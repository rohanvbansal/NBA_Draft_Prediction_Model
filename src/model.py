"""
Model training, evaluation, and persistence for the NBA Draft Prediction Model.

Three classifiers are compared:
  * Logistic Regression  (linear baseline)
  * Random Forest        (ensemble, handles non-linearity well)
  * XGBoost              (gradient-boosted trees, typically best performance)

The best model (by validation ROC-AUC) is saved to disk with ``joblib``.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_classifiers() -> Dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        ),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[object, str, Dict, StandardScaler]:
    """Train all classifiers and return the best one.

    Logistic Regression uses a StandardScaler fitted on the training data;
    tree-based models receive the raw (unscaled) features.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.

    Returns:
        Tuple of (best_model, best_model_name, results_dict, scaler).
        ``results_dict`` maps model name → dict with ``accuracy`` and
        ``roc_auc`` on the validation set.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns
    )

    classifiers = _build_classifiers()
    results: Dict[str, Dict] = {}
    best_model = None
    best_model_name = ""
    best_auc = -1.0

    for name, clf in classifiers.items():
        if name == "logistic_regression":
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_val_scaled)
            y_prob = clf.predict_proba(X_val_scaled)[:, 1]
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            y_prob = clf.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)
        results[name] = {"accuracy": acc, "roc_auc": auc, "model": clf}
        logger.info("%s → accuracy=%.4f  AUC=%.4f", name, acc, auc)

        if auc > best_auc:
            best_auc = auc
            best_model = clf
            best_model_name = name

    logger.info("Best model: %s (AUC=%.4f)", best_model_name, best_auc)
    return best_model, best_model_name, results, scaler


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scaler: StandardScaler | None = None,
    model_name: str = "model",
    plot_dir: str | None = None,
) -> Dict:
    """Evaluate a trained model on the test set and optionally save plots.

    Args:
        model: Trained classifier.
        X_test: Test feature matrix.
        y_test: Test labels.
        scaler: StandardScaler used during training (required for logistic
            regression; pass ``None`` for tree-based models).
        model_name: String identifier used in log messages and plot titles.
        plot_dir: Directory where plots are saved.  If ``None``, no plots
            are saved.

    Returns:
        Dict with ``accuracy``, ``roc_auc``, and ``classification_report``.
    """
    if scaler is not None and model_name == "logistic_regression":
        X_eval = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    else:
        X_eval = X_test

    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["Undrafted", "Drafted"])

    logger.info("Test accuracy: %.4f", acc)
    logger.info("Test ROC-AUC:  %.4f", auc)
    logger.info("\n%s", report)

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        _plot_confusion_matrix(y_test, y_pred, model_name, plot_dir)
        _plot_roc_curve(y_test, y_prob, model_name, plot_dir)
        if hasattr(model, "feature_importances_"):
            feature_names = list(X_test.columns)
            _plot_feature_importance(model, feature_names, model_name, plot_dir)

    return {"accuracy": acc, "roc_auc": auc, "classification_report": report}


def _plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    plot_dir: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Undrafted", "Drafted"],
        yticklabels=["Undrafted", "Drafted"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix – {model_name}")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    path = os.path.join(plot_dir, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", path)


def _plot_roc_curve(
    y_true: pd.Series,
    y_prob: np.ndarray,
    model_name: str,
    plot_dir: str,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve – {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = os.path.join(plot_dir, f"roc_curve_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curve → %s", path)


def _plot_feature_importance(
    model: object,
    feature_names: list[str],
    model_name: str,
    plot_dir: str,
    top_n: int = 15,
) -> None:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_indices = indices[::-1]  # reverse for horizontal bar chart (highest at top)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in top_indices],
        importances[top_indices],
    )
    ax.set_title(f"Feature Importance – {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    path = os.path.join(plot_dir, f"feature_importance_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance → %s", path)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(
    model: object,
    scaler: StandardScaler,
    label_encoders: dict,
    feature_columns: list[str],
    model_name: str,
    model_dir: str = DEFAULT_MODEL_DIR,
) -> str:
    """Persist trained artifacts to disk.

    Saves a single ``.joblib`` bundle containing the model, scaler,
    label encoders, and feature column list.

    Args:
        model: Trained classifier.
        scaler: Fitted StandardScaler.
        label_encoders: Fitted LabelEncoder instances keyed by column name.
        feature_columns: Ordered list of feature column names.
        model_name: Name used to derive the filename.
        model_dir: Directory where the bundle is saved.

    Returns:
        Full path to the saved bundle.
    """
    os.makedirs(model_dir, exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_columns": feature_columns,
        "model_name": model_name,
    }
    path = os.path.join(model_dir, f"{model_name}.joblib")
    joblib.dump(bundle, path)
    logger.info("Model bundle saved → %s", path)
    return path


def load_model(path: str) -> dict:
    """Load a persisted model bundle from disk.

    Args:
        path: Path to the ``.joblib`` bundle.

    Returns:
        Bundle dict with keys ``model``, ``scaler``, ``label_encoders``,
        ``feature_columns``, and ``model_name``.

    Raises:
        FileNotFoundError: If the bundle does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model bundle not found: {path}")
    return joblib.load(path)
