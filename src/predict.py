"""
Main entry point for training and evaluating the NBA Draft Prediction Model.

Usage (from repo root)::

    python -m src.predict --data data/nba_draft_data.csv --plots models/plots

Or train and then predict on new players::

    python -m src.predict \\
        --data data/nba_draft_data.csv \\
        --predict data/new_players.csv \\
        --plots models/plots
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

# Allow running as a script from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import (
    load_data,
    preprocess_data,
    get_features_and_target,
    split_data,
)
from src.feature_engineering import add_engineered_features, get_all_feature_columns
from src.model import (
    train_models,
    evaluate_model,
    save_model,
    load_model,
    DEFAULT_MODEL_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_pipeline(
    data_path: str,
    model_dir: str = DEFAULT_MODEL_DIR,
    plot_dir: str | None = None,
) -> str:
    """Run the full training pipeline.

    1. Load raw CSV data.
    2. Preprocess (encode, impute).
    3. Add engineered features.
    4. Split into train / val / test.
    5. Train three classifiers; select the best by validation AUC.
    6. Evaluate the best model on the held-out test set.
    7. Save the model bundle.

    Args:
        data_path: Path to the player CSV file.
        model_dir: Directory where the trained model bundle is saved.
        plot_dir: Optional directory to write evaluation plots.

    Returns:
        Path to the saved model bundle.
    """
    logger.info("=== NBA Draft Prediction Model – Training Pipeline ===")

    # 1. Load
    df_raw = load_data(data_path)
    logger.info("Dataset shape: %s", df_raw.shape)

    # 2. Preprocess
    df, label_encoders = preprocess_data(df_raw, fit_encoders=True)

    # 3. Feature engineering
    df = add_engineered_features(df)

    # 4. Split
    X, y = get_features_and_target_extended(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    logger.info(
        "Split sizes – train: %d  val: %d  test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    # 5. Train
    best_model, best_name, results, scaler = train_models(
        X_train, y_train, X_val, y_val
    )
    _print_results_table(results)

    # 6. Evaluate on test set
    logger.info("--- Final test-set evaluation: %s ---", best_name)
    evaluate_model(
        best_model, X_test, y_test,
        scaler=scaler,
        model_name=best_name,
        plot_dir=plot_dir,
    )

    # 7. Save
    bundle_path = save_model(
        best_model,
        scaler,
        label_encoders,
        list(X_train.columns),
        best_name,
        model_dir=model_dir,
    )
    return bundle_path


def predict_pipeline(bundle_path: str, input_path: str) -> pd.DataFrame:
    """Load a saved model and run inference on new player data.

    Args:
        bundle_path: Path to the ``.joblib`` model bundle.
        input_path: Path to a CSV file with new player statistics.

    Returns:
        DataFrame with ``player_id``, ``name``, ``draft_probability``, and
        ``predicted_drafted`` columns.
    """
    logger.info("=== NBA Draft Prediction Model – Inference ===")

    bundle = load_model(bundle_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    label_encoders = bundle["label_encoders"]
    feature_columns = bundle["feature_columns"]
    model_name = bundle["model_name"]

    df_raw = pd.read_csv(input_path)

    # Keep ID columns for output
    id_cols = [c for c in ["player_id", "name"] if c in df_raw.columns]
    df_ids = df_raw[id_cols].copy() if id_cols else pd.DataFrame()

    df, _ = preprocess_data(df_raw, label_encoders=label_encoders, fit_encoders=False)
    df = add_engineered_features(df)

    # Align columns
    missing = [c for c in feature_columns if c not in df.columns]
    for col in missing:
        df[col] = 0.0
    X = df[feature_columns]

    if model_name == "logistic_regression":
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        probs = model.predict_proba(X_scaled)[:, 1]
    else:
        probs = model.predict_proba(X)[:, 1]

    threshold = 0.5
    preds = (probs >= threshold).astype(int)

    result = df_ids.copy()
    result["draft_probability"] = probs.round(4)
    result["predicted_drafted"] = preds

    logger.info("Predictions for %d players:", len(result))
    for _, row in result.iterrows():
        status = "DRAFTED" if row["predicted_drafted"] else "UNDRAFTED"
        name = row.get("name", "Unknown")
        logger.info("  %-20s  prob=%.3f  → %s", name, row["draft_probability"], status)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_features_and_target_extended(df: pd.DataFrame):
    """Return X and y using all feature columns (including engineered)."""
    feature_cols = get_all_feature_columns()
    # Keep only columns that exist in df
    feature_cols = [c for c in feature_cols if c in df.columns]
    from src.data_loader import TARGET_COLUMN
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def _print_results_table(results: dict) -> None:
    print("\n" + "=" * 50)
    print(f"{'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['roc_auc']:>10.4f}")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NBA Draft Prediction Model – train and predict."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to player statistics CSV file.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Directory to save/load model bundles.",
    )
    parser.add_argument(
        "--plots",
        default=None,
        help="Directory to save evaluation plots (optional).",
    )
    parser.add_argument(
        "--predict",
        default=None,
        help="Path to CSV file of new players to predict (optional). "
             "If provided, the script loads an existing model bundle instead "
             "of training.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.predict:
        # Find the latest bundle in model_dir
        bundles = [
            f for f in os.listdir(args.model_dir) if f.endswith(".joblib")
        ]
        if not bundles:
            logger.error(
                "No model bundles found in %s. Run training first.", args.model_dir
            )
            sys.exit(1)
        bundle_path = os.path.join(args.model_dir, sorted(bundles)[-1])
        results = predict_pipeline(bundle_path, args.predict)
        print(results.to_string(index=False))
    else:
        train_pipeline(args.data, model_dir=args.model_dir, plot_dir=args.plots)


if __name__ == "__main__":
    main()
