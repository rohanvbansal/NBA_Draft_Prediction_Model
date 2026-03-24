"""
Unit tests for the NBA Draft Prediction Model.

Tests cover data loading, preprocessing, feature engineering, model training,
and inference.
"""

from __future__ import annotations

import os
import sys
import tempfile
import pytest
import numpy as np
import pandas as pd

# Ensure the repo root is on the path
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from data.generate_sample_data import generate_sample_data
from src.data_loader import (
    load_data,
    preprocess_data,
    get_features_and_target,
    split_data,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)
from src.feature_engineering import add_engineered_features, get_all_feature_columns
from src.model import (
    train_models,
    evaluate_model,
    save_model,
    load_model,
)
from src.predict import get_features_and_target_extended, train_pipeline, predict_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_df():
    """Small synthetic dataset used across tests."""
    return generate_sample_data(n_players=200, seed=0)


@pytest.fixture(scope="module")
def preprocessed_df(raw_df):
    df, encoders = preprocess_data(raw_df, fit_encoders=True)
    return df, encoders


@pytest.fixture(scope="module")
def engineered_df(preprocessed_df):
    df, encoders = preprocessed_df
    return add_engineered_features(df), encoders


# ---------------------------------------------------------------------------
# Data generation tests
# ---------------------------------------------------------------------------

class TestDataGeneration:
    def test_shape(self, raw_df):
        assert len(raw_df) == 200

    def test_columns_present(self, raw_df):
        for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_draft_rate_reasonable(self, raw_df):
        rate = raw_df[TARGET_COLUMN].mean()
        assert 0.10 <= rate <= 0.60, f"Draft rate {rate:.2%} is outside expected range"

    def test_no_all_nan_column(self, raw_df):
        for col in NUMERIC_FEATURES:
            assert raw_df[col].notna().any(), f"Column {col} is entirely NaN"

    def test_reproducibility(self):
        df1 = generate_sample_data(n_players=50, seed=7)
        df2 = generate_sample_data(n_players=50, seed=7)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_load_data_from_csv(self, raw_df):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            raw_df.to_csv(f.name, index=False)
            path = f.name
        try:
            loaded = load_data(path)
            assert len(loaded) == len(raw_df)
        finally:
            os.unlink(path)

    def test_load_data_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent/path/data.csv")

    def test_load_data_missing_columns(self):
        df_bad = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df_bad.to_csv(f.name, index=False)
            path = f.name
        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                load_data(path)
        finally:
            os.unlink(path)

    def test_preprocess_no_nans(self, raw_df):
        # Introduce NaN deliberately
        df_with_nan = raw_df.copy()
        df_with_nan.loc[0, "ppg"] = np.nan
        df, _ = preprocess_data(df_with_nan, fit_encoders=True)
        assert df[NUMERIC_FEATURES].isna().sum().sum() == 0

    def test_preprocess_categorical_encoded(self, preprocessed_df):
        df, _ = preprocessed_df
        for col in CATEGORICAL_FEATURES:
            assert pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]), \
                f"Column {col} not numerically encoded"

    def test_split_sizes(self, raw_df):
        df, _ = preprocess_data(raw_df, fit_encoders=True)
        X, y = get_features_and_target(df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(X), "Split rows don't add up to total"
        assert len(X_test) > 0
        assert len(X_val) > 0
        assert len(X_train) > 0

    def test_split_stratification(self, raw_df):
        df, _ = preprocess_data(raw_df, fit_encoders=True)
        X, y = get_features_and_target(df)
        _, _, _, y_train, _, y_test = split_data(X, y)
        # Both splits should contain both classes
        assert set(y_train.unique()) == {0, 1}
        assert set(y_test.unique()) == {0, 1}


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    def test_engineered_columns_exist(self, engineered_df):
        df, _ = engineered_df
        for col in ["scoring_efficiency", "versatility_score", "defensive_impact",
                    "assist_to_turnover", "bmi", "points_per_game_scaled"]:
            assert col in df.columns, f"Missing engineered column: {col}"

    def test_no_nans_after_engineering(self, engineered_df):
        df, _ = engineered_df
        all_features = get_all_feature_columns()
        present = [c for c in all_features if c in df.columns]
        assert df[present].isna().sum().sum() == 0, "NaN values found after feature engineering"

    def test_scoring_efficiency_formula(self, engineered_df):
        df, _ = engineered_df
        expected = df["ppg"] * df["fg_pct"]
        pd.testing.assert_series_equal(df["scoring_efficiency"], expected, check_names=False)

    def test_bmi_positive(self, engineered_df):
        df, _ = engineered_df
        assert (df["bmi"] > 0).all(), "BMI values should be positive"

    def test_get_all_feature_columns_count(self):
        cols = get_all_feature_columns()
        assert len(cols) == len(set(cols)), "Duplicate column names in feature list"
        assert len(cols) > 0


# ---------------------------------------------------------------------------
# Model training & evaluation tests
# ---------------------------------------------------------------------------

class TestModel:
    @pytest.fixture(scope="class")
    def trained_artifacts(self, raw_df):
        df, encoders = preprocess_data(raw_df, fit_encoders=True)
        df = add_engineered_features(df)
        X, y = get_features_and_target_extended(df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        best_model, best_name, results, scaler = train_models(
            X_train, y_train, X_val, y_val
        )
        return best_model, best_name, results, scaler, X_test, y_test, encoders, list(X_train.columns)

    def test_results_have_expected_keys(self, trained_artifacts):
        _, _, results, *_ = trained_artifacts
        for key in ["logistic_regression", "random_forest", "xgboost"]:
            assert key in results
            assert "accuracy" in results[key]
            assert "roc_auc" in results[key]

    def test_auc_above_random(self, trained_artifacts):
        _, _, results, *_ = trained_artifacts
        for name, metrics in results.items():
            assert metrics["roc_auc"] > 0.5, \
                f"{name} AUC {metrics['roc_auc']:.4f} is not above random"

    def test_accuracy_reasonable(self, trained_artifacts):
        _, _, results, *_ = trained_artifacts
        for name, metrics in results.items():
            assert 0.5 <= metrics["accuracy"] <= 1.0, \
                f"{name} accuracy {metrics['accuracy']:.4f} is unreasonable"

    def test_evaluate_model(self, trained_artifacts):
        model, name, _, scaler, X_test, y_test, *_ = trained_artifacts
        result = evaluate_model(model, X_test, y_test, scaler=scaler, model_name=name)
        assert "accuracy" in result
        assert "roc_auc" in result
        assert "classification_report" in result

    def test_save_and_load_model(self, trained_artifacts):
        model, name, _, scaler, X_test, y_test, encoders, feature_cols = trained_artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = save_model(
                model, scaler, encoders, feature_cols, name, model_dir=tmpdir
            )
            assert os.path.exists(bundle_path)
            bundle = load_model(bundle_path)
            assert "model" in bundle
            assert "scaler" in bundle
            assert "label_encoders" in bundle
            assert "feature_columns" in bundle


# ---------------------------------------------------------------------------
# End-to-end pipeline tests
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_train_pipeline(self, raw_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            raw_df.to_csv(data_path, index=False)
            bundle_path = train_pipeline(data_path, model_dir=tmpdir)
            assert os.path.exists(bundle_path)

    def test_predict_pipeline(self, raw_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            raw_df.to_csv(data_path, index=False)
            bundle_path = train_pipeline(data_path, model_dir=tmpdir)

            # Use a subset as new players
            new_players = raw_df.drop(columns=[TARGET_COLUMN]).head(10)
            new_path = os.path.join(tmpdir, "new_players.csv")
            new_players.to_csv(new_path, index=False)

            results = predict_pipeline(bundle_path, new_path)
            assert len(results) == 10
            assert "draft_probability" in results.columns
            assert "predicted_drafted" in results.columns
            assert results["draft_probability"].between(0, 1).all()
            assert results["predicted_drafted"].isin([0, 1]).all()
