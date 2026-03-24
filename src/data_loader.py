"""
Data loading and preprocessing utilities for the NBA Draft Prediction Model.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "age",
    "height_inches",
    "weight_lbs",
    "games_played",
    "ppg",
    "rpg",
    "apg",
    "spg",
    "bpg",
    "topg",
    "fg_pct",
    "three_pct",
    "ft_pct",
    "per",
    "win_shares",
]

CATEGORICAL_FEATURES = ["position", "conference"]

TARGET_COLUMN = "drafted"

# Columns that are identifiers and should not be used as features
ID_COLUMNS = ["player_id", "name", "draft_year"]


def load_data(filepath: str) -> pd.DataFrame:
    """Load player data from a CSV file.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Raw DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows from %s", len(df), filepath)

    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def preprocess_data(
    df: pd.DataFrame,
    label_encoders: dict | None = None,
    fit_encoders: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Clean and encode the raw DataFrame.

    Numeric columns are coerced to float and missing values are filled with
    column medians.  Categorical columns are label-encoded.

    Args:
        df: Raw input DataFrame.
        label_encoders: Pre-fitted LabelEncoder instances (used during
            inference to ensure consistent encoding).
        fit_encoders: If True, fit new LabelEncoders; otherwise use the
            provided ones.

    Returns:
        Tuple of (preprocessed DataFrame, label_encoder dict).
    """
    df = df.copy()

    # Coerce numeric columns
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Encode categorical columns
    if label_encoders is None:
        label_encoders = {}

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()
        if fit_encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            # Handle unseen categories gracefully
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])

    return df, label_encoders


def get_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target vector y.

    Args:
        df: Preprocessed DataFrame.

    Returns:
        Tuple of (X, y).
    """
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split data into training, validation, and test sets.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Fraction of data held out for the test set.
        val_size: Fraction of the remaining data held out for validation.
        random_state: Random seed.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
