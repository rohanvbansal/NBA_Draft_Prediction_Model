"""
Feature engineering for the NBA Draft Prediction Model.

Constructs composite and ratio features that tend to be more predictive
of draft outcomes than raw per-game statistics alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the DataFrame.

    The following composite features are added:

    * ``scoring_efficiency`` – PPG adjusted for field-goal percentage.
    * ``versatility_score`` – Sum of PPG + RPG + APG (box-score versatility).
    * ``defensive_impact`` – SPG + BPG combined defensive contribution.
    * ``assist_to_turnover`` – APG divided by TOPG (ball security proxy).
    * ``bmi`` – Body-mass index proxy (weight / height²) scaled to real BMI.
    * ``points_per_minute`` – PPG divided by games_played (volume proxy).

    Args:
        df: DataFrame that must already contain the raw numeric columns
            produced by :func:`src.data_loader.preprocess_data`.

    Returns:
        DataFrame with additional engineered columns appended.
    """
    df = df.copy()

    df["scoring_efficiency"] = df["ppg"] * df["fg_pct"]

    df["versatility_score"] = df["ppg"] + df["rpg"] + df["apg"]

    df["defensive_impact"] = df["spg"] + df["bpg"]

    safe_topg = df["topg"].replace(0, np.nan)
    df["assist_to_turnover"] = (df["apg"] / safe_topg).fillna(0.0)

    # BMI: weight(lbs) / height(in)^2 * 703
    df["bmi"] = df["weight_lbs"] / (df["height_inches"] ** 2) * 703

    df["points_per_game_scaled"] = df["ppg"] / (df["games_played"].replace(0, np.nan)).fillna(1)

    # Final guard: fill any remaining NaN values introduced by divisions
    engineered_cols = [
        "scoring_efficiency",
        "versatility_score",
        "defensive_impact",
        "assist_to_turnover",
        "bmi",
        "points_per_game_scaled",
    ]
    for col in engineered_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(0.0)

    return df


def get_all_feature_columns() -> list[str]:
    """Return the full ordered list of feature columns used for modelling.

    Returns:
        List of column names.
    """
    engineered = [
        "scoring_efficiency",
        "versatility_score",
        "defensive_impact",
        "assist_to_turnover",
        "bmi",
        "points_per_game_scaled",
    ]
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES + engineered
