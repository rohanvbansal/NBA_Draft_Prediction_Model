# NBA Draft Prediction Model

A machine-learning pipeline that predicts whether a college basketball player
will be selected in the NBA Draft based on their college statistics.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Features](#features)
- [Models](#models)
- [Results](#results)
- [Running Tests](#running-tests)

---

## Overview

The model ingests per-game college basketball statistics (points, rebounds,
assists, shooting percentages, etc.) alongside physical measurements and
advanced metrics, then trains three classifiers to predict draft outcomes:

| Classifier | Notes |
|---|---|
| Logistic Regression | Linear baseline with StandardScaler |
| Random Forest | 200-tree ensemble, handles non-linearity |
| XGBoost | Gradient-boosted trees, strong out-of-the-box |

The best classifier (by validation ROC-AUC) is selected automatically and
saved to disk with `joblib`.

---

## Project Structure

```
NBA_Draft_Prediction_Model/
├── data/
│   ├── generate_sample_data.py   # Generates nba_draft_data.csv
│   └── nba_draft_data.csv        # Sample dataset (1000 players)
├── models/                       # Saved model bundles (.joblib)
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Loading, preprocessing, train/val/test split
│   ├── feature_engineering.py   # Engineered composite features
│   ├── model.py                  # Training, evaluation, persistence
│   └── predict.py                # Main CLI entry point
├── tests/
│   └── test_model.py             # 24 pytest unit & integration tests
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate (or provide) player data

```bash
python data/generate_sample_data.py
```

This creates `data/nba_draft_data.csv` with 1 000 synthetic college player
records.  Replace this file with real data in the same format to train on
actual draft history.

### 3. Train the model

```bash
python -m src.predict --data data/nba_draft_data.csv --plots models/plots
```

The script:

1. Loads and preprocesses the data.
2. Engineers additional composite features.
3. Trains Logistic Regression, Random Forest, and XGBoost classifiers.
4. Selects the best model by validation ROC-AUC.
5. Evaluates the winner on the held-out test set.
6. Saves the model bundle to `models/<model_name>.joblib`.
7. Writes confusion matrix, ROC curve, and feature-importance plots to
   `models/plots/` (if `--plots` is provided).

### 4. Predict on new players

Prepare a CSV with the same columns as the training file (minus the `drafted`
column), then run:

```bash
python -m src.predict \
    --data data/nba_draft_data.csv \
    --model-dir models \
    --predict data/new_players.csv
```

Output columns: `player_id`, `name`, `draft_probability`, `predicted_drafted`.

---

## Features

### Raw statistics

| Column | Description |
|---|---|
| `age` | Age at time of draft |
| `height_inches` | Height in inches |
| `weight_lbs` | Weight in pounds |
| `games_played` | Games played in college season |
| `ppg` | Points per game |
| `rpg` | Rebounds per game |
| `apg` | Assists per game |
| `spg` | Steals per game |
| `bpg` | Blocks per game |
| `topg` | Turnovers per game |
| `fg_pct` | Field goal percentage |
| `three_pct` | Three-point percentage |
| `ft_pct` | Free throw percentage |
| `per` | Player Efficiency Rating |
| `win_shares` | Win Shares |
| `position` | PG / SG / SF / PF / C |
| `conference` | College conference |

### Engineered features

| Feature | Formula |
|---|---|
| `scoring_efficiency` | `ppg × fg_pct` |
| `versatility_score` | `ppg + rpg + apg` |
| `defensive_impact` | `spg + bpg` |
| `assist_to_turnover` | `apg / topg` |
| `bmi` | `weight / height² × 703` |
| `points_per_game_scaled` | `ppg / games_played` |

---

## Models

Three classifiers are trained and compared:

- **Logistic Regression** – scaled features, `class_weight="balanced"`.
- **Random Forest** – 200 trees, max depth 8, `class_weight="balanced"`.
- **XGBoost** – 200 trees, learning rate 0.05, log-loss evaluation metric.

The model with the highest validation ROC-AUC is saved and used for
inference.

---

## Results

On the included 1 000-player synthetic dataset (80 / 10 / 10 train-val-test
split):

| Model | Val Accuracy | Val ROC-AUC |
|---|---|---|
| Logistic Regression | ~0.90 | ~0.96 |
| Random Forest | ~0.91 | **~0.98** |
| XGBoost | ~0.91 | ~0.97 |

Test-set results for the best model (Random Forest): **~92% accuracy,
~0.979 ROC-AUC**.

---

## Running Tests

```bash
pytest tests/ -v
```

24 tests covering data generation, preprocessing, feature engineering, model
training, persistence, and the full end-to-end pipeline.