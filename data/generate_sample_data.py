"""
Script to generate a realistic sample NBA draft dataset.

This generates synthetic college basketball player statistics that mirror
the types of data available from sources such as Sports Reference (college
basketball statistics combined with NBA draft history).
"""

import numpy as np
import pandas as pd
import os

SEED = 42
N_PLAYERS = 1000
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "nba_draft_data.csv")


def generate_sample_data(n_players: int = N_PLAYERS, seed: int = SEED) -> pd.DataFrame:
    """Generate a realistic synthetic dataset of college basketball players.

    Features are calibrated so that drafted players tend to have better
    statistics, mimicking real draft selection patterns.

    Args:
        n_players: Number of player records to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with player statistics and draft outcome labels.
    """
    rng = np.random.default_rng(seed)

    # ~30% of prospects get drafted (roughly 60 picks / ~200 notable prospects)
    drafted = rng.integers(0, 2, size=n_players, dtype=int)
    drafted = (rng.random(n_players) < 0.30).astype(int)

    # Helper: drafted players have higher baseline for most stats
    def stat(base_drafted, base_undrafted, std, drafted_flag):
        return np.where(
            drafted_flag,
            rng.normal(base_drafted, std, n_players),
            rng.normal(base_undrafted, std, n_players),
        )

    # Physical measurements
    height_inches = stat(79, 76, 3, drafted)          # ~6'7" vs ~6'4"
    weight_lbs = stat(215, 205, 20, drafted)

    # Per-game statistics
    ppg = np.clip(stat(18, 12, 5, drafted), 0, 40)
    rpg = np.clip(stat(7.0, 5.5, 2.5, drafted), 0, 20)
    apg = np.clip(stat(3.5, 2.5, 1.5, drafted), 0, 12)
    spg = np.clip(stat(1.5, 1.0, 0.6, drafted), 0, 5)
    bpg = np.clip(stat(1.2, 0.7, 0.6, drafted), 0, 5)
    topg = np.clip(stat(2.5, 2.0, 1.0, drafted), 0, 8)   # turnovers

    # Shooting percentages
    fg_pct = np.clip(stat(0.48, 0.44, 0.06, drafted), 0.2, 0.75)
    three_pct = np.clip(stat(0.36, 0.33, 0.07, drafted), 0.1, 0.60)
    ft_pct = np.clip(stat(0.78, 0.72, 0.09, drafted), 0.3, 1.0)

    # Advanced metrics
    per = np.clip(stat(22, 16, 5, drafted), 5, 40)
    win_shares = np.clip(stat(5.5, 3.0, 2.5, drafted), 0, 15)

    # Contextual
    games_played = np.clip(
        rng.integers(20, 40, n_players).astype(float), 10, 40
    )
    age = np.clip(rng.normal(20, 1, n_players), 18, 25)

    positions = ["PG", "SG", "SF", "PF", "C"]
    position = rng.choice(positions, n_players)

    conferences = ["ACC", "Big Ten", "Big 12", "SEC", "Pac-12", "Big East", "Other"]
    conference_weights = [0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.20]
    conference = rng.choice(conferences, n_players, p=conference_weights)

    draft_year = rng.choice(range(2010, 2024), n_players)

    player_ids = [f"P{str(i).zfill(4)}" for i in range(1, n_players + 1)]
    names = [f"Player_{pid}" for pid in player_ids]

    df = pd.DataFrame(
        {
            "player_id": player_ids,
            "name": names,
            "draft_year": draft_year,
            "age": np.round(age, 1),
            "position": position,
            "conference": conference,
            "height_inches": np.round(height_inches, 0).astype(int),
            "weight_lbs": np.round(weight_lbs, 0).astype(int),
            "games_played": games_played.astype(int),
            "ppg": np.round(ppg, 1),
            "rpg": np.round(rpg, 1),
            "apg": np.round(apg, 1),
            "spg": np.round(spg, 1),
            "bpg": np.round(bpg, 1),
            "topg": np.round(topg, 1),
            "fg_pct": np.round(fg_pct, 3),
            "three_pct": np.round(three_pct, 3),
            "ft_pct": np.round(ft_pct, 3),
            "per": np.round(per, 1),
            "win_shares": np.round(win_shares, 1),
            "drafted": drafted,
        }
    )

    return df


if __name__ == "__main__":
    df = generate_sample_data()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} player records → {OUTPUT_PATH}")
    print(df.head())
    print(f"\nDraft rate: {df['drafted'].mean():.1%}")
