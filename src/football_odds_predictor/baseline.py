"""Evaluate simple closing-odds baselines on V1 validation folds."""

import argparse
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import TARGET_COLUMNS, load_dataset, validation_folds
from .features import HISTORY_LENGTH, MATCH_STATS, SIDES


def summarize_form(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Keep current Elo and average each team's recent statistics."""
    fields = (
        "was_home", "opponent_elo",
        *(f"{stat}_{direction}" for stat in MATCH_STATS for direction in ("for", "against")),
    )
    summaries = {
        f"{side}_mean_{field}": fixtures[
            [f"{side}_match_{i}_{field}" for i in range(1, HISTORY_LENGTH + 1)]
        ].mean(axis=1, skipna=False)
        for side in SIDES for field in fields
    }
    return fixtures[["home_elo", "away_elo"]].join(pd.DataFrame(summaries, index=fixtures.index))


def ridge_predictions(train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(summarize_form(train), np.log(train[TARGET_COLUMNS].to_numpy()))
    log_odds = model.predict(summarize_form(validation))
    return np.exp(np.maximum(log_odds, np.log(1.01)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    parser.add_argument("--matches", type=Path, default=Path("data/processed/matches.csv"))
    parser.add_argument("--model", choices=("median", "ridge"), default="median")
    args = parser.parse_args()

    dataset = load_dataset(args.features, args.matches)
    log_errors = []
    decimal_errors = []
    for season, train, validation in validation_folds(dataset):
        if args.model == "ridge":
            predictions = ridge_predictions(train, validation)
            prediction_summary = ""
        else:
            predictions = train[TARGET_COLUMNS].median().to_numpy()
            odds = " / ".join(f"{value:.2f}" for value in predictions)
            prediction_summary = f" | odds={odds}"
        actual = validation[TARGET_COLUMNS].to_numpy()
        log_mae = np.abs(np.log(actual) - np.log(predictions)).mean()
        decimal_mae = np.abs(actual - predictions).mean()
        log_errors.append(log_mae)
        decimal_errors.append(decimal_mae)
        print(
            f"{season}/{(season + 1) % 100:02d} | train={len(train):,} "
            f"| validation={len(validation):,}{prediction_summary} "
            f"| log MAE={log_mae:.4f} | decimal MAE={decimal_mae:.4f}"
        )

    print(f"Fold mean | log MAE={mean(log_errors):.4f} | decimal MAE={mean(decimal_errors):.4f}")


if __name__ == "__main__":
    main()
