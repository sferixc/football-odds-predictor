"""Evaluate the median closing-odds baseline on V1 validation folds."""

import argparse
from pathlib import Path
from statistics import mean

import numpy as np

from .dataset import TARGET_COLUMNS, load_dataset, validation_folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    parser.add_argument("--matches", type=Path, default=Path("data/processed/matches.csv"))
    args = parser.parse_args()

    dataset = load_dataset(args.features, args.matches)
    log_errors = []
    decimal_errors = []
    for season, train, validation in validation_folds(dataset):
        predictions = train[TARGET_COLUMNS].median().to_numpy()
        actual = validation[TARGET_COLUMNS].to_numpy()
        log_mae = np.abs(np.log(actual) - np.log(predictions)).mean()
        decimal_mae = np.abs(actual - predictions).mean()
        log_errors.append(log_mae)
        decimal_errors.append(decimal_mae)
        odds = " / ".join(f"{value:.2f}" for value in predictions)
        print(
            f"{season}/{(season + 1) % 100:02d} | train={len(train):,} "
            f"| validation={len(validation):,} | odds={odds} "
            f"| log MAE={log_mae:.4f} | decimal MAE={decimal_mae:.4f}"
        )

    print(f"Fold mean | log MAE={mean(log_errors):.4f} | decimal MAE={mean(decimal_errors):.4f}")


if __name__ == "__main__":
    main()
