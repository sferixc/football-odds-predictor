"""Check calibration and score the fixed sporting models on 2025/26."""

import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from ..data.dataset import final_test_split, validation_folds
from ..data.sporting import player_dataset, team_dataset
from ..models.sporting import predict_saved
from ..training.fit_sporting import DEFAULT_MODELS, PLAYER_LINES, TEAM_MODELS, fit_model
from ..training.sporting import TEAM_THRESHOLDS, add_team_outcomes

BIN_COUNT = 10


def reliability(actual: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    """Summarize prediction and observed frequency in fixed 10-point bands."""
    outcomes = ("1", "X", "2") if probabilities.shape[1] == 3 else ("Under", "Over")
    stats = []
    for index in range(0 if len(outcomes) == 3 else 1, len(outcomes)):
        predicted = probabilities[:, index]
        observed = actual == index
        bands = np.minimum((predicted * BIN_COUNT).astype(int), BIN_COUNT - 1)
        for band in np.unique(bands):
            mask = bands == band
            stats.append({
                "outcome": outcomes[index], "low": band / BIN_COUNT,
                "high": (band + 1) / BIN_COUNT, "count": int(mask.sum()),
                "mean_probability": float(predicted[mask].mean()),
                "observed_rate": float(observed[mask].mean()),
            })
    return stats


def assess_target(
    rows: pd.DataFrame, inputs: list[str], target: str, model_type: str,
    threshold: float | None, models: dict | None,
) -> list[dict]:
    eligible = rows.dropna(subset=[*inputs, target])
    if models is None:
        folds = ((train, validation) for _, train, validation in validation_folds(eligible))
    else:
        folds = (final_test_split(eligible),)
    key = (target, threshold)
    predictions, outcomes = [], []
    for train, evaluation in folds:
        fitted = models if models is not None else {
            key: fit_model(train, inputs, target, model_type, threshold),
        }
        cutoff = evaluation["source_kickoff"].min().date()
        probabilities = predict_saved(fitted, evaluation[inputs], target, cutoff, threshold)
        labels = evaluation[target] if threshold is None else evaluation[target].gt(threshold)
        predictions.append(probabilities)
        outcomes.append(labels.to_numpy(dtype=int))

    probabilities = np.concatenate(predictions)
    actual = np.concatenate(outcomes)
    classes = np.arange(probabilities.shape[1])
    loss = log_loss(actual, probabilities, labels=classes)
    brier = np.square(probabilities - np.eye(len(classes))[actual]).sum(axis=1).mean()
    if threshold is not None:
        brier /= 2
    bins = reliability(actual, probabilities)
    calibration_gap = sum(
        row["count"] * abs(row["mean_probability"] - row["observed_rate"]) for row in bins
    ) / sum(row["count"] for row in bins)
    name = target if threshold is None else f"{target}>{threshold:g}"
    print(f"{name:<24} n={len(actual):>6,} | log loss={loss:.4f} | "
          f"Brier={brier:.4f} | calibration gap={calibration_gap:.1%}")
    return [{"target": name, **row} for row in bins]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validation", "test"))
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    args = parser.parse_args()
    models = None
    if args.phase == "test":
        with args.models.open("rb") as handle:
            models = pickle.load(handle)

    teams, team_inputs = team_dataset(Path("data/processed/features.csv"), Path("data/raw/football-data"))
    add_team_outcomes(teams)
    players, player_inputs = player_dataset(Path("data/raw/xgtables-understat"))
    results = []
    for target, model_type in TEAM_MODELS.items():
        results.extend(assess_target(teams, team_inputs, target, model_type, TEAM_THRESHOLDS.get(target), models))
    for threshold in PLAYER_LINES:
        results.extend(assess_target(players, player_inputs, "shots", "xgboost", threshold, models))
    output = Path(f"data/processed/sporting-reliability-{args.phase}.csv")
    pd.DataFrame(results).to_csv(output, index=False)
    print(f"Saved reliability bands to {output}")


if __name__ == "__main__":
    main()
