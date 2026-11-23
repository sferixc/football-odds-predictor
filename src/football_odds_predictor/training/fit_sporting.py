"""Fit and save the sporting models selected on historical validation folds."""

import argparse
from pathlib import Path
import pickle
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ..data.dataset import FIRST_SEASON, TEST_SEASON
from ..data.sporting import player_dataset, team_dataset
from .sporting import TEAM_THRESHOLDS, XGB_SETTINGS, add_team_outcomes, fit_epoch_classifier

DEFAULT_MODELS = Path("data/processed/sporting-models.pkl")
TEAM_MODELS = {
    "1x2": "sgd",
    "match_goals": "sgd",
    "first_half_goals": "frequency",
    "second_half_goals": "sgd",
    "home_yellows": "sgd",
    "away_yellows": "sgd",
    "match_yellows": "sgd",
    "match_corners": "sgd",
    "home_corners": "logistic",
    "away_corners": "xgboost",
}
PLAYER_LINES = (0.5, 1.5, 2.5)


def fit_model(rows: pd.DataFrame, inputs: list[str], target: str, model_type: str,
              threshold: float | None) -> dict:
    """Return fitted parameters with the exact input columns and target line."""
    train = rows.loc[rows["season_start_year"].between(FIRST_SEASON, TEST_SEASON - 1)]
    train = train.dropna(subset=[*inputs, target])
    labels = train[target] if threshold is None else train[target].gt(threshold)
    y = labels.to_numpy(dtype=int)
    x = train[inputs].to_numpy()
    classes = np.arange(3 if threshold is None else 2)
    if not np.array_equal(np.unique(y), classes):
        raise ValueError(f"Training data is missing classes for {target} {threshold}")
    estimator = frequency = None
    if model_type == "frequency":
        frequency = np.bincount(y) / len(y)
    elif model_type == "sgd":
        scaler, classifier, _ = fit_epoch_classifier(
            x, y, train["source_kickoff"].dt.normalize(), classes,
        )
        estimator = make_pipeline(scaler, classifier)
    else:
        estimator = (
            XGBClassifier(**XGB_SETTINGS) if model_type == "xgboost"
            else make_pipeline(StandardScaler(), LogisticRegression())
        )
        estimator.fit(x, y)
    return {
        "inputs": inputs, "estimator": estimator, "frequency": frequency,
        "training_end": train["source_kickoff"].max().date(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODELS)
    args = parser.parse_args()
    teams, team_inputs = team_dataset(Path("data/processed/features.csv"), Path("data/raw/football-data"))
    add_team_outcomes(teams)
    players, player_inputs = player_dataset(Path("data/raw/xgtables-understat"))
    models = {}
    for target, model_type in TEAM_MODELS.items():
        line = TEAM_THRESHOLDS.get(target)
        models[(target, line)] = fit_model(teams, team_inputs, target, model_type, line)
    for line in PLAYER_LINES:
        models[("shots", line)] = fit_model(players, player_inputs, "shots", "xgboost", line)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.output.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            pickle.dump(models, temporary)
            temporary.close()
            temporary_path.replace(args.output)
        finally:
            temporary_path.unlink(missing_ok=True)
    print(f"Saved {len(models)} trained models to {args.output}")


if __name__ == "__main__":
    main()
