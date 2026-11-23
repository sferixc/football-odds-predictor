"""Fit and chronologically evaluate domestic corner-count models."""

from pathlib import Path
import pickle
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import poisson

from ..data.dataset import final_test_split, validation_folds
from ..data.sporting import team_dataset
from ..models.corners import CORNER_LINES, CORNER_MODELS

RAW = Path("data/raw/football-data")
FEATURES = Path("data/processed/features.csv")
LEAGUES = ("E0", "D1", "SP1")
TARGETS = ("home_corners", "away_corners", "match_corners")
ALPHA = 1.0


def model_for(inputs: list[str]) -> object:
    """Use the same scaled Poisson count approach for fitting and prediction."""
    return make_pipeline(StandardScaler(), PoissonRegressor(alpha=ALPHA, max_iter=1000))


def brier_by_lines(actual: np.ndarray, probabilities: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Score the binary Over event at each half-line, then average over fixtures."""
    observed = actual[:, None] > lines[None, :]
    return np.square(probabilities - observed).mean(axis=0)


def evaluate(league: str, rows: pd.DataFrame, target: str, inputs: list[str]) -> None:
    lines = CORNER_LINES[target]
    fold_scores = []
    for season, earlier, later in validation_folds(rows):
        estimator = model_for(inputs).fit(earlier[inputs], earlier[target])
        predicted_mean = estimator.predict(later[inputs])
        probabilities = poisson.sf(lines[None, :], predicted_mean[:, None])
        score = brier_by_lines(later[target].to_numpy(), probabilities, lines)

        baseline_mean = earlier[target].mean()
        baseline_probabilities = poisson.sf(lines, baseline_mean)
        baseline_score = brier_by_lines(
            later[target].to_numpy(), np.broadcast_to(baseline_probabilities, probabilities.shape), lines,
        )
        fold_scores.append((score.mean(), baseline_score.mean()))
        print(f"{league} | {target} | {season}/{season + 1} | "
              f"Poisson Brier {score.mean():.4f} | mean-count baseline {baseline_score.mean():.4f}")

    model_mean = np.mean([score[0] for score in fold_scores])
    baseline_mean = np.mean([score[1] for score in fold_scores])
    print(f"{league} | {target} | validation mean | "
          f"Poisson {model_mean:.4f} | mean-count baseline {baseline_mean:.4f}")


def fit_league(league: str, rows: pd.DataFrame, inputs: list[str]) -> dict:
    league_rows = rows.loc[rows["league"].eq(league)].dropna(subset=[*inputs, *TARGETS])
    for target in TARGETS:
        evaluate(league, league_rows, target, inputs)

    train, test = final_test_split(league_rows)
    models = {}
    for target in TARGETS:
        estimator = model_for(inputs).fit(train[inputs], train[target])
        predicted_mean = estimator.predict(test[inputs])
        lines = CORNER_LINES[target]
        probabilities = poisson.sf(lines[None, :], predicted_mean[:, None])
        score = brier_by_lines(test[target].to_numpy(), probabilities, lines).mean()
        baseline_probabilities = poisson.sf(lines, train[target].mean())
        baseline_score = brier_by_lines(
            test[target].to_numpy(), np.broadcast_to(baseline_probabilities, probabilities.shape), lines,
        ).mean()
        print(f"{league} | {target} | held-out 2025/26 | "
              f"Poisson Brier {score:.4f} | mean-count baseline {baseline_score:.4f}")
        models[target] = {"estimator": estimator, "inputs": inputs}

    return {"training_end": train["source_kickoff"].max().date(), "models": models}


def main() -> None:
    rows, inputs = team_dataset(FEATURES, RAW)
    rows["match_corners"] = rows["home_corners"] + rows["away_corners"]
    bundle = {"leagues": {league: fit_league(league, rows, inputs) for league in LEAGUES}}
    save_bundle(CORNER_MODELS, bundle)


def save_bundle(destination: Path, bundle: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        path = Path(temporary.name)
        try:
            pickle.dump(bundle, temporary)
            temporary.close()
            path.replace(destination)
        finally:
            path.unlink(missing_ok=True)
    print(f"Saved corner-count models to {destination}")


if __name__ == "__main__":
    main()
