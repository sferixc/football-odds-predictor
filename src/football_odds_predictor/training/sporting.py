"""Compare sporting probability models on expanding chronological folds.

The historical frequency is the reference score. Each candidate predicts
probabilities for the same validation fixtures for a given selection.
"""

from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ..data.dataset import validation_folds
from ..data.sporting import player_dataset, team_dataset

SEED = 42
MAX_EPOCHS = 30
PATIENCE = 5
SGD_SETTINGS = {
    "loss": "log_loss", "random_state": SEED, "shuffle": False,
    "average": True, "learning_rate": "constant", "eta0": 0.01, "alpha": 0.001,
}
XGB_SETTINGS = {
    "n_estimators": 150, "max_depth": 2, "learning_rate": 0.05,
    "random_state": SEED, "n_jobs": 1,
}
TEAM_THRESHOLDS = {
    "match_goals": 2.5,
    "first_half_goals": 0.5,
    "second_half_goals": 1.5,
    "home_yellows": 2.5,
    "away_yellows": 2.5,
    "match_yellows": 4.5,
    "home_corners": 4.5,
    "away_corners": 4.5,
    "match_corners": 9.5,
}


def add_team_outcomes(rows: pd.DataFrame) -> None:
    """Derive labels from the match itself, for evaluation and model fitting."""
    difference = rows["home_goals"] - rows["away_goals"]
    # The 1X2 class order is home win, draw, away win.
    rows["1x2"] = np.select([difference.gt(0), difference.eq(0)], [0, 1], default=2)
    rows.loc[difference.isna(), "1x2"] = np.nan
    for scope, stat in (
        ("match", "goals"), ("first_half", "goals"), ("second_half", "goals"),
        ("match", "yellows"), ("match", "corners"),
    ):
        suffix = stat if scope == "match" else f"{scope}_{stat}"
        rows[f"{scope}_{stat}"] = rows[f"home_{suffix}"] + rows[f"away_{suffix}"]


def epoch_predictions(
    x: np.ndarray, y: np.ndarray, dates: pd.Series, outer_x: np.ndarray, classes: np.ndarray,
) -> tuple[np.ndarray, int]:
    scaler, model, epoch = fit_epoch_classifier(x, y, dates, classes)
    return model.predict_proba(scaler.transform(outer_x)), epoch


def fit_epoch_classifier(
    x: np.ndarray, y: np.ndarray, dates: pd.Series, classes: np.ndarray,
) -> tuple[StandardScaler, SGDClassifier, int]:
    """Choose an epoch on the last fifth of training dates, then refit on the fold."""
    sorted_days = dates.drop_duplicates().sort_values()
    # Pick one date boundary so that fixtures on the same day stay together.
    inner_validation = dates.ge(sorted_days.iloc[int(len(sorted_days) * 0.8)]).to_numpy()
    scaler = StandardScaler().fit(x[~inner_validation])
    fit_x = scaler.transform(x[~inner_validation])
    validation_x = scaler.transform(x[inner_validation])
    fit_y, validation_y = y[~inner_validation], y[inner_validation]
    model = SGDClassifier(**SGD_SETTINGS)
    rng = np.random.default_rng(SEED)
    best_loss = np.inf
    best_epoch = 1
    for epoch in range(1, MAX_EPOCHS + 1):
        # Shuffle the earlier rows within each epoch. Later dates score the epoch.
        order = rng.permutation(len(fit_y))
        model.partial_fit(fit_x[order], fit_y[order], classes=classes)
        loss = log_loss(validation_y, model.predict_proba(validation_x), labels=classes)
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
        if epoch - best_epoch >= PATIENCE:
            break

    # Refit from scratch on the entire training fold for the selected number of epochs.
    scaler = StandardScaler().fit(x)
    fit_x = scaler.transform(x)
    model = SGDClassifier(**SGD_SETTINGS)
    rng = np.random.default_rng(SEED)
    for _ in range(best_epoch):
        order = rng.permutation(len(y))
        model.partial_fit(fit_x[order], y[order], classes=classes)
    return scaler, model, best_epoch


def evaluate_market(
    rows: pd.DataFrame, inputs: list[str], target: str, threshold: float | None = None,
) -> None:
    """Score frequency, logistic regression, epoch-trained SGD, and XGBoost."""
    if rows["source_kickoff"].isna().any():
        raise ValueError("Expected a date for every observation")
    eligible = rows.dropna(subset=[*inputs, target])
    classes = np.arange(3 if threshold is None else 2)
    fold_scores = []
    for season, train, validation in validation_folds(eligible):
        x = train[inputs].to_numpy()
        validation_x = validation[inputs].to_numpy()
        train_target, actual = train[target].to_numpy(), validation[target].to_numpy()
        if threshold is not None:
            # A line of 1.5 shots becomes a yes/no label for two or more shots.
            train_target, actual = train_target > threshold, actual > threshold
        y, actual = train_target.astype(int), actual.astype(int)
        if not np.array_equal(np.unique(y), classes):
            raise ValueError(f"Missing training classes for {target} in {season}")
        frequency = np.bincount(y) / len(y)
        baseline = np.broadcast_to(frequency, (len(actual), len(classes)))
        logistic = make_pipeline(StandardScaler(), LogisticRegression())
        logistic.fit(x, y)
        gradient, epoch = epoch_predictions(
            x, y, train["source_kickoff"].dt.normalize(), validation_x, classes,
        )
        trees = XGBClassifier(**XGB_SETTINGS)
        trees.fit(x, y)
        predictions = (
            baseline, logistic.predict_proba(validation_x), gradient, trees.predict_proba(validation_x),
        )
        scores = [log_loss(actual, p, labels=classes) for p in predictions]
        expected = np.eye(len(classes))[actual]
        # For binary events, report Brier on the probability of the event.
        brier = [np.square(p - expected).sum(axis=1).mean() / (2 if len(classes) == 2 else 1)
                 for p in predictions]
        fold_scores.append([*scores, *brier])
        print(
            f"{target} {threshold if threshold is not None else ''} | {season} | "
            f"{len(train)}/{len(validation)} | log loss "
            f"{'/'.join(f'{score:.4f}' for score in scores)} | epoch {epoch}",
            flush=True,
        )
        if target == "1x2" and season == 2022:
            match = validation["home_team"].eq("Arsenal") & validation["away_team"].eq("Chelsea")
            if match.any():
                probabilities = predictions[1][match.to_numpy()][0]
                print(f"Arsenal–Chelsea logistic 1/X/2: {probabilities.round(4).tolist()}", flush=True)

    averages = [mean(column) for column in zip(*fold_scores)]
    print(f"Mean | {target} {threshold if threshold is not None else ''} | "
          f"log loss {'/'.join(f'{x:.4f}' for x in averages[:4])} | "
          f"Brier {'/'.join(f'{x:.4f}' for x in averages[4:])}", flush=True)


def main() -> None:
    print("Model order: historical frequency / logistic regression / epoch-trained SGD / XGBoost",
          flush=True)
    teams, team_inputs = team_dataset(
        Path("data/processed/features.csv"), Path("data/raw/football-data"),
    )
    add_team_outcomes(teams)
    evaluate_market(teams, team_inputs, "1x2")
    for target, threshold in TEAM_THRESHOLDS.items():
        evaluate_market(teams, team_inputs, target, threshold)

    players, player_inputs = player_dataset(Path("data/raw/xgtables-understat"))
    for threshold in (0.5, 1.5, 2.5):
        evaluate_market(players, player_inputs, "shots", threshold)


if __name__ == "__main__":
    main()
