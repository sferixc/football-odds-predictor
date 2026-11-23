"""Prepare the match rows that the model experiments share.

load_dataset() pairs each match's inputs with its three actual closing odds.
validation_folds() hands the experiment one training and validation pair at a time.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FIXTURE_COLUMNS
from .preprocessing import TARGET_ODDS

TARGET_COLUMNS = list(TARGET_ODDS.values())
FIRST_SEASON = 2021
TEST_SEASON = 2025


def load_dataset(features_path: Path, matches_path: Path) -> pd.DataFrame:
    """Return one row per fixture with a complete history and three valid odds.

    The inputs come from features.csv. From matches.csv, read only the columns
    that identify the fixture and the Bet365 odds the models aim to predict.
    """
    features = pd.read_csv(features_path, parse_dates=["source_kickoff"])
    matches = pd.read_csv(
        matches_path,
        usecols=[*FIXTURE_COLUMNS, *TARGET_COLUMNS],
        parse_dates=["source_kickoff"],
    )
    return join_dataset(features, matches)


def join_dataset(features: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Pair features and closing odds by fixture, then apply V1 eligibility."""
    fixture_key = list(FIXTURE_COLUMNS)
    matches = matches[[*fixture_key, *TARGET_COLUMNS]]
    if features[fixture_key].isna().any().any() or matches[fixture_key].isna().any().any():
        raise ValueError("Both CSVs must contain complete fixture keys")
    # Match by league, season, kickoff, and both teams so CSV row order can differ.
    dataset = features.merge(matches, on=fixture_key, validate="one_to_one")
    if len(dataset) != len(features) or len(dataset) != len(matches):
        raise ValueError("Features and matches must contain the same fixtures")

    odds = dataset[TARGET_COLUMNS]
    valid_odds = (np.isfinite(odds) & odds.gt(1)).all(axis=1)
    eligible = dataset.loc[dataset["history_complete"].eq(True) & valid_odds]
    return eligible.sort_values("source_kickoff", ignore_index=True)


def validation_folds(
    dataset: pd.DataFrame,
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield (validation season, training rows, validation rows) for each fold.

    The first fold learns from 2021/22 and scores 2022/23. The next fold adds
    2022/23 to training and scores 2023/24. The 2025/26 season is held for testing.
    """
    for season in range(FIRST_SEASON + 1, TEST_SEASON):
        train, validation = _split_at_season(dataset, season)
        yield season, train, validation


def final_test_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on 2021/22 through 2024/25 and score the held-out 2025/26 season."""
    return _split_at_season(dataset, TEST_SEASON)


def _split_at_season(
    dataset: pd.DataFrame, season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = dataset["season_start_year"]
    train = dataset.loc[seasons.ge(FIRST_SEASON) & seasons.lt(season)]
    evaluation = dataset.loc[seasons.eq(season)]
    if train.empty or evaluation.empty:
        raise ValueError(f"Missing training or evaluation data for season {season}")
    if train["source_kickoff"].max() >= evaluation["source_kickoff"].min():
        raise ValueError(f"Training and evaluation dates overlap for season {season}")
    return train, evaluation
