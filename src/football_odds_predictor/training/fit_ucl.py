"""Fit and audit UCL and Europa League forecasts with shared club history."""

import argparse
from collections import Counter
from pathlib import Path
import pickle
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import log_loss, mean_poisson_deviance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data.europe import CURRENT_LEAGUES, EUROPE, INPUTS, build_europe_history, read_europe_games
from ..data.europe_elo import COUNTRY_CODES, ELO_ARCHIVE, EuropeanEloSeeds
from ..models.ucl import UCL_MODELS, club_key
from .europe_cards import fit_europe_cards

COVERAGE = Path("data/processed/europe-coverage.csv")


def fit_competition(rows, european_only, competition):
    train = rows.loc[rows.season.between(2021, 2024)]
    test = rows.loc[rows.season.eq(2025)]
    old_train = european_only.loc[european_only.season.between(2021, 2024)]
    old_test = european_only.loc[european_only.season.eq(2025)]
    inputs = list(INPUTS)
    result = make_pipeline(StandardScaler(), LogisticRegression())
    result.fit(train[inputs].to_numpy(), train.result)
    old_result = make_pipeline(StandardScaler(), LogisticRegression())
    old_result.fit(old_train[inputs[:3]].to_numpy(), old_train.result)

    goals = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.5))
    earlier, validation = train.loc[train.season.lt(2024)], train.loc[train.season.eq(2024)]
    goals.fit(earlier[inputs].to_numpy(), earlier.goals)
    validation_score = mean_poisson_deviance(validation.goals, goals.predict(validation[inputs].to_numpy()))
    mean_score = mean_poisson_deviance(validation.goals, np.full(len(validation), earlier.goals.mean()))
    if mean_score <= validation_score:
        goals = DummyRegressor()
    goals.fit(train[inputs].to_numpy(), train.goals)
    x = test[inputs].to_numpy()
    print(f"{competition} | {len(train)} train / {len(test)} held-out")
    print(f"  1X2 log loss: {log_loss(test.result, result.predict_proba(x)):.4f} combined, "
          f"{log_loss(old_test.result, old_result.predict_proba(old_test[inputs[:3]].to_numpy())):.4f} Europe-only")
    print(f"  Goals deviance: {mean_poisson_deviance(test.goals, goals.predict(x)):.4f} "
          f"({type(goals).__name__}; selected on 2024/25)")
    return {"result": result, "goals": goals}


def score_seeds(rows, previous):
    """Compare identical 1X2 models on identical chronological validation fixtures."""
    keys = ["game_id", "date", "competition", "season"]
    if not rows.loc[rows.season.between(2021, 2025), keys].reset_index(drop=True).equals(
        previous.loc[previous.season.between(2021, 2025), keys].reset_index(drop=True)
    ):
        raise ValueError("Elo variants must use the same historical fixtures")

    for competition in sorted(EUROPE):
        for season in (2022, 2023, 2024, 2025):
            scores = []
            for dataset in (previous, rows):
                fixtures = dataset.loc[dataset.competition.eq(competition)]
                train = fixtures.loc[fixtures.season.between(2021, season - 1)]
                validation = fixtures.loc[fixtures.season.eq(season)]
                estimator = make_pipeline(StandardScaler(), LogisticRegression())
                estimator.fit(train[list(INPUTS)].to_numpy(), train.result)
                scores.append(log_loss(validation.result, estimator.predict_proba(
                    validation[list(INPUTS)].to_numpy(),
                )))
            split = "holdout" if season == 2025 else "validation"
            print(f"{competition} {season}/{season + 1} {split} 1X2 log loss "
                  f"| former Elo {scores[0]:.4f} | dated Elo {scores[1]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elo-source", type=Path, default=ELO_ARCHIVE)
    args = parser.parse_args()
    if not args.elo_source.is_dir():
        parser.error(f"Elo archive missing: {args.elo_source}. See CONTRIBUTING.md for the pinned download.")
    games, names = read_europe_games(club_key)
    seeds = EuropeanEloSeeds(args.elo_source, names, club_key,
                             {league: COUNTRY_CODES[country] for country, league in CURRENT_LEAGUES.items()})
    audit = {}
    rows, timeline = build_europe_history(games, seeds=seeds, initial_audit=audit)
    previous, _ = build_europe_history(games)
    print(f"Initial Elo coverage: {dict(Counter(item[2] for item in audit.values()))}")
    for name in ("OFI Crete", "Manchester City"):
        club = names.get(club_key(name))
        if club in audit:
            debut, elo, source, as_of = audit[club]
            print(f"{name}: {elo:.2f} ({source}, {as_of}) before {debut}")
    score_seeds(rows, previous)
    baseline, _ = build_europe_history(games, mixed=False)
    models = {competition: fit_competition(
        rows.loc[rows.competition.eq(competition)], baseline.loc[baseline.competition.eq(competition)], competition,
    ) for competition in sorted(EUROPE)}

    card_models, referee_cards = fit_europe_cards(games, rows)
    for competition, cards in card_models.items():
        models[competition]["cards"] = cards

    latest = games.loc[games.season.ge(2024)]
    european = latest.loc[latest.competition_id.isin(EUROPE)]
    domestic = latest.loc[latest.competition_type.eq("domestic_league")]
    records = []
    for side in ("home", "away"):
        records.append(domestic[[f"{side}_club_id", "competition_id", "date"]].rename(
            columns={f"{side}_club_id": "club_id", "date": "domestic_through"},
        ))
    leagues = pd.concat(records).sort_values("domestic_through").drop_duplicates("club_id", keep="last")
    clubs = pd.concat([
        european[[f"{side}_club_id", f"{side}_club_name"]].set_axis(["club_id", "club"], axis=1)
        for side in ("home", "away")
    ]).drop_duplicates("club_id")
    coverage = clubs.merge(leagues, on="club_id", how="left").sort_values("club")
    COVERAGE.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(COVERAGE, index=False)
    print(f"Domestic coverage since 2024/25: {coverage.competition_id.notna().sum()}/{len(coverage)} clubs "
          f"across {coverage.competition_id.nunique()} leagues")
    for competition, group in european.groupby("competition_id"):
        known = group.referee.fillna("").str.strip().ne("").sum()
        print(f"{competition} historical referee coverage: {known}/{len(group)}")
    bundle = {"names": names, "referee_cards": referee_cards, "timeline": timeline, "competitions": models,
              "elo_initialization": audit,
              "domestic": set(leagues.club_id), "training_end": rows.loc[rows.season.le(2024), "season"].max()}
    with NamedTemporaryFile(dir=UCL_MODELS.parent, delete=False) as temporary:
        path = Path(temporary.name)
        try:
            pickle.dump(bundle, temporary)
            temporary.close()
            path.replace(UCL_MODELS)
        finally:
            path.unlink(missing_ok=True)
    print(f"Saved {UCL_MODELS} and {COVERAGE}")


if __name__ == "__main__":
    main()
