"""Evaluate card forecasts with chronological referee history."""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data.europe import INPUTS
from ..data.europe_cards import CARD_EVENTS, CARD_LINEUPS, SCOPES, card_targets
from ..models.ucl import MIN_REFEREE_MATCHES, referee_key

LINES = np.arange(0.5, 10, 1)


def fit_europe_cards(games, inputs):
    if not (CARD_EVENTS.is_file() and CARD_LINEUPS.is_file()):
        print("European card archive missing. See CONTRIBUTING.md.")
        return {}, {}
    rows = inputs.merge(card_targets(games), on="game_id", validate="many_to_one").sort_values("date")
    rows["official"] = rows["referee"].map(referee_key)
    days = rows.groupby(["official", "date"])["match"].agg(["sum", "count"])
    prior = days.groupby(level="official").cumsum() - days
    prior_mean = prior["sum"] / prior["count"].where(prior["count"].ge(MIN_REFEREE_MATCHES))
    rows = rows.join(prior_mean.rename("referee_mean"), on=["official", "date"])
    referee_history = {
        official: list(zip(group.index.get_level_values("date").date,
                           group["sum"].cumsum(), group["count"].cumsum()))
        for official, group in days.groupby(level="official")
    }
    models = {}
    for competition, subset in rows.groupby("competition"):
        early = subset.loc[subset.season.lt(2024)]
        paired_early = early.dropna(subset=["referee_mean"])
        validation = subset.loc[subset.season.eq(2024)].dropna(subset=["referee_mean"])
        train = subset.loc[subset.season.le(2024)]
        test = subset.loc[subset.season.eq(2025)]
        scopes = {}
        for scope in SCOPES:
            fitted = {}
            scores = {}
            for variant, columns in (("team", list(INPUTS)), ("referee", [*INPUTS, "referee_mean"])):
                model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0))
                model.fit(paired_early[columns].to_numpy(), paired_early[scope])
                scores[variant] = brier(validation[scope], model.predict(validation[columns].to_numpy()))
                available = train.dropna(subset=columns)
                model.fit(available[columns].to_numpy(), available[scope])
                fitted[variant] = model
            use_referee = scores["referee"] < scores["team"]
            with_ref = test["referee_mean"].notna().to_numpy()
            expected = fitted["team"].predict(test[list(INPUTS)].to_numpy())
            if use_referee and with_ref.any():
                expected[with_ref] = fitted["referee"].predict(
                    test.loc[with_ref, [*INPUTS, "referee_mean"]].to_numpy(),
                )
            print(f"{competition} cards {scope} | {len(train)} train / {len(test)} test | "
                  f"2024/25 paired Brier {scores['team']:.4f}/{scores['referee']:.4f} "
                  f"({len(validation)} matches) | selected {'referee' if use_referee else 'team'} | "
                  f"test {brier(test[scope], expected):.4f}")
            scopes[scope] = {**fitted, "use_referee": use_referee}
        models[competition] = scopes
    archived_count = inputs["season"].le(2025).sum()
    print(f"European card labels: {len(rows)} of {archived_count} archived league-stage fixtures")
    return models, referee_history


def brier(labels: pd.Series, means: np.ndarray) -> float:
    observed = labels.to_numpy()[:, None] > LINES
    predicted = poisson.sf(LINES[None, :], means[:, None])
    return float(np.square(observed - predicted).mean())
