"""Chronological team ratings and inference for European league-stage markets."""

from bisect import bisect_left
from pathlib import Path
import pickle
import re
import unicodedata

from scipy.stats import poisson

from ..data.europe import pre_match_inputs

UCL_MODELS = Path("data/processed/ucl-models.pkl")


def club_key(name: str) -> str:
    name = "".join(c for c in unicodedata.normalize("NFKD", name.casefold()) if not unicodedata.combining(c))
    name = re.sub(r"^(?:fc|cf|ac) | (?:fc|cf|ac)$", "", name)
    return re.sub(r"[^a-z0-9]+", "", name)


def load_ucl_models(models: dict) -> None:
    if UCL_MODELS.is_file():
        with UCL_MODELS.open("rb") as handle:
            models["ucl"] = pickle.load(handle)


def referee_key(name):
    return club_key(name) if isinstance(name, str) and name.strip() else None


def predict_ucl(bundle, fixture, kickoff, competition="CL", referee=None):
    """Read each club's shared rating and form strictly before kickoff day."""
    season = kickoff.year - (kickoff.month < 7)
    if (bundle is None or "competitions" not in bundle or season <= bundle["training_end"]
            or kickoff.month not in (9, 10, 11, 12, 1)):
        return None
    teams = fixture["event_name"].split("·")
    if len(teams) != 2:
        return None
    home, away = (bundle["names"].get(club_key(name.strip())) for name in teams)
    if home is None or away is None or home == away:
        return None
    ratings, history, last_dates = {}, {}, []
    for team in (home, away):
        timeline = bundle["timeline"].get(team, ())
        index = bisect_left(timeline, kickoff.date(), key=lambda row: row[0])
        if index < 1:
            return None
        day, ratings[team], history[team] = timeline[index - 1]
        if len(history[team]) < 5 or (kickoff.date() - day).days > 60:
            return None
        last_dates.append(str(day))
    x = [pre_match_inputs(home, away, ratings, history)]
    models = bundle["competitions"][competition]
    card_forecasts = {}
    referee_history = bundle.get("referee_cards", {}).get(referee_key(referee), ())
    ref_index = bisect_left(referee_history, kickoff.date(), key=lambda row: row[0])
    prior = referee_history[ref_index - 1] if ref_index else None
    for scope, variants in models.get("cards", {}).items():
        use_referee = bool(variants["use_referee"] and prior and prior[2] >= 3)
        if use_referee:
            expected = variants["referee"].predict([[*x[0], prior[1] / prior[2]]])[0]
        else:
            expected = variants["team"].predict(x)[0]
        card_forecasts[scope] = (float(expected), use_referee)
    return (
        models["result"].predict_proba(x)[0], float(models["goals"].predict(x)[0]),
        last_dates, all(team in bundle["domestic"] for team in (home, away)), card_forecasts,
    )


def goal_probabilities(line: float, expected_goals: float) -> tuple[float, float]:
    return float(poisson.cdf(line, expected_goals)), float(poisson.sf(line, expected_goals))
