"""Report independent 1X2 probabilities alongside one captured Superbet offer."""

import argparse
from datetime import timedelta
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..data.dataset import FIRST_SEASON, TEST_SEASON
from ..data.features import build_features
from ..data.preprocessing import load_matches
from ..data.sporting import read_team_matches, team_dataset, team_features
from ..data.superbet_snapshot import check_window, open_price, simple_promotions, utc_timestamp
from ..training.sporting import add_team_outcomes, fit_epoch_classifier
from .offer import assess_offer

LEAGUES = {
    106: ("E0", "Europe/London"),
    245: ("D1", "Europe/Berlin"),
    98: ("SP1", "Europe/Madrid"),
}
RAW = Path("data/raw")


def read_snapshot(path: Path, probe: bool) -> tuple[dict, str]:
    """Accept timed captures by default and require an opt-in for untimed probes."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "event" in data:
        event = data["event"]
        observed = utc_timestamp(data["observed_at_utc"])
        check_window(utc_timestamp(event["fixture"]["utc_date"]), observed)
        return event, observed.isoformat()
    if probe and len(data.get("events", [])) == 1:
        return data["events"][0], "untimed probe"
    raise ValueError("Expected a timed snapshot. Use --probe for a single untimed test response.")


def predict_fixture(event: dict) -> tuple[np.ndarray, list[pd.Timestamp]]:
    fixture = event["fixture"]
    league, zone = LEAGUES[int(fixture["tournament_id"])]
    kickoff = utc_timestamp(fixture["utc_date"]).astimezone(ZoneInfo(zone))
    cutoff = pd.Timestamp(kickoff.date())
    home, away = fixture["event_name"].split("·", maxsplit=1)

    historical = read_team_matches(RAW / "football-data")
    recent = read_team_matches(RAW / "football-data-2627")
    history = pd.concat([historical, recent], ignore_index=True)
    history = history.loc[
        history["league"].eq(league) & history["source_kickoff"].lt(cutoff)
        & history["home_goals"].notna() & history["away_goals"].notna()
    ]
    last_matches = []
    for team in (home, away):
        appearances = history.loc[history["home_team"].eq(team) | history["away_team"].eq(team)]
        if appearances.empty:
            raise ValueError(f"Team {team!r} could not be matched to the results archive")
        last_matches.append(appearances["source_kickoff"].max())
    season = kickoff.year - int(kickoff.month < 7)
    upcoming = pd.DataFrame([{
        "league": league, "season_start_year": season,
        "source_kickoff": pd.Timestamp(kickoff.replace(tzinfo=None)),
        "home_team": home, "away_team": away,
    }])
    # A future fixture contributes an empty result. Earlier completed games provide its form and Elo.
    matches = pd.concat([history, upcoming], ignore_index=True)
    warmup = load_matches(RAW / "football-data-warmup")
    ratings = build_features(matches, elo_warmup=warmup)
    feature_rows, inputs = team_features(matches, ratings)
    current = feature_rows.iloc[[-1]][inputs]
    if current.isna().any().any():
        raise ValueError("The fixture is missing complete five-match inputs")

    dataset, _ = team_dataset(Path("data/processed/features.csv"), RAW / "football-data")
    add_team_outcomes(dataset)
    train = dataset.loc[
        dataset["season_start_year"].ge(FIRST_SEASON)
        & dataset["season_start_year"].lt(TEST_SEASON)
        & dataset["source_kickoff"].lt(cutoff)
    ].dropna(subset=[*inputs, "1x2"])
    scaler, model, _ = fit_epoch_classifier(
        train[inputs].to_numpy(), train["1x2"].to_numpy(dtype=int),
        train["source_kickoff"].dt.normalize(), np.arange(3),
    )
    return model.predict_proba(scaler.transform(current.to_numpy()))[0], last_matches


def print_report(event: dict, observed: str, stress: Decimal | None) -> None:
    probabilities, last_matches = predict_fixture(event)
    fixture = event["fixture"]
    print(f"{fixture['event_name'].replace('·', ' – ')} | kickoff {fixture['utc_date']}")
    print(f"Offer: {observed} | Latest results used: {' / '.join(str(d.date()) for d in last_matches)}")
    if observed == "untimed probe":
        print("Replay of an untimed offer. Probabilities use the local results archive.")
    if stress is not None:
        print(f"Provisional color uses a {stress * 100:g} percentage-point reduction.")
    print("Return is calculated at the higher displayed price.")
    print(f"{'1X2':<4} {'Win':>7} {'Loss':>7} {'Fair':>7} {'Superbet':>9} {'SuperCota':>10} "
          f"{'Est. return':>12} {'Prudent':>9} {'Color':>8}")
    prices = {
        odd["metadata"]["name"]: odd["price"]
        for market in event["markets"] if market["id"] == 547
        for odd in market["odds"] if open_price(odd)
    }
    promotions = {
        offer["selection"]: offer["odds"]
        for offer in simple_promotions(event) if offer["type"] == "super_cota"
    }
    for selection, probability in zip(("1", "X", "2"), probabilities):
        p = Decimal(str(probability))
        standard, promoted = prices.get(selection), promotions.get(selection)
        available = [price for price in (standard, promoted) if price is not None]
        returns = conservative = signal = "—"
        if available:
            score = assess_offer(
                odds=Decimal(str(max(available))), win_probability=p,
                conservative_win_probability=p if stress is None else max(Decimal(0), p - stress),
                stake=Decimal(1), budget=Decimal(1),
            )
            returns = f"{score.expected_return:+.1%}"
            if stress is not None:
                conservative, signal = f"{score.conservative_return:+.1%}", score.signal
        standard_text = f"{standard:.2f}" if standard is not None else "—"
        promoted_text = f"{promoted:.2f}" if promoted is not None else "—"
        print(f"{selection:<4} {p:>7.1%} {1 - p:>7.1%} {1 / p:>7.2f} "
              f"{standard_text:>9} {promoted_text:>10} "
              f"{returns:>12} {conservative:>9} {signal:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--probe", action="store_true", help="Allow a single untimed test response")
    parser.add_argument("--stress-pp", type=Decimal, help="Optional probability reduction in percentage points")
    args = parser.parse_args()
    stress = args.stress_pp
    if stress is not None and (not stress.is_finite() or not 0 < stress < 100):
        parser.error("--stress-pp must be between 0 and 100")
    try:
        event, observed = read_snapshot(args.snapshot, args.probe)
        print_report(event, observed, stress / 100 if stress is not None else None)
    except (InvalidOperation, KeyError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
