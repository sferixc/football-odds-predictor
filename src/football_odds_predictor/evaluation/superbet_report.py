"""Report supported football selections alongside one captured Superbet offer."""

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import pickle
import re
import unicodedata
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..data.features import build_features
from ..data.preprocessing import load_matches
from ..data.sporting import (
    read_player_appearances, read_team_matches, team_features,
)
from ..data.superbet_snapshot import check_window, open_price, simple_promotions, utc_timestamp
from ..models.sporting import predict_saved as predict_saved
from ..training.fit_sporting import DEFAULT_MODELS, PLAYER_LINES
from ..training.sporting import TEAM_THRESHOLDS
from .offer import assess_offer

LEAGUES = {
    106: ("E0", "Europe/London", "epl"),
    245: ("D1", "Europe/Berlin", "bundesliga"),
    98: ("SP1", "Europe/Madrid", "la-liga"),
}
RAW = Path("data/raw")
TEAM_MARKETS = {
    "goals": (
        (200734, "match_goals", "Match goals"),
        (200735, "first_half_goals", "First-half goals"),
        (200738, "second_half_goals", "Second-half goals"),
    ),
    "cards": (
        (None, "home_yellows", "Home yellow cards"),
        (None, "away_yellows", "Away yellow cards"),
        (None, "match_yellows", "Match yellow cards"),
    ),
    "corners": (
        (704, "match_corners", "Match corners"),
        (713, "home_corners", "Home corners"),
        (733, "away_corners", "Away corners"),
    ),
}
GROUPS = ("1x2", "goals", "cards", "corners", "shots")


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


def team_inputs_for_event(event: dict) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    fixture = event["fixture"]
    league, zone, _ = LEAGUES[int(fixture["tournament_id"])]
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

    return current, last_matches


def line_selection(name: str, specifiers: dict, threshold: float) -> str | None:
    """Confirm both the exact line and the direction in the provider response."""
    match = re.search(r"(?:^| - )(Peste|Sub) ([\d.,]+)$", name)
    line = specifiers.get("total")
    if match and line is not None and Decimal(str(line)) == Decimal(str(threshold)):
        if Decimal(match[2].replace(",", ".")) == Decimal(str(threshold)):
            return match[1]
    return None


def team_prices(event: dict, market_id: int | None, target: str, line: float,
                promotions: list[dict]) -> tuple[dict, dict]:
    standard, promoted = {}, {}
    for market in event["markets"]:
        if market_id is None or market["id"] != market_id:
            continue
        for odd in market["odds"]:
            metadata = odd["metadata"]
            side = line_selection(metadata["name"], metadata.get("specifiers") or {}, line)
            if side and open_price(odd):
                standard[side] = odd["price"]
    for offer in promotions:
        if offer["market"] == target:
            side = line_selection(offer["selection"], offer.get("specifiers") or {}, line)
            if side:
                promoted[side] = max(promoted.get(side, 0), offer["odds"])
    return standard, promoted


def _player_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name.casefold())
    return "".join(letter for letter in decomposed if letter.isalnum())


def player_report(event: dict, player_filter: str | None, promotions: list[dict], models: dict) -> None:
    """Show conditional shot probabilities and prices without scoring settlement risk."""
    _, zone, league = LEAGUES[int(event["fixture"]["tournament_id"])]
    kickoff = utc_timestamp(event["fixture"]["utc_date"]).astimezone(ZoneInfo(zone))
    cutoff = pd.Timestamp(kickoff.date())
    history = read_player_appearances(RAW / "xgtables-understat")
    history = history.loc[history["source_kickoff"].lt(cutoff)]
    league_history = history.loc[history["league"].eq(league)].copy()
    league_history["name_key"] = league_history["player"].map(_player_name)
    offers = []
    for market in event["markets"]:
        if market["id"] != 236218:
            continue
        for odd in market["odds"]:
            metadata = odd["metadata"]
            specifiers = metadata.get("specifiers") or {}
            name = specifiers.get("player_name")
            team = metadata.get("extra", {}).get("team")
            if name and team in ("1", "2") and open_price(odd):
                if player_filter is None or _player_name(player_filter) in _player_name(name):
                    offers.append((name, team, metadata["name"], specifiers, odd["price"]))

    rows = {}
    skipped = set()
    for name, team, *_ in offers:
        if name in rows or name in skipped:
            continue
        appearances = league_history.loc[league_history["name_key"].eq(_player_name(name))]
        recent = appearances.sort_values("source_kickoff").tail(5)
        if appearances["player_id"].nunique() != 1 or len(recent) < 5 or recent["minutes"].mean() < 45:
            skipped.add(name)
            continue
        rows[name] = {
            "at_home": int(team == "1"), "mean_minutes": recent["minutes"].mean(),
            "mean_shots": recent["shots"].mean(), "last_match": recent["source_kickoff"].max().date(),
        }

    print("\n[Player total shots] Probabilities conditional on playing. Offer value and color pending settlement checks.")
    if not rows:
        print("No offered players with an unambiguous name and enough prior minutes.")
        return
    features = pd.DataFrame.from_dict(rows, orient="index")
    print(f"{'Player and selection':<43} {'Prob.':>7} {'Superbet':>9} {'Promo':>8} {'Last result':>12}")
    for threshold in PLAYER_LINES:
        probabilities = dict(zip(features.index, predict_saved(
            models, features, "shots", cutoff.date(), threshold,
        )[:, 1]))
        boosted = {}
        for offer in promotions:
            if offer["market"] != "player_total_shots":
                continue
            specifiers = offer.get("specifiers") or {}
            side = line_selection(offer["selection"], specifiers, threshold)
            player_id = specifiers.get("player_id")
            if side and player_id:
                key = (player_id, side)
                boosted[key] = max(boosted.get(key, 0), offer["odds"])
        for name, _, label, specifiers, price in offers:
            side = line_selection(label, specifiers, threshold)
            if side and name in probabilities:
                probability = probabilities[name] if side == "Peste" else 1 - probabilities[name]
                promo = boosted.get((specifiers.get("player_id"), side))
                promo_text = f"{promo:.2f}" if promo is not None else "—"
                print(f"{label:<43} {probability:>7.1%} {price:>9.2f} "
                      f"{promo_text:>8} {str(rows[name]['last_match']):>12}")
    if skipped:
        print(f"Skipped {len(skipped)} players with ambiguous identity or insufficient prior minutes.")


def print_selections(
    label: str, selections: tuple[str, ...], probabilities: np.ndarray,
    prices: dict, promotions: dict, stress: Decimal | None,
) -> None:
    print(f"\n[{label}]")
    print(f"{'Selection':<10} {'Win':>7} {'Loss':>7} {'Fair':>7} {'Superbet':>9} {'Promo':>8} "
          f"{'Est. return':>12} {'Prudent':>9} {'Color':>8}")
    for selection, probability in zip(selections, probabilities):
        p = Decimal(str(probability))
        standard, promoted = prices.get(selection), promotions.get(selection)
        available = [price for price in (standard, promoted) if price is not None]
        returns = conservative = signal = "—"
        if available and p > 0:
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
        fair = f"{1 / p:.2f}" if p > 0 else "—"
        print(f"{selection:<10} {p:>7.1%} {1 - p:>7.1%} {fair:>7} "
              f"{standard_text:>9} {promoted_text:>8} "
              f"{returns:>12} {conservative:>9} {signal:>8}")


def print_report(
    event: dict, observed: str, stress: Decimal | None, models: dict,
    groups: tuple[str, ...] = GROUPS, player_filter: str | None = None,
) -> None:
    fixture = event["fixture"]
    if int(fixture["tournament_id"]) not in LEAGUES:
        raise ValueError("This report covers Premier League, Bundesliga, and La Liga")
    print(f"{fixture['event_name'].replace('·', ' – ')} | kickoff {fixture['utc_date']}")
    print(f"Offer: {observed}")
    if observed == "untimed probe":
        print("Replay of an untimed offer. Probabilities use the local results archive.")
    if stress is not None:
        print(f"Provisional color uses a {stress * 100:g} percentage-point reduction.")
    print("Return is calculated at the higher displayed price.")
    promotions = simple_promotions(event)
    zone = LEAGUES[int(fixture["tournament_id"])][1]
    cutoff = utc_timestamp(fixture["utc_date"]).astimezone(ZoneInfo(zone)).date()

    if any(group != "shots" for group in groups):
        current, last_matches = team_inputs_for_event(event)
        print(f"Latest team results used (home / away): {' / '.join(str(d.date()) for d in last_matches)}")
        if "1x2" in groups:
            prices = {
                odd["metadata"]["name"]: odd["price"]
                for market in event["markets"] if market["id"] == 547
                for odd in market["odds"] if open_price(odd)
            }
            boosted = {}
            for offer in promotions:
                if offer["market"] == "1x2":
                    selection = offer["selection"]
                    boosted[selection] = max(boosted.get(selection, 0), offer["odds"])
            print_selections(
                "Full-time result", ("1", "X", "2"),
                predict_saved(models, current, "1x2", cutoff)[0], prices, boosted, stress,
            )

        for group in groups:
            for market_id, target, label in TEAM_MARKETS.get(group, ()):
                threshold = TEAM_THRESHOLDS[target]
                standard, boosted = team_prices(event, market_id, target, threshold, promotions)
                probabilities = predict_saved(models, current, target, cutoff, threshold)[0]
                print_selections(
                    f"{label} ({threshold:g})", ("Sub", "Peste"), probabilities,
                    standard, boosted, stress,
                )
            if group == "cards":
                print("Yellow-card probabilities only. Superbet total cards has a separate settlement definition.")
    if "shots" in groups:
        player_report(event, player_filter, promotions, models)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, nargs="?", help="Defaults to the most recently saved capture")
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS, help="Locally trained model file")
    parser.add_argument("--probe", action="store_true", help="Allow a single untimed test response")
    parser.add_argument("--stress-pp", type=Decimal, help="Optional probability reduction in percentage points")
    parser.add_argument("--market", action="append", choices=GROUPS, help="Show only this market group")
    parser.add_argument("--player", help="Filter player shot selections by partial name")
    args = parser.parse_args()
    stress = args.stress_pp
    if stress is not None and (not stress.is_finite() or not 0 < stress < 100):
        parser.error("--stress-pp must be between 0 and 100")
    try:
        snapshot = args.snapshot
        if snapshot is None:
            snapshot = max(
                (RAW / "superbet").glob("*.json"),
                key=lambda path: (path.stat().st_mtime_ns, path.name),
                default=None,
            )
            if snapshot is None:
                raise ValueError("No saved captures yet. Run the watcher or pass a snapshot path.")
        event, observed = read_snapshot(snapshot, args.probe)
        groups = tuple(dict.fromkeys(args.market)) if args.market else (("shots",) if args.player else GROUPS)
        if not args.models.is_file():
            raise ValueError("Model file missing. First run football_odds_predictor.training.fit_sporting.")
        with args.models.open("rb") as handle:
            models = pickle.load(handle)
        print_report(event, observed, stress / 100 if stress is not None else None, models, groups, args.player)
    except (InvalidOperation, KeyError, OSError, ValueError, pickle.UnpicklingError, EOFError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
