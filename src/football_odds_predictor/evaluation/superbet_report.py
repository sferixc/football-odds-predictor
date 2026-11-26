"""Report supported football selections alongside one captured Superbet offer."""

import argparse
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import pickle
import re
import sys
import unicodedata
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..data.features import build_features
from ..data.preprocessing import load_matches
from ..data.referees import pre_match_referee
from ..data.sporting import (
    read_player_appearances, read_team_matches, team_features,
)
from ..data.superbet_snapshot import EUROPEAN_TOURNAMENTS, check_window, open_price, simple_promotions, utc_timestamp
from ..models.cards import CARD_LINES, load_card_models, card_probabilities
from ..models.corners import CORNER_LINES, corner_probabilities, european_corner_inputs, load_corner_models
from ..models.sporting import predict_saved as predict_saved
from ..models.ucl import MIN_REFEREE_MATCHES, goal_probabilities, load_ucl_models, predict_ucl
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
CARD_OFFER_MARKETS = {
    690: "match",
    700: "home",
    708: "away",
}
BADGE_STYLE = {
    "red": "97;41", "yellow": "30;103", "orange": "30;48;5;208",
    "green": "30;102", "purple": "97;45",
}


def color_badge(signal: str, highlighted: bool) -> str:
    """Pad the label before applying ANSI codes so the table stays aligned."""
    if not highlighted or signal not in BADGE_STYLE:
        return f"{signal:>8}"
    return f"\033[{BADGE_STYLE[signal]}m{signal.upper().center(8)}\033[0m"


def read_snapshot(path: Path, probe: bool) -> tuple[dict, str]:
    """Accept timed captures by default and require an opt-in for untimed probes."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "event" in data:
        event = data["event"]
        observed = utc_timestamp(data["observed_at_utc"])
        check_window(utc_timestamp(event["fixture"]["utc_date"]), observed,
                     early=data.get("early") is True, archived=True)
        if "referee_appointment" in data:
            event = {**event, "referee_appointment": data["referee_appointment"]}
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


def card_offer_sections(event: dict) -> list[dict]:
    """Display bookmaker card totals using the provider's separate settlement market."""
    sections = []
    for market in event["markets"]:
        scope = CARD_OFFER_MARKETS.get(market["id"])
        if scope is None:
            continue
        rows = []
        for odd in market["odds"]:
            metadata = odd["metadata"]
            specifiers = metadata.get("specifiers") or {}
            line = specifiers.get("total")
            if line is None or not odd.get("display"):
                continue
            side = line_selection(metadata["name"], specifiers, line)
            if side:
                direction = "Over" if side == "Peste" else "Under"
                rows.append({
                    "selection": f"{direction} {line}",
                    "standard": odd["price"] if open_price(odd) else None,
                })
        if rows:
            sections.append({
                "group": "cards", "name": f"Superbet total cards · {scope}", "rows": rows,
                "quote_only": True, "card_scope": scope,
                "note": "Superbet counts yellow and red cards using its own settlement rules.",
            })
    return sections


def european_offer_sections(event: dict, groups: tuple[str, ...], player_filter: str | None) -> list[dict]:
    """Expose UEFA prices without applying domestic probability models."""
    definitions = {547: ("1x2", "Full-time result"), 236218: ("shots", "Player total shots")}
    for group in ("goals", "corners"):
        definitions.update({market_id: (group, label) for market_id, _, label in TEAM_MARKETS[group]})
    sections = card_offer_sections(event) if "cards" in groups else []
    for market in event["markets"]:
        if market["id"] not in definitions:
            continue
        group, name = definitions[market["id"]]
        if group not in groups:
            continue
        rows = []
        for odd in market["odds"]:
            if not open_price(odd):
                continue
            label = odd["metadata"]["name"]
            if group == "shots" and player_filter and player_filter.casefold() not in label.casefold():
                continue
            label = re.sub(r"\b(?:Peste|Sub) (?=\d)",
                           lambda match: "Over " if match[0] == "Peste " else "Under ", label)
            rows.append({"selection": label, "standard": odd["price"]})
        if rows:
            sections.append({
                "group": group, "name": name, "rows": rows, "quote_only": True, "market_id": market["id"],
            })
    return sections


def _player_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name.casefold())
    return "".join(letter for letter in decomposed if letter.isalnum())


def player_rows(
    event: dict, player_filter: str | None, promotions: list[dict], models: dict,
) -> tuple[list[dict], str]:
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

    if not rows:
        return [], "No offered players with an unambiguous name and enough prior minutes."
    features = pd.DataFrame.from_dict(rows, orient="index")
    results = []
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
                results.append({
                    "selection": f"{name} · {'Over' if side == 'Peste' else 'Under'} {threshold:g}",
                    "fair": float(1 / probability) if probability > 0 else None,
                    "standard": price,
                    "promotion": boosted.get((specifiers.get("player_id"), side)),
                    "last_result": str(rows[name]["last_match"]),
                })
    note = "Model odds assume the player takes part. Value, loss risk, and color are pending settlement checks."
    if skipped:
        note += f" Skipped {len(skipped)} players with ambiguous identity or insufficient prior minutes."
    return results, note


def selection_rows(selections, probabilities, prices, promotions, stress) -> list[dict]:
    """Return display-ready values shared by the terminal and dashboard."""
    rows = []
    for selection, probability in zip(selections, probabilities):
        p = Decimal(str(probability))
        standard, promoted = prices.get(selection), promotions.get(selection)
        price = promoted if promoted is not None else standard
        row = {
            "selection": {"Sub": "Under", "Peste": "Over"}.get(selection, selection),
            "loss": float(1 - p),
            "fair": float(1 / p) if p > 0 else None, "standard": standard,
            "promotion": promoted, "return": None, "prudent": None, "signal": None,
        }
        if price is not None and p > 0:
            score = assess_offer(
                odds=Decimal(str(price)), win_probability=p,
                conservative_win_probability=p if stress is None else max(Decimal(0), p - stress),
                stake=Decimal(1), budget=Decimal(1),
            )
            row["return"] = float(score.expected_return)
            if stress is not None:
                row["prudent"] = float(score.conservative_return)
                row["signal"] = score.signal
        rows.append(row)
    return rows


def corner_sections(event, bundle, league, current, cutoff, promotions, stress):
    """Use the same line, price and risk display for domestic and UEFA corners."""
    lines = {target: set(defaults) for target, defaults in CORNER_LINES.items()}
    targets = {market: target for market, target, _ in TEAM_MARKETS["corners"]}
    for market in event["markets"]:
        target = targets.get(market["id"])
        if target is None:
            continue
        for odd in market["odds"]:
            metadata = odd["metadata"]
            line = (metadata.get("specifiers") or {}).get("total")
            if re.fullmatch(r"\d+\.5(?:0*)", str(line)):
                lines[target].add(float(line))
    for offer in promotions:
        line = (offer.get("specifiers") or {}).get("total")
        if offer["market"] in lines and re.fullmatch(r"\d+\.5(?:0*)", str(line)):
            lines[offer["market"]].add(float(line))
    lines = {target: sorted(values) for target, values in lines.items()}
    forecasts = corner_probabilities(bundle, league, current, cutoff, lines)
    if forecasts is None:
        return []
    sections = []
    for market, target, label in TEAM_MARKETS["corners"]:
        rows = []
        for line, over in zip(lines[target], forecasts[target]):
            standard, boosted = team_prices(event, market, target, line, promotions)
            scored = selection_rows(("Sub", "Peste"), (1 - over, over), standard, boosted, stress)
            for row in scored:
                row["selection"] += f" {line:g}"
            rows.extend(scored)
        section = {"group": "corners", "name": label, "rows": rows, "quote_only": False}
        if league in ("CL", "EL"):
            home, away = (int(current[f"{side}_observed_corners"].iat[0]) for side in ("home", "away"))
            section["note"] = f"Corner data in the last five matches: home {home}/5, away {away}/5."
        sections.append(section)
    return sections


def referee_note(referee, used, count, minimum=1):
    if not isinstance(referee, str) or not referee.strip():
        return "Team model only · referee assignment unavailable"
    referee = referee.strip()
    matches = f"{count} prior {'match' if count == 1 else 'matches'}"
    if used:
        return f"Referee used · {referee} · {matches}"
    if count < minimum:
        return f"Team model only · {referee} · {matches}, need at least {minimum}"
    return f"Team model only · {referee} · referee model unavailable for this market"


def build_report(
    event: dict, observed: str, stress: Decimal | None, models: dict,
    groups: tuple[str, ...] = GROUPS, player_filter: str | None = None,
) -> dict:
    """Build market sections from one snapshot and the saved models."""
    fixture = event["fixture"]
    report = {
        "fixture": fixture["event_name"].replace("·", " – "),
        "kickoff": fixture["utc_date"], "observed": observed,
        "stress_pp": float(stress * 100) if stress is not None else None,
        "last_results": [], "sections": [],
    }
    tournament = int(fixture["tournament_id"])
    if tournament in EUROPEAN_TOURNAMENTS:
        competition = EUROPEAN_TOURNAMENTS[tournament]
        kickoff = utc_timestamp(fixture["utc_date"])
        observed_at = None if observed == "untimed probe" else utc_timestamp(observed)
        cutoff = min(kickoff, observed_at) if observed_at is not None else kickoff
        report["competition"] = "Champions League" if competition == "CL" else "Europa League"
        report["sections"] = european_offer_sections(event, groups, player_filter)
        if "corners" in groups:
            bundle = models.get("europe_corner_counts")
            current = european_corner_inputs(bundle, fixture, cutoff)
            corners = corner_sections(event, bundle, competition, current, cutoff,
                                      simple_promotions(event), stress)
            if corners:
                report["sections"] = [section for section in report["sections"] if section["group"] != "corners"]
                report["sections"].extend(corners)
        referee = pre_match_referee(event, cutoff) if observed_at is not None else fixture.get("referee")
        forecast = predict_ucl(
            models.get("ucl"), fixture, kickoff, competition, referee, observed_at=observed_at,
        )
        if forecast is not None:
            result, expected_goals, last_dates, has_domestic_history, card_forecasts = forecast
            promotions = simple_promotions(event)
            for section in report["sections"]:
                market = section.get("market_id")
                if market not in (547, 200734):
                    continue
                prices = {row["selection"]: row["standard"] for row in section["rows"]}
                if market == 547:
                    boosted = {}
                    for offer in promotions:
                        if offer["market"] == "1x2":
                            selection = offer["selection"]
                            boosted[selection] = max(boosted.get(selection, 0), offer["odds"])
                    rows = selection_rows(("1", "X", "2"), result, prices, boosted, stress)
                else:
                    rows = []
                    lines = {float(match[1]) for selection in prices
                             if (match := re.fullmatch(r"(?:Over|Under) (\d+\.5)", selection))}
                    for line in sorted(lines):
                        standard, boosted = team_prices(event, market, "match_goals", line, promotions)
                        rows.extend(selection_rows(
                            ("Sub", "Peste"), goal_probabilities(line, expected_goals), standard, boosted, stress,
                        ))
                        for row in rows[-2:]:
                            row["selection"] += f" {line:g}"
                    if not rows:
                        continue
                section.update(rows=rows, quote_only=False)
            if "cards" in groups:
                for market, scope in CARD_OFFER_MARKETS.items():
                    if scope not in card_forecasts:
                        continue
                    expected, uses_referee, prior_matches = card_forecasts[scope]
                    rows = []
                    quotes = next((section["rows"] for section in report["sections"]
                                   if section.get("card_scope") == scope), ())
                    lines = set(CARD_LINES)
                    lines.update(float(match[1]) for quote in quotes
                                 if (match := re.fullmatch(r"(?:Over|Under) (\d+\.5)", quote["selection"])))
                    for line in sorted(lines):
                        standard, boosted = team_prices(event, market, f"{scope}_card_points", line, promotions)
                        selections = ("Sub", "Peste")
                        scored = selection_rows(
                            selections, goal_probabilities(line, expected), standard, boosted, stress,
                        )
                        for row in scored:
                            row["selection"] += f" {line:g}"
                        rows.extend(scored)
                    report["sections"].append({
                        "group": "cards", "card_scope": scope, "name": f"{scope.title()} total cards",
                        "rows": rows, "quote_only": False,
                        "note": "Card totals reconstructed from events and lineups.",
                        "referee_note": referee_note(referee, uses_referee, prior_matches, MIN_REFEREE_MATCHES),
                    })
                report["european_card_model"] = bool(card_forecasts)
            report["last_results"] = last_dates
            report["domestic_history_complete"] = has_domestic_history
        report["offer_only"] = all(section["quote_only"] for section in report["sections"])
        return report
    if tournament not in LEAGUES:
        raise ValueError("This report covers Premier League, Bundesliga, La Liga, Champions League, and Europa League")
    sections = report["sections"]
    promotions = simple_promotions(event)
    zone = LEAGUES[int(fixture["tournament_id"])][1]
    cutoff = utc_timestamp(fixture["utc_date"]).astimezone(ZoneInfo(zone)).date()

    if any(group != "shots" for group in groups):
        current, last_matches = team_inputs_for_event(event)
        report["last_results"] = [str(result.date()) for result in last_matches]
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
            sections.append({
                "group": "1x2", "name": "Full-time result",
                "rows": selection_rows(("1", "X", "2"),
                                       predict_saved(models, current, "1x2", cutoff)[0],
                                       prices, boosted, stress),
            })

        league = LEAGUES[int(fixture["tournament_id"])][0]
        card_quotes = card_offer_sections(event) if "cards" in groups else []
        referee = pre_match_referee(event, utc_timestamp(observed)) if (
            "cards" in groups and observed != "untimed probe"
        ) else fixture.get("referee")
        for group in groups:
            if group == "corners":
                corners = corner_sections(event, models.get("corner_counts"), league, current,
                                          cutoff, promotions, stress)
                if corners:
                    sections.extend(corners)
                    continue
            for market_id, target, label in TEAM_MARKETS.get(group, ()):
                if group == "cards":
                    scope = target.split("_", 1)[0]
                    bundle = models.get("yellow_card_counts", {}).get(league)
                    proxy = bundle is not None and f"{scope}_card_points" in bundle["models"]
                    forecast = f"{scope}_card_points" if proxy else target
                    overs, with_referee, prior_matches = card_probabilities(
                        bundle, current, forecast, cutoff, referee,
                    )
                    if overs is not None:
                        prices = {
                            row["selection"]: row["standard"] for section in card_quotes
                            if section["card_scope"] == scope for row in section["rows"]
                        } if proxy else {}
                        rows = []
                        for line, over in zip(CARD_LINES, overs):
                            rows.extend(selection_rows(
                                (f"Under {line:g}", f"Over {line:g}"),
                                (1 - over, over), prices, {}, stress,
                            ))
                        sections.append({
                            "group": group, "name": f"{scope.title()} total cards (proxy)" if proxy else label,
                            "rows": rows,
                            "card_scope": scope, "proxy": proxy,
                            "note": (
                                "Estimated total uses aggregate yellows + 2 × reds. "
                                "Second yellows and bench cards may differ from settlement."
                                if proxy else "Model estimates for yellow cards only."
                            ),
                            "referee_note": referee_note(referee, with_referee, prior_matches),
                        })
                        continue
                threshold = TEAM_THRESHOLDS[target]
                standard, boosted = team_prices(event, market_id, target, threshold, promotions)
                probabilities = predict_saved(models, current, target, cutoff, threshold)[0]
                selections = ((f"Under {threshold:g}", f"Over {threshold:g}")
                              if group == "cards" else ("Sub", "Peste"))
                sections.append({
                    "group": group, "name": f"{label} ({threshold:g})",
                    "rows": selection_rows(selections, probabilities, standard, boosted, stress),
                    **({"card_scope": target.split("_", 1)[0],
                        "referee_note": "Team model only · referee model unavailable for this market"}
                       if group == "cards" else {}),
                    "note": "Model estimates for yellow cards only." if group == "cards" else "",
                })
        if "cards" in groups:
            sections.extend(card_quotes)
    if "shots" in groups:
        rows, note = player_rows(event, player_filter, promotions, models)
        sections.append({"group": "shots", "name": "Player total shots", "rows": rows, "note": note})
    return report


def print_report(
    event: dict, observed: str, stress: Decimal | None, models: dict,
    groups: tuple[str, ...] = GROUPS, player_filter: str | None = None, highlighted: bool = False,
) -> None:
    report = build_report(event, observed, stress, models, groups, player_filter)
    print(f"{report['fixture']} | kickoff {report['kickoff']}")
    print(f"Offer: {observed}")
    if report.get("offer_only"):
        print(f"{report['competition']} · Superbet prices only")
    else:
        if observed == "untimed probe":
            print("Replay of an untimed offer. Probabilities use the local results archive.")
        if stress is not None:
            print(f"Provisional color uses a {stress * 100:g} percentage-point reduction.")
        print("Return uses Promo when available and Superbet otherwise.")
    if report["last_results"]:
        print(f"Latest team results used (home / away): {' / '.join(report['last_results'])}")

    def fmt(value, style):
        return format(value, style) if value is not None else "—"

    for section in report["sections"]:
        print(f"\n[{section['name']}]")
        if section.get("referee_note"):
            print(section["referee_note"])
        if section.get("quote_only"):
            print(f"{'Selection':<14} {'Superbet':>9}")
            for row in section["rows"]:
                price = "Closed" if row["standard"] is None else fmt(row["standard"], ".2f")
                print(f"{row['selection']:<14} {price:>9}")
        elif section["group"] == "shots":
            print(f"{'Player and selection':<43} {'Model odds':>10} {'Superbet':>9} {'Promo':>8} {'Last result':>12}")
            for row in section["rows"]:
                print(f"{row['selection']:<43} {fmt(row['fair'], '.2f'):>10} {fmt(row['standard'], '.2f'):>9} "
                      f"{fmt(row['promotion'], '.2f'):>8} {row['last_result']:>12}")
        else:
            print(f"{'Selection':<10} {'Model odds':>10} {'Superbet':>9} {'Promo':>8} {'Loss risk':>9} "
                  f"{'Est. return':>12} {'Prudent':>9} {'Signal':>8}")
            for row in section["rows"]:
                print(f"{row['selection']:<10} {fmt(row['fair'], '.2f'):>10} {fmt(row['standard'], '.2f'):>9} "
                      f"{fmt(row['promotion'], '.2f'):>8} {row['loss']:>9.1%} {fmt(row['return'], '+.1%'):>12} "
                      f"{fmt(row['prudent'], '+.1%'):>9} {color_badge(row['signal'] or '—', highlighted)}")
        if section.get("note"):
            print(section["note"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, nargs="?", help="Defaults to the most recently saved capture")
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS, help="Locally trained model file")
    parser.add_argument("--probe", action="store_true", help="Allow a single untimed test response")
    parser.add_argument("--stress-pp", type=Decimal, help="Optional probability reduction in percentage points")
    parser.add_argument("--market", action="append", choices=GROUPS, help="Show only this market group")
    parser.add_argument("--player", help="Filter player shot selections by partial name")
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto",
                        help="Highlight signal labels with terminal background colors")
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
        if not args.models.is_file() and int(event["fixture"]["tournament_id"]) not in EUROPEAN_TOURNAMENTS:
            raise ValueError("Model file missing. First run football_odds_predictor.training.fit_sporting.")
        models = {}
        if int(event["fixture"]["tournament_id"]) not in EUROPEAN_TOURNAMENTS:
            with args.models.open("rb") as handle:
                models = pickle.load(handle)
            load_card_models(models)
        else:
            load_ucl_models(models)
        load_corner_models(models)
        highlighted = args.color == "always" or (
            args.color == "auto" and sys.stdout.isatty() and "NO_COLOR" not in os.environ
            and os.environ.get("TERM") != "dumb"
        )
        print_report(
            event, observed, stress / 100 if stress is not None else None, models,
            groups, args.player, highlighted,
        )
    except (InvalidOperation, KeyError, OSError, ValueError, pickle.UnpicklingError, EOFError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
