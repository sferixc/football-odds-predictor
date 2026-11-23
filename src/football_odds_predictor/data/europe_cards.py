"""Reconstruct European card totals from lineups, substitutions and cards."""

from collections import defaultdict

import pandas as pd

from .referees import ARCHIVE

CARD_EVENTS = ARCHIVE.with_name("game_events.csv.gz")
CARD_LINEUPS = ARCHIVE.with_name("game_lineups.csv.gz")
SCOPES = ("home", "away", "match")


def card_weight(description: str) -> int | None:
    """Second-yellow dismissals contribute the red, with no extra yellow."""
    description = description.casefold().strip()
    if description.startswith(("red card", "second yellow")):
        return 2
    if "yellow card" in description.partition(",")[0]:
        return 1
    return None


def count_cards(events, lineups, clubs):
    starters = defaultdict(set)
    reserves = defaultdict(set)
    for row in lineups.itertuples(index=False):
        if row.type == "starting_lineup":
            starters[row.club_id].add(row.player_id)
        elif row.type == "substitutes":
            reserves[row.club_id].add(row.player_id)
    if any(len(starters[club]) != 11 for club in clubs):
        return None

    intervals = {club: {player: [0, 91] for player in starters[club]} for club in clubs}
    substitutions = events.loc[events.type.eq("Substitutions")].sort_values("minute")
    if substitutions.empty:
        return None
    for row in substitutions.itertuples(index=False):
        active = intervals.get(row.club_id)
        if (active is None or row.player_id not in active or row.player_in_id in active
                or pd.isna(row.player_in_id) or pd.isna(row.minute)):
            return None
        active[row.player_id][1] = row.minute
        active[row.player_in_id] = [row.minute, 91]

    totals = dict.fromkeys(clubs, 0)
    observed = set()
    cards = events.loc[events.type.eq("Cards")].sort_values("minute")
    for row in cards.itertuples(index=False):
        if row.club_id not in totals or pd.isna(row.minute):
            return None
        description = str(row.description)
        if "after final whistle" in description.casefold() or row.minute > 90:
            continue
        span = intervals[row.club_id].get(row.player_id)
        if span is None:
            if row.player_id in reserves[row.club_id] or pd.isna(row.player_id):
                continue
            # Some source IDs identify staff; omit the match if we cannot establish eligibility.
            return None
        if row.minute in span:
            return None
        if not span[0] < row.minute < span[1]:
            continue
        weight = card_weight(description)
        if weight is None:
            return None
        key = (row.club_id, row.player_id, weight)
        if key in observed:
            return None
        observed.add(key)
        totals[row.club_id] += weight
    return tuple(totals[club] for club in clubs)


def card_targets(games: pd.DataFrame) -> pd.DataFrame:
    historical = games.loc[
        games.competition_id.isin(("CL", "EL")) & games.season.between(2021, 2025)
        & games["round"].str.startswith("Group", na=False) & games.game_id.notna()
    ]
    ids = set(historical.game_id)
    events = pd.read_csv(CARD_EVENTS, usecols=[
        "game_id", "type", "club_id", "player_id", "player_in_id", "minute", "description",
    ])
    lineups = pd.read_csv(CARD_LINEUPS, usecols=["game_id", "club_id", "player_id", "type"])
    event_groups = {game: rows for game, rows in events.loc[events.game_id.isin(ids)].groupby("game_id")}
    lineup_groups = {game: rows for game, rows in lineups.loc[lineups.game_id.isin(ids)].groupby("game_id")}
    targets = []
    for row in historical.itertuples(index=False):
        if row.game_id not in event_groups or row.game_id not in lineup_groups:
            continue
        totals = count_cards(
            event_groups[row.game_id], lineup_groups[row.game_id], (row.home_club_id, row.away_club_id),
        )
        if totals is not None:
            targets.append((row.game_id, *totals, sum(totals)))
    return pd.DataFrame(targets, columns=["game_id", *SCOPES])
