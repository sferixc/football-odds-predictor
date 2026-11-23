"""Join historical referee assignments from the Transfermarkt games archive."""

from pathlib import Path

import pandas as pd

ARCHIVE = Path("data/raw/transfermarkt/games.csv.gz")
APPOINTMENTS = Path("data/raw/referee-appointments.csv")

# Names here use Football-Data's team spelling on the right.
CLUBS = {
    "D1": {
        "1. Fußballclub Heidenheim 1846": "Heidenheim", "1.FC Köln": "FC Koln",
        "1.FC Union Berlin": "Union Berlin", "1.FSV Mainz 05": "Mainz",
        "Arminia Bielefeld": "Bielefeld", "Bayer 04 Leverkusen": "Leverkusen",
        "Borussia Dortmund": "Dortmund", "Borussia Mönchengladbach": "M'gladbach",
        "Eintracht Frankfurt": "Ein Frankfurt", "FC Augsburg": "Augsburg",
        "FC Schalke 04": "Schalke 04", "FC St. Pauli": "St Pauli",
        "Hamburger SV": "Hamburg", "Hertha BSC": "Hertha", "SC Freiburg": "Freiburg",
        "SV Darmstadt 98": "Darmstadt", "SV Werder Bremen": "Werder Bremen",
        "SpVgg Greuther Fürth": "Greuther Furth", "TSG 1899 Hoffenheim": "Hoffenheim",
        "VfB Stuttgart": "Stuttgart", "VfL Bochum": "Bochum", "VfL Wolfsburg": "Wolfsburg",
    },
    "SP1": {
        "Athletic Bilbao": "Ath Bilbao", "Atlético de Madrid": "Ath Madrid",
        "CA Osasuna": "Osasuna", "CD Leganés": "Leganes", "Celta de Vigo": "Celta",
        "Cádiz CF": "Cadiz", "Deportivo Alavés": "Alaves", "Elche CF": "Elche",
        "FC Barcelona": "Barcelona", "Getafe CF": "Getafe", "Girona FC": "Girona",
        "Granada CF": "Granada", "Levante UD": "Levante", "RCD Espanyol Barcelona": "Espanol",
        "RCD Mallorca": "Mallorca", "Rayo Vallecano": "Vallecano",
        "Real Betis Balompié": "Betis", "Real Oviedo": "Oviedo", "Real Sociedad": "Sociedad",
        "Real Valladolid CF": "Valladolid", "Sevilla FC": "Sevilla", "UD Almería": "Almeria",
        "UD Las Palmas": "Las Palmas", "Valencia CF": "Valencia", "Villarreal CF": "Villarreal",
    },
}


def add_archive_referees(rows: pd.DataFrame, league: str) -> pd.DataFrame:
    if league not in CLUBS or not ARCHIVE.is_file():
        return rows
    mapping = {
        "date": "match_date", "home_club_name": "home_team", "away_club_name": "away_team",
        "home_club_goals": "home_goals", "away_club_goals": "away_goals",
        "referee": "archive_referee",
    }
    games = pd.read_csv(ARCHIVE, usecols=["competition_id", *mapping], parse_dates=["date"])
    competition = {"D1": "L1", "SP1": "ES1"}[league]
    games = games.loc[games["competition_id"].eq(competition)].rename(columns=mapping)
    for side in ("home_team", "away_team"):
        games[side] = games[side].replace(CLUBS[league])
    keys = ["match_date", "home_team", "away_team", "home_goals", "away_goals"]
    merged = rows.merge(games[[*keys, "archive_referee"]], on=keys, how="left", validate="one_to_one")
    merged["referee"] = merged["referee"].fillna(merged.pop("archive_referee"))
    return merged


def pre_match_referee(event: dict, cutoff) -> str | None:
    """Read a named referee or an appointment observed before the offer snapshot."""
    fixture = event["fixture"]
    referee = fixture.get("referee")
    if isinstance(referee, str) and referee.strip():
        return referee
    automatic = event.get("referee_appointment") or {}
    name = automatic.get("name")
    observed = pd.to_datetime(automatic.get("observed_at_utc"), utc=True, errors="coerce")
    if not isinstance(name, str) or not name.strip() or pd.isna(observed) or observed > cutoff:
        name = None
    if not APPOINTMENTS.is_file():
        return name
    rows = pd.read_csv(APPOINTMENTS, dtype={"event_id": "string"})
    timestamps = pd.to_datetime(rows["observed_at_utc"], utc=True, errors="coerce")
    matches = rows.loc[rows["event_id"].eq(str(event["event_id"])) & timestamps.le(cutoff)]
    if matches.empty:
        return name
    index = timestamps.loc[matches.index].idxmax()
    if name is not None and observed > timestamps.loc[index]:
        return name
    return matches.loc[index, "referee"]
