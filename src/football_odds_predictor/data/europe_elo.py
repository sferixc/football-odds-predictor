"""Read dated public ratings for clubs entering the shared European timeline."""

from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd

ELO_ARCHIVE = Path("data/raw/sporteden-2026-11-26")
ELO_YEARS = range(2019, 2027)
COUNTRY_CODES = {
    "AUT": "AT", "BEL": "BE", "CRO": "HR", "CYP": "CY", "CZE": "CZ", "DEN": "DK",
    "ENG": "GB-ENG", "ESP": "ES", "FIN": "FI", "FRA": "FR", "GER": "DE", "GRE": "GR",
    "ITA": "IT", "KAZ": "KZ", "NED": "NL", "NOR": "NO", "POL": "PL", "POR": "PT",
    "ROU": "RO", "RUS": "RU", "SCO": "GB-SCT", "SUI": "CH", "SVK": "SK", "SVN": "SI",
    "SWE": "SE", "TUR": "TR", "UKR": "UA", "WAL": "GB-WLS",
}
EUROPE_CODES = set(COUNTRY_CODES.values()) | {
    "AD", "AL", "AM", "AZ", "BA", "BG", "BY", "EE", "FO", "GB-NIR", "GE", "GI", "HU",
    "IE", "IL", "IS", "LI", "LT", "LU", "LV", "MD", "ME", "MK", "MT", "RS", "SM", "XK",
}
MAX_CLUB_AGE = pd.Timedelta(days=180)
# Match provider names to existing Transfermarkt or Open Matchday identities.
CLUB_ALIASES = {
    "OFI": "OFI Crete", "Man City": "Manchester City", "Man United": "Manchester United",
    "Brighton": "Brighton & Hove Albion", "Wolves": "Wolverhampton Wanderers",
    "Forest": "Nottingham Forest", "Spurs": "Tottenham Hotspur",
    "Atletico": "Atlético de Madrid", "Athletic Club": "Athletic Bilbao",
    "Paris SG": "Paris Saint-Germain", "Paris SG FC": "Paris Saint-Germain",
    "Marseille": "Olympique Marseille", "Lyon": "Olympique Lyon", "Lille": "LOSC Lille",
    "Bayern": "Bayern Munich", "Dortmund": "Borussia Dortmund", "Leipzig": "RB Leipzig",
    "Frankfurt": "Eintracht Frankfurt", "Leverkusen": "Bayer 04 Leverkusen",
    "Gladbach": "Borussia Mönchengladbach",
    "Internazionale": "Inter Milan", "Milan": "AC Milan", "Napoli": "SSC Napoli",
    "Roma": "AS Roma", "Lazio": "SS Lazio", "Juventus": "Juventus FC",
    "Benfica": "SL Benfica", "Sporting": "Sporting CP", "Porto": "FC Porto",
    "Brugge": "Club Brugge", "Genk": "KRC Genk", "Gent": "KAA Gent",
    "Union SG": "Union Saint-Gilloise", "Bodoe Glimt": "Bodo Glimt",
    "AEK": "AEK Athens", "Aris": "Aris Thessaloniki",
    "Lech": "Lech Poznan", "Legia": "Legia Warszawa", "Lechia": "Lechia Gdansk",
}
# The source also rates Bournemouth FC, a different club from AFC Bournemouth.
VERIFIED_TEAM_IDS = {"AFC Bournemouth": 10087}


class EuropeanEloSeeds:
    """Use the latest published snapshot strictly earlier than a club's debut day."""

    def __init__(self, path, names, club_key, league_countries):
        rows = pd.concat([
            pd.read_csv(path / f"{year}.csv", comment="#", encoding="utf-8-sig",
                        usecols=["date", "team_id", "team", "country_code", "elo"], parse_dates=["date"])
            for year in ELO_YEARS
        ], ignore_index=True)
        rows = rows.loc[rows.country_code.isin(EUROPE_CODES)].copy()
        if rows.empty or rows.isna().any().any() or not np.isfinite(rows.elo).all():
            raise ValueError(f"Missing or invalid European Elo values in {path}")
        if rows.duplicated(["date", "team_id"]).any():
            raise ValueError(f"Duplicate Elo team and date in {path}")
        self.first_snapshot = rows.date.min()
        aliases = {club_key(source): club_key(target) for source, target in CLUB_ALIASES.items()}

        def resolve(name):
            key = club_key(name)
            return names.get(aliases.get(key, key), names.get(key))

        rows["club_id"] = rows.team.map(resolve).astype(object)
        # Reject ambiguous provider identities instead of silently merging different clubs.
        matched = rows.dropna(subset=["club_id"]).copy()
        for name, team_id in VERIFIED_TEAM_IDS.items():
            club = names.get(club_key(name))
            matched = matched.loc[matched.club_id.ne(club) | matched.team_id.eq(team_id)]
        identities = matched.groupby("club_id").team_id.transform("nunique")
        matched = matched.loc[identities.eq(1)]
        self.clubs = {
            club: (group.date.tolist(), group.elo.tolist())
            for club, group in matched.sort_values("date").groupby("club_id", sort=False)
        }
        medians = rows.groupby(["date", "country_code"]).elo.median()
        self.countries = {
            country: (group.index.get_level_values("date").tolist(), group.tolist())
            for country, group in medians.groupby(level="country_code")
        }
        global_medians = medians.groupby(level="date").median()
        self.global_prior = global_medians.index.tolist(), global_medians.tolist()
        self.league_countries = league_countries

    def initial(self, club, day, league=None):
        if club in self.clubs:
            dates, elos = self.clubs[club]
            index = bisect_left(dates, day) - 1
            if index >= 0 and day - dates[index] <= MAX_CLUB_AGE:
                return float(elos[index]), "club", dates[index]
        country = self.league_countries.get(league)
        if country in self.countries:
            dates, elos = self.countries[country]
            index = bisect_left(dates, day) - 1
            if index >= 0:
                return float(elos[index]), "country median", dates[index]
        dates, elos = self.global_prior
        index = bisect_left(dates, day) - 1
        if index >= 0:
            return float(elos[index]), "European median", dates[index]
        raise ValueError(f"No Elo snapshot before {day.date()}")
