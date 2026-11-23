"""Find published pre-match referee appointments for watched fixtures."""

from collections import defaultdict
from datetime import UTC, datetime
from functools import lru_cache
from html import unescape
import json
import re
from urllib.request import urlopen

from ..models.ucl import club_key
from .europe import ALIASES
from .referees import CLUBS

OBSERVER = "https://www.referee.observer/fixtures"
UEFA_LIST = (
    "https://www.uefa.com/uefachampionsleague/news/"
    "02a9-218880e7a2aa-96adf63aa534-1000--who-is-the-referee-which-officials-are-in-charge-of-the-uefa/"
)
UEFA_MATCH = "https://match.uefa.com/v5/matches/"
ALIASES_BY_KEY = {club_key(a): club_key(b) for a, b in ALIASES.items()}
ALIASES_BY_KEY.update(
    (club_key(alias), club_key(original))
    for mapping in CLUBS.values() for original, alias in mapping.items()
)
LEAGUE_NAMES = {"LaLiga": "La Liga", "LaLiga EA Sports": "La Liga", "La Liga EA Sports": "La Liga"}


def text_content(markup):
    return " ".join(unescape(re.sub(r"<[^>]*>", "", markup)).split())


def teams_key(name, separator):
    teams = name.split(separator)
    if len(teams) != 2:
        return None
    keys = [club_key(team.strip()) for team in teams]
    return tuple(ALIASES_BY_KEY.get(key, key) for key in keys)


@lru_cache(maxsize=24)
def fetch(url, period):
    """Share a fresh source response among fixtures checked together."""
    try:
        with urlopen(url, timeout=4) as response:
            return response.read().decode("utf-8"), datetime.now(UTC)
    except (OSError, UnicodeError):
        return None


def observer_appointments(markup):
    appointments = defaultdict(set)
    for section in re.findall(r"<section\b[^>]*>(.*?)</section>", markup, re.S):
        heading = re.search(r"<h2\b[^>]*>(.*?)</h2>", section, re.S)
        if not heading:
            continue
        try:
            day = datetime.strptime(text_content(heading[1]), "%A, %d %B %Y").date()
        except ValueError:
            continue
        cards = re.split(r'<a\b[^>]*href="/match/\d+"[^>]*>', section)[1:]
        for card in cards:
            name, _, details = card.partition("</a>")
            teams = teams_key(text_content(name), " v ")
            league = re.search(r"<span\b[^>]*>([^<>]+?)\s*·\s*\d{4}/\d{2,4}</span>", details)
            referee = re.search(r'<a\b[^>]*href="/referee/[^"]+"[^>]*>(.*?)</a>', details, re.S)
            if teams and league and referee:
                competition = text_content(league[1]).removeprefix("UEFA ")
                competition = LEAGUE_NAMES.get(competition, competition)
                full_name = text_content(referee[1])
                if full_name:
                    appointments[day, competition, teams].add(full_name)
    return appointments


def uefa_ids(markup, teams):
    links = re.findall(
        r'<a\b[^>]*href="(?:https://www\.uefa\.com)?/uefachampionsleague/match/(\d+)--[^"]*"[^>]*>(.*?)</a>',
        markup, re.S,
    )
    return {identifier for identifier, label in links
            if teams_key(text_content(label).partition(" – ")[0], " vs ") == teams}


def uefa_referee(payload, kickoff):
    value = payload.get("kickOffTime", {}).get("dateTime")
    if not value or datetime.fromisoformat(value) != kickoff or payload.get("status") != "UPCOMING":
        return None
    names = {entry["person"]["translations"]["name"]["EN"] for entry in payload.get("referees", [])
             if entry.get("role") == "REFEREE"}
    return next(iter(names)) if len(names) == 1 else None


def find_appointment(fixture, now, fresh=False):
    teams = teams_key(fixture.name, "·")
    if teams is None or now >= fixture.kickoff:
        return None
    period = int(now.timestamp()) // (60 if fresh else 3600)
    sources = [OBSERVER]
    if fixture.league == "Champions League":
        sources.insert(0, UEFA_LIST)
    for source in sources:
        try:
            response = fetch(source, (fresh, period))
            if response is None:
                continue
            markup, observed = response
            if source == UEFA_LIST:
                ids = uefa_ids(markup, teams)
                if len(ids) != 1:
                    continue
                source = UEFA_MATCH + next(iter(ids))
                response = fetch(source, (fresh, period))
                if response is None:
                    continue
                body, observed = response
                name = uefa_referee(json.loads(body), fixture.kickoff)
            else:
                names = observer_appointments(markup).get((fixture.kickoff.date(), fixture.league, teams), ())
                name = next(iter(names)) if len(names) == 1 else None
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if name and observed < fixture.kickoff:
            return {"name": name, "source": source, "observed_at_utc": observed.isoformat()}
    return None
