"""Discover the three project leagues and capture each upcoming fixture once."""

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlencode
from urllib.request import urlopen

from .superbet_snapshot import ENDPOINT, capture, utc_timestamp

LEAGUES = {106: "Premier League", 245: "Bundesliga", 98: "La Liga"}
REFRESH_INTERVAL = timedelta(hours=1)
TARGET_LEAD = timedelta(minutes=15)
MINIMUM_LEAD = timedelta(minutes=10)


@dataclass(frozen=True)
class Fixture:
    event_id: int
    name: str
    league: str
    kickoff: datetime

    @property
    def capture_time(self) -> datetime:
        return self.kickoff - TARGET_LEAD


def scheduled_fixtures(payload: dict, now: datetime) -> list[Fixture]:
    """Keep upcoming football fixtures in the leagues covered by the model."""
    fixtures = {}
    for event in payload["events"]:
        fixture = event["fixture"]
        league = LEAGUES.get(fixture["tournament_id"])
        if league is None or fixture["sport_id"] != 5:
            continue
        kickoff = utc_timestamp(fixture["utc_date"])
        if kickoff <= now:
            continue
        event_id = int(event["event_id"])
        fixtures[event_id] = Fixture(event_id, fixture["event_name"], league, kickoff)
    return sorted(fixtures.values(), key=lambda fixture: (fixture.kickoff, fixture.event_id))


def discover(now: datetime) -> list[Fixture]:
    """Read the next week of fixture dates from the prematch listing."""
    start = now.replace(minute=0, second=0, microsecond=0)
    end = start + timedelta(days=7)
    parameters = {
        "startDate": start.isoformat(timespec="milliseconds"),
        "endDate": end.isoformat(timespec="milliseconds"),
        "index": "active-prematch",
        "sports": "5",
        "tournaments": ",".join(str(league) for league in sorted(LEAGUES)),
    }
    with urlopen(f"{ENDPOINT}?{urlencode(parameters)}", timeout=15) as response:
        return scheduled_fixtures(json.load(response), now)


def next_capture(
    fixtures: list[Fixture],
    now: datetime,
    output_dir: Path,
    attempted: set[tuple[int, datetime]],
) -> Fixture | None:
    """Find the earliest eligible job that has not already been captured or attempted."""
    return next((
        fixture for fixture in fixtures
        if fixture.kickoff - now > MINIMUM_LEAD
        and (fixture.event_id, fixture.kickoff) not in attempted
        and not (output_dir / f"{fixture.event_id}.json").exists()
    ), None)


def watch(output_dir: Path) -> None:
    """Refresh the schedule, then wait for either a capture or the next refresh."""
    fixtures = []
    attempted: set[tuple[int, datetime]] = set()
    refresh_at = datetime.now(UTC)
    while True:
        now = datetime.now(UTC)
        if now >= refresh_at:
            try:
                fixtures = discover(now)
            except (KeyError, OSError, ValueError) as error:
                print(f"Schedule refresh failed: {error}", file=sys.stderr, flush=True)
            else:
                # A kickoff change makes a new scheduled attempt possible for that fixture.
                attempted.intersection_update((f.event_id, f.kickoff) for f in fixtures)
                print(f"{now.isoformat()} | {len(fixtures)} upcoming fixtures", flush=True)
            refresh_at = now + REFRESH_INTERVAL

        now = datetime.now(UTC)
        fixture = next_capture(fixtures, now, output_dir, attempted)
        if fixture is not None and fixture.capture_time <= now:
            # Record the attempt before fetching to avoid repeating a failed request in the loop.
            attempted.add((fixture.event_id, fixture.kickoff))
            try:
                capture(fixture.event_id, fixture.kickoff, output_dir)
            except (KeyError, OSError, ValueError) as error:
                print(f"{fixture.name}: {error}", file=sys.stderr, flush=True)
            continue

        wake_at = min(refresh_at, fixture.capture_time) if fixture is not None else refresh_at
        time.sleep(max(0, min(60, (wake_at - now).total_seconds())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/superbet"))
    parser.add_argument("--list", action="store_true", help="List upcoming fixtures without capturing odds")
    args = parser.parse_args()
    try:
        if args.list:
            for fixture in discover(datetime.now(UTC)):
                print(f"{fixture.capture_time.isoformat()} | {fixture.league} | {fixture.name}")
        else:
            watch(args.output_dir)
    except KeyboardInterrupt:
        print("Stopped Superbet capture.")
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"Schedule check failed: {error}\n")


if __name__ == "__main__":
    main()
