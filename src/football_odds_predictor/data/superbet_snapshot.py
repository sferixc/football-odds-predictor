"""Save one public Superbet Romania offer before a selected fixture."""

import argparse
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

ENDPOINT = "https://production-superbet-offer-ro.freetls.fastly.net/v3/ro-RO/events"


def utc_timestamp(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None:
        raise ValueError("Kickoff must include a time zone, such as Z or +02:00")
    return timestamp.astimezone(UTC)


def check_window(kickoff: datetime, observed_at: datetime) -> None:
    if not timedelta(minutes=10) <= kickoff - observed_at <= timedelta(minutes=20):
        raise ValueError("Capture must take place 10–20 minutes before kickoff")


def read_event(payload: dict, event_id: int, kickoff: datetime) -> dict:
    """Check that the response contains the requested fixture and offer."""
    events = [event for event in payload["events"] if int(event["event_id"]) == event_id]
    if len(events) != 1:
        raise ValueError(f"Expected one event with ID {event_id}")
    event = events[0]
    if utc_timestamp(event["fixture"]["utc_date"]) != kickoff:
        raise ValueError("Kickoff differs from the supplied time. Check the fixture before retrying.")
    if not event.get("markets"):
        raise ValueError("The fixture has no markets in the response")
    return event


def capture(event_id: int, kickoff: datetime, output_dir: Path) -> Path:
    """Fetch and save a single offer inside the pre-match window."""
    output = output_dir / f"{event_id}.json"
    if event_id <= 0 or output.exists():
        raise ValueError("Expected a positive event ID with no existing snapshot")
    check_window(kickoff, datetime.now(UTC))
    url = f"{ENDPOINT}?{urlencode({'events': event_id})}"
    with urlopen(url, timeout=15) as response:
        payload = json.load(response)
    observed_at = datetime.now(UTC)
    check_window(kickoff, observed_at)
    event = read_event(payload, event_id, kickoff)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(
            {"observed_at_utc": observed_at.isoformat(), "source": url, "event": event},
            handle, ensure_ascii=False, indent=2,
        )
    print(f"Saved {len(event['markets'])} markets to {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_id", type=int, help="Numeric ID after e- in the Superbet match URL")
    parser.add_argument("--kickoff", required=True, help="Scheduled kickoff with explicit time zone")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/superbet"))
    args = parser.parse_args()
    try:
        capture(args.event_id, utc_timestamp(args.kickoff), args.output_dir)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"Snapshot failed: {error}\n")


if __name__ == "__main__":
    main()
