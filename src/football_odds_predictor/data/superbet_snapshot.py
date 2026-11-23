"""Save one public Superbet Romania offer before a selected fixture."""

import argparse
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

UCL_TOURNAMENT = 80794
UEL_TOURNAMENT = 688
EUROPEAN_TOURNAMENTS = {UCL_TOURNAMENT: "CL", UEL_TOURNAMENT: "EL"}
ENDPOINT = "https://production-superbet-offer-ro.freetls.fastly.net/v3/ro-RO/events"
# Source market IDs mapped to the event names used in the sporting analysis.
SIMPLE_MARKETS = {
    "547": "1x2",
    "200734": "match_goals",
    "200735": "first_half_goals",
    "200738": "second_half_goals",
    "704": "match_corners",
    "713": "home_corners",
    "733": "away_corners",
    "690": "match_card_points",
    "700": "home_card_points",
    "708": "away_card_points",
    "236218": "player_total_shots",
}


def open_price(odd: dict) -> bool:
    """Check the flags and price used for currently selectable outcomes."""
    price = odd.get("price")
    return (
        odd.get("status") == 1
        and odd.get("display") is True
        and isinstance(price, (int, float))
        and math.isfinite(price)
        and price > 1
    )


def simple_promotions(event: dict) -> list[dict]:
    """Extract open single selections that map to a supported sporting market."""
    # Match the promotional and standard 1X2 markets by outcome (1, X, or 2).
    standard = {}
    for market in event["markets"]:
        if market["id"] == 547:
            standard.update(
                (odd["metadata"]["name"], odd["price"])
                for odd in market["odds"] if open_price(odd)
            )

    promotions = []
    for market in event["markets"]:
        if market["id"] != 201013:
            continue
        for odd in market["odds"]:
            selection = odd["metadata"]["name"].removeprefix("SC-")
            if open_price(odd) and selection in ("1", "X", "2"):
                promotions.append({
                    "type": "super_cota",
                    "market": "1x2",
                    "selection": selection,
                    "odds": odd["price"],
                    "standard_odds": standard.get(selection),
                    "offer_id": odd["uuid"],
                })

    for market in event.get("priceboosts", []):
        for odd in market["odds"]:
            if not open_price(odd):
                continue
            # More than one component needs a joint probability and stays in the raw offer.
            metadata = odd["metadata"]
            components = metadata.get("odd_components") or []
            if len(components) != 1:
                continue
            component = components[0]
            market_name = SIMPLE_MARKETS.get(str(component.get("universal_market_id")))
            selection = component.get("odd_name")
            if market_name is None or not selection:
                continue
            original = metadata.get("extra", {}).get("originalPrice")
            promotions.append({
                "type": "price_boost",
                "market": market_name,
                "selection": selection,
                "specifiers": component.get("specifiers", {}),
                "odds": odd["price"],
                "original_price_reference": original,
                "offer_id": odd["uuid"],
            })
    return promotions


def utc_timestamp(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None:
        raise ValueError("Kickoff must include a time zone, such as Z or +02:00")
    return timestamp.astimezone(UTC)


def check_window(kickoff: datetime, observed_at: datetime) -> None:
    """Require an observed price 10 to 20 minutes before scheduled kickoff."""
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
    # Record completion time and recheck the window in case the request ran long.
    observed_at = datetime.now(UTC)
    check_window(kickoff, observed_at)
    event = read_event(payload, event_id, kickoff)
    promotions = simple_promotions(event)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        # Keep the original event so future parsers can re-read markets we skip today.
        json.dump(
            {
                "observed_at_utc": observed_at.isoformat(),
                "source": url,
                "simple_promotions": promotions,
                "event": event,
            },
            handle, ensure_ascii=False, indent=2,
        )
    print(f"Saved {len(event['markets'])} markets and {len(promotions)} simple promotions to {output}", flush=True)
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
