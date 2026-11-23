"""Serve the local match dashboard using saved models and offer snapshots."""

import argparse
from decimal import Decimal, InvalidOperation
from errno import EADDRINUSE
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import pickle
from datetime import UTC, datetime, timedelta
from time import monotonic
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

from ..data.referees import APPOINTMENTS
from ..data.superbet_watch import discover
from ..data.superbet_snapshot import utc_timestamp
from ..models.cards import load_card_models
from ..models.ucl import load_ucl_models
from ..training.fit_sporting import DEFAULT_MODELS
from .superbet_report import RAW, build_report, read_snapshot

PAGE = Path(__file__).with_name("dashboard.html")
DASHBOARD_ZONE = ZoneInfo("Europe/Amsterdam")


def snapshots() -> dict:
    """Index saved captures and local probes without exposing arbitrary file paths."""
    items = {}
    for folder, probe in (("superbet", False), ("superbet-probes", True)):
        for path in (RAW / folder).glob("*.json"):
            try:
                event, observed = read_snapshot(path, probe)
                fixture = event["fixture"]
                key = f"{folder}/{path.stem}"
                items[key] = {
                    "id": key, "name": fixture["event_name"].replace("·", " – "),
                    "kickoff": fixture["utc_date"], "probe": probe, "path": path,
                }
            except (KeyError, OSError, ValueError):
                continue
    return dict(sorted(
        items.items(), key=lambda item: (not item[1]["probe"], item[1]["kickoff"]), reverse=True,
    ))


def handler_for(models: dict) -> type[BaseHTTPRequestHandler]:
    schedule_cache: dict = {"at": 0.0, "fixtures": [], "error": None}

    def today_fixtures() -> tuple[list[dict], str | None, str]:
        """Combine today's public schedule with saved offers; refresh schedule every 2 minutes."""
        now = datetime.now(UTC)
        today = now.astimezone(DASHBOARD_ZONE).date()
        if monotonic() - schedule_cache["at"] >= 120:
            try:
                schedule_cache["fixtures"] = discover(now)
                schedule_cache["error"] = None
            except (KeyError, OSError, ValueError) as error:
                schedule_cache["error"] = str(error)
            schedule_cache["at"] = monotonic()

        merged = {}
        for fixture in schedule_cache["fixtures"]:
            if fixture.kickoff.astimezone(DASHBOARD_ZONE).date() != today:
                continue
            merged[fixture.event_id] = {
                "id": f"scheduled/{fixture.event_id}", "event_id": fixture.event_id,
                "name": fixture.name.replace("·", " – "), "league": fixture.league,
                "kickoff": fixture.kickoff.isoformat(), "captured": False, "probe": False,
            }

        captures = snapshots()
        for item in captures.values():
            try:
                kickoff = utc_timestamp(item["kickoff"])
                if kickoff.astimezone(DASHBOARD_ZONE).date() != today:
                    continue
                event_id = int(item["path"].stem)
                merged[event_id] = {
                    "id": item["id"], "event_id": event_id, "name": item["name"],
                    "league": "", "kickoff": item["kickoff"], "captured": True,
                    "probe": item["probe"],
                }
            except (ValueError, OSError):
                continue
        fixtures = sorted(merged.values(), key=lambda item: item["kickoff"])
        return fixtures, schedule_cache["error"], today.isoformat()

    @lru_cache(maxsize=16)
    def report(path: Path, modified: int, appointments_modified: int, probe: bool, stress: Decimal | None) -> dict:
        # Modification times invalidate the cache when local offers or referee appointments change.
        event, observed = read_snapshot(path, probe)
        return build_report(event, observed, stress, models)

    class DashboardHandler(BaseHTTPRequestHandler):
        def send_body(self, body: bytes, media_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", media_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value, status: int = 200) -> None:
            self.send_body(json.dumps(value, allow_nan=False).encode(), "application/json", status)

        def do_GET(self) -> None:
            request = urlsplit(self.path)
            if request.path == "/":
                self.send_body(PAGE.read_bytes(), "text/html; charset=utf-8")
                return
            if request.path not in ("/api/matches", "/api/report"):
                self.send_json({"error": "Not found."}, 404)
                return
            if request.path == "/api/matches":
                fixtures, error, today = today_fixtures()
                self.send_json({"date": today, "fixtures": fixtures, "schedule_error": error})
                return
            parameters = parse_qs(request.query)
            match_id = parameters.get("match", [""])[0]
            if match_id.startswith("scheduled/"):
                event_id = match_id.split("/", maxsplit=1)[1]
                fixture = next((f for f in today_fixtures()[0] if str(f["event_id"]) == event_id), None)
                if fixture is None:
                    self.send_json({"error": "Today's fixture is no longer on the schedule."}, 404)
                else:
                    self.send_json({
                        "waiting": True, "fixture": fixture["name"], "kickoff": fixture["kickoff"],
                        "competition": fixture["league"],
                    })
                return
            item = snapshots().get(match_id)
            if item is None:
                self.send_json({"error": "Capture not found."}, 404)
                return
            try:
                stress = Decimal(parameters.get("stress", ["0"])[0])
                if not stress.is_finite() or not 0 <= stress < 100:
                    raise ValueError("Sensitivity must be from 0 to 99 percentage points.")
                path = item["path"]
                appointments_modified = APPOINTMENTS.stat().st_mtime_ns if APPOINTMENTS.is_file() else 0
                result = report(
                    path, path.stat().st_mtime_ns, appointments_modified, item["probe"],
                    stress / 100 if stress else None,
                )
                self.send_json(result)
            except (InvalidOperation, KeyError, OSError, ValueError) as error:
                self.send_json({"error": str(error)}, 422)

    return DashboardHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    args = parser.parse_args()
    if not args.models.is_file():
        parser.error("Model file missing. Run football_odds_predictor.training.fit_sporting first.")
    with args.models.open("rb") as handle:
        models = pickle.load(handle)
    load_card_models(models)
    load_ucl_models(models)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(models))
    except OSError as error:
        if error.errno == EADDRINUSE:
            parser.exit(1, f"Port {args.port} is already in use. Stop the earlier dashboard "
                        "or rerun with --port 0 to pick a free port.\n")
        parser.exit(1, f"Could not start the dashboard on port {args.port}: {error.strerror}\n")
    with server:
        print(f"Dashboard: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Stopped dashboard.")


if __name__ == "__main__":
    main()
