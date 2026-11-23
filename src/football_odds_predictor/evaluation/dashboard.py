"""Serve the local match dashboard using saved models and offer snapshots."""

import argparse
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import pickle
from urllib.parse import parse_qs, urlsplit

from ..training.fit_sporting import DEFAULT_MODELS
from .superbet_report import RAW, build_report, read_snapshot

PAGE = Path(__file__).with_name("dashboard.html")


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
    @lru_cache(maxsize=16)
    def report(path: Path, modified: int, probe: bool, stress: Decimal | None) -> dict:
        # The modification time invalidates a cache entry if the local snapshot changes.
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
            items = snapshots()
            if request.path == "/api/matches":
                self.send_json([{k: v for k, v in item.items() if k != "path"} for item in items.values()])
                return
            parameters = parse_qs(request.query)
            item = items.get(parameters.get("match", [""])[0])
            if item is None:
                self.send_json({"error": "Capture not found."}, 404)
                return
            try:
                stress = Decimal(parameters.get("stress", ["0"])[0])
                if not stress.is_finite() or not 0 <= stress < 100:
                    raise ValueError("Sensitivity must be from 0 to 99 percentage points.")
                path = item["path"]
                result = report(path, path.stat().st_mtime_ns, item["probe"], stress / 100 if stress else None)
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
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(models)) as server:
        print(f"Dashboard: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Stopped dashboard.")


if __name__ == "__main__":
    main()
