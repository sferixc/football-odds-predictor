"""Download and audit historical Understat player data published by xGTables."""

import argparse
import csv
import io
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

COMPETITIONS = ("epl", "bundesliga", "la-liga")
SEASONS = range(2021, 2026)
REQUIRED_COLUMNS = {
    "fixtures": {"fixture_id", "date", "home", "away"},
    "appearances": {"fixture_id", "player_id", "side", "position", "minutes", "shots"},
    "shots": {"shot_id", "fixture_id", "player_id", "side", "result"},
}
BASE_URL = "https://www.xgtables.io/history"


def download_archive(root: Path) -> None:
    """Fetch each published CSV once, leaving existing local files intact."""
    for competition in COMPETITIONS:
        for season in SEASONS:
            folder = root / competition / str(season)
            folder.mkdir(parents=True, exist_ok=True)
            for table, required in REQUIRED_COLUMNS.items():
                path = folder / f"{table}.csv"
                if path.exists():
                    continue
                url = f"{BASE_URL}/{competition}/{season}/{path.name}"
                with urlopen(url, timeout=30) as response:
                    contents = response.read()
                reader = csv.DictReader(io.StringIO(contents.decode("utf-8-sig")))
                if not required.issubset(reader.fieldnames or []) or next(reader, None) is None:
                    raise ValueError(f"Empty or unexpected CSV from {url}")
                # Rename only after receiving and checking the entire response.
                temporary = path.with_suffix(".download")
                temporary.write_bytes(contents)
                temporary.replace(path)


def audit_archive(root: Path) -> None:
    """Count coverage and report source discrepancies without filling gaps."""
    for competition in COMPETITIONS:
        counts = dict(fixtures=0, appearances=0, shots=0, no_shot_map=0, shot_count_mismatches=0)
        for season in SEASONS:
            folder = root / competition / str(season)
            fixtures, appearances, shots = (
                pd.read_csv(folder / f"{table}.csv", usecols=sorted(columns))
                for table, columns in REQUIRED_COLUMNS.items()
            )
            key = ["fixture_id", "side", "player_id"]
            if (fixtures["fixture_id"].duplicated().any() or appearances.duplicated(key).any()
                    or shots["shot_id"].duplicated().any()):
                raise ValueError(f"Duplicate fixture, appearance, or shot in {folder}")
            if not all(table["fixture_id"].isin(fixtures["fixture_id"]).all()
                       for table in (appearances, shots)):
                raise ValueError(f"Unmatched fixture IDs in {folder}")
            # Understat own-goal events can use the opposing side and do not count as shots.
            shot_events = shots.loc[shots["result"].ne("OwnGoal")]
            shot_counts = shot_events.groupby(key).size()
            reported = appearances.set_index(key)["shots"]
            if not shot_counts.index.isin(reported.index).all():
                raise ValueError(f"Shot without a matching player appearance in {folder}")
            # An absent fixture shot map differs from zero shots by one player.
            mapped = appearances["fixture_id"].isin(shots["fixture_id"]).to_numpy()
            recorded = shot_counts.reindex(reported.index, fill_value=0)
            counts["fixtures"] += len(fixtures)
            counts["appearances"] += len(appearances)
            counts["shots"] += len(shots)
            counts["no_shot_map"] += (~fixtures["fixture_id"].isin(shots["fixture_id"])).sum()
            counts["shot_count_mismatches"] += reported[mapped].ne(recorded[mapped]).sum()
        print(competition, " | ".join(f"{name}={count:,}" for name, count in counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/xgtables-understat"))
    parser.add_argument("--download", action="store_true", help="Fetch any missing archive files")
    args = parser.parse_args()
    if args.download:
        download_archive(args.root)
    audit_archive(args.root)


if __name__ == "__main__":
    main()
