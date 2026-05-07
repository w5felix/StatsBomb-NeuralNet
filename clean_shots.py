#!/usr/bin/env python3
"""
Clean up the dataset to include only shots information across all matches.

Outputs:
- data/clean/shots.csv
- data/clean/shots.jsonl

Fields included per shot:
- match_id (from events filename)
- event_id
- period
- timestamp
- minute
- second
- team
- player
- location_x
- location_y
- outcome
- xg

Usage:
  python3 clean_shots.py
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable

ROOT = Path(__file__).resolve().parent
EVENTS_DIR = ROOT / "data" / "events"
OUT_DIR = ROOT / "data" / "clean"
CSV_PATH = OUT_DIR / "shots.csv"
JSONL_PATH = OUT_DIR / "shots.jsonl"


def iter_event_files() -> Iterable[Path]:
    if not EVENTS_DIR.exists():
        raise SystemExit(f"Events directory not found: {EVENTS_DIR}")
    yield from sorted(EVENTS_DIR.glob("*.json"), key=lambda p: p.name)


def extract_shot_records(file_path: Path) -> Iterable[Dict[str, Any]]:
    match_id = file_path.stem  # e.g., 7444
    with file_path.open("r", encoding="utf-8") as f:
        try:
            events = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse {file_path}: {e}")
    for ev in events:
        # Filter only shot events
        ev_type = ((ev.get("type") or {}).get("name") or "").lower()
        if ev_type != "shot":
            continue
        shot = ev.get("shot") or {}
        loc = ev.get("location") or [None, None]
        outcome = (shot.get("outcome") or {}).get("name")
        xg = shot.get("statsbomb_xg")
        rec = {
            "match_id": match_id,
            "event_id": ev.get("id"),
            "period": ev.get("period"),
            "timestamp": ev.get("timestamp"),
            "minute": ev.get("minute"),
            "second": ev.get("second"),
            "team": ((ev.get("team") or {}).get("name")),
            "player": ((ev.get("player") or {}).get("name")),
            "location_x": loc[0] if isinstance(loc, list) and len(loc) > 0 else None,
            "location_y": loc[1] if isinstance(loc, list) and len(loc) > 1 else None,
            "outcome": outcome,
            "xg": xg,
        }
        yield rec


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "match_id",
        "event_id",
        "period",
        "timestamp",
        "minute",
        "second",
        "team",
        "player",
        "location_x",
        "location_y",
        "outcome",
        "xg",
    ]

    total = 0
    # Write CSV and JSONL simultaneously
    with CSV_PATH.open("w", encoding="utf-8", newline="") as csvfile, JSONL_PATH.open(
        "w", encoding="utf-8"
    ) as jsonlfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for path in iter_event_files():
            for rec in extract_shot_records(path):
                writer.writerow(rec)
                jsonlfile.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1

    print(f"Wrote {total} shots to:\n - {CSV_PATH}\n - {JSONL_PATH}")


if __name__ == "__main__":
    main()
