"""
Manual WC results updater.

Usage:
    python -m scripts.update_wc
    run()   ← called from the Fixtures page sidebar button
"""

import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.football_api import get_wc_matches, get_wc_scorers
from src.data.team_mappings import normalize
from src.db.client import get_client, slugify


def _team_slug(api_name: Optional[str]) -> Optional[str]:
    if not api_name:
        return None
    name = normalize(api_name)
    slug = slugify(name)
    doc = get_client().collection("teams").document(slug).get()
    return slug if doc.exists else None


def update_fixtures() -> int:
    matches = get_wc_matches()
    db = get_client()
    updated = 0

    for m in matches:
        home_slug = _team_slug(m["homeTeam"]["name"])
        away_slug = _team_slug(m["awayTeam"]["name"])
        if not home_slug or not away_slug:
            continue

        ft = (m.get("score") or {}).get("fullTime") or {}
        group_raw = m.get("group") or ""
        group = group_raw.replace("GROUP_", "").strip() or None

        update_data = {
            "home_goals": ft.get("home"),
            "away_goals": ft.get("away"),
            "status":     m.get("status", "SCHEDULED"),
            "group_name": group,
        }
        try:
            db.collection("wc_fixtures").document(str(m["id"])).set(
                update_data, merge=True
            )
            updated += 1
        except Exception as exc:
            print(f"  Skip fixture {m['id']}: {exc}")
        time.sleep(0.01)

    return updated


def update_player_stats() -> int:
    scorers = get_wc_scorers()
    db = get_client()
    updated = 0

    for entry in scorers:
        api_pid = entry["player"]["id"]
        player_doc_id = str(api_pid)
        player_ref = db.collection("players").document(player_doc_id)
        player_snap = player_ref.get()
        if not player_snap.exists:
            continue

        update_data = {
            "goals":          entry.get("goals", 0) or 0,
            "assists":        entry.get("assists", 0) or 0,
            "yellow_cards":   entry.get("yellowCards", 0) or 0,
            "red_cards":      entry.get("redCards", 0) or 0,
            "minutes_played": entry.get("playedMatches", 0) or 0,
        }
        try:
            db.collection("wc_player_stats").document(player_doc_id).set(
                update_data, merge=True
            )
            updated += 1
        except Exception as exc:
            print(f"  Skip player stats {api_pid}: {exc}")

    return updated


def run() -> tuple:
    print("[Update] Syncing WC fixtures…")
    f = update_fixtures()
    print(f"[Update] {f} fixtures updated")

    print("[Update] Syncing player stats…")
    p = update_player_stats()
    print(f"[Update] {p} player records updated")
    return f, p


if __name__ == "__main__":
    run()
