"""
First-run database seeding for Firestore.

Collections populated:
  teams · team_recent_matches · wc_fixtures · players · wc_player_stats
"""

import time
from typing import Optional

import pandas as pd

from src.data.football_api import (
    get_team_squad,
    get_wc_matches,
    get_wc_scorers,
    get_wc_teams,
)
from src.data.historical import get_team_pre_wc_matches, load
from src.data.team_mappings import normalize
from src.db.client import get_client, slugify

# ── helpers ──────────────────────────────────────────────────────────────────

_slug_cache: dict[str, str] = {}  # canonical_name → slug


def _ensure_team(csv_name: str) -> str:
    """Return the Firestore doc ID (slug) for a team, creating it if missing."""
    if csv_name in _slug_cache:
        return _slug_cache[csv_name]

    db = get_client()
    slug = slugify(csv_name)
    doc = db.collection("teams").document(slug).get()
    if not doc.exists:
        db.collection("teams").document(slug).set(
            {"name": csv_name, "country_code": csv_name[:3].upper()}
        )
    _slug_cache[csv_name] = slug
    return slug


def _group_from(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    return raw.replace("GROUP_", "").strip() or None


# ── step 1: teams ─────────────────────────────────────────────────────────────

def seed_teams() -> dict[int, str]:
    """
    Upsert WC 2026 teams from API.
    Returns {api_id: slug}.
    """
    print("\n[Seeder] Step 1 — teams")
    api_teams = get_wc_teams()
    if not api_teams:
        print("  WARNING: No teams from API.")
        return {}

    db = get_client()
    api_to_slug: dict[int, str] = {}

    for t in api_teams:
        api_name = t.get("name") or t.get("shortName", "Unknown")
        csv_name = normalize(api_name)
        slug = slugify(csv_name)

        doc_data = {
            "name":         csv_name,
            "country_code": (t.get("tla") or csv_name[:3]).upper()[:3],
            "api_id":       t["id"],
            "group_name":   _group_from(t.get("group")),
        }
        db.collection("teams").document(slug).set(doc_data, merge=True)
        api_to_slug[t["id"]] = slug
        _slug_cache[csv_name] = slug
        time.sleep(0.02)

    print(f"  Upserted {len(api_to_slug)} teams")
    return api_to_slug


# ── step 2: recent matches ────────────────────────────────────────────────────

def seed_recent_matches(api_to_slug: dict[int, str], n: int = 15) -> None:
    """Insert last `n` pre-WC matches per team from the historical CSV."""
    print(f"\n[Seeder] Step 2 — last {n} pre-WC matches per team")
    hist_df = load()
    db = get_client()
    total = 0

    for slug in api_to_slug.values():
        team_doc = db.collection("teams").document(slug).get()
        if not team_doc.exists:
            continue
        csv_name = team_doc.to_dict().get("name", slug)

        matches = get_team_pre_wc_matches(hist_df, csv_name, n=n)
        if matches.empty:
            print(f"  WARNING: no historical data for '{csv_name}'")
            continue

        for _, row in matches.iterrows():
            home_name = str(row["home_team"])
            away_name = str(row["away_team"])
            home_slug = _ensure_team(home_name)
            away_slug = _ensure_team(away_name)
            date_str = str(row["date"].date())
            doc_id = f"{home_slug}_{away_slug}_{date_str}"

            doc_data = {
                "home_team_id":   home_slug,
                "home_team_name": home_name,
                "away_team_id":   away_slug,
                "away_team_name": away_name,
                "match_date":     date_str,
                "competition":    str(row.get("tournament", "")),
                "home_goals":     int(row["home_score"]),
                "away_goals":     int(row["away_score"]),
                "neutral":        bool(row.get("neutral", False)),
            }
            try:
                db.collection("team_recent_matches").document(doc_id).set(
                    doc_data, merge=True
                )
                total += 1
            except Exception as exc:
                print(f"  Skip {doc_id}: {exc}")
            time.sleep(0.01)

        print(f"  {csv_name}: {len(matches)} matches")

    print(f"  Total rows written: {total}")


# ── step 3: WC fixtures ───────────────────────────────────────────────────────

def seed_wc_fixtures(api_to_slug: dict[int, str]) -> None:
    """Fetch all WC 2026 fixtures from API and write to wc_fixtures."""
    print("\n[Seeder] Step 3 — WC fixtures")
    matches = get_wc_matches()
    if not matches:
        print("  WARNING: no fixtures from API.")
        return

    db = get_client()
    inserted = 0

    for m in matches:
        home_raw = m["homeTeam"].get("name")
        away_raw = m["awayTeam"].get("name")
        if not home_raw or not away_raw:
            # Knockout TBD — store fixture shell without team references
            ft = (m.get("score") or {}).get("fullTime") or {}
            shell = {
                "home_team_id": None, "home_team_name": "TBD",
                "away_team_id": None, "away_team_name": "TBD",
                "match_date": m.get("utcDate"),
                "stage": m.get("stage", "KNOCKOUT"),
                "group_name": None,
                "home_goals": ft.get("home"), "away_goals": ft.get("away"),
                "status": m.get("status", "SCHEDULED"),
                "venue": m.get("venue"), "api_match_id": m["id"],
            }
            try:
                db.collection("wc_fixtures").document(str(m["id"])).set(shell, merge=True)
                inserted += 1
            except Exception as exc:
                print(f"  Skip TBD fixture {m['id']}: {exc}")
            time.sleep(0.01)
            continue

        home_api = m["homeTeam"]["id"]
        away_api = m["awayTeam"]["id"]
        home_slug = api_to_slug.get(home_api, slugify(normalize(home_raw)))
        away_slug = api_to_slug.get(away_api, slugify(normalize(away_raw)))

        home_name = normalize(home_raw)
        away_name = normalize(away_raw)

        ft = (m.get("score") or {}).get("fullTime") or {}
        doc_data = {
            "home_team_id":   home_slug,
            "home_team_name": home_name,
            "away_team_id":   away_slug,
            "away_team_name": away_name,
            "match_date":     m.get("utcDate"),
            "stage":          m.get("stage", "GROUP_STAGE"),
            "group_name":     _group_from(m.get("group")),
            "home_goals":     ft.get("home"),
            "away_goals":     ft.get("away"),
            "status":         m.get("status", "SCHEDULED"),
            "venue":          m.get("venue"),
            "api_match_id":   m["id"],
        }
        group = _group_from(m.get("group"))
        try:
            db.collection("wc_fixtures").document(str(m["id"])).set(doc_data, merge=True)
            # Back-fill group_name onto team docs (teams endpoint doesn't include it)
            if group:
                db.collection("teams").document(home_slug).set({"group_name": group}, merge=True)
                db.collection("teams").document(away_slug).set({"group_name": group}, merge=True)
            inserted += 1
        except Exception as exc:
            print(f"  Skip fixture {m['id']}: {exc}")
        time.sleep(0.01)

    print(f"  Inserted/updated {inserted} fixtures")


# ── step 4: players ───────────────────────────────────────────────────────────

def seed_players(api_to_slug: dict[int, str]) -> dict[int, str]:
    """
    Fetch squad per team and write to players collection.
    Returns {api_player_id: player_doc_id}.
    """
    print("\n[Seeder] Step 4 — player squads")
    db = get_client()
    api_pid_to_doc: dict[int, str] = {}

    for api_team_id, team_slug in api_to_slug.items():
        team_doc = db.collection("teams").document(team_slug).get()
        team_name = team_doc.to_dict().get("name", team_slug) if team_doc.exists else team_slug

        squad = get_team_squad(api_team_id)
        for p in squad:
            player_doc_id = str(p["id"])
            doc_data = {
                "team_id":       team_slug,
                "team_name":     team_name,
                "name":          p.get("name", "Unknown"),
                "position":      p.get("position", ""),
                "date_of_birth": p.get("dateOfBirth"),
                "nationality":   p.get("nationality", ""),
                "api_id":        p["id"],
            }
            try:
                db.collection("players").document(player_doc_id).set(doc_data, merge=True)
                api_pid_to_doc[p["id"]] = player_doc_id
            except Exception as exc:
                print(f"  Skip player {p.get('name')}: {exc}")
            time.sleep(0.005)

        print(f"  {team_name}: {len(squad)} players")

    return api_pid_to_doc


# ── step 5: player stats ──────────────────────────────────────────────────────

def seed_player_stats(api_pid_to_doc: dict[int, str]) -> None:
    """Fetch WC top scorers and write to wc_player_stats."""
    print("\n[Seeder] Step 5 — WC player stats")
    scorers = get_wc_scorers()
    if not scorers:
        print("  No scorer data yet.")
        return

    db = get_client()
    updated = 0

    for entry in scorers:
        api_pid = entry["player"]["id"]
        doc_id = api_pid_to_doc.get(api_pid)
        if not doc_id:
            continue

        player_data = db.collection("players").document(doc_id).get()
        player_name = (player_data.to_dict() or {}).get("name", "?") if player_data.exists else "?"
        team_id = (player_data.to_dict() or {}).get("team_id", "") if player_data.exists else ""
        team_name = (player_data.to_dict() or {}).get("team_name", "") if player_data.exists else ""

        doc_data = {
            "player_id":      doc_id,
            "player_name":    player_name,
            "team_id":        team_id,
            "team_name":      team_name,
            "goals":          entry.get("goals", 0) or 0,
            "assists":        entry.get("assists", 0) or 0,
            "yellow_cards":   entry.get("yellowCards", 0) or 0,
            "red_cards":      entry.get("redCards", 0) or 0,
            "minutes_played": entry.get("playedMatches", 0) or 0,
        }
        try:
            db.collection("wc_player_stats").document(doc_id).set(doc_data, merge=True)
            updated += 1
        except Exception as exc:
            print(f"  Skip stats {api_pid}: {exc}")

    print(f"  Updated {updated} player records")


# ── entry point ───────────────────────────────────────────────────────────────

def run_full_seed(n_recent: int = 15) -> None:
    api_to_slug = seed_teams()
    seed_recent_matches(api_to_slug, n=n_recent)
    seed_wc_fixtures(api_to_slug)
    api_pid_to_doc = seed_players(api_to_slug)
    seed_player_stats(api_pid_to_doc)
    print("\n[Seeder] Done.")
