"""
Thin client for the football-data.org v4 API.
Free tier: 10 requests / minute — a 6.5 s delay is inserted after each call.
"""

import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

_BASE = "https://api.football-data.org/v4"
_RATE_DELAY = 6.5  # seconds between calls (free tier: 10 req/min)
_WC_CODE = "WC"


def _headers() -> dict:
    return {"X-Auth-Token": os.getenv("FOOTBALL_DATA_API_KEY", "")}


def _get(path: str, params: Optional[dict] = None) -> dict:
    resp = requests.get(f"{_BASE}/{path}", headers=_headers(), params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(_RATE_DELAY)
    return resp.json()


def get_wc_teams() -> list[dict]:
    """Return list of team objects for WC 2026."""
    try:
        data = _get(f"competitions/{_WC_CODE}/teams")
        return data.get("teams", [])
    except requests.HTTPError as exc:
        print(f"[API] Error fetching WC teams: {exc}")
        return []


def get_wc_matches(status: Optional[str] = None) -> list[dict]:
    """Return WC 2026 match objects, optionally filtered by status."""
    params: dict = {}
    if status:
        params["status"] = status
    try:
        data = _get(f"competitions/{_WC_CODE}/matches", params)
        return data.get("matches", [])
    except requests.HTTPError as exc:
        print(f"[API] Error fetching WC matches: {exc}")
        return []


def get_wc_scorers(limit: int = 100) -> list[dict]:
    """Return top scorer objects for WC 2026."""
    try:
        data = _get(f"competitions/{_WC_CODE}/scorers", {"limit": limit})
        return data.get("scorers", [])
    except requests.HTTPError as exc:
        print(f"[API] Error fetching WC scorers: {exc}")
        return []


def get_team_squad(api_team_id: int) -> list[dict]:
    """Return squad player objects for a single team."""
    try:
        data = _get(f"teams/{api_team_id}")
        return data.get("squad", [])
    except requests.HTTPError as exc:
        print(f"[API] Error fetching squad for team {api_team_id}: {exc}")
        return []
