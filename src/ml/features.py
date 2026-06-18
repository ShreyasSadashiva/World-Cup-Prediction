"""
Feature engineering for match outcome prediction.
Team IDs are Firestore document slugs (strings), e.g. "brazil", "united-states".
"""

import numpy as np
import pandas as pd

from src.db.client import get_client

_FEATURE_COLS = [
    "home_win_rate", "home_draw_rate", "home_loss_rate",
    "home_avg_goals_scored", "home_avg_goals_conceded", "home_avg_goal_diff",
    "home_weighted_form", "home_clean_sheet_rate", "home_scoring_rate",
    "away_win_rate", "away_draw_rate", "away_loss_rate",
    "away_avg_goals_scored", "away_avg_goals_conceded", "away_avg_goal_diff",
    "away_weighted_form", "away_clean_sheet_rate", "away_scoring_rate",
    "win_rate_diff", "form_diff", "goal_diff_diff",
    "goals_scored_diff", "goals_conceded_diff",
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_goal_diff", "h2h_n",
    "neutral", "wc_form_diff",
]

FEATURE_COLS = _FEATURE_COLS


def _team_stats(team_id: str) -> dict:
    """Compute form stats from team_recent_matches in Firestore."""
    db = get_client()

    home_docs = db.collection("team_recent_matches").where(
        "home_team_id", "==", team_id
    ).limit(15).stream()
    away_docs = db.collection("team_recent_matches").where(
        "away_team_id", "==", team_id
    ).limit(15).stream()

    records = []
    for doc in home_docs:
        d = doc.to_dict()
        gs, gc = d.get("home_goals", 0), d.get("away_goals", 0)
        records.append({
            "date": d.get("match_date", ""),
            "gs": gs, "gc": gc,
            "result": "W" if gs > gc else ("D" if gs == gc else "L"),
        })
    for doc in away_docs:
        d = doc.to_dict()
        gs, gc = d.get("away_goals", 0), d.get("home_goals", 0)
        records.append({
            "date": d.get("match_date", ""),
            "gs": gs, "gc": gc,
            "result": "W" if gs > gc else ("D" if gs == gc else "L"),
        })

    if not records:
        return _default_team_stats()

    records.sort(key=lambda x: x["date"])
    records = records[-15:]
    n = len(records)

    results = [r["result"] for r in records]
    gs_arr = np.array([r["gs"] for r in records], dtype=float)
    gc_arr = np.array([r["gc"] for r in records], dtype=float)
    form_pts = np.array(
        [3 if r == "W" else (1 if r == "D" else 0) for r in results], dtype=float
    )
    weights = np.exp(np.linspace(-1, 0, n))

    return {
        "win_rate":          results.count("W") / n,
        "draw_rate":         results.count("D") / n,
        "loss_rate":         results.count("L") / n,
        "avg_goals_scored":  float(gs_arr.mean()),
        "avg_goals_conceded":float(gc_arr.mean()),
        "avg_goal_diff":     float((gs_arr - gc_arr).mean()),
        "weighted_form":     float(np.average(form_pts, weights=weights)),
        "clean_sheet_rate":  float((gc_arr == 0).mean()),
        "scoring_rate":      float((gs_arr > 0).mean()),
    }


def _wc_form(team_id: str) -> float:
    """Win rate in finished WC 2026 matches for this team."""
    db = get_client()

    home_docs = list(
        db.collection("wc_fixtures")
        .where("home_team_id", "==", team_id)
        .where("status", "==", "FINISHED")
        .stream()
    )
    away_docs = list(
        db.collection("wc_fixtures")
        .where("away_team_id", "==", team_id)
        .where("status", "==", "FINISHED")
        .stream()
    )

    wins, total = 0, len(home_docs) + len(away_docs)
    for doc in home_docs:
        d = doc.to_dict()
        if (d.get("home_goals") or 0) > (d.get("away_goals") or 0):
            wins += 1
    for doc in away_docs:
        d = doc.to_dict()
        if (d.get("away_goals") or 0) > (d.get("home_goals") or 0):
            wins += 1

    return wins / total if total > 0 else 0.33


def _h2h_features(home_id: str, away_id: str) -> dict:
    """Head-to-head stats across team_recent_matches + finished WC fixtures."""
    db = get_client()

    def _stream(coll, hid, aid):
        return list(
            db.collection(coll).where("home_team_id", "==", hid).where("away_team_id", "==", aid).stream()
        )

    wc_filter_docs = lambda hid, aid: [
        d for d in db.collection("wc_fixtures")
        .where("home_team_id", "==", hid)
        .where("away_team_id", "==", aid)
        .stream()
        if d.to_dict().get("status") == "FINISHED"
    ]

    h_home = [d.to_dict() for d in _stream("team_recent_matches", home_id, away_id)] + \
             [d.to_dict() for d in wc_filter_docs(home_id, away_id)]
    h_swap = [d.to_dict() for d in _stream("team_recent_matches", away_id, home_id)] + \
             [d.to_dict() for d in wc_filter_docs(away_id, home_id)]

    if not h_home and not h_swap:
        return {"h2h_home_win_rate": 0.45, "h2h_draw_rate": 0.27, "h2h_goal_diff": 0.0, "h2h_n": 0}

    home_wins = sum(1 for r in h_home if (r.get("home_goals") or 0) > (r.get("away_goals") or 0))
    home_wins += sum(1 for r in h_swap if (r.get("away_goals") or 0) > (r.get("home_goals") or 0))
    draws = sum(
        1 for r in (h_home + h_swap)
        if (r.get("home_goals") or 0) == (r.get("away_goals") or 0)
    )
    gd = (
        [(r.get("home_goals", 0) or 0) - (r.get("away_goals", 0) or 0) for r in h_home] +
        [(r.get("away_goals", 0) or 0) - (r.get("home_goals", 0) or 0) for r in h_swap]
    )
    total = len(h_home) + len(h_swap)

    return {
        "h2h_home_win_rate": home_wins / total,
        "h2h_draw_rate":     draws / total,
        "h2h_goal_diff":     float(np.mean(gd)),
        "h2h_n":             total,
    }


def _default_team_stats() -> dict:
    return {
        "win_rate": 0.33, "draw_rate": 0.33, "loss_rate": 0.33,
        "avg_goals_scored": 1.1, "avg_goals_conceded": 1.1, "avg_goal_diff": 0.0,
        "weighted_form": 1.0, "clean_sheet_rate": 0.3, "scoring_rate": 0.7,
    }


def build_match_features(home_team_id: str, away_team_id: str, neutral: bool = False) -> dict:
    """Build the full 29-feature dict for a match prediction."""
    h = _team_stats(home_team_id)
    a = _team_stats(away_team_id)
    h2h = _h2h_features(home_team_id, away_team_id)
    wc_h = _wc_form(home_team_id)
    wc_a = _wc_form(away_team_id)

    features: dict = {}
    for k, v in h.items():
        features[f"home_{k}"] = v
    for k, v in a.items():
        features[f"away_{k}"] = v

    features["win_rate_diff"]       = h["win_rate"] - a["win_rate"]
    features["form_diff"]           = h["weighted_form"] - a["weighted_form"]
    features["goal_diff_diff"]      = h["avg_goal_diff"] - a["avg_goal_diff"]
    features["goals_scored_diff"]   = h["avg_goals_scored"] - a["avg_goals_scored"]
    features["goals_conceded_diff"] = h["avg_goals_conceded"] - a["avg_goals_conceded"]
    features.update(h2h)
    features["neutral"]      = int(neutral)
    features["wc_form_diff"] = wc_h - wc_a

    return {k: features.get(k, 0.0) for k in FEATURE_COLS}


def match_result_label(home_goals: int, away_goals: int, perspective: str = "home") -> str:
    """Return 'W', 'D', or 'L' from the given perspective."""
    if home_goals == away_goals:
        return "D"
    if perspective == "home":
        return "W" if home_goals > away_goals else "L"
    return "W" if away_goals > home_goals else "L"
