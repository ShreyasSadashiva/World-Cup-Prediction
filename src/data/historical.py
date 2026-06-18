"""
Loader for the martj42 international football results CSV dataset.
URL: https://raw.githubusercontent.com/martj42/international_results/master/results.csv

Columns: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral
"""

import io
from typing import Optional

import pandas as pd
import requests

_CSV_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
_cache: Optional[pd.DataFrame] = None

WC_2026_START = pd.Timestamp("2026-06-11")
WC_TOURNAMENT_KEYWORDS = ["FIFA World Cup 2026", "FIFA World Cup qualification"]


def load(force_refresh: bool = False) -> pd.DataFrame:
    """Download and cache the full historical results CSV."""
    global _cache
    if _cache is not None and not force_refresh:
        return _cache

    print("[Historical] Downloading results CSV…")
    resp = requests.get(_CSV_URL, timeout=90)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    df["date"] = pd.to_datetime(df["date"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    _cache = df
    print(f"[Historical] Loaded {len(df):,} matches")
    return df


def get_team_pre_wc_matches(
    df: pd.DataFrame,
    team_name: str,
    n: int = 15,
    exclude_wc_qualifiers: bool = False,
) -> pd.DataFrame:
    """
    Return up to `n` matches for `team_name` that finished before WC 2026 started.
    Excludes WC 2026 matches; optionally excludes WC qualification matches too.
    """
    mask = (df["home_team"] == team_name) | (df["away_team"] == team_name)
    team_df = df[mask & (df["date"] < WC_2026_START)].copy()

    # Always exclude 2026 WC matches
    team_df = team_df[~team_df["tournament"].str.contains("World Cup 2026", case=False, na=False)]

    if exclude_wc_qualifiers:
        team_df = team_df[~team_df["tournament"].str.contains("qualification", case=False, na=False)]

    return team_df.sort_values("date").tail(n)


def get_h2h(df: pd.DataFrame, team1: str, team2: str) -> pd.DataFrame:
    """Return all historical matches between team1 and team2 (any order)."""
    mask = (
        ((df["home_team"] == team1) & (df["away_team"] == team2))
        | ((df["home_team"] == team2) & (df["away_team"] == team1))
    )
    return df[mask].sort_values("date").copy()
