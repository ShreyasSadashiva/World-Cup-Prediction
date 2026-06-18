"""
Train the WC 2026 match outcome predictor.

Downloads the martj42 historical CSV, builds a rolling-window feature set
(no data leakage), trains XGBoost, runs 5-fold CV, and saves the model.

Usage (from project root):
    python -m scripts.train_model
"""

import io
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.ml.features import FEATURE_COLS
from src.ml.model import save

_CSV_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
_MIN_DATE = "2000-01-01"
_MIN_MATCHES = 7  # minimum prior matches needed to compute valid features
_N_WINDOW = 15    # rolling window size (matches last 15)


# ── feature computation (standalone, no DB) ──────────────────────────────────

def _team_stats_at(df: pd.DataFrame, team: str, before_date) -> Optional[dict]:
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    hist = df[mask & (df["date"] < before_date)].sort_values("date").tail(_N_WINDOW)

    if len(hist) < _MIN_MATCHES:
        return None

    results, gs_list, gc_list = [], [], []
    for _, r in hist.iterrows():
        if r["home_team"] == team:
            gs, gc = r["home_score"], r["away_score"]
        else:
            gs, gc = r["away_score"], r["home_score"]
        gs_list.append(gs)
        gc_list.append(gc)
        results.append("W" if gs > gc else ("D" if gs == gc else "L"))

    n = len(results)
    gs_arr = np.array(gs_list, dtype=float)
    gc_arr = np.array(gc_list, dtype=float)
    form_pts = np.array([3 if r == "W" else (1 if r == "D" else 0) for r in results], dtype=float)
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


def _h2h_at(df: pd.DataFrame, home: str, away: str, before_date) -> dict:
    mask = (
        ((df["home_team"] == home) & (df["away_team"] == away))
        | ((df["home_team"] == away) & (df["away_team"] == home))
    )
    h2h = df[mask & (df["date"] < before_date)]
    if h2h.empty:
        return {"h2h_home_win_rate": 0.45, "h2h_draw_rate": 0.27, "h2h_goal_diff": 0.0, "h2h_n": 0}

    hw = sum(1 for _, r in h2h.iterrows()
             if (r["home_team"] == home and r["home_score"] > r["away_score"])
             or (r["home_team"] == away and r["away_score"] > r["home_score"]))
    draws = sum(1 for _, r in h2h.iterrows() if r["home_score"] == r["away_score"])
    gd = [
        (r["home_score"] - r["away_score"]) if r["home_team"] == home
        else (r["away_score"] - r["home_score"])
        for _, r in h2h.iterrows()
    ]
    total = len(h2h)
    return {
        "h2h_home_win_rate": hw / total,
        "h2h_draw_rate":     draws / total,
        "h2h_goal_diff":     float(np.mean(gd)),
        "h2h_n":             total,
    }


def build_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df[df["date"] >= _MIN_DATE].copy()
    print(f"[Train] Processing {len(df):,} matches from {_MIN_DATE}…")

    rows, targets = [], []
    for idx, (_, m) in enumerate(df.iterrows()):
        if idx % 2000 == 0:
            print(f"  {idx:,} / {len(df):,}")

        h_stats = _team_stats_at(df, m["home_team"], m["date"])
        a_stats = _team_stats_at(df, m["away_team"], m["date"])
        if h_stats is None or a_stats is None:
            continue

        h2h = _h2h_at(df, m["home_team"], m["away_team"], m["date"])

        feat: dict = {}
        for k, v in h_stats.items():
            feat[f"home_{k}"] = v
        for k, v in a_stats.items():
            feat[f"away_{k}"] = v
        feat["win_rate_diff"]       = h_stats["win_rate"] - a_stats["win_rate"]
        feat["form_diff"]           = h_stats["weighted_form"] - a_stats["weighted_form"]
        feat["goal_diff_diff"]      = h_stats["avg_goal_diff"] - a_stats["avg_goal_diff"]
        feat["goals_scored_diff"]   = h_stats["avg_goals_scored"] - a_stats["avg_goals_scored"]
        feat["goals_conceded_diff"] = h_stats["avg_goals_conceded"] - a_stats["avg_goals_conceded"]
        feat.update(h2h)
        feat["neutral"]      = int(bool(m.get("neutral", False)))
        feat["wc_form_diff"] = 0.0  # 0 during training; populated at prediction time

        hs, as_ = m["home_score"], m["away_score"]
        target = "H" if hs > as_ else ("D" if hs == as_ else "A")

        rows.append({k: feat.get(k, 0.0) for k in FEATURE_COLS})
        targets.append(target)

    print(f"[Train] Built {len(rows):,} training samples")
    return pd.DataFrame(rows), pd.Series(targets)


def main():
    print("[Train] Downloading historical data…")
    resp = requests.get(_CSV_URL, timeout=90)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df["date"] = pd.to_datetime(df["date"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    print(f"[Train] Loaded {len(df):,} matches total")

    X, y = build_training_data(df)
    print(f"\n[Train] Class distribution:\n{y.value_counts().to_string()}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    print("\n[Train] Running 5-fold cross-validation…")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y_enc, cv=cv, scoring="accuracy")
    print(f"[Train] CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    print("[Train] Fitting final model…")
    model.fit(X, y_enc)
    save(model, list(X.columns))
    print("[Train] Done!")


if __name__ == "__main__":
    main()
