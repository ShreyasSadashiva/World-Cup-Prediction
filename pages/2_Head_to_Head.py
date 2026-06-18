"""
Head-to-Head — matrix view + pair drill-down.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.db.client import get_client
from src.ml.features import match_result_label

st.set_page_config(page_title="Head to Head · WC 2026", page_icon="🤝", layout="wide")
st.title("🤝 Head to Head")


@st.cache_data(ttl=300)
def load_teams() -> pd.DataFrame:
    docs = get_client().collection("teams").stream()
    rows = [{"id": d.id, **d.to_dict()} for d in docs]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df.sort_values("name") if not df.empty else df


@st.cache_data(ttl=300)
def load_all_h2h_raw() -> list[dict]:
    """Load all match rows (recent + finished WC) for matrix computation."""
    db = get_client()
    recent = [d.to_dict() for d in db.collection("team_recent_matches").stream()]
    wc = [d.to_dict() for d in db.collection("wc_fixtures").stream()
          if d.to_dict().get("status") == "FINISHED"]
    return recent + wc


@st.cache_data(ttl=300)
def load_pair_history(slug1: str, slug2: str) -> pd.DataFrame:
    db = get_client()
    records = []

    for coll, extra_key in [("team_recent_matches", "competition"), ("wc_fixtures", "stage")]:
        for hid, aid in [(slug1, slug2), (slug2, slug1)]:
            docs = (
                db.collection(coll)
                .where("home_team_id", "==", hid)
                .where("away_team_id", "==", aid)
                .stream()
            )
            for doc in docs:
                d = doc.to_dict()
                if coll == "wc_fixtures" and d.get("home_goals") is None:
                    continue
                records.append({
                    "Date":        d.get("match_date", ""),
                    "home_slug":   hid,
                    "Home":        d.get("home_team_name", hid),
                    "HG":          d.get("home_goals", 0),
                    "AG":          d.get("away_goals", 0),
                    "Away":        d.get("away_team_name", aid),
                    "Competition": d.get(extra_key, ""),
                })

    return pd.DataFrame(records).sort_values("Date", ascending=False) if records else pd.DataFrame()


teams_df = load_teams()
if teams_df.empty:
    st.error("No teams in DB — run `python -m scripts.seed_db` first.")
    st.stop()

team_names = teams_df["name"].tolist()
name_to_slug = dict(zip(teams_df["name"], teams_df["id"]))

view = st.radio("View", ["Pair Detail", "Matrix Overview"], horizontal=True)

# ── PAIR DETAIL ───────────────────────────────────────────────────────────────
if view == "Pair Detail":
    c1, c2 = st.columns(2)
    with c1:
        t1 = st.selectbox("Team 1", team_names, index=0)
    with c2:
        t2 = st.selectbox("Team 2", team_names, index=min(1, len(team_names) - 1))

    if t1 == t2:
        st.warning("Select two different teams.")
        st.stop()

    slug1, slug2 = name_to_slug[t1], name_to_slug[t2]
    history = load_pair_history(slug1, slug2)

    if history.empty:
        st.info(f"No historical matches found between **{t1}** and **{t2}**.")
    else:
        t1_wins = draws = t2_wins = 0
        for _, row in history.iterrows():
            if row["home_slug"] == slug1:
                res = match_result_label(int(row["HG"]), int(row["AG"]), "home")
            else:
                res = match_result_label(int(row["HG"]), int(row["AG"]), "away")
            if res == "W": t1_wins += 1
            elif res == "D": draws += 1
            else: t2_wins += 1

        total = t1_wins + draws + t2_wins
        total_goals = int(history["HG"].sum() + history["AG"].sum())

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Meetings", total)
        mc2.metric(f"{t1} Wins", t1_wins)
        mc3.metric("Draws", draws)
        mc4.metric(f"{t2} Wins", t2_wins)
        mc5.metric("Total Goals", total_goals)

        fig = go.Figure(go.Pie(
            labels=[f"{t1} Wins", "Draws", f"{t2} Wins"],
            values=[t1_wins, draws, t2_wins],
            marker_colors=["#198754", "#6c757d", "#0d6efd"], hole=0.5,
        ))
        fig.update_layout(height=320, template="plotly_dark", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Match History")
        display = history[["Date", "Home", "HG", "AG", "Away", "Competition"]].copy()
        display["Date"] = display["Date"].str[:10]
        display.columns = ["Date", "Home", "HG", "AG", "Away", "Competition"]
        st.dataframe(display, use_container_width=True, hide_index=True)

# ── MATRIX OVERVIEW ───────────────────────────────────────────────────────────
else:
    st.subheader("Win-rate matrix (row = home team, col = away team)")
    all_matches = load_all_h2h_raw()

    if not all_matches:
        st.info("No match data yet.")
        st.stop()

    slugs = teams_df["id"].tolist()
    names = teams_df["name"].tolist()
    n = len(slugs)
    matrix = np.full((n, n), np.nan)
    slug_idx = {s: i for i, s in enumerate(slugs)}

    for m in all_matches:
        hi = slug_idx.get(m.get("home_team_id"))
        ai = slug_idx.get(m.get("away_team_id"))
        if hi is None or ai is None or hi == ai:
            continue
        hg = m.get("home_goals") or 0
        ag = m.get("away_goals") or 0
        if np.isnan(matrix[hi][ai]):
            matrix[hi][ai] = 0.0
        matrix[hi][ai] = (matrix[hi][ai] + (1 if hg > ag else 0)) / 1

    # Recalculate properly
    counts = np.zeros((n, n))
    wins = np.zeros((n, n))
    for m in all_matches:
        hi = slug_idx.get(m.get("home_team_id"))
        ai = slug_idx.get(m.get("away_team_id"))
        if hi is None or ai is None or hi == ai:
            continue
        hg = m.get("home_goals") or 0
        ag = m.get("away_goals") or 0
        counts[hi][ai] += 1
        if hg > ag:
            wins[hi][ai] += 1

    with np.errstate(invalid="ignore"):
        matrix = np.where(counts > 0, wins / counts, np.nan)

    has_data = ~np.all(np.isnan(matrix), axis=1) & ~np.all(np.isnan(matrix), axis=0)
    mat_f = matrix[has_data][:, has_data]
    names_f = [names[i] for i in range(n) if has_data[i]]

    if mat_f.size == 0:
        st.info("No H2H data available yet.")
    else:
        text_mat = [[f"{v:.0%}" if not np.isnan(v) else "" for v in row] for row in mat_f]
        fig = go.Figure(go.Heatmap(
            z=mat_f, x=names_f, y=names_f,
            text=text_mat, texttemplate="%{text}",
            colorscale="RdYlGn", zmin=0, zmax=1,
            colorbar=dict(title="Win Rate"),
        ))
        fig.update_layout(
            title="H2H Home-Team Win Rate Matrix",
            height=max(500, len(names_f) * 22), template="plotly_dark",
            xaxis=dict(tickangle=-45), margin=dict(l=100, b=120),
        )
        st.plotly_chart(fig, use_container_width=True)
