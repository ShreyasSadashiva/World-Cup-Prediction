"""
FIFA World Cup 2026 Prediction App — Dashboard
"""

import pandas as pd
import streamlit as st

from src.db.client import get_client
from src.ml.model import is_trained

st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=300)
def load_teams() -> pd.DataFrame:
    docs = get_client().collection("teams").stream()
    rows = [{"id": d.id, **d.to_dict()} for d in docs]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data(ttl=120)
def load_wc_fixtures() -> pd.DataFrame:
    docs = get_client().collection("wc_fixtures").stream()
    rows = [{"id": d.id, **d.to_dict()} for d in docs]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce", utc=True)
    return df.sort_values("match_date")


def compute_standings(fixtures: pd.DataFrame) -> pd.DataFrame:
    finished = fixtures[fixtures["status"] == "FINISHED"].copy()
    if finished.empty:
        return pd.DataFrame()

    stats: dict[str, dict] = {}

    def _init(name, group):
        if name not in stats:
            stats[name] = {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0, "Group": group or ""}

    for _, row in finished.iterrows():
        home = row.get("home_team_name", "?")
        away = row.get("away_team_name", "?")
        group = row.get("group_name", "")
        hg = int(row.get("home_goals") or 0)
        ag = int(row.get("away_goals") or 0)

        _init(home, group); _init(away, group)
        stats[home]["P"] += 1; stats[away]["P"] += 1
        stats[home]["GF"] += hg; stats[home]["GA"] += ag
        stats[away]["GF"] += ag; stats[away]["GA"] += hg

        if hg > ag:
            stats[home]["W"] += 1; stats[home]["Pts"] += 3
            stats[away]["L"] += 1
        elif hg == ag:
            stats[home]["D"] += 1; stats[home]["Pts"] += 1
            stats[away]["D"] += 1; stats[away]["Pts"] += 1
        else:
            stats[away]["W"] += 1; stats[away]["Pts"] += 3
            stats[home]["L"] += 1

    df = pd.DataFrame(stats).T.reset_index().rename(columns={"index": "Team"})
    df["GD"] = df["GF"].astype(int) - df["GA"].astype(int)
    return df.sort_values(["Group", "Pts", "GD"], ascending=[True, False, False])


# ── page ─────────────────────────────────────────────────────────────────────
st.title("⚽ FIFA World Cup 2026 — Prediction Dashboard")
st.caption("Powered by XGBoost · Data: football-data.org + martj42 historical dataset")

if not is_trained():
    st.warning("**Model not trained yet.** Run `python -m scripts.train_model` to train the predictor.", icon="⚠️")

teams_df = load_teams()
if teams_df.empty:
    st.error("**Database is empty.** Run `python -m scripts.seed_db` to populate WC data.", icon="🔴")
    st.stop()

fixtures_df = load_wc_fixtures()

col_fixtures, col_standings = st.columns([1, 1], gap="large")

with col_fixtures:
    st.subheader("📅 Upcoming Matches")
    upcoming = fixtures_df[fixtures_df["status"] == "SCHEDULED"].head(8) if not fixtures_df.empty else pd.DataFrame()

    if upcoming.empty:
        st.info("No upcoming matches scheduled.")
    else:
        for _, row in upcoming.iterrows():
            home = row.get("home_team_name", "TBD")
            away = row.get("away_team_name", "TBD")
            date_str = row["match_date"].strftime("%b %d, %H:%M UTC") if pd.notna(row.get("match_date")) else "TBD"
            group = row.get("group_name") or ""
            stage = f"Group {group}" if group else (row.get("stage", "")).replace("_", " ").title()
            st.markdown(f"**{home}** vs **{away}**  \n<small>{date_str} · {stage}</small>", unsafe_allow_html=True)
            st.divider()

    st.subheader("✅ Recent Results")
    finished = fixtures_df[fixtures_df["status"] == "FINISHED"] if not fixtures_df.empty else pd.DataFrame()

    if finished.empty:
        st.info("No finished matches yet.")
    else:
        for _, row in finished.tail(6).iloc[::-1].iterrows():
            home = row.get("home_team_name", "?")
            away = row.get("away_team_name", "?")
            hg = int(row.get("home_goals") or 0)
            ag = int(row.get("away_goals") or 0)
            date_str = row["match_date"].strftime("%b %d") if pd.notna(row.get("match_date")) else ""
            st.markdown(f"**{home} {hg} – {ag} {away}** · <small>{date_str}</small>", unsafe_allow_html=True)

with col_standings:
    st.subheader("🏆 Group Standings")
    standings = compute_standings(fixtures_df)

    if standings.empty:
        st.info("Standings will appear once matches have been played.")
    else:
        groups = sorted(standings["Group"].dropna().unique())
        if groups:
            tabs = st.tabs([f"Group {g}" for g in groups])
            for tab, g in zip(tabs, groups):
                with tab:
                    grp = standings[standings["Group"] == g][
                        ["Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"]
                    ].reset_index(drop=True)
                    grp.index += 1
                    st.dataframe(grp, use_container_width=True, hide_index=False)

st.divider()
c1, c2, c3, c4 = st.columns(4)
total = len(fixtures_df) if not fixtures_df.empty else 0
played = len(fixtures_df[fixtures_df["status"] == "FINISHED"]) if not fixtures_df.empty else 0
goals = int(
    (fixtures_df[fixtures_df["status"] == "FINISHED"]["home_goals"].fillna(0) +
     fixtures_df[fixtures_df["status"] == "FINISHED"]["away_goals"].fillna(0)).sum()
) if not fixtures_df.empty else 0

c1.metric("Total Fixtures", total)
c2.metric("Played", played)
c3.metric("Total Goals", goals)
c4.metric("Teams", len(teams_df))
