import pandas as pd
import streamlit as st

from src.db.client import get_client
from src.ml.model import is_trained
from src.ui.styles import apply_styles, top_nav, flag, match_card, standings_table

st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_styles()
top_nav()


@st.cache_data(ttl=300)
def load_teams() -> pd.DataFrame:
    docs = get_client().collection("teams").stream()
    rows = [{"id": d.id, **d.to_dict()} for d in docs]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data(ttl=60)
def load_fixtures() -> pd.DataFrame:
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
    stats: dict = {}

    def _init(name, group):
        if name not in stats:
            stats[name] = {"name": name, "P": 0, "W": 0, "D": 0, "L": 0,
                           "GF": 0, "GA": 0, "GD": 0, "Pts": 0, "Group": group or ""}

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
            stats[home]["W"] += 1; stats[home]["Pts"] += 3; stats[away]["L"] += 1
        elif hg == ag:
            stats[home]["D"] += 1; stats[home]["Pts"] += 1
            stats[away]["D"] += 1; stats[away]["Pts"] += 1
        else:
            stats[away]["W"] += 1; stats[away]["Pts"] += 3; stats[home]["L"] += 1

    df = pd.DataFrame(stats.values())
    df["GD"] = df["GF"] - df["GA"]
    return df.sort_values(["Group", "Pts", "GD"], ascending=[True, False, False])


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:8px">
  <h1 style="margin:0">⚽ WC 2026 Predictor</h1>
  <p class="page-sub">Live predictions · Group standings · Team form · Player stats</p>
</div>
""", unsafe_allow_html=True)

if not is_trained():
    st.warning("Model not trained — run `python -m scripts.train_model`", icon="⚠️")

teams_df = load_teams()
if teams_df.empty:
    st.error("Database empty — run `python -m scripts.seed_db`", icon="🔴")
    st.stop()

fixtures_df = load_fixtures()
finished_df = fixtures_df[fixtures_df["status"] == "FINISHED"] if not fixtures_df.empty else pd.DataFrame()
upcoming_df = fixtures_df[fixtures_df["status"] == "SCHEDULED"] if not fixtures_df.empty else pd.DataFrame()

# ── Top stats bar ──────────────────────────────────────────────────────────
total = len(fixtures_df) if not fixtures_df.empty else 0
played = len(finished_df)
goals = int((finished_df["home_goals"].fillna(0) + finished_df["away_goals"].fillna(0)).sum()) if not finished_df.empty else 0
avg_g = f"{goals/played:.1f}" if played else "—"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Fixtures", total)
c2.metric("Matches Played", played)
c3.metric("Goals Scored", goals)
c4.metric("Avg Goals / Match", avg_g)

st.divider()

# ── Main layout ────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-label">Upcoming Matches</div>', unsafe_allow_html=True)
    if upcoming_df.empty:
        st.info("No upcoming matches.")
    else:
        for _, row in upcoming_df.head(6).iterrows():
            home = row.get("home_team_name", "TBD")
            away = row.get("away_team_name", "TBD")
            dt = row["match_date"].strftime("%a %d %b · %H:%M UTC") if pd.notna(row.get("match_date")) else "TBD"
            g = row.get("group_name") or ""
            st.markdown(
                match_card(home, away, date=dt, group=g, status="SCHEDULED"),
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-label">Recent Results</div>', unsafe_allow_html=True)
    if finished_df.empty:
        st.info("No results yet.")
    else:
        for _, row in finished_df.tail(5).iloc[::-1].iterrows():
            home = row.get("home_team_name", "?")
            away = row.get("away_team_name", "?")
            hg = int(row.get("home_goals") or 0)
            ag = int(row.get("away_goals") or 0)
            dt = row["match_date"].strftime("%a %d %b") if pd.notna(row.get("match_date")) else ""
            g = row.get("group_name") or ""
            st.markdown(
                match_card(home, away, hg, ag, date=dt, group=g, status="FINISHED"),
                unsafe_allow_html=True,
            )

with col_right:
    st.markdown('<div class="section-label">Group Standings</div>', unsafe_allow_html=True)
    standings = compute_standings(fixtures_df)

    if standings.empty:
        st.info("Standings will appear once matches are played.")
    else:
        groups = sorted(standings["Group"].dropna().unique())
        if groups:
            tabs = st.tabs([f"Group {g}" for g in groups])
            for tab, g in zip(tabs, groups):
                with tab:
                    grp = standings[standings["Group"] == g].reset_index(drop=True)
                    rows = []
                    for i, r in grp.iterrows():
                        rows.append({
                            "name": r["name"], "P": int(r["P"]), "W": int(r["W"]),
                            "D": int(r["D"]), "L": int(r["L"]), "GF": int(r["GF"]),
                            "GA": int(r["GA"]), "GD": int(r["GD"]), "Pts": int(r["Pts"]),
                        })
                    st.markdown(standings_table(rows), unsafe_allow_html=True)
                    st.markdown(
                        '<p style="font-size:.65rem;color:#1e3a1e;margin-top:6px">'
                        '<span style="color:#22c55e">■</span> Qualify &nbsp; '
                        '<span style="color:#f59e0b">■</span> May qualify (best 3rd)</p>',
                        unsafe_allow_html=True,
                    )
