"""
WC Fixtures — all WC 2026 matches grouped by group/stage, with manual sync button.
"""

import pandas as pd
import streamlit as st

from src.db.client import get_client

st.set_page_config(page_title="WC Fixtures · WC 2026", page_icon="🗓️", layout="wide")
st.title("🗓️ WC 2026 Fixtures")


@st.cache_data(ttl=120)
def load_fixtures() -> pd.DataFrame:
    docs = get_client().collection("wc_fixtures").stream()
    rows = [{"id": d.id, **d.to_dict()} for d in docs]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["match_date"] = pd.to_datetime(df.get("match_date"), errors="coerce", utc=True)
    return df.sort_values("match_date")


def _score(row) -> str:
    if row.get("status") == "FINISHED":
        return f"**{int(row.get('home_goals') or 0)} – {int(row.get('away_goals') or 0)}**"
    if row.get("status") == "IN_PLAY":
        return "🔴 LIVE"
    return "vs"


def _date(val) -> str:
    try:
        return pd.to_datetime(val, utc=True).strftime("%a %d %b · %H:%M UTC")
    except Exception:
        return str(val)[:16] if val else "TBD"


# ── sidebar sync ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("🔄 Data Sync")
    if st.button("Update WC Results", type="primary", use_container_width=True):
        with st.spinner("Fetching latest results from football-data.org…"):
            try:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from scripts.update_wc import run
                f_upd, p_upd = run()
                st.success(f"Updated {f_upd} fixtures · {p_upd} player records")
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Update failed: {exc}")
    st.caption("Pulls latest finished match scores and player stats from the API.")

fixtures_df = load_fixtures()

if fixtures_df.empty:
    st.error("No fixtures in DB — run `python -m scripts.seed_db` first.")
    st.stop()

# ── status filter ─────────────────────────────────────────────────────────────
status_filter = st.radio("Filter", ["All", "Scheduled", "Finished", "Live"], horizontal=True)
status_map = {"Scheduled": "SCHEDULED", "Finished": "FINISHED", "Live": "IN_PLAY"}
if status_filter != "All":
    fixtures_df = fixtures_df[fixtures_df["status"] == status_map[status_filter]]

group_matches = fixtures_df[fixtures_df["group_name"].notna()].copy() if "group_name" in fixtures_df.columns else pd.DataFrame()
knockout_matches = fixtures_df[fixtures_df["group_name"].isna()].copy() if "group_name" in fixtures_df.columns else fixtures_df

groups = sorted(group_matches["group_name"].dropna().unique()) if not group_matches.empty else []

if groups:
    extra_tabs = ["🏆 Knockout"] if not knockout_matches.empty else []
    tabs = st.tabs([f"Group {g}" for g in groups] + extra_tabs)

    for i, g in enumerate(groups):
        with tabs[i]:
            grp = group_matches[group_matches["group_name"] == g]
            for _, row in grp.iterrows():
                home = row.get("home_team_name", "TBD")
                away = row.get("away_team_name", "TBD")
                col1, col2, col3 = st.columns([3, 2, 3])
                with col1:
                    st.markdown(f"### {home}")
                with col2:
                    st.markdown(f"<div style='text-align:center;padding-top:8px'>{_score(row)}</div>",
                                unsafe_allow_html=True)
                with col3:
                    st.markdown(f"### {away}")
                st.caption(f"{_date(row.get('match_date'))} · {row.get('venue', '')}")
                st.divider()

    if not knockout_matches.empty and extra_tabs:
        with tabs[-1]:
            for stage in knockout_matches["stage"].dropna().unique():
                st.subheader(stage.replace("_", " ").title())
                for _, row in knockout_matches[knockout_matches["stage"] == stage].iterrows():
                    home = row.get("home_team_name", "TBD")
                    away = row.get("away_team_name", "TBD")
                    col1, col2, col3 = st.columns([3, 2, 3])
                    with col1:
                        st.markdown(f"**{home}**")
                    with col2:
                        st.markdown(f"<div style='text-align:center'>{_score(row)}</div>",
                                    unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"**{away}**")
                    st.caption(_date(row.get("match_date")))
                    st.divider()
else:
    for _, row in fixtures_df.iterrows():
        home = row.get("home_team_name", "TBD")
        away = row.get("away_team_name", "TBD")
        st.markdown(f"**{home}** {_score(row)} **{away}** · {_date(row.get('match_date'))}")
