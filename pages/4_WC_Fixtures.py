import pandas as pd
import streamlit as st

from src.db.client import get_client
from src.ui.styles import apply_styles, top_nav, match_card

st.set_page_config(page_title="WC Fixtures · WC 2026", page_icon="🗓️", layout="wide")
apply_styles()
top_nav()


@st.cache_data(ttl=60)
def load_fixtures() -> pd.DataFrame:
    docs = get_client().collection("wc_fixtures").stream()
    rows = [{"id": d.id, **d.to_dict()} for d in docs]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["match_date"] = pd.to_datetime(df.get("match_date"), errors="coerce", utc=True)
    return df.sort_values("match_date")


def _date(val) -> str:
    try:
        return pd.to_datetime(val, utc=True).strftime("%a %d %b · %H:%M UTC")
    except Exception:
        return str(val)[:16] if val else "TBD"


# ── Sidebar sync ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔄 Data Sync")
    if st.button("Update WC Results", type="primary", use_container_width=True):
        with st.spinner("Fetching from football-data.org…"):
            try:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from scripts.update_wc import run
                f_upd, p_upd = run()
                st.success(f"✅ {f_upd} fixtures · {p_upd} player records updated")
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Update failed: {exc}")
    st.caption("Pulls latest finished match scores and player stats from the API.")

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<h1>🗓️ WC 2026 Fixtures</h1>
<p class="page-sub">All 104 matches · Group stage &amp; Knockout · Click sidebar to sync latest results</p>
""", unsafe_allow_html=True)

fixtures_df = load_fixtures()
if fixtures_df.empty:
    st.error("No fixtures — run `python -m scripts.seed_db` first.")
    st.stop()

# ── Status filter chips ─────────────────────────────────────────────────────
status_filter = st.radio(
    "Filter", ["All", "Scheduled", "Finished", "Live"],
    horizontal=True, label_visibility="collapsed",
)
status_map = {"Scheduled": "SCHEDULED", "Finished": "FINISHED", "Live": "IN_PLAY"}
if status_filter != "All":
    fixtures_df = fixtures_df[fixtures_df["status"] == status_map[status_filter]]

# ── Counts ──────────────────────────────────────────────────────────────────
total = len(fixtures_df)
done  = (fixtures_df["status"] == "FINISHED").sum()
sched = (fixtures_df["status"] == "SCHEDULED").sum()

c1, c2, c3 = st.columns(3)
c1.metric("Total",    total)
c2.metric("Finished", int(done))
c3.metric("Scheduled",int(sched))

st.divider()

# ── Group stage tabs ────────────────────────────────────────────────────────
group_df    = fixtures_df[fixtures_df["group_name"].notna()].copy() if "group_name" in fixtures_df.columns else pd.DataFrame()
knockout_df = fixtures_df[fixtures_df["group_name"].isna()].copy()  if "group_name" in fixtures_df.columns else fixtures_df

groups = sorted(group_df["group_name"].dropna().unique()) if not group_df.empty else []

if groups:
    extra = ["🏆 Knockout"] if not knockout_df.empty else []
    tabs  = st.tabs([f"Group {g}" for g in groups] + extra)

    for tab, g in zip(tabs, groups):
        with tab:
            grp = group_df[group_df["group_name"] == g]
            col_a, col_b = st.columns(2)
            for i, (_, row) in enumerate(grp.iterrows()):
                home   = row.get("home_team_name", "TBD")
                away   = row.get("away_team_name", "TBD")
                status = row.get("status", "SCHEDULED")
                hg     = int(row.get("home_goals") or 0) if status == "FINISHED" else None
                ag     = int(row.get("away_goals") or 0) if status == "FINISHED" else None
                dt     = _date(row.get("match_date"))
                venue  = row.get("venue") or ""
                col = col_a if i % 2 == 0 else col_b
                with col:
                    st.markdown(
                        match_card(home, away, hg, ag, date=dt, group=g, venue=venue, status=status),
                        unsafe_allow_html=True,
                    )

    if not knockout_df.empty and extra:
        with tabs[-1]:
            stages = knockout_df["stage"].dropna().unique() if "stage" in knockout_df.columns else []
            for stage in stages:
                stage_df = knockout_df[knockout_df["stage"] == stage]
                st.markdown(
                    f'<div class="section-label">{stage.replace("_"," ").title()}</div>',
                    unsafe_allow_html=True,
                )
                col_a, col_b = st.columns(2)
                for i, (_, row) in enumerate(stage_df.iterrows()):
                    home   = row.get("home_team_name", "TBD")
                    away   = row.get("away_team_name", "TBD")
                    status = row.get("status", "SCHEDULED")
                    hg     = int(row.get("home_goals") or 0) if status == "FINISHED" else None
                    ag     = int(row.get("away_goals") or 0) if status == "FINISHED" else None
                    dt     = _date(row.get("match_date"))
                    col = col_a if i % 2 == 0 else col_b
                    with col:
                        st.markdown(
                            match_card(home, away, hg, ag, date=dt, stage=stage, status=status),
                            unsafe_allow_html=True,
                        )
else:
    col_a, col_b = st.columns(2)
    for i, (_, row) in enumerate(fixtures_df.iterrows()):
        home   = row.get("home_team_name", "TBD")
        away   = row.get("away_team_name", "TBD")
        status = row.get("status", "SCHEDULED")
        hg     = int(row.get("home_goals") or 0) if status == "FINISHED" else None
        ag     = int(row.get("away_goals") or 0) if status == "FINISHED" else None
        dt     = _date(row.get("match_date"))
        col = col_a if i % 2 == 0 else col_b
        with col:
            st.markdown(match_card(home, away, hg, ag, date=dt, status=status), unsafe_allow_html=True)
