import pandas as pd
import streamlit as st

from src.db.client import get_client
from src.ml.features import build_match_features, match_result_label
from src.ml.model import is_trained, predict
from src.ui.styles import apply_styles, flag, form_strip, prob_bars, stat_grid, vs_banner

st.set_page_config(page_title="Match Predictor · WC 2026", page_icon="🎯", layout="wide")
apply_styles()


@st.cache_data(ttl=300)
def load_teams() -> pd.DataFrame:
    docs = get_client().collection("teams").stream()
    rows = [{"id": d.id, **d.to_dict()} for d in docs]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df.sort_values("name") if not df.empty else df


@st.cache_data(ttl=300)
def load_recent(team_id: str) -> list[dict]:
    db = get_client()
    home = [d.to_dict() for d in
            db.collection("team_recent_matches").where("home_team_id", "==", team_id).limit(15).stream()]
    away = [d.to_dict() for d in
            db.collection("team_recent_matches").where("away_team_id", "==", team_id).limit(15).stream()]
    records = []
    for d in home:
        gs, gc = d.get("home_goals", 0), d.get("away_goals", 0)
        records.append({"Date": d.get("match_date", ""), "Opponent": d.get("away_team_name", "?"),
                        "GS": gs, "GC": gc, "Result": match_result_label(gs, gc, "home"),
                        "Venue": "H", "Comp": d.get("competition", "")})
    for d in away:
        gs, gc = d.get("away_goals", 0), d.get("home_goals", 0)
        records.append({"Date": d.get("match_date", ""), "Opponent": d.get("home_team_name", "?"),
                        "GS": gs, "GC": gc,
                        "Result": match_result_label(d.get("home_goals", 0), d.get("away_goals", 0), "away"),
                        "Venue": "A", "Comp": d.get("competition", "")})
    return sorted(records, key=lambda x: x["Date"], reverse=True)[:15]


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<h1>🎯 Match Predictor</h1>
<p class="page-sub">Select two teams to get ML-powered win · draw · loss probabilities</p>
""", unsafe_allow_html=True)

if not is_trained():
    st.warning("Model not trained — run `python -m scripts.train_model`", icon="⚠️")

teams_df = load_teams()
if teams_df.empty:
    st.error("No teams — run `python -m scripts.seed_db` first.")
    st.stop()

team_names = teams_df["name"].tolist()

# ── Team selectors ─────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    home_name = st.selectbox("🏠 Home / Team 1", team_names, index=0)
with c2:
    away_name = st.selectbox("✈️ Away / Team 2", team_names, index=min(1, len(team_names) - 1))

neutral = st.checkbox("Neutral venue", value=True)

if home_name == away_name:
    st.warning("Select two different teams.")
    st.stop()

home_row = teams_df[teams_df["name"] == home_name].iloc[0]
away_row = teams_df[teams_df["name"] == away_name].iloc[0]
home_id  = str(home_row["id"])
away_id  = str(away_row["id"])
home_grp = str(home_row.get("group_name") or "")
away_grp = str(away_row.get("group_name") or "")

# ── VS banner ──────────────────────────────────────────────────────────────
st.markdown(vs_banner(home_name, home_grp, away_name, away_grp), unsafe_allow_html=True)

# ── Prediction ─────────────────────────────────────────────────────────────
features = build_match_features(home_id, away_id, neutral=neutral)
result   = predict(features)

if not result.get("model_available", True):
    st.info("Showing estimated probabilities — train the model for accurate predictions.")

# Outcome label + colour
outcome_styles = {
    "Home Win": ("#22c55e", f"{flag(home_name)} {home_name} Win"),
    "Draw":     ("#f59e0b", "Draw"),
    "Away Win": ("#3b82f6", f"{flag(away_name)} {away_name} Win"),
}
oc, olabel = outcome_styles.get(result["predicted"], ("#94a3b8", result["predicted"]))

st.markdown(f"""
<div style="text-align:center;margin:8px 0 16px">
  <span style="font-size:.7rem;color:#334155;text-transform:uppercase;letter-spacing:.12em">Prediction</span><br>
  <span style="font-size:1.8rem;font-weight:900;color:{oc}">{olabel}</span>
  <span style="font-size:.8rem;color:#475569;margin-left:8px">({result['confidence']:.1%} confidence)</span>
</div>
""", unsafe_allow_html=True)

# Probability bars
st.markdown(
    prob_bars(result["home_win"], result["draw"], result["away_win"], home_name, away_name),
    unsafe_allow_html=True,
)

st.divider()

# ── Stats comparison ───────────────────────────────────────────────────────
st.markdown('<div class="section-label">Team Stats Comparison — Last 15 Pre-WC Matches</div>', unsafe_allow_html=True)

stat_pairs = [
    ("Win Rate",          "home_win_rate",          "away_win_rate",          ".0%"),
    ("Avg Goals Scored",  "home_avg_goals_scored",  "away_avg_goals_scored",  ".2f"),
    ("Avg Goals Conceded","home_avg_goals_conceded", "away_avg_goals_conceded",".2f"),
    ("Goal Difference",   "home_avg_goal_diff",     "away_avg_goal_diff",     "+.2f"),
    ("Form Score",        "home_weighted_form",     "away_weighted_form",     ".2f"),
    ("Clean Sheets",      "home_clean_sheet_rate",  "away_clean_sheet_rate",  ".0%"),
]

rows_html = ""
for label, hk, ak, fmt in stat_pairs:
    hv = features.get(hk, 0)
    av = features.get(ak, 0)
    better_h = hv >= av
    hc = "#22c55e" if better_h else "#94a3b8"
    ac = "#22c55e" if not better_h else "#94a3b8"
    rows_html += f"""
<tr>
  <td style="text-align:right;padding:10px 16px;font-size:.9rem;font-weight:700;color:{hc}">{hv:{fmt}}</td>
  <td style="text-align:center;padding:10px 12px;font-size:.72rem;color:#334155;
             text-transform:uppercase;letter-spacing:.08em;white-space:nowrap">{label}</td>
  <td style="text-align:left;padding:10px 16px;font-size:.9rem;font-weight:700;color:{ac}">{av:{fmt}}</td>
</tr>"""

st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f0f24,#141432);border:1px solid #1e1e3e;
            border-radius:14px;overflow:hidden;margin:8px 0">
  <div style="display:flex;padding:12px 16px;background:#0a0a1e;border-bottom:1px solid #1a1a3a">
    <div style="flex:1;text-align:right;font-size:.8rem;font-weight:800;color:#f8fafc">
      {flag(home_name)} {home_name}
    </div>
    <div style="width:180px"></div>
    <div style="flex:1;text-align:left;font-size:.8rem;font-weight:800;color:#f8fafc">
      {flag(away_name)} {away_name}
    </div>
  </div>
  <table style="width:100%;border-collapse:collapse">{rows_html}</table>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── H2H ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Head to Head</div>', unsafe_allow_html=True)

h2h_n = int(features.get("h2h_n", 0))
if h2h_n == 0:
    st.info("No historical head-to-head data found.")
else:
    hw = round(features.get("h2h_home_win_rate", 0) * h2h_n)
    dr = round(features.get("h2h_draw_rate", 0) * h2h_n)
    aw = h2h_n - hw - dr
    hf, af = flag(home_name), flag(away_name)

    st.markdown(f"""
<div style="display:flex;gap:12px;margin:8px 0">
  <div class="stat-card" style="flex:1;text-align:center">
    <div class="sc-val" style="color:#22c55e">{hw}</div>
    <div class="sc-lbl">{hf} {home_name} Wins</div>
  </div>
  <div class="stat-card" style="flex:1;text-align:center">
    <div class="sc-val" style="color:#f59e0b">{dr}</div>
    <div class="sc-lbl">Draws</div>
  </div>
  <div class="stat-card" style="flex:1;text-align:center">
    <div class="sc-val" style="color:#3b82f6">{aw}</div>
    <div class="sc-lbl">{af} {away_name} Wins</div>
  </div>
  <div class="stat-card" style="flex:1;text-align:center">
    <div class="sc-val">{h2h_n}</div>
    <div class="sc-lbl">Total Meetings</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Recent form ─────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Recent Form</div>', unsafe_allow_html=True)

fc1, fc2 = st.columns(2)
for col, tid, tname in [(fc1, home_id, home_name), (fc2, away_id, away_name)]:
    with col:
        st.markdown(f"**{flag(tname)} {tname}**")
        records = load_recent(tid)
        if not records:
            st.info("No data.")
        else:
            results = [r["Result"] for r in records]
            st.markdown(form_strip(results), unsafe_allow_html=True)
            df = pd.DataFrame(records)[["Date", "Opponent", "Venue", "GS", "GC", "Comp"]]
            df["Date"] = df["Date"].str[:10]
            st.dataframe(df, use_container_width=True, hide_index=True)
