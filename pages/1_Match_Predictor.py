"""
Match Predictor — select two teams and get ML win/draw/loss probabilities.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.db.client import get_client
from src.ml.features import build_match_features, match_result_label
from src.ml.model import is_trained, predict

st.set_page_config(page_title="Match Predictor · WC 2026", page_icon="🎯", layout="wide")
st.title("🎯 Match Predictor")
st.caption("Select any two WC 2026 teams to get an ML-powered prediction.")

if not is_trained():
    st.warning("Model not trained yet — run `python -m scripts.train_model`.", icon="⚠️")


@st.cache_data(ttl=300)
def load_teams() -> pd.DataFrame:
    docs = get_client().collection("teams").stream()
    rows = [{"id": d.id, **d.to_dict()} for d in docs]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df.sort_values("name") if not df.empty else df


@st.cache_data(ttl=300)
def load_recent_matches(team_id: str) -> pd.DataFrame:
    db = get_client()
    home_docs = db.collection("team_recent_matches").where("home_team_id", "==", team_id).limit(15).stream()
    away_docs = db.collection("team_recent_matches").where("away_team_id", "==", team_id).limit(15).stream()

    records = []
    for doc in home_docs:
        d = doc.to_dict()
        gs, gc = d.get("home_goals", 0), d.get("away_goals", 0)
        records.append({
            "Date": d.get("match_date", ""), "Opponent": d.get("away_team_name", "?"),
            "GS": gs, "GC": gc, "Result": match_result_label(gs, gc, "home"),
            "Venue": "H", "Competition": d.get("competition", ""),
        })
    for doc in away_docs:
        d = doc.to_dict()
        gs, gc = d.get("away_goals", 0), d.get("home_goals", 0)
        records.append({
            "Date": d.get("match_date", ""), "Opponent": d.get("home_team_name", "?"),
            "GS": gs, "GC": gc, "Result": match_result_label(d.get("home_goals", 0), d.get("away_goals", 0), "away"),
            "Venue": "A", "Competition": d.get("competition", ""),
        })

    return pd.DataFrame(records).sort_values("Date", ascending=False).head(15) if records else pd.DataFrame()


teams_df = load_teams()
if teams_df.empty:
    st.error("No teams in DB — run `python -m scripts.seed_db` first.")
    st.stop()

team_names = teams_df["name"].tolist()

col1, col_mid, col2 = st.columns([5, 1, 5])
with col1:
    home_name = st.selectbox("🏠 Home Team", team_names, index=0, key="home")
with col_mid:
    st.markdown("<div style='text-align:center;padding-top:32px;font-size:1.4em;font-weight:bold'>vs</div>",
                unsafe_allow_html=True)
with col2:
    away_name = st.selectbox("✈️ Away Team", team_names, index=min(1, len(team_names) - 1), key="away")

neutral = st.checkbox("Neutral venue", value=False)

home_id = str(teams_df[teams_df["name"] == home_name]["id"].values[0])
away_id = str(teams_df[teams_df["name"] == away_name]["id"].values[0])

st.divider()

if home_name == away_name:
    st.warning("Please select two different teams.")
    st.stop()

features = build_match_features(home_id, away_id, neutral=neutral)
result = predict(features)

if not result.get("model_available", True):
    st.info("Showing estimated probabilities — train the model for real predictions.")

fig = go.Figure(go.Bar(
    x=[result["home_win"], result["draw"], result["away_win"]],
    y=[f"🏠 {home_name}", "Draw", f"✈️ {away_name}"],
    orientation="h",
    marker_color=["#198754", "#6c757d", "#0d6efd"],
    text=[f"{v:.1%}" for v in [result["home_win"], result["draw"], result["away_win"]]],
    textposition="outside",
))
fig.update_layout(
    title=f"Win Probability — {home_name} vs {away_name}",
    xaxis=dict(tickformat=".0%", range=[0, 1]),
    height=260, margin=dict(l=10, r=80, t=40, b=10), template="plotly_dark",
)
st.plotly_chart(fig, use_container_width=True)

outcome_colors = {"Home Win": "green", "Draw": "orange", "Away Win": "red"}
color = outcome_colors.get(result["predicted"], "grey")
st.markdown(
    f"**Prediction:** <span style='color:{color};font-size:1.3em;font-weight:bold'>{result['predicted']}</span> "
    f"<small>(confidence: {result['confidence']:.1%})</small>",
    unsafe_allow_html=True,
)

st.divider()
st.subheader("📊 Team Stats Comparison (last 15 pre-WC matches)")

stats_labels = {
    "Win Rate":          ("home_win_rate",          "away_win_rate"),
    "Avg Goals Scored":  ("home_avg_goals_scored",  "away_avg_goals_scored"),
    "Avg Goals Conced.": ("home_avg_goals_conceded","away_avg_goals_conceded"),
    "Avg Goal Diff":     ("home_avg_goal_diff",     "away_avg_goal_diff"),
    "Form Score":        ("home_weighted_form",     "away_weighted_form"),
    "Clean Sheet Rate":  ("home_clean_sheet_rate",  "away_clean_sheet_rate"),
}

rows = []
for label, (hk, ak) in stats_labels.items():
    rows.append({"Stat": label, home_name: f"{features.get(hk, 0):.2f}", away_name: f"{features.get(ak, 0):.2f}"})

st.dataframe(pd.DataFrame(rows).set_index("Stat"), use_container_width=True)

st.subheader(f"🤝 Head to Head · {home_name} vs {away_name}")
h2h_n = int(features.get("h2h_n", 0))
if h2h_n == 0:
    st.info("No historical head-to-head matches found.")
else:
    hw = features.get("h2h_home_win_rate", 0) * h2h_n
    dr = features.get("h2h_draw_rate", 0) * h2h_n
    aw = h2h_n - hw - dr

    fig2 = go.Figure(go.Pie(
        labels=[f"{home_name} Wins", "Draws", f"{away_name} Wins"],
        values=[hw, dr, aw],
        marker_colors=["#198754", "#6c757d", "#0d6efd"], hole=0.45,
    ))
    fig2.update_layout(height=300, template="plotly_dark", margin=dict(t=20))
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.metric(f"{home_name} Wins", int(round(hw)))
        st.metric("Draws", int(round(dr)))
        st.metric(f"{away_name} Wins", int(round(aw)))
        st.metric("Total Meetings", h2h_n)
        st.metric("Avg Goal Diff (home perspective)", f"{features.get('h2h_goal_diff', 0):.2f}")

st.divider()
st.subheader("📋 Recent Form")
fc1, fc2 = st.columns(2)

def _badge(r: str) -> str:
    c = {"W": "#198754", "D": "#6c757d", "L": "#dc3545"}.get(r, "#6c757d")
    return f'<span style="background:{c};color:white;padding:1px 6px;border-radius:3px;font-weight:bold;font-size:0.8em">{r}</span>'

for col, tid, tname in [(fc1, home_id, home_name), (fc2, away_id, away_name)]:
    with col:
        st.markdown(f"**{tname}**")
        form_df = load_recent_matches(tid)
        if form_df.empty:
            st.info("No match data.")
        else:
            form_df["R"] = form_df["Result"].apply(_badge)
            st.markdown(
                form_df[["Date", "Opponent", "GS", "GC", "R", "Venue"]].to_html(escape=False, index=False),
                unsafe_allow_html=True,
            )
