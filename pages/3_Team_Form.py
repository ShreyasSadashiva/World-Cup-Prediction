"""
Team Form — last 15 pre-WC matches per team.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.db.client import get_client
from src.ml.features import match_result_label

st.set_page_config(page_title="Team Form · WC 2026", page_icon="📈", layout="wide")
st.title("📈 Team Form")
st.caption("Last 15 international matches before WC 2026 kick-off")


@st.cache_data(ttl=300)
def load_teams() -> pd.DataFrame:
    docs = get_client().collection("teams").stream()
    rows = [{"id": d.id, **d.to_dict()} for d in docs]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df.sort_values("name") if not df.empty else df


@st.cache_data(ttl=300)
def load_team_matches(team_id: str) -> pd.DataFrame:
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
            "GS": gs, "GC": gc,
            "Result": match_result_label(d.get("home_goals", 0), d.get("away_goals", 0), "away"),
            "Venue": "A", "Competition": d.get("competition", ""),
        })

    return pd.DataFrame(records).sort_values("Date", ascending=False).head(15) if records else pd.DataFrame()


teams_df = load_teams()
if teams_df.empty:
    st.error("No teams in DB — run `python -m scripts.seed_db` first.")
    st.stop()

team_name = st.selectbox("Select Team", teams_df["name"].tolist())
team_id = str(teams_df[teams_df["name"] == team_name]["id"].values[0])
group = teams_df[teams_df["name"] == team_name]["group_name"].values[0]

st.markdown(f"**Group {group}** · WC 2026" if group else "**WC 2026**")

matches_df = load_team_matches(team_id)

if matches_df.empty:
    st.info("No recent match data found for this team.")
    st.stop()

st.subheader("Form Strip (most recent → oldest)")
colors = {"W": "#198754", "D": "#6c757d", "L": "#dc3545"}
badges = " ".join(
    f'<span style="background:{colors[r]};color:white;padding:3px 10px;border-radius:4px;'
    f'font-weight:bold;font-size:1.1em;margin:2px">{r}</span>'
    for r in matches_df["Result"].tolist()
)
st.markdown(badges, unsafe_allow_html=True)

st.divider()
n = len(matches_df)
wins = (matches_df["Result"] == "W").sum()
draws = (matches_df["Result"] == "D").sum()
losses = (matches_df["Result"] == "L").sum()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Matches", n)
c2.metric("Wins", wins)
c3.metric("Draws", draws)
c4.metric("Losses", losses)
c5.metric("Avg Scored", f"{matches_df['GS'].mean():.2f}")
c6.metric("Avg Conceded", f"{matches_df['GC'].mean():.2f}")

st.divider()
st.subheader("Goals Per Match")

chart_df = matches_df.iloc[::-1]
labels = [f"{r['Date'][:10]} vs {r['Opponent']}" for _, r in chart_df.iterrows()]

fig = go.Figure()
fig.add_trace(go.Bar(name="Scored", x=labels, y=chart_df["GS"].tolist(), marker_color="#198754"))
fig.add_trace(go.Bar(name="Conceded", x=labels, y=chart_df["GC"].tolist(), marker_color="#dc3545"))
fig.update_layout(
    barmode="group", template="plotly_dark", height=350,
    xaxis=dict(tickangle=-45), margin=dict(b=120, t=20),
    legend=dict(orientation="h", y=1.05),
)
st.plotly_chart(fig, use_container_width=True)

col_pie, col_table = st.columns([1, 2])
with col_pie:
    fig2 = go.Figure(go.Pie(
        labels=["Wins", "Draws", "Losses"], values=[wins, draws, losses],
        marker_colors=["#198754", "#6c757d", "#dc3545"], hole=0.45,
    ))
    fig2.update_layout(height=300, template="plotly_dark", margin=dict(t=10))
    st.plotly_chart(fig2, use_container_width=True)

with col_table:
    st.subheader("Match Log")
    display = matches_df[["Date", "Opponent", "Venue", "GS", "GC", "Result", "Competition"]].copy()
    display["Date"] = display["Date"].str[:10]

    def _colour(val):
        c = {"W": "#19875440", "D": "#6c757d40", "L": "#dc354540"}.get(val, "")
        return f"background-color: {c}"

    st.dataframe(
        display.style.applymap(_colour, subset=["Result"]),
        use_container_width=True, hide_index=True,
    )
