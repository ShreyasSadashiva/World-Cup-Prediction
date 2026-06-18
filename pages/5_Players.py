"""
Players — WC 2026 top scorers and team squad viewer.
"""

import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.db.client import get_client

st.set_page_config(page_title="Players · WC 2026", page_icon="👤", layout="wide")
st.title("👤 Players")


@st.cache_data(ttl=120)
def load_scorers() -> pd.DataFrame:
    docs = get_client().collection("wc_player_stats").stream()
    rows = [d.to_dict() for d in docs]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("goals", ascending=False)


@st.cache_data(ttl=300)
def load_teams() -> pd.DataFrame:
    docs = get_client().collection("teams").stream()
    rows = [{"id": d.id, **d.to_dict()} for d in docs]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df.sort_values("name") if not df.empty else df


@st.cache_data(ttl=300)
def load_squad(team_id: str) -> pd.DataFrame:
    docs = get_client().collection("players").where("team_id", "==", team_id).stream()
    rows = []
    for doc in docs:
        p = doc.to_dict()
        dob = p.get("date_of_birth")
        age = ""
        if dob:
            try:
                born = pd.to_datetime(dob)
                age = str((datetime.date(2026, 6, 11) - born.date()).days // 365)
            except Exception:
                pass
        rows.append({
            "Name":        p.get("name", "?"),
            "Position":    p.get("position", ""),
            "Age":         age,
            "Nationality": p.get("nationality", ""),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


tab1, tab2 = st.tabs(["🏅 WC Top Scorers", "👥 Team Squad"])

with tab1:
    scorers_df = load_scorers()
    if scorers_df.empty:
        st.info(
            "No scorer data yet — update WC results via the Fixtures page "
            "or run `python -m scripts.update_wc`."
        )
    else:
        display_cols = [c for c in ["player_name", "team_name", "goals", "assists",
                                     "yellow_cards", "red_cards", "minutes_played"]
                        if c in scorers_df.columns]
        display = scorers_df[display_cols].copy()
        display.columns = [c.replace("_", " ").title() for c in display_cols]
        display.index = range(1, len(display) + 1)
        st.dataframe(display, use_container_width=True)

        top15 = scorers_df.head(15)
        if not top15.empty and "player_name" in top15.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Goals", x=top15["player_name"], y=top15.get("goals", []),
                marker_color="#198754",
            ))
            fig.add_trace(go.Bar(
                name="Assists", x=top15["player_name"], y=top15.get("assists", []),
                marker_color="#0d6efd",
            ))
            fig.update_layout(
                barmode="group", template="plotly_dark", height=380,
                xaxis=dict(tickangle=-45), margin=dict(b=130, t=20),
                legend=dict(orientation="h", y=1.05),
                title="Top 15 Players — Goals & Assists",
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    teams_df = load_teams()
    if teams_df.empty:
        st.error("No teams in DB.")
    else:
        team_name = st.selectbox("Select Team", teams_df["name"].tolist())
        team_id = str(teams_df[teams_df["name"] == team_name]["id"].values[0])
        group = teams_df[teams_df["name"] == team_name].get("group_name", pd.Series([None])).values[0]
        if group:
            st.markdown(f"**Group {group}**")

        squad_df = load_squad(team_id)
        if squad_df.empty:
            st.info("No squad data — seed the DB to fetch player squads.")
        else:
            pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
            squad_df["_order"] = squad_df["Position"].map(pos_order).fillna(9)
            squad_df = squad_df.sort_values("_order").drop(columns="_order")

            pos_labels = {"GK": "🧤 Goalkeepers", "DEF": "🛡️ Defenders",
                          "MID": "⚙️ Midfielders", "FWD": "⚡ Forwards"}
            for pos in ["GK", "DEF", "MID", "FWD"]:
                grp = squad_df[squad_df["Position"] == pos]
                if not grp.empty:
                    st.subheader(pos_labels[pos])
                    st.dataframe(
                        grp[["Name", "Age", "Nationality"]].reset_index(drop=True),
                        use_container_width=True, hide_index=True,
                    )
