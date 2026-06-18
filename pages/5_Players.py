import datetime
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.db.client import get_client
from src.ui.styles import apply_styles, top_nav, flag

st.set_page_config(page_title="Players · WC 2026", page_icon="👤", layout="wide")
apply_styles()
top_nav()


@st.cache_data(ttl=120)
def load_scorers() -> pd.DataFrame:
    docs = get_client().collection("wc_player_stats").stream()
    rows = [d.to_dict() for d in docs]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("goals", ascending=False)


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
            "Name":     p.get("name", "?"),
            "Position": p.get("position", ""),
            "Age":      age,
            "Nat":      p.get("nationality", ""),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<h1>👤 Players</h1>
<p class="page-sub">WC 2026 top scorers · assists · squad viewer</p>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🏅 Top Scorers", "👥 Team Squad"])

# ── TOP SCORERS ──────────────────────────────────────────────────────────────
with tab1:
    scorers_df = load_scorers()
    if scorers_df.empty:
        st.info("No scorer data yet — update results via the Fixtures page.")
    else:
        # Leaderboard
        st.markdown('<div class="section-label">Goals Leaderboard</div>', unsafe_allow_html=True)

        rows_html = ""
        rank_colors = ["#f59e0b", "#94a3b8", "#cd7f32"]
        for i, (_, r) in enumerate(scorers_df.head(20).iterrows()):
            rank = i + 1
            rc   = rank_colors[i] if i < 3 else "#334155"
            tf   = flag(r.get("team_name", ""))
            rows_html += f"""
<div class="lb-row">
  <div class="lb-rank {'top3' if rank <= 3 else ''}" style="color:{rc}">{rank}</div>
  <div style="flex:1">
    <div class="lb-name">{tf} {r.get('player_name','?')}</div>
    <div class="lb-team">{r.get('team_name','')}</div>
  </div>
  <div style="display:flex;gap:16px;align-items:center">
    <div style="text-align:center">
      <div class="lb-stat">{r.get('goals',0)}</div>
      <div style="font-size:.62rem;color:#334155;text-transform:uppercase">Goals</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:1rem;font-weight:700;color:#8b5cf6;min-width:24px;text-align:right">
        {r.get('assists',0)}
      </div>
      <div style="font-size:.62rem;color:#334155;text-transform:uppercase">Ast</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:.9rem;font-weight:700;color:#f59e0b;min-width:20px;text-align:right">
        {r.get('yellow_cards',0)}
      </div>
      <div style="font-size:.62rem;color:#334155;text-transform:uppercase">YC</div>
    </div>
  </div>
</div>"""

        st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f0f24,#141432);
            border:1px solid #1e1e3e;border-radius:14px;overflow:hidden">
  {rows_html}
</div>""", unsafe_allow_html=True)

        # Goals + assists chart
        st.markdown('<div class="section-label">Top 15 — Goals &amp; Assists</div>', unsafe_allow_html=True)
        top15 = scorers_df.head(15)
        labels = [f"{flag(r.get('team_name',''))} {r.get('player_name','?')}" for _, r in top15.iterrows()]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Goals", x=labels, y=top15.get("goals", []),
            marker=dict(color="#22c55e", opacity=0.9),
        ))
        fig.add_trace(go.Bar(
            name="Assists", x=labels, y=top15.get("assists", []),
            marker=dict(color="#8b5cf6", opacity=0.8),
        ))
        fig.update_layout(
            barmode="group", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=360, margin=dict(l=0, r=0, t=10, b=140),
            font=dict(family="Inter", color="#94a3b8"),
            xaxis=dict(tickangle=-45, tickfont=dict(size=10), gridcolor="#1a1a3a"),
            yaxis=dict(gridcolor="#1a1a3a"),
            legend=dict(orientation="h", y=1.08, font=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

# ── SQUAD VIEWER ────────────────────────────────────────────────────────────
with tab2:
    teams_df = load_teams()
    if teams_df.empty:
        st.error("No teams in DB.")
    else:
        team_name = st.selectbox("Select Team", teams_df["name"].tolist(), key="squad_sel")
        row = teams_df[teams_df["name"] == team_name].iloc[0]
        team_id = str(row["id"])
        group   = str(row.get("group_name") or "")
        f_      = flag(team_name)

        st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f0f24,#160d2e);
            border:1px solid #1e1e3e;border-radius:14px;padding:20px 24px;
            display:flex;align-items:center;gap:16px;margin:8px 0 16px">
  <span style="font-size:3rem">{f_}</span>
  <div>
    <div style="font-size:1.5rem;font-weight:900;color:#f8fafc">{team_name}</div>
    {'<span class="group-chip" style="margin-top:4px;display:inline-block">Group '+group+'</span>' if group else ''}
  </div>
</div>""", unsafe_allow_html=True)

        squad_df = load_squad(team_id)
        if squad_df.empty:
            st.info("No squad data — seed the DB to fetch player squads.")
        else:
            pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
            squad_df["_ord"] = squad_df["Position"].map(pos_order).fillna(9)
            squad_df = squad_df.sort_values("_ord").drop(columns="_ord")

            pos_info = {
                "GK":  ("🧤", "#0ea5e9", "Goalkeepers"),
                "DEF": ("🛡️", "#22c55e", "Defenders"),
                "MID": ("⚙️", "#8b5cf6", "Midfielders"),
                "FWD": ("⚡", "#f59e0b", "Forwards"),
            }
            for pos, (icon, color, label) in pos_info.items():
                grp = squad_df[squad_df["Position"] == pos].reset_index(drop=True)
                if grp.empty:
                    continue
                st.markdown(
                    f'<div class="section-label" style="color:{color};border-left-color:{color}">'
                    f'{icon} {label} ({len(grp)})</div>',
                    unsafe_allow_html=True,
                )
                # Grid of player cards
                cols = st.columns(4)
                for i, (_, p) in enumerate(grp.iterrows()):
                    with cols[i % 4]:
                        st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f0f24,#141432);
            border:1px solid #1e1e3e;border-radius:10px;padding:12px 14px;margin:4px 0;
            border-left:3px solid {color}">
  <div style="font-size:.88rem;font-weight:700;color:#e2e8f0">{p["Name"]}</div>
  <div style="font-size:.72rem;color:#475569;margin-top:3px">
    Age {p.get("Age","?")} &nbsp;·&nbsp; {p.get("Nat","")[:20]}
  </div>
</div>""", unsafe_allow_html=True)
