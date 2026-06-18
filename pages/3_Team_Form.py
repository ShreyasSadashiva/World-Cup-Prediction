import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.db.client import get_client
from src.ml.features import match_result_label
from src.ui.styles import apply_styles, top_nav, flag, form_strip, stat_grid

st.set_page_config(page_title="Team Form · WC 2026", page_icon="📈", layout="wide")
apply_styles()
top_nav()


@st.cache_data(ttl=300)
def load_teams() -> pd.DataFrame:
    docs = get_client().collection("teams").stream()
    rows = [{"id": d.id, **d.to_dict()} for d in docs]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df.sort_values("name") if not df.empty else df


@st.cache_data(ttl=300)
def load_matches(team_id: str) -> pd.DataFrame:
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
                        "Venue": "H", "Competition": d.get("competition", "")})
    for d in away:
        gs, gc = d.get("away_goals", 0), d.get("home_goals", 0)
        records.append({"Date": d.get("match_date", ""), "Opponent": d.get("home_team_name", "?"),
                        "GS": gs, "GC": gc,
                        "Result": match_result_label(d.get("home_goals", 0), d.get("away_goals", 0), "away"),
                        "Venue": "A", "Competition": d.get("competition", "")})
    return pd.DataFrame(records).sort_values("Date", ascending=False).head(15) if records else pd.DataFrame()


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<h1>📈 Team Form</h1>
<p class="page-sub">Last 15 international matches before WC 2026 kick-off</p>
""", unsafe_allow_html=True)

teams_df = load_teams()
if teams_df.empty:
    st.error("No teams — run `python -m scripts.seed_db` first.")
    st.stop()

team_name = st.selectbox("Select Team", teams_df["name"].tolist(), label_visibility="collapsed")
row = teams_df[teams_df["name"] == team_name].iloc[0]
team_id = str(row["id"])
group = str(row.get("group_name") or "")

# Team hero card
f_ = flag(team_name)
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f0f24,#160d2e,#0d162e);
            border:1px solid #1e1e3e;border-radius:16px;padding:24px 28px;
            display:flex;align-items:center;gap:20px;margin:8px 0 16px">
  <span style="font-size:3.5rem;line-height:1">{f_}</span>
  <div>
    <div style="font-size:1.8rem;font-weight:900;color:#f8fafc;line-height:1">{team_name}</div>
    {'<div style="margin-top:6px"><span class="group-chip">Group '+group+'</span></div>' if group else ''}
  </div>
</div>
""", unsafe_allow_html=True)

matches_df = load_matches(team_id)

if matches_df.empty:
    st.info("No recent match data found for this team.")
    st.stop()

# ── Form strip ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Recent Form</div>', unsafe_allow_html=True)
results = matches_df["Result"].tolist()
st.markdown(form_strip(results), unsafe_allow_html=True)

# ── Stat cards ──────────────────────────────────────────────────────────────
n = len(matches_df)
wins   = (matches_df["Result"] == "W").sum()
draws  = (matches_df["Result"] == "D").sum()
losses = (matches_df["Result"] == "L").sum()
avg_gs = matches_df["GS"].mean()
avg_gc = matches_df["GC"].mean()
clean  = (matches_df["GC"] == 0).sum()
win_rt = wins / n if n else 0

st.markdown(stat_grid([
    (f"{wins}", "Wins"),
    (f"{draws}", "Draws"),
    (f"{losses}", "Losses"),
    (f"{win_rt:.0%}", "Win Rate"),
    (f"{avg_gs:.2f}", "Avg Scored"),
    (f"{avg_gc:.2f}", "Avg Conceded"),
    (f"{int(clean)}", "Clean Sheets"),
    (f"{avg_gs - avg_gc:+.2f}", "Goal Diff"),
]), unsafe_allow_html=True)

st.divider()

# ── Goals chart ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Goals Per Match</div>', unsafe_allow_html=True)

chart_df = matches_df.iloc[::-1]
labels = [f"{r['Date'][:10]} · {r['Opponent']}" for _, r in chart_df.iterrows()]

fig = go.Figure()
fig.add_trace(go.Bar(
    name="Scored",   x=labels, y=chart_df["GS"].tolist(),
    marker=dict(color="#22c55e", opacity=0.85),
))
fig.add_trace(go.Bar(
    name="Conceded", x=labels, y=chart_df["GC"].tolist(),
    marker=dict(color="#ef4444", opacity=0.7),
))
fig.update_layout(
    barmode="group", template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=320, margin=dict(l=0, r=0, t=10, b=120),
    font=dict(family="Inter", color="#94a3b8"),
    xaxis=dict(tickangle=-40, tickfont=dict(size=10), gridcolor="#1a1a3a"),
    yaxis=dict(gridcolor="#1a1a3a"),
    legend=dict(orientation="h", y=1.08, font=dict(size=11)),
)
st.plotly_chart(fig, use_container_width=True)

# ── Match log ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Match Log</div>', unsafe_allow_html=True)

log_rows = ""
for _, r in matches_df.iterrows():
    res = r["Result"]
    badge_cls = f"fb fb-{res}"
    opp_flag = flag(r["Opponent"])
    score_color = "#22c55e" if res == "W" else ("#f59e0b" if res == "D" else "#ef4444")
    log_rows += f"""
<tr>
  <td style="color:#475569;font-size:.78rem">{str(r["Date"])[:10]}</td>
  <td><span style="font-size:.85rem;color:#e2e8f0;font-weight:600">{opp_flag} {r["Opponent"]}</span></td>
  <td style="text-align:center">
    <span style="font-size:.72rem;color:#475569;background:#0a0a1e;padding:2px 8px;border-radius:6px">{r["Venue"]}</span>
  </td>
  <td style="text-align:center;font-weight:800;color:{score_color};font-size:1rem">{r["GS"]}–{r["GC"]}</td>
  <td style="text-align:center"><span class="{badge_cls}">{res}</span></td>
  <td style="font-size:.72rem;color:#334155">{r.get("Competition","")[:35]}</td>
</tr>"""

st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f0f24,#141432);border:1px solid #1e1e3e;
            border-radius:14px;overflow:hidden">
  <table class="st-table" style="font-size:.85rem">
    <thead><tr>
      <th>Date</th><th>Opponent</th><th>Venue</th>
      <th>Score</th><th>Result</th><th>Competition</th>
    </tr></thead>
    <tbody>{log_rows}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)
