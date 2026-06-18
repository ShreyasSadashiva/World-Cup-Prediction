import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.db.client import get_client
from src.ml.features import match_result_label
from src.ui.styles import apply_styles, top_nav, flag

st.set_page_config(page_title="Head to Head · WC 2026", page_icon="🤝", layout="wide")
apply_styles()
top_nav()


@st.cache_data(ttl=300)
def load_teams() -> pd.DataFrame:
    docs = get_client().collection("teams").stream()
    rows = [{"id": d.id, **d.to_dict()} for d in docs]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df.sort_values("name") if not df.empty else df


@st.cache_data(ttl=300)
def load_all_matches() -> list[dict]:
    db = get_client()
    recent = [d.to_dict() for d in db.collection("team_recent_matches").stream()]
    wc = [d.to_dict() for d in db.collection("wc_fixtures").stream()
          if d.to_dict().get("status") == "FINISHED"]
    return recent + wc


@st.cache_data(ttl=300)
def load_pair_history(slug1: str, slug2: str) -> list[dict]:
    db = get_client()
    records = []
    for coll, ek in [("team_recent_matches", "competition"), ("wc_fixtures", "stage")]:
        for hid, aid in [(slug1, slug2), (slug2, slug1)]:
            for doc in (db.collection(coll)
                          .where("home_team_id", "==", hid)
                          .where("away_team_id", "==", aid)
                          .stream()):
                d = doc.to_dict()
                if coll == "wc_fixtures" and d.get("home_goals") is None:
                    continue
                records.append({
                    "Date":       d.get("match_date", ""),
                    "home_slug":  hid,
                    "Home":       d.get("home_team_name", hid),
                    "HG":         d.get("home_goals", 0),
                    "AG":         d.get("away_goals", 0),
                    "Away":       d.get("away_team_name", aid),
                    "Comp":       d.get(ek, ""),
                })
    return sorted(records, key=lambda x: x["Date"], reverse=True)


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<h1>🤝 Head to Head</h1>
<p class="page-sub">Historical matchups between any two WC 2026 teams · full matrix view</p>
""", unsafe_allow_html=True)

teams_df = load_teams()
if teams_df.empty:
    st.error("No teams — run `python -m scripts.seed_db` first.")
    st.stop()

team_names   = teams_df["name"].tolist()
name_to_slug = dict(zip(teams_df["name"], teams_df["id"]))

view = st.radio("View", ["Pair Detail", "Matrix Overview"], horizontal=True, label_visibility="collapsed")

# ── PAIR DETAIL ─────────────────────────────────────────────────────────────
if view == "Pair Detail":
    c1, c2 = st.columns(2)
    with c1:
        t1 = st.selectbox("Team 1", team_names, index=0)
    with c2:
        t2 = st.selectbox("Team 2", team_names, index=min(1, len(team_names) - 1))

    if t1 == t2:
        st.warning("Select two different teams.")
        st.stop()

    slug1, slug2 = name_to_slug[t1], name_to_slug[t2]
    history = load_pair_history(slug1, slug2)

    if not history:
        st.markdown(f"""
<div style="background:#0f0f24;border:1px solid #1e1e3e;border-radius:14px;padding:32px;text-align:center">
  <div style="font-size:2.5rem;margin-bottom:12px">{flag(t1)} 🆚 {flag(t2)}</div>
  <div style="color:#475569;font-size:.9rem">No historical meetings found</div>
</div>""", unsafe_allow_html=True)
    else:
        t1_wins = draws = t2_wins = 0
        for r in history:
            if r["home_slug"] == slug1:
                res = match_result_label(int(r["HG"]), int(r["AG"]), "home")
            else:
                res = match_result_label(int(r["HG"]), int(r["AG"]), "away")
            if res == "W": t1_wins += 1
            elif res == "D": draws += 1
            else: t2_wins += 1

        total = t1_wins + draws + t2_wins
        total_goals = sum(int(r["HG"]) + int(r["AG"]) for r in history)
        f1, f2 = flag(t1), flag(t2)

        # Header with flags
        st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f0f24,#160d2e,#0d162e);
            border:1px solid #1e1e3e;border-radius:16px;padding:28px 32px;margin:12px 0;
            display:flex;align-items:center;justify-content:space-around;text-align:center">
  <div>
    <div style="font-size:3rem">{f1}</div>
    <div style="font-size:1.5rem;font-weight:900;color:#f8fafc;margin:6px 0">{t1_wins}</div>
    <div style="font-size:.7rem;color:#475569;text-transform:uppercase;letter-spacing:.1em">Wins</div>
  </div>
  <div>
    <div style="font-size:1.3rem;font-weight:900;color:#334155">VS</div>
    <div style="margin-top:12px;font-size:1.8rem;font-weight:900;color:#f59e0b">{draws}</div>
    <div style="font-size:.7rem;color:#475569;text-transform:uppercase;letter-spacing:.1em">Draws</div>
  </div>
  <div>
    <div style="font-size:3rem">{f2}</div>
    <div style="font-size:1.5rem;font-weight:900;color:#f8fafc;margin:6px 0">{t2_wins}</div>
    <div style="font-size:.7rem;color:#475569;text-transform:uppercase;letter-spacing:.1em">Wins</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Donut chart
        col_chart, col_stats = st.columns([1, 1])
        with col_chart:
            fig = go.Figure(go.Pie(
                labels=[f"{t1} Wins", "Draws", f"{t2} Wins"],
                values=[t1_wins, draws, t2_wins],
                marker_colors=["#22c55e", "#f59e0b", "#3b82f6"],
                hole=0.6,
                textinfo="label+percent",
                textfont=dict(color="#e2e8f0", size=11),
            ))
            fig.update_layout(
                height=300, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_stats:
            st.markdown(f"""
<div class="stat-grid" style="flex-direction:column">
  <div class="stat-card"><div class="sc-val">{total}</div><div class="sc-lbl">Total Meetings</div></div>
  <div class="stat-card"><div class="sc-val">{total_goals}</div><div class="sc-lbl">Total Goals</div></div>
  <div class="stat-card">
    <div class="sc-val">{total_goals/total:.1f}</div>
    <div class="sc-lbl">Avg Goals/Match</div>
  </div>
</div>""", unsafe_allow_html=True)

        # Match history table
        st.markdown('<div class="section-label">Match History</div>', unsafe_allow_html=True)
        rows_html = ""
        for r in history:
            is_t1_home = r["home_slug"] == slug1
            res_from_t1 = match_result_label(int(r["HG"]), int(r["AG"]),
                                              "home" if is_t1_home else "away")
            rc = {"W": "#22c55e", "D": "#f59e0b", "L": "#ef4444"}.get(res_from_t1, "#94a3b8")
            hf, af = flag(r["Home"]), flag(r["Away"])
            rows_html += f"""
<tr>
  <td style="color:#475569;font-size:.78rem">{str(r["Date"])[:10]}</td>
  <td style="font-weight:600;color:#e2e8f0">{hf} {r["Home"]}</td>
  <td style="text-align:center;font-weight:900;font-size:1.1rem;color:#f8fafc">{r["HG"]} – {r["AG"]}</td>
  <td style="font-weight:600;color:#e2e8f0">{af} {r["Away"]}</td>
  <td style="text-align:center">
    <span style="background:{rc}22;color:{rc};padding:2px 8px;border-radius:5px;
                 font-weight:700;font-size:.78rem;border:1px solid {rc}44">
      {res_from_t1}
    </span>
  </td>
  <td style="font-size:.72rem;color:#334155">{r["Comp"][:35]}</td>
</tr>"""

        st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f0f24,#141432);border:1px solid #1e1e3e;
            border-radius:14px;overflow:hidden">
  <table class="st-table">
    <thead><tr>
      <th>Date</th><th>Home</th><th>Score</th><th>Away</th><th>Result</th><th>Competition</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>""", unsafe_allow_html=True)

# ── MATRIX OVERVIEW ─────────────────────────────────────────────────────────
else:
    st.markdown('<div class="section-label">Home-team win rate matrix — all WC team pairs</div>', unsafe_allow_html=True)
    all_matches = load_all_matches()

    if not all_matches:
        st.info("No match data yet.")
        st.stop()

    slugs = teams_df["id"].tolist()
    names = teams_df["name"].tolist()
    n = len(slugs)
    slug_idx = {s: i for i, s in enumerate(slugs)}
    counts = np.zeros((n, n))
    wins   = np.zeros((n, n))

    for m in all_matches:
        hi = slug_idx.get(m.get("home_team_id"))
        ai = slug_idx.get(m.get("away_team_id"))
        if hi is None or ai is None or hi == ai:
            continue
        hg = m.get("home_goals") or 0
        ag = m.get("away_goals") or 0
        counts[hi][ai] += 1
        if hg > ag: wins[hi][ai] += 1

    with np.errstate(invalid="ignore"):
        matrix = np.where(counts > 0, wins / counts, np.nan)

    has_data = ~np.all(np.isnan(matrix), axis=1) & ~np.all(np.isnan(matrix), axis=0)
    mat_f    = matrix[has_data][:, has_data]
    names_f  = [names[i] for i in range(n) if has_data[i]]

    if mat_f.size == 0:
        st.info("Not enough H2H data yet.")
    else:
        text_mat = [[f"{v:.0%}" if not np.isnan(v) else "" for v in row] for row in mat_f]
        fig = go.Figure(go.Heatmap(
            z=mat_f, x=names_f, y=names_f,
            text=text_mat, texttemplate="%{text}",
            colorscale=[[0, "#7f1d1d"], [0.5, "#1e3a5f"], [1, "#14532d"]],
            zmin=0, zmax=1,
            colorbar=dict(
                title="Win Rate", tickformat=".0%",
                tickfont=dict(color="#94a3b8"),
                titlefont=dict(color="#94a3b8"),
                bgcolor="#0f0f24",
            ),
            hovertemplate="%{y} vs %{x}: %{text}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="Head-to-Head Win Rate Matrix (row team wins)", font=dict(color="#cbd5e1")),
            height=max(600, len(names_f) * 22),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickangle=-45, tickfont=dict(size=9, color="#64748b")),
            yaxis=dict(tickfont=dict(size=9, color="#64748b")),
            margin=dict(l=120, b=140, t=50),
        )
        st.plotly_chart(fig, use_container_width=True)
