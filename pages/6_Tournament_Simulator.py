"""
Tournament Simulator — simulates the rest of WC 2026 using current results
and the ML model for unplayed matches.

Modes:
  Single run  — always picks the highest-probability outcome
  Monte Carlo — samples from probability distributions N times,
                shows each team's chance of winning the tournament
"""

import random
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.db.client import get_client
from src.ml.features import FEATURE_COLS
from src.ml.model import is_trained, load as load_model
from src.ui.styles import apply_styles, top_nav, bracket_card, champion_banner, flag, standings_table

# ── constants ─────────────────────────────────────────────────────────────────
_ROUNDS = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"]


# ── data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Loading tournament data…")
def load_sim_data() -> dict:
    db = get_client()

    teams = {d.id: {"id": d.id, **d.to_dict()} for d in db.collection("teams").stream()}
    fixtures = [{"id": d.id, **d.to_dict()} for d in db.collection("wc_fixtures").stream()]
    recent_all = [d.to_dict() for d in db.collection("team_recent_matches").stream()]

    return {"teams": teams, "fixtures": fixtures, "recent_all": recent_all}


# ── pre-compute team form features (no DB calls during simulation) ────────────

def _stats_from_records(records: list[dict]) -> dict:
    if not records:
        return {"win_rate": .33, "draw_rate": .33, "loss_rate": .33,
                "avg_goals_scored": 1.1, "avg_goals_conceded": 1.1, "avg_goal_diff": 0.0,
                "weighted_form": 1.0, "clean_sheet_rate": .3, "scoring_rate": .7}

    records = sorted(records, key=lambda r: r.get("date", ""))[-15:]
    n = len(records)
    results, gs_list, gc_list = [], [], []
    for r in records:
        gs, gc = r["gs"], r["gc"]
        gs_list.append(gs); gc_list.append(gc)
        results.append("W" if gs > gc else ("D" if gs == gc else "L"))

    gs_arr = np.array(gs_list, dtype=float)
    gc_arr = np.array(gc_list, dtype=float)
    form_pts = np.array([3 if r == "W" else (1 if r == "D" else 0) for r in results], dtype=float)
    weights = np.exp(np.linspace(-1, 0, n))

    return {
        "win_rate":          results.count("W") / n,
        "draw_rate":         results.count("D") / n,
        "loss_rate":         results.count("L") / n,
        "avg_goals_scored":  float(gs_arr.mean()),
        "avg_goals_conceded":float(gc_arr.mean()),
        "avg_goal_diff":     float((gs_arr - gc_arr).mean()),
        "weighted_form":     float(np.average(form_pts, weights=weights)),
        "clean_sheet_rate":  float((gc_arr == 0).mean()),
        "scoring_rate":      float((gs_arr > 0).mean()),
    }


def build_all_features(data: dict) -> dict[str, dict]:
    """Pre-compute form stats for every WC team from in-memory data."""
    team_ids = set(data["teams"].keys())
    team_records: dict[str, list] = defaultdict(list)

    for m in data["recent_all"]:
        hid = m.get("home_team_id")
        aid = m.get("away_team_id")
        date = m.get("match_date", "")
        if hid in team_ids:
            team_records[hid].append({"date": date, "gs": m.get("home_goals", 0),
                                      "gc": m.get("away_goals", 0)})
        if aid in team_ids:
            team_records[aid].append({"date": date, "gs": m.get("away_goals", 0),
                                      "gc": m.get("home_goals", 0)})

    # WC form (wins in finished WC fixtures)
    wc_wins: dict[str, int] = defaultdict(int)
    wc_played: dict[str, int] = defaultdict(int)
    for f in data["fixtures"]:
        if f.get("status") != "FINISHED":
            continue
        hid, aid = f.get("home_team_id"), f.get("away_team_id")
        hg, ag = f.get("home_goals") or 0, f.get("away_goals") or 0
        if hid:
            wc_played[hid] += 1
            if hg > ag: wc_wins[hid] += 1
        if aid:
            wc_played[aid] += 1
            if ag > hg: wc_wins[aid] += 1

    features: dict[str, dict] = {}
    for tid in team_ids:
        stats = _stats_from_records(team_records[tid])
        played = wc_played[tid]
        stats["wc_form"] = wc_wins[tid] / played if played else 0.33
        features[tid] = stats

    return features


# ── single match prediction (no DB) ──────────────────────────────────────────

def _predict_match(home_id: str, away_id: str, all_feats: dict[str, dict],
                   mode: str = "monte_carlo", allow_draw: bool = True) -> str:
    """Return 'home', 'draw', or 'away'. In knockout (allow_draw=False) draws go to penalties."""
    h = all_feats.get(home_id, all_feats.get(list(all_feats)[0]))
    a = all_feats.get(away_id, all_feats.get(list(all_feats)[0]))

    feat = {
        **{f"home_{k}": v for k, v in h.items() if k != "wc_form"},
        **{f"away_{k}": v for k, v in a.items() if k != "wc_form"},
        "win_rate_diff":        h["win_rate"] - a["win_rate"],
        "form_diff":            h["weighted_form"] - a["weighted_form"],
        "goal_diff_diff":       h["avg_goal_diff"] - a["avg_goal_diff"],
        "goals_scored_diff":    h["avg_goals_scored"] - a["avg_goals_scored"],
        "goals_conceded_diff":  h["avg_goals_conceded"] - a["avg_goals_conceded"],
        "h2h_home_win_rate": 0.45, "h2h_draw_rate": 0.27,
        "h2h_goal_diff": 0.0,  "h2h_n": 0,
        "neutral":       1,
        "wc_form_diff":  h["wc_form"] - a["wc_form"],
    }
    feat_vec = {k: feat.get(k, 0.0) for k in FEATURE_COLS}

    model, encoder, feature_cols = load_model()
    if model is None:
        hw, dr, aw = 0.40, 0.25, 0.35
    else:
        import pandas as _pd
        X = _pd.DataFrame([feat_vec])[feature_cols].fillna(0.0)
        proba = model.predict_proba(X)[0]
        cls = encoder.classes_  # ["A","D","H"]
        hw = float(proba[list(cls).index("H")])
        dr = float(proba[list(cls).index("D")])
        aw = float(proba[list(cls).index("A")])

    if not allow_draw:
        # Eliminate draw: redistribute proportionally to home/away
        total = hw + aw
        hw, aw = hw / total, aw / total
        dr = 0.0

    if mode == "deterministic":
        if hw >= dr and hw >= aw: return "home"
        if dr >= hw and dr >= aw: return "draw"
        return "away"
    else:
        r = random.random()
        if r < hw: return "home"
        if r < hw + dr: return "draw"
        return "away"


def _sim_score(home_id: str, away_id: str, outcome: str, all_feats: dict) -> tuple[int, int]:
    """Generate a plausible scoreline consistent with the predicted outcome."""
    h = all_feats.get(home_id, {})
    a = all_feats.get(away_id, {})
    exp_h = (h.get("avg_goals_scored", 1.2) + a.get("avg_goals_conceded", 1.2)) / 2
    exp_a = (a.get("avg_goals_scored", 1.0) + h.get("avg_goals_conceded", 1.0)) / 2

    for _ in range(20):
        hg = np.random.poisson(max(exp_h, 0.5))
        ag = np.random.poisson(max(exp_a, 0.5))
        if outcome == "home" and hg > ag: return int(hg), int(ag)
        if outcome == "draw" and hg == ag: return int(hg), int(ag)
        if outcome == "away" and ag > hg: return int(hg), int(ag)

    # Fallback
    if outcome == "home": return 1, 0
    if outcome == "draw": return 1, 1
    return 0, 1


# ── group stage ───────────────────────────────────────────────────────────────

def simulate_group_stage(data: dict, all_feats: dict, mode: str) -> dict:
    """
    Returns:
        {
          group_name: {
            team_id: {"name", "P","W","D","L","GF","GA","GD","Pts", "simulated_pts"},
          },
          "_match_log": [{home_name, away_name, hg, ag, simulated, group}]
        }
    """
    # Build group → teams mapping
    groups: dict[str, set] = defaultdict(set)
    for f in data["fixtures"]:
        g = f.get("group_name")
        if not g:
            continue
        hid, aid = f.get("home_team_id"), f.get("away_team_id")
        if hid: groups[g].add(hid)
        if aid: groups[g].add(aid)

    # Init standings
    standings: dict[str, dict[str, dict]] = {}
    for g, tids in groups.items():
        standings[g] = {}
        for tid in tids:
            name = data["teams"].get(tid, {}).get("name", tid)
            standings[g][tid] = {"name": name, "P": 0, "W": 0, "D": 0, "L": 0,
                                 "GF": 0, "GA": 0, "GD": 0, "Pts": 0, "sim_pts": 0}

    match_log = []

    for f in data["fixtures"]:
        g = f.get("group_name")
        if not g:
            continue
        hid, aid = f.get("home_team_id"), f.get("away_team_id")
        if not hid or not aid:
            continue
        if hid not in standings.get(g, {}):
            continue

        if f.get("status") == "FINISHED":
            hg = int(f.get("home_goals") or 0)
            ag = int(f.get("away_goals") or 0)
            outcome = "home" if hg > ag else ("draw" if hg == ag else "away")
            simulated = False
        else:
            outcome = _predict_match(hid, aid, all_feats, mode=mode, allow_draw=True)
            hg, ag = _sim_score(hid, aid, outcome, all_feats)
            simulated = True

        # Update standings
        for sid, sg, gc in [(hid, hg, ag), (aid, ag, hg)]:
            if sid in standings[g]:
                s = standings[g][sid]
                s["P"] += 1; s["GF"] += sg; s["GA"] += gc
                if sg > gc: s["W"] += 1; s["Pts"] += 3
                elif sg == gc: s["D"] += 1; s["Pts"] += 1
                else: s["L"] += 1
                if simulated: s["sim_pts"] += (3 if sg > gc else (1 if sg == gc else 0))

        match_log.append({
            "group": g,
            "home": data["teams"].get(hid, {}).get("name", hid),
            "away": data["teams"].get(aid, {}).get("name", aid),
            "hg": hg, "ag": ag, "simulated": simulated,
        })

    # Compute GD and sort
    for g in standings:
        for s in standings[g].values():
            s["GD"] = s["GF"] - s["GA"]
        standings[g] = dict(
            sorted(standings[g].items(),
                   key=lambda kv: (kv[1]["Pts"], kv[1]["GD"], kv[1]["GF"]),
                   reverse=True)
        )

    standings["_match_log"] = match_log
    return standings


# ── qualifier selection ───────────────────────────────────────────────────────

def get_qualifiers(standings: dict) -> list[dict]:
    """
    Returns 32 teams:
      - Top 2 from each of 12 groups (24 teams)
      - 8 best third-place finishers
    Each entry: {team_id, team_name, qualified_as, seed}
    """
    qualifiers = []
    third_place = []

    groups = sorted(k for k in standings if k != "_match_log")
    for g in groups:
        group_teams = list(standings[g].items())
        if len(group_teams) < 1: continue

        for pos, (tid, stats) in enumerate(group_teams):
            label = f"Group {g} — {'1st' if pos == 0 else ('2nd' if pos == 1 else ('3rd' if pos == 2 else '4th'))}"
            if pos < 2:
                qualifiers.append({
                    "team_id": tid, "team_name": stats["name"],
                    "qualified_as": label, "position": pos,
                    "group": g, "pts": stats["Pts"],
                    "gd": stats["GD"], "gf": stats["GF"],
                })
            elif pos == 2:
                third_place.append({
                    "team_id": tid, "team_name": stats["name"],
                    "qualified_as": label, "position": 2,
                    "group": g, "pts": stats["Pts"],
                    "gd": stats["GD"], "gf": stats["GF"],
                })

    # Best 8 third-place teams
    third_place.sort(key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
    for t in third_place[:8]:
        t["qualified_as"] += " (best 3rd)"
        qualifiers.append(t)

    # Seed: group winners (by group order) first, then runners-up, then best 3rd
    winners = [q for q in qualifiers if q["position"] == 0]
    runners_up = [q for q in qualifiers if q["position"] == 1]
    thirds = [q for q in qualifiers if q["position"] == 2]

    seeded = winners + runners_up + thirds
    for i, q in enumerate(seeded):
        q["seed"] = i + 1

    return seeded


# ── knockout simulator ────────────────────────────────────────────────────────

def simulate_knockout(qualifiers: list[dict], all_feats: dict,
                      mode: str) -> list[dict]:
    """
    Simulate all knockout rounds. Returns a list of round dicts:
    [{"round": str, "matches": [{"home", "away", "hg", "ag", "winner"}]}]
    """
    # Seed order: 1 vs 32, 2 vs 31, ..., 16 vs 17
    bracket = [qualifiers[i]["team_id"] for i in range(len(qualifiers))]
    name_map = {q["team_id"]: q["team_name"] for q in qualifiers}
    # Add any team names from data that might appear
    for q in qualifiers:
        name_map[q["team_id"]] = q["team_name"]

    rounds = []
    remaining = bracket[:32]  # ensure exactly 32

    for round_name in _ROUNDS:
        if len(remaining) < 2:
            break
        matches = []
        next_round = []
        mid = len(remaining) // 2

        for i in range(mid):
            home_id = remaining[i]
            away_id = remaining[len(remaining) - 1 - i]
            outcome = _predict_match(home_id, away_id, all_feats,
                                     mode=mode, allow_draw=False)
            hg, ag = _sim_score(home_id, away_id, outcome, all_feats)
            winner_id = home_id if outcome == "home" else away_id
            matches.append({
                "home_id":   home_id,
                "away_id":   away_id,
                "home_name": name_map.get(home_id, home_id),
                "away_name": name_map.get(away_id, away_id),
                "hg": hg, "ag": ag,
                "winner_id":   winner_id,
                "winner_name": name_map.get(winner_id, winner_id),
            })
            next_round.append(winner_id)

        rounds.append({"round": round_name, "matches": matches})
        remaining = next_round

    return rounds


# ── full tournament run ───────────────────────────────────────────────────────

def run_tournament(data: dict, all_feats: dict, mode: str) -> dict:
    group_results = simulate_group_stage(data, all_feats, mode)
    qualifiers = get_qualifiers(group_results)
    bracket = simulate_knockout(qualifiers, all_feats, mode)
    champion_id = bracket[-1]["matches"][0]["winner_id"] if bracket else None
    champion_name = data["teams"].get(champion_id, {}).get("name", "?") if champion_id else "?"
    return {
        "group_results": group_results,
        "qualifiers":    qualifiers,
        "bracket":       bracket,
        "champion_id":   champion_id,
        "champion_name": champion_name,
    }


def run_monte_carlo(data: dict, all_feats: dict, n: int) -> dict:
    """Run n simulations. Returns win counts and round-reach counts per team."""
    wins: dict[str, int] = defaultdict(int)
    final_appearances: dict[str, int] = defaultdict(int)
    semis: dict[str, int] = defaultdict(int)
    quarters: dict[str, int] = defaultdict(int)

    progress = st.progress(0, text="Running simulations…")
    for i in range(n):
        result = run_tournament(data, all_feats, mode="monte_carlo")
        wins[result["champion_id"]] += 1

        bracket = result["bracket"]
        round_names = [r["round"] for r in bracket]

        for r_idx, r in enumerate(bracket):
            for m in r["matches"]:
                if r["round"] == "Final":
                    final_appearances[m["home_id"]] += 1
                    final_appearances[m["away_id"]] += 1
                elif r["round"] == "Semi-final":
                    semis[m["home_id"]] += 1
                    semis[m["away_id"]] += 1
                elif r["round"] == "Quarter-final":
                    quarters[m["home_id"]] += 1
                    quarters[m["away_id"]] += 1

        if i % max(1, n // 50) == 0:
            progress.progress((i + 1) / n, text=f"Running simulations… {i+1}/{n}")

    progress.empty()
    return {
        "wins":    wins,
        "finals":  final_appearances,
        "semis":   semis,
        "quarters":quarters,
        "n":       n,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Tournament Simulator · WC 2026", page_icon="🏆", layout="wide")
apply_styles()
top_nav()

st.markdown("""
<h1>🏆 Tournament Simulator</h1>
<p class="page-sub">Simulates the rest of WC 2026 using current results + ML model · Single run or Monte Carlo</p>
""", unsafe_allow_html=True)

if not is_trained():
    st.warning("Model not trained — run `python -m scripts.train_model` first.", icon="⚠️")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Simulation Settings")
    sim_mode = st.radio("Mode", ["Single Run", "Monte Carlo"], index=0)
    n_sims = 500
    if sim_mode == "Monte Carlo":
        n_sims = st.slider("Simulations", min_value=100, max_value=2000, value=500, step=100)
    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)
    st.caption(
        "**Single Run** — always picks the most likely outcome.\n\n"
        "**Monte Carlo** — samples probabilities N times and shows each team's win %."
    )

data = load_sim_data()
if not data["teams"]:
    st.error("No data — run `python -m scripts.seed_db` first.")
    st.stop()

if "sim_result" not in st.session_state:
    st.session_state.sim_result = None
if "mc_result" not in st.session_state:
    st.session_state.mc_result = None

if run_btn:
    with st.spinner("Building team features…"):
        all_feats = build_all_features(data)
    if sim_mode == "Single Run":
        with st.spinner("Simulating tournament…"):
            st.session_state.sim_result = run_tournament(data, all_feats, mode="deterministic")
        st.session_state.mc_result = None
    else:
        st.session_state.sim_result = run_tournament(data, all_feats, mode="monte_carlo")
        st.session_state.mc_result = run_monte_carlo(data, all_feats, n=n_sims)

result = st.session_state.sim_result

if result is None:
    st.markdown("""
<div style="background:linear-gradient(135deg,#0f0f24,#141432);border:1px solid #1e1e3e;
            border-radius:16px;padding:48px;text-align:center;margin-top:32px">
  <div style="font-size:3rem;margin-bottom:12px">🏆</div>
  <div style="font-size:1.1rem;font-weight:700;color:#cbd5e1;margin-bottom:8px">
    Ready to simulate
  </div>
  <div style="font-size:.85rem;color:#475569">
    Choose a mode in the sidebar and click <strong style="color:#38bdf8">▶ Run Simulation</strong>
  </div>
</div>""", unsafe_allow_html=True)
    st.stop()

# ── Champion banner ───────────────────────────────────────────────────────────
st.markdown(champion_banner(result["champion_name"]), unsafe_allow_html=True)

# ── Monte Carlo probabilities ─────────────────────────────────────────────────
mc = st.session_state.mc_result
if mc:
    st.markdown('<div class="section-label">Tournament Win Probabilities</div>', unsafe_allow_html=True)
    st.caption(f"Based on {mc['n']:,} simulated tournaments")

    prob_rows = []
    for tid, w in sorted(mc["wins"].items(), key=lambda x: -x[1]):
        name = data["teams"].get(tid, {}).get("name", tid)
        prob_rows.append({
            "Team":     name,
            "Win %":    w / mc["n"],
            "Final %":  mc["finals"].get(tid, 0) / mc["n"],
            "Semi %":   mc["semis"].get(tid, 0) / mc["n"],
            "QF %":     mc["quarters"].get(tid, 0) / mc["n"],
        })

    prob_df = pd.DataFrame(prob_rows).head(20)
    max_win = max(prob_df["Win %"]) if not prob_df.empty else 0.01

    # Horizontal bar chart with flags
    y_labels = [f"{flag(r['Team'])} {r['Team']}" for _, r in prob_df.iterrows()]
    fig = go.Figure(go.Bar(
        x=prob_df["Win %"] * 100,
        y=y_labels,
        orientation="h",
        marker=dict(
            color=prob_df["Win %"] * 100,
            colorscale=[[0,"#1e3a5f"],[0.5,"#0ea5e9"],[1,"#f59e0b"]],
            showscale=False,
        ),
        text=[f"  {v:.1%}" for v in prob_df["Win %"]],
        textposition="outside",
        textfont=dict(color="#cbd5e1", size=11),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, max_win * 135], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#cbd5e1")),
        height=max(400, len(prob_df) * 28),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=80, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full probability table"):
        disp = prob_df.copy()
        for c in ["Win %", "Final %", "Semi %", "QF %"]:
            disp[c] = disp[c].apply(lambda v: f"{v:.1%}")
        st.dataframe(disp, use_container_width=True, hide_index=True)

    st.divider()

# ── Knockout bracket ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Knockout Bracket</div>', unsafe_allow_html=True)

bracket = result["bracket"]
if bracket:
    cols = st.columns(len(bracket))
    for col, rnd in zip(cols, bracket):
        with col:
            st.markdown(
                f'<div style="font-size:.68rem;font-weight:700;color:#38bdf8;text-transform:uppercase;'
                f'letter-spacing:.1em;margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid #1e1e3e">'
                f'{rnd["round"]}</div>',
                unsafe_allow_html=True,
            )
            for m in rnd["matches"]:
                st.markdown(
                    bracket_card(m["home_name"], m["away_name"], m["hg"], m["ag"],
                                 m["winner_id"], m["home_id"]),
                    unsafe_allow_html=True,
                )

st.divider()

# ── Qualifiers ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">32 Teams that Qualify</div>', unsafe_allow_html=True)

qual_rows = ""
for q in result["qualifiers"]:
    pos_color = "#22c55e" if q["position"] < 2 else "#f59e0b"
    f_ = flag(q["team_name"])
    qual_rows += f"""
<div style="display:flex;align-items:center;gap:10px;padding:8px 14px;
            border-bottom:1px solid #0f0f22">
  <span style="font-size:.72rem;font-weight:800;color:#334155;min-width:24px">{q['seed']}</span>
  <span style="font-size:1rem">{f_}</span>
  <span style="font-size:.88rem;font-weight:700;color:#e2e8f0;flex:1">{q['team_name']}</span>
  <span style="font-size:.65rem;color:{pos_color};background:{pos_color}22;
               padding:2px 8px;border-radius:10px;border:1px solid {pos_color}44">
    {q['qualified_as'].split('—')[1].strip() if '—' in q['qualified_as'] else q['qualified_as']}
  </span>
</div>"""

c1, c2 = st.columns(2)
items = result["qualifiers"]
mid = len(items) // 2
with c1:
    st.markdown(f'<div style="background:#0f0f24;border:1px solid #1e1e3e;border-radius:14px;overflow:hidden">'
                + "".join(
                    f'<div style="display:flex;align-items:center;gap:10px;padding:8px 14px;border-bottom:1px solid #0f0f22">'
                    f'<span style="font-size:.72rem;font-weight:800;color:#334155;min-width:24px">{q["seed"]}</span>'
                    f'<span>{flag(q["team_name"])}</span>'
                    f'<span style="font-size:.88rem;font-weight:700;color:#e2e8f0;flex:1">{q["team_name"]}</span>'
                    f'<span style="font-size:.65rem;color:#22c55e;background:#22c55e22;padding:2px 8px;border-radius:10px">'
                    f'{q["qualified_as"].split("—")[1].strip() if "—" in q["qualified_as"] else q["qualified_as"]}</span>'
                    f'</div>'
                    for q in items[:mid]
                ) + "</div>",
                unsafe_allow_html=True)
with c2:
    st.markdown(f'<div style="background:#0f0f24;border:1px solid #1e1e3e;border-radius:14px;overflow:hidden">'
                + "".join(
                    f'<div style="display:flex;align-items:center;gap:10px;padding:8px 14px;border-bottom:1px solid #0f0f22">'
                    f'<span style="font-size:.72rem;font-weight:800;color:#334155;min-width:24px">{q["seed"]}</span>'
                    f'<span>{flag(q["team_name"])}</span>'
                    f'<span style="font-size:.88rem;font-weight:700;color:#e2e8f0;flex:1">{q["team_name"]}</span>'
                    f'<span style="font-size:.65rem;color:#22c55e;background:#22c55e22;padding:2px 8px;border-radius:10px">'
                    f'{q["qualified_as"].split("—")[1].strip() if "—" in q["qualified_as"] else q["qualified_as"]}</span>'
                    f'</div>'
                    for q in items[mid:]
                ) + "</div>",
                unsafe_allow_html=True)

st.divider()

# ── Simulated group standings ─────────────────────────────────────────────────
st.markdown('<div class="section-label">Simulated Group Standings</div>', unsafe_allow_html=True)
st.caption("✅ Actual results preserved · 🔮 Remaining matches simulated")

group_results = result["group_results"]
groups = sorted(k for k in group_results if k != "_match_log")

with st.expander("Match Log"):
    logs = group_results.get("_match_log", [])
    if logs:
        log_df = pd.DataFrame(logs)
        log_df["Score"] = log_df.apply(lambda r: f"{r['hg']}–{r['ag']}", axis=1)
        log_df["Type"]  = log_df["simulated"].apply(lambda v: "🔮 Sim" if v else "✅ Real")
        st.dataframe(
            log_df[["group","home","Score","away","Type"]].rename(
                columns={"group":"Grp","home":"Home","away":"Away"}),
            use_container_width=True, hide_index=True,
        )

tab_cols = st.tabs([f"Group {g}" for g in groups])
for tab, g in zip(tab_cols, groups):
    with tab:
        rows = [
            {"name": s["name"], "P": s["P"], "W": s["W"], "D": s["D"], "L": s["L"],
             "GF": s["GF"], "GA": s["GA"], "GD": s["GD"], "Pts": s["Pts"]}
            for tid, s in group_results[g].items()
        ]
        st.markdown(standings_table(rows), unsafe_allow_html=True)
