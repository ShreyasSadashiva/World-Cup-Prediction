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
st.title("🏆 Tournament Simulator")
st.caption("Simulates the remaining WC 2026 matches using current results + ML model predictions.")

if not is_trained():
    st.warning("Model not trained — run `python -m scripts.train_model` first.", icon="⚠️")

# ── sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Simulation Settings")
    sim_mode = st.radio("Mode", ["Single Run", "Monte Carlo"], index=0)
    n_sims = 500
    if sim_mode == "Monte Carlo":
        n_sims = st.slider("Simulations", min_value=100, max_value=2000, value=500, step=100)
    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)
    st.caption(
        "**Single Run** — always picks the most likely outcome.\n\n"
        "**Monte Carlo** — samples from probability distributions and shows "
        "each team's chance of winning the tournament."
    )

data = load_sim_data()
if not data["teams"]:
    st.error("No data in DB — run `python -m scripts.seed_db` first.")
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
    st.info("Configure settings in the sidebar and click **▶ Run Simulation** to start.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  CHAMPION BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);
                border:2px solid #f4c430;border-radius:12px;
                padding:24px;text-align:center;margin-bottom:24px">
        <div style="font-size:2.8em">🥇</div>
        <div style="color:#f4c430;font-size:2em;font-weight:bold;margin:8px 0">
            {result['champion_name']}
        </div>
        <div style="color:#aaa;font-size:1em">Predicted WC 2026 Champion</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
#  MONTE CARLO WIN PROBABILITIES
# ══════════════════════════════════════════════════════════════════════════════
mc = st.session_state.mc_result
if mc:
    st.subheader("📊 Tournament Win Probabilities")
    st.caption(f"Based on {mc['n']} simulated tournaments")

    prob_rows = []
    for tid, w in sorted(mc["wins"].items(), key=lambda x: -x[1]):
        name = data["teams"].get(tid, {}).get("name", tid)
        prob_rows.append({
            "Team":          name,
            "Win %":         w / mc["n"],
            "Final %":       mc["finals"].get(tid, 0) / mc["n"],
            "Semi-final %":  mc["semis"].get(tid, 0) / mc["n"],
            "Quarter-final %": mc["quarters"].get(tid, 0) / mc["n"],
        })

    prob_df = pd.DataFrame(prob_rows).head(20)

    fig = go.Figure(go.Bar(
        x=prob_df["Win %"] * 100,
        y=prob_df["Team"],
        orientation="h",
        marker=dict(
            color=prob_df["Win %"] * 100,
            colorscale="YlOrRd",
            showscale=False,
        ),
        text=[f"{v:.1%}" for v in prob_df["Win %"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top 20 teams by tournament win probability",
        xaxis=dict(title="Win probability (%)", range=[0, max(prob_df["Win %"]) * 130]),
        height=520,
        template="plotly_dark",
        margin=dict(l=10, r=80, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed probability table
    with st.expander("Full probability table"):
        display_df = prob_df.copy()
        for col in ["Win %", "Final %", "Semi-final %", "Quarter-final %"]:
            display_df[col] = display_df[col].apply(lambda v: f"{v:.1%}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  KNOCKOUT BRACKET
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🗂️ Knockout Bracket")

bracket = result["bracket"]
if bracket:
    cols = st.columns(len(bracket))
    for col, rnd in zip(cols, bracket):
        with col:
            st.markdown(f"**{rnd['round']}**")
            for m in rnd["matches"]:
                home_won = m["winner_id"] == m["home_id"]
                h_style = "**" if home_won else ""
                a_style = "**" if not home_won else ""
                score = f"{m['hg']}–{m['ag']}"
                st.markdown(
                    f"<div style='background:#1e1e2e;border-radius:6px;"
                    f"padding:8px 10px;margin:4px 0;font-size:0.85em'>"
                    f"{'🏆 ' if home_won else ''}{h_style}{m['home_name']}{h_style}<br>"
                    f"<span style='color:#888;font-size:0.8em'>{score}</span><br>"
                    f"{'🏆 ' if not home_won else ''}{a_style}{m['away_name']}{a_style}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  QUALIFIERS TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🎫 32 Teams that Qualify")

qual_df = pd.DataFrame([
    {"Seed": q["seed"], "Team": q["team_name"], "Qualified as": q["qualified_as"]}
    for q in result["qualifiers"]
])
c1, c2, c3 = st.columns(3)
third = len(qual_df) // 3
for col, chunk in zip([c1, c2, c3], [qual_df.iloc[:third], qual_df.iloc[third:2*third], qual_df.iloc[2*third:]]):
    with col:
        st.dataframe(chunk.reset_index(drop=True), use_container_width=True, hide_index=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  GROUP STAGE STANDINGS
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📋 Simulated Group Standings")
st.caption("✅ Actual result  ·  🔮 Simulated result")

group_results = result["group_results"]
groups = sorted(k for k in group_results if k != "_match_log")

# Show match log
with st.expander("Match Log (all group stage results)"):
    logs = group_results.get("_match_log", [])
    if logs:
        log_df = pd.DataFrame(logs)
        log_df["Score"] = log_df.apply(lambda r: f"{r['hg']}–{r['ag']}", axis=1)
        log_df["Type"] = log_df["simulated"].apply(lambda v: "🔮 Simulated" if v else "✅ Actual")
        st.dataframe(
            log_df[["group", "home", "Score", "away", "Type"]].rename(
                columns={"group": "Group", "home": "Home", "away": "Away"}
            ),
            use_container_width=True, hide_index=True,
        )

# Group tabs
n_groups = len(groups)
tab_cols = st.tabs([f"Group {g}" for g in groups])

for tab, g in zip(tab_cols, groups):
    with tab:
        rows = []
        for pos, (tid, s) in enumerate(group_results[g].items()):
            qualifier_badge = "🟢" if pos < 2 else ("🟡" if pos == 2 else "⚪")
            rows.append({
                "": qualifier_badge,
                "Team": s["name"],
                "P": s["P"], "W": s["W"], "D": s["D"], "L": s["L"],
                "GF": s["GF"], "GA": s["GA"], "GD": s["GD"], "Pts": s["Pts"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("🟢 Qualifies  🟡 May qualify as best 3rd  ⚪ Eliminated")
