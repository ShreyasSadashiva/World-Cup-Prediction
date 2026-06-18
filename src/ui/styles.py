"""
Shared UI design system — CSS injection + HTML component helpers.
Import apply_styles() at the top of every Streamlit page.
"""

from typing import Optional

import streamlit as st

# ── Country flag emoji map ─────────────────────────────────────────────────
FLAGS: dict[str, str] = {
    "Algeria": "🇩🇿", "Argentina": "🇦🇷", "Australia": "🇦🇺", "Austria": "🇦🇹",
    "Belgium": "🇧🇪", "Bolivia": "🇧🇴", "Bosnia-Herzegovina": "🇧🇦",
    "Brazil": "🇧🇷", "Cameroon": "🇨🇲", "Canada": "🇨🇦", "Cape Verde Islands": "🇨🇻",
    "Chile": "🇨🇱", "Colombia": "🇨🇴", "Costa Rica": "🇨🇷", "Croatia": "🇭🇷",
    "Curaçao": "🇨🇼", "Czech Republic": "🇨🇿", "DR Congo": "🇨🇩",
    "Denmark": "🇩🇰", "Ecuador": "🇪🇨", "Egypt": "🇪🇬", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "France": "🇫🇷", "Germany": "🇩🇪", "Ghana": "🇬🇭", "Haiti": "🇭🇹",
    "Honduras": "🇭🇳", "Hungary": "🇭🇺", "Iran": "🇮🇷", "Iraq": "🇮🇶",
    "Ivory Coast": "🇨🇮", "Jamaica": "🇯🇲", "Japan": "🇯🇵", "Jordan": "🇯🇴",
    "Mexico": "🇲🇽", "Morocco": "🇲🇦", "Netherlands": "🇳🇱", "New Zealand": "🇳🇿",
    "Nigeria": "🇳🇬", "Norway": "🇳🇴", "Panama": "🇵🇦", "Paraguay": "🇵🇾",
    "Peru": "🇵🇪", "Poland": "🇵🇱", "Portugal": "🇵🇹", "Qatar": "🇶🇦",
    "Romania": "🇷🇴", "Saudi Arabia": "🇸🇦", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Senegal": "🇸🇳", "Serbia": "🇷🇸", "South Africa": "🇿🇦", "South Korea": "🇰🇷",
    "Spain": "🇪🇸", "Sweden": "🇸🇪", "Switzerland": "🇨🇭", "Tunisia": "🇹🇳",
    "Turkey": "🇹🇷", "Ukraine": "🇺🇦", "United States": "🇺🇸", "Uruguay": "🇺🇾",
    "Uzbekistan": "🇺🇿", "Venezuela": "🇻🇪",
}

def flag(team: str) -> str:
    return FLAGS.get(team, "🏳️")


# ── CSS ────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── BASE ─────────────────────────────────────────────────────────── */
.stApp, [data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg,#070714 0%,#080820 60%,#0a0818 100%) !important;
    font-family:'Inter','Segoe UI',sans-serif !important;
}
#MainMenu,footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent !important;border:none !important}
.main .block-container{padding:0.5rem 2rem 4rem;max-width:1380px}

/* ── HIDE SIDEBAR AUTO-NAV ───────────────────────────────────────── */
[data-testid="stSidebarNav"]{display:none !important}

/* ── TOP NAV ──────────────────────────────────────────────────────── */
.wc-topnav{
    background:linear-gradient(90deg,#060614,#0a0820);
    border-bottom:1px solid #1a1a3a;
    padding:0 24px;
    display:flex;align-items:center;
    gap:4px;margin-bottom:24px;
    border-radius:0 0 12px 12px;
}
.wc-topnav .nav-brand{
    font-size:1rem;font-weight:900;
    background:linear-gradient(135deg,#38bdf8,#818cf8);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;
    padding:14px 16px 14px 0;
    margin-right:8px;white-space:nowrap;
    border-right:1px solid #1a1a3a;
}
/* Style st.page_link elements inside the nav */
.wc-topnav a[data-testid="stPageLink-NavLink"],
.wc-topnav a[data-testid="stPageLink-NavLink"]:visited{
    color:#64748b !important;font-size:.78rem !important;
    font-weight:600 !important;text-decoration:none !important;
    padding:8px 14px !important;border-radius:8px !important;
    transition:all .2s !important;white-space:nowrap !important;
    display:flex !important;align-items:center !important;gap:5px !important;
}
.wc-topnav a[data-testid="stPageLink-NavLink"]:hover{
    color:#e2e8f0 !important;background:rgba(56,189,248,.08) !important;
}
.wc-topnav a[data-testid="stPageLink-NavLink"][aria-current="page"]{
    color:#38bdf8 !important;background:rgba(56,189,248,.1) !important;
    border-bottom:2px solid #38bdf8 !important;border-radius:8px 8px 0 0 !important;
}
/* Columns inside nav: remove gap/padding */
.wc-topnav [data-testid="stHorizontalBlock"]{gap:0 !important;align-items:center !important}
.wc-topnav [data-testid="stColumn"]{padding:0 !important;min-width:0 !important}
.wc-topnav [data-testid="stColumn"] [data-testid="stPageLink"]{margin:0 !important}

/* ── SIDEBAR ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#060616,#0a0820) !important;
    border-right:1px solid #1a1a3a !important;
}
[data-testid="stSidebar"] *{color:#cbd5e1 !important}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{color:#f8fafc !important}
[data-testid="stSidebar"] hr{border-color:#1a1a3a !important}

/* ── TYPOGRAPHY ──────────────────────────────────────────────────── */
h1{
    font-size:2.4rem !important;font-weight:900 !important;
    letter-spacing:-0.03em !important;line-height:1.1 !important;
    background:linear-gradient(135deg,#38bdf8 0%,#818cf8 55%,#c084fc 100%);
    -webkit-background-clip:text !important;-webkit-text-fill-color:transparent !important;
    background-clip:text !important;margin-bottom:2px !important;
}
h2{
    font-size:0.75rem !important;font-weight:700 !important;
    color:#38bdf8 !important;text-transform:uppercase !important;
    letter-spacing:0.12em !important;margin:2rem 0 0.8rem !important;
    padding-left:10px;border-left:3px solid #38bdf8;
    -webkit-text-fill-color:#38bdf8 !important;
}
h3{color:#cbd5e1 !important;font-weight:700 !important;font-size:1rem !important}
p{color:#94a3b8 !important}
.stCaption p,[data-testid="stCaptionContainer"] p{color:#475569 !important;font-size:0.78rem !important}

/* ── BUTTONS ──────────────────────────────────────────────────────── */
.stButton>button{
    background:linear-gradient(135deg,#0ea5e9,#8b5cf6) !important;
    border:none !important;color:#fff !important;
    font-weight:700 !important;border-radius:10px !important;
    font-size:0.9rem !important;letter-spacing:0.02em !important;
    transition:all .2s !important;
}
.stButton>button:hover{
    opacity:.88 !important;transform:translateY(-1px) !important;
    box-shadow:0 6px 24px rgba(14,165,233,.4) !important;
}
.stButton>button:active{transform:none !important}

/* ── TABS ─────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{
    background:transparent !important;
    border-bottom:1px solid #1e1e3e !important;gap:2px;
}
.stTabs [data-baseweb="tab"]{
    background:transparent !important;color:#475569 !important;
    font-weight:600 !important;font-size:0.82rem !important;
    padding:8px 18px !important;border-radius:8px 8px 0 0 !important;
    text-transform:uppercase;letter-spacing:0.06em;
}
.stTabs [aria-selected="true"]{
    background:rgba(56,189,248,.08) !important;color:#38bdf8 !important;
    border-bottom:2px solid #38bdf8 !important;
}

/* ── SELECT / INPUT ───────────────────────────────────────────────── */
[data-testid="stSelectbox"]>div>div{
    background:#0f0f26 !important;border:1px solid #1e1e3e !important;
    color:#e2e8f0 !important;border-radius:10px !important;
}
[data-testid="stTextInput"]>div>div>input{
    background:#0f0f26 !important;border:1px solid #1e1e3e !important;
    color:#e2e8f0 !important;border-radius:10px !important;
}

/* ── METRICS ──────────────────────────────────────────────────────── */
[data-testid="metric-container"]{
    background:linear-gradient(135deg,#0f0f24,#141430) !important;
    border:1px solid #1e1e3e !important;border-radius:14px !important;
    padding:18px !important;
}
[data-testid="stMetricValue"]{color:#f8fafc !important;font-weight:800 !important}
[data-testid="stMetricLabel"]{
    color:#475569 !important;font-size:0.7rem !important;
    text-transform:uppercase !important;letter-spacing:0.1em !important;
}

/* ── DATAFRAME ───────────────────────────────────────────────────── */
[data-testid="stDataFrame"]{border:1px solid #1e1e3e !important;border-radius:12px !important;overflow:hidden}
[data-testid="stDataFrame"] thead th{background:#0c0c20 !important;color:#475569 !important;font-size:0.72rem !important;letter-spacing:.08em;text-transform:uppercase}
[data-testid="stDataFrame"] tbody td{background:#0f0f24 !important;color:#cbd5e1 !important}
[data-testid="stDataFrame"] tbody tr:hover td{background:#141432 !important}

/* ── RADIO / CHECKBOX ────────────────────────────────────────────── */
.stRadio label p,.stCheckbox label p{color:#94a3b8 !important}
.stRadio [data-testid="stWidgetLabel"] p{color:#cbd5e1 !important;font-weight:600}

/* ── SLIDER ──────────────────────────────────────────────────────── */
.stSlider [data-testid="stWidgetLabel"] p{color:#94a3b8 !important}
.stSlider .rc-slider-track,.stSlider .rc-slider-handle{background:#0ea5e9 !important;border-color:#0ea5e9 !important}

/* ── PROGRESS ────────────────────────────────────────────────────── */
.stProgress>div>div{background:linear-gradient(90deg,#0ea5e9,#8b5cf6) !important}

/* ── DIVIDER ──────────────────────────────────────────────────────── */
hr{border-color:#1a1a3a !important;margin:1.5rem 0 !important}

/* ── ALERTS ───────────────────────────────────────────────────────── */
.stAlert{border-radius:12px !important;border:1px solid #1e1e3e !important}

/* ── EXPANDER ─────────────────────────────────────────────────────── */
[data-testid="stExpander"]{
    background:#0f0f24 !important;border:1px solid #1e1e3e !important;
    border-radius:12px !important;
}
[data-testid="stExpander"] summary{color:#94a3b8 !important}

/* ── CUSTOM COMPONENTS ─────────────────────────────────────────────── */

/* Page subtitle */
.page-sub{color:#475569;font-size:0.82rem;margin-top:-6px;margin-bottom:24px;letter-spacing:.02em}

/* Section label */
.section-label{
    font-size:.68rem;font-weight:700;text-transform:uppercase;
    letter-spacing:.14em;color:#38bdf8;
    border-left:3px solid #38bdf8;padding-left:10px;
    margin:2rem 0 .8rem;
}

/* Match card */
.match-card{
    background:linear-gradient(135deg,#0f0f24 0%,#141432 100%);
    border:1px solid #1e1e3e;border-radius:14px;
    padding:16px 20px;margin:8px 0;
    transition:border-color .2s,box-shadow .2s;
}
.match-card:hover{border-color:#38bdf8;box-shadow:0 4px 24px rgba(56,189,248,.12)}
.mc-meta{font-size:.67rem;color:#334155;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px}
.mc-body{display:flex;align-items:center;justify-content:space-between;gap:12px}
.mc-team{flex:1;font-size:1rem;font-weight:700;color:#e2e8f0}
.mc-team.home{text-align:right}.mc-team.away{text-align:left}
.mc-flag{font-size:1.3em;vertical-align:middle;margin:0 4px}
.mc-score{
    font-size:1.4rem;font-weight:900;color:#f8fafc;
    background:#1a1a36;border-radius:8px;
    padding:6px 18px;min-width:80px;text-align:center;
}
.mc-score.live{color:#ef4444;background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.3);animation:pulse 1.5s infinite}
.mc-score.upcoming{font-size:.9rem;color:#38bdf8;background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.25);padding:8px 14px}
.mc-foot{font-size:.67rem;color:#334155;text-align:center;margin-top:8px}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.65}}

/* Team vs banner */
.vs-banner{
    background:linear-gradient(135deg,#0f0f24,#160d2e,#0d162e);
    border:1px solid #1e1e3e;border-radius:18px;padding:32px 40px;margin:16px 0;
}
.vs-block{text-align:center}
.vs-flag{font-size:4rem;line-height:1;display:block;margin-bottom:8px}
.vs-name{font-size:1.5rem;font-weight:900;color:#f8fafc;margin-bottom:4px}
.vs-group{font-size:.7rem;color:#334155;text-transform:uppercase;letter-spacing:.12em}
.vs-divider{font-size:1.5rem;font-weight:900;color:#1e1e3e;padding:0 24px;align-self:center}

/* Probability bars */
.prob-container{margin:4px 0}
.prob-row{margin:12px 0}
.prob-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px}
.prob-team{font-size:.82rem;font-weight:600;color:#cbd5e1}
.prob-pct{font-size:.82rem;font-weight:700;color:#f8fafc}
.prob-track{background:#1a1a36;border-radius:6px;height:10px;overflow:hidden}
.prob-fill{height:10px;border-radius:6px;transition:width .6s cubic-bezier(.4,0,.2,1)}
.prob-fill.home{background:linear-gradient(90deg,#22c55e,#16a34a)}
.prob-fill.draw{background:linear-gradient(90deg,#f59e0b,#d97706)}
.prob-fill.away{background:linear-gradient(90deg,#3b82f6,#2563eb)}
.prob-label-mid{font-size:.72rem;color:#334155;text-align:center;margin-top:2px}

/* Form badges */
.form-strip{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:12px 0}
.fb{
    width:30px;height:30px;border-radius:7px;
    display:inline-flex;align-items:center;justify-content:center;
    font-weight:800;font-size:.82rem;
}
.fb-W{background:#15803d;color:#86efac;box-shadow:0 2px 8px rgba(21,128,61,.4)}
.fb-D{background:#1e3a5f;color:#93c5fd;box-shadow:0 2px 8px rgba(30,58,95,.4)}
.fb-L{background:#7f1d1d;color:#fca5a5;box-shadow:0 2px 8px rgba(127,29,29,.4)}

/* Stat cards grid */
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:14px 0}
.stat-card{
    background:linear-gradient(135deg,#0f0f24,#141432);
    border:1px solid #1e1e3e;border-radius:12px;
    padding:16px 14px;text-align:center;
}
.sc-val{font-size:1.8rem;font-weight:900;color:#f8fafc;line-height:1}
.sc-lbl{font-size:.67rem;color:#475569;text-transform:uppercase;letter-spacing:.12em;margin-top:5px}
.sc-accent{color:#38bdf8}

/* Standings table */
.st-table{width:100%;border-collapse:collapse}
.st-table th{
    background:#0a0a1e;color:#334155;font-size:.65rem;
    font-weight:700;text-transform:uppercase;letter-spacing:.1em;
    padding:8px 12px;border-bottom:1px solid #1a1a3a;text-align:center;
}
.st-table th:first-child{text-align:left;padding-left:16px}
.st-table td{
    padding:11px 12px;border-bottom:1px solid #0f0f22;
    color:#cbd5e1;font-size:.88rem;text-align:center;
}
.st-table td:first-child{text-align:left;padding-left:16px;font-weight:600;color:#e2e8f0}
.st-table tr:hover td{background:rgba(56,189,248,.04)}
.st-table .row-q1{border-left:3px solid #22c55e}
.st-table .row-q2{border-left:3px solid #22c55e}
.st-table .row-m3{border-left:3px solid #f59e0b}
.st-table .row-out{border-left:3px solid #1e1e3e}
.st-pts{font-weight:900;color:#f8fafc}
.st-team-cell{display:flex;align-items:center;gap:8px}

/* Champion banner */
.champ-banner{
    background:linear-gradient(135deg,#0d0d22,#170d2e,#0d1a22);
    border:2px solid #f59e0b;border-radius:18px;padding:32px;text-align:center;
    box-shadow:0 0 50px rgba(245,158,11,.2),inset 0 0 50px rgba(245,158,11,.04);
    margin:16px 0;
}
.champ-trophy{font-size:3.5rem;margin-bottom:8px;display:block}
.champ-country{font-size:.7rem;color:#92400e;text-transform:uppercase;letter-spacing:.15em;margin-bottom:6px}
.champ-name{font-size:2.6rem;font-weight:900;color:#fbbf24;line-height:1;margin-bottom:8px}
.champ-flag{font-size:2rem;display:block;margin:4px 0}
.champ-sub{font-size:.75rem;color:#78350f;letter-spacing:.08em;text-transform:uppercase}

/* Bracket card */
.bracket-card{
    background:linear-gradient(135deg,#0f0f24,#141432);
    border:1px solid #1e1e3e;border-radius:10px;
    padding:10px 14px;margin:5px 0;font-size:.82rem;
}
.bracket-card.winner{border-color:#38bdf8;box-shadow:0 2px 12px rgba(56,189,248,.15)}
.bc-team{
    display:flex;align-items:center;justify-content:space-between;
    padding:4px 0;color:#94a3b8;
}
.bc-team.won{color:#f8fafc;font-weight:700}
.bc-goals{
    font-weight:800;font-size:1rem;
    background:#1a1a36;padding:2px 8px;border-radius:5px;color:#e2e8f0;
}
.bc-goals.won{color:#38bdf8}

/* Player leaderboard */
.lb-row{
    display:flex;align-items:center;gap:14px;
    padding:12px 16px;border-bottom:1px solid #0f0f22;
}
.lb-row:hover{background:rgba(56,189,248,.04)}
.lb-rank{font-size:1rem;font-weight:900;color:#334155;min-width:24px}
.lb-rank.top3{color:#f59e0b}
.lb-name{flex:1;font-weight:700;color:#e2e8f0;font-size:.92rem}
.lb-team{font-size:.72rem;color:#475569}
.lb-stat{font-size:1.2rem;font-weight:900;color:#38bdf8;min-width:30px;text-align:right}

/* Group badge chip */
.group-chip{
    display:inline-block;padding:2px 10px;
    background:rgba(56,189,248,.12);color:#38bdf8;
    border:1px solid rgba(56,189,248,.3);border-radius:20px;
    font-size:.68rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
}
.live-chip{
    background:rgba(239,68,68,.12);color:#ef4444;
    border-color:rgba(239,68,68,.3);
}
.done-chip{background:rgba(100,116,139,.1);color:#64748b;border-color:#1e1e3e}

/* Navigation page title */
.nav-title{font-size:.7rem;color:#334155;text-transform:uppercase;letter-spacing:.1em;margin-bottom:20px}
</style>
"""

def apply_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


_NAV_PAGES = [
    ("⚽", "Dashboard",   "app.py"),
    ("🎯", "Predictor",   "pages/1_Match_Predictor.py"),
    ("🤝", "H2H",         "pages/2_Head_to_Head.py"),
    ("📈", "Team Form",   "pages/3_Team_Form.py"),
    ("🗓️", "Fixtures",    "pages/4_WC_Fixtures.py"),
    ("👤", "Players",     "pages/5_Players.py"),
    ("🏆", "Simulator",   "pages/6_Tournament_Simulator.py"),
]


def top_nav() -> None:
    """Render a horizontal top navigation bar using st.page_link."""
    brand = '<div class="nav-brand">⚽ WC 2026</div>'
    # Wrap everything in the wc-topnav div
    st.markdown(f'<div class="wc-topnav">{brand}', unsafe_allow_html=True)
    cols = st.columns(len(_NAV_PAGES))
    for col, (icon, label, path) in zip(cols, _NAV_PAGES):
        with col:
            st.page_link(path, label=f"{icon} {label}", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── HTML component helpers ─────────────────────────────────────────────────

def match_card(
    home: str, away: str,
    home_goals: Optional[int] = None, away_goals: Optional[int] = None,
    date: str = "", group: str = "", stage: str = "", venue: str = "",
    status: str = "SCHEDULED",
) -> str:
    meta_parts = []
    if group: meta_parts.append(f"Group {group}")
    elif stage: meta_parts.append(stage.replace("_", " ").title())
    if date: meta_parts.append(date)
    if venue: meta_parts.append(venue)
    meta = " &nbsp;·&nbsp; ".join(meta_parts)

    hf, af = flag(home), flag(away)

    if status == "FINISHED" and home_goals is not None:
        score_html = f'<div class="mc-score">{home_goals} – {away_goals}</div>'
        foot = '<span class="done-chip" style="font-size:.65rem;padding:2px 8px;border-radius:10px;background:rgba(100,116,139,.1);color:#475569;border:1px solid #1e1e3e">Full Time</span>'
    elif status == "IN_PLAY":
        score_html = f'<div class="mc-score live">🔴 {home_goals} – {away_goals}</div>'
        foot = '<span class="live-chip" style="font-size:.65rem;padding:2px 8px;border-radius:10px;background:rgba(239,68,68,.12);color:#ef4444;border:1px solid rgba(239,68,68,.3)">● Live</span>'
    else:
        score_html = '<div class="mc-score upcoming">vs</div>'
        foot = ""

    return f"""
<div class="match-card">
  <div class="mc-meta">{meta}</div>
  <div class="mc-body">
    <div class="mc-team home"><span class="mc-flag">{hf}</span> {home}</div>
    {score_html}
    <div class="mc-team away">{away} <span class="mc-flag">{af}</span></div>
  </div>
  {f'<div class="mc-foot">{foot}</div>' if foot else ''}
</div>"""


def form_strip(results: list[str]) -> str:
    badges = "".join(f'<span class="fb fb-{r}">{r}</span>' for r in results)
    return f'<div class="form-strip">{badges}</div>'


def stat_grid(items: list[tuple]) -> str:
    """items = [(value, label), ...]"""
    cards = "".join(
        f'<div class="stat-card"><div class="sc-val">{v}</div>'
        f'<div class="sc-lbl">{l}</div></div>'
        for v, l in items
    )
    return f'<div class="stat-grid">{cards}</div>'


def prob_bars(home_win: float, draw: float, away_win: float,
              home_name: str, away_name: str) -> str:
    hw_w = f"{home_win * 100:.0f}%"
    dr_w = f"{draw * 100:.0f}%"
    aw_w = f"{away_win * 100:.0f}%"
    hf, af = flag(home_name), flag(away_name)
    return f"""
<div class="prob-container">
  <div class="prob-row">
    <div class="prob-header">
      <span class="prob-team">{hf} {home_name}</span>
      <span class="prob-pct" style="color:#22c55e">{hw_w}</span>
    </div>
    <div class="prob-track"><div class="prob-fill home" style="width:{hw_w}"></div></div>
  </div>
  <div class="prob-row">
    <div class="prob-header">
      <span class="prob-team">Draw</span>
      <span class="prob-pct" style="color:#f59e0b">{dr_w}</span>
    </div>
    <div class="prob-track"><div class="prob-fill draw" style="width:{dr_w}"></div></div>
  </div>
  <div class="prob-row">
    <div class="prob-header">
      <span class="prob-team">{af} {away_name}</span>
      <span class="prob-pct" style="color:#3b82f6">{aw_w}</span>
    </div>
    <div class="prob-track"><div class="prob-fill away" style="width:{aw_w}"></div></div>
  </div>
</div>"""


def vs_banner(home: str, home_group: str, away: str, away_group: str) -> str:
    hf, af = flag(home), flag(away)
    hg = f"Group {home_group}" if home_group else ""
    ag = f"Group {away_group}" if away_group else ""
    return f"""
<div class="vs-banner">
  <div style="display:flex;align-items:center;justify-content:space-between">
    <div class="vs-block" style="flex:1">
      <span class="vs-flag">{hf}</span>
      <div class="vs-name">{home}</div>
      <div class="vs-group">{hg}</div>
    </div>
    <div class="vs-divider">VS</div>
    <div class="vs-block" style="flex:1">
      <span class="vs-flag">{af}</span>
      <div class="vs-name">{away}</div>
      <div class="vs-group">{ag}</div>
    </div>
  </div>
</div>"""


def champion_banner(name: str, conf: float = 0) -> str:
    f_ = flag(name)
    conf_str = f"Confidence: {conf:.1%}" if conf else "Predicted Champion"
    return f"""
<div class="champ-banner">
  <span class="champ-trophy">🏆</span>
  <div class="champ-country">FIFA World Cup 2026</div>
  <div class="champ-flag">{f_}</div>
  <div class="champ-name">{name}</div>
  <div class="champ-sub">{conf_str}</div>
</div>"""


def standings_table(rows: list[dict], show_group: bool = False) -> str:
    """
    rows: list of dicts with keys: name, P, W, D, L, GF, GA, GD, Pts, position (0-3)
    """
    extra_th = "<th>GRP</th>" if show_group else ""
    header = f"""
<thead><tr>
  <th style="width:32px">#</th><th>Team</th>
  {extra_th}<th>P</th><th>W</th><th>D</th><th>L</th>
  <th>GF</th><th>GA</th><th>GD</th><th>Pts</th>
</tr></thead>"""

    pos_class = ["row-q1", "row-q2", "row-m3", "row-out"]
    body_rows = ""
    for i, r in enumerate(rows):
        pc = pos_class[min(i, 3)]
        f_ = flag(r["name"])
        extra_td = f"<td>{r.get('group','')}</td>" if show_group else ""
        gd_color = "color:#22c55e" if r["GD"] > 0 else ("color:#ef4444" if r["GD"] < 0 else "color:#94a3b8")
        body_rows += f"""
<tr class="{pc}">
  <td style="color:#334155;font-weight:700">{i+1}</td>
  <td><div class="st-team-cell">{f_} {r["name"]}</div></td>
  {extra_td}
  <td>{r["P"]}</td><td>{r["W"]}</td><td>{r["D"]}</td><td>{r["L"]}</td>
  <td>{r["GF"]}</td><td>{r["GA"]}</td>
  <td style="{gd_color}">{r["GD"]:+d}</td>
  <td class="st-pts">{r["Pts"]}</td>
</tr>"""

    return f'<table class="st-table">{header}<tbody>{body_rows}</tbody></table>'


def bracket_card(home: str, away: str, hg: int, ag: int, winner_id: str,
                 home_id: str) -> str:
    home_won = winner_id == home_id
    hf, af = flag(home), flag(away)
    h_cls = "won" if home_won else ""
    a_cls = "won" if not home_won else ""
    hg_cls = "won" if home_won else ""
    ag_cls = "won" if not home_won else ""
    card_cls = "winner" if True else ""
    return f"""
<div class="bracket-card {card_cls}">
  <div class="bc-team {h_cls}">
    <span>{hf} {home}</span>
    <span class="bc-goals {hg_cls}">{hg}</span>
  </div>
  <div class="bc-team {a_cls}">
    <span>{af} {away}</span>
    <span class="bc-goals {ag_cls}">{ag}</span>
  </div>
</div>"""
