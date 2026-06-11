import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, date
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data_prep import (load_data, get_team_features, compute_global_averages,
                       build_ml_dataset, build_elo_timeline,
                       get_h2h_advantage, get_continental_form, detect_confederation,
                       CONF_TOURNAMENTS, load_fifa_rankings, get_fifa_points,
                       FIFA_NAME_MAP)
from predictor import full_predict, estimate_phi, fit_elo_parameters
import predictor as _predictor
from simulator import simulate_tournament, GROUPS
from ml_model import train_model, load_model, predict_ml
import friends as _friends

st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Inter:wght@400;500;600;700&display=swap');

    [data-testid="stSidebar"]        { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }

    /* ── Base ── */
    .stApp {
        background: #050d1f;
        background-image:
            radial-gradient(ellipse at 50% -10%, rgba(0,198,255,0.07) 0%, transparent 55%),
            radial-gradient(ellipse at 95% 90%,  rgba(59,130,246,0.04) 0%, transparent 45%);
        font-family: 'Inter', sans-serif;
    }
    .main .block-container { padding: 0.5rem 2rem 2rem; max-width: 100%; }

    /* ── Typography ── */
    h1 { font-family:'Orbitron',sans-serif !important; color:#e2e8f0 !important;
         letter-spacing:0.04em; font-weight:700 !important; }
    h2, h3, h4 { font-family:'Inter',sans-serif !important; color:#94a3b8 !important; font-weight:600 !important; }
    p, li { color:#cbd5e1; }

    /* ── Glass metric cards with shimmer ── */
    .metric-card {
        background: linear-gradient(135deg, #0b1628 0%, #0d1f38 100%);
        border: 1px solid rgba(0,198,255,0.14);
        border-radius: 12px;
        padding: 18px 14px;
        text-align: center;
        margin: 4px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.55);
        transition: transform 0.22s, box-shadow 0.22s, border-color 0.22s;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: -100%; width: 60%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0,198,255,0.07), transparent);
        transition: left 0.55s ease;
        pointer-events: none;
    }
    .metric-card:hover { transform: translateY(-2px);
        box-shadow: 0 10px 36px rgba(0,0,0,0.65), 0 0 24px rgba(0,198,255,0.1);
        border-color: rgba(0,198,255,0.38); }
    .metric-card:hover::before { left: 160%; }
    .metric-card h3 { font-family:'Orbitron',sans-serif; font-size:1.9rem;
                      margin:0; font-weight:700; color:white !important; }
    .metric-card p  { color:#475569; margin:0; font-size:0.75rem;
                      text-transform:uppercase; letter-spacing:0.09em; margin-top:4px; }

    /* ── Match cards ── */
    .match-card {
        background: linear-gradient(135deg, #0b1628, #0d1f38);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        position: relative;
        overflow: hidden;
        transition: border-color 0.22s, box-shadow 0.22s;
    }
    .match-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #1d4ed8, #00c6ff, #1d4ed8);
        opacity: 0;
        transition: opacity 0.22s;
    }
    .match-card:hover { border-color: rgba(0,198,255,0.28);
        box-shadow: 0 4px 24px rgba(0,198,255,0.08); }
    .match-card:hover::after { opacity: 1; }

    /* ── Prob bar animation ── */
    @keyframes growBar {
        from { width: 0 !important; }
        to   { width: var(--w); }
    }
    .prob-fill {
        width: var(--w);
        animation: growBar 0.7s cubic-bezier(0.4,0,0.2,1) both;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.88rem; color: white;
        overflow: hidden; white-space: nowrap; gap: 5px;
    }

    /* ── Pulsing live dot ── */
    @keyframes livepulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(0,198,255,0.8); }
        60%       { box-shadow: 0 0 0 6px rgba(0,198,255,0); }
    }
    .live-dot {
        display: inline-block; width: 8px; height: 8px;
        background: #00c6ff; border-radius: 50%;
        animation: livepulse 1.8s ease-in-out infinite;
        vertical-align: middle; margin-right: 5px;
    }

    /* ── Info strip ── */
    .info-strip {
        background: #0b1628;
        border-bottom: 1px solid rgba(0,198,255,0.12);
        padding: 9px 20px;
        display: flex; gap: 24px; align-items: center;
        font-size: 0.78rem; flex-wrap: wrap;
        margin-bottom: 6px;
    }

    /* ── Tab navbar ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #0b1628;
        border-bottom: 1px solid rgba(0,198,255,0.1);
        padding: 0 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px; padding: 0 20px;
        color: #475569; font-size: 0.85rem; font-weight: 500;
        background: transparent; border-bottom: 2px solid transparent;
        border-radius: 0; letter-spacing: 0.02em;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00c6ff;
        background: rgba(0,198,255,0.05);
    }
    .stTabs [aria-selected="true"] {
        color: #00c6ff !important;
        background: rgba(0,198,255,0.07);
        border-bottom: 2px solid #00c6ff !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-panel"]     { padding: 20px 0; }

    /* ── Streamlit overrides ── */
    [data-testid="stMetricValue"] { font-family:'Orbitron',sans-serif !important; color:#00c6ff !important; }
    [data-testid="stMetricLabel"] { color:#475569 !important; font-size:0.75rem !important;
                                    text-transform:uppercase; letter-spacing:0.08em; }
    .stSelectbox label, .stCheckbox label { color:#00c6ff !important; font-size:0.8rem !important;
                                            text-transform:uppercase; letter-spacing:0.06em; }
    .stNumberInput label { color:#00c6ff !important; font-size:0.8rem !important; text-transform:uppercase; }
    .stDateInput label   { color:#00c6ff !important; font-size:0.8rem !important; text-transform:uppercase; }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0369a1, #0ea5e9);
        color: white; font-weight: 600; border: none; border-radius: 8px;
        font-family: 'Orbitron', sans-serif; font-size: 0.8rem; letter-spacing: 0.06em;
    }
    .stButton > button[kind="primary"]:hover { opacity: 0.88; }
    .stButton > button {
        background: #0b1628; color: #00c6ff;
        border: 1px solid rgba(0,198,255,0.25); border-radius: 8px;
    }

    /* ── Misc ── */
    hr { border-color: rgba(0,198,255,0.08) !important; }
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    .streamlit-expanderHeader  { color: #00c6ff !important; font-size: 0.85rem !important; }
    [data-testid="stInfo"]    { background: rgba(0,198,255,0.06); border-color: rgba(0,198,255,0.3); }
    [data-testid="stSuccess"] { background: rgba(74,222,128,0.06); border-color: rgba(74,222,128,0.3); }
    [data-testid="stWarning"] { background: rgba(251,191,36,0.06); border-color: rgba(251,191,36,0.3); }
</style>
""", unsafe_allow_html=True)

# ─── Data & model ─────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_model_and_elo(df):
    _, current_elo = build_elo_timeline(df[df['home_score'].notna()])
    fifa_df = load_fifa_rankings()
    df_feat = build_ml_dataset(df, min_year=2005)
    model = load_model()
    acc = None
    if model is None:
        with st.spinner("Training ML model... (~40 seconds, one-time only)"):
            model, acc, _ = train_model(df_feat)
    return model, current_elo, fifa_df, acc, df_feat

df, shootouts = get_data()
model, current_elo, fifa_df, _model_acc, _df_feat = get_model_and_elo(df)

_goals = np.concatenate([
    df.loc[df['home_score'].notna(), 'home_score'].values,
    df.loc[df['home_score'].notna(), 'away_score'].values,
])
_predictor.PHI = estimate_phi(_goals)


def _build_elo_fitting_records(df_feat, model, avg_h, avg_a, n=3000):
    from ml_model import add_derived_features, FEATURE_COLS
    from predictor import (nb_score_matrix, poisson_probs, disagreement_weight,
                           blend_probs, entropy_injection, LAMBDA_FLOOR_GENERAL)
    sample = df_feat.sort_values('date').tail(n).copy()
    sample_aug = add_derived_features(sample)
    proba  = model.predict_proba(sample_aug[FEATURE_COLS])
    cls_names = ['away_win', 'draw', 'home_win']
    cls_order = [list(model.classes_).index(c) for c in [0, 1, 2]]
    result_to_outcome = {0: 'away_win', 1: 'draw', 2: 'home_win'}
    records = []
    for i, (_, row) in enumerate(sample_aug.iterrows()):
        try:
            p_arr = proba[i]
            p_ml  = {cls_names[j]: float(p_arr[cls_order[j]]) * 100 for j in range(3)}
            h_scr = max(float(row.get('h_scored',   avg_h)), 0.1)
            h_con = max(float(row.get('h_conceded', avg_a)), 0.1)
            a_scr = max(float(row.get('a_scored',   avg_a)), 0.1)
            a_con = max(float(row.get('a_conceded', avg_h)), 0.1)
            neutral  = bool(row.get('is_neutral', 0))
            home_adv = 1.0 if neutral else 1.1
            lh = float(np.clip(h_scr * a_con / max(avg_h, 0.1) * home_adv, LAMBDA_FLOOR_GENERAL, 5.5))
            la = float(np.clip(a_scr * h_con / max(avg_a, 0.1),             LAMBDA_FLOOR_GENERAL, 5.5))
            sm = nb_score_matrix(lh, la, phi=_predictor.PHI)
            p_poisson = poisson_probs(sm)
            w, d = disagreement_weight(p_poisson, p_ml)
            p_blend = blend_probs(p_poisson, p_ml, w)
            p_mix, _ = entropy_injection(p_blend, d)
            records.append({'elo_diff': float(row.get('elo_diff', 0)),
                            'outcome':  result_to_outcome[int(row['result'])],
                            'p_mix':    p_mix})
        except Exception:
            continue
    return records


_avg_h, _avg_a = compute_global_averages(df, pd.Timestamp('2024-01-01'))
_elo_records   = _build_elo_fitting_records(_df_feat, model, _avg_h, _avg_a, n=3000)
_k_hat, _alpha_hat = fit_elo_parameters(_elo_records)
_predictor.K_ELO      = _k_hat
_predictor.ALPHA_AWAY = _alpha_hat

WC2026 = df[
    (df['tournament'] == 'FIFA World Cup') &
    (df['date'].dt.year == 2026)
].copy().reset_index(drop=True)

def _validate_fifa_name_map(groups, fifa_df):
    if fifa_df is None:
        return
    ranked_nations = set(fifa_df['country_full'].unique())
    missing = [f"{t} → '{FIFA_NAME_MAP.get(t,t)}'"
               for teams in groups.values() for t in teams
               if FIFA_NAME_MAP.get(t, t) not in ranked_nations]
    if missing:
        st.warning("⚠️ No FIFA ranking data for: " + ", ".join(missing))

_validate_fifa_name_map(GROUPS, fifa_df)

ALL_WC_TEAMS = sorted(pd.concat([WC2026['home_team'], WC2026['away_team']]).unique().tolist())
TODAY = pd.Timestamp(datetime.today().date())

TEAM_FLAGS = {
    'Mexico': '🇲🇽', 'South Africa': '🇿🇦', 'South Korea': '🇰🇷', 'Czech Republic': '🇨🇿',
    'Canada': '🇨🇦', 'Bosnia and Herzegovina': '🇧🇦', 'Switzerland': '🇨🇭', 'Qatar': '🇶🇦',
    'Brazil': '🇧🇷', 'Morocco': '🇲🇦', 'Scotland': '🏴󠁧󠁢󠁳󠁣󠁴󠁿', 'Haiti': '🇭🇹',
    'United States': '🇺🇸', 'Australia': '🇦🇺', 'Turkey': '🇹🇷', 'Paraguay': '🇵🇾',
    'Germany': '🇩🇪', 'Curaçao': '🇨🇼', 'Ivory Coast': '🇨🇮', 'Ecuador': '🇪🇨',
    'Netherlands': '🇳🇱', 'Japan': '🇯🇵', 'Sweden': '🇸🇪', 'Tunisia': '🇹🇳',
    'Belgium': '🇧🇪', 'Iran': '🇮🇷', 'New Zealand': '🇳🇿', 'Egypt': '🇪🇬',
    'Spain': '🇪🇸', 'Cape Verde': '🇨🇻', 'Saudi Arabia': '🇸🇦', 'Uruguay': '🇺🇾',
    'France': '🇫🇷', 'Senegal': '🇸🇳', 'Iraq': '🇮🇶', 'Norway': '🇳🇴',
    'Argentina': '🇦🇷', 'Algeria': '🇩🇿', 'Austria': '🇦🇹', 'Jordan': '🇯🇴',
    'Portugal': '🇵🇹', 'DR Congo': '🇨🇩', 'Uzbekistan': '🇺🇿', 'Colombia': '🇨🇴',
    'England': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'Croatia': '🇭🇷', 'Ghana': '🇬🇭', 'Panama': '🇵🇦',
}

def flag(team):
    return TEAM_FLAGS.get(team, '🏳')

# ─── Helpers ─────────────────────────────────────────────────────────────────
def get_features(team):
    elo = current_elo.get(team, 1500)
    fifa_pts, fifa_rank = get_fifa_points(fifa_df, team, TODAY)
    return get_team_features(df, team, TODAY, n_matches=20, elo=elo,
                             fifa_points=fifa_pts, fifa_rank=fifa_rank, fifa_df=fifa_df)

def run_prediction(home_team, away_team, neutral=True):
    hf = get_features(home_team)
    af = get_features(away_team)
    if hf is None or af is None:
        return None, None
    avg_h, avg_a = compute_global_averages(df, TODAY)
    h2h = get_h2h_advantage(df, home_team, away_team, n=10)
    result = full_predict(hf, af, model, avg_h, avg_a, neutral=neutral, h2h_diff=h2h)
    blend = {'home_win': result['home_win'], 'draw': result['draw'], 'away_win': result['away_win']}
    return result, blend

def get_team_form_string(team, n=10):
    matches = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['home_score'].notna())
    ].sort_values('date').tail(n)
    badges = []
    for _, row in matches.iterrows():
        s, c = (row['home_score'], row['away_score']) if row['home_team'] == team \
               else (row['away_score'], row['home_score'])
        badges.append("W" if s > c else ("D" if s == c else "L"))
    return badges

def render_match_card(home_team, away_team, blend, match_date=None, most_likely=None):
    """Broadcast-style card used for every upcoming fixture."""
    if blend is None:
        return
    hw = blend['home_win']
    d  = blend['draw']
    aw = blend['away_win']
    hf = flag(home_team)
    af = flag(away_team)

    # Pre-compute optional sections — no nested expressions inside the main f-string
    if match_date:
        header_html = (
            '<div style="text-align:center;margin-bottom:14px">'
            '<span style="font-family:\'Orbitron\',sans-serif;font-size:0.65rem;'
            'font-weight:700;color:#00c6ff;letter-spacing:0.18em">'
            '<span class="live-dot"></span>'
            f'FIFA WORLD CUP 2026 &middot; {match_date}'
            '</span></div>'
        )
    else:
        header_html = ''

    if most_likely:
        ml_html = (
            '<div style="color:#475569;font-size:0.7rem;margin-top:10px;letter-spacing:0.06em">'
            f'Most likely: <b style="color:#f59e0b">'
            f'{home_team} {most_likely[0]}&ndash;{most_likely[1]} {away_team}'
            '</b></div>'
        )
    else:
        ml_html = ''

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0b1628 0%,#0f2040 100%);'
        f'border:1px solid rgba(0,198,255,0.18);border-radius:14px;'
        f'padding:22px 24px;margin-bottom:6px;position:relative;overflow:hidden">'
        f'<div style="position:absolute;top:0;left:0;width:40%;height:100%;'
        f'background:linear-gradient(90deg,rgba(29,78,216,0.12),transparent)"></div>'
        f'<div style="position:absolute;top:0;right:0;width:40%;height:100%;'
        f'background:linear-gradient(270deg,rgba(185,28,28,0.12),transparent)"></div>'
        f'<div style="position:absolute;top:0;left:0;right:0;height:2px;'
        f'background:linear-gradient(90deg,transparent,rgba(0,198,255,0.6),transparent)"></div>'
        f'<div style="position:relative;z-index:1">'
        + header_html +
        f'<div style="display:flex;justify-content:space-around;align-items:center">'
        f'<div style="text-align:center;flex:1">'
        f'<div style="font-size:2.2rem;line-height:1;margin-bottom:8px">{hf}</div>'
        f'<div style="font-family:\'Orbitron\',sans-serif;font-size:0.82rem;font-weight:700;'
        f'color:#e2e8f0;letter-spacing:0.03em">{home_team}</div>'
        f'<div style="font-family:\'Orbitron\',sans-serif;font-size:1.55rem;font-weight:900;'
        f'color:#60a5fa;margin-top:6px;text-shadow:0 0 16px rgba(96,165,250,0.35)">{hw}%</div>'
        f'</div>'
        f'<div style="text-align:center;padding:0 20px">'
        f'<div style="font-family:\'Orbitron\',sans-serif;font-size:1.5rem;font-weight:900;'
        f'color:#00c6ff;text-shadow:0 0 24px rgba(0,198,255,0.5);letter-spacing:0.12em">VS</div>'
        f'<div style="color:#334155;font-family:\'Orbitron\',sans-serif;'
        f'font-size:0.6rem;margin-top:8px;letter-spacing:0.1em">DRAW {d}%</div>'
        + ml_html +
        f'</div>'
        f'<div style="text-align:center;flex:1">'
        f'<div style="font-size:2.2rem;line-height:1;margin-bottom:8px">{af}</div>'
        f'<div style="font-family:\'Orbitron\',sans-serif;font-size:0.82rem;font-weight:700;'
        f'color:#e2e8f0;letter-spacing:0.03em">{away_team}</div>'
        f'<div style="font-family:\'Orbitron\',sans-serif;font-size:1.55rem;font-weight:900;'
        f'color:#f87171;margin-top:6px;text-shadow:0 0 16px rgba(248,113,113,0.35)">{aw}%</div>'
        f'</div>'
        f'</div>'   # close flex row
        f'</div>'   # close z-index wrapper
        f'</div>',  # close outer card
        unsafe_allow_html=True
    )
    render_prob_bar(home_team, away_team, blend)


def render_prob_bar(home_team, away_team, blend):
    hw = blend['home_win']; d = blend['draw']; aw = blend['away_win']
    total = hw + d + aw
    if total == 0:
        return
    hw_pct = hw / total * 100; d_pct = d / total * 100; aw_pct = aw / total * 100
    hf = flag(home_team); af = flag(away_team)
    st.markdown(f"""
    <div style="margin:10px 0 18px 0">
        <div style="display:flex;border-radius:8px;overflow:hidden;height:46px;
                    box-shadow:0 4px 20px rgba(0,0,0,0.6)">
            <div class="prob-fill" style="--w:{hw_pct:.1f}%;
                background:linear-gradient(90deg,#1e3a8a,#1d4ed8)">
                <span style="font-size:0.9rem">{hf}</span>
                <span style="font-family:'Orbitron',sans-serif;font-size:0.9rem">{hw}%</span>
            </div>
            <div class="prob-fill" style="--w:{d_pct:.1f}%;background:#0d1830;
                border-left:1px solid rgba(255,255,255,0.05);
                border-right:1px solid rgba(255,255,255,0.05);
                font-family:'Orbitron',sans-serif;font-size:0.8rem;color:#475569">
                {d}%
            </div>
            <div class="prob-fill" style="--w:{aw_pct:.1f}%;
                background:linear-gradient(90deg,#7f1d1d,#b91c1c)">
                <span style="font-family:'Orbitron',sans-serif;font-size:0.9rem">{aw}%</span>
                <span style="font-size:0.9rem">{af}</span>
            </div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:5px;
                    font-size:0.72rem;font-weight:600;letter-spacing:0.06em;padding:0 3px">
            <span style="color:#60a5fa">{home_team[:20]}</span>
            <span style="color:#334155">DRAW</span>
            <span style="color:#f87171">{away_team[:20]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Info strip ───────────────────────────────────────────────────────────────
_acc_str = f"{_model_acc*100:.1f}%" if _model_acc is not None else "~65%"
st.markdown(f"""
<div class="info-strip">
    <div style="display:flex;align-items:center;gap:0">
        <span class="live-dot"></span>
        <span style="font-family:'Orbitron',sans-serif;font-size:0.82rem;font-weight:700;
                     color:#00c6ff;letter-spacing:0.1em">WC 2026</span>
    </div>
    <span style="color:#1e3a5f">│</span>
    <span style="color:#475569">Model: <b style="color:#e2e8f0">Poisson + GBT</b></span>
    <span style="color:#475569">Accuracy:
        <b style="color:#f59e0b;font-family:'Orbitron',sans-serif">{_acc_str}</b></span>
    <span style="color:#475569">Baseline: <b style="color:#e2e8f0">48%</b></span>
    <span style="color:#475569">Elo k:
        <b style="color:#e2e8f0;font-family:'Orbitron',sans-serif">{_predictor.K_ELO:.4f}</b></span>
    <span style="color:#475569">α_away:
        <b style="color:#e2e8f0;font-family:'Orbitron',sans-serif">{_predictor.ALPHA_AWAY:.3f}</b></span>
    <span style="color:#475569">Data: <b style="color:#e2e8f0">49,000+ matches</b></span>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_home, tab_fixtures, tab_predict, tab_stats, tab_h2h, tab_sim, tab_live, tab_friends = st.tabs([
    "⚽  Home", "📅  Fixtures", "🔮  Predict", "📊  Team Stats",
    "🏆  H2H", "🎲  Simulator", "📥  Live Results", "👥  Friends"
])

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    # ── Stats bar ──
    total_matches = len(WC2026)
    played_count  = int(WC2026['home_score'].notna().sum())
    remaining     = total_matches - played_count

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f'<div class="metric-card"><h3>{total_matches}</h3><p>Total Fixtures</p></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric-card"><h3 style="color:#f59e0b">{played_count}</h3><p>Played</p></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="metric-card"><h3 style="color:#00c6ff">{remaining}</h3><p>Upcoming</p></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="metric-card"><h3>{len(ALL_WC_TEAMS)}</h3><p>Teams</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Featured match hero ──
    upcoming_all = WC2026[WC2026['home_score'].isna()]
    if len(upcoming_all) > 0:
        feat = upcoming_all.iloc[0]
        _, feat_blend = run_prediction(feat['home_team'], feat['away_team'])
        feat_date = pd.to_datetime(feat['date']).strftime('%B %d, %Y')

        hw_disp = feat_blend['home_win'] if feat_blend else '--'
        dr_disp = feat_blend['draw']     if feat_blend else '--'
        aw_disp = feat_blend['away_win'] if feat_blend else '--'

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0b1628 0%,#0f2040 100%);
                    border:1px solid rgba(0,198,255,0.22);border-radius:16px;
                    padding:32px;margin-bottom:8px;position:relative;overflow:hidden">
            <div style="position:absolute;top:0;left:0;width:42%;height:100%;
                        background:linear-gradient(90deg,rgba(29,78,216,0.14),transparent)"></div>
            <div style="position:absolute;top:0;right:0;width:42%;height:100%;
                        background:linear-gradient(270deg,rgba(185,28,28,0.14),transparent)"></div>
            <div style="position:absolute;top:0;left:0;right:0;height:2px;
                        background:linear-gradient(90deg,transparent,#00c6ff,transparent)"></div>
            <div style="position:relative;z-index:1">
                <div style="text-align:center;margin-bottom:22px">
                    <span style="font-family:'Orbitron',sans-serif;font-size:0.68rem;font-weight:700;
                                 color:#00c6ff;letter-spacing:0.2em">
                        <span class="live-dot"></span>NEXT MATCH · FIFA WORLD CUP 2026 · {feat_date}
                    </span>
                </div>
                <div style="display:flex;justify-content:space-around;align-items:center">
                    <div style="text-align:center;flex:1">
                        <div style="font-size:3.2rem;line-height:1;margin-bottom:10px">{flag(feat['home_team'])}</div>
                        <div style="font-family:'Orbitron',sans-serif;font-size:0.95rem;font-weight:700;
                                    color:#e2e8f0;letter-spacing:0.04em">{feat['home_team']}</div>
                        <div style="font-family:'Orbitron',sans-serif;font-size:2rem;font-weight:900;
                                    color:#60a5fa;margin-top:8px;
                                    text-shadow:0 0 20px rgba(96,165,250,0.4)">{hw_disp}%</div>
                    </div>
                    <div style="text-align:center;padding:0 28px">
                        <div style="font-family:'Orbitron',sans-serif;font-size:2rem;font-weight:900;
                                    color:#00c6ff;text-shadow:0 0 30px rgba(0,198,255,0.6);
                                    letter-spacing:0.12em">VS</div>
                        <div style="color:#334155;font-family:'Orbitron',sans-serif;
                                    font-size:0.65rem;margin-top:10px;letter-spacing:0.1em">
                            DRAW {dr_disp}%
                        </div>
                    </div>
                    <div style="text-align:center;flex:1">
                        <div style="font-size:3.2rem;line-height:1;margin-bottom:10px">{flag(feat['away_team'])}</div>
                        <div style="font-family:'Orbitron',sans-serif;font-size:0.95rem;font-weight:700;
                                    color:#e2e8f0;letter-spacing:0.04em">{feat['away_team']}</div>
                        <div style="font-family:'Orbitron',sans-serif;font-size:2rem;font-weight:900;
                                    color:#f87171;margin-top:8px;
                                    text-shadow:0 0 20px rgba(248,113,113,0.4)">{aw_disp}%</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if feat_blend:
            render_prob_bar(feat['home_team'], feat['away_team'], feat_blend)

    # ── Next 4 matches ──
    st.markdown("### Upcoming Fixtures")
    rest = upcoming_all.iloc[1:5]
    for _, row in rest.iterrows():
        _, blend = run_prediction(row['home_team'], row['away_team'])
        match_date = pd.to_datetime(row['date']).strftime('%B %d, %Y')
        render_match_card(row['home_team'], row['away_team'], blend, match_date=match_date)

    st.markdown("---")
    st.markdown("### Pipeline Overview")
    c1, c2 = st.columns(2)
    with c1: st.info("**Gradient Boosting + Poisson Blend**\n\nCombines statistical Poisson expected goals model with a machine learning classifier trained on 20,000+ international matches from 2005–2025.")
    with c2: st.info("**Features:** Goals scored/conceded · Recent form (last 20) · H2H record (last 10) · Tournament weight · Elo rating · FIFA ranking · Home advantage")

# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════
with tab_fixtures:
    st.markdown("# World Cup 2026 — Fixtures")
    fix_tab1, fix_tab2 = st.tabs(["📋  Upcoming", "✅  Results"])

    with fix_tab1:
        upcoming_fx = WC2026[WC2026['home_score'].isna()].copy()
        for d_val, group in upcoming_fx.groupby('date'):
            date_label = pd.to_datetime(d_val).strftime('%A, %B %d')
            st.markdown(f"### 📅 {date_label}")
            for _, row in group.iterrows():
                poisson, blend = run_prediction(row['home_team'], row['away_team'])
                ml = poisson['most_likely'] if poisson else None
                render_match_card(row['home_team'], row['away_team'], blend, most_likely=ml)
            st.markdown("<br>", unsafe_allow_html=True)

    with fix_tab2:
        played_fx = WC2026[WC2026['home_score'].notna()].copy()
        if len(played_fx) == 0:
            st.info("No results yet.")
        else:
            played_fx['date_str'] = pd.to_datetime(played_fx['date']).dt.strftime('%b %d')
            for _, row in played_fx.sort_values('date', ascending=False).iterrows():
                h, a = int(row['home_score']), int(row['away_score'])
                border = "#1d4ed8" if h > a else ("#dc2626" if h < a else "#334155")
                st.markdown(f"""
                <div class="match-card" style="border-color:{border}44">
                    <div style="color:#334155;font-size:0.7rem;font-weight:600;letter-spacing:0.08em">
                        {row['date_str']} · FIFA WORLD CUP
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px">
                        <span style="color:#e2e8f0;font-size:1rem;font-weight:600">
                            {flag(row['home_team'])} {row['home_team']}
                        </span>
                        <span style="font-family:'Orbitron',sans-serif;font-size:1.6rem;
                                     font-weight:900;color:#f59e0b;
                                     text-shadow:0 0 16px rgba(245,158,11,0.4)">{h} – {a}</span>
                        <span style="color:#e2e8f0;font-size:1rem;font-weight:600">
                            {row['away_team']} {flag(row['away_team'])}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("# Match Predictor")

    col1, col2, col3 = st.columns([5, 1, 5])
    with col1:
        home_team = st.selectbox("Home / Team 1", ALL_WC_TEAMS,
                                 index=ALL_WC_TEAMS.index('Brazil') if 'Brazil' in ALL_WC_TEAMS else 0,
                                 format_func=lambda t: f"{flag(t)} {t}")
    with col2:
        st.markdown("<div style='text-align:center;margin-top:32px;font-family:Orbitron,sans-serif;font-size:1rem;color:#00c6ff;letter-spacing:0.1em'>VS</div>", unsafe_allow_html=True)
    with col3:
        away_team = st.selectbox("Away / Team 2", ALL_WC_TEAMS,
                                 index=ALL_WC_TEAMS.index('Morocco') if 'Morocco' in ALL_WC_TEAMS else 1,
                                 format_func=lambda t: f"{flag(t)} {t}")

    neutral = st.checkbox("Neutral venue (World Cup)", value=True)

    if home_team == away_team:
        st.warning("Please select two different teams.")
    else:
        with st.spinner("Running 8-step prediction pipeline..."):
            pred_result, blend = run_prediction(home_team, away_team, neutral=neutral)
            hf = get_features(home_team)
            af = get_features(away_team)

        if pred_result is None:
            st.error("Could not generate prediction — insufficient data for one or both teams.")
        else:
            st.markdown("---")
            st.markdown("### Win Probabilities")
            render_prob_bar(home_team, away_team, blend)

            col1, col2, col3, col4 = st.columns(4)
            with col1: st.markdown(f'<div class="metric-card"><h3 style="color:#60a5fa">{pred_result["home_win"]}%</h3><p>{home_team} Win</p></div>', unsafe_allow_html=True)
            with col2: st.markdown(f'<div class="metric-card"><h3 style="color:#64748b">{pred_result["draw"]}%</h3><p>Draw</p></div>', unsafe_allow_html=True)
            with col3: st.markdown(f'<div class="metric-card"><h3 style="color:#f87171">{pred_result["away_win"]}%</h3><p>{away_team} Win</p></div>', unsafe_allow_html=True)
            with col4:
                _eps = pred_result['entropy_eps']
                _conf_label = "🟢 HIGH" if _eps < 5 else ("🟡 MED" if _eps < 15 else "🔴 LOW")
                _conf_color = "#4ade80" if _eps < 5 else ("#f59e0b" if _eps < 15 else "#f87171")
                st.markdown(f'<div class="metric-card"><h3 style="color:{_conf_color};font-size:1.3rem">{_conf_label}</h3><p>Confidence · ε={_eps}%</p></div>', unsafe_allow_html=True)

            with st.expander("🔬 Pipeline Breakdown (8-step)"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**NB Poisson**")
                    st.caption(f"{pred_result['poisson_home_win']}% / {pred_result['poisson_draw']}% / {pred_result['poisson_away_win']}%")
                with c2:
                    st.markdown("**ML Model**")
                    st.caption(f"{pred_result['ml_home_win']}% / {pred_result['ml_draw']}% / {pred_result['ml_away_win']}%")
                with c3:
                    st.markdown("**Disagreement**")
                    st.caption(f"{pred_result['disagreement']:.1f} pts → {pred_result['blend_weight_poisson']}% NB / {100-pred_result['blend_weight_poisson']:.0f}% ML")
                st.caption(f"ε={pred_result['entropy_eps']}% · Elo Δ={pred_result['elo_diff']:.0f} · xG {pred_result['lambda_home']}–{pred_result['lambda_away']}")

            st.markdown("---")
            st.markdown("### Score Probability Matrix")
            col1, col2 = st.columns([3, 2])

            with col1:
                max_show = 6
                matrix = pred_result['score_matrix'][:max_show, :max_show] * 100
                hover = [[f"{flag(home_team)} {home_team} {i}–{j} {away_team} {flag(away_team)}<br><b>{matrix[i][j]:.2f}%</b>"
                          for j in range(max_show)] for i in range(max_show)]
                text_labels = [[f"{matrix[i][j]:.1f}%" for j in range(max_show)] for i in range(max_show)]
                fig_p = go.Figure(go.Heatmap(
                    z=matrix, x=list(range(max_show)), y=list(range(max_show)),
                    text=text_labels, texttemplate="%{text}",
                    textfont={"size": 11, "color": "white", "family": "Orbitron"},
                    hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
                    colorscale=[[0,'#050d1f'],[0.3,'#0d2456'],[0.7,'#1a4db5'],[1,'#00c6ff']],
                    showscale=False,
                ))
                ml_i, ml_j = pred_result['most_likely']
                if ml_i < max_show and ml_j < max_show:
                    fig_p.add_shape(type='rect',
                        x0=ml_j-0.5, x1=ml_j+0.5, y0=ml_i-0.5, y1=ml_i+0.5,
                        line=dict(color='#f59e0b', width=3),
                        fillcolor='rgba(0,0,0,0)')
                fig_p.update_layout(
                    title=dict(text="Hover cells for exact probability",
                               font=dict(color='#475569', size=12)),
                    xaxis=dict(
                        title=dict(text=f"{flag(away_team)} {away_team} Goals",
                                   font=dict(color='#f87171')),
                        tickmode='array', tickvals=list(range(max_show)),
                        tickfont=dict(color='#94a3b8', family='Orbitron', size=10),
                        gridcolor='rgba(255,255,255,0.03)'),
                    yaxis=dict(
                        title=dict(text=f"{flag(home_team)} {home_team} Goals",
                                   font=dict(color='#60a5fa')),
                        tickmode='array', tickvals=list(range(max_show)),
                        tickfont=dict(color='#94a3b8', family='Orbitron', size=10),
                        gridcolor='rgba(255,255,255,0.03)', autorange='reversed'),
                    paper_bgcolor='#080e1f',
                    plot_bgcolor='#080e1f',
                    font=dict(color='#e2e8f0', family='Inter'),
                    height=380, margin=dict(l=70, r=20, t=40, b=60),
                )
                st.plotly_chart(fig_p, width='stretch')

            with col2:
                st.markdown("#### Top 10 Most Likely Scores")
                for i, (h_g, a_g, prob) in enumerate(pred_result['top_scores'][:10]):
                    bar_w = int(prob / pred_result['top_scores'][0][2] * 100)
                    medal = ["🥇","🥈","🥉"][i] if i < 3 else f"**{i+1}.**"
                    st.markdown(f"""
                    <div style="margin:4px 0;display:flex;align-items:center;gap:8px">
                        <span style="width:24px;font-size:0.85rem">{medal}</span>
                        <span style="color:#e2e8f0;font-family:'Orbitron',sans-serif;
                                     font-size:0.9rem;font-weight:700;width:56px">{h_g}–{a_g}</span>
                        <div style="flex:1;background:rgba(255,255,255,0.05);border-radius:4px;height:14px">
                            <div style="width:{bar_w}%;background:linear-gradient(90deg,#0369a1,#00c6ff);
                                        height:100%;border-radius:4px"></div>
                        </div>
                        <span style="color:#475569;width:38px;text-align:right;
                                     font-family:'Orbitron',sans-serif;font-size:0.75rem">{prob}%</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Team Comparison")
            col1, col2 = st.columns(2)
            conf_labels = {'CAF':'AFCON','UEFA':'Euros','CONMEBOL':'Copa América',
                           'CONCACAF':'Gold Cup','AFC':'Asian Cup','OFC':'OFC Cup'}

            def form_badges(tn):
                return " ".join(["🟢" if b=="W" else "🟡" if b=="D" else "🔴"
                                  for b in get_team_form_string(tn, 8)])

            with col1:
                st.markdown(f"#### {flag(home_team)} {home_team}")
                st.metric("Adj Goals Scored",   f"{hf['adj_scored']:.2f}")
                st.metric("Adj Goals Conceded", f"{hf['avg_conceded']:.2f}")
                st.metric("Overall Form",       f"{hf['form']:.0%}")
                conf_h = detect_confederation(df, home_team)
                st.metric(f"{conf_labels.get(conf_h,'Continental')} Form", f"{hf['cont_form']:.0%}")
                st.markdown(f"**Recent:** {form_badges(home_team)}")
                st.caption(f"Elo: {hf['elo']:.0f} · FIFA: {hf['fifa_points']:.0f} · {hf['matches_played']} matches")

            with col2:
                st.markdown(f"#### {flag(away_team)} {away_team}")
                st.metric("Adj Goals Scored",   f"{af['adj_scored']:.2f}")
                st.metric("Adj Goals Conceded", f"{af['adj_conceded']:.2f}")
                st.metric("Overall Form",       f"{af['form']:.0%}")
                conf_a = detect_confederation(df, away_team)
                st.metric(f"{conf_labels.get(conf_a,'Continental')} Form", f"{af['cont_form']:.0%}")
                st.markdown(f"**Recent:** {form_badges(away_team)}")
                st.caption(f"Elo: {af['elo']:.0f} · FIFA: {af['fifa_points']:.0f} · {af['matches_played']} matches")

            st.markdown("---")
            st.markdown("### Expected Goals")
            col1, col2 = st.columns(2)
            with col1: st.markdown(f'<div class="metric-card"><h3 style="color:#60a5fa">{pred_result["lambda_home"]}</h3><p>{home_team} xG</p></div>', unsafe_allow_html=True)
            with col2: st.markdown(f'<div class="metric-card"><h3 style="color:#f87171">{pred_result["lambda_away"]}</h3><p>{away_team} xG</p></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TEAM STATS
# ══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    st.markdown("# Team Statistics")
    team = st.selectbox("Select a team", ALL_WC_TEAMS, format_func=lambda t: f"{flag(t)} {t}")

    f = get_features(team)
    if f is None:
        st.error("No data available for this team.")
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.markdown(f'<div class="metric-card"><h3 style="color:#60a5fa">{f["adj_scored"]:.2f}</h3><p>Goals Scored</p></div>', unsafe_allow_html=True)
        with col2: st.markdown(f'<div class="metric-card"><h3 style="color:#f87171">{f["adj_conceded"]:.2f}</h3><p>Goals Conceded</p></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="metric-card"><h3 style="color:#f59e0b">{f["form"]:.0%}</h3><p>Overall Form</p></div>', unsafe_allow_html=True)
        with col4:
            conf_ts = detect_confederation(df, team)
            conf_lbl = {'CAF':'AFCON','UEFA':'Euros','CONMEBOL':'Copa','CONCACAF':'Gold Cup','AFC':'Asian Cup','OFC':'OFC'}.get(conf_ts, 'Conf')
            st.markdown(f'<div class="metric-card"><h3 style="color:#00c6ff">{f["cont_form"]:.0%}</h3><p>{conf_lbl} Form</p></div>', unsafe_allow_html=True)
        with col5: st.markdown(f'<div class="metric-card"><h3>{f["matches_played"]}</h3><p>Matches</p></div>', unsafe_allow_html=True)

        st.markdown("---")

        matches_ts = df[
            ((df['home_team'] == team) | (df['away_team'] == team)) &
            (df['home_score'].notna())
        ].sort_values('date').tail(20).copy()

        history_ts = []
        for _, row in matches_ts.iterrows():
            if row['home_team'] == team:
                s, c, opp, venue = row['home_score'], row['away_score'], row['away_team'], "H"
            else:
                s, c, opp, venue = row['away_score'], row['home_score'], row['home_team'], "A"
            history_ts.append({
                'Date': pd.to_datetime(row['date']).strftime('%Y-%m-%d'),
                'Opponent': opp, 'Score': f"{int(s)}-{int(c)}",
                'Result': "W" if s>c else ("D" if s==c else "L"),
                'Venue': venue, 'Tournament': row['tournament'][:30]
            })

        def color_result(val):
            if val == "W": return "background-color:#0f2d1a;color:#4ade80"
            if val == "L": return "background-color:#2d0f0f;color:#f87171"
            return "background-color:#1a1f2d;color:#f59e0b"

        st.markdown("#### Last 20 Matches")
        st.dataframe(pd.DataFrame(history_ts).style.map(color_result, subset=['Result']),
                     width='stretch', hide_index=True)

        st.markdown("---")
        st.markdown("#### Goals per Match")
        scored_list, conceded_list, dates_list = [], [], []
        for _, row in matches_ts.iterrows():
            if row['home_team'] == team:
                scored_list.append(row['home_score']); conceded_list.append(row['away_score'])
            else:
                scored_list.append(row['away_score']); conceded_list.append(row['home_score'])
            dates_list.append(pd.to_datetime(row['date']).strftime('%m/%d'))

        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor('#050d1f')
        ax.set_facecolor('#080e1f')
        x = range(len(dates_list))
        ax.plot(x, scored_list,   color='#00c6ff', marker='o', linewidth=2, label='Scored',   markersize=5)
        ax.plot(x, conceded_list, color='#f87171', marker='s', linewidth=2, label='Conceded', markersize=5)
        ax.fill_between(x, scored_list,   alpha=0.1, color='#00c6ff')
        ax.fill_between(x, conceded_list, alpha=0.1, color='#f87171')
        ax.set_xticks(x); ax.set_xticklabels(dates_list, rotation=45, color='#475569', fontsize=8)
        ax.tick_params(colors='#475569')
        for spine in ['bottom','left']: ax.spines[spine].set_color('#0d1f38')
        for spine in ['top','right']:   ax.spines[spine].set_visible(False)
        ax.legend(facecolor='#080e1f', labelcolor='#94a3b8', framealpha=0.8)
        ax.set_title(f"{team} — Goals per Match", color='#94a3b8', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown(f"#### {flag(team)} WC 2026 Schedule")
        team_wc = WC2026[(WC2026['home_team'] == team) | (WC2026['away_team'] == team)]
        for _, row in team_wc.iterrows():
            d_str = pd.to_datetime(row['date']).strftime('%b %d')
            if pd.isna(row['home_score']):
                _, blend_ts = run_prediction(row['home_team'], row['away_team'])
                score_str = "vs"
            else:
                blend_ts  = None
                score_str = f"{int(row['home_score'])}–{int(row['away_score'])}"
            st.markdown(f"**{d_str}** — {flag(row['home_team'])} {row['home_team']} {score_str} {flag(row['away_team'])} {row['away_team']}")
            if blend_ts:
                render_prob_bar(row['home_team'], row['away_team'], blend_ts)

        st.markdown("---")
        st.markdown("#### Group Stage Qualification Odds")
        if st.button(f"Simulate {team}'s Group (1,000 runs)", key="grp_sim"):
            _team_group, _team_group_name = None, None
            for _gn, _gt in GROUPS.items():
                if team in _gt:
                    _team_group, _team_group_name = _gt, _gn; break
            if _team_group is None:
                st.info("This team is not in the WC 2026 group stage.")
            else:
                from simulator import simulate_group as _sim_group
                _fc_g  = {t: get_features(t) for t in _team_group}
                _mc_g  = {}
                _avg_h_g, _avg_a_g = compute_global_averages(df, TODAY)

                def _grp_predict(h, a, neutral=True):
                    _k = (h,a,neutral)
                    if _k not in _mc_g:
                        _hf,_af = _fc_g.get(h),_fc_g.get(a)
                        _mc_g[_k] = full_predict(_hf,_af,model,_avg_h_g,_avg_a_g,neutral=neutral) \
                            if _hf and _af else {'home_win':33.3,'draw':33.3,'away_win':33.3,'lambda_home':1.5,'lambda_away':1.0}
                    return _mc_g[_k]

                _finish = {t: [0]*4 for t in _team_group}
                with st.spinner(f"Simulating Group {_team_group_name}..."):
                    for _ in range(1000):
                        _s, _ = _sim_group(_team_group, _grp_predict)
                        for _p,_t in enumerate(_s): _finish[_t][_p] += 1

                _q_pct = (_finish[team][0]+_finish[team][1])/1000*100
                st.metric(f"{flag(team)} {team} Qualification — Group {_team_group_name}", f"{_q_pct:.1f}%",
                          help="Probability of finishing 1st or 2nd")
                _sorted = sorted(_team_group, key=lambda t: -(_finish[t][0]+_finish[t][1]))
                _cols_g = st.columns(len(_team_group))
                for _ci, _t in enumerate(_sorted):
                    with _cols_g[_ci]:
                        st.markdown(f"**{flag(_t)} {_t}**")
                        st.metric("Qualify", f"{(_finish[_t][0]+_finish[_t][1])/1000*100:.1f}%")
                        st.metric("1st",     f"{_finish[_t][0]/1000*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# H2H
# ══════════════════════════════════════════════════════════════════════════════
with tab_h2h:
    st.markdown("# Head to Head")
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", ALL_WC_TEAMS,
                             index=ALL_WC_TEAMS.index('Brazil') if 'Brazil' in ALL_WC_TEAMS else 0,
                             format_func=lambda t: f"{flag(t)} {t}")
    with col2:
        team2 = st.selectbox("Team 2", ALL_WC_TEAMS,
                             index=ALL_WC_TEAMS.index('Morocco') if 'Morocco' in ALL_WC_TEAMS else 1,
                             format_func=lambda t: f"{flag(t)} {t}")

    if team1 == team2:
        st.warning("Select two different teams.")
    else:
        h2h_df = df[
            (((df['home_team']==team1)&(df['away_team']==team2))|
             ((df['home_team']==team2)&(df['away_team']==team1)))&
            df['home_score'].notna()
        ].sort_values('date', ascending=False)

        if len(h2h_df) == 0:
            st.info(f"No historical matches found between {team1} and {team2}.")
        else:
            t1w = t2w = drs = t1g = t2g = 0
            for _, row in h2h_df.iterrows():
                s1, s2 = (row['home_score'], row['away_score']) if row['home_team']==team1 \
                          else (row['away_score'], row['home_score'])
                t1g += s1; t2g += s2
                if s1>s2: t1w+=1
                elif s1<s2: t2w+=1
                else: drs+=1

            col1,col2,col3,col4,col5 = st.columns(5)
            with col1: st.markdown(f'<div class="metric-card"><h3 style="color:#60a5fa">{t1w}</h3><p>{team1} Wins</p></div>', unsafe_allow_html=True)
            with col2: st.markdown(f'<div class="metric-card"><h3 style="color:#64748b">{drs}</h3><p>Draws</p></div>', unsafe_allow_html=True)
            with col3: st.markdown(f'<div class="metric-card"><h3 style="color:#f87171">{t2w}</h3><p>{team2} Wins</p></div>', unsafe_allow_html=True)
            with col4: st.markdown(f'<div class="metric-card"><h3 style="color:#60a5fa">{t1g}</h3><p>{team1} Goals</p></div>', unsafe_allow_html=True)
            with col5: st.markdown(f'<div class="metric-card"><h3 style="color:#f87171">{t2g}</h3><p>{team2} Goals</p></div>', unsafe_allow_html=True)

            total_h = t1w+t2w+drs
            st.markdown("#### Overall Record")
            render_prob_bar(team1, team2, {'home_win':round(t1w/total_h*100,1),
                                           'draw':round(drs/total_h*100,1),
                                           'away_win':round(t2w/total_h*100,1)})
            st.markdown("---")
            st.markdown(f"#### All {len(h2h_df)} Matches")

            rows_h2h = []
            for _, row in h2h_df.iterrows():
                s1,s2 = (int(row['home_score']),int(row['away_score'])) if row['home_team']==team1 \
                         else (int(row['away_score']),int(row['home_score']))
                rows_h2h.append({'Date':pd.to_datetime(row['date']).strftime('%Y-%m-%d'),
                                  team1:s1, team2:s2,
                                  'Result':"W" if s1>s2 else ("D" if s1==s2 else "L"),
                                  'Tournament':row['tournament'][:35],'Venue':row['city']})

            def color_h2h(val):
                if val=="W": return "background-color:#0f2d1a;color:#4ade80"
                if val=="L": return "background-color:#2d0f0f;color:#f87171"
                return "background-color:#1a1f2d;color:#f59e0b"

            st.dataframe(pd.DataFrame(rows_h2h).style.map(color_h2h, subset=['Result']),
                         width='stretch', hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.markdown("# Tournament Simulator")
    st.markdown("Monte Carlo simulation of the full WC 2026 bracket.")

    col1, col2 = st.columns([2,1])
    with col1: n_sims = st.select_slider("Simulations", options=[500,1000,2000,5000], value=1000)
    with col2: run_sim = st.button("▶  RUN SIMULATION", type="primary", width='stretch')

    if run_sim:
        with st.spinner(f"Simulating {n_sims:,} tournaments..."):
            avg_h_s, avg_a_s = compute_global_averages(df, TODAY)
            fc_s = {t: get_features(t) for teams in GROUPS.values() for t in teams}
            mc_s = {}

            def sim_fn(h, a, neutral=True):
                k = (h,a,neutral)
                if k not in mc_s:
                    hf_s,af_s = fc_s.get(h),fc_s.get(a)
                    mc_s[k] = full_predict(hf_s,af_s,model,avg_h_s,avg_a_s,neutral=neutral) \
                        if hf_s and af_s else {'home_win':33.3,'draw':33.3,'away_win':33.3,'lambda_home':1.5,'lambda_away':1.0}
                return mc_s[k]

            sim_res = simulate_tournament(sim_fn, n_simulations=n_sims)

        sorted_r = sorted(sim_res.items(), key=lambda x: -x[1]['champion'])
        st.markdown(f"---\n### Championship Probabilities — {n_sims:,} Simulations")

        for i, (t_s, s) in enumerate(sorted_r[:10]):
            medal  = ["🥇","🥈","🥉"][i] if i < 3 else f"**{i+1}.**"
            bar_w  = int(s['champion'] / sorted_r[0][1]['champion'] * 100)
            rank_c = ["#f59e0b","#94a3b8","#b45309"][i] if i < 3 else "#334155"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;margin:5px 0;
                        padding:11px 16px;
                        background:linear-gradient(135deg,#0b1628,#0d1f38);
                        border:1px solid rgba(255,255,255,0.05);border-radius:10px;
                        transition:all 0.2s">
                <span style="width:30px;font-size:1.05rem;text-align:center">{medal}</span>
                <span style="font-size:1.4rem">{flag(t_s)}</span>
                <span style="color:#e2e8f0;font-weight:600;width:160px;
                             font-size:0.95rem">{t_s}</span>
                <div style="flex:1;background:rgba(255,255,255,0.05);
                            border-radius:5px;height:20px;overflow:hidden">
                    <div style="width:{bar_w}%;
                                background:linear-gradient(90deg,#0369a1,#00c6ff);
                                height:100%;border-radius:5px;
                                animation:growBar 0.9s cubic-bezier(0.4,0,0.2,1) both"></div>
                </div>
                <span style="font-family:'Orbitron',sans-serif;color:#f59e0b;
                             font-weight:700;width:52px;text-align:right;
                             font-size:0.9rem">{s['champion']}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---\n#### Full Tournament Odds")
        sim_rows = [{'Team':f"{flag(t)} {t}",'R16':f"{s['r16']}%",
                     'QF':f"{s['qf']}%",'SF':f"{s['sf']}%",
                     'Final':f"{s['final']}%",'🏆 Win':f"{s['champion']}%"}
                    for t,s in sorted_r]
        st.dataframe(pd.DataFrame(sim_rows), width='stretch', hide_index=True)

        st.markdown("---\n#### Group Qualification Odds")
        for g_name, teams in GROUPS.items():
            with st.expander(f"Group {g_name}"):
                for t_g in sorted(teams, key=lambda t: -sim_res[t]['r16']):
                    sg = sim_res[t_g]
                    st.markdown(f"**{flag(t_g)} {t_g}** — Qualify: **{sg['r16']}%** | QF: {sg['qf']}% | Win: {sg['champion']}%")
    else:
        st.info("Click **▶ RUN SIMULATION** to simulate the full WC 2026 bracket.")
        st.markdown("- Group stage → R32 → R16 → QF → SF → Final\n- Draws in knockouts resolved via Elo-weighted penalty shootout\n- 48 teams · 12 groups · 1,000–5,000 simulations")

# ══════════════════════════════════════════════════════════════════════════════
# LIVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown("# Live Results — WC 2026")
    st.markdown("Submit match results to update predictions, form, and Elo in real time.")

    RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'results.csv')

    st.markdown("### Add a Result")
    all_teams_live = sorted({t for teams in GROUPS.values() for t in teams})
    col1, col2, col3 = st.columns(3)
    with col1:
        live_home = st.selectbox("Home Team", all_teams_live, key="lh")
        live_home_score = st.number_input("Home Goals", min_value=0, max_value=20, value=0, step=1, key="lhs")
    with col2:
        st.markdown("<div style='padding-top:2rem;text-align:center;font-family:Orbitron,sans-serif;font-size:1.2rem;color:#00c6ff;letter-spacing:0.1em;font-weight:700'>VS</div>", unsafe_allow_html=True)
    with col3:
        live_away = st.selectbox("Away Team", [t for t in all_teams_live if t!=live_home], key="la")
        live_away_score = st.number_input("Away Goals", min_value=0, max_value=20, value=0, step=1, key="las")

    col_a, col_b, col_c = st.columns(3)
    with col_a: live_date = st.date_input("Match Date", value=datetime.today().date(), key="ld")
    with col_b: live_tournament = st.selectbox("Tournament", ["FIFA World Cup","FIFA World Cup qualification","Friendly"], key="lt")
    with col_c: live_neutral = st.checkbox("Neutral Venue", value=True, key="ln")

    _on_cloud = not os.path.isabs(__file__) or "/mount/src/" in __file__
    if _on_cloud:
        st.warning("⚠️ **Cloud deployment:** results submitted here are not permanently saved. "
                   "To persist new results, add them to `results.csv` and push a new commit to GitHub.")

    if st.button("Submit Result", type="primary"):
        if live_home == live_away:
            st.error("Home and away teams must be different.")
        else:
            new_row = pd.DataFrame([{
                'date':pd.Timestamp(live_date),'home_team':live_home,'away_team':live_away,
                'home_score':int(live_home_score),'away_score':int(live_away_score),
                'tournament':live_tournament,'city':'WC 2026',
                'country':'USA/Canada/Mexico','neutral':live_neutral,
            }])
            existing = pd.read_csv(RESULTS_PATH, parse_dates=['date'])
            dup = ((existing['date'].dt.date==live_date)&
                   (existing['home_team']==live_home)&(existing['away_team']==live_away))
            if dup.any():
                st.warning("Result already exists.")
            else:
                updated = pd.concat([existing, new_row], ignore_index=True)
                updated.to_csv(RESULTS_PATH, index=False)
                st.cache_data.clear(); st.cache_resource.clear()
                _n_wc = int(((updated['tournament']=='FIFA World Cup')&
                              (pd.to_datetime(updated['date']).dt.year==2026)&
                              updated['home_score'].notna()).sum())
                if _n_wc > 0 and _n_wc % 5 == 0:
                    _pkl = os.path.join(os.path.dirname(__file__), 'ml_model.pkl')
                    if os.path.exists(_pkl): os.remove(_pkl)
                    st.info(f"🔄 Auto-retrain queued — {_n_wc} WC results. Reload to retrain.")
                st.success(f"✅ {live_home} {int(live_home_score)}–{int(live_away_score)} {live_away} added. Reload to update predictions.")

    st.markdown("---")
    st.markdown("### WC 2026 Results")
    wc_played = df[(df['tournament']=='FIFA World Cup')&
                   (df['date'].dt.year==2026)&
                   df['home_score'].notna()].sort_values('date', ascending=False).copy()

    if wc_played.empty:
        st.info("No WC 2026 results recorded yet.")
    else:
        wc_played['Score'] = (wc_played['home_score'].astype(int).astype(str)
                              +' – '+wc_played['away_score'].astype(int).astype(str))
        st.dataframe(
            wc_played[['date','home_team','Score','away_team','tournament']]
            .rename(columns={'date':'Date','home_team':'Home','away_team':'Away','tournament':'Tournament'})
            .style.format({'Date':lambda x:x.strftime('%Y-%m-%d')}),
            width='stretch', hide_index=True)
        st.caption(f"**{len(wc_played)} match(es) recorded**")

    if len(wc_played) >= 3:
        st.markdown("---")
        st.markdown("### Brier Score")
        st.caption("Lower is better · Perfect=0.000 · Baseline≈0.52 · Random=0.667")
        if st.button("Compute Brier Score", key="brier_btn"):
            with st.spinner("Computing..."):
                _avg_h_b, _avg_a_b = compute_global_averages(df, TODAY)
                _bs_rows = []
                for _, _r in wc_played.iterrows():
                    _hf_b = get_features(_r['home_team']); _af_b = get_features(_r['away_team'])
                    if _hf_b is None or _af_b is None: continue
                    _p = full_predict(_hf_b,_af_b,model,_avg_h_b,_avg_a_b,neutral=True)
                    _ph,_pd,_pa = _p['home_win']/100, _p['draw']/100, _p['away_win']/100
                    _hs,_as = _r['home_score'],_r['away_score']
                    _bs = (_ph-(1 if _hs>_as else 0))**2+(_pd-(1 if _hs==_as else 0))**2+(_pa-(1 if _hs<_as else 0))**2
                    _bs_rows.append({'Match':f"{_r['home_team']} vs {_r['away_team']}",
                                     'H%':f"{_p['home_win']}%",'D%':f"{_p['draw']}%",'A%':f"{_p['away_win']}%",
                                     'Result':'H' if _hs>_as else ('D' if _hs==_as else 'A'),
                                     'Brier':round(_bs,3)})
            if _bs_rows:
                _ab = float(np.mean([r['Brier'] for r in _bs_rows]))
                _g  = "Excellent" if _ab<0.4 else ("Good" if _ab<0.5 else ("Fair" if _ab<0.6 else "Poor"))
                c1,c2,c3 = st.columns(3)
                with c1: st.metric("Avg Brier", f"{_ab:.3f}")
                with c2: st.metric("Matches",   len(_bs_rows))
                with c3: st.metric("Grade",      _g)
                st.dataframe(pd.DataFrame(_bs_rows), width='stretch', hide_index=True)

    st.markdown("---")
    st.markdown("### Calibrate Probabilities")
    wc_cal = df[(df['tournament']=='FIFA World Cup')&(df['date'].dt.year==2026)&df['home_score'].notna()]
    n_cal  = len(wc_cal)
    if n_cal < 10:
        st.info(f"Calibration requires 10 WC results. **{n_cal}/10 recorded.**")
    else:
        st.success(f"✅ {n_cal} results — enough to calibrate.")
        if st.button("Fit Calibrator Now", type="primary"):
            with st.spinner("Fitting isotonic regression..."):
                _CAL = _predictor.CALIBRATOR
                preds, actuals = [], []
                _ahc, _aac = compute_global_averages(df, TODAY)
                for _, row in wc_cal.iterrows():
                    _hf_c,_af_c = get_features(row['home_team']),get_features(row['away_team'])
                    if _hf_c is None or _af_c is None: continue
                    _rc = full_predict(_hf_c,_af_c,model,_ahc,_aac,neutral=True)
                    preds.append({'home_win':_rc['home_win']/100,'draw':_rc['draw']/100,'away_win':_rc['away_win']/100})
                    _hs,_as = row['home_score'],row['away_score']
                    actuals.append('home_win' if _hs>_as else ('draw' if _hs==_as else 'away_win'))
            if len(preds) >= 10:
                _CAL.fit(preds, actuals); _predictor.CALIBRATOR = _CAL
                st.success(f"Calibrator fitted on {len(preds)} matches. Reload to apply.")
            else:
                st.warning("Not enough valid pairs — check team name mappings.")

# ══════════════════════════════════════════════════════════════════════════════
# FRIENDS
# ══════════════════════════════════════════════════════════════════════════════
with tab_friends:
    st.markdown("# 👥 Friend Group Predictions")
    st.markdown("Create a private group, share the invite code, and compete against your friends on every match.")

    if not _friends.is_configured():
        st.error("Supabase is not configured. Add `supabase_url` and `supabase_key` to `.streamlit/secrets.toml`.")
        st.stop()

    # ── Session state ──────────────────────────────────────────────────────
    if "fr_username"  not in st.session_state: st.session_state.fr_username  = ""
    if "fr_group_id"  not in st.session_state: st.session_state.fr_group_id  = ""
    if "fr_group_name" not in st.session_state: st.session_state.fr_group_name = ""
    if "fr_invite_code" not in st.session_state: st.session_state.fr_invite_code = ""

    # ── Step 1: identity ───────────────────────────────────────────────────
    if not st.session_state.fr_username:
        st.markdown("### Step 1 — Choose your name")
        col_u, _ = st.columns([2, 3])
        with col_u:
            name_input = st.text_input("Your display name", placeholder="e.g. Mohammed", key="fr_name_inp")
            if st.button("Continue", type="primary", key="fr_name_btn"):
                if name_input.strip():
                    st.session_state.fr_username = name_input.strip()
                    st.rerun()
                else:
                    st.error("Please enter a name.")
        st.stop()

    st.markdown(f"**Playing as:** `{st.session_state.fr_username}`  "
                f"<span style='color:#475569;font-size:0.8rem'>(reload to switch)</span>",
                unsafe_allow_html=True)
    st.divider()

    # ── Step 2: create or join a group ────────────────────────────────────
    if not st.session_state.fr_group_id:
        st.markdown("### Step 2 — Create or join a group")
        col_c, col_j = st.columns(2)

        with col_c:
            st.markdown("**Create a new group**")
            grp_name = st.text_input("Group name", placeholder="e.g. Office WC Preds", key="fr_grp_name")
            if st.button("Create Group", type="primary", key="fr_create_btn"):
                if grp_name.strip():
                    grp, err = _friends.create_group(grp_name, st.session_state.fr_username)
                    if err:
                        st.error(err)
                    else:
                        st.session_state.fr_group_id    = grp["id"]
                        st.session_state.fr_group_name  = grp["name"]
                        st.session_state.fr_invite_code = grp["invite_code"]
                        st.rerun()
                else:
                    st.error("Enter a group name.")

        with col_j:
            st.markdown("**Join an existing group**")
            inv_code = st.text_input("Invite code", placeholder="e.g. A3BX9K", key="fr_inv_code")
            if st.button("Join Group", key="fr_join_btn"):
                if inv_code.strip():
                    grp, err = _friends.join_group(inv_code, st.session_state.fr_username)
                    if err:
                        st.error(err)
                    else:
                        st.session_state.fr_group_id    = grp["id"]
                        st.session_state.fr_group_name  = grp["name"]
                        st.session_state.fr_invite_code = grp["invite_code"]
                        st.rerun()
                else:
                    st.error("Enter an invite code.")
        st.stop()

    # ── Group header ───────────────────────────────────────────────────────
    gid  = st.session_state.fr_group_id
    gname = st.session_state.fr_group_name
    gcode = st.session_state.fr_invite_code

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0b1628,#0f2040);'
        f'border:1px solid rgba(0,198,255,0.2);border-radius:12px;padding:16px 22px;margin-bottom:16px">'
        f'<span style="font-family:\'Orbitron\',sans-serif;font-size:1rem;color:#e2e8f0;font-weight:700">'
        f'{gname}</span>'
        f'<span style="margin-left:18px;font-family:\'Orbitron\',sans-serif;font-size:0.75rem;'
        f'color:#00c6ff;letter-spacing:0.18em;background:rgba(0,198,255,0.1);'
        f'border:1px solid rgba(0,198,255,0.3);border-radius:6px;padding:4px 10px">'
        f'INVITE: {gcode}</span>'
        f'<span style="margin-left:12px;color:#475569;font-size:0.75rem">'
        f'Share this code with friends so they can join</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    members = _friends.get_members(gid)
    st.caption(f"{len(members)} member(s): {', '.join(m['username'] for m in members)}")
    st.divider()

    # ── Inner tabs ─────────────────────────────────────────────────────────
    ft_picks, ft_board, ft_all = st.tabs(["🎯  Submit Picks", "🏆  Leaderboard", "📋  All Picks"])

    # ── Submit Picks ───────────────────────────────────────────────────────
    with ft_picks:
        st.markdown("### Upcoming Matches — make your prediction")
        st.caption("Correct outcome = **2 pts** · Exact scoreline = **5 pts total**")

        upcoming_fr = WC2026[WC2026['home_score'].isna()].copy().sort_values('date')
        my_picks    = {(p['home_team'], p['away_team']): p for p in _friends.get_my_picks(gid, st.session_state.fr_username)}

        if upcoming_fr.empty:
            st.info("No upcoming fixtures left to pick.")
        else:
            for _, row in upcoming_fr.iterrows():
                ht, at = row['home_team'], row['away_team']
                mdate  = pd.to_datetime(row['date']).date()
                existing = my_picks.get((ht, at), {})
                prev_winner = existing.get('predicted_winner', None)
                prev_hs     = existing.get('predicted_home_score', None)
                prev_as     = existing.get('predicted_away_score', None)
                hf, af = flag(ht), flag(at)

                with st.expander(
                    f"{hf} {ht}  vs  {af} {at}  —  "
                    f"{pd.to_datetime(row['date']).strftime('%b %d')}  "
                    f"{'✅ picked' if prev_winner else ''}",
                    expanded=prev_winner is None
                ):
                    col_w, col_hs, col_as = st.columns([3, 1, 1])
                    safe_key = f"{ht}_{at}".replace(" ", "_")
                    with col_w:
                        winner_opts = [f"{hf} {ht} win", "Draw", f"{af} {at} win"]
                        winner_idx  = ({"home": 0, "draw": 1, "away": 2}.get(prev_winner, 0)
                                       if prev_winner else 0)
                        winner_sel  = st.radio("Outcome", winner_opts, index=winner_idx,
                                               horizontal=True, key=f"fr_w_{safe_key}")
                        winner_val  = ["home", "draw", "away"][winner_opts.index(winner_sel)]
                    with col_hs:
                        hs_val = st.number_input(f"{ht} goals", 0, 20,
                                                 value=int(prev_hs) if prev_hs is not None else 1,
                                                 key=f"fr_hs_{safe_key}")
                    with col_as:
                        as_val = st.number_input(f"{at} goals", 0, 20,
                                                 value=int(prev_as) if prev_as is not None else 0,
                                                 key=f"fr_as_{safe_key}")

                    if st.button("Save pick", key=f"fr_save_{safe_key}"):
                        ok, err = _friends.submit_pick(
                            gid, st.session_state.fr_username,
                            ht, at, mdate, winner_val, hs_val, as_val
                        )
                        if ok:
                            st.success("Pick saved!")
                            st.rerun()
                        else:
                            st.error(f"Error: {err}")

    # ── Leaderboard ────────────────────────────────────────────────────────
    with ft_board:
        st.markdown("### Leaderboard")
        lb = _friends.compute_leaderboard(gid, df)
        if lb.empty:
            st.info("No picks have been scored yet — results are added after each match.")
        else:
            lb.columns = ["Player", "Points", "Correct", "Exact Score", "Total Picks"]
            st.dataframe(lb, use_container_width=True)
            st.caption("Correct = right outcome (2 pts) · Exact Score = right scoreline (5 pts total)")

    # ── All Picks ──────────────────────────────────────────────────────────
    with ft_all:
        st.markdown("### All Picks in This Group")
        all_p = _friends.get_all_picks(gid)
        if not all_p:
            st.info("No picks submitted yet.")
        else:
            ap_df = pd.DataFrame([{
                "Player":   p["username"],
                "Match":    f"{p['home_team']} vs {p['away_team']}",
                "Date":     p["match_date"],
                "Pick":     {"home": f"🏠 {p['home_team']}", "draw": "🤝 Draw",
                              "away": f"✈️ {p['away_team']}"}.get(p["predicted_winner"], p["predicted_winner"]),
                "Score":    (f"{p['predicted_home_score']}–{p['predicted_away_score']}"
                             if p["predicted_home_score"] is not None else "—"),
            } for p in all_p])
            st.dataframe(ap_df, use_container_width=True, hide_index=True)
