import pandas as pd
import numpy as np
import os as _os

_BASE = _os.path.dirname(_os.path.abspath(__file__))

# ─── Name mapping: results.csv → FIFA ranking CSV ────────────────────────────
FIFA_NAME_MAP = {
    'Cape Verde':     'Cabo Verde',
    'Curaçao':        'Curacao',
    'Czech Republic': 'Czechia',
    'DR Congo':       'Congo DR',
    'Iran':           'IR Iran',
    'Ivory Coast':    "Côte d'Ivoire",
    'South Korea':    'Korea Republic',
    'United States':  'USA',
    'North Korea':    'Korea DPR',
}

# ─── Confederation strength factors (Fix #1) ─────────────────────────────────
# Based on average FIFA ranking of confederation members + World Cup performance
CONFEDERATION_STRENGTH = {
    'UEFA':     1.00,   # baseline — strongest confederation
    'CONMEBOL': 0.98,
    'CONCACAF': 0.85,
    'CAF':      0.82,
    'AFC':      0.80,
    'OFC':      0.72,
    'OTHER':    0.80,
}

CONF_TOURNAMENTS = {
    'CAF':      ['African Cup of Nations', 'African Cup of Nations qualification'],
    'UEFA':     ['UEFA Euro', 'UEFA Euro qualification', 'UEFA Nations League'],
    'CONMEBOL': ['Copa América', 'Copa América qualification'],
    'CONCACAF': ['Gold Cup', 'CONCACAF Nations League', 'CONCACAF Championship'],
    'AFC':      ['AFC Asian Cup', 'AFC Asian Cup qualification'],
    'OFC':      ['Oceania Nations Cup'],
}
ALL_CONF_TOURNAMENTS = [t for ts in CONF_TOURNAMENTS.values() for t in ts]

TOURNAMENT_WEIGHTS = {
    'FIFA World Cup': 3.0,
    'FIFA World Cup qualification': 2.0,
    'UEFA Euro': 2.5, 'Copa América': 2.5,
    'African Cup of Nations': 2.0, 'AFC Asian Cup': 2.0,
    'Gold Cup': 1.8, 'UEFA Nations League': 1.8,
    'CONCACAF Nations League': 1.5, 'Friendly': 1.0,
}

def load_former_names():
    """
    Load former_names.csv (columns: current, former, start_date, end_date).
    Returns empty DataFrame if file not found.
    """
    path = _os.path.join(_BASE, 'former_names.csv')
    if not _os.path.exists(path):
        return pd.DataFrame(columns=['current', 'former', 'start_date', 'end_date'])
    fn = pd.read_csv(path, parse_dates=['start_date', 'end_date'])
    fn['start_date'] = pd.to_datetime(fn['start_date'], errors='coerce')
    fn['end_date']   = pd.to_datetime(fn['end_date'],   errors='coerce')
    return fn

def resolve_team_names(df, former_names):
    """
    Replace historic team names in results DataFrame with current names.
    Each row in former_names covers a date range when 'former' was used.
    If end_date is NaT, the mapping is open-ended.
    """
    if former_names.empty:
        return df
    df = df.copy()
    for _, row in former_names.iterrows():
        former  = row['current'] if pd.isna(row.get('former')) else row['former']
        current = row['current']
        start   = row['start_date']
        end     = row['end_date']
        mask_home = df['home_team'] == former
        mask_away = df['away_team'] == former
        if pd.notna(start):
            mask_home &= df['date'] >= start
            mask_away &= df['date'] >= start
        if pd.notna(end):
            mask_home &= df['date'] <= end
            mask_away &= df['date'] <= end
        df.loc[mask_home, 'home_team'] = current
        df.loc[mask_away, 'away_team'] = current
    return df

def load_data():
    df = pd.read_csv(_os.path.join(_BASE, 'results.csv'), parse_dates=['date'])
    shootouts = pd.read_csv(_os.path.join(_BASE, 'shootouts.csv'), parse_dates=['date'])
    former_names = load_former_names()
    df = resolve_team_names(df, former_names)
    return df, shootouts

def load_fifa_rankings():
    files = [f for f in _os.listdir(_BASE) if f.startswith('fifa_ranking') and f.endswith('.csv')]
    if not files:
        return None
    dfs = [pd.read_csv(_os.path.join(_BASE, f)) for f in files]
    fifa = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['country_full','rank_date'])
    fifa['rank_date'] = pd.to_datetime(fifa['rank_date'])
    return fifa.sort_values(['country_full','rank_date']).reset_index(drop=True)

def get_fifa_points(fifa_df, team, as_of_date):
    if fifa_df is None:
        return 1500.0, 100
    fifa_name = FIFA_NAME_MAP.get(team, team)
    past = fifa_df[
        (fifa_df['country_full'] == fifa_name) &
        (fifa_df['rank_date'] <= as_of_date)
    ]
    if len(past) == 0:
        return 1500.0, 100
    latest = past.iloc[-1]
    pts  = float(latest['total_points']) if pd.notna(latest['total_points']) else 1500.0
    rank = int(latest['rank']) if pd.notna(latest['rank']) else 100
    return pts, rank

def detect_confederation(df, team):
    matches = df[(df['home_team']==team)|(df['away_team']==team)]
    for conf, tournaments in CONF_TOURNAMENTS.items():
        if matches['tournament'].isin(tournaments).any():
            return conf
    return 'OTHER'

def _ewm_avg(results, span):
    """
    ewm-weighted average matching pandas ewm(span=span, ignore_na=True).
    Weights: most-recent observation gets highest weight.
    Returns None when results is empty.
    """
    if not results:
        return None
    n = len(results)
    alpha = 2.0 / (span + 1)
    w = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)], dtype=float)
    w /= w.sum()
    return float(np.average(results, weights=w))

# ─── Fix #1: Opponent-quality-adjusted stats ────────────────────────────────
def get_opponent_rank(fifa_df, team, as_of_date):
    pts, rank = get_fifa_points(fifa_df, team, as_of_date)
    return rank

def quality_adjusted_stats(df, fifa_df, team, as_of_date, n=20):
    """
    Compute goals scored/conceded adjusted for opponent quality.
    vs rank 1-20:  weight 1.5x
    vs rank 21-50: weight 1.2x
    vs rank 51-100: weight 1.0x
    vs rank 100+:  weight 0.7x
    """
    matches = df[
        ((df['home_team']==team)|(df['away_team']==team)) &
        (df['date'] < as_of_date) & (df['home_score'].notna())
    ].sort_values('date').tail(n)

    if len(matches) < 3:
        return None, None, None, None

    scored_list, conceded_list, weights = [], [], []

    for _, row in matches.iterrows():
        opp = row['away_team'] if row['home_team']==team else row['home_team']
        s   = row['home_score'] if row['home_team']==team else row['away_score']
        c   = row['away_score'] if row['home_team']==team else row['home_score']

        _, opp_rank = get_fifa_points(fifa_df, opp, as_of_date)

        # Opponent quality weight
        if opp_rank <= 20:   oq_w = 1.5
        elif opp_rank <= 50: oq_w = 1.2
        elif opp_rank <= 100: oq_w = 1.0
        else:                 oq_w = 0.7

        # Tournament weight
        t_w = TOURNAMENT_WEIGHTS.get(row['tournament'], 1.0)

        scored_list.append(s)
        conceded_list.append(c)
        weights.append(oq_w * t_w)

    weights = np.array(weights)
    # Exponential time decay: w = exp(-days / T_scale), T_scale = 10 years
    T_SCALE_DAYS = 10 * 365.25
    match_dates = pd.to_datetime(matches['date'])
    days_ago = np.array([(as_of_date - d).days for d in match_dates], dtype=float)
    days_ago = np.clip(days_ago, 0, None)
    recency = np.exp(-days_ago / T_SCALE_DAYS)
    combined = weights * recency
    combined /= combined.sum()

    adj_scored   = float(np.average(scored_list, weights=combined))
    adj_conceded = float(np.average(conceded_list, weights=combined))

    return adj_scored, adj_conceded, combined, matches

# ─── Fix #2: Single primary strength = FIFA points (Elo as recency correction)
# Reference distributions (approx. from FIFA ranking data):
#   FIFA points: mean ≈ 1000, std ≈ 300
#   Elo:         mean ≈ 1500, std ≈ 150
_FIFA_MEAN, _FIFA_STD = 1000.0, 300.0
_ELO_MEAN,  _ELO_STD  = 1500.0, 150.0

def get_primary_strength(fifa_pts, elo, alpha=0.75):
    """
    Z-score normalize both ratings to a common scale, then blend.
    alpha=0.75 weights FIFA (current ranking) over Elo while letting recent
    form still influence predictions for teams whose ranking lags their results.
    Output is in standardized units; callers should treat it as a relative signal.
    """
    z_fifa = (fifa_pts - _FIFA_MEAN) / _FIFA_STD
    z_elo  = (elo       - _ELO_MEAN)  / _ELO_STD
    return alpha * z_fifa + (1 - alpha) * z_elo

# ─── Core team features ───────────────────────────────────────────────────────
def get_team_features(df, team, as_of_date, n_matches=20,
                      elo=None, fifa_points=None, fifa_rank=None, fifa_df=None):
    
    # Quality-adjusted stats (Fix #1)
    if fifa_df is not None:
        adj_scored, adj_conceded, combined, matches = quality_adjusted_stats(
            df, fifa_df, team, as_of_date, n=n_matches)
    else:
        matches = df[
            ((df['home_team']==team)|(df['away_team']==team)) &
            (df['date'] < as_of_date) & (df['home_score'].notna())
        ].sort_values('date').tail(n_matches)
        adj_scored = adj_conceded = combined = None

    if matches is None or len(matches) < 3:
        return None

    gs, gc, rs, home_rs, away_rs, wc_rs = [], [], [], [], [], []
    for _, row in matches.iterrows():
        s, c = (row['home_score'], row['away_score']) if row['home_team']==team \
               else (row['away_score'], row['home_score'])
        result = 1.0 if s > c else (0.5 if s == c else 0.0)
        gs.append(s); gc.append(c); rs.append(result)
        if row['home_team'] == team:
            home_rs.append(result)
        else:
            away_rs.append(result)
        if row['tournament'] in ('FIFA World Cup', 'FIFA World Cup qualification'):
            wc_rs.append(result)

    # ewm weights matching training's ewm(span=20, ignore_na=True)
    n = len(matches)
    alpha20 = 2.0 / (20 + 1)
    ewm_w = np.array([(1 - alpha20) ** (n - 1 - i) for i in range(n)], dtype=float)
    ewm_w /= ewm_w.sum()

    gd_list = np.array(gs) - np.array(gc)

    # WC form: ewm(span=10) over WC-only matches — matches training's roll_wc_form
    _wc = _ewm_avg(wc_rs, span=10)
    wc_form = _wc if _wc is not None else float(np.average(rs, weights=ewm_w))

    # Continental form: ewm(span=10) via get_continental_form
    cont_form = get_continental_form(df, team, as_of_date)

    # Home/away form: ewm(span=20) over split — matches training's roll_home/away_form
    _hf = _ewm_avg(home_rs, span=20)
    _af = _ewm_avg(away_rs, span=20)
    home_form = _hf if _hf is not None else float(np.average(rs, weights=ewm_w))
    away_form = _af if _af is not None else float(np.average(rs, weights=ewm_w))

    # Fix #2: single primary strength
    elo_val  = elo if elo is not None else 1500.0
    fifa_val = fifa_points if fifa_points is not None else 1500.0
    primary_strength = get_primary_strength(fifa_val, elo_val)

    # Team-specific φ via Bayesian shrinkage (all goals, home and away)
    from predictor import team_phi as _team_phi
    all_goals = gs + gc   # scored + conceded goals — captures goal variance profile
    phi_team  = _team_phi(all_goals)

    return {
        'team':             team,
        # Raw stats (ewm-weighted, matches training roll_scored/roll_conceded)
        'avg_scored':       float(np.average(gs, weights=ewm_w)),
        'avg_conceded':     float(np.average(gc, weights=ewm_w)),
        # Quality-adjusted stats (Fix #1) — opponent quality + calendar decay
        'adj_scored':       adj_scored if adj_scored is not None else float(np.average(gs, weights=ewm_w)),
        'adj_conceded':     adj_conceded if adj_conceded is not None else float(np.average(gc, weights=ewm_w)),
        # Team-specific NB dispersion (Bayesian shrinkage toward global φ)
        'phi':              phi_team,
        # Form (all ewm-weighted, matching training roll_* columns exactly)
        'form':             float(np.average(rs, weights=ewm_w)),
        'wc_form':          wc_form,
        'cont_form':        cont_form,
        'home_form':        home_form,
        'away_form':        away_form,
        # Extra stats
        'avg_gd':           float(np.average(gd_list, weights=ewm_w)),
        'clean_sheet_rate': float(np.mean(np.array(gc) == 0)),
        'scored_2p_rate':   float(np.mean(np.array(gs) >= 2)),
        # Ratings (Fix #2)
        'elo':              elo_val,
        'fifa_points':      fifa_val,
        'fifa_rank':        fifa_rank if fifa_rank is not None else 100,
        'primary_strength': primary_strength,
        'matches_played':   len(matches),
        'last_match':       matches['date'].max(),
    }

def get_h2h_advantage(df, team1, team2, n=10, shootouts=None):
    h2h = df[
        (((df['home_team']==team1)&(df['away_team']==team2))|
         ((df['home_team']==team2)&(df['away_team']==team1))) &
        (df['home_score'].notna())
    ].sort_values('date').tail(n)
    if len(h2h) == 0: return 0.0

    # Build a lookup of penalty winners for draws: (date, home, away) -> winner
    penalty_winners = {}
    if shootouts is not None and len(shootouts) > 0:
        for _, sr in shootouts.iterrows():
            key = (str(sr['date'])[:10], sr['home_team'], sr['away_team'])
            penalty_winners[key] = sr['winner']

    now = pd.Timestamp.now()
    T_DAYS = 4 * 365.25
    pts, total_w = 0.0, 0.0
    for _, row in h2h.iterrows():
        days_ago = max((now - pd.to_datetime(row['date'])).days, 0)
        w = float(np.exp(-days_ago / T_DAYS))
        hs, as_ = row['home_score'], row['away_score']
        if hs > as_:
            result = 1.0 if row['home_team'] == team1 else 0.0
        elif as_ > hs:
            result = 1.0 if row['away_team'] == team1 else 0.0
        else:
            # Draw in 90 min — check if penalties decided the tie
            key = (str(row['date'])[:10], row['home_team'], row['away_team'])
            pen_winner = penalty_winners.get(key)
            if pen_winner == team1:
                result = 0.75  # penalty win counts more than draw, less than win
            elif pen_winner is not None and pen_winner != team1:
                result = 0.25  # penalty loss
            else:
                result = 0.5
        pts += w * result
        total_w += w
    return (pts / total_w - 0.5) if total_w > 0 else 0.0

def get_continental_form(df, team, as_of_date, n=8):
    best_matches, best_date = None, pd.Timestamp('1900-01-01')
    for conf, tournaments in CONF_TOURNAMENTS.items():
        cm = df[
            ((df['home_team']==team)|(df['away_team']==team)) &
            df['tournament'].isin(tournaments) &
            (df['date'] < as_of_date) & (df['home_score'].notna())
        ]
        if len(cm) > 0 and cm['date'].max() > best_date:
            best_date = cm['date'].max(); best_matches = cm
    if best_matches is None: return 0.5
    recent = best_matches.sort_values('date').tail(n)
    results = []
    for _, row in recent.iterrows():
        s, c = (row['home_score'], row['away_score']) if row['home_team']==team \
               else (row['away_score'], row['home_score'])
        results.append(1.0 if s > c else (0.5 if s == c else 0.0))
    v = _ewm_avg(results, span=10)
    return v if v is not None else 0.5

def compute_global_averages(df, as_of_date):
    recent = df[
        (df['date'] < as_of_date) &
        (df['date'] >= as_of_date - pd.Timedelta(days=365*4)) &
        (df['home_score'].notna())
    ]
    return recent['home_score'].mean(), recent['away_score'].mean()

# ─── Elo timeline (kept for recency signal) ───────────────────────────────────
def build_elo_timeline(df, k=30, initial=1500, min_year=None, shootouts=None):
    # Build penalty lookup: (date_str, home, away) -> winner
    pen_winners = {}
    if shootouts is not None and len(shootouts) > 0:
        for _, sr in shootouts.iterrows():
            pen_winners[(str(sr['date'])[:10], sr['home_team'], sr['away_team'])] = sr['winner']

    elo = {}
    records = []
    sorted_df = df[df['home_score'].notna()].sort_values('date')

    def _update(sorted_slice):
        for _, row in sorted_slice.iterrows():
            h, a = row['home_team'], row['away_team']
            r_h = elo.get(h, initial); r_a = elo.get(a, initial)
            exp_h = 1 / (1 + 10**((r_a - r_h)/400))
            hs, as_ = row['home_score'], row['away_score']
            if hs > as_:
                act_h = 1.0
            elif as_ > hs:
                act_h = 0.0
            else:
                # Check penalties — winner gets 0.65, loser gets 0.35
                key = (str(row['date'])[:10], h, a)
                pen = pen_winners.get(key)
                act_h = 0.65 if pen == h else (0.35 if pen == a else 0.5)
            gd = abs(hs - as_)
            mult = 1 + (gd>1)*0.5 + (gd>3)*0.5
            elo[h] = r_h + k * mult * (act_h - exp_h)
            elo[a] = r_a + k * mult * ((1-act_h) - (1-exp_h))
            records.append({'date': row['date'], 'team': h, 'elo': elo[h]})
            records.append({'date': row['date'], 'team': a, 'elo': elo[a]})

    if min_year is not None:
        cutoff = pd.Timestamp(f'{min_year}-01-01')
        _update(sorted_df[sorted_df['date'] < cutoff])
        for team in elo:
            elo[team] = initial + 0.75 * (elo[team] - initial)
        records.clear()
        _update(sorted_df[sorted_df['date'] >= cutoff])
    else:
        _update(sorted_df)

    return pd.DataFrame(records), elo

# ─── Fast ML Dataset Builder ──────────────────────────────────────────────────
def build_ml_dataset(df, min_year=2005):
    df_c = df[df['home_score'].notna()].copy().reset_index(drop=True)
    elo_df, _ = build_elo_timeline(df_c)
    fifa_df = load_fifa_rankings()

    home = df_c[['date','home_team','away_team','home_score','away_score','tournament','neutral']].copy()
    home.columns = ['date','team','opp','scored','conceded','tournament','neutral']; home['is_home']=True
    away = df_c[['date','away_team','home_team','away_score','home_score','tournament','neutral']].copy()
    away.columns = ['date','team','opp','scored','conceded','tournament','neutral']; away['is_home']=False
    long = pd.concat([home,away],ignore_index=True).sort_values(['team','date']).reset_index(drop=True)

    long['result']     = (long['scored']>long['conceded']).astype(float) + \
                         (long['scored']==long['conceded']).astype(float)*0.5
    long['is_wc']      = long['tournament'].isin(['FIFA World Cup','FIFA World Cup qualification']).astype(float)
    long['is_cont']    = long['tournament'].isin(ALL_CONF_TOURNAMENTS).astype(float)
    long['goal_diff']  = long['scored'] - long['conceded']
    long['clean_sheet']= (long['conceded']==0).astype(float)
    long['scored_2p']  = (long['scored']>=2).astype(float)
    long['home_result']= long['result'].where(long['is_home'], np.nan)
    long['away_result']= long['result'].where(~long['is_home'], np.nan)

    # Rolling stats
    long['roll_scored']      = long.groupby('team')['scored'].transform(lambda x: x.shift(1).ewm(span=20, min_periods=3).mean())
    long['roll_conceded']    = long.groupby('team')['conceded'].transform(lambda x: x.shift(1).ewm(span=20, min_periods=3).mean())
    long['roll_form']        = long.groupby('team')['result'].transform(lambda x: x.shift(1).ewm(span=20, min_periods=3).mean())
    long['roll_gd']          = long.groupby('team')['goal_diff'].transform(lambda x: x.shift(1).ewm(span=20, min_periods=3).mean())
    long['roll_clean_sheet'] = long.groupby('team')['clean_sheet'].transform(lambda x: x.shift(1).ewm(span=20, min_periods=3).mean())
    long['roll_scored_2p']   = long.groupby('team')['scored_2p'].transform(lambda x: x.shift(1).ewm(span=20, min_periods=3).mean())
    # ignore_na=True: NaN positions (e.g. away matches for home_form) don't
    # advance the decay counter — matches inference which only pools matching rows
    long['roll_home_form']   = long.groupby('team')['home_result'].transform(
        lambda x: x.shift(1).ewm(span=20, min_periods=2, ignore_na=True).mean()
    ).fillna(long['roll_form'])
    long['roll_away_form']   = long.groupby('team')['away_result'].transform(
        lambda x: x.shift(1).ewm(span=20, min_periods=2, ignore_na=True).mean()
    ).fillna(long['roll_form'])

    # WC/continental form: mask non-matching rows as NaN then ewm(ignore_na=True)
    long['_wc_r']   = long['result'].where(long['is_wc']   == 1, np.nan)
    long['_cont_r'] = long['result'].where(long['is_cont'] == 1, np.nan)
    long['roll_wc_form']   = long.groupby('team')['_wc_r'].transform(
        lambda x: x.shift(1).ewm(span=10, min_periods=1, ignore_na=True).mean()
    ).fillna(long['roll_form'])
    long['roll_cont_form'] = long.groupby('team')['_cont_r'].transform(
        lambda x: x.shift(1).ewm(span=10, min_periods=1, ignore_na=True).mean()
    ).fillna(long['roll_form'])
    long.drop(columns=['_wc_r', '_cont_r'], inplace=True)

    # Elo merge
    elo_sorted = elo_df.sort_values(['team','date']).reset_index(drop=True)
    elo_list = []
    for team, grp in long.groupby('team', sort=False):
        team_elo = elo_sorted[elo_sorted['team']==team][['date','elo']].copy()
        if len(team_elo)==0:
            grp2 = grp[['date']].copy(); grp2['elo']=1500.0
        else:
            grp2 = pd.merge_asof(grp[['date']].copy().sort_values('date'),
                                 team_elo.sort_values('date'), on='date', direction='backward')
            grp2['elo'] = grp2['elo'].fillna(1500.0)
        grp2.index = grp.index
        elo_list.append(grp2['elo'])
    long['elo'] = pd.concat(elo_list).sort_index()

    # FIFA points merge
    if fifa_df is not None:
        fifa_sorted = fifa_df.sort_values(['country_full','rank_date']).reset_index(drop=True)
        fifa_list = []
        for team, grp in long.groupby('team', sort=False):
            fifa_name = FIFA_NAME_MAP.get(team, team)
            team_fifa = fifa_sorted[fifa_sorted['country_full']==fifa_name][['rank_date','total_points','rank']].copy()
            team_fifa = team_fifa.rename(columns={'rank_date':'date','total_points':'fifa_pts','rank':'fifa_rank'})
            if len(team_fifa)==0:
                grp2 = grp[['date']].copy(); grp2['fifa_pts']=1500.0; grp2['fifa_rank']=100
            else:
                grp2 = pd.merge_asof(grp[['date']].copy().sort_values('date'),
                                     team_fifa.sort_values('date'), on='date', direction='backward')
                grp2['fifa_pts']  = grp2['fifa_pts'].fillna(1500.0)
                grp2['fifa_rank'] = grp2['fifa_rank'].fillna(100)
            grp2.index = grp.index
            fifa_list.append(grp2[['fifa_pts','fifa_rank']])
        long_fifa = pd.concat(fifa_list).sort_index()
        long['fifa_pts']  = long_fifa['fifa_pts']
        long['fifa_rank'] = long_fifa['fifa_rank']
    else:
        long['fifa_pts']  = long['elo']
        long['fifa_rank'] = 50

    # Primary strength = Fix #2: z-score normalized blend (matches get_primary_strength)
    long['primary_str'] = (
        0.7 * (long['fifa_pts'] - _FIFA_MEAN) / _FIFA_STD +
        0.3 * (long['elo']      - _ELO_MEAN)  / _ELO_STD
    )

    # Split home/away
    cols = ['date','team','roll_scored','roll_conceded','roll_form','roll_wc_form',
            'roll_cont_form','roll_gd','roll_clean_sheet','roll_scored_2p',
            'roll_home_form','roll_away_form','elo','fifa_pts','fifa_rank','primary_str']

    hs  = long[long['is_home']][cols].copy()
    hs.columns  = ['date','home_team','h_scored','h_conceded','h_form','h_wc_form',
                   'h_cont_form','h_gd','h_clean_sheet','h_scored_2p',
                   'h_home_form','h_away_form','h_elo','h_fifa','h_fifa_rank','h_str']
    as_ = long[~long['is_home']][cols].copy()
    as_.columns = ['date','away_team','a_scored','a_conceded','a_form','a_wc_form',
                   'a_cont_form','a_gd','a_clean_sheet','a_scored_2p',
                   'a_home_form','a_away_form','a_elo','a_fifa','a_fifa_rank','a_str']

    m = df_c.merge(hs,on=['date','home_team']).merge(as_,on=['date','away_team'])
    m = m[m['date'].dt.year>=min_year].dropna(subset=['h_scored','a_scored','h_elo','a_elo'])
    m['result']     = np.where(m['home_score']>m['away_score'],2,
                      np.where(m['home_score']==m['away_score'],1,0))
    m['is_neutral'] = m['neutral'].isin([True,'TRUE','True']).astype(int)
    m['elo_diff']   = m['h_elo']  - m['a_elo']
    m['fifa_diff']  = m['h_fifa'] - m['a_fifa']
    m['str_diff']   = m['h_str']  - m['a_str']   # Fix #2: primary signal

    # H2H: pre-build sorted history per team-pair for O(1) per-match lookup
    from bisect import bisect_left as _bisect_left
    _h2h_lookup = {}
    for _, _r in df_c[df_c['home_score'].notna()].sort_values('date').iterrows():
        _h, _a = _r['home_team'], _r['away_team']
        _hs, _as = float(_r['home_score']), float(_r['away_score'])
        _ta, _tb = (_h, _a) if _h <= _a else (_a, _h)
        _key = (_ta, _tb)
        if _key not in _h2h_lookup:
            _h2h_lookup[_key] = {'dates': [], 'results': []}
        _res = (1.0 if _hs > _as else (0.5 if _hs == _as else 0.0)) if _h == _ta \
               else (1.0 if _as > _hs else (0.5 if _hs == _as else 0.0))
        _h2h_lookup[_key]['dates'].append(_r['date'])
        _h2h_lookup[_key]['results'].append(_res)

    def _h2h_adv(home_team, away_team, match_date, n=10):
        ta, tb = (home_team, away_team) if home_team <= away_team else (away_team, home_team)
        key = (ta, tb)
        if key not in _h2h_lookup:
            return 0.0
        dates = _h2h_lookup[key]['dates']
        results = _h2h_lookup[key]['results']
        idx = _bisect_left(dates, match_date)  # exclusive: only before match_date
        if idx == 0:
            return 0.0
        past = results[max(0, idx - n):idx]
        pts = sum(past) / len(past) - 0.5
        return pts if home_team == ta else -pts

    m['h2h_diff'] = [_h2h_adv(r['home_team'], r['away_team'], r['date'])
                     for _, r in m.iterrows()]

    return m[['date',
              'h_scored','h_conceded','h_form','h_wc_form','h_cont_form',
              'h_gd','h_clean_sheet','h_scored_2p','h_home_form','h_away_form',
              'h_elo','h_fifa','h_str',
              'a_scored','a_conceded','a_form','a_wc_form','a_cont_form',
              'a_gd','a_clean_sheet','a_scored_2p','a_home_form','a_away_form',
              'a_elo','a_fifa','a_str',
              'elo_diff','fifa_diff','str_diff','h2h_diff','is_neutral','result']]
