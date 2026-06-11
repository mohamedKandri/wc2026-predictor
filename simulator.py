"""
World Cup 2026 Monte Carlo bracket simulator.
Runs N simulations of the full tournament using match probabilities.
Outputs championship odds for all 48 teams.
"""
import numpy as np
import pandas as pd
from collections import defaultdict

# WC 2026 group stage structure
GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Switzerland', 'Qatar'],
    'C': ['Brazil', 'Morocco', 'Scotland', 'Haiti'],
    'D': ['United States', 'Australia', 'Turkey', 'Paraguay'],
    'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Iran', 'New Zealand', 'Egypt'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

def simulate_match(home_team, away_team, predict_fn, neutral=True):
    """
    Simulate a single match.
    Returns (home_team, away_team, outcome, lambda_home, lambda_away).
    lambdas come from the prediction model so callers can use them for goal sampling.
    """
    result = predict_fn(home_team, away_team, neutral=neutral)
    if result is None:
        return home_team, away_team, 'home_win', 1.5, 1.0

    hw = result['home_win'] / 100
    d  = result['draw'] / 100
    aw = result['away_win'] / 100
    total = hw + d + aw
    hw, d, aw = hw/total, d/total, aw/total

    r = np.random.random()
    if r < hw:
        outcome = 'home_win'
    elif r < hw + d:
        outcome = 'draw'
    else:
        outcome = 'away_win'

    lh = result.get('lambda_home', 1.5)
    la = result.get('lambda_away', 1.0)
    return home_team, away_team, outcome, lh, la

def simulate_knockout_match(team1, team2, predict_fn):
    """
    Knockout: draw → AET/penalty shootout.
    Shootout probability is Elo-weighted: stronger team wins ~52-54% historically.
    P(team1 wins shootout) = sigmoid(elo_diff * 0.003)
    where elo_diff = team1_elo - team2_elo from the prediction result.
    """
    _, _, outcome, *_ = simulate_match(team1, team2, predict_fn, neutral=True)
    if outcome == 'home_win':
        return team1
    elif outcome == 'away_win':
        return team2
    else:
        # Get Elo diff from prediction for shootout weighting
        result = predict_fn(team1, team2, neutral=True)
        elo_diff = result.get('elo_diff', 0) if result else 0
        # sigmoid(elo_diff * 0.003): 200 gap → ~53%, 0 gap → 50%
        p_team1 = float(1 / (1 + np.exp(-elo_diff * 0.003)))
        return team1 if np.random.random() < p_team1 else team2

def simulate_group(group_teams, predict_fn):
    """
    Simulate a 4-team group. Returns sorted standings.
    3pts win, 1pt draw, 0pts loss.
    Goal counts sampled from prediction-model lambdas (not hardcoded 1.5).
    """
    points = defaultdict(int)
    gf = defaultdict(int)
    ga = defaultdict(int)

    for i in range(len(group_teams)):
        for j in range(i+1, len(group_teams)):
            h, a = group_teams[i], group_teams[j]
            _, _, outcome, lh, la = simulate_match(h, a, predict_fn, neutral=False)
            if outcome == 'home_win':
                points[h] += 3
                g = int(np.random.poisson(max(lh, 0.3)))
                gf[h] += g; ga[a] += g
            elif outcome == 'away_win':
                points[a] += 3
                g = int(np.random.poisson(max(la, 0.3)))
                gf[a] += g; ga[h] += g
            else:
                points[h] += 1; points[a] += 1
                g = int(np.random.poisson(max((lh + la) / 2, 0.3)))
                gf[h] += g; ga[a] += g
                gf[a] += g; ga[h] += g

    standings = sorted(
        group_teams,
        key=lambda t: (points[t], gf[t]-ga[t], gf[t], np.random.random()),
        reverse=True
    )
    return standings, points

# Fixed R32 bracket structure for WC 2026
_CROSS_PAIRINGS = [
    ('A','B'), ('B','A'),
    ('C','D'), ('D','C'),
    ('E','F'), ('F','E'),
    ('G','H'), ('H','G'),
]
_THIRD_SLOT_GROUPS = ['I', 'J', 'K', 'L']

def _assign_thirds_from_pool(slot_groups, pool):
    """
    Greedily assign third-place teams from pool to bracket slots, avoiding
    a third-place team facing a team from its own group.
    pool: mutable list of [team, group_name], sorted best-first. Modified in-place.
    slot_groups: list of group letters whose runner-up/winner each slot faces.
    Falls back to best available if no conflict-free assignment exists.
    """
    result = []
    for sg in slot_groups:
        assigned = False
        for idx, (team, tg) in enumerate(pool):
            if tg != sg:
                result.append(team)
                pool.pop(idx)
                assigned = True
                break
        if not assigned and pool:
            result.append(pool.pop(0)[0])
    return result

def simulate_tournament(predict_fn, n_simulations=10000):
    """
    Run full WC 2026 simulation n times.
    Returns probability distributions for each team.
    """
    stats = {
        'r16': defaultdict(int),
        'qf':  defaultdict(int),
        'sf':  defaultdict(int),
        'f':   defaultdict(int),
        'w':   defaultdict(int),
    }

    group_names = list(GROUPS.keys())

    for sim in range(n_simulations):
        # Group stage
        qualifiers = {}  # group → [1st, 2nd, 3rd (best 3rd)]
        third_place = []

        for g_name, teams in GROUPS.items():
            standings, points = simulate_group(teams, predict_fn)
            qualifiers[g_name] = standings
            # Track 3rd place for best-of-3rd selection
            third_team = standings[2]
            third_pts  = points[third_team]
            third_place.append((third_pts, g_name, third_team))

        # Best 8 third-place teams advance (WC 2026: 12 groups, 8 best 3rds)
        third_place.sort(reverse=True)
        thirds_pool = [[t, g] for _, g, t in third_place[:8]]  # mutable, best-first

        # Assign thirds to bracket slots, preventing same-group rematches
        ru_thirds = _assign_thirds_from_pool(list(_THIRD_SLOT_GROUPS), thirds_pool)
        w_thirds  = _assign_thirds_from_pool(list(_THIRD_SLOT_GROUPS), thirds_pool)

        # ── WC 2026 fixed R32 bracket ─────────────────────────────────────────
        r32_pairs = []
        for wg, rg in _CROSS_PAIRINGS:
            r32_pairs.append((qualifiers[wg][0], qualifiers[rg][1]))
        for i, g in enumerate(_THIRD_SLOT_GROUPS):
            r32_pairs.append((qualifiers[g][1], ru_thirds[i]))   # runner-up vs third
        for i, g in enumerate(_THIRD_SLOT_GROUPS):
            r32_pairs.append((qualifiers[g][0], w_thirds[i]))    # winner vs third
        # r32_pairs: 8 + 4 + 4 = 16 matches, 32 unique teams

        r32_flat = [t for pair in r32_pairs for t in pair]
        for t in r32_flat:
            stats['r16'][t] += 1

        def run_round(teams):
            winners = []
            for i in range(0, len(teams), 2):
                if i + 1 < len(teams):
                    w = simulate_knockout_match(teams[i], teams[i+1], predict_fn)
                    winners.append(w)
                else:
                    winners.append(teams[i])
            return winners

        # R32(32→16) → R16(16→8) → QF(8→4) → SF(4→2) → Final(2→1)
        r16_teams   = run_round(r32_flat)
        for t in r16_teams: stats['qf'][t] += 1

        qf_teams    = run_round(r16_teams)
        for t in qf_teams: stats['sf'][t] += 1

        sf_teams    = run_round(qf_teams)
        final_teams = run_round(sf_teams)
        for t in final_teams: stats['f'][t] += 1

        champion = run_round(final_teams)
        if champion:
            stats['w'][champion[0]] += 1

    # Convert to percentages
    all_teams = [t for teams in GROUPS.values() for t in teams]
    results = {}
    for team in all_teams:
        results[team] = {
            'r16':      round(stats['r16'][team] / n_simulations * 100, 1),
            'qf':       round(stats['qf'][team]  / n_simulations * 100, 1),
            'sf':       round(stats['sf'][team]  / n_simulations * 100, 1),
            'final':    round(stats['f'][team]   / n_simulations * 100, 1),
            'champion': round(stats['w'][team]   / n_simulations * 100, 1),
        }

    return results
