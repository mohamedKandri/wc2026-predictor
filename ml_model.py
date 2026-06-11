"""
ML Discriminative Model — Fix B compliant.
Features: ONLY strength, form, context signals.
ZERO goal-rate features (α, β, λ) to prevent Poisson leakage.

Classifier priority (best available at runtime):
  1. XGBoost  — best accuracy, regularization, native NaN handling
  2. LightGBM — leaf-wise trees, fast, native NaN handling
  3. HistGradientBoosting (sklearn) — same leaf-wise approach, zero extra install
"""
import pandas as pd
import numpy as np
import pickle, os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_model.pkl')

# Strict signal separation:
#   ML sees only DIFFERENTIALS (relative strength) + form + context
#   Raw Elo/FIFA absolutes are reserved for Step 6 (elo_logit_constraint only)
#   This prevents double-counting: same signal influencing both ML and logit prior
FEATURE_COLS = [
    # Strength differentials only — relative, not absolute
    'str_diff', 'elo_diff', 'fifa_diff',
    # Form signals (per-team and differential)
    'h_form', 'h_wc_form', 'h_cont_form', 'h_home_form', 'h_away_form',
    'a_form', 'a_wc_form', 'a_cont_form', 'a_home_form', 'a_away_form',
    'form_diff', 'wc_form_diff', 'cont_form_diff',
    # Goal difference (net result, not rate)
    'h_gd', 'a_gd', 'gd_diff',
    # Head-to-head: home team's win rate vs away team over last 10 H2H, centred at 0
    'h2h_diff',
    # Context
    'is_neutral'
]


def _best_classifier():
    """Return the best available gradient boosting classifier."""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
        ), 'XGBoost', False  # (clf, name, needs_scaler)
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ), 'LightGBM', False
    except ImportError:
        pass

    # Zero-install fallback: sklearn's LightGBM-style implementation
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=400,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=15,
        l2_regularization=0.1,
        random_state=42,
    ), 'HistGradientBoosting', False  # handles NaN natively, no scaler needed


def add_derived_features(df):
    df = df.copy()
    df['str_diff']       = df['h_str']       - df['a_str']
    df['elo_diff']       = df['h_elo']        - df['a_elo']
    df['fifa_diff']      = df['h_fifa']       - df['a_fifa']
    df['form_diff']      = df['h_form']       - df['a_form']
    df['wc_form_diff']   = df['h_wc_form']    - df['a_wc_form']
    df['cont_form_diff'] = df['h_cont_form']  - df['a_cont_form']
    df['gd_diff']        = df['h_gd']         - df['a_gd']
    return df


def train_model(df_features, tune=False):
    df = add_derived_features(df_features)
    for col in ['h_home_form', 'h_away_form', 'a_home_form', 'a_away_form']:
        base = col.replace('home_form', 'form').replace('away_form', 'form')
        df[col] = df[col].fillna(df[base])

    # Temporal split: train on older matches, test on recent ones.
    # Random split leaks future data into training on time-series data.
    df = df.sort_values('date').reset_index(drop=True)
    split = int(len(df) * 0.8)
    X_train = df[FEATURE_COLS].iloc[:split]
    y_train = df['result'].iloc[:split]
    X_test  = df[FEATURE_COLS].iloc[split:]
    y_test  = df['result'].iloc[split:]

    clf, clf_name, needs_scaler = _best_classifier()
    print(f"[ml_model] Using {clf_name}")

    if needs_scaler:
        model = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    else:
        # Tree-based models don't need scaling; all three options handle NaN natively
        model = Pipeline([('clf', clf)])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                target_names=['Away Win', 'Draw', 'Home Win'], output_dict=True)
    print(f"[ml_model] Test accuracy ({clf_name}): {acc*100:.1f}%")

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    return model, acc, report


def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None


def predict_ml(model, home_f, away_f, is_neutral=False, elo_diff=0, fifa_diff=0, h2h_diff=0.0):
    """Fix B: only strength/form inputs, no goal rates."""
    hps = home_f.get('primary_strength', 0.0)  # 0.0 = average team on z-score scale
    aps = away_f.get('primary_strength', 0.0)
    row = {
        'h_str':       hps,
        'a_str':       aps,
        'h_elo':       home_f.get('elo', 1500),
        'a_elo':       away_f.get('elo', 1500),
        'elo_diff':    elo_diff,
        'h_fifa':      home_f.get('fifa_points', 1500),
        'a_fifa':      away_f.get('fifa_points', 1500),
        'fifa_diff':   fifa_diff,
        'h_form':      home_f['form'],
        'h_wc_form':   home_f['wc_form'],
        'h_cont_form': home_f.get('cont_form', 0.5),
        'h_home_form': home_f.get('home_form', home_f['form']),
        'h_away_form': home_f.get('away_form', home_f['form']),
        'a_form':      away_f['form'],
        'a_wc_form':   away_f['wc_form'],
        'a_cont_form': away_f.get('cont_form', 0.5),
        'a_home_form': away_f.get('home_form', away_f['form']),
        'a_away_form': away_f.get('away_form', away_f['form']),
        'h_gd':        home_f.get('avg_gd', 0),
        'a_gd':        away_f.get('avg_gd', 0),
        'h2h_diff':    float(h2h_diff),
        'is_neutral':  int(is_neutral),
    }
    df = add_derived_features(pd.DataFrame([row]))
    proba = model.predict_proba(df[FEATURE_COLS])[0]
    label = {0: 'away_win', 1: 'draw', 2: 'home_win'}
    probs = {label[c]: round(p * 100, 1) for c, p in zip(model.classes_, proba)}
    return probs, label[model.predict(df[FEATURE_COLS])[0]]
