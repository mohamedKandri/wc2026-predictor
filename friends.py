"""
Supabase-backed friend group predictions for WC 2026.

Scoring:
  2 pts  — correct outcome (H / D / A)
  +3 pts — exact scoreline bonus (total 5 pts)

Tables required in Supabase (see README or project SQL):
  groups   (id, name, invite_code, created_by, created_at)
  members  (id, group_id, username, joined_at)
  picks    (id, group_id, username, home_team, away_team, match_date,
            predicted_winner, predicted_home_score, predicted_away_score, submitted_at)
"""
import random, string
import streamlit as st
import pandas as pd


# ── Supabase client ────────────────────────────────────────────────────────────

def is_configured() -> bool:
    return "supabase_url" in st.secrets and "supabase_key" in st.secrets


def _client():
    from supabase import create_client
    return create_client(st.secrets["supabase_url"], st.secrets["supabase_key"])


# ── Group management ───────────────────────────────────────────────────────────

def _gen_code() -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))


def create_group(name: str, username: str):
    """Create a new group, add creator as first member. Returns (group_dict, error)."""
    try:
        sb = _client()
        code = _gen_code()
        res = sb.table("groups").insert({
            "name": name.strip(),
            "invite_code": code,
            "created_by": username.strip(),
        }).execute()
        grp = res.data[0]
        sb.table("members").insert({"group_id": grp["id"], "username": username.strip()}).execute()
        return grp, None
    except Exception as e:
        return None, str(e)


def join_group(code: str, username: str):
    """Join an existing group by invite code. Returns (group_dict, error)."""
    try:
        sb = _client()
        res = sb.table("groups").select("*").eq("invite_code", code.strip().upper()).execute()
        if not res.data:
            return None, "Invalid invite code — double-check and try again."
        grp = res.data[0]
        existing = (sb.table("members").select("id")
                      .eq("group_id", grp["id"]).eq("username", username.strip()).execute())
        if not existing.data:
            sb.table("members").insert({
                "group_id": grp["id"], "username": username.strip()
            }).execute()
        return grp, None
    except Exception as e:
        return None, str(e)


def get_members(group_id: str) -> list:
    try:
        sb = _client()
        return sb.table("members").select("username,joined_at").eq("group_id", group_id).execute().data
    except Exception:
        return []


# ── Picks ──────────────────────────────────────────────────────────────────────

def submit_pick(group_id: str, username: str, home_team: str, away_team: str,
                match_date, predicted_winner: str,
                pred_home: int | None = None, pred_away: int | None = None):
    """Upsert a pick (one pick per user per match per group). Returns (ok, error)."""
    try:
        sb = _client()
        sb.table("picks").upsert({
            "group_id": group_id,
            "username": username,
            "home_team": home_team,
            "away_team": away_team,
            "match_date": str(match_date),
            "predicted_winner": predicted_winner,
            "predicted_home_score": pred_home,
            "predicted_away_score": pred_away,
        }, on_conflict="group_id,username,home_team,away_team,match_date").execute()
        return True, None
    except Exception as e:
        return False, str(e)


def get_my_picks(group_id: str, username: str) -> list:
    try:
        sb = _client()
        return (sb.table("picks").select("*")
                  .eq("group_id", group_id).eq("username", username)
                  .order("match_date").execute().data)
    except Exception:
        return []


def get_all_picks(group_id: str) -> list:
    try:
        sb = _client()
        return (sb.table("picks").select("*")
                  .eq("group_id", group_id)
                  .order("match_date").execute().data)
    except Exception:
        return []


# ── Leaderboard ────────────────────────────────────────────────────────────────

def compute_leaderboard(group_id: str, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score every pick against actual results.
    Returns a leaderboard DataFrame sorted by points desc.
    """
    picks = get_all_picks(group_id)
    if not picks:
        return pd.DataFrame()

    scored = []
    for p in picks:
        pts, status = 0, "pending"

        match = results_df[
            (results_df["home_team"] == p["home_team"]) &
            (results_df["away_team"] == p["away_team"]) &
            (results_df["home_score"].notna())
        ]
        if not match.empty:
            m = match.iloc[0]
            hs = int(m["home_score"])
            as_ = int(m["away_score"])
            actual = "home" if hs > as_ else ("draw" if hs == as_ else "away")
            if p["predicted_winner"] == actual:
                pts = 2
                exact = (p["predicted_home_score"] == hs and p["predicted_away_score"] == as_)
                if exact:
                    pts = 5
                status = "exact" if exact else "correct"
            else:
                status = "wrong"

        scored.append({
            "username":   p["username"],
            "match":      f"{p['home_team']} vs {p['away_team']}",
            "date":       p["match_date"],
            "pick":       p["predicted_winner"],
            "pred_score": (f"{p['predicted_home_score']}–{p['predicted_away_score']}"
                           if p["predicted_home_score"] is not None else "—"),
            "points":     pts,
            "status":     status,
        })

    if not scored:
        return pd.DataFrame()

    df = pd.DataFrame(scored)
    lb = (df.groupby("username")
            .agg(
                points  = ("points",  "sum"),
                correct = ("status",  lambda x: (x.isin(["correct", "exact"])).sum()),
                exact   = ("status",  lambda x: (x == "exact").sum()),
                picks   = ("status",  "count"),
            )
            .reset_index()
            .sort_values("points", ascending=False)
            .reset_index(drop=True))
    lb.index += 1
    return lb
