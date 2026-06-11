-- Run this in: supabase.com → your project → SQL Editor → New query

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS groups (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name         TEXT NOT NULL,
    invite_code  TEXT UNIQUE NOT NULL,
    created_by   TEXT NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS members (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_id   UUID REFERENCES groups(id) ON DELETE CASCADE,
    username   TEXT NOT NULL,
    joined_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(group_id, username)
);

CREATE TABLE IF NOT EXISTS picks (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_id              UUID REFERENCES groups(id) ON DELETE CASCADE,
    username              TEXT NOT NULL,
    home_team             TEXT NOT NULL,
    away_team             TEXT NOT NULL,
    match_date            DATE NOT NULL,
    predicted_winner      TEXT NOT NULL CHECK (predicted_winner IN ('home','draw','away')),
    predicted_home_score  INTEGER,
    predicted_away_score  INTEGER,
    submitted_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(group_id, username, home_team, away_team, match_date)
);
