"""
SQLite Database Layer
---------------------
Manages persistent storage for TripWeaver AI.

Tables:
  searches          — every query the user makes (destination, type, timestamp)
  itineraries       — saved trip plans with full text
  user_preferences  — per-session travel style, budget tier, accommodation preference

All operations are synchronous and thread-safe via a module-level lock.
The DB file is created automatically on first use.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# DB lives next to this file: backend/database/tripweaver.db
_DB_PATH = Path(__file__).resolve().parent / "tripweaver.db"
_lock = threading.Lock()


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS searches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    query       TEXT    NOT NULL,
    query_type  TEXT    NOT NULL DEFAULT 'general',
    destination TEXT,
    result_summary TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS itineraries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT    NOT NULL,
    destination  TEXT    NOT NULL,
    days         INTEGER,
    budget       REAL,
    travel_style TEXT,
    content      TEXT    NOT NULL,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_preferences (
    session_id      TEXT    PRIMARY KEY,
    travel_style    TEXT,
    budget_tier     TEXT,
    accommodation   TEXT,
    home_city       TEXT,
    currency        TEXT    DEFAULT 'INR',
    interests       TEXT,   -- JSON array
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_searches_session   ON searches(session_id);
CREATE INDEX IF NOT EXISTS idx_searches_dest      ON searches(destination);
CREATE INDEX IF NOT EXISTS idx_itineraries_session ON itineraries(session_id);
"""


# ── Connection helper ─────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # better concurrent read performance
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call multiple times."""
    with _lock:
        conn = _get_conn()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()


# ── Searches ──────────────────────────────────────────────────────────────────

def save_search(
    session_id: str,
    query: str,
    query_type: str = "general",
    destination: Optional[str] = None,
    result_summary: Optional[str] = None,
) -> int:
    """Insert a search record. Returns the new row id."""
    with _lock:
        conn = _get_conn()
        try:
            cur = conn.execute(
                """INSERT INTO searches
                   (session_id, query, query_type, destination, result_summary)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, query, query_type, destination, result_summary),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def get_recent_searches(
    session_id: str,
    limit: int = 10,
    destination: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return recent searches for a session, optionally filtered by destination."""
    with _lock:
        conn = _get_conn()
        try:
            if destination:
                rows = conn.execute(
                    """SELECT * FROM searches
                       WHERE session_id = ? AND destination = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (session_id, destination, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM searches
                       WHERE session_id = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (session_id, limit),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def get_popular_destinations(limit: int = 5) -> List[Dict[str, Any]]:
    """Return the most searched destinations across all sessions."""
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                """SELECT destination, COUNT(*) as count
                   FROM searches
                   WHERE destination IS NOT NULL
                   GROUP BY destination
                   ORDER BY count DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


# ── Itineraries ───────────────────────────────────────────────────────────────

def save_itinerary(
    session_id: str,
    destination: str,
    content: str,
    days: Optional[int] = None,
    budget: Optional[float] = None,
    travel_style: Optional[str] = None,
) -> int:
    """Save a generated itinerary. Returns the new row id."""
    with _lock:
        conn = _get_conn()
        try:
            cur = conn.execute(
                """INSERT INTO itineraries
                   (session_id, destination, days, budget, travel_style, content)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, destination, days, budget, travel_style, content),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def get_itineraries(
    session_id: str,
    destination: Optional[str] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Return saved itineraries for a session."""
    with _lock:
        conn = _get_conn()
        try:
            if destination:
                rows = conn.execute(
                    """SELECT * FROM itineraries
                       WHERE session_id = ? AND destination = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (session_id, destination, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM itineraries
                       WHERE session_id = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (session_id, limit),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def get_itinerary_by_id(itinerary_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single itinerary by its ID."""
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM itineraries WHERE id = ?", (itinerary_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


# ── User Preferences ──────────────────────────────────────────────────────────

def save_preferences(session_id: str, **kwargs) -> None:
    """
    Upsert user preferences for a session.
    Accepted kwargs: travel_style, budget_tier, accommodation,
                     home_city, currency, interests (list → stored as JSON)
    """
    if "interests" in kwargs and isinstance(kwargs["interests"], list):
        kwargs["interests"] = json.dumps(kwargs["interests"])

    fields = {k: v for k, v in kwargs.items() if v is not None}
    if not fields:
        return

    with _lock:
        conn = _get_conn()
        try:
            # Check if row exists
            existing = conn.execute(
                "SELECT session_id FROM user_preferences WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            if existing:
                set_clause = ", ".join(f"{k} = ?" for k in fields)
                set_clause += ", updated_at = datetime('now')"
                conn.execute(
                    f"UPDATE user_preferences SET {set_clause} WHERE session_id = ?",
                    (*fields.values(), session_id),
                )
            else:
                cols = "session_id, " + ", ".join(fields.keys())
                placeholders = "?, " + ", ".join("?" * len(fields))
                conn.execute(
                    f"INSERT INTO user_preferences ({cols}) VALUES ({placeholders})",
                    (session_id, *fields.values()),
                )
            conn.commit()
        finally:
            conn.close()


def get_preferences(session_id: str) -> Dict[str, Any]:
    """Return preferences for a session, or empty dict if none saved."""
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM user_preferences WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return {}
            prefs = dict(row)
            if prefs.get("interests"):
                try:
                    prefs["interests"] = json.loads(prefs["interests"])
                except (json.JSONDecodeError, TypeError):
                    prefs["interests"] = []
            return prefs
        finally:
            conn.close()


def format_preferences_for_prompt(prefs: Dict[str, Any]) -> str:
    """Format saved preferences as a string to inject into the system prompt."""
    if not prefs:
        return ""
    parts = ["📋 Saved User Preferences:"]
    if prefs.get("travel_style"):
        parts.append(f"  • Travel Style: {prefs['travel_style'].title()}")
    if prefs.get("budget_tier"):
        parts.append(f"  • Budget Tier: {prefs['budget_tier'].title()}")
    if prefs.get("accommodation"):
        parts.append(f"  • Preferred Stay: {prefs['accommodation'].title()}")
    if prefs.get("home_city"):
        parts.append(f"  • Home City: {prefs['home_city']}")
    if prefs.get("interests"):
        interests = prefs["interests"]
        if isinstance(interests, list):
            parts.append(f"  • Interests: {', '.join(interests)}")
    return "\n".join(parts)


# Initialise DB on module import
init_db()
