"""
SQLite persistence layer.

Tables:
  users       — accounts (students + admins), lock/violation state
  violations  — one row per detected MALPRACTICE frame, with image path
  settings    — single-row global config (threshold, frame skip, max violations)

This is intentionally simple (no migrations, no connection pool) since it's a
single-machine research/demo dashboard. For real deployment, swap in a proper
DB + password hashing library (bcrypt/argon2) instead of sha256.
"""

import sqlite3
import hashlib
import time
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "data" / "agwfn.db"
CAPTURES_DIR = Path(__file__).parent / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)
DB_PATH.parent.mkdir(exist_ok=True)


@contextmanager
def get_conn():
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                role TEXT NOT NULL CHECK(role IN ('student','admin')),
                locked INTEGER NOT NULL DEFAULT 0,
                disallowed INTEGER NOT NULL DEFAULT 0,
                violation_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                timestamp REAL NOT NULL,
                image_path TEXT NOT NULL,
                state TEXT NOT NULL,
                flagged_classes TEXT,
                reviewed INTEGER NOT NULL DEFAULT 0,
                decision TEXT,
                FOREIGN KEY(username) REFERENCES users(username)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                threshold REAL NOT NULL DEFAULT 0.45,
                frame_skip INTEGER NOT NULL DEFAULT 15,
                max_violations INTEGER NOT NULL DEFAULT 5
            )
        """)
        conn.execute("""
            INSERT OR IGNORE INTO settings (id, threshold, frame_skip, max_violations)
            VALUES (1, 0.45, 15, 5)
        """)

    _seed_demo_accounts()


def _seed_demo_accounts():
    demo_users = [
        ("admin", "admin123", "Administrator", "admin"),
        ("student1", "student123", "Student One", "student"),
        ("student2", "student123", "Student Two", "student"),
    ]
    with get_conn() as conn:
        for username, pwd, full_name, role in demo_users:
            conn.execute("""
                INSERT OR IGNORE INTO users
                    (username, password_hash, full_name, role, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (username, hash_password(pwd), full_name, role, time.time()))


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def get_user(username: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def verify_login(username: str, password: str):
    user = get_user(username)
    if user and user["password_hash"] == hash_password(password):
        return user
    return None


def create_user(username: str, password: str, full_name: str, role: str = "student") -> bool:
    if get_user(username):
        return False
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO users (username, password_hash, full_name, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (username, hash_password(password), full_name, role, time.time()))
    return True


def list_students():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM users WHERE role = 'student' ORDER BY username"
        ).fetchall()
        return [dict(r) for r in rows]


def increment_violation(username: str) -> int:
    """Increments violation_count and returns the new count."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET violation_count = violation_count + 1 WHERE username = ?",
            (username,),
        )
        row = conn.execute(
            "SELECT violation_count FROM users WHERE username = ?", (username,)
        ).fetchone()
        return row["violation_count"]


def lock_user(username: str):
    with get_conn() as conn:
        conn.execute("UPDATE users SET locked = 1 WHERE username = ?", (username,))


def set_user_decision(username: str, allow: bool):
    """Admin decision after reviewing captured violations.
    allow=True  -> unlock, reset violation count, clear disallowed flag
    allow=False -> keep locked, mark disallowed
    """
    with get_conn() as conn:
        if allow:
            conn.execute("""
                UPDATE users SET locked = 0, disallowed = 0, violation_count = 0
                WHERE username = ?
            """, (username,))
        else:
            conn.execute("""
                UPDATE users SET locked = 1, disallowed = 1
                WHERE username = ?
            """, (username,))


def is_locked(username: str) -> bool:
    user = get_user(username)
    return bool(user and user["locked"])


# ---------------------------------------------------------------------------
# Violations
# ---------------------------------------------------------------------------

def log_violation(username: str, image_path: str, state: str, flagged_classes: str):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO violations (username, timestamp, image_path, state, flagged_classes)
            VALUES (?, ?, ?, ?, ?)
        """, (username, time.time(), image_path, state, flagged_classes))


def get_violations(username: str = None):
    with get_conn() as conn:
        if username:
            rows = conn.execute(
                "SELECT * FROM violations WHERE username = ? ORDER BY timestamp DESC",
                (username,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM violations ORDER BY timestamp DESC"
            ).fetchall()
        return [dict(r) for r in rows]


def get_pending_reviews():
    """Students who are locked but not yet given a final admin decision
    (disallowed=0 means no decision has been recorded since the last lock)."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM users
            WHERE role = 'student' AND locked = 1 AND disallowed = 0
            ORDER BY username
        """).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def get_settings():
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM settings WHERE id = 1").fetchone()
        return dict(row)


def update_settings(threshold: float, frame_skip: int, max_violations: int):
    with get_conn() as conn:
        conn.execute("""
            UPDATE settings SET threshold = ?, frame_skip = ?, max_violations = ?
            WHERE id = 1
        """, (threshold, frame_skip, max_violations))
