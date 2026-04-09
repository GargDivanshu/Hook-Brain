import json
import os
import sqlite3
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("HOOKBRAIN_DATA_DIR", os.path.join(BASE_DIR, "data"))
DB_PATH = os.path.join(DATA_DIR, "hookbrain.db")


def _conn():
    os.makedirs(DATA_DIR, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS scans (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                hook_text      TEXT    NOT NULL,
                brain_data     TEXT    NOT NULL,
                viral_score    REAL,
                mechanic       TEXT,
                parent_scan_id INTEGER,
                created_at     TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rewrites (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                source_scan_id INTEGER NOT NULL,
                mechanic       TEXT    NOT NULL,
                hook_text      TEXT    NOT NULL,
                why_text       TEXT,
                provider       TEXT,
                child_scan_id  INTEGER,
                created_at     TEXT    NOT NULL
            );
        """)


def save_scan(hook_text, data, parent_scan_id=None, mechanic=None):
    created_at = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO scans
               (hook_text, brain_data, viral_score, mechanic, parent_scan_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                hook_text,
                json.dumps(data),
                data.get("viral", {}).get("viral_score"),
                mechanic,
                parent_scan_id,
                created_at,
            ),
        )
        return {
            "id": cur.lastrowid,
            "hook_text": hook_text,
            "brain_data": data,
            "viral_score": data.get("viral", {}).get("viral_score"),
            "mechanic": mechanic,
            "parent_scan_id": parent_scan_id,
            "created_at": created_at,
        }


def get_history(limit=60):
    with _conn() as c:
        rows = c.execute(
            """SELECT id, hook_text, viral_score, mechanic, parent_scan_id, created_at
               FROM scans
               ORDER BY created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_scan(scan_id):
    with _conn() as c:
        row = c.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    if not row:
        return None
    data = dict(row)
    try:
        data["brain_data"] = json.loads(data["brain_data"])
    except Exception:
        pass
    return data


def save_rewrites(source_scan_id, rewrites, provider=None):
    created_at = datetime.now(timezone.utc).isoformat()
    out = []
    with _conn() as c:
        for rewrite in rewrites:
            cur = c.execute(
                """INSERT INTO rewrites
                   (source_scan_id, mechanic, hook_text, why_text, provider, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    source_scan_id,
                    rewrite.get("mechanic"),
                    rewrite.get("hook"),
                    rewrite.get("why"),
                    provider,
                    created_at,
                ),
            )
            out.append({
                "id": cur.lastrowid,
                "source_scan_id": source_scan_id,
                "mechanic": rewrite.get("mechanic"),
                "hook": rewrite.get("hook"),
                "why": rewrite.get("why"),
                "provider": provider,
                "child_scan_id": None,
                "created_at": created_at,
            })
    return out


def attach_rewrite_scan(rewrite_id, child_scan_id):
    with _conn() as c:
        c.execute(
            "UPDATE rewrites SET child_scan_id = ? WHERE id = ?",
            (child_scan_id, rewrite_id),
        )


def get_rewrites_for_scan(source_scan_id):
    with _conn() as c:
        rows = c.execute(
            """
            SELECT
                r.id,
                r.source_scan_id,
                r.mechanic,
                r.hook_text,
                r.why_text,
                r.provider,
                r.child_scan_id,
                r.created_at,
                s.viral_score AS result_viral_score,
                s.brain_data  AS result_brain_data
            FROM rewrites r
            LEFT JOIN scans s ON s.id = r.child_scan_id
            WHERE r.source_scan_id = ?
            ORDER BY r.id ASC
            """,
            (source_scan_id,),
        ).fetchall()

    out = []
    for row in rows:
        item = dict(row)
        if item.get("result_brain_data"):
            try:
                item["result_brain_data"] = json.loads(item["result_brain_data"])
            except Exception:
                pass
        out.append(item)
    return out
