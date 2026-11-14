"""Compact, resumable SQLite storage for bulk Reddit collection.

Only selected posts and selected comments are retained.  API pages and the
unselected candidates used to rank them live only in memory.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class BulkStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA wal_autocheckpoint=1000")
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                config_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS work_items (
                run_id TEXT NOT NULL,
                subreddit TEXT NOT NULL,
                day TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                post_count INTEGER,
                comment_count INTEGER,
                error TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, subreddit, day)
            );
            CREATE INDEX IF NOT EXISTS work_status_idx
                ON work_items(run_id, status, subreddit, day);
            CREATE TABLE IF NOT EXISTS posts (
                run_id TEXT NOT NULL,
                id TEXT NOT NULL,
                subreddit TEXT,
                post_day TEXT NOT NULL,
                created_utc INTEGER,
                title TEXT,
                selftext TEXT,
                score INTEGER,
                num_comments INTEGER,
                author TEXT,
                url TEXT,
                link_flair_text TEXT,
                over_18 INTEGER,
                PRIMARY KEY (run_id, id)
            );
            CREATE TABLE IF NOT EXISTS comments (
                run_id TEXT NOT NULL,
                id TEXT NOT NULL,
                post_id TEXT NOT NULL,
                subreddit TEXT,
                created_utc INTEGER,
                author TEXT,
                score INTEGER,
                num_replies INTEGER NOT NULL,
                body TEXT,
                parent_id TEXT,
                PRIMARY KEY (run_id, id)
            );
            CREATE INDEX IF NOT EXISTS comments_post_idx
                ON comments(run_id, post_id);
            """
        )
        self.conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_run(self, run_id: str, config: dict[str, Any]) -> None:
        existing = self.conn.execute(
            "SELECT config_json FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        encoded = json.dumps(config, sort_keys=True)
        if existing and existing["config_json"] != encoded:
            raise ValueError("run_id already exists with a different configuration")
        if not existing:
            self.conn.execute(
                "INSERT INTO runs VALUES (?, ?, ?)", (run_id, self._now(), encoded)
            )
            self.conn.commit()

    def add_work_items(self, run_id: str, subreddits: Iterable[str], days: Iterable[str]) -> None:
        now = self._now()
        days = tuple(days)
        self.conn.executemany(
            """INSERT OR IGNORE INTO work_items
               (run_id, subreddit, day, updated_at) VALUES (?, ?, ?, ?)""",
            ((run_id, subreddit, day, now) for subreddit in subreddits for day in days),
        )
        self.conn.commit()

    def claim_next(self, run_id: str, retry_failed: bool) -> tuple[str, str] | None:
        statuses = "('pending', 'failed')" if retry_failed else "('pending')"
        row = self.conn.execute(
            f"SELECT subreddit, day FROM work_items WHERE run_id = ? AND status IN {statuses} "
            "ORDER BY subreddit, day LIMIT 1",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        updated = self.conn.execute(
            """UPDATE work_items SET status = 'running', attempts = attempts + 1,
               error = NULL, updated_at = ?
               WHERE run_id = ? AND subreddit = ? AND day = ? AND status != 'completed'""",
            (self._now(), run_id, row["subreddit"], row["day"]),
        )
        self.conn.commit()
        return (row["subreddit"], row["day"]) if updated.rowcount else None

    def recover_interrupted_work(self, run_id: str) -> int:
        """Return work claimed by a process that stopped before checkpointing it."""
        cursor = self.conn.execute(
            """UPDATE work_items SET status = 'pending', error = 'recovered after interruption',
               updated_at = ? WHERE run_id = ? AND status = 'running'""",
            (self._now(), run_id),
        )
        self.conn.commit()
        return cursor.rowcount

    def preview(self, run_id: str, table: str, limit: int) -> list[sqlite3.Row]:
        if table not in {"posts", "comments"}:
            raise ValueError("table must be posts or comments")
        return self.conn.execute(
            f"SELECT * FROM {table} WHERE run_id = ? ORDER BY created_utc DESC LIMIT ?",
            (run_id, limit),
        ).fetchall()

    def save_day(
        self,
        run_id: str,
        subreddit: str,
        day: str,
        posts: list[dict[str, Any]],
        comments: list[dict[str, Any]],
    ) -> None:
        """Atomically persist selected data and its completion checkpoint."""
        with self.conn:
            self.conn.executemany(
                """INSERT OR REPLACE INTO posts VALUES
                   (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (run_id, str(p.get("id", "")).removeprefix("t3_"), subreddit, day,
                     p.get("created_utc"), p.get("title"), p.get("selftext"), p.get("score"),
                     p.get("num_comments"), p.get("author"), p.get("url"),
                     p.get("link_flair_text"), int(bool(p.get("over_18"))))
                    for p in posts
                ],
            )
            self.conn.executemany(
                """INSERT OR REPLACE INTO comments VALUES
                   (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (run_id, str(c.get("id", "")).removeprefix("t1_"),
                     str(c.get("post_id", "")).removeprefix("t3_"), c.get("subreddit"),
                     c.get("created_utc"), c.get("author"), c.get("score"), c["num_replies"],
                     c.get("body"), c.get("parent_id"))
                    for c in comments
                ],
            )
            self.conn.execute(
                """UPDATE work_items SET status = 'completed', post_count = ?, comment_count = ?,
                   error = NULL, updated_at = ? WHERE run_id = ? AND subreddit = ? AND day = ?""",
                (len(posts), len(comments), self._now(), run_id, subreddit, day),
            )

    def mark_failed(self, run_id: str, subreddit: str, day: str, error: str) -> None:
        self.conn.execute(
            """UPDATE work_items SET status = 'failed', error = ?, updated_at = ?
               WHERE run_id = ? AND subreddit = ? AND day = ?""",
            (error[:1000], self._now(), run_id, subreddit, day),
        )
        self.conn.commit()

    def progress(self, run_id: str) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS count FROM work_items WHERE run_id = ? GROUP BY status",
            (run_id,),
        )
        return {row["status"]: row["count"] for row in rows}

    def close(self) -> None:
        self.conn.close()
