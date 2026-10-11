"""SQLite paper metadata and judgment status persistence management (Task 3.2).

Responsible for managing:
- arxiv_id (PK): Unique paper identifier
- title: Paper title
- published_date: Publish/update time (ISO string)
- is_wm: Whether it is a robotics world model (1 or 0)
- insight_json: PaperInsight structured analysis JSON string (nullable)
- created_at: Local persistence timestamp (ISO string)

Features deduplication checks: skips processing if arxiv_id already exists, ensuring write idempotency.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

from src.core.schema import PaperInsight, RawPaper

load_dotenv()

logger = logging.getLogger(__name__)

# Default SQLite database path
DEFAULT_SQLITE_PATH = Path(__file__).resolve().parents[2] / "data" / "papers.db"


class DBManager:
    """SQLite paper persistence and status record manager."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize database connection manager.

        Args:
            db_path: Database file path. Defaults to env var SQLITE_DB_PATH or data/papers.db.
                     Pass ':memory:' to enable an in-memory database (commonly used for tests).
        """
        raw_path = db_path or os.getenv("SQLITE_DB_PATH") or DEFAULT_SQLITE_PATH
        if str(raw_path) == ":memory:":
            self.db_path: str = ":memory:"
            self._mem_conn: sqlite3.Connection | None = sqlite3.connect(
                ":memory:", timeout=10.0, check_same_thread=False
            )
            self._mem_conn.row_factory = sqlite3.Row
        else:
            path_obj = Path(raw_path).resolve()
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(path_obj)
            self._mem_conn = None

        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection context manager, auto-committing transactions and releasing file connections."""
        if self._mem_conn is not None:
            with self._mem_conn:
                yield self._mem_conn
        else:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

    def _init_db(self) -> None:
        """Create table and necessary indexes."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            published_date TEXT NOT NULL,
            is_wm INTEGER NOT NULL,
            insight_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_papers_is_wm ON papers(is_wm);
        CREATE INDEX IF NOT EXISTS idx_papers_published_date ON papers(published_date);
        """
        with self._get_connection() as conn:
            conn.executescript(create_table_sql)
            conn.commit()
        logger.debug("Database initialization complete: %s", self.db_path)

    def exists(self, arxiv_id: str) -> bool:
        """Check if the given arxiv_id already exists in the database."""
        query = "SELECT 1 FROM papers WHERE arxiv_id = ? LIMIT 1;"
        with self._get_connection() as conn:
            cursor = conn.execute(query, (arxiv_id,))
            return cursor.fetchone() is not None

    def insert_paper(
        self,
        arxiv_id: str,
        title: str,
        published_date: datetime | str,
        is_wm: bool,
        insight: PaperInsight | dict[str, Any] | None = None,
        created_at: datetime | str | None = None,
    ) -> bool:
        """Insert metadata and judgment results for a single paper.

        Built-in deduplication: skips insertion and returns False if arxiv_id already exists.

        Returns:
            bool: True on successful insertion; False if already exists and skipped.
        """
        if self.exists(arxiv_id):
            logger.info(
                "Paper %s already exists in SQLite, skipping insertion", arxiv_id
            )
            return False

        # Format date
        if isinstance(published_date, datetime):
            pub_date_str = published_date.astimezone(UTC).isoformat()
        else:
            pub_date_str = str(published_date)

        # Format creation time
        if created_at is None:
            created_at_str = datetime.now(UTC).isoformat()
        elif isinstance(created_at, datetime):
            created_at_str = created_at.astimezone(UTC).isoformat()
        else:
            created_at_str = str(created_at)

        # Serialize insight
        insight_json_str: str | None = None
        if insight is not None:
            if isinstance(insight, PaperInsight):
                insight_json_str = insight.model_dump_json()
            elif isinstance(insight, dict):
                insight_json_str = json.dumps(insight, ensure_ascii=False)
            else:
                insight_json_str = str(insight)

        insert_sql = """
        INSERT INTO papers (
            arxiv_id, title, published_date, is_wm, insight_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?);
        """
        with self._get_connection() as conn:
            conn.execute(
                insert_sql,
                (
                    arxiv_id,
                    title,
                    pub_date_str,
                    1 if is_wm else 0,
                    insight_json_str,
                    created_at_str,
                ),
            )
            conn.commit()

        logger.info(
            "Paper %s successfully persisted to SQLite (is_wm=%s)", arxiv_id, is_wm
        )
        return True

    def save_paper(
        self,
        paper: RawPaper,
        insight: PaperInsight | None = None,
    ) -> bool:
        """Convenience method to save based on domain model.

        If insight is not provided, defaults to non-world model (is_wm=False).
        """
        is_wm = insight.is_world_model if insight is not None else False
        return self.insert_paper(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            published_date=paper.published_date,
            is_wm=is_wm,
            insight=insight,
        )

    def get_paper(self, arxiv_id: str) -> dict[str, Any] | None:
        """Get a single paper record by arxiv_id."""
        query = "SELECT * FROM papers WHERE arxiv_id = ?;"
        with self._get_connection() as conn:
            row = conn.execute(query, (arxiv_id,)).fetchone()
            if row is None:
                return None
            return dict(row)

    def get_paper_insight(self, arxiv_id: str) -> PaperInsight | None:
        """Get structured judgment model (PaperInsight) for a single paper."""
        record = self.get_paper(arxiv_id)
        if not record or not record.get("insight_json"):
            return None
        try:
            return PaperInsight.model_validate_json(record["insight_json"])
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning(
                "Failed to parse insight_json for paper %s: %s", arxiv_id, exc
            )
            return None

    def list_papers(
        self,
        is_wm: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query paper list, supporting filtering by world model status and pagination."""
        conditions: list[str] = []
        params: list[Any] = []

        if is_wm is not None:
            conditions.append("is_wm = ?")
            params.append(1 if is_wm else 0)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
        SELECT * FROM papers
        {where_clause}
        ORDER BY published_date DESC, created_at DESC
        LIMIT ? OFFSET ?;
        """
        params.extend([limit, offset])

        with self._get_connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [dict(row) for row in rows]

    def count_papers(self, is_wm: bool | None = None) -> int:
        """Count total papers or papers by category."""
        params: tuple[Any, ...]
        if is_wm is not None:
            query = "SELECT COUNT(*) FROM papers WHERE is_wm = ?;"
            params = (1 if is_wm else 0,)
        else:
            query = "SELECT COUNT(*) FROM papers;"
            params = ()

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return int(cursor.fetchone()[0])

    def delete_paper(self, arxiv_id: str) -> bool:
        """Delete a specific paper and its metadata from SQLite."""
        clean_id = arxiv_id.strip()
        query = "DELETE FROM papers WHERE arxiv_id = ?;"
        with self._get_connection() as conn:
            cursor = conn.execute(query, (clean_id,))
            conn.commit()
            success = cursor.rowcount > 0
            if success:
                logger.info("Removed paper %s from SQLite database", clean_id)
            return success

    def batch_delete_papers(self, arxiv_ids: Sequence[str]) -> int:
        """Batch delete specified papers from SQLite.

        Returns:
            int: Number of records successfully deleted.
        """
        if not arxiv_ids:
            return 0
        cleaned_ids = [aid.strip() for aid in arxiv_ids if aid.strip()]
        if not cleaned_ids:
            return 0

        placeholders = ", ".join(["?"] * len(cleaned_ids))
        query = f"DELETE FROM papers WHERE arxiv_id IN ({placeholders});"
        with self._get_connection() as conn:
            cursor = conn.execute(query, tuple(cleaned_ids))
            conn.commit()
            deleted_count = cursor.rowcount
            logger.info("Batch deleted %d papers from SQLite", deleted_count)
            return deleted_count

    def batch_insert_papers(
        self,
        papers: Sequence[tuple[RawPaper, PaperInsight | None]],
    ) -> tuple[int, int]:
        """Batch insert paper list.

        Returns:
            tuple[int, int]: (successfully inserted count, skipped duplicate count)
        """
        inserted_count = 0
        skipped_count = 0
        for paper, insight in papers:
            if self.save_paper(paper, insight):
                inserted_count += 1
            else:
                skipped_count += 1
        return inserted_count, skipped_count

    def close(self) -> None:
        """Close database connection (mainly used to release in-memory connection)."""
        if self._mem_conn is not None:
            self._mem_conn.close()
            self._mem_conn = None


__all__ = ["DEFAULT_SQLITE_PATH", "DBManager"]
