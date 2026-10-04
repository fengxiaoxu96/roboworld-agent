"""SQLite 论文元数据与研判状态持久化管理 (Task 3.2).

负责管理：
- arxiv_id (PK): 论文唯一标识
- title: 论文标题
- published_date: 发布/更新时间 (ISO 字符串)
- is_wm: 是否属于机器人世界模型 (1 或 0)
- insight_json: PaperInsight 结构化分析 JSON 字符串 (可为空)
- created_at: 本地落库时间 (ISO 字符串)

具备写入去重检查：若 arxiv_id 已存在则跳过处理，确保写入幂等性。
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

# 默认 SQLite 数据库路径
DEFAULT_SQLITE_PATH = Path(__file__).resolve().parents[2] / "data" / "papers.db"


class DBManager:
    """SQLite 论文持久化与状态记录管理器。"""

    def __init__(self, db_path: Path | str | None = None) -> None:
        """初始化数据库连接管理器。

        Args:
            db_path: 数据库文件路径。默认为环境变量 SQLITE_DB_PATH 或 data/papers.db。
                     传入 ':memory:' 可启用内存数据库（常用于测试）。
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
        """获取数据库连接上下文管理器，自动提交事务并在文件模式下释放连接。"""
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
        """建表与创建必要索引。"""
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
        logger.debug("数据库初始化完成: %s", self.db_path)

    def exists(self, arxiv_id: str) -> bool:
        """检查指定 arxiv_id 是否已经入库。"""
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
        """写入单篇论文元数据与研判结论。

        内置去重逻辑：若 arxiv_id 已存在，则跳过写入并返回 False。

        Returns:
            bool: 成功插入返回 True；已存在跳过返回 False。
        """
        if self.exists(arxiv_id):
            logger.info("论文 %s 已存在于 SQLite，跳过写入", arxiv_id)
            return False

        # 格式化日期
        if isinstance(published_date, datetime):
            pub_date_str = published_date.astimezone(UTC).isoformat()
        else:
            pub_date_str = str(published_date)

        # 格式化创建时间
        if created_at is None:
            created_at_str = datetime.now(UTC).isoformat()
        elif isinstance(created_at, datetime):
            created_at_str = created_at.astimezone(UTC).isoformat()
        else:
            created_at_str = str(created_at)

        # 序列化 insight
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

        logger.info("论文 %s 成功落库 SQLite (is_wm=%s)", arxiv_id, is_wm)
        return True

    def save_paper(
        self,
        paper: RawPaper,
        insight: PaperInsight | None = None,
    ) -> bool:
        """基于领域模型便捷保存。

        若未提供 insight，则视为初筛暂定或非世界模型 (is_wm=False)。
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
        """按 arxiv_id 获取单篇论文记录。"""
        query = "SELECT * FROM papers WHERE arxiv_id = ?;"
        with self._get_connection() as conn:
            row = conn.execute(query, (arxiv_id,)).fetchone()
            if row is None:
                return None
            return dict(row)

    def get_paper_insight(self, arxiv_id: str) -> PaperInsight | None:
        """获取单篇论文的结构化研判模型 (PaperInsight)。"""
        record = self.get_paper(arxiv_id)
        if not record or not record.get("insight_json"):
            return None
        try:
            return PaperInsight.model_validate_json(record["insight_json"])
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("解析论文 %s 的 insight_json 失败: %s", arxiv_id, exc)
            return None

    def list_papers(
        self,
        is_wm: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """查询论文列表，支持按是否世界模型过滤与分页。"""
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
        """统计论文总数或分类数量。"""
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
        """从 SQLite 删除指定论文及其元数据记录。"""
        clean_id = arxiv_id.strip()
        query = "DELETE FROM papers WHERE arxiv_id = ?;"
        with self._get_connection() as conn:
            cursor = conn.execute(query, (clean_id,))
            conn.commit()
            success = cursor.rowcount > 0
            if success:
                logger.info("已从 SQLite 数据库移除论文 %s", clean_id)
            return success

    def batch_delete_papers(self, arxiv_ids: Sequence[str]) -> int:
        """从 SQLite 批量删除指定论文列表。

        Returns:
            int: 实际成功删除的记录条数。
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
            logger.info("已从 SQLite 批量删除 %d 篇论文", deleted_count)
            return deleted_count

    def batch_insert_papers(
        self,
        papers: Sequence[tuple[RawPaper, PaperInsight | None]],
    ) -> tuple[int, int]:
        """批量写入论文列表。

        Returns:
            tuple[int, int]: (成功写入数, 跳过重复数)
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
        """关闭数据库连接（主要用于释放内存连接）。"""
        if self._mem_conn is not None:
            self._mem_conn.close()
            self._mem_conn = None


__all__ = ["DEFAULT_SQLITE_PATH", "DBManager"]
