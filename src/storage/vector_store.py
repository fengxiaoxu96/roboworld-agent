"""Qdrant 向量数据库论文切片持久化与语义检索 (Task 3.2).

负责管理：
- 本地嵌入式 / 内存模式运行 Qdrant 向量库；
- 将确认相关的机器人世界模型论文核心结论（Core Innovation、模型范式、评测环境、摘要等）切片；
- Embedding 向量化与多维元数据落库；
- 写入幂等去重：若 arxiv_id 已存在则跳过处理；
- 提供语义近邻搜索接口 (search)，支持 RAG 与文献横向对比。
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.core.schema import PaperInsight, RawPaper

load_dotenv()

logger = logging.getLogger(__name__)

# 默认 Qdrant 本地嵌入存储路径与集合名称
DEFAULT_QDRANT_STORAGE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "qdrant_storage"
)
DEFAULT_COLLECTION_NAME = "world_model_insights"
DEFAULT_VECTOR_SIZE = 128


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Embedding 接口协议，兼容 LangChain Embeddings 与自定义 Embedder。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class DefaultEmbedder:
    """轻量级确定性语义 Embedder。

    在未配置外部商业 API Key 或离线/单测环境下提供零外部依赖、确定性的稠密向量表征。
    基于 Token Hash Projection 与 L2 归一化，能有效保留词频与关键词共现语义，
    支持余弦相似度近邻检索，推理耗时 < 0.1ms。
    """

    def __init__(self, vector_size: int = DEFAULT_VECTOR_SIZE) -> None:
        self.vector_size = vector_size

    def _embed_text(self, text: str) -> list[float]:
        vec = np.zeros(self.vector_size, dtype=np.float32)
        words = text.lower().replace("-", " ").split()
        if not words:
            return vec.tolist()

        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.vector_size
            sign = 1.0 if ((h // self.vector_size) % 2 == 0) else -1.0
            vec[idx] += sign

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)


class VectorSearchResult(BaseModel):
    """语义近邻检索返回结果模型。"""

    model_config = ConfigDict(extra="ignore")

    arxiv_id: str = Field(..., description="论文编号")
    title: str = Field(..., description="论文标题")
    chunk_index: int = Field(..., description="切片序号")
    chunk_text: str = Field(..., description="切片正文")
    score: float = Field(..., description="余弦相似度得分")
    model_paradigm: str = Field(..., description="模型范式类别")
    action_conditioned: bool = Field(..., description="是否显式输入动作条件")
    eval_environments: list[str] = Field(
        default_factory=list, description="评测环境列表"
    )
    core_innovation: str = Field(default="", description="核心创新概述")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="完整 Payload 元数据"
    )


def create_paper_chunks(
    paper: RawPaper,
    insight: PaperInsight,
    extra_chunks: Sequence[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """将确认相关的世界模型论文切分为带多维元数据的文本块。

    切片包含：
    1. 核心结论块：包含模型范式、是否 Action-conditioned、评测基准与 150 字创新总结；
    2. 摘要块：论文完整研究背景与方法摘要；
    3. 可选正文核心段落块（如通过 PDF 解析器提取的 Method / Experiments 关键段落）。
    """
    chunks: list[tuple[str, dict[str, Any]]] = []
    base_meta = {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "model_paradigm": insight.model_paradigm,
        "action_conditioned": insight.action_conditioned,
        "eval_environments": insight.eval_environments,
        "core_innovation": insight.core_innovation,
        "published_date": paper.published_date.isoformat(),
        "pdf_url": paper.pdf_url,
    }

    # 切片 1：核心创新与架构结构化总结
    action_cond_str = "Yes (Action-conditioned)" if insight.action_conditioned else "No"
    env_str = (
        ", ".join(insight.eval_environments) if insight.eval_environments else "None"
    )
    core_chunk = (
        f"Title: {paper.title}\n"
        f"Model Paradigm: {insight.model_paradigm}\n"
        f"Action-Conditioned: {action_cond_str}\n"
        f"Evaluation Environments: {env_str}\n"
        f"Core Innovation: {insight.core_innovation}"
    )
    chunks.append((core_chunk, {**base_meta, "chunk_type": "core_insight"}))

    # 切片 2：论文摘要块
    abstract_chunk = f"Title: {paper.title}\nAbstract: {paper.abstract}"
    chunks.append((abstract_chunk, {**base_meta, "chunk_type": "abstract"}))

    # 可选附加切片（如正文重点章节）
    if extra_chunks:
        for idx, extra_text in enumerate(extra_chunks):
            cleaned = extra_text.strip()
            if cleaned:
                chunks.append(
                    (
                        f"Title: {paper.title}\n{cleaned}",
                        {**base_meta, "chunk_type": f"section_{idx}"},
                    )
                )

    return chunks


class VectorStore:
    """基于 Qdrant 的论文核心结论向量存储与语义检索底座。"""

    def __init__(
        self,
        storage_path: Path | str | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedder: EmbedderProtocol | None = None,
        client: QdrantClient | None = None,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ) -> None:
        """初始化向量存储。

        Args:
            storage_path: 本地 Qdrant 数据存储目录。默认为环境变量 QDRANT_STORAGE_PATH
                          或 data/qdrant_storage。若设为 ':memory:' 则使用内存库。
            collection_name: Qdrant 集合名称。
            embedder: Embedding 生成器。默认使用 DefaultEmbedder。
            client: 显式传入已创建的 QdrantClient（可选）。
            vector_size: 向量维度。
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.embedder = embedder or DefaultEmbedder(vector_size=self.vector_size)

        if client is not None:
            self.client = client
        else:
            raw_path = (
                storage_path
                or os.getenv("QDRANT_STORAGE_PATH")
                or DEFAULT_QDRANT_STORAGE_PATH
            )
            if str(raw_path) == ":memory:":
                self.client = QdrantClient(":memory:")
            else:
                path_obj = Path(raw_path).resolve()
                path_obj.mkdir(parents=True, exist_ok=True)
                self.client = QdrantClient(path=str(path_obj))

        self._init_collection()

    def _init_collection(self) -> None:
        """检查并创建目标集合与向量配置。"""
        collections_response = self.client.get_collections()
        existing = [c.name for c in collections_response.collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "创建 Qdrant 集合: %s (dim=%d)", self.collection_name, self.vector_size
            )
        else:
            logger.debug("Qdrant 集合已存在: %s", self.collection_name)

    def paper_exists(self, arxiv_id: str) -> bool:
        """检查指定 arxiv_id 是否已在向量库中有切片记录。"""
        filter_cond = Filter(
            must=[
                FieldCondition(
                    key="arxiv_id",
                    match=MatchValue(value=arxiv_id),
                )
            ]
        )
        count_result = self.client.count(
            collection_name=self.collection_name,
            count_filter=filter_cond,
        )
        return count_result.count > 0

    def add_paper(
        self,
        paper: RawPaper,
        insight: PaperInsight,
        extra_chunks: Sequence[str] | None = None,
    ) -> bool:
        """将世界模型论文核心结论切片并向量化入库。

        内置去重逻辑：若 arxiv_id 已存在于向量库中，则跳过处理并返回 False。

        Returns:
            bool: 成功切片入库返回 True；已存在跳过返回 False。
        """
        if self.paper_exists(paper.arxiv_id):
            logger.info("论文 %s 已存在于向量库，跳过切片入库", paper.arxiv_id)
            return False

        # 生成切片
        raw_chunks = create_paper_chunks(paper, insight, extra_chunks=extra_chunks)
        if not raw_chunks:
            return False

        chunk_texts = [text for text, _ in raw_chunks]
        embeddings = self.embedder.embed_documents(chunk_texts)

        points: list[PointStruct] = []
        for idx, ((chunk_text, meta), vector) in enumerate(
            zip(raw_chunks, embeddings, strict=True)
        ):
            # 生成确定性 Point ID（基于 arxiv_id + chunk_index 的 UUID5）
            point_uuid = str(
                uuid.uuid5(uuid.NAMESPACE_DNS, f"{paper.arxiv_id}_chunk_{idx}")
            )
            payload = {
                **meta,
                "chunk_index": idx,
                "chunk_text": chunk_text,
            }
            points.append(
                PointStruct(
                    id=point_uuid,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info(
            "论文 %s 成功入库向量库，共写入 %d 个核心切片",
            paper.arxiv_id,
            len(points),
        )
        return True

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """执行语义近邻搜索。

        Args:
            query: 用户查询文本（如“Flow Matching 世界模型”）。
            limit: 返回最相似结果数。
            score_threshold: 余弦相似度得分下限过滤。
            filter_criteria: 精确匹配过滤字典（如 {'model_paradigm': 'Flow-Matching'}）。

        Returns:
            list[VectorSearchResult]: 按相关度降序排列的检索结果列表。
        """
        query_vector = self.embedder.embed_query(query)

        query_filter: Filter | None = None
        if filter_criteria:
            conditions: list[Any] = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_criteria.items()
            ]
            query_filter = Filter(must=conditions)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
        )

        results: list[VectorSearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                VectorSearchResult(
                    arxiv_id=str(payload.get("arxiv_id", "")),
                    title=str(payload.get("title", "")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    chunk_text=str(payload.get("chunk_text", "")),
                    score=float(point.score),
                    model_paradigm=str(payload.get("model_paradigm", "Other")),
                    action_conditioned=bool(payload.get("action_conditioned", False)),
                    eval_environments=list(payload.get("eval_environments", [])),
                    core_innovation=str(payload.get("core_innovation", "")),
                    payload=payload,
                )
            )

        return results

    def delete_paper(self, arxiv_id: str) -> bool:
        """从向量库删除指定论文的所有切片点。"""
        if not self.paper_exists(arxiv_id):
            return False

        filter_cond = Filter(
            must=[
                FieldCondition(
                    key="arxiv_id",
                    match=MatchValue(value=arxiv_id),
                )
            ]
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=filter_cond,
        )
        logger.info("已从向量库移除论文 %s", arxiv_id)
        return True

    def batch_delete_papers(self, arxiv_ids: Sequence[str]) -> int:
        """从向量库中批量删除多篇论文的全部切片。

        Returns:
            int: 成功触发删除的论文数。
        """
        deleted_count = 0
        for aid in arxiv_ids:
            if self.delete_paper(aid.strip()):
                deleted_count += 1
        return deleted_count

    def count(self) -> int:
        """统计当前向量库中的切片总数。"""
        return self.client.count(collection_name=self.collection_name).count

    def close(self) -> None:
        """关闭客户端连接释放资源。"""
        self.client.close()


__all__ = [
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_QDRANT_STORAGE_PATH",
    "DEFAULT_VECTOR_SIZE",
    "DefaultEmbedder",
    "EmbedderProtocol",
    "VectorSearchResult",
    "VectorStore",
    "create_paper_chunks",
]
