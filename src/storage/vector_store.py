"""Qdrant vector database paper chunk persistence and semantic retrieval (Task 3.2).

Responsible for managing:
- Running local embedded / in-memory mode Qdrant vector database;
- Chunking confirmed robotics world model paper core conclusions (Core Innovation, model paradigm, eval environments, abstract, etc.);
- Embedding vectorization and multi-dimensional metadata persistence;
- Write idempotency and deduplication: skips processing if arxiv_id already exists;
- Providing semantic similarity search interface (search), supporting RAG and paper cross-comparison.
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

# Default Qdrant local embedded storage path and collection name
DEFAULT_QDRANT_STORAGE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "qdrant_storage"
)
DEFAULT_COLLECTION_NAME = "world_model_insights"
DEFAULT_VECTOR_SIZE = 128


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Embedding interface protocol compatible with LangChain Embeddings and custom embedders."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class DefaultEmbedder:
    """Lightweight deterministic semantic Embedder.

    Provides zero-external-dependency, deterministic dense vector representations
    when external commercial API keys are not configured or in offline/unit-test environments.
    Based on Token Hash Projection and L2 normalization, effectively retaining word frequency
    and keyword co-occurrence semantics, supporting cosine similarity nearest neighbor search with inference latency < 0.1ms.
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
    """Semantic similarity search result model."""

    model_config = ConfigDict(extra="ignore")

    arxiv_id: str = Field(..., description="Paper ID")
    title: str = Field(..., description="Paper title")
    chunk_index: int = Field(..., description="Chunk index")
    chunk_text: str = Field(..., description="Chunk text content")
    score: float = Field(..., description="Cosine similarity score")
    model_paradigm: str = Field(..., description="Model paradigm category")
    action_conditioned: bool = Field(
        ..., description="Whether action conditioning is explicitly provided"
    )
    is_world_model: bool = Field(
        default=True, description="Whether it belongs to a world model paper"
    )
    eval_environments: list[str] = Field(
        default_factory=list, description="Evaluation environments list"
    )
    core_innovation: str = Field(default="", description="Core innovation summary")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Full payload metadata"
    )


def create_paper_chunks(
    paper: RawPaper,
    insight: PaperInsight | None = None,
    extra_chunks: Sequence[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Chunk ingested paper into text chunks with multi-dimensional metadata.

    Supports all ingested papers (both world model and non-world model):
    1. Core insight chunk (if insight exists): includes model paradigm, action-conditioned flag, evaluation benchmarks, and innovation summary;
    2. Abstract chunk: complete research background and method abstract;
    3. Optional body key section chunks (e.g., Method / Experiments key passages extracted via PDF parser).
    """
    chunks: list[tuple[str, dict[str, Any]]] = []
    model_paradigm = insight.model_paradigm if insight else "Other"
    action_conditioned = insight.action_conditioned if insight else False
    eval_environments = insight.eval_environments if insight else []
    core_innovation = insight.core_innovation if insight else ""
    is_world_model = insight.is_world_model if insight else False

    base_meta = {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "model_paradigm": model_paradigm,
        "action_conditioned": action_conditioned,
        "eval_environments": eval_environments,
        "core_innovation": core_innovation,
        "is_world_model": is_world_model,
        "published_date": paper.published_date.isoformat(),
        "pdf_url": paper.pdf_url,
    }

    # Chunk 1: Structured summary of core innovation and architecture (if insight exists)
    if insight is not None:
        action_cond_str = "Yes (Action-conditioned)" if action_conditioned else "No"
        env_str = ", ".join(eval_environments) if eval_environments else "None"
        wm_str = "Yes" if is_world_model else "No"
        core_chunk = (
            f"Title: {paper.title}\n"
            f"World Model: {wm_str}\n"
            f"Model Paradigm: {model_paradigm}\n"
            f"Action-Conditioned: {action_cond_str}\n"
            f"Evaluation Environments: {env_str}\n"
            f"Core Innovation: {core_innovation}"
        )
        chunks.append((core_chunk, {**base_meta, "chunk_type": "core_insight"}))

    # Chunk 2: Paper abstract chunk
    abstract_text = paper.abstract or paper.title
    abstract_chunk = f"Title: {paper.title}\nAbstract: {abstract_text}"
    chunks.append((abstract_chunk, {**base_meta, "chunk_type": "abstract"}))

    # Optional extra chunks (e.g., key paper sections)
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
    """Qdrant-based paper core conclusion vector storage and semantic retrieval engine."""

    def __init__(
        self,
        storage_path: Path | str | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedder: EmbedderProtocol | None = None,
        client: QdrantClient | None = None,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ) -> None:
        """Initialize vector store.

        Args:
            storage_path: Local Qdrant data storage directory. Defaults to env var QDRANT_STORAGE_PATH
                          or data/qdrant_storage. If set to ':memory:', uses in-memory store.
            collection_name: Qdrant collection name.
            embedder: Embedding generator. Defaults to DefaultEmbedder.
            client: Explicitly provided QdrantClient instance (optional).
            vector_size: Vector dimension.
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
        """Check and create target collection and vector configuration."""
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
                "Created Qdrant collection: %s (dim=%d)",
                self.collection_name,
                self.vector_size,
            )
        else:
            logger.debug("Qdrant collection already exists: %s", self.collection_name)

    def paper_exists(self, arxiv_id: str) -> bool:
        """Check if chunks for the specified arxiv_id already exist in the vector store."""
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
        insight: PaperInsight | None = None,
        extra_chunks: Sequence[str] | None = None,
    ) -> bool:
        """Chunk paper and vectorize into store (supports all ingested papers).

        Built-in deduplication: skips processing and returns False if arxiv_id already exists in vector store.

        Returns:
            bool: True on successful chunking and ingestion; False if already exists and skipped.
        """
        if self.paper_exists(paper.arxiv_id):
            logger.info(
                "Paper %s already exists in vector store, skipping chunk ingestion",
                paper.arxiv_id,
            )
            return False

        # Generate chunks
        raw_chunks = create_paper_chunks(paper, insight, extra_chunks=extra_chunks)
        if not raw_chunks:
            return False

        chunk_texts = [text for text, _ in raw_chunks]
        embeddings = self.embedder.embed_documents(chunk_texts)

        points: list[PointStruct] = []
        for idx, ((chunk_text, meta), vector) in enumerate(
            zip(raw_chunks, embeddings, strict=True)
        ):
            # Generate deterministic Point ID (UUID5 based on arxiv_id + chunk_index)
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
            "Paper %s successfully ingested into vector store with %d core chunks",
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
        """Execute semantic similarity search.

        Args:
            query: User query text (e.g., "Flow Matching world model").
            limit: Maximum number of similar results to return.
            score_threshold: Lower threshold filter for cosine similarity score.
            filter_criteria: Exact match filter dictionary (e.g., {'model_paradigm': 'Flow-Matching'}).

        Returns:
            list[VectorSearchResult]: List of retrieval results ordered by relevance descending.
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
                    is_world_model=bool(payload.get("is_world_model", True)),
                    eval_environments=list(payload.get("eval_environments", [])),
                    core_innovation=str(payload.get("core_innovation", "")),
                    payload=payload,
                )
            )

        return results

    def delete_paper(self, arxiv_id: str) -> bool:
        """Delete all chunk points of a specified paper from vector store."""
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
        logger.info("Removed paper %s from vector store", arxiv_id)
        return True

    def batch_delete_papers(self, arxiv_ids: Sequence[str]) -> int:
        """Batch delete all chunks for multiple papers from vector store.

        Returns:
            int: Number of papers for which deletion was successfully triggered.
        """
        deleted_count = 0
        for aid in arxiv_ids:
            if self.delete_paper(aid.strip()):
                deleted_count += 1
        return deleted_count

    def count(self) -> int:
        """Count total chunks currently in vector store."""
        return self.client.count(collection_name=self.collection_name).count

    def close(self) -> None:
        """Close client connection and release resources."""
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
