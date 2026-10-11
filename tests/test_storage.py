"""Unit and integration tests: Hybrid persistent storage layer (Task 3.2: SQLite + Vector DB)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.core.schema import PaperInsight, RawPaper
from src.storage import DBManager, VectorSearchResult, VectorStore, create_paper_chunks


def _create_sample_paper(
    arxiv_id: str = "2401.00001",
    title: str = "WorldGym: Action-Conditioned World Models for Policy Evaluation",
    abstract: str = (
        "We present WorldGym, an action-conditioned world model framework "
        "designed for robot manipulation and simulation-to-real transfer."
    ),
    published_date: datetime | None = None,
) -> RawPaper:
    return RawPaper(
        arxiv_id=arxiv_id,
        title=title,
        authors=["Alice Smith", "Bob Jones"],
        abstract=abstract,
        categories=["cs.RO", "cs.AI"],
        published_date=published_date or datetime(2024, 1, 15, 12, 0, tzinfo=UTC),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    )


def _create_sample_insight(
    is_world_model: bool = True,
    model_paradigm: str = "Flow-Matching",
    action_conditioned: bool = True,
    eval_environments: list[str] | None = None,
    core_innovation: str = (
        "First application of Flow-Matching to robot action-conditioned latent space prediction, "
        "achieving efficient policy evaluation on the CALVIN benchmark."
    ),
) -> PaperInsight:
    return PaperInsight(
        is_world_model=is_world_model,
        model_paradigm=model_paradigm,  # type: ignore[arg-type]
        action_conditioned=action_conditioned,
        eval_environments=eval_environments or ["CALVIN", "Franka Real Robot"],
        core_innovation=core_innovation,
    )


# =========================================================================
# 1. SQLite persistence tests (DBManager)
# =========================================================================


def test_sqlite_in_memory_crud() -> None:
    db = DBManager(db_path=":memory:")

    paper = _create_sample_paper()
    insight = _create_sample_insight()

    # 1. First insertion
    inserted = db.save_paper(paper, insight)
    assert inserted is True
    assert db.exists(paper.arxiv_id) is True

    # 2. Read verification
    record = db.get_paper(paper.arxiv_id)
    assert record is not None
    assert record["arxiv_id"] == "2401.00001"
    assert record["title"] == paper.title
    assert record["is_wm"] == 1
    assert "Flow-Matching" in record["insight_json"]

    # 3. Structured Insight parsing
    loaded_insight = db.get_paper_insight(paper.arxiv_id)
    assert loaded_insight is not None
    assert loaded_insight.is_world_model is True
    assert loaded_insight.model_paradigm == "Flow-Matching"
    assert loaded_insight.action_conditioned is True


def test_sqlite_deduplication_and_idempotency() -> None:
    """Verify deduplication check: skips if arxiv_id already exists, idempotent on duplicate writes."""
    db = DBManager(db_path=":memory:")
    paper = _create_sample_paper("2401.00002")
    insight = _create_sample_insight()

    # First write succeeds
    assert db.save_paper(paper, insight) is True
    assert db.count_papers() == 1

    # Second write is deduplicated and skipped
    assert db.save_paper(paper, insight) is False
    assert db.count_papers() == 1


def test_sqlite_non_world_model_paper() -> None:
    """Verify non-world model (pre-filter skipped / failed classification) is properly recorded in SQLite."""
    db = DBManager(db_path=":memory:")
    paper = _create_sample_paper("2401.00003", title="General Video Synthesis")
    insight = _create_sample_insight(
        is_world_model=False,
        core_innovation="Pure video generation model, lacking physical action closed loop.",
    )

    assert db.save_paper(paper, insight) is True
    record = db.get_paper("2401.00003")
    assert record is not None
    assert record["is_wm"] == 0

    # List filtering test
    wm_list = db.list_papers(is_wm=True)
    non_wm_list = db.list_papers(is_wm=False)
    assert len(wm_list) == 0
    assert len(non_wm_list) == 1
    assert non_wm_list[0]["arxiv_id"] == "2401.00003"


def test_sqlite_file_persistence(tmp_path: Path) -> None:
    """Verify file-based SQLite persistence and multi-instance reading."""
    db_file = tmp_path / "papers.db"
    db1 = DBManager(db_path=db_file)

    paper = _create_sample_paper("2401.00004")
    db1.save_paper(paper)

    # Open second instance reading same file
    db2 = DBManager(db_path=db_file)
    assert db2.exists("2401.00004") is True
    assert db2.count_papers() == 1

    # Deletion test
    assert db2.delete_paper("2401.00004") is True
    assert db2.exists("2401.00004") is False


# =========================================================================
# 2. Qdrant vector database tests (VectorStore)
# =========================================================================


def test_create_paper_chunks() -> None:
    paper = _create_sample_paper()
    insight = _create_sample_insight()
    chunks = create_paper_chunks(
        paper, insight, extra_chunks=["Methodology excerpt text"]
    )

    assert len(chunks) == 3
    # Chunk 1: Core innovation
    core_text, core_meta = chunks[0]
    assert "Core Innovation" in core_text
    assert core_meta["model_paradigm"] == "Flow-Matching"
    assert core_meta["action_conditioned"] is True

    # Chunk 2: Abstract
    abs_text, abs_meta = chunks[1]
    assert "Abstract:" in abs_text
    assert abs_meta["chunk_type"] == "abstract"

    # Chunk 3: Additional section
    extra_text, extra_meta = chunks[2]
    assert "Methodology excerpt text" in extra_text
    assert extra_meta["chunk_type"] == "section_0"


def test_vector_store_add_and_deduplication() -> None:
    """Verify vector store chunk insertion and deduplication logic."""
    vstore = VectorStore(storage_path=":memory:", collection_name="test_collection")
    paper = _create_sample_paper("2401.10001")
    insight = _create_sample_insight()

    # 1. First ingestion
    added = vstore.add_paper(paper, insight)
    assert added is True
    assert vstore.paper_exists(paper.arxiv_id) is True
    # Defaults to 2 chunks (core innovation + abstract)
    assert vstore.count() == 2

    # 2. Second ingestion deduplicated and skipped, count unchanged
    added_again = vstore.add_paper(paper, insight)
    assert added_again is False
    assert vstore.count() == 2


def test_vector_store_semantic_search() -> None:
    """Verify acceptance criteria: semantic similarity search via vector store."""
    vstore = VectorStore(storage_path=":memory:", collection_name="test_search")

    # Paper 1: Flow-Matching robot arm world model
    p1 = _create_sample_paper(
        arxiv_id="2401.11111",
        title="Flow Matching World Models for Robotic Manipulation",
        abstract="We introduce continuous flow matching to learn robot dynamics models.",
    )
    ins1 = _create_sample_insight(
        model_paradigm="Flow-Matching",
        core_innovation="Robot dynamics prediction model based on continuous flow matching.",
    )

    # Paper 2: Diffusion autonomous driving world model
    p2 = _create_sample_paper(
        arxiv_id="2401.22222",
        title="Diffusion World Models in Autonomous Driving",
        abstract="A diffusion-based visual simulator for highway traffic prediction.",
    )
    ins2 = _create_sample_insight(
        model_paradigm="Diffusion-based",
        core_innovation="Multi-agent diffusion prediction model for highway driving scenarios.",
    )

    # Paper 3: Transformer discrete world model
    p3 = _create_sample_paper(
        arxiv_id="2401.33333",
        title="Discrete Autoregressive Latent Dynamics",
        abstract="Tokenizing robotics observations for autoregressive sequence modeling.",
    )
    ins3 = _create_sample_insight(
        model_paradigm="Transformer/AutoRegressive",
        core_innovation="Autoregressive Transformer discrete world model.",
    )

    assert vstore.add_paper(p1, ins1) is True
    assert vstore.add_paper(p2, ins2) is True
    assert vstore.add_paper(p3, ins3) is True

    # 1. Query "flow matching robot dynamics"
    results1 = vstore.search("flow matching robot dynamics", limit=2)
    assert len(results1) > 0
    assert isinstance(results1[0], VectorSearchResult)
    # Most relevant should be Paper 1 (Flow-Matching)
    assert results1[0].arxiv_id == "2401.11111"
    assert results1[0].model_paradigm == "Flow-Matching"
    assert results1[0].score > 0.3

    # 2. Query "autonomous driving diffusion"
    results2 = vstore.search("autonomous driving diffusion", limit=2)
    assert len(results2) > 0
    assert results2[0].arxiv_id == "2401.22222"
    assert results2[0].model_paradigm == "Diffusion-based"


def test_vector_store_filtered_search() -> None:
    """Verify vector search with metadata filtering fields."""
    vstore = VectorStore(storage_path=":memory:", collection_name="test_filter")

    p1 = _create_sample_paper(arxiv_id="2401.40001", title="Model A")
    ins1 = _create_sample_insight(model_paradigm="Diffusion-based")
    p2 = _create_sample_paper(arxiv_id="2401.40002", title="Model B")
    ins2 = _create_sample_insight(model_paradigm="Flow-Matching")

    vstore.add_paper(p1, ins1)
    vstore.add_paper(p2, ins2)

    # Exact filter Flow-Matching
    filtered_results = vstore.search(
        "world model",
        limit=5,
        filter_criteria={"model_paradigm": "Flow-Matching"},
    )
    assert len(filtered_results) > 0
    assert all(r.model_paradigm == "Flow-Matching" for r in filtered_results)


def test_vector_store_delete() -> None:
    vstore = VectorStore(storage_path=":memory:", collection_name="test_del")
    paper = _create_sample_paper("2401.50001")
    insight = _create_sample_insight()

    vstore.add_paper(paper, insight)
    assert vstore.count() == 2

    assert vstore.delete_paper("2401.50001") is True
    assert vstore.count() == 0
    assert vstore.paper_exists("2401.50001") is False


# =========================================================================
# 3. Hybrid persistence coordination and workflow simulation tests
# =========================================================================


def test_hybrid_storage_workflow(tmp_path: Path) -> None:
    """Simulate SQLite and Vector DB hybrid persistence coordination in Phase 4 workflow:

    - Relevant papers (is_wm=True): written to SQLite and chunked into vector store;
    - Irrelevant papers (is_wm=False): recorded in SQLite only, not added to vector store;
    - Duplicate writes: all idempotently skipped.
    """
    db = DBManager(db_path=tmp_path / "papers.db")
    vstore = VectorStore(storage_path=tmp_path / "qdrant")

    # 1. Paper A (confirmed world model)
    paper_wm = _create_sample_paper("2401.90001", title="Real World Model")
    insight_wm = _create_sample_insight(is_world_model=True)

    db_res_wm = db.save_paper(paper_wm, insight_wm)
    vstore_res_wm = vstore.add_paper(paper_wm, insight_wm)
    assert db_res_wm is True
    assert vstore_res_wm is True

    # 2. Paper B (non-world model, e.g., pure image generation)
    paper_skip = _create_sample_paper("2401.90002", title="Pure Image GAN")
    insight_skip = _create_sample_insight(
        is_world_model=False,
        core_innovation="General image generation without state dynamics modeling.",
    )

    db_res_skip = db.save_paper(paper_skip, insight_skip)
    # Workflow simulation: routed conditionally, non-world models skip vector store
    assert db_res_skip is True
    assert vstore.paper_exists(paper_skip.arxiv_id) is False

    # 3. Check database state
    assert db.count_papers(is_wm=True) == 1
    assert db.count_papers(is_wm=False) == 1
    assert vstore.count() == 2  # Only 2 chunks from paper_wm

    # 4. Duplicate write idempotency verification
    assert db.save_paper(paper_wm, insight_wm) is False
    assert vstore.add_paper(paper_wm, insight_wm) is False
    assert db.count_papers() == 2
    assert vstore.count() == 2

    # 5. Vector search only retrieves relevant world model literature
    search_hits = vstore.search("world model", limit=5)
    assert len(search_hits) > 0
    assert all(hit.arxiv_id == "2401.90001" for hit in search_hits)


def test_batch_delete_papers(tmp_path: Path) -> None:
    """Test batch paper deletion functionality in DBManager and VectorStore."""
    db = DBManager(db_path=tmp_path / "batch_del.db")
    vs = VectorStore(storage_path=tmp_path / "batch_del_vs")

    p1 = _create_sample_paper("2408.00001", "Paper 1")
    p2 = _create_sample_paper("2408.00002", "Paper 2")
    ins1 = _create_sample_insight()
    ins2 = _create_sample_insight()

    db.save_paper(p1, ins1)
    db.save_paper(p2, ins2)
    vs.add_paper(p1, ins1)
    vs.add_paper(p2, ins2)

    assert db.count_papers() == 2
    assert vs.paper_exists("2408.00001") is True
    assert vs.paper_exists("2408.00002") is True

    # Batch deletion
    del_count_db = db.batch_delete_papers(["2408.00001", "2408.00002", "non_existing"])
    assert del_count_db == 2
    assert db.count_papers() == 0

    del_count_vs = vs.batch_delete_papers(["2408.00001", "2408.00002", "non_existing"])
    assert del_count_vs == 2
    assert vs.paper_exists("2408.00001") is False
    assert vs.paper_exists("2408.00002") is False
