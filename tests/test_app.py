"""Streamlit interactive Q&A interface and RAG engine test suite (Task 5.2).

Test coverage:
1. RAGSynthesizer vector search and answer generation (comparative analysis, general Q&A, empty results fallback, LLM mode);
2. Original literature citation extraction and formatting (arXiv ID, title, links, action-conditioned and environments);
3. Multidimensional category filtering and keyword data filtering logic (filter_papers_data);
4. Headless end-to-end frontend startup and component interaction tests based on Streamlit AppTest.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.app import (
    LANGUAGES,
    RAGSynthesizer,
    TRANSLATIONS,
    batch_delete_paper_records,
    delete_paper_record,
    filter_papers_data,
    get_text,
    parse_insight_payload,
    sync_db_to_vector_store,
)
from src.core.schema import PaperInsight, RawPaper
from src.storage.db_manager import DBManager
from src.storage.vector_store import VectorSearchResult, VectorStore

APP_SCRIPT_PATH = str(Path(__file__).resolve().parents[1] / "src" / "app.py")


@pytest.fixture
def sample_papers_and_insights() -> list[tuple[RawPaper, PaperInsight]]:
    """Provide standard world model and comparison samples for testing."""
    p1 = RawPaper(
        arxiv_id="2405.00001",
        title="FlowWorld: Flow Matching for Action-Conditioned Dynamics",
        authors=["Alice", "Bob"],
        abstract="We introduce FlowWorld, an action-conditioned continuous normalizing flow world model.",
        categories=["cs.RO", "cs.AI"],
        published_date=datetime(2026, 9, 20, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/2405.00001",
    )
    ins1 = PaperInsight(
        is_world_model=True,
        model_paradigm="Flow-Matching",
        action_conditioned=True,
        eval_environments=["CALVIN", "Real Robot Franka"],
        core_innovation="High-speed action-conditioned physical dynamics modeling based on continuous flow matching",
    )

    p2 = RawPaper(
        arxiv_id="2405.00002",
        title="DiffDynamics: Diffusion World Models for Manipulation",
        authors=["Charlie", "David"],
        abstract="DiffDynamics models robotic state transitions via latent diffusion with explicit action guidance.",
        categories=["cs.RO"],
        published_date=datetime(2026, 9, 21, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/2405.00002",
    )
    ins2 = PaperInsight(
        is_world_model=True,
        model_paradigm="Diffusion-based",
        action_conditioned=True,
        eval_environments=["RLBench", "Isaac Gym"],
        core_innovation="Denoising diffusion dynamics and long-horizon closed-loop planning",
    )

    p3 = RawPaper(
        arxiv_id="2405.00003",
        title="GenVideo: Open-domain Video Generation",
        authors=["Eve"],
        abstract="A text-to-video foundation model without action conditioning.",
        categories=["cs.CV"],
        published_date=datetime(2026, 9, 22, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/2405.00003",
    )
    ins3 = PaperInsight(
        is_world_model=False,
        model_paradigm="Diffusion-based",
        action_conditioned=False,
        eval_environments=[],
        core_innovation="Open-source text-to-video foundation model",
    )

    return [(p1, ins1), (p2, ins2), (p3, ins3)]


@pytest.fixture
def populated_vector_store(
    sample_papers_and_insights: list[tuple[RawPaper, PaperInsight]],
) -> VectorStore:
    """Initialize in-memory vector store containing test chunks."""
    vs = VectorStore(storage_path=":memory:")
    for paper, insight in sample_papers_and_insights:
        vs.add_paper(paper, insight)
    return vs


def test_rag_synthesizer_empty_results(populated_vector_store: VectorStore) -> None:
    """Test graceful suggestion prompts when search results are empty."""
    synthesizer = RAGSynthesizer(vector_store=populated_vector_store)
    empty_res: list[VectorSearchResult] = []
    answer = synthesizer.generate_answer(
        "Quantum computing in bio-pharma", empty_res, lang="en"
    )

    assert "No highly relevant literature found" in answer
    assert "Suggestions" in answer


def test_rag_synthesizer_comparison_query(
    populated_vector_store: VectorStore,
) -> None:
    """Test comparative Q&A across multiple papers (tables, mechanism analysis, and citations)."""
    synthesizer = RAGSynthesizer(vector_store=populated_vector_store)
    query = "Compare Flow Matching and Diffusion world model methods"
    results = synthesizer.search(query, limit=5)

    assert len(results) > 0
    answer = synthesizer.generate_answer(query, results, lang="en")

    # Verify comparison table and key fields
    assert "Comparative Analysis" in answer
    assert "| Paradigm |" in answer
    assert "Flow-Matching" in answer
    assert "Diffusion-based" in answer
    assert "Action-conditioned" in answer

    # Verify standard citations
    assert "### 📚 Original Literature Citations" in answer
    assert "2405.00001" in answer
    assert "https://arxiv.org/pdf/2405.00001" in answer
    assert "https://arxiv.org/abs/2405.00001" in answer


def test_rag_synthesizer_general_query(
    populated_vector_store: VectorStore,
) -> None:
    """Test general academic Q&A summary."""
    synthesizer = RAGSynthesizer(vector_store=populated_vector_store)
    query = "FlowWorld dynamics on CALVIN benchmark"
    results = synthesizer.search(query, limit=3)

    assert len(results) > 0
    answer = synthesizer.generate_answer(query, results, lang="en")

    assert "Literature Analysis & Paper Summary" in answer
    assert "Original Literature Citations" in answer
    assert "2405.00001" in answer


def test_rag_synthesizer_with_mock_llm(
    populated_vector_store: VectorStore,
) -> None:
    """Test invocation branch when external LLM is configured."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "[LLM Analysis] Flow Matching converges faster than diffusion.\n\n### 📚 References\n- 2405.00001"
    mock_llm.invoke.return_value = mock_response

    synthesizer = RAGSynthesizer(vector_store=populated_vector_store, llm=mock_llm)
    results = synthesizer.search("Flow Matching features", limit=2)
    answer = synthesizer.generate_answer("Flow Matching features", results, lang="en")

    mock_llm.invoke.assert_called_once()
    assert "[LLM Analysis]" in answer


def test_filter_papers_data() -> None:
    """Test multidimensional condition combination filtering logic."""
    mock_db_papers = [
        {
            "arxiv_id": "2405.00001",
            "title": "Flow Matching World Models",
            "published_date": "2026-09-20",
            "is_wm": 1,
            "insight_json": json.dumps(
                {
                    "model_paradigm": "Flow-Matching",
                    "action_conditioned": True,
                    "eval_environments": ["CALVIN"],
                    "core_innovation": "Flow Matching continuous flow world model",
                }
            ),
        },
        {
            "arxiv_id": "2405.00002",
            "title": "Diffusion Dynamics for Robots",
            "published_date": "2026-09-21",
            "is_wm": 1,
            "insight_json": json.dumps(
                {
                    "model_paradigm": "Diffusion-based",
                    "action_conditioned": False,
                    "eval_environments": ["RLBench"],
                    "core_innovation": "Diffusion prediction without action conditioning",
                }
            ),
        },
        {
            "arxiv_id": "2405.00003",
            "title": "Non-WM Video Generator",
            "published_date": "2026-09-22",
            "is_wm": 0,
            "insight_json": "{}",
        },
    ]

    # 1. Default to world models only
    res1 = filter_papers_data(mock_db_papers, world_model_only=True)
    assert len(res1) == 2

    # 2. Paradigm filter
    res2 = filter_papers_data(
        mock_db_papers,
        selected_paradigms=["Flow-Matching"],
        world_model_only=True,
    )
    assert len(res2) == 1
    assert res2[0]["arxiv_id"] == "2405.00001"

    # 3. Action-conditioned only
    res3 = filter_papers_data(
        mock_db_papers,
        action_conditioned_only=True,
        world_model_only=True,
    )
    assert len(res3) == 1
    assert res3[0]["arxiv_id"] == "2405.00001"

    # 4. Keyword search
    res4 = filter_papers_data(
        mock_db_papers,
        search_keyword="diffusion",
        world_model_only=True,
    )
    assert len(res4) == 1
    assert res4[0]["arxiv_id"] == "2405.00002"

    # 5. Show all (including non-world-models)
    res5 = filter_papers_data(mock_db_papers, world_model_only=False)
    assert len(res5) == 3


def test_parse_insight_payload() -> None:
    """Test insight_json string parsing safety."""
    assert parse_insight_payload(None) == {}
    assert parse_insight_payload("") == {}
    assert parse_insight_payload("invalid json") == {}
    data = parse_insight_payload('{"model_paradigm": "Flow-Matching"}')
    assert data.get("model_paradigm") == "Flow-Matching"


def test_streamlit_app_renders_reliably(tmp_path: Any, monkeypatch: Any) -> None:
    """Use Streamlit AppTest to verify frontend startup and basic component rendering."""
    db_file = tmp_path / "test_papers.db"
    qdrant_dir = tmp_path / "test_qdrant"

    # Prepopulate a test record
    db = DBManager(db_path=db_file)
    db.insert_paper(
        arxiv_id="2405.12345",
        title="Sample Embodied World Model",
        published_date="2026-09-25T10:00:00Z",
        is_wm=True,
        insight={
            "model_paradigm": "Transformer/AutoRegressive",
            "action_conditioned": True,
            "eval_environments": ["CALVIN"],
            "core_innovation": "Autoregressive embodied dynamics model",
        },
    )

    monkeypatch.setenv("SQLITE_DB_PATH", str(db_file))
    monkeypatch.setenv("QDRANT_STORAGE_PATH", str(qdrant_dir))
    st.cache_resource.clear()

    # Load app.py with AppTest
    at = AppTest.from_file(APP_SCRIPT_PATH, default_timeout=15)
    at.run()

    # Verify no unhandled exceptions
    assert not at.exception

    # Verify primary page title
    assert len(at.title) >= 1
    assert "RoboWorld Agent" in at.title[0].value

    # Verify tabs exist
    assert len(at.tabs) >= 2


def test_streamlit_app_chat_interaction(tmp_path: Any, monkeypatch: Any) -> None:
    """Use Streamlit AppTest to verify Q&A dialog interaction."""
    db_file = tmp_path / "test_papers.db"
    qdrant_dir = tmp_path / "test_qdrant"

    monkeypatch.setenv("SQLITE_DB_PATH", str(db_file))
    monkeypatch.setenv("QDRANT_STORAGE_PATH", str(qdrant_dir))
    st.cache_resource.clear()

    at = AppTest.from_file(APP_SCRIPT_PATH, default_timeout=15)
    at.run()
    assert not at.exception

    # Verify initial assistant welcome message
    assert len(at.chat_message) >= 1

    # Simulate user input question
    at.chat_input[0].set_value("Compare Flow Matching and Diffusion models").run()
    assert not at.exception

    # Verify Q&A messages update (user query + assistant answer)
    assert len(at.chat_message) >= 3
    user_msg = at.chat_message[1].markdown[0].value
    assistant_msg = at.chat_message[2].markdown[0].value
    assert "Compare Flow Matching and Diffusion models" in user_msg
    assert len(assistant_msg) > 10


def test_rag_synthesizer_searches_all_papers_including_non_wm(
    populated_vector_store: VectorStore,
) -> None:
    """Test that RAG searches all literature, including non-world-model papers (e.g. GenVideo)."""
    synthesizer = RAGSynthesizer(vector_store=populated_vector_store)
    # Query non-world-model literature
    results = synthesizer.search("GenVideo Open-domain Video Generation", limit=3)
    assert len(results) > 0
    matched_ids = [r.arxiv_id for r in results]
    assert "2405.00003" in matched_ids

    # Generate answer to verify it includes the non-world-model literature
    answer = synthesizer.generate_answer(
        "Introduce GenVideo video generation foundation", results, lang="en"
    )
    assert "2405.00003" in answer
    assert "GenVideo" in answer


def test_sync_db_to_vector_store(tmp_path: Path) -> None:
    """Test syncing un-chunked papers from SQLite to Qdrant vector store."""
    db = DBManager(db_path=tmp_path / "sync_test.db")
    vs = VectorStore(storage_path=tmp_path / "sync_qdrant")

    # Insert one WM and one non-WM paper into SQLite
    db.insert_paper(
        arxiv_id="2406.11111",
        title="WM Paper",
        published_date=datetime(2026, 9, 20, tzinfo=UTC),
        is_wm=True,
        insight={"model_paradigm": "Diffusion-based", "core_innovation": "WM dyn"},
    )
    db.insert_paper(
        arxiv_id="2406.22222",
        title="Non-WM Video Paper",
        published_date=datetime(2026, 9, 21, tzinfo=UTC),
        is_wm=False,
    )

    # Vector store is initially empty
    assert vs.paper_exists("2406.11111") is False
    assert vs.paper_exists("2406.22222") is False

    # Execute sync
    synced = sync_db_to_vector_store(db, vs)
    assert synced == 2
    assert vs.paper_exists("2406.11111") is True
    assert vs.paper_exists("2406.22222") is True

    # Sync again: should be idempotent and skip
    synced_again = sync_db_to_vector_store(db, vs)
    assert synced_again == 0


def test_delete_paper_record(tmp_path: Path) -> None:
    """Test delete_paper_record synchronizes cleanup in SQLite and Qdrant."""
    db = DBManager(db_path=tmp_path / "del_record.db")
    vs = VectorStore(storage_path=tmp_path / "del_record_vs")

    p = RawPaper(
        arxiv_id="2407.99999",
        title="Test Delete Paper",
        authors=["Author"],
        abstract="Abstract",
        categories=["cs.RO"],
        published_date=datetime(2026, 9, 20, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/2407.99999",
    )
    ins = PaperInsight(
        is_world_model=True,
        model_paradigm="Flow-Matching",
        action_conditioned=True,
        eval_environments=["CALVIN"],
        core_innovation="Innovation",
    )

    db.save_paper(p, ins)
    vs.add_paper(p, ins)

    assert db.exists("2407.99999") is True
    assert vs.paper_exists("2407.99999") is True

    # Execute synchronized deletion
    success = delete_paper_record("2407.99999", db, vs)
    assert success is True
    assert db.exists("2407.99999") is False
    assert vs.paper_exists("2407.99999") is False


def test_batch_delete_paper_records(tmp_path: Path) -> None:
    """Test batch_delete_paper_records batch-synchronizes cleanup in SQLite and Qdrant."""
    db = DBManager(db_path=tmp_path / "batch_del.db")
    vs = VectorStore(storage_path=tmp_path / "batch_del_vs")

    for aid in ["2401.00001", "2401.00002", "2401.00003"]:
        p = RawPaper(
            arxiv_id=aid,
            title=f"Paper {aid}",
            authors=["Author"],
            abstract="Abstract",
            categories=["cs.RO"],
            published_date=datetime(2026, 9, 20, tzinfo=UTC),
            pdf_url=f"https://arxiv.org/pdf/{aid}",
        )
        ins = PaperInsight(
            is_world_model=True,
            model_paradigm="Flow-Matching",
            action_conditioned=True,
            eval_environments=["CALVIN"],
            core_innovation="Innovation",
        )
        db.save_paper(p, ins)
        vs.add_paper(p, ins)

    assert db.exists("2401.00001") is True
    assert db.exists("2401.00002") is True
    assert db.exists("2401.00003") is True

    # Batch delete two papers
    del_count = batch_delete_paper_records(["2401.00001", "2401.00002"], db, vs)
    assert del_count == 2
    assert db.exists("2401.00001") is False
    assert db.exists("2401.00002") is False
    assert db.exists("2401.00003") is True
    assert vs.paper_exists("2401.00001") is False
    assert vs.paper_exists("2401.00002") is False
    assert vs.paper_exists("2401.00003") is True


def test_streamlit_app_delete_interaction(tmp_path: Any, monkeypatch: Any) -> None:
    """Use Streamlit AppTest to test single paper deletion and batch deletion interactions with sync cleanup."""
    db_file = tmp_path / "test_del_app.db"
    qdrant_dir = tmp_path / "test_del_app_vs"

    db = DBManager(db_path=db_file)
    db.insert_paper(
        arxiv_id="2405.88888",
        title="Single Paper To Delete",
        published_date="2026-09-25T10:00:00Z",
        is_wm=True,
    )
    db.insert_paper(
        arxiv_id="2405.11111",
        title="Batch Paper One",
        published_date="2026-09-26T10:00:00Z",
        is_wm=True,
    )
    db.insert_paper(
        arxiv_id="2405.22222",
        title="Batch Paper Two",
        published_date="2026-09-27T10:00:00Z",
        is_wm=True,
    )
    db.insert_paper(
        arxiv_id="2405.33333",
        title="Filtered Paper Distinct",
        published_date="2026-09-28T10:00:00Z",
        is_wm=True,
    )

    monkeypatch.setenv("SQLITE_DB_PATH", str(db_file))
    monkeypatch.setenv("QDRANT_STORAGE_PATH", str(qdrant_dir))
    st.cache_resource.clear()

    at = AppTest.from_file(APP_SCRIPT_PATH, default_timeout=15)
    at.run()
    assert not at.exception
    assert db.exists("2405.88888") is True

    # 1. Verify sidebar no longer contains input deletion components
    sidebar_text_input_keys = [w.key for w in at.sidebar.text_input]
    assert "sidebar_del_input" not in sidebar_text_input_keys
    sidebar_button_keys = [w.key for w in at.sidebar.button]
    assert "sidebar_del_btn" not in sidebar_button_keys

    # 2. Verify single card delete button and confirmation dialog in library tab
    assert len(at.tabs) >= 2
    del_btn = at.tabs[1].button(key="del_btn_2405.88888")
    assert del_btn is not None
    del_btn.click().run()
    assert not at.exception
    # Verify confirmation dialog pops up; record not yet physically deleted from database
    assert db.exists("2405.88888") is True
    # Verify cancel button in dialog: after clicking cancel, record remains and view stays on library tab
    cancel_btn = at.button(key="btn_dialog_cancel_delete")
    assert cancel_btn is not None
    cancel_btn.click().run()
    assert not at.exception
    assert db.exists("2405.88888") is True
    assert at.session_state["active_tab_id"] == "library"

    # Click card delete button again to trigger dialog, then click confirm delete
    del_btn = at.tabs[1].button(key="del_btn_2405.88888")
    del_btn.click().run()
    confirm_btn = at.button(key="btn_dialog_confirm_delete")
    assert confirm_btn is not None
    confirm_btn.click().run()
    assert not at.exception
    # Verify record is physically deleted from database and active tab remains library
    assert db.exists("2405.88888") is False
    assert at.session_state["active_tab_id"] == "library"

    # 3. Verify batch deletion without filtering (select 2405.11111 and 2405.22222)
    chk1 = at.tabs[1].checkbox(key="paper_select_2405.11111")
    chk2 = at.tabs[1].checkbox(key="paper_select_2405.22222")
    assert chk1 is not None and chk2 is not None
    chk1.check().run()
    chk2.check().run()
    assert not at.exception

    batch_del_btn = at.tabs[1].button(key="btn_batch_delete")
    assert batch_del_btn is not None
    batch_del_btn.click().run()
    assert not at.exception
    # Dialog triggered; records still exist before confirmation
    assert db.exists("2405.11111") is True
    assert db.exists("2405.22222") is True
    # Click confirm delete in dialog
    at.button(key="btn_dialog_confirm_delete").click().run()
    assert not at.exception
    assert db.exists("2405.11111") is False
    assert db.exists("2405.22222") is False
    assert db.exists("2405.33333") is True
    assert at.session_state["active_tab_id"] == "library"

    # 4. Verify batch deletion with filtering
    # Search "Distinct" keyword to filter only 2405.33333
    at.sidebar.text_input[0].set_value("Distinct").run()
    assert not at.exception
    chk_filtered = at.tabs[1].checkbox(key="paper_select_2405.33333")
    assert chk_filtered is not None
    chk_filtered.check().run()
    assert not at.exception
    at.tabs[1].button(key="btn_batch_delete").click().run()
    assert not at.exception
    assert db.exists("2405.33333") is True
    at.button(key="btn_dialog_confirm_delete").click().run()
    assert not at.exception
    assert db.exists("2405.33333") is False
    assert at.session_state["active_tab_id"] == "library"

    # 5. Verify select all and deselect all button interactions
    db.insert_paper(
        arxiv_id="2405.44444",
        title="Check All Paper A",
        published_date="2026-09-29T10:00:00Z",
        is_wm=True,
    )
    db.insert_paper(
        arxiv_id="2405.55555",
        title="Check All Paper B",
        published_date="2026-09-30T10:00:00Z",
        is_wm=True,
    )
    at.sidebar.text_input[0].set_value("").run()
    assert not at.exception

    # Click select all
    at.tabs[1].button(key="btn_select_all").click().run()
    assert not at.exception
    assert at.tabs[1].checkbox(key="paper_select_2405.44444").value is True
    assert at.tabs[1].checkbox(key="paper_select_2405.55555").value is True

    # Click deselect all
    at.tabs[1].button(key="btn_deselect_all").click().run()
    assert not at.exception
    assert at.tabs[1].checkbox(key="paper_select_2405.44444").value is False
    assert at.tabs[1].checkbox(key="paper_select_2405.55555").value is False


def test_get_text_translations() -> None:
    """Test get_text correctly returns translations and supports formatting."""
    assert get_text("metric_total", "zh") == TRANSLATIONS["zh"]["metric_total"]
    assert get_text("metric_total", "en") == "Total Papers"

    zh_del = get_text("delete_success", "zh", arxiv_id="2405.00001")
    en_del = get_text("delete_success", "en", arxiv_id="2405.00001")
    assert (
        "2405.00001" in zh_del
        and TRANSLATIONS["zh"]["delete_success"].format(arxiv_id="2405.00001") == zh_del
    )
    assert "2405.00001" in en_del and "Successfully" in en_del


def test_rag_synthesizer_bilingual_answers(
    populated_vector_store: VectorStore,
) -> None:
    """Test RAGSynthesizer retrieval and comparative output in English mode."""
    synthesizer = RAGSynthesizer(vector_store=populated_vector_store)

    # Empty result test
    empty_ans_en = synthesizer.generate_answer("non-existent-query-xyz", [], lang="en")
    assert "No highly relevant literature found" in empty_ans_en
    assert "Suggestions" in empty_ans_en

    # Comparative Q&A English test
    results = synthesizer.search("Flow Matching vs Diffusion", limit=3)
    assert len(results) > 0
    ans_en = synthesizer.generate_answer(
        "Compare Flow Matching and Diffusion", results, lang="en"
    )
    assert "Comparative Analysis" in ans_en
    assert "Original Literature Citations" in ans_en
    assert "arXiv ID" in ans_en


def test_streamlit_app_language_switch(tmp_path: Any, monkeypatch: Any) -> None:
    """Use Streamlit AppTest to verify language selector switching functionality."""
    db_file = tmp_path / "test_lang_app.db"
    qdrant_dir = tmp_path / "test_lang_app_vs"

    db = DBManager(db_path=db_file)
    db.insert_paper(
        arxiv_id="2405.12345",
        title="Test World Model Paper",
        published_date="2026-09-25T10:00:00Z",
        is_wm=True,
    )

    monkeypatch.setenv("SQLITE_DB_PATH", str(db_file))
    monkeypatch.setenv("QDRANT_STORAGE_PATH", str(qdrant_dir))
    st.cache_resource.clear()

    at = AppTest.from_file(APP_SCRIPT_PATH, default_timeout=15)
    at.run()
    assert not at.exception

    # 1. Default language is Chinese
    assert TRANSLATIONS["zh"]["app_title"].split()[0] in at.title[0].value
    assert TRANSLATIONS["zh"]["tab_chat"] in at.tabs[0].label
    lang_selector = at.sidebar.selectbox(key="language_select")
    assert lang_selector.value == LANGUAGES["zh"]
    assert at.session_state["active_tab_id"] == "chat"

    # 2. Switch to English under chat page, page stays on chat tab
    lang_selector.select("English").run()
    assert not at.exception
    assert "Embodied World Models" in at.title[0].value
    assert "Interactive Academic Q&A" in at.tabs[0].label
    assert at.sidebar.selectbox(key="language_select").value == "English"
    assert at.session_state["active_tab_id"] == "chat"

    # 3. Switch to literature library tab
    at.session_state["active_tab_id"] = "library"
    at.run()
    assert not at.exception
    assert at.session_state["active_tab_id"] == "library"
    assert "Literature Library" in at.session_state["main_tabs"]

    # 4. Switch language to Chinese while in literature library, verify page remains on library tab
    at.sidebar.selectbox(key="language_select").select(LANGUAGES["zh"]).run()
    assert not at.exception
    assert TRANSLATIONS["zh"]["app_title"].split()[0] in at.title[0].value
    assert TRANSLATIONS["zh"]["tab_library"] in at.session_state["main_tabs"]
    assert at.session_state["active_tab_id"] == "library"

    # 5. Switch back to English, still stably remains on library tab
    at.sidebar.selectbox(key="language_select").select("English").run()
    assert not at.exception
    assert "Literature Library" in at.session_state["main_tabs"]
    assert at.session_state["active_tab_id"] == "library"
