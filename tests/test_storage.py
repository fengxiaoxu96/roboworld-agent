"""单测与集成测试：混合持久化存储层 (Task 3.2: SQLite + Vector DB)."""

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
        "首次将 Flow-Matching 应用于机器人动作条件化潜空间预测，"
        "在 CALVIN 基准上实现了高效的策略评估。"
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
# 1. SQLite 持久化测试 (DBManager)
# =========================================================================


def test_sqlite_in_memory_crud() -> None:
    db = DBManager(db_path=":memory:")

    paper = _create_sample_paper()
    insight = _create_sample_insight()

    # 1. 首次插入
    inserted = db.save_paper(paper, insight)
    assert inserted is True
    assert db.exists(paper.arxiv_id) is True

    # 2. 读取验证
    record = db.get_paper(paper.arxiv_id)
    assert record is not None
    assert record["arxiv_id"] == "2401.00001"
    assert record["title"] == paper.title
    assert record["is_wm"] == 1
    assert "Flow-Matching" in record["insight_json"]

    # 3. 结构化 Insight 解析
    loaded_insight = db.get_paper_insight(paper.arxiv_id)
    assert loaded_insight is not None
    assert loaded_insight.is_world_model is True
    assert loaded_insight.model_paradigm == "Flow-Matching"
    assert loaded_insight.action_conditioned is True


def test_sqlite_deduplication_and_idempotency() -> None:
    """验证去重检查：若 arxiv_id 已存在则跳过处理，重复写入幂等有效。"""
    db = DBManager(db_path=":memory:")
    paper = _create_sample_paper("2401.00002")
    insight = _create_sample_insight()

    # 第一次写入成功
    assert db.save_paper(paper, insight) is True
    assert db.count_papers() == 1

    # 第二次写入被去重并跳过
    assert db.save_paper(paper, insight) is False
    assert db.count_papers() == 1


def test_sqlite_non_world_model_paper() -> None:
    """验证非世界模型（初筛跳过/未通过分类）正常记入 SQLite。"""
    db = DBManager(db_path=":memory:")
    paper = _create_sample_paper("2401.00003", title="General Video Synthesis")
    insight = _create_sample_insight(
        is_world_model=False,
        core_innovation="纯视频生成模型，缺乏物理动作闭环。",
    )

    assert db.save_paper(paper, insight) is True
    record = db.get_paper("2401.00003")
    assert record is not None
    assert record["is_wm"] == 0

    # 列表过滤测试
    wm_list = db.list_papers(is_wm=True)
    non_wm_list = db.list_papers(is_wm=False)
    assert len(wm_list) == 0
    assert len(non_wm_list) == 1
    assert non_wm_list[0]["arxiv_id"] == "2401.00003"


def test_sqlite_file_persistence(tmp_path: Path) -> None:
    """验证基于文件的 SQLite 真实持久化与多实例读取。"""
    db_file = tmp_path / "papers.db"
    db1 = DBManager(db_path=db_file)

    paper = _create_sample_paper("2401.00004")
    db1.save_paper(paper)

    # 开启第二个实例读取同一文件
    db2 = DBManager(db_path=db_file)
    assert db2.exists("2401.00004") is True
    assert db2.count_papers() == 1

    # 删除测试
    assert db2.delete_paper("2401.00004") is True
    assert db2.exists("2401.00004") is False


# =========================================================================
# 2. Qdrant 向量数据库测试 (VectorStore)
# =========================================================================


def test_create_paper_chunks() -> None:
    paper = _create_sample_paper()
    insight = _create_sample_insight()
    chunks = create_paper_chunks(
        paper, insight, extra_chunks=["Methodology excerpt text"]
    )

    assert len(chunks) == 3
    # 切片 1：核心创新
    core_text, core_meta = chunks[0]
    assert "Core Innovation" in core_text
    assert core_meta["model_paradigm"] == "Flow-Matching"
    assert core_meta["action_conditioned"] is True

    # 切片 2：摘要
    abs_text, abs_meta = chunks[1]
    assert "Abstract:" in abs_text
    assert abs_meta["chunk_type"] == "abstract"

    # 切片 3：附加章节
    extra_text, extra_meta = chunks[2]
    assert "Methodology excerpt text" in extra_text
    assert extra_meta["chunk_type"] == "section_0"


def test_vector_store_add_and_deduplication() -> None:
    """验证向量库切片写入与去重逻辑。"""
    vstore = VectorStore(storage_path=":memory:", collection_name="test_collection")
    paper = _create_sample_paper("2401.10001")
    insight = _create_sample_insight()

    # 1. 首次入库
    added = vstore.add_paper(paper, insight)
    assert added is True
    assert vstore.paper_exists(paper.arxiv_id) is True
    # 默认产生 2 个切片（核心创新 + 摘要）
    assert vstore.count() == 2

    # 2. 二次入库被去重跳过，数量不变
    added_again = vstore.add_paper(paper, insight)
    assert added_again is False
    assert vstore.count() == 2


def test_vector_store_semantic_search() -> None:
    """验证验收标准：可通过向量库进行语义近邻搜索。"""
    vstore = VectorStore(storage_path=":memory:", collection_name="test_search")

    # 论文 1：Flow-Matching 机械臂世界模型
    p1 = _create_sample_paper(
        arxiv_id="2401.11111",
        title="Flow Matching World Models for Robotic Manipulation",
        abstract="We introduce continuous flow matching to learn robot dynamics models.",
    )
    ins1 = _create_sample_insight(
        model_paradigm="Flow-Matching",
        core_innovation="基于连续流匹配的机器人动态预测模型。",
    )

    # 论文 2：Diffusion 自动驾驶世界模型
    p2 = _create_sample_paper(
        arxiv_id="2401.22222",
        title="Diffusion World Models in Autonomous Driving",
        abstract="A diffusion-based visual simulator for highway traffic prediction.",
    )
    ins2 = _create_sample_insight(
        model_paradigm="Diffusion-based",
        core_innovation="针对高速公路驾驶场景的多智能体扩散预测模型。",
    )

    # 论文 3：Transformer 离散世界模型
    p3 = _create_sample_paper(
        arxiv_id="2401.33333",
        title="Discrete Autoregressive Latent Dynamics",
        abstract="Tokenizing robotics observations for autoregressive sequence modeling.",
    )
    ins3 = _create_sample_insight(
        model_paradigm="Transformer/AutoRegressive",
        core_innovation="自回归 Transformer 离散世界模型。",
    )

    assert vstore.add_paper(p1, ins1) is True
    assert vstore.add_paper(p2, ins2) is True
    assert vstore.add_paper(p3, ins3) is True

    # 1. 查询“flow matching robot dynamics”
    results1 = vstore.search("flow matching robot dynamics", limit=2)
    assert len(results1) > 0
    assert isinstance(results1[0], VectorSearchResult)
    # 最相关的应为论文 1 (Flow-Matching)
    assert results1[0].arxiv_id == "2401.11111"
    assert results1[0].model_paradigm == "Flow-Matching"
    assert results1[0].score > 0.3

    # 2. 查询“autonomous driving diffusion”
    results2 = vstore.search("autonomous driving diffusion", limit=2)
    assert len(results2) > 0
    assert results2[0].arxiv_id == "2401.22222"
    assert results2[0].model_paradigm == "Diffusion-based"


def test_vector_store_filtered_search() -> None:
    """验证基于元数据过滤字段的向量搜索。"""
    vstore = VectorStore(storage_path=":memory:", collection_name="test_filter")

    p1 = _create_sample_paper(arxiv_id="2401.40001", title="Model A")
    ins1 = _create_sample_insight(model_paradigm="Diffusion-based")
    p2 = _create_sample_paper(arxiv_id="2401.40002", title="Model B")
    ins2 = _create_sample_insight(model_paradigm="Flow-Matching")

    vstore.add_paper(p1, ins1)
    vstore.add_paper(p2, ins2)

    # 精确过滤 Flow-Matching
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
# 3. 混合持久化协同与工作流语义模拟测试
# =========================================================================


def test_hybrid_storage_workflow(tmp_path: Path) -> None:
    """模拟 Phase 4 编排流程中 SQLite 与 Vector DB 的混合落库协同：

    - 相关论文 (is_wm=True)：写入 SQLite 并切片入向量库；
    - 不相关论文 (is_wm=False)：仅记入 SQLite，不入向量库；
    - 二次写入：全部幂等跳过。
    """
    db = DBManager(db_path=tmp_path / "papers.db")
    vstore = VectorStore(storage_path=tmp_path / "qdrant")

    # 1. 论文 A（确认是世界模型）
    paper_wm = _create_sample_paper("2401.90001", title="Real World Model")
    insight_wm = _create_sample_insight(is_world_model=True)

    db_res_wm = db.save_paper(paper_wm, insight_wm)
    vstore_res_wm = vstore.add_paper(paper_wm, insight_wm)
    assert db_res_wm is True
    assert vstore_res_wm is True

    # 2. 论文 B（非世界模型，如单纯图像生成）
    paper_skip = _create_sample_paper("2401.90002", title="Pure Image GAN")
    insight_skip = _create_sample_insight(
        is_world_model=False,
        core_innovation="通用图像生成，无状态动态建模。",
    )

    db_res_skip = db.save_paper(paper_skip, insight_skip)
    # 模拟流程：根据条件路由，非世界模型不入向量库
    assert db_res_skip is True
    assert vstore.paper_exists(paper_skip.arxiv_id) is False

    # 3. 检查数据库状态
    assert db.count_papers(is_wm=True) == 1
    assert db.count_papers(is_wm=False) == 1
    assert vstore.count() == 2  # 仅 paper_wm 的 2 个切片

    # 4. 重复写入幂等性验证
    assert db.save_paper(paper_wm, insight_wm) is False
    assert vstore.add_paper(paper_wm, insight_wm) is False
    assert db.count_papers() == 2
    assert vstore.count() == 2

    # 5. 向量检索仅检索出相关世界模型文献
    search_hits = vstore.search("world model", limit=5)
    assert len(search_hits) > 0
    assert all(hit.arxiv_id == "2401.90001" for hit in search_hits)
