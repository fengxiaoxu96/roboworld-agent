# 项目研发任务书：arXiv 机器人世界模型追踪智能体 (RoboWorld-Agent)

**项目名称**：RoboWorld-Agent  
**核心目标**：针对机器人世界模型（World Models / Latent Dynamics）垂直领域，构建具备数据感知、语义判断、结构化提取、长文本工具调用、混合持久化存储与自动化推送的完整智能体系统。

---

## 阶段一：环境初始化与数据契约定义 (Data & Contract First)

### 任务 1.1：开发环境初始化与工程脚手架搭建
- **目标**：建立规范的 Python 开发环境、依赖锁定与目录拓扑。
- **技术栈**：Python 3.11+, `uv`, `ruff`, `pre-commit`。
- **产出路径**：
  - 根目录：`pyproject.toml`, `uv.lock`, `.gitignore`, `.env.example`, `.pre-commit-config.yaml`
  - 核心子目录：`src/core/`, `src/fetchers/`, `src/agents/`, `src/tools/`, `src/storage/`, `src/workflows/`, `tests/`, `data/`
- **验收标准**：通过 `uv run python` 能正常导入所有基础依赖，Pre-commit 钩子生效，Git 状态干净。

### 任务 1.2：定义领域数据模型 (Pydantic Schemas)
- **目标**：完成系统核心数据流通强类型 Schema 定义，确保所有下游模块有明确数据契约。
- **文件位置**：`src/core/schema.py`
- **关键结构定义**：
  - `RawPaper`：包含 `arxiv_id`, `title`, `authors`, `abstract`, `categories`, `published_date`, `pdf_url`。
  - `PaperInsight`：
    - `is_world_model`: bool（是否具备环境动态预测/潜空间建模属性）
    - `model_paradigm`: Literal['Diffusion-based', 'Transformer/AutoRegressive', 'Flow-Matching', 'State-Space/Mamba', 'Other']
    - `action_conditioned`: bool（是否显式输入 Action）
    - `eval_environments`: list[str]（验证基准，如 Isaac Gym/Lab, RLBench, CALVIN, Real Robot）
    - `core_innovation`: str（150字以内核心贡献概述）
- **验收标准**：编写 `tests/test_schema.py`，确保合法数据成功反序列化，非法枚举或缺项抛出 `ValidationError`。

---

## 阶段二：数据采集与核心推理 Agent 开发 (Perception & Cognition)

### 任务 2.1：arXiv 采集器与前置规则初筛 (Fetcher & Rule Filter)
- **目标**：低延迟、防限流抓取目标分类论文，并用规则粗筛过滤无关文献。
- **文件位置**：`src/fetchers/arxiv_fetcher.py`
- **具体要求**：
  - 查询分类：`cs.RO`, `cs.AI`, `cs.LG`。
  - 支持参数：`days_back`（近几天更新）、`max_results`。
  - 正则初筛：针对 Title 和 Abstract 匹配关键词集合（如 `world model`, `latent dynamics`, `video prediction`, `forward model`, `embodied` 等）。
  - 实现异常捕获与指数退避重试（Exponential Backoff）。
- **验收标准**：单次运行能返回清洗后的 `list[RawPaper]`，并打印粗筛丢弃比例日志。

### 任务 2.2：分类与结构化提取 Agent (Classifier & Extractor)
- **目标**：通过 LLM Structured Outputs / Tool Calling 稳定生成结构化洞察。
- **文件位置**：`src/agents/classifier_agent.py`
- **具体要求**：
  - 编写 System Prompt，明确定义机器人世界模型的边界（严格区分仅生成画面的通用视频模型与支持动作闭环的具身世界模型）。
  - 强制输出 `PaperInsight` 模型，禁止自由 Markdown 文本输出。
  - 构建 Golden Dataset 测试集（10 篇标准世界模型论文 + 10 篇边缘混淆论文）。
- **验收标准**：在 Golden Dataset 上分类准确率 >= 90%，无 JSON 解析崩坏。

---

## 阶段三：长文本工具与存储底座集成 (Tools & Storage)

### 任务 3.1：PDF 定向章节解析工具 (Tool Use)
- **目标**：在摘要语义不足以做出确切判断时，赋能 Agent 查阅论文正文。
- **文件位置**：`src/tools/pdf_parser.py`
- **具体要求**：
  - 基于 `PyMuPDF` (fitz) 或 `pypdf` 实现 PDF 纯文本流提取。
  - 正则匹配关键二级标题，仅定向截取 `Methodology`、`Method`、`Experiments` 核心段落，过滤参考文献与长公式。
  - 封装为标准 Agent Tool：`get_paper_section(arxiv_id: str, section_name: str) -> str`。
- **验收标准**：输入指定 arXiv 论文 ID，3秒内返回清理后的对应段落文本。

### 任务 3.2：混合持久化存储层 (SQLite + Vector DB)
- **目标**：建立去重、状态记录及支持多维检索的数据底座。
- **文件位置**：`src/storage/db_manager.py`, `src/storage/vector_store.py`
- **具体要求**：
  - **SQLite**：建表管理 `arxiv_id (PK)`, `title`, `published_date`, `is_wm`, `insight_json`, `created_at`。
  - **Vector DB**：本地部署/嵌入模式运行 `Qdrant` 或 `Chroma`，将已确认的相关论文核心结论切片后 Embedding 入库。
  - 写入逻辑包含去重检查：若 `arxiv_id` 已存在则跳过处理。
- **验收标准**：重复写入幂等有效，可通过向量库进行语义近邻搜索。

---

## 阶段四：工作流状态编排与容错 (Workflow Orchestration)

### 任务 4.1：基于 LangGraph 的主状态图编排
- **目标**：使用状态机编排完整的 Agent 处理链条，实现条件分支与错误隔离。
- **文件位置**：`src/workflows/graph.py`
- **工作流节点与流转**：
  1. `FetchNode`：拉取并粗筛 arXiv 元数据。
  2. `FilterNode`：LLM 对摘要进行第一轮分类判定。
  3. `Conditional Router`：
     - 若置信度高且不相关 -> `SkipNode`（仅记入 SQLite，不入向量库）。
     - 若置信度高且相关 -> `ExtractNode`。
     - 若置信度低/信息不足 -> `ToolCallNode`（调用 PDF 解析正文关键章节） -> 重新判定。
  4. `StorageNode`：完成元数据与向量切片落库。
  5. `ReportNode`：格式化今日处理成果。
- **验收标准**：单个样本失败不阻塞整体流水线，生成完整执行 Trace 状态。

---

## 阶段五：自动化分发、UI 与监控评估 (Delivery & Observability)

### 任务 5.1：Markdown 简报生成与定时触发
- **目标**：实现无人值守的自动化早报输出。
- **文件位置**：`src/workflows/scheduler.py`, `src/tools/reporter.py`
- **具体要求**：
  - 使用 Jinja2 渲染规范的 Markdown 简报（按模型范式归类，高亮 Action-conditioned、基准环境与核心创新）。
  - 支持保存本地 Markdown，或通过 Webhook 推送至通信工具/邮箱。
  - 支持系统定时触发（如通过 Crontab 或 APScheduler 设置每日固定时间运行）。
- **验收标准**：单次流水线结束后，能在指定输出目录产出结构优雅的日报文档。

### 任务 5.2：Streamlit 交互式问答界面
- **目标**：提供文献知识库的可视化检索与多篇论文横向对比问答。
- **文件位置**：`src/app.py`
- **具体要求**：
  - 侧边栏/主面板展示已收录文献列表，支持按分类标签过滤。
  - 提供 Chat 对话框，挂载向量检索实现 RAG 问答（如“对比上周入库的使用 Flow Matching 的世界模型方法”）。
- **验收标准**：前端能稳定启动，检索问答响应准确且包含原始文献引用。

### 任务 5.3：可观测性追踪与质量评估 (Tracing & Eval)
- **目标**：掌控 Token 消耗，监控 Prompt 迭代对研判准确率的影响。
- **具体要求**：
  - 全流程接入 LangSmith 或 Phoenix，记录所有 LLM 调用拓扑。
  - 建立自动化 Eval 测试脚本：每次修改 Prompt 或模型参数时，跑一遍 Golden Dataset 并输出 Precision/Recall 报告。
- **验收标准**：可通过可视化平台查看节点 Trace；测试集改动可量化验证。