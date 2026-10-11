"""机器人世界模型分类与结构化提取 Agent (Task 2.2 / Task 4.1).

定义机器人世界模型与通用视频生成模型的边界：
1. 具身世界模型必须具备状态转移或环境动态预测能力，且显式支持动作闭环 (Action-conditioned)；
2. 纯文本生成画面的通用视频模型（无动作输入、无动力学转移）不属于具身世界模型；
3. 提供摘要初筛 (classify) 与深度结构化提取 (extract_insight) 功能；
4. 当摘要信息模糊或无法确认动作条件时，输出 low 置信度，驱动 ToolCallNode 获取正文。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.core.llm import get_llm
from src.core.schema import ModelParadigm, PaperInsight, RawPaper

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一名资深的具身智能与机器人世界模型（Robotics World Models）领域科学家。
你的职责是严谨研判论文是否属于“具身机器人世界模型”，并提取结构化洞察。

【核心研判边界】：
1. 必须是【机器人世界模型】：核心是学习环境的物理转移与因果动力学模型 $p(s_{t+1} | s_t, a_t)$ 或潜空间动态 (Latent Dynamics)。
2. 严格区分【动作闭环】与【通用视频生成】：
   - 具身世界模型必须显式接收动作（Action/Control/Proprioception）作为输入条件，或用于闭环策略规划与评估 (Imagination-based Planning / Policy Evaluation)；
   - 仅根据文本提示词生成视频画面的通用生成模型（如纯 Sora-like 模型，未接收机器人动作、不可用于动作闭环控制），一律判为 False。
3. 若论文摘要语义不足以做出确切判断（例如提及视频预测但未说明是否输入机器人动作），置信度必须给出 'low' 并标记 needs_full_text=True，以便后续调阅正文章节进一步确证。
"""


class FilterDecision(BaseModel):
    """FilterNode 第一轮分类判定输出模型。"""

    is_world_model: bool = Field(
        ...,
        description="是否属于具身机器人世界模型",
    )
    confidence: Literal["high", "low"] = Field(
        ...,
        description="判定置信度。若摘要信息不足判断动作条件则为 'low'",
    )
    needs_full_text: bool = Field(
        default=False,
        description="是否需要调阅正文章节进一步研判",
    )
    reasoning: str = Field(
        ...,
        description="判定理由与关键线索说明",
    )


class ClassifierAgent:
    """分类与结构化提取 Agent。"""

    def __init__(
        self,
        llm: Any | None = None,
        auto_init_llm: bool = True,
    ) -> None:
        """初始化分类与结构化提取 Agent。

        Args:
            llm: 显式指定的大模型实例。若为 None 且 auto_init_llm 为 True，
                 将调用 get_llm(temperature=0.0) 自动初始化通用 LLM。
            auto_init_llm: 是否自动尝试根据环境配置初始化通用 LLM。
        """
        if llm is not None:
            self.llm = llm
        elif auto_init_llm:
            self.llm = get_llm(temperature=0.0)
        else:
            self.llm = None

    def classify(
        self,
        paper: RawPaper,
        section_text: str | None = None,
    ) -> FilterDecision:
        """对论文执行相关性与置信度判定。

        若提供 section_text（通过 PDF 解析工具调阅的正文），则结合正文综合研判。
        """
        # 若传入了 LLM，尝试调用 LLM 结构化输出
        if self.llm is not None:
            try:
                return self._classify_with_llm(paper, section_text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM 分类调用失败，回退到规则分析器: %s", exc)

        # 默认高质量规则启发式研判（保证离线与无 API Key 场景下的可靠性）
        return self._classify_with_heuristics(paper, section_text)

    def extract_insight(
        self,
        paper: RawPaper,
        section_text: str | None = None,
    ) -> PaperInsight:
        """对确认相关的论文提取完整的 PaperInsight 结构化洞察。"""
        if self.llm is not None:
            try:
                return self._extract_with_llm(paper, section_text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM 提取调用失败，回退到规则分析器: %s", exc)

        return self._extract_with_heuristics(paper, section_text)

    def _classify_with_heuristics(
        self,
        paper: RawPaper,
        section_text: str | None = None,
    ) -> FilterDecision:
        """基于领域本体与关键词规则的分类研判。"""
        content = f"{paper.title}\n{paper.abstract}".lower()
        full_context = f"{content}\n{section_text.lower()}" if section_text else content

        # 明确不相关的负向信号（通用非具身领域、纯静态感知、无动力学强化学习）
        negative_signals = [
            "text-to-image",
            "text-to-video only",
            "text-to-video generation",
            "social media",
            "financial stock",
            "medical imaging segmentation",
            "nlp sentiment",
            "talking head video synthesis",
            "audio generation",
            "point cloud registration",
            "without dynamics",
            "without training an environmental transition model",
            "human preference modeling",
            "2d bounding box object detection",
            "surround-view camera images",
        ]
        if any(neg in content for neg in negative_signals):
            return FilterDecision(
                is_world_model=False,
                confidence="high",
                needs_full_text=False,
                reasoning="论文主题属于与机器人物理交互无关的通用领域或缺乏环境动力学建模。",
            )

        # 核心正向信号：世界模型 / 潜空间动态
        has_wm_keyword = any(
            kw in full_context
            for kw in [
                "world model",
                "world models",
                "latent dynamics",
                "forward model",
                "transition model",
                "generative world",
                "dynamic simulator",
            ]
        )

        # 机器人 / 具身 / 物理环境信号
        has_robot_keyword = any(
            kw in full_context
            for kw in [
                "robot",
                "robotic",
                "manipulation",
                "embodied",
                "policy",
                "locomotion",
                "calvin",
                "rlbench",
                "franka",
                "driving",
                "vehicle",
                "autonomous driving",
                "dexterous",
                "hands",
                "isaac gym",
                "deepmind control",
                "aloha",
                "robomimic",
                "kitchen",
                "meta-world",
                "bridge data",
                "bimanual",
                "control suite",
            ]
        )

        # 显式动作条件化信号
        has_action_cond = any(
            kw in full_context
            for kw in [
                "action-conditioned",
                "action conditioned",
                "action condition",
                "conditioned on action",
                "conditioned on robot action",
                "predict future observation given action",
                "action inputs",
                "action sequence",
                "control actions",
                "motor control actions",
                "action trajectories",
                "actions as inputs",
            ]
        )

        # 若已调阅正文且包含动作与动态模型，判定为高置信度世界模型
        if section_text:
            if (has_wm_keyword or has_action_cond) and has_robot_keyword:
                return FilterDecision(
                    is_world_model=True,
                    confidence="high",
                    needs_full_text=False,
                    reasoning="结合正文关键章节确证，该方法具备明确的机器人动作条件转移与环境动态预测能力。",
                )
            return FilterDecision(
                is_world_model=False,
                confidence="high",
                needs_full_text=False,
                reasoning="调阅正文后确认缺乏显式的机器人动作条件化动态预测机制。",
            )

        # 纯摘要判定逻辑
        # 1. 明确的机器人动作条件化世界模型
        if has_wm_keyword and has_action_cond and has_robot_keyword:
            return FilterDecision(
                is_world_model=True,
                confidence="high",
                needs_full_text=False,
                reasoning="摘要明确提出面向机器人的动作条件化世界模型/潜空间动态方法。",
            )

        # 2. 提及世界模型或视频预测但未明确动作闭环 -> 置信度低，触发 ToolCallNode
        if has_wm_keyword and has_robot_keyword and not has_action_cond:
            return FilterDecision(
                is_world_model=True,
                confidence="low",
                needs_full_text=True,
                reasoning="摘要提及机器人动态/世界模型，但尚未明确是否包含动作闭环条件，需查阅正文方法章节确证。",
            )

        # 3. 提及 video prediction 但机器人关联模糊 -> 置信度低
        if "video prediction" in content and has_robot_keyword:
            return FilterDecision(
                is_world_model=False,
                confidence="low",
                needs_full_text=True,
                reasoning="摘要包含机器人视频预测，需调阅正文判断是否属于具备物理动作条件的具身世界模型。",
            )

        # 4. 其他常规无强关联论文 -> 置信度高且不相关
        return FilterDecision(
            is_world_model=False,
            confidence="high",
            needs_full_text=False,
            reasoning="摘要未呈现机器人动作条件化世界模型或潜空间动力学核心属性。",
        )

    def _extract_with_heuristics(
        self,
        paper: RawPaper,
        section_text: str | None = None,
    ) -> PaperInsight:
        """基于上下文信息提取结构化 PaperInsight。"""
        full_text = f"{paper.title}\n{paper.abstract}\n{section_text or ''}".lower()
        title_lower = paper.title.lower()

        # 1. 推断模型范式（优先考虑标题与架构明确指定的架构，再结合正文关键词）
        paradigm: ModelParadigm = "Other"
        if (
            "flow matching" in title_lower
            or "flow-matching" in title_lower
            or "continuous flow" in title_lower
        ):
            paradigm = "Flow-Matching"
        elif (
            "mamba" in title_lower
            or "state-space" in title_lower
            or "state space" in title_lower
            or "ssm" in title_lower
        ):
            paradigm = "State-Space/Mamba"
        elif (
            "diffusion" in title_lower
            or "denoising" in title_lower
            or "unisim" in title_lower
            or "drivedreamer" in title_lower
        ):
            paradigm = "Diffusion-based"
        elif (
            "transformer" in title_lower
            or "autoregressive" in title_lower
            or "robogen" in title_lower
            or "dreamer" in title_lower
        ):
            paradigm = "Transformer/AutoRegressive"
        elif any(
            w in full_text
            for w in [
                "flow matching",
                "rectified flow",
                "flow-matching",
                "continuous flow",
            ]
        ):
            paradigm = "Flow-Matching"
        elif any(w in full_text for w in ["mamba", "state-space", "ssm"]):
            paradigm = "State-Space/Mamba"
        elif any(
            w in full_text
            for w in [
                "diffusion",
                "denoising",
                "score-based",
                "ddpm",
                "video prediction",
                "forward simulation",
            ]
        ):
            paradigm = "Diffusion-based"
        elif any(
            w in full_text
            for w in [
                "transformer",
                "autoregressive",
                "gpt",
                "discrete token",
                "tokenizer",
                "recurrent state space",
                "rssm",
            ]
        ):
            paradigm = "Transformer/AutoRegressive"

        # 2. 动作条件化
        action_cond = any(
            w in full_text
            for w in [
                "action",
                "action-conditioned",
                "control",
                "conditioned on action",
                "policy evaluation",
            ]
        )

        # 3. 评测基准识别
        known_envs = [
            "CALVIN",
            "RLBench",
            "Isaac Gym",
            "Isaac Lab",
            "RoboSuite",
            "Meta-World",
            "Franka",
            "UR5",
            "Real Robot",
            "Habitat",
            "MuJoCo",
            "ALOHA",
        ]
        detected_envs = [
            env
            for env in known_envs
            if re.search(rf"\b{re.escape(env.lower())}\b", full_text)
        ]
        if not detected_envs:
            detected_envs = ["Simulation / Real Robot Benchmark"]

        # 4. 150字核心贡献概述
        core_innov = (
            f"提出针对机器人具身交互的 {paradigm} 世界模型，"
            f"支持动作条件化未来状态预测与策略评估（验证于 {', '.join(detected_envs[:2])}）。"
        )
        if len(core_innov) > 150:
            core_innov = core_innov[:147] + "..."

        return PaperInsight(
            is_world_model=True,
            model_paradigm=paradigm,
            action_conditioned=action_cond,
            eval_environments=detected_envs,
            core_innovation=core_innov,
        )

    def _classify_with_llm(
        self, paper: RawPaper, section_text: str | None = None
    ) -> FilterDecision:
        assert self.llm is not None
        user_prompt = f"论文标题: {paper.title}\n摘要: {paper.abstract}\n"
        if section_text:
            user_prompt += f"\n【提取的正文关键章节】:\n{section_text[:3000]}\n"

        structured_llm = self.llm.with_structured_output(FilterDecision)
        result = structured_llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        if isinstance(result, FilterDecision):
            return result
        if isinstance(result, dict):
            return FilterDecision.model_validate(result)
        raise ValueError(f"无法将 LLM 分类结果解析为 FilterDecision: {result}")

    def _extract_with_llm(
        self, paper: RawPaper, section_text: str | None = None
    ) -> PaperInsight:
        assert self.llm is not None
        user_prompt = (
            f"已确认该论文属于机器人世界模型。\n"
            f"论文标题: {paper.title}\n摘要: {paper.abstract}\n"
        )
        if section_text:
            user_prompt += f"\n【提取的正文关键章节】:\n{section_text[:3000]}\n"

        structured_llm = self.llm.with_structured_output(PaperInsight)
        result = structured_llm.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        if isinstance(result, PaperInsight):
            return result
        if isinstance(result, dict):
            return PaperInsight.model_validate(result)
        raise ValueError(f"无法将 LLM 提取结果解析为 PaperInsight: {result}")


__all__ = ["SYSTEM_PROMPT", "ClassifierAgent", "FilterDecision"]
