"""Robotics world model classification and structured insight extraction agent (Task 2.2 / Task 4.1).

Defines the boundary between robotics world models and generic video generation models:
1. Embodied world models must possess state transition or environment dynamics prediction capabilities,
   and explicitly support closed-loop action conditioning (Action-conditioned);
2. Generic video generation models driven solely by text prompts (without action inputs or dynamics transitions)
   do not qualify as embodied world models;
3. Provides abstract initial screening (classify) and deep structured extraction (extract_insight);
4. Outputs low confidence when abstract information is ambiguous or cannot confirm action conditioning,
   triggering ToolCallNode to retrieve key full-text sections.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.core.llm import get_llm
from src.core.schema import ModelParadigm, PaperInsight, RawPaper

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior research scientist in Embodied AI and Robotics World Models.
Your responsibility is to rigorously analyze whether research papers belong to "Robotics World Models" and extract structured insights.

[Core Decision Boundaries]:
1. Must be a [Robotics World Model]: The core is learning the environment's physical transitions and causal dynamics model p(s_{t+1} | s_t, a_t) or latent dynamics.
2. Strictly distinguish [Closed-loop Action Conditioning] from [Generic Video Generation]:
   - An embodied world model must explicitly take action (Action/Control/Proprioception) as conditional inputs, or be used for imagination-based planning and policy evaluation;
   - Generic video generative models driven solely by text prompts (e.g., pure Sora-like models without robot action conditioning or closed-loop control) must always be classified as False.
3. If abstract semantics are insufficient to reach a conclusive judgment (e.g., mentions video prediction without confirming action conditioning), confidence must be set to 'low' and needs_full_text=True to trigger full text section retrieval.
"""


class FilterDecision(BaseModel):
    """FilterNode classification decision output model."""

    is_world_model: bool = Field(
        ...,
        description="Whether the paper belongs to embodied robotics world models",
    )
    confidence: Literal["high", "low"] = Field(
        ...,
        description="Decision confidence. Set to 'low' if abstract information is insufficient to confirm action conditioning",
    )
    needs_full_text: bool = Field(
        default=False,
        description="Whether key full-text sections must be retrieved for conclusive judgment",
    )
    reasoning: str = Field(
        ...,
        description="Decision rationale and key evidence explanation",
    )


class ClassifierAgent:
    """Classification and structured insight extraction agent."""

    def __init__(
        self,
        llm: Any | None = None,
        auto_init_llm: bool = True,
    ) -> None:
        """Initialize the classification and insight extraction agent.

        Args:
            llm: Explicit LLM instance. If None and auto_init_llm is True,
                 get_llm(temperature=0.0) is called to initialize a universal LLM.
            auto_init_llm: Whether to automatically initialize universal LLM based on environment config.
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
        """Perform relevance and confidence classification on a paper.

        If section_text (retrieved via PDF parser tool) is provided, evaluates jointly with the full text.
        """
        # If LLM is provided, attempt structured output invocation
        if self.llm is not None:
            try:
                return self._classify_with_llm(paper, section_text)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LLM classification invocation failed, falling back to heuristic analyzer: %s",
                    exc,
                )

        # Default high-quality heuristic rule evaluation (ensures reliability offline and without API keys)
        return self._classify_with_heuristics(paper, section_text)

    def extract_insight(
        self,
        paper: RawPaper,
        section_text: str | None = None,
    ) -> PaperInsight:
        """Extract structured PaperInsight from confirmed relevant papers."""
        if self.llm is not None:
            try:
                return self._extract_with_llm(paper, section_text)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LLM insight extraction invocation failed, falling back to heuristic analyzer: %s",
                    exc,
                )

        return self._extract_with_heuristics(paper, section_text)

    def _classify_with_heuristics(
        self,
        paper: RawPaper,
        section_text: str | None = None,
    ) -> FilterDecision:
        """Domain ontology and keyword-rule-based classification analysis."""
        content = f"{paper.title}\n{paper.abstract}".lower()
        full_context = f"{content}\n{section_text.lower()}" if section_text else content

        # Negative signals: general non-embodied domains, static perception, dynamics-free RL
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
                reasoning="The paper belongs to a general domain unrelated to physical robot interaction or lacks environmental dynamics modeling.",
            )

        # Core positive signals: world model / latent dynamics
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

        # Robot / embodied / physical environment signals
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

        # Explicit action-conditioned signals
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

        # If key section text has been retrieved, evaluate with full text
        if section_text:
            if (has_wm_keyword or has_action_cond) and has_robot_keyword:
                return FilterDecision(
                    is_world_model=True,
                    confidence="high",
                    needs_full_text=False,
                    reasoning="Confirmed by key section review, the method clearly demonstrates robot action-conditioned state transitions and environmental dynamics prediction capabilities.",
                )
            return FilterDecision(
                is_world_model=False,
                confidence="high",
                needs_full_text=False,
                reasoning="After reviewing full text, confirmed lack of explicit robot action-conditioned dynamics prediction mechanism.",
            )

        # Pure abstract decision logic
        # 1. Explicit action-conditioned world model
        if has_wm_keyword and has_action_cond and has_robot_keyword:
            return FilterDecision(
                is_world_model=True,
                confidence="high",
                needs_full_text=False,
                reasoning="Abstract explicitly proposes an action-conditioned world model or latent dynamics method for robotics.",
            )

        # 2. Mentions world model or video prediction but action conditioning unconfirmed -> low confidence, triggers ToolCallNode
        if has_wm_keyword and has_robot_keyword and not has_action_cond:
            return FilterDecision(
                is_world_model=True,
                confidence="low",
                needs_full_text=True,
                reasoning="Abstract mentions robot dynamics or world model, but action-conditioned closed-loop is unconfirmed; requires methodology section review.",
            )

        # 3. Mentions video prediction with vague robotics association -> low confidence
        if "video prediction" in content and has_robot_keyword:
            return FilterDecision(
                is_world_model=False,
                confidence="low",
                needs_full_text=True,
                reasoning="Abstract includes robot video prediction; requires full text review to confirm whether it is an embodied world model with physical action conditioning.",
            )

        # 4. Standard irrelevant paper -> high confidence, not relevant
        return FilterDecision(
            is_world_model=False,
            confidence="high",
            needs_full_text=False,
            reasoning="Abstract does not exhibit core properties of robot action-conditioned world models or latent dynamics.",
        )

    def _extract_with_heuristics(
        self,
        paper: RawPaper,
        section_text: str | None = None,
    ) -> PaperInsight:
        """Extract structured PaperInsight based on contextual information."""
        full_text = f"{paper.title}\n{paper.abstract}\n{section_text or ''}".lower()
        title_lower = paper.title.lower()

        # 1. Infer model paradigm
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

        # 2. Action conditioning
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

        # 3. Benchmark environments identification
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

        # 4. Core innovation summary within 150 words
        core_innov = (
            f"Proposes a {paradigm} world model for embodied robotic interaction, "
            f"supporting action-conditioned future state prediction and policy evaluation (evaluated on {', '.join(detected_envs[:2])})."
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
        user_prompt = f"Paper Title: {paper.title}\nAbstract: {paper.abstract}\n"
        if section_text:
            user_prompt += f"\n[Extracted Key Sections]:\n{section_text[:3000]}\n"

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
        raise ValueError(
            f"Failed to parse LLM classification result into FilterDecision: {result}"
        )

    def _extract_with_llm(
        self, paper: RawPaper, section_text: str | None = None
    ) -> PaperInsight:
        assert self.llm is not None
        user_prompt = (
            f"The paper has been confirmed as a robotics world model.\n"
            f"Paper Title: {paper.title}\nAbstract: {paper.abstract}\n"
        )
        if section_text:
            user_prompt += f"\n[Extracted Key Sections]:\n{section_text[:3000]}\n"

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
        raise ValueError(
            f"Failed to parse LLM extraction result into PaperInsight: {result}"
        )


__all__ = ["SYSTEM_PROMPT", "ClassifierAgent", "FilterDecision"]
