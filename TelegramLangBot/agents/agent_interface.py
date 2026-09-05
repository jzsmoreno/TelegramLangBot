"""Base interfaces and types for the multi-agent review system."""

from __future__ import annotations

import abc
import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ReviewScore(Enum):
    """Overall score a review agent assigns to a response."""

    PASS = auto()
    NEEDS_IMPROVEMENT = auto()
    FAIL = auto()


class AgentReviewType(Enum):
    """Categories of review that agents can produce."""

    CLARITY = "clarity"
    HELPFULNESS = "helpfulness"
    TONE = "tone"
    SAFETY = "safety"


@dataclass
class AgentReview:
    """The result of a single agent's review.

    Attributes
    ----------
    agent_name : str
        Human-readable name of the reviewing agent (e.g. \"ClarityAgent\").
    review_type : AgentReviewType
        The category this review belongs to.
    score : ReviewScore
        Overall assessment (merged rule-based + LLM).
    feedback : str
        Short explanation of the findings (empty when PASS).
    suggested_fix : str | None
        Concrete replacement text when the agent can provide one, otherwise ``None``.
    llm_review_used : bool
        Whether an LLM was used to supplement the review.
    llm_review_duration_ms : float
        Time the LLM review call took in milliseconds (0 if not used).
    llm_score : ReviewScore | None
        The isolated LLM score before merging with rule-based results
        (``None`` when LLM review was not used).
    llm_feedback : str
        The isolated LLM feedback before merging with rule-based results
        (empty when LLM review was not used).
    """

    agent_name: str
    review_type: AgentReviewType
    score: ReviewScore
    feedback: str = ""
    suggested_fix: str | None = None
    llm_review_used: bool = False
    llm_review_duration_ms: float = 0.0
    llm_score: ReviewScore | None = None
    llm_feedback: str = ""


logger = logging.getLogger(__name__)


class BaseAgent(abc.ABC):
    """Abstract agent that reviews an LLM response for a specific quality dimension.

    Subclasses must implement :meth:`review`.
    """

    def __init__(self, llm: Any = None) -> None:
        self._llm = llm

    @property
    @abc.abstractmethod
    def agent_name(self) -> str:
        """Human-readable agent name used in logs and :class:`AgentReview`."""
        ...

    @property
    @abc.abstractmethod
    def review_type(self) -> AgentReviewType:
        """The category this agent is responsible for."""
        ...

    @property
    @abc.abstractmethod
    def llm_prompt(self) -> str:
        """System prompt used when running LLM-based review on this dimension."""
        ...

    async def _run_llm_review(self, response: str) -> AgentReview | None:
        """Run an LLM-based review and parse its structured JSON response."""
        if self._llm is None:
            return None

        import time

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=self.llm_prompt),
            HumanMessage(content=f"RESPONSE TO REVIEW:\n{response}"),
        ]

        t0 = time.monotonic()

        try:
            result = await self._llm.ainvoke(messages)
        except Exception:
            logger.exception(
                "LLM review failed for %s — falling back to rule-based",
                self.agent_name,
            )
            return None

        elapsed_ms = (time.monotonic() - t0) * 1000

        if hasattr(result, "content"):
            content = result.content
        else:
            content = result

        if isinstance(content, list):
            text_parts: list[str] = []

            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    block_text = block.get("text")
                    if isinstance(block_text, str):
                        text_parts.append(block_text)

            raw = "".join(text_parts).strip()
        else:
            raw = str(content).strip()

        if not raw:
            logger.warning(
                "LLM review for %s returned empty output",
                self.agent_name,
            )
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            cleaned = raw.strip()

            if cleaned.startswith("```") and cleaned.endswith("```"):
                lines = cleaned.splitlines()

                if lines:
                    lines = lines[1:]

                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]

                cleaned = "\n".join(lines).strip()

            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning(
                    "LLM review for %s returned invalid JSON: %r",
                    self.agent_name,
                    raw[:1000],
                )
                return None

        if not isinstance(data, dict):
            logger.warning(
                "LLM review for %s returned JSON %s instead of an object: %r",
                self.agent_name,
                type(data).__name__,
                raw[:500],
            )
            return None

        score_raw = data.get("score")

        if not isinstance(score_raw, str):
            logger.warning(
                "LLM review for %s has invalid score type: %r",
                self.agent_name,
                score_raw,
            )
            return None

        score_map = {
            "PASS": ReviewScore.PASS,
            "NEEDS_IMPROVEMENT": ReviewScore.NEEDS_IMPROVEMENT,
            "FAIL": ReviewScore.FAIL,
        }

        score = score_map.get(score_raw.strip().upper())

        if score is None:
            logger.warning(
                "LLM review for %s returned unknown score: %r",
                self.agent_name,
                score_raw,
            )
            return None

        feedback_raw = data.get("feedback", "")

        if not isinstance(feedback_raw, str):
            logger.warning(
                "LLM review for %s has invalid feedback type: %s",
                self.agent_name,
                type(feedback_raw).__name__,
            )
            return None

        feedback = feedback_raw.strip()

        return AgentReview(
            agent_name=self.agent_name,
            review_type=self.review_type,
            score=score,
            feedback=feedback,
            llm_review_used=True,
            llm_review_duration_ms=round(elapsed_ms, 1),
            llm_score=score,
            llm_feedback=feedback,
        )

    @abc.abstractmethod
    async def review(
        self,
        response: str,
        context: dict[str, Any] | None = None,
    ) -> AgentReview:
        """Inspect *response* and return an :class:`AgentReview`.

        Parameters
        ----------
        response : str
            The LLM response text (already processed by ``clean_telegram_html``).
        context : dict | None
            Optional extra info such as ``user_message``, ``chat_history``,
            ``language``, etc.

        Returns
        -------
        AgentReview
            Assessment with score, feedback, and optional suggested fix.
        """
        ...

    @staticmethod
    def _build_review(
        score: ReviewScore,
        feedback: str = "",
        suggested_fix: str | None = None,
        agent_name: str = "BaseAgent",
        review_type: AgentReviewType = AgentReviewType.CLARITY,
        llm_review_used: bool = False,
        llm_review_duration_ms: float = 0.0,
        llm_score: ReviewScore | None = None,
        llm_feedback: str = "",
    ) -> AgentReview:
        """Convenience factory for building a review."""
        return AgentReview(
            agent_name=agent_name,
            review_type=review_type,
            score=score,
            feedback=feedback,
            suggested_fix=suggested_fix,
            llm_review_used=llm_review_used,
            llm_review_duration_ms=llm_review_duration_ms,
            llm_score=llm_score,
            llm_feedback=llm_feedback,
        )
