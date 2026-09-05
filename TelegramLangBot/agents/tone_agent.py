"""Review agent focused on tone, friendliness, and professionalism."""

from __future__ import annotations

import re
from typing import Any

from .agent_config import (
    AGENT_DEFINITIONS,
    TONE_ARROGANT_PATTERNS,
    TONE_ES_PATTERNS,
    TONE_LLM_PROMPT,
    TONE_PASSIVE_AGGRESSIVE_PATTERNS,
    TONE_ROBOTIC_PATTERNS,
)
from .agent_interface import AgentReview, AgentReviewType, BaseAgent, ReviewScore


class ToneAgent(BaseAgent):
    """Checks whether an LLM response uses an appropriate tone.

    Flags overly formal, robotic, passive-aggressive, or arrogant language
    that harms user experience. Combines regex patterns with optional
    LLM-based analysis.
    """

    _ROBOTIC_PATTERNS = TONE_ROBOTIC_PATTERNS
    _PASSIVE_AGGRESSIVE_PATTERNS = TONE_PASSIVE_AGGRESSIVE_PATTERNS
    _ARROGANT_PATTERNS = TONE_ARROGANT_PATTERNS
    _ES_TONE_PATTERNS = TONE_ES_PATTERNS

    def __init__(self, llm: Any = None) -> None:
        super().__init__(llm=llm)

    @property
    def agent_name(self) -> str:
        return AGENT_DEFINITIONS["tone"]["name"]

    @property
    def review_type(self) -> AgentReviewType:
        return AgentReviewType.TONE

    @property
    def llm_prompt(self) -> str:
        return TONE_LLM_PROMPT

    async def review(
        self,
        response: str,
        context: dict[str, Any] | None = None,
    ) -> AgentReview:
        """Run tone checks on the response."""
        issues: list[str] = []

        all_patterns = (
            self._ROBOTIC_PATTERNS
            + self._PASSIVE_AGGRESSIVE_PATTERNS
            + self._ARROGANT_PATTERNS
            + self._ES_TONE_PATTERNS
        )

        for pattern, feedback in all_patterns:
            if re.search(pattern, response):
                issues.append(feedback)

        llm_review = await self._run_llm_review(response)
        llm_score = llm_review.score if llm_review else None
        llm_feedback = llm_review.feedback if llm_review else ""
        llm_used = llm_review is not None
        llm_duration = llm_review.llm_review_duration_ms if llm_review else 0.0
        if llm_review is not None and llm_review.score != ReviewScore.PASS:
            issues.append(f"[LLM] {llm_review.feedback}")

        if not issues:
            return self._build_review(
                ReviewScore.PASS,
                agent_name=self.agent_name,
                review_type=self.review_type,
                llm_review_used=llm_used,
                llm_review_duration_ms=llm_duration,
                llm_score=llm_score,
                llm_feedback=llm_feedback,
            )

        has_fail = any(r.score == ReviewScore.FAIL for r in ([llm_review] if llm_review else []))
        if has_fail:
            return AgentReview(
                agent_name=self.agent_name,
                review_type=self.review_type,
                score=ReviewScore.FAIL,
                feedback="; ".join(issues),
                llm_review_used=llm_used,
                llm_review_duration_ms=llm_duration,
                llm_score=llm_score,
                llm_feedback=llm_feedback,
            )

        return AgentReview(
            agent_name=self.agent_name,
            review_type=self.review_type,
            score=ReviewScore.NEEDS_IMPROVEMENT,
            feedback="; ".join(issues),
            llm_review_used=llm_used,
            llm_review_duration_ms=llm_duration,
            llm_score=llm_score,
            llm_feedback=llm_feedback,
        )
