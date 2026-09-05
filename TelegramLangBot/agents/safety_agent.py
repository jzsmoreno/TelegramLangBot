"""Review agent focused on safety, ethics, and policy compliance."""

from __future__ import annotations

import re
from typing import Any

from .agent_config import (
    AGENT_DEFINITIONS,
    SAFETY_COT_LEAK_PATTERNS,
    SAFETY_DANGEROUS_PATTERNS,
    SAFETY_LLM_PROMPT,
    SAFETY_MISINFO_PATTERNS,
    SAFETY_PII_PATTERN,
)
from .agent_interface import AgentReview, AgentReviewType, BaseAgent, ReviewScore


class SafetyAgent(BaseAgent):
    """Checks whether an LLM response is safe and policy-compliant.

    Flags harmful, offensive, or misleading content.  Also catches
    leaked internal reasoning / chain-of-thought (the system prompt
    explicitly forbids this — see ``agent_config.py``).

    Combines regex patterns with an optional LLM-powered review pass.
    """

    _DANGEROUS_PATTERNS = SAFETY_DANGEROUS_PATTERNS
    _COT_LEAK_PATTERNS = SAFETY_COT_LEAK_PATTERNS
    _MISINFO_PATTERNS = SAFETY_MISINFO_PATTERNS

    _PII_PATTERN = re.compile(SAFETY_PII_PATTERN)

    def __init__(self, llm: Any = None) -> None:
        super().__init__(llm=llm)

    @property
    def agent_name(self) -> str:
        return AGENT_DEFINITIONS["safety"]["name"]

    @property
    def review_type(self) -> AgentReviewType:
        return AgentReviewType.SAFETY

    @property
    def llm_prompt(self) -> str:
        return SAFETY_LLM_PROMPT

    async def review(
        self,
        response: str,
        context: dict[str, Any] | None = None,
    ) -> AgentReview:
        """Run safety checks."""
        issues: list[str] = []
        critical: list[str] = []

        for pattern, feedback in self._DANGEROUS_PATTERNS:
            if re.search(pattern, response):
                critical.append(feedback)

        llm_review = await self._run_llm_review(response)
        llm_score = llm_review.score if llm_review else None
        llm_feedback = llm_review.feedback if llm_review else ""
        llm_used = llm_review is not None
        llm_duration = llm_review.llm_review_duration_ms if llm_review else 0.0
        if llm_review is not None:
            if llm_review.score == ReviewScore.FAIL:
                critical.append(f"[LLM] {llm_review.feedback}")
            elif llm_review.score != ReviewScore.PASS:
                issues.append(f"[LLM] {llm_review.feedback}")

        if critical:
            return AgentReview(
                agent_name=self.agent_name,
                review_type=self.review_type,
                score=ReviewScore.FAIL,
                feedback="; ".join(critical),
                llm_review_used=llm_used,
                llm_review_duration_ms=llm_duration,
                llm_score=llm_score,
                llm_feedback=llm_feedback,
            )

        for pattern, feedback in self._COT_LEAK_PATTERNS:
            if re.search(pattern, response):
                issues.append(feedback)

        for pattern, feedback in self._MISINFO_PATTERNS:
            if re.search(pattern, response):
                issues.append(feedback)

        pii_matches = self._PII_PATTERN.findall(response)
        if pii_matches:
            issues.append(
                "Response may contain personal/sensitive data patterns "
                f"({len(pii_matches)} matches). Verify this is not leaked PII."
            )

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
