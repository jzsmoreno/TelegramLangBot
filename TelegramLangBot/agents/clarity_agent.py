"""Review agent focused on response clarity and readability."""

from __future__ import annotations

import re
from typing import Any

from .agent_config import (
    AGENT_DEFINITIONS,
    CLARITY_ECHO_REPEAT_WORDS,
    CLARITY_ERROR_PATTERNS,
    CLARITY_LLM_PROMPT,
    CLARITY_TOO_SHORT_MAX_LEN,
    CLARITY_WALL_OF_TEXT_THRESHOLD,
)
from .agent_interface import AgentReview, AgentReviewType, BaseAgent, ReviewScore


class ClarityAgent(BaseAgent):
    """Checks whether an LLM response is clear, well-structured, and easy to read.

    Combines rule-based heuristics with an optional LLM-powered pass for deeper
    semantic checks.
    """

    _ERROR_PATTERNS = CLARITY_ERROR_PATTERNS

    _TOO_SHORT_PATTERN = re.compile(f"^.{{1,{CLARITY_TOO_SHORT_MAX_LEN}}}$")

    _WALL_OF_TEXT_RE = re.compile(rf"[^\n]{{{CLARITY_WALL_OF_TEXT_THRESHOLD},}}")

    _ECHO_PATTERN = re.compile(CLARITY_ECHO_REPEAT_WORDS)

    _MARKDOWN_BOLD_RE = re.compile(r"\*\*[^*]+\*\*")
    _MARKDOWN_ITALIC_RE = re.compile(r"(?<!\*)\*[^*\n]+\*(?!\*)")

    def __init__(self, llm: Any = None) -> None:
        super().__init__(llm=llm)

    @property
    def agent_name(self) -> str:
        return AGENT_DEFINITIONS["clarity"]["name"]

    @property
    def review_type(self) -> AgentReviewType:
        return AgentReviewType.CLARITY

    @property
    def llm_prompt(self) -> str:
        return CLARITY_LLM_PROMPT

    async def review(
        self,
        response: str,
        context: dict[str, Any] | None = None,
    ) -> AgentReview:
        """Run clarity checks and return an :class:`AgentReview`."""
        issues: list[str] = []
        suggestions: list[str] = []

        for pattern, feedback in self._ERROR_PATTERNS:
            if re.search(pattern, response):
                return self._build_review(
                    ReviewScore.FAIL,
                    feedback,
                    agent_name=self.agent_name,
                    review_type=self.review_type,
                )

        if self._TOO_SHORT_PATTERN.match(response.strip()):
            issues.append("Response is very short — it may not fully address the question.")
        else:
            wall_matches = self._WALL_OF_TEXT_RE.findall(response)
            if wall_matches:
                issues.append(
                    f"Found {len(wall_matches)} unbroken block(s) of text over 500 chars. "
                    "Consider adding line breaks, bullet points, or sections for readability."
                )
                improved = response
                for long_line in wall_matches:
                    chunks = re.split(r"\.\s+", long_line)
                    if len(chunks) > 1:
                        broken = ".\n".join(chunks[:3])
                        improved = improved.replace(long_line, broken, 1)
                if improved != response:
                    suggestions.append(improved)

        if self._ECHO_PATTERN.search(response):
            issues.append("Response contains echoed / repeated phrases — it may be unnatural.")

        if self._MARKDOWN_BOLD_RE.search(response):
            issues.append(
                "Residual Markdown bold (`**`) detected. " "Telegram won't render this correctly."
            )
        if self._MARKDOWN_ITALIC_RE.search(response):
            issues.append(
                "Residual Markdown italic (`*`) detected. " "Telegram won't render this correctly."
            )

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

        if suggestions:
            return AgentReview(
                agent_name=self.agent_name,
                review_type=self.review_type,
                score=ReviewScore.NEEDS_IMPROVEMENT,
                feedback="; ".join(issues),
                suggested_fix=suggestions[0],
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
