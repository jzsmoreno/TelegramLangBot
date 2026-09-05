"""Review agent focused on response helpfulness and completeness."""

from __future__ import annotations

import re
from typing import Any

from .agent_config import (
    AGENT_DEFINITIONS,
    HELPFULNESS_DEFLECTION_PATTERNS,
    HELPFULNESS_ECHO_QUESTION_RE,
    HELPFULNESS_LAZY_LIST_RE,
    HELPFULNESS_LLM_PROMPT,
    HELPFULNESS_MIN_AFTER_ECHO,
    HELPFULNESS_MIN_DEFINITION_RESPONSE,
    HELPFULNESS_MIN_NUMBERED_ITEMS,
)
from .agent_interface import AgentReview, AgentReviewType, BaseAgent, ReviewScore


class HelpfulnessAgent(BaseAgent):
    """Checks whether an LLM response actually helps the user.

    Uses rule-based heuristics combined with an optional LLM-powered pass:
    verifies the response doesn't deflect, doesn't just echo the question
    back, and provides substantive content.
    """

    _DEFLECTION_PATTERNS = HELPFULNESS_DEFLECTION_PATTERNS

    _ECHO_QUESTION_RE = re.compile(HELPFULNESS_ECHO_QUESTION_RE)

    _LAZY_LIST_RE = re.compile(HELPFULNESS_LAZY_LIST_RE, re.MULTILINE)

    def __init__(self, llm: Any = None) -> None:
        super().__init__(llm=llm)

    @property
    def agent_name(self) -> str:
        return AGENT_DEFINITIONS["helpfulness"]["name"]

    @property
    def review_type(self) -> AgentReviewType:
        return AgentReviewType.HELPFULNESS

    @property
    def llm_prompt(self) -> str:
        return HELPFULNESS_LLM_PROMPT

    async def review(
        self,
        response: str,
        context: dict[str, Any] | None = None,
    ) -> AgentReview:
        """Run helpfulness checks."""
        issues: list[str] = []

        stripped = response.strip()

        for pattern, feedback in self._DEFLECTION_PATTERNS[:-1]:
            if re.search(pattern, stripped):
                if "note:" not in feedback.lower():
                    issues.append(feedback)
                    break

        echo_match = self._ECHO_QUESTION_RE.search(stripped[:200])
        if echo_match:
            after_echo = stripped[echo_match.end() :].strip(" ,.")
            if len(after_echo) < HELPFULNESS_MIN_AFTER_ECHO:
                issues.append(
                    "Response echoes the user's question back but provides no substantive answer."
                )

        numbered_items = self._LAZY_LIST_RE.findall(stripped)
        if len(numbered_items) >= HELPFULNESS_MIN_NUMBERED_ITEMS:
            issues.append(
                f"Response contains {len(numbered_items)} short numbered items "
                "that may be placeholders rather than real content. "
                "Consider expanding each point."
            )

        if context and "user_question" in context:
            user_q = str(context["user_question"]).lower().rstrip("? !.")
            if re.match(r"^(what|qu[ée]|cu[aá]l|define|explain)\b", user_q):
                if len(stripped) < HELPFULNESS_MIN_DEFINITION_RESPONSE:
                    issues.append(
                        "User asked a definition / explanation question but "
                        "the response is very short — likely incomplete."
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

        critical_issue = any("deflects" in i for i in issues)
        return AgentReview(
            agent_name=self.agent_name,
            review_type=self.review_type,
            score=ReviewScore.FAIL if critical_issue else ReviewScore.NEEDS_IMPROVEMENT,
            feedback="; ".join(issues),
            llm_review_used=llm_used,
            llm_review_duration_ms=llm_duration,
            llm_score=llm_score,
            llm_feedback=llm_feedback,
        )
