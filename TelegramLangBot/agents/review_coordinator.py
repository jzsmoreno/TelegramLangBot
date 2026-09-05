"""Coordinate multiple review agents and optionally rewrite LLM responses."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Final

from langchain_core.messages import HumanMessage, SystemMessage

from .agent_config import REWRITE_SYSTEM_PROMPT
from .agent_interface import AgentReview, BaseAgent, ReviewScore
from .clarity_agent import ClarityAgent
from .helpfulness_agent import HelpfulnessAgent
from .review_result import ReviewResult
from .safety_agent import SafetyAgent
from .tone_agent import ToneAgent

logger = logging.getLogger(__name__)


class ReviewCoordinator:
    """Coordinate response review, correction, and optional LLM rewriting.

    Parameters
    ----------
    agents:
        Custom review agents. When omitted, the default five review agents
        are created.
    llm:
        Optional LangChain-compatible LLM supporting ``ainvoke(messages)``.
        The LLM is used to rewrite responses that require improvement.
    """

    _REWRITE_SYSTEM_PROMPT: Final[str] = REWRITE_SYSTEM_PROMPT

    def __init__(
        self,
        agents: list[BaseAgent] | None = None,
        llm: Any | None = None,
    ) -> None:
        """Initialize the review coordinator."""
        self._llm = llm
        self._agents = agents if agents is not None else self._create_default_agents(llm)

    @staticmethod
    def _create_default_agents(llm: Any | None) -> list[BaseAgent]:
        """Create the default response-review agents."""
        return [
            ClarityAgent(llm=llm),
            HelpfulnessAgent(llm=llm),
            ToneAgent(llm=llm),
            SafetyAgent(llm=llm),
        ]

    async def review(
        self,
        response: str,
        context: dict[str, Any] | None = None,
    ) -> ReviewResult:
        """Review a response, apply fixes, and optionally request a rewrite."""
        reviews = await self._run_all_agents(response, context)

        current_response = self._apply_suggested_fixes(
            response=response,
            reviews=reviews,
        )

        if not reviews:
            logger.warning("No review agents returned a result.")
            return ReviewResult(
                original_response=response,
                final_response=current_response,
                passed=False,
                requires_rewrite=False,
                agent_reviews=reviews,
            )

        scores = [review.score for review in reviews]
        all_pass = all(score == ReviewScore.PASS for score in scores)
        any_fail = any(score == ReviewScore.FAIL for score in scores)
        needs_improvement = any(score == ReviewScore.NEEDS_IMPROVEMENT for score in scores)

        if all_pass:
            logger.info("All review agents passed the response.")
            return ReviewResult(
                original_response=response,
                final_response=current_response,
                passed=True,
                requires_rewrite=False,
                agent_reviews=reviews,
            )

        feedback_lines = self._collect_feedback(reviews)

        if self._llm is not None and (any_fail or needs_improvement):
            logger.info(
                "Requesting LLM rewrite for %d review issue(s).",
                len(feedback_lines),
            )

            try:
                rewritten = await self._request_rewrite(
                    response=current_response,
                    feedback_lines=feedback_lines,
                    context=context,
                )
            except Exception:
                logger.exception("LLM rewrite failed.")
                rewritten = None

            if rewritten:
                return ReviewResult(
                    original_response=response,
                    final_response=rewritten,
                    passed=False,
                    requires_rewrite=True,
                    agent_reviews=reviews,
                    rewrite_feedback="; ".join(feedback_lines),
                )

        return ReviewResult(
            original_response=response,
            final_response=current_response,
            passed=False,
            requires_rewrite=False,
            agent_reviews=reviews,
        )

    async def _run_all_agents(
        self,
        response: str,
        context: dict[str, Any] | None,
    ) -> list[AgentReview]:
        """Run all configured review agents concurrently."""
        if not self._agents:
            return []

        tasks = [agent.review(response, context=context) for agent in self._agents]

        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        reviews: list[AgentReview] = []

        for agent, result in zip(self._agents, results):
            if isinstance(result, Exception):
                logger.error(
                    "Review agent %s failed: %s",
                    agent.agent_name,
                    result,
                )
                continue

            reviews.append(result)

            if result.score != ReviewScore.PASS:
                logger.debug(
                    "[%s] %s: %s",
                    result.score.name,
                    result.agent_name,
                    result.feedback,
                )

        return reviews

    @staticmethod
    def _apply_suggested_fixes(
        response: str,
        reviews: list[AgentReview],
    ) -> str:
        """Apply available agent-provided fixes in review order."""
        current_response = response

        for review in reviews:
            suggested_fix = review.suggested_fix

            if not suggested_fix or suggested_fix == current_response:
                continue

            logger.info(
                "Applying suggested fix from %s.",
                review.agent_name,
            )

            current_response = suggested_fix

        return current_response

    @staticmethod
    def _collect_feedback(
        reviews: list[AgentReview],
    ) -> list[str]:
        """Collect unique non-passing reviewer feedback."""
        seen: set[str] = set()
        feedback_lines: list[str] = []

        for review in reviews:
            if review.score == ReviewScore.PASS:
                continue

            feedback = review.feedback.strip()

            if not feedback or feedback in seen:
                continue

            seen.add(feedback)
            feedback_lines.append(f"[{review.review_type.value}] {feedback}")

        return feedback_lines

    async def _request_rewrite(
        self,
        response: str,
        feedback_lines: list[str],
        context: dict[str, Any] | None,
    ) -> str | None:
        """Request an improved response from the configured LLM."""
        if self._llm is None:
            return None

        user_content = self._build_rewrite_prompt(
            response=response,
            feedback_lines=feedback_lines,
            context=context,
        )

        messages = [
            SystemMessage(content=self._REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        result = await self._llm.ainvoke(messages)
        rewritten = self._extract_llm_content(result)

        if not rewritten or rewritten == response.strip():
            return None

        return rewritten

    @staticmethod
    def _build_rewrite_prompt(
        response: str,
        feedback_lines: list[str],
        context: dict[str, Any] | None,
    ) -> str:
        """Build the user message supplied to the rewrite LLM."""
        sections = [
            "ORIGINAL RESPONSE:",
            response.strip(),
            "",
            "REVIEWER FEEDBACK:",
        ]

        sections.extend(f"- {feedback}" for feedback in feedback_lines)

        if context:
            sections.extend(
                [
                    "",
                    "RELEVANT CONTEXT:",
                    str(context),
                ]
            )

        sections.extend(
            [
                "",
                "Return only the improved response text.",
            ]
        )

        return "\n".join(sections)

    @staticmethod
    def _extract_llm_content(result: Any) -> str | None:
        """Extract and normalize text content from an LLM response."""
        content = getattr(result, "content", result)

        if content is None:
            return None

        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item) for item in content
            ).strip()
        else:
            text = str(content).strip()

        return text or None
