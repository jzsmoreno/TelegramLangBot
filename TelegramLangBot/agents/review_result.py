"""Final review result produced by the multi-agent review system."""

from __future__ import annotations

from dataclasses import dataclass, field

from .agent_interface import AgentReview


@dataclass
class ReviewResult:
    """Holds the outcome of a multi-agent review pass.

    Attributes
    ----------
    original_response : str
        The LLM response that was reviewed (after ``clean_telegram_html``).
    final_response : str
        The response that should be sent to Telegram — either the original
        (if it passed) or an improved version.
    passed : bool
        ``True`` when all agents passed — no XL rewrite was needed.
    requires_rewrite : bool
        ``True`` when the coordinator asked the LLM to rewrite.
    agent_reviews : list[AgentReview]
        Full set of individual agent reviews.
    rewrite_feedback : str
        Accumulated feedback sent to the LLM during rewrite, if any.
    """

    original_response: str
    final_response: str
    passed: bool
    requires_rewrite: bool = False
    agent_reviews: list[AgentReview] = field(default_factory=list)
    rewrite_feedback: str = ""
