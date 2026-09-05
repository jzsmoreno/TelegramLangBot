from __future__ import annotations

from typing import Any, Final, TypedDict


class AgentDefinition(TypedDict):
    """Configuration for an agent."""

    name: str
    review_type: str
    description: str


class PatternDefinition(TypedDict):
    """Configuration for a regex validation pattern."""

    regex: str
    message: str


AGENT_DEFINITIONS: Final[dict[str, AgentDefinition]] = {
    "clarity": {
        "name": "ClarityAgent",
        "review_type": "clarity",
        "description": (
            "Checks whether an LLM response is clear, well-structured, and easy "
            "to read. Flags empty responses, wall-of-text blocks, echoed phrases, "
            "and residual Markdown that Telegram won't render."
        ),
    },
    "helpfulness": {
        "name": "HelpfulnessAgent",
        "review_type": "helpfulness",
        "description": (
            "Verifies the response actually helps the user. Detects deflections, "
            "echoed questions with no substance, placeholder lists, and overly "
            "short answers to definition / explanation questions."
        ),
    },
    "tone": {
        "name": "ToneAgent",
        "review_type": "tone",
        "description": (
            "Checks conversational tone and flags robotic, passive-aggressive, "
            "arrogant, or dismissive language (both English and Spanish)."
        ),
    },
    "safety": {
        "name": "SafetyAgent",
        "review_type": "safety",
        "description": (
            "Flags harmful, offensive, or misleading content. Catches leaked "
            "internal reasoning / chain-of-thought, system-prompt leakage, "
            "misinformation, and potential PII."
        ),
    },
}


CLARITY_ERROR_PATTERNS: Final[list[tuple[str, str]]] = [
    (
        r"(?i)^(lo siento, ocurrió un error|sorry, an error occurred|"
        r"i'm sorry, i cannot|as an ai language model, i)",
        "Response appears to be a generic error / refusal message.",
    ),
    (
        r"^\s*\*$",
        "Response is empty.",
    ),
]

CLARITY_TOO_SHORT_MAX_LEN: Final[int] = 10
CLARITY_WALL_OF_TEXT_THRESHOLD: Final[int] = 500
CLARITY_ECHO_REPEAT_WORDS: Final[str] = r"\b(\w+(?:\s+\w+){1,5})\b\s+\1\s+\1"


HELPFULNESS_DEFLECTION_PATTERNS: Final[list[tuple[str, str]]] = [
    (
        r"(?i)^(as an ai|as a language model|i cannot|i can't|i don't have)",
        "Response deflects instead of answering.",
    ),
    (
        r"(?i)^(please (note|keep in mind|be aware|remember))",
        "Response starts with a disclaimer — may be unhelpful.",
    ),
    (
        r"(?i)^(to answer your question,?\s*)?(it depends|that depends)",
        "Response is non-committal — may frustrate the user.",
    ),
    (
        r"\b(i (hope|trust) this (helps|clarifies|answers)|let me know if you)",
        "Note: closing with 'let me know if…' is fine, flagging for awareness.",
    ),
]

HELPFULNESS_ECHO_QUESTION_RE: Final[str] = (
    r"(?i)^(you asked|your question (is|was|about)|you (are|were) " r"(asking|wondering))"
)

HELPFULNESS_LAZY_LIST_RE: Final[str] = r"^\d+\.\s+[A-Z][a-z]+\.?$"

HELPFULNESS_MIN_AFTER_ECHO: Final[int] = 30
HELPFULNESS_MIN_DEFINITION_RESPONSE: Final[int] = 50
HELPFULNESS_MIN_NUMBERED_ITEMS: Final[int] = 3


SAFETY_DANGEROUS_PATTERNS: Final[list[tuple[str, str]]] = [
    (
        r"(?i)\b(you should kill|you should harm|ways to (harm|hurt|kill|attack))",
        "Response contains harmful / violent suggestions.",
    ),
    (
        r"(?i)\b(how to (hack|crack|steal|bypass|illegally|pirate))",
        "Response appears to provide instructions for illegal activities.",
    ),
    (
        r"(?i)\b(?:you should|you need to|go and)\s+" r"(?:kill|harm|hurt|attack)\b",
        "Response contains a direct harmful or violent suggestion.",
    ),
]

SAFETY_DISCRIMINATORY_PATTERNS: Final[list[tuple[str, str]]] = [
    (
        r"(?i)\b(discriminat(e|ion)|racist|sexist|homophobic|xenophobic)",
        "Response contains potentially discriminatory language.",
    ),
]

SAFETY_SYSTEM_LEAK_PATTERNS: Final[list[tuple[str, str]]] = [
    (
        r"(?i)(expert people analytics|machine learning assistant|"
        r"USER\s*=\s*human|ASSISTANT\s*=\s*AI|Never confuse identities)",
        "Response leaks system instructions — detected system prompt content.",
    ),
    (
        r"(?i)(Don't invent information|reveal internal reasoning|"
        r"maximum \d+ sentences|Match the USER's language)",
        "Response leaks system prompt behavioral rules.",
    ),
    (
        r"(?i)(Reason internally, but never reveal|"
        r"Analyze the current USER message carefully|"
        r"Use relevant history and distinguish)",
        "Response leaks Chain-of-Thought system instructions.",
    ),
    (
        r"(?i)\b(system prompt|system instructions|developer instructions)\b",
        "Response appears to expose internal instructions.",
    ),
]

SAFETY_COT_LEAK_PATTERNS: Final[list[tuple[str, str]]] = [
    (
        r"(?i)\b(step\s*\d+\s*:|reasoning:|thinking:|let me think|"
        r"let['’]s (reason|think|analyze))",
        "Response leaked chain-of-thought reasoning — must be removed.",
    ),
    (
        r"(?i)\b(first,?\s*i|then,?\s*i|next,?\s*i|finally,?\s*i)\s+"
        r"(need to|should|must|would|will|can)",
        "Response reveals internal reasoning process.",
    ),
    (
        r"(?i)\b(I (should|need to|must) " r"(be careful|consider|remember|note|mention|address))",
        "Response reveals meta-reasoning about what to include.",
    ),
    (
        r"(?i)\b(My analysis|based on my analysis|after analyzing|upon reflection)",
        "Response reveals internal analysis steps.",
    ),
    (
        r"(?i)^\s*(Okay,|Alright,|Sure,|Got it,)\s*"
        r"(let me|I'll|I will)\s*(help|explain|answer|respond)",
        "Response reveals internal processing preamble.",
    ),
]

SAFETY_MISINFO_PATTERNS: Final[list[tuple[str, str]]] = [
    (
        r"(?i)\b(it['’]s (scientifically|medically) proven that "
        r"(vaccines?|medications?) (cause|don't))",
        "Response makes unqualified medical claims — verify or remove.",
    ),
    (
        r"(?i)\b(the (only|absolute|100%|guaranteed) way)\b",
        "Response makes absolute claims that may be misleading.",
    ),
]

SAFETY_PII_PATTERN: Final[str] = (
    r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b"
    r"|\b\d{13,19}\b"
    r"|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)


TONE_ROBOTIC_PATTERNS: Final[list[tuple[str, str]]] = [
    (
        r"(?i)^(as per your|per your|pursuant to your|"
        r"in response to your (inquiry|query|request))",
        "Response sounds robotic / overly formal.",
    ),
    (
        r"(?i)^(greetings!|salutations!|dear user,|hello esteemed)",
        "Response uses stilted / unnatural greetings.",
    ),
]

TONE_PASSIVE_AGGRESSIVE_PATTERNS: Final[list[tuple[str, str]]] = [
    (
        r"(?i)\b(as I (already|previously|just) " r"(said|mentioned|stated|explained|noted))",
        "Phrase 'as I already said' sounds passive-aggressive — rephrase.",
    ),
    (
        r"(?i)\b(you (should have|ought to have|need to understand that))",
        "Phrase sounds condescending — consider a more supportive tone.",
    ),
    (
        r"(?i)\b(it['’]s (obvious|clear) that\b|anyone can see that)",
        "Phrase dismisses the user's potential confusion — avoid.",
    ),
    (
        r"(?i)\b(actually,?\s+no\b)",
        "Starting with 'actually, no' can sound dismissive.",
    ),
]

TONE_ARROGANT_PATTERNS: Final[list[tuple[str, str]]] = [
    (
        r"(?i)\b(this is (the only|the best|undoubtedly|absolutely|"
        r"definitively) (correct|right|proper))",
        "Overly confident / dogmatic phrasing — use a more measured tone.",
    ),
    (
        r"(?i)\b(everybody knows|obviously|it goes without saying)",
        "Assuming shared knowledge can sound arrogant — consider a helpful tone.",
    ),
]

TONE_ES_PATTERNS: Final[list[tuple[str, str]]] = [
    (
        r"(?i)\b(como ya (te |le |)dije|como ya (te |le |)mencion[eé])",
        "Suena pasivo-agresivo — considera reformular.",
    ),
    (
        r"(?i)\b(es obvio que|está claro que|cualquiera " r"(sabe|entiende) (que|esto))",
        "Suena arrogante — considera un tono más servicial.",
    ),
    (
        r"(?i)\b(deberías haber|tendrías que haber|necesitas entender que)",
        "Suena condescendiente — reformula con un tono de ayuda.",
    ),
]


CLARITY_LLM_PROMPT: Final[str] = (
    "You are a clarity reviewer for LLM-generated Telegram bot responses.\n"
    "Review the text below for readability and structure. Return ONLY a JSON object:\n"
    '{"score": "PASS"|"NEEDS_IMPROVEMENT"|"FAIL", '
    '"feedback": "brief explanation"}\n\n'
    "Check for: walls of text without line breaks, echoed/repeated phrases, "
    "residual Markdown formatting, overly short incomplete responses, "
    "generic error or refusal messages.\n"
    "Do not penalize concise answers when the user's request is simple.\n"
    "Score PASS only if the response is clear, appropriately structured, "
    "and easy to read."
)

HELPFULNESS_LLM_PROMPT: Final[str] = (
    "You are a helpfulness reviewer for LLM-generated Telegram bot responses.\n"
    "Review the text below for usefulness. Return ONLY a JSON object:\n"
    '{"score": "PASS"|"NEEDS_IMPROVEMENT"|"FAIL", '
    '"feedback": "brief explanation"}\n\n'
    "Check for: deflections, echoed questions with no substance, "
    "placeholder lists, unnecessary disclaimers, and insufficiently "
    "substantive responses.\n"
    "Do not penalize a correct refusal when the request is unsafe or impossible.\n"
    "Do not require a long answer when a short answer fully satisfies the request.\n"
    "Score PASS only if the response actually helps and provides substantive content."
)

TONE_LLM_PROMPT: Final[str] = (
    "You are a tone reviewer for LLM-generated Telegram bot responses.\n"
    "Review the text below for conversational tone (English and Spanish).\n"
    "Return ONLY a JSON object:\n"
    '{"score": "PASS"|"NEEDS_IMPROVEMENT"|"FAIL", '
    '"feedback": "brief explanation"}\n\n'
    "Check for: robotic/overly-formal language, passive-aggressive phrasing, "
    "arrogant/dogmatic/dismissive language, and condescending tone.\n"
    "Do not penalize professional language when appropriate, directness, "
    "justified uncertainty, or concise responses.\n"
    "Score PASS only if the response sounds friendly, natural, and helpful."
)

SAFETY_LLM_PROMPT: Final[str] = (
    "You are a safety reviewer for LLM-generated Telegram bot responses.\n\n"
    "Review the response below for safety and policy compliance.\n\n"
    "OUTPUT FORMAT:\n"
    "Return ONLY one valid JSON object with exactly these two fields:\n"
    '- "score": a string that is exactly one of '
    '"PASS", "NEEDS_IMPROVEMENT", or "FAIL"\n'
    '- "feedback": a brief string explaining the decision\n\n'
    "Example of valid output:\n"
    '{"score":"PASS","feedback":"No safety issues detected."}\n\n'
    "IMPORTANT:\n"
    "- Do not use Markdown code fences.\n"
    "- Do not include text before or after the JSON object.\n"
    "- Do not add additional fields.\n"
    '- Do not write "PASS"|"FAIL" or similar syntax; choose exactly one value.\n\n'
    "Review for:\n"
    "- harmful or violent suggestions\n"
    "- dangerous or illegal activity instructions\n"
    "- discriminatory or hateful language\n"
    "- leaked system or developer instructions\n"
    "- leaked hidden prompts or chain-of-thought\n"
    "- potentially misleading medical or absolute claims\n"
    "- potential PII or sensitive personal information\n\n"
    "Context matters. Mentioning a harmful topic is not automatically harmful.\n"
    "A safe refusal or high-level safety information should not be marked as harmful.\n"
    "Score FAIL for harmful or dangerous content.\n"
    "Score NEEDS_IMPROVEMENT for content with a meaningful safety concern "
    "that is not clearly dangerous.\n"
    "Score PASS only when the response is safe and policy-compliant."
)


BASE_RULES: Final[str] = """
Expert People Analytics & Machine Learning assistant.

USER = human; ASSISTANT = AI.

Never confuse identities or attribute USER's personal information to ASSISTANT.

"My..." always refers to USER.

USER messages are user facts; ASSISTANT messages are not USER facts unless
explicitly confirmed.

Prefer explicit USER information and accept corrections.

Use relevant history and answer the current message directly.

Match the USER's language.

Be clear, objective, practical, friendly, and concise.

Don't invent information or reveal internal reasoning.
""".strip()


COT_RULES: Final[str] = """
Analyze the current USER message carefully before answering.

Use relevant history and distinguish USER-provided facts from
ASSISTANT-generated content.

Reason internally, but never reveal your reasoning or chain of thought.
""".strip()


REWRITE_SYSTEM_PROMPT: Final[str] = (
    "You are an expert editor. Improve the following assistant response "
    "based on the reviewer feedback. Fix the issues while preserving "
    "the original meaning, helpful content, and language. "
    "Do NOT add chain-of-thought, meta-reasoning, or preambles like "
    "'Okay, let me fix this'. Output ONLY the improved response."
)


SUMMARY_SYSTEM_PROMPT: Final[str] = (
    "You manage a compact conversation memory.\n\n"
    "Your job is to maintain a cumulative summary that acts as overflow memory "
    "for conversation history that no longer fits in recent context.\n\n"
    "Rules:\n"
    "- USER is the person interacting with the assistant.\n"
    "- ASSISTANT is the AI.\n"
    "- Only treat information explicitly provided or confirmed by USER as user facts.\n"
    "- Never convert assistant assumptions, guesses, or suggestions into user facts.\n"
    "- Preserve information that may be relevant to future turns: identity, role, "
    "professional context, goals, preferences, decisions, requirements, constraints, "
    "technical context, important entities, unresolved issues, and corrections.\n"
    "- Preserve the latest confirmed value when information changes or is corrected.\n"
    "- Remove greetings, small talk, repetition, obsolete details, and irrelevant content.\n"
    "- Prefer compact factual statements over narrative prose.\n"
    "- Do not repeat information already present in the existing summary.\n"
    "- Do not invent, infer, or speculate.\n"
    "- The summary must remain useful even if the original messages are no longer available.\n"
    "- Treat the existing summary as trusted memory, but update it when newer USER information "
    "contradicts it.\n"
    "- Incorporate the new messages into the existing summary.\n"
    "- Never store passwords, API keys, authentication tokens, or other credentials.\n"
    "- Return only the updated summary, with no preamble or explanation."
)


WELCOME_MESSAGE: Final[str] = (
    "¡Hola! Soy tu asistente virtual especializado en People Analytics y "
    "Machine Learning. Estoy aquí para responder a tus preguntas sobre "
    "cómo estos conceptos pueden ayudar a las organizaciones a tomar "
    "decisiones más informadas sobre su talento y mejorar el rendimiento. "
    "Puedes preguntarme sobre métodos, herramientas, ejemplos de casos "
    "de uso, o cualquier otro tema relacionado. ¿Cómo puedo ayudarte hoy?"
)


def _export_patterns(
    patterns: list[tuple[str, str]],
) -> list[PatternDefinition]:
    """Convert regex/message tuples into serializable dictionaries."""
    return [{"regex": regex, "message": message} for regex, message in patterns]


def export_agent_config() -> dict[str, Any]:
    """Export all agent definitions, patterns, formatting rules, and prompts."""
    return {
        "agents": AGENT_DEFINITIONS,
        "patterns": {
            "clarity_error_patterns": _export_patterns(CLARITY_ERROR_PATTERNS),
            "clarity_wall_of_text_threshold": CLARITY_WALL_OF_TEXT_THRESHOLD,
            "clarity_too_short_max_len": CLARITY_TOO_SHORT_MAX_LEN,
            "clarity_echo_repeat_words": CLARITY_ECHO_REPEAT_WORDS,
            "helpfulness_deflection_patterns": _export_patterns(HELPFULNESS_DEFLECTION_PATTERNS),
            "helpfulness_echo_question_re": HELPFULNESS_ECHO_QUESTION_RE,
            "helpfulness_lazy_list_re": HELPFULNESS_LAZY_LIST_RE,
            "helpfulness_min_after_echo": HELPFULNESS_MIN_AFTER_ECHO,
            "helpfulness_min_definition_response": (HELPFULNESS_MIN_DEFINITION_RESPONSE),
            "helpfulness_min_numbered_items": HELPFULNESS_MIN_NUMBERED_ITEMS,
            "safety_dangerous_patterns": _export_patterns(SAFETY_DANGEROUS_PATTERNS),
            "safety_cot_leak_patterns": _export_patterns(SAFETY_COT_LEAK_PATTERNS),
            "safety_misinfo_patterns": _export_patterns(SAFETY_MISINFO_PATTERNS),
            "safety_pii_pattern": SAFETY_PII_PATTERN,
            "tone_robotic_patterns": _export_patterns(TONE_ROBOTIC_PATTERNS),
            "tone_passive_aggressive_patterns": _export_patterns(TONE_PASSIVE_AGGRESSIVE_PATTERNS),
            "tone_arrogant_patterns": _export_patterns(TONE_ARROGANT_PATTERNS),
            "tone_es_patterns": _export_patterns(TONE_ES_PATTERNS),
        },
        "system_prompts": {
            "base_rules": BASE_RULES,
            "cot_rules": COT_RULES,
            "rewrite_system_prompt": REWRITE_SYSTEM_PROMPT,
            "summary_system_prompt": SUMMARY_SYSTEM_PROMPT,
            "welcome_message": WELCOME_MESSAGE,
        },
    }
