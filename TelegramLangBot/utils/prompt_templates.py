from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

BASE_RULES = """
Expert People Analytics & Machine Learning assistant.

USER = human; ASSISTANT = AI.
Never confuse identities or attribute USER's personal information to ASSISTANT.
"My..." always refers to USER.
USER messages are user facts; ASSISTANT messages are not USER facts unless explicitly confirmed.
Prefer explicit USER information and accept corrections.

Use relevant history and answer the current message directly.
Match the USER's language.
Be clear, objective, practical, friendly, and concise.
Don't invent information or reveal internal reasoning.
Maximum 3 sentences.
"""

COT_RULES = """
Analyze the current USER message carefully before answering.
Use relevant history and distinguish USER-provided facts from ASSISTANT-generated content.
Reason internally, but never reveal your reasoning or chain of thought.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=BASE_RULES),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("CURRENT USER MESSAGE:\n{pregunta}"),
    ]
)

cot_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=BASE_RULES + COT_RULES),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("CURRENT USER MESSAGE:\n{pregunta}"),
    ]
)
