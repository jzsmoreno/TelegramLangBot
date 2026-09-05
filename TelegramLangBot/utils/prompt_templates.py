from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from ..agents.agent_config import BASE_RULES, COT_RULES

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
