import asyncio
import logging
import os
import sys
from functools import wraps
from typing import Any, Callable, Coroutine

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .utils.config import load_config, parse_config
from .utils.prompt_templates import cot_prompt, prompt
from .utils.telegram_format import clean_telegram_html

load_dotenv(find_dotenv())


config = load_config("./TelegramLangBot/config.ini")
config = parse_config(config)
users_admin = config["security"]["users_admin"]


def get_env(key: str) -> str:
    value = os.getenv(key)

    if not value:
        logging.error("Missing required env var %s", key)
        sys.exit(1)

    return value


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION", "2023-03-15-preview")
TOKEN = get_env("TOKEN")

AZURE_MODEL = os.getenv("AZURE_MODEL", "gpt-4o")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "gemma-3-4b-it")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gemma-3-4b-it")

MAX_TURNS = 10
MAX_HISTORY_MESSAGES = MAX_TURNS * 2


welcome_message = (
    "¡Hola! Soy tu asistente virtual especializado en People Analytics y "
    "Machine Learning. Estoy aquí para responder a tus preguntas sobre cómo "
    "estos conceptos pueden ayudar a las organizaciones a tomar decisiones "
    "más informadas sobre su talento y mejorar el rendimiento. Puedes "
    "preguntarme sobre métodos, herramientas, ejemplos de casos de uso, o "
    "cualquier otro tema relacionado. ¿Cómo puedo ayudarte hoy?"
)


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


if AZURE_ENDPOINT:
    summary_llm = AzureChatOpenAI(
        model=SUMMARY_MODEL,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version=API_VERSION,
        temperature=0.2,
        max_tokens=1024,
    )
else:
    summary_llm = ChatOpenAI(
        model=SUMMARY_MODEL,
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        temperature=0.2,
        max_tokens=1024,
    )


if AZURE_ENDPOINT:
    llm = AzureChatOpenAI(
        model=AZURE_MODEL,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version=API_VERSION,
        temperature=0.2,
        max_tokens=2048,
    )
else:
    llm = ChatOpenAI(
        model=LOCAL_MODEL,
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        temperature=0.2,
        max_tokens=2048,
    )


chain = prompt | llm | StrOutputParser()
cot_chain = cot_prompt | llm | StrOutputParser()


conversation_history: dict[
    int,
    list[HumanMessage | AIMessage],
] = {}

conversation_summaries: dict[int, str] = {}

chat_cot_mode: dict[int, bool] = {}


SUMMARY_SYSTEM_PROMPT = (
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
    "- Return only the updated summary, with no preamble or explanation."
)


def get_history(
    chat_id: int,
) -> list[HumanMessage | AIMessage]:
    return conversation_history.setdefault(chat_id, [])


def format_messages(
    messages: list[HumanMessage | AIMessage],
) -> str:
    formatted_messages: list[str] = []

    for message in messages:
        if isinstance(message, HumanMessage):
            role = "USER"
        elif isinstance(message, AIMessage):
            role = "ASSISTANT"
        else:
            role = type(message).__name__

        formatted_messages.append(f"{role}: {message.content}")

    return "\n".join(formatted_messages)


async def summarize_context(
    chat_id: int,
    messages: list[HumanMessage | AIMessage],
) -> None:
    if not messages:
        return

    existing_summary = conversation_summaries.get(
        chat_id,
        "",
    )

    conversation = (
        "EXISTING SUMMARY:\n"
        f"{existing_summary or 'No previous summary exists.'}\n\n"
        "MESSAGES MOVING TO LONG-TERM MEMORY:\n"
        f"{format_messages(messages)}\n\n"
        "Update the existing summary using these messages."
    )

    summary_messages = [
        SystemMessage(
            content=SUMMARY_SYSTEM_PROMPT,
        ),
        HumanMessage(
            content=conversation,
        ),
    ]

    try:
        response = await summary_llm.ainvoke(
            summary_messages,
        )

        summary = response.content

        if not isinstance(summary, str):
            summary = str(summary)

        summary = summary.strip()

        if not summary:
            logging.warning(
                "Empty summary generated | chat_id=%s",
                chat_id,
            )
            return

        conversation_summaries[chat_id] = summary

        logging.info(
            "Summary overflow updated | chat_id=%s | " "messages_summarized=%s | summary_length=%s",
            chat_id,
            len(messages),
            len(summary),
        )

    except Exception:
        logging.exception(
            "Error while updating overflow summary | chat_id=%s",
            chat_id,
        )


async def trim_history(
    chat_id: int,
) -> None:
    history = get_history(chat_id)

    if len(history) <= MAX_HISTORY_MESSAGES:
        return

    overflow_count = len(history) - MAX_HISTORY_MESSAGES

    if overflow_count % 2 != 0:
        overflow_count -= 1

    if overflow_count <= 0:
        return

    overflow = history[:overflow_count]

    await summarize_context(
        chat_id=chat_id,
        messages=overflow,
    )

    conversation_history[chat_id] = history[overflow_count:]


def log_history(
    chat_id: int,
    question: str,
) -> None:
    history = get_history(chat_id)
    summary = conversation_summaries.get(
        chat_id,
        "",
    )

    logging.info(
        "Memory diagnostic | chat_id=%s | history_messages=%s | "
        "history_turns=%s | has_summary=%s | cot=%s",
        chat_id,
        len(history),
        len(history) // 2,
        bool(summary),
        chat_cot_mode.get(chat_id, False),
    )

    print("\n" + "=" * 80)
    print(f"CHAT ID: {chat_id}")
    print(f"HISTORY LENGTH: {len(history)}")
    print(f"SUMMARY AVAILABLE: {bool(summary)}")
    print("-" * 80)

    if summary:
        print("CONVERSATION SUMMARY:")
        print(summary)
        print("-" * 80)

    print("RECENT HISTORY:")

    for index, message in enumerate(
        history,
        start=1,
    ):
        if isinstance(message, HumanMessage):
            role = "USER"
        elif isinstance(message, AIMessage):
            role = "ASSISTANT"
        else:
            role = type(message).__name__

        print(f"{index}. {role}: {message.content}")

    print("-" * 80)
    print(f"CURRENT QUESTION: {question}")
    print("=" * 80 + "\n")


async def ask_chatgpt(
    chat_id: int,
    question: str,
) -> str:
    try:
        history = get_history(chat_id)

        summary = conversation_summaries.get(
            chat_id,
            "",
        )

        use_cot = chat_cot_mode.get(
            chat_id,
            False,
        )

        log_history(
            chat_id,
            question,
        )

        selected_chain = cot_chain if use_cot else chain

        answer = await selected_chain.ainvoke(
            {
                "history": history,
                "context_summary": summary,
                "pregunta": question,
            }
        )

        history.append(
            HumanMessage(
                content=question,
            )
        )

        history.append(
            AIMessage(
                content=answer,
            )
        )

        await trim_history(chat_id)

        return clean_telegram_html(answer)

    except Exception:
        logging.exception(
            "Error while querying the model | chat_id=%s",
            chat_id,
        )

        return "Lo siento, ocurrió un error al procesar tu pregunta."


def restricted(
    func: Callable[..., Coroutine[Any, Any, Any]],
) -> Callable[..., Coroutine[Any, Any, Any]]:
    @wraps(func)
    async def wrapped(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if update.effective_user is None:
            return None

        user_id = update.effective_user.id

        if user_id not in users_admin:
            if update.message:
                await update.message.reply_text("Lo siento, no tienes permiso para usar este bot.")

            return None

        return await func(
            update,
            context,
            *args,
            **kwargs,
        )

    return wrapped


@restricted
async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if update.effective_chat is None:
        return

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=welcome_message,
    )


@restricted
async def hello(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if update.message is None or update.effective_user is None:
        return

    await update.message.reply_text(f"¡Hola! {update.effective_user.first_name}")


async def keep_typing(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
) -> None:
    try:
        while True:
            await context.bot.send_chat_action(
                chat_id=chat_id,
                action="typing",
            )

            await asyncio.sleep(4)

    except asyncio.CancelledError:
        pass


@restricted
async def handle_question(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if update.effective_chat is None or update.message is None:
        return

    question = update.message.text

    if not question:
        return

    chat_id = update.effective_chat.id

    typing_task = asyncio.create_task(
        keep_typing(
            context,
            chat_id,
        )
    )

    try:
        response = await ask_chatgpt(
            chat_id,
            question,
        )

        await update.message.reply_text(
            response,
            parse_mode=ParseMode.HTML,
        )

    finally:
        typing_task.cancel()


async def clear_memory(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if update.effective_chat is None or update.message is None:
        return

    chat_id = update.effective_chat.id

    conversation_history.pop(
        chat_id,
        None,
    )

    conversation_summaries.pop(
        chat_id,
        None,
    )

    await update.message.reply_text("🧹 Memoria de conversación borrada. Empezamos de nuevo.")


@restricted
async def show_memory(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if update.effective_chat is None or update.message is None:
        return

    chat_id = update.effective_chat.id

    history = get_history(chat_id)

    summary = conversation_summaries.get(
        chat_id,
        "",
    )

    use_cot = chat_cot_mode.get(
        chat_id,
        False,
    )

    await update.message.reply_text(
        "🧠 Estado de memoria\n\n"
        f"Chat ID: {chat_id}\n"
        f"Mensajes recientes: {len(history)}\n"
        f"Turnos recientes: {len(history) // 2}/{MAX_TURNS}\n"
        f"Resumen disponible: {'Sí' if summary else 'No'}\n"
        f"Longitud del resumen: {len(summary)} caracteres\n"
        f"Modo CoT: {'ON' if use_cot else 'OFF'}"
    )


@restricted
async def show_summary(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if update.effective_chat is None or update.message is None:
        return

    chat_id = update.effective_chat.id

    summary = conversation_summaries.get(
        chat_id,
    )

    if not summary:
        await update.message.reply_text("🧠 No existe un resumen acumulado para esta conversación.")

        return

    await update.message.reply_text(f"🧠 Resumen acumulado\n\n{summary}")


async def get_user_id(
    update: Update,
    context: CallbackContext,
) -> None:
    if update.message is None or update.effective_user is None:
        return

    await update.message.reply_text(f"Your user ID is: {update.effective_user.id}")


@restricted
async def set_cot(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if update.effective_chat is None or update.message is None:
        return

    chat_id = update.effective_chat.id

    if not context.args:
        current_mode = chat_cot_mode.get(
            chat_id,
            False,
        )

        await update.message.reply_text(
            f"Uso: /cot on | off\n" f"Estado actual: {'ON' if current_mode else 'OFF'}"
        )

        return

    option = context.args[0].lower()

    if option == "on":
        chat_cot_mode[chat_id] = True

        await update.message.reply_text("Modo CoT activado.")

    elif option == "off":
        chat_cot_mode[chat_id] = False

        await update.message.reply_text("Modo CoT desactivado.")

    else:
        await update.message.reply_text("Uso: /cot on | off")


def build_application():
    application = ApplicationBuilder().token(TOKEN).build()

    application.add_handler(
        CommandHandler(
            "clear",
            clear_memory,
        )
    )

    application.add_handler(
        CommandHandler(
            "memory",
            show_memory,
        )
    )

    application.add_handler(
        CommandHandler(
            "summary",
            show_summary,
        )
    )

    application.add_handler(
        CommandHandler(
            "start",
            start,
        )
    )

    application.add_handler(
        CommandHandler(
            "hello",
            hello,
        )
    )

    application.add_handler(
        CommandHandler(
            "id",
            get_user_id,
        )
    )

    application.add_handler(
        CommandHandler(
            "cot",
            set_cot,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_question,
        )
    )

    return application


def main() -> None:
    application = build_application()
    application.run_polling()


if __name__ == "__main__":
    main()
