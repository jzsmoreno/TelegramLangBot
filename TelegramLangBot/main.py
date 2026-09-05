import asyncio
import datetime
import io
import json
import logging
import os
import sys
import time
from functools import wraps
from typing import Any, Callable, Coroutine

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from telegram import Update
from telegram.constants import ParseMode
from telegram.error import NetworkError, RetryAfter, TimedOut
from telegram.ext import (
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .agents.agent_config import SUMMARY_SYSTEM_PROMPT as SUMMARY_SYSTEM_PROMPT_CONFIG
from .agents.agent_config import WELCOME_MESSAGE
from .agents.review_coordinator import ReviewCoordinator
from .monitor_server import MonitorServer
from .utils.config import load_config, parse_config
from .utils.prompt_templates import cot_prompt, prompt
from .utils.telegram_format import clean_telegram_html

load_dotenv(find_dotenv())

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

config = load_config("./TelegramLangBot/config.ini")
config = parse_config(config)
users_admin = config["security"]["users_admin"]


def get_env(key: str) -> str:
    value = os.getenv(key)

    if not value:
        logger.error("Missing required env var %s", key)
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

welcome_message = WELCOME_MESSAGE


if AZURE_ENDPOINT:
    summary_llm = AzureChatOpenAI(
        model=SUMMARY_MODEL,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version=API_VERSION,
        temperature=0.8,
        max_tokens=1024,
    )

    llm = AzureChatOpenAI(
        model=AZURE_MODEL,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version=API_VERSION,
        temperature=0.8,
        max_tokens=2048,
    )
else:
    summary_llm = ChatOpenAI(
        model=SUMMARY_MODEL,
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        temperature=0.8,
        max_tokens=1024,
    )

    llm = ChatOpenAI(
        model=LOCAL_MODEL,
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        temperature=0.8,
        max_tokens=2048,
    )


chain = prompt | llm | StrOutputParser()
cot_chain = cot_prompt | llm | StrOutputParser()

review_coordinator = ReviewCoordinator(llm=llm)


conversation_history: dict[
    int,
    list[HumanMessage | AIMessage],
] = {}

conversation_summaries: dict[int, str] = {}

chat_cot_mode: dict[int, bool] = {}

SUMMARY_SYSTEM_PROMPT = SUMMARY_SYSTEM_PROMPT_CONFIG

background_tasks: set[asyncio.Task[Any]] = set()


def create_background_task(coroutine: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    task = asyncio.create_task(coroutine)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return task


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
            logger.warning(
                "Empty summary generated | chat_id=%s",
                chat_id,
            )
            return

        conversation_summaries[chat_id] = summary

        logger.info(
            "Summary overflow updated | chat_id=%s | messages_summarized=%s | summary_length=%s",
            chat_id,
            len(messages),
            len(summary),
        )

    except asyncio.CancelledError:
        raise

    except Exception:
        logger.exception(
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

    if conversation_history.get(chat_id) is history:
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

    logger.info(
        "Memory diagnostic | chat_id=%s | history_messages=%s | "
        "history_turns=%s | has_summary=%s | cot=%s",
        chat_id,
        len(history),
        len(history) // 2,
        bool(summary),
        chat_cot_mode.get(chat_id, False),
    )

    if not logger.isEnabledFor(logging.DEBUG):
        return

    logger.debug(
        "\n%s\nCHAT ID: %s\nHISTORY LENGTH: %s\nSUMMARY AVAILABLE: %s\n%s",
        "=" * 80,
        chat_id,
        len(history),
        bool(summary),
        "-" * 80,
    )

    if summary:
        logger.debug(
            "CONVERSATION SUMMARY:\n%s\n%s",
            summary,
            "-" * 80,
        )

    logger.debug("RECENT HISTORY:")

    for index, message in enumerate(history, start=1):
        if isinstance(message, HumanMessage):
            role = "USER"
        elif isinstance(message, AIMessage):
            role = "ASSISTANT"
        else:
            role = type(message).__name__

        logger.debug(
            "%d. %s: %s",
            index,
            role,
            message.content,
        )

    logger.debug(
        "%s\nCURRENT QUESTION: %s\n%s",
        "-" * 80,
        question,
        "=" * 80,
    )


async def _run_review_and_report(
    chat_id: int,
    question: str,
    raw_answer: str,
    user_id: int,
) -> str:
    """Review the answer through all agents, apply corrections, and report to monitor.

    Returns the best final response after all agent reviews and corrections.
    """
    formatted = clean_telegram_html(raw_answer)

    review_result = await review_coordinator.review(
        formatted,
        context={"user_question": question},
    )

    reviews_data = [
        {
            "agent_name": review.agent_name,
            "review_type": review.review_type.value,
            "score": review.score.name,
            "feedback": review.feedback,
            "suggested_fix": review.suggested_fix,
        }
        for review in review_result.agent_reviews
    ]

    for index, review in enumerate(
        review_result.agent_reviews,
        start=1,
    ):
        try:
            monitor.send_agent_step(
                chat_id,
                review.agent_name,
                review.review_type.value,
                review.score.name,
                review.feedback,
                review.suggested_fix,
                index,
                len(review_result.agent_reviews),
                formatted,
            )

            if review.llm_review_used:
                monitor.send_agent_llm_activity(
                    chat_id,
                    review.agent_name,
                    review.review_type.value,
                    review.llm_score.name if review.llm_score else "N/A",
                    review.llm_feedback or "No LLM feedback",
                    review.llm_review_duration_ms,
                    review.score.name,
                )

        except Exception:
            logger.exception(
                "Failed to report review agent result | chat_id=%s | agent=%s",
                chat_id,
                review.agent_name,
            )

    try:
        monitor.send_review(
            chat_id,
            review_result.passed,
            review_result.requires_rewrite,
            reviews_data,
            review_result.rewrite_feedback,
        )
    except Exception:
        logger.exception(
            "Failed to report review result | chat_id=%s",
            chat_id,
        )

    final_text = clean_telegram_html(
        review_result.final_response,
    )

    try:
        monitor.send_response(
            chat_id,
            final_text,
            len(final_text),
            chat_cot_mode.get(chat_id, False),
        )
    except Exception:
        logger.exception(
            "Failed to report final response | chat_id=%s",
            chat_id,
        )

    return final_text


async def ask_chatgpt(
    chat_id: int,
    question: str,
    user_id: int = 0,
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

        prompt_chars = len(question) + sum(len(str(message.content)) for message in history)

        t0 = time.monotonic()

        answer = await selected_chain.ainvoke(
            {
                "history": history,
                "context_summary": summary,
                "pregunta": question,
            }
        )

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        est_prompt_tokens = max(
            1,
            prompt_chars // 4,
        )

        est_comp_tokens = max(
            1,
            len(answer) // 4,
        )

        try:
            monitor.send_token_usage(
                chat_id,
                user_id,
                est_prompt_tokens,
                est_comp_tokens,
                elapsed_ms,
            )
        except Exception:
            logger.exception(
                "Failed to report token usage | chat_id=%s",
                chat_id,
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

        create_background_task(trim_history(chat_id))

        return answer

    except asyncio.CancelledError:
        raise

    except Exception:
        logger.exception(
            "Error while querying the model | chat_id=%s",
            chat_id,
        )

        try:
            monitor.send_error(
                chat_id,
                "LLM query failed",
            )
        except Exception:
            logger.exception(
                "Failed to report LLM error | chat_id=%s",
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
                try:
                    await update.message.reply_text(
                        "Lo siento, no tienes permiso para usar este bot."
                    )
                except (NetworkError, TimedOut) as exc:
                    logger.warning(
                        "Failed to send permission message | user_id=%s | error=%s",
                        user_id,
                        exc,
                    )

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

    try:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=welcome_message,
        )
    except (NetworkError, TimedOut) as exc:
        logger.warning(
            "Failed to send start message | chat_id=%s | error=%s",
            update.effective_chat.id,
            exc,
        )


@restricted
async def hello(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if update.message is None or update.effective_user is None:
        return

    try:
        await update.message.reply_text(f"¡Hola! {update.effective_user.first_name}")
    except (NetworkError, TimedOut) as exc:
        logger.warning(
            "Failed to send hello message | user_id=%s | error=%s",
            update.effective_user.id,
            exc,
        )


async def keep_typing(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
) -> None:
    try:
        while True:
            try:
                await context.bot.send_chat_action(
                    chat_id=chat_id,
                    action="typing",
                )

            except RetryAfter as exc:
                await asyncio.sleep(
                    max(
                        1,
                        int(exc.retry_after),
                    )
                )

            except (NetworkError, TimedOut) as exc:
                logger.warning(
                    "Typing indicator network error | chat_id=%s | error=%s",
                    chat_id,
                    exc,
                )

            await asyncio.sleep(4)

    except asyncio.CancelledError:
        raise


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
    user_id = update.effective_user.id if update.effective_user else 0

    username = (
        update.effective_user.username
        if update.effective_user and update.effective_user.username
        else ""
    )

    try:
        monitor.send_request(
            chat_id,
            user_id,
            question,
            username,
        )
    except Exception:
        logger.exception(
            "Failed to report incoming request | chat_id=%s",
            chat_id,
        )

    typing_task = asyncio.create_task(
        keep_typing(
            context,
            chat_id,
        )
    )

    try:
        raw_answer = await ask_chatgpt(
            chat_id,
            question,
            user_id,
        )

        # Run review through all agents and apply corrections BEFORE sending to user
        final_answer = await _run_review_and_report(
            chat_id=chat_id,
            question=question,
            raw_answer=raw_answer,
            user_id=user_id,
        )

        try:
            await update.message.reply_text(
                final_answer,
                parse_mode=ParseMode.HTML,
            )

        except (NetworkError, TimedOut) as exc:
            logger.warning(
                "Network error sending answer | chat_id=%s | error=%s",
                chat_id,
                exc,
            )
            return

        except Exception:
            logger.warning(
                "Failed to send answer as HTML, retrying as plain text | chat_id=%s",
                chat_id,
            )

            try:
                await update.message.reply_text(
                    final_answer,
                )
            except (NetworkError, TimedOut) as exc:
                logger.warning(
                    "Network error sending plain answer | chat_id=%s | error=%s",
                    chat_id,
                    exc,
                )
                return

    except asyncio.CancelledError:
        raise

    except Exception:
        logger.exception(
            "Failed to handle question | chat_id=%s",
            chat_id,
        )

    finally:
        typing_task.cancel()

        try:
            await typing_task
        except asyncio.CancelledError:
            pass


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

    try:
        monitor.send_clear(chat_id)
    except Exception:
        logger.exception(
            "Failed to report memory clear | chat_id=%s",
            chat_id,
        )

    try:
        await update.message.reply_text("🧹 Memoria de conversación borrada. Empezamos de nuevo.")
    except (NetworkError, TimedOut) as exc:
        logger.warning(
            "Failed to send memory clear response | chat_id=%s | error=%s",
            chat_id,
            exc,
        )


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


@restricted
async def export_data(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Export conversation data as a JSON file. Admin-only.

    Usage: /export [user_id]
      - No argument: exports all conversations.
      - With user_id: exports only conversations involving that user.
    """
    if update.effective_chat is None or update.message is None:
        return

    user_id_filter: int | None = None

    if context.args:
        try:
            user_id_filter = int(context.args[0])
        except ValueError:
            await update.message.reply_text(
                "Uso: /export [user_id]\n" "Ejemplo: /export 1505936813"
            )
            return

    all_conversations = monitor._conversations

    if user_id_filter is not None:
        filtered: dict[int, list[dict[str, Any]]] = {}

        for chat_id, messages in all_conversations.items():
            matching = [message for message in messages if message.get("user_id") == user_id_filter]

            if matching:
                filtered[chat_id] = matching

        export_data_dict: dict[str, Any] = {
            "export_info": {
                "user_id_filter": user_id_filter,
                "chat_count": len(filtered),
                "exported_at": datetime.datetime.now().isoformat(),
            },
            "conversations": filtered,
        }

        filename = f"export_user_{user_id_filter}.json"

    else:
        export_data_dict = {
            "export_info": {
                "all_users": True,
                "chat_count": len(all_conversations),
                "exported_at": datetime.datetime.now().isoformat(),
            },
            "conversations": dict(all_conversations),
        }

        filename = "export_all_users.json"

    json_str = json.dumps(
        export_data_dict,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    byte_io = io.BytesIO(json_str.encode("utf-8"))

    byte_io.name = filename

    await update.message.reply_document(
        document=byte_io,
        filename=filename,
        caption=(
            f"Export {'filtered by user ' + str(user_id_filter) if user_id_filter else 'all users'}\n"
            f"{export_data_dict['export_info']['chat_count']} conversation(s) exported."
        ),
    )


def build_application():
    """Build the Telegram bot application with all handlers."""
    application = ApplicationBuilder().token(TOKEN).concurrent_updates(True).build()

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
        CommandHandler(
            "export",
            export_data,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_question,
        )
    )

    return application


MONITOR_PORT = int(
    os.getenv(
        "MONITOR_PORT",
        "8080",
    )
)

monitor = MonitorServer(
    port=MONITOR_PORT,
)


def main() -> None:
    monitor.start()

    application = build_application()

    try:
        application.run_polling(
            drop_pending_updates=True,
        )

    except KeyboardInterrupt:
        logger.info("Bot shutdown requested")

    except Exception:
        logger.exception("Telegram application stopped unexpectedly")


if __name__ == "__main__":
    main()
