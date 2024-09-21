import logging
from telegram import Update
from telegram.ext import (
    filters,
    CallbackContext,
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
)
from functools import wraps
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv, find_dotenv
from utils.config import load_config, parse_config
import os

load_dotenv(find_dotenv())

config = load_config("./TelegramLangBot/config.ini")
config = parse_config(config)
users_admin = config["security"]["users_admin"]

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"]
API_VERSION = os.environ["API_VERSION"]
TOKEN = os.environ["TOKEN"]

welcome_message = "¡Hola! Soy tu asistente virtual especializado en People Analytics y Machine Learning. Estoy aquí para responder a tus preguntas sobre cómo estos conceptos pueden ayudar a las organizaciones a tomar decisiones más informadas sobre su talento y mejorar el rendimiento. Puedes preguntarme sobre métodos, herramientas, ejemplos de casos de uso, o cualquier otro tema relacionado. ¿Cómo puedo ayudarte hoy?"
prompt = ChatPromptTemplate.from_template(
    "Actúa como un asistente virtual experto en People Analytics y Machine Learning. "
    "Tu tarea es proporcionar información clara y concisa sobre cómo estos campos pueden ser aplicados en la gestión del talento y la toma de decisiones en organizaciones. "
    "Responde a la pregunta: {pregunta}, de manera amigable y accesible, ofreciendo ejemplos relevantes y recomendaciones de ser posible. "
    "Mantén un enfoque en la utilidad y la aplicabilidad práctica en tu respuesta. "
    "Utiliza un máximo de tres frases y se breve en tu respuesta."
)


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

llm = AzureChatOpenAI(
    model="gpt-4o",
    azure_endpoint=AZURE_ENDPOINT,
    api_key=OPENAI_API_KEY,
    api_version="2023-03-15-preview",
)

chain = prompt | llm | StrOutputParser()


def restricted(func):
    @wraps(func)
    async def wrapped(update, context, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id not in users_admin:
            print(f"You cannot use this bot :(")
            return
        return await func(update, context, *args, **kwargs)

    return wrapped

@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=welcome_message)

@restricted
async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f"¡Hola! {update.effective_user.first_name}")


async def ask_chatgpt(question):
    try:
        return chain.invoke({"pregunta": question})
    except Exception as e:
        return f"An error occurred: {e}"

@restricted
async def handle_question(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    question = update.message.text
    response = await ask_chatgpt(question)
    await update.message.reply_text(response)

@restricted
async def get_user_id(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    await update.message.reply_text(f"Your user ID is: {user_id}")


if __name__ == "__main__":
    application = ApplicationBuilder().token(TOKEN).build()

    start_handler = CommandHandler("start", start)
    hola_handler = CommandHandler("hello", hello)
    id_handler = CommandHandler("id", get_user_id)
    gpt_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question)

    application.add_handler(start_handler)
    application.add_handler(hola_handler)
    application.add_handler(id_handler)
    application.add_handler(gpt_handler)

    application.run_polling()
