from typing import cast

from telegram import Chat, Update
from telegram.ext import ContextTypes

from TelegramLangBot.models import bot_handler

WELCOME_MSG = "¡Hola! Soy tu asistente virtual especializado en People Analytics y Machine Learning. Estoy aquí para responder a tus preguntas sobre cómo estos conceptos pueden ayudar a las organizaciones a tomar decisiones más informadas sobre su talento y mkjorar el rendimiento. Puedes preguntarme sobre métodos, herramientas, ejemplos de casos de uso, o cualquier otro tema relacionado. ¿Cómo puedo ayudarte hoy?"


@bot_handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id: int = cast(Chat, update.effective_chat).id
    await context.bot.send_message(chat_id=chat_id, text=WELCOME_MSG)
