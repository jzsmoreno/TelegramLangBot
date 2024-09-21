import os

import requests as req
from telegram import Update
from telegram.ext import ContextTypes

from TelegramLangBot.models import bot_handler


@bot_handler
async def joke(update: Update, context: ContextTypes.DEFAULT_TYPE):
    assert update.message and update.message.text, "No text in this message, sorry"
    if not context.args:
        return await update.message.reply_text(f"Tienes que hacer una pregunta junto al comando")

    msg = " ".join(context.args)
    api_url = os.environ["API_URL"] + "/chat/chat-completion?bot_name=JokeBot"

    res = req.post(api_url, json={"text": msg})
    if res.status_code != 200:
        return await update.message.reply_text(f"Error en la respuesta del servidor: {res.text}")

    data = res.json()
    return await update.message.reply_text(data["data"]["message"])
