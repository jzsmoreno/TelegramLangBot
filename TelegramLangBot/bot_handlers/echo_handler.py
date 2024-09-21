from telegram import Update

from TelegramLangBot.models import bot_handler


@bot_handler
async def echo(update: Update, _):
    assert update.message and update.message.text, "No text in this message, sorry"
    current_msg: str | None = update.message.text
    await update.message.reply_text(current_msg.upper())
