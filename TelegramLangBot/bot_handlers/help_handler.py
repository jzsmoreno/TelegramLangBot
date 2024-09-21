from telegram import Update
from telegram.ext import ContextTypes


async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    assert update.message, "No message sent"
    if not context.args:
        return await update.message.reply_text(
            "Usa /help <comando> para obtener ayuda, por ejemplo /help joke"
        )
    command = context.args[0]
    if command == "joke":
        return await update.message.reply_text(
            "Usa /joke <tema> para obtener un chiste sobre el tema que quieras\n"
            "Por ejemplo, /joke programming"
        )
    else:
        return await update.message.reply_text(
            f"No se que es {command}, usa /help <comando> para obtener ayuda, por ejemplo /help joke"
        )
