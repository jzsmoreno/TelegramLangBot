from functools import wraps

from telegram import Update
from telegram.ext import ContextTypes


def bot_handler(func):
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        assert update.message is not None, "This handler only works with messages"
        assert update.effective_chat is not None, "This handler only works with messages from chats"

        return await func(update, context, *args, **kwargs)

    return wrapped
