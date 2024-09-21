from enum import Enum

from telegram.ext import ApplicationBuilder, BaseHandler, CommandHandler, MessageHandler
from telegram.ext._utils.types import HandlerCallback


class SupportedHandlerTypes(str, Enum):
    COMMAND = "command"
    MESSAGE = "message"

    # Allowing dynamic creation of new values
    @classmethod
    def _missing_(cls, value):
        # If an unsupported value is provided, return it as a string
        return value


class TelegramApplication:
    def __init__(self, token: str):
        self.app = ApplicationBuilder().token(token).build()

    def add_handler(
        self,
        callback: HandlerCallback,
        handler_type: SupportedHandlerTypes = SupportedHandlerTypes.COMMAND,
        **kwargs,
    ):
        HandlerType = BaseHandler
        if handler_type == SupportedHandlerTypes.COMMAND:
            HandlerType = CommandHandler
            kwargs["command"] = callback.__name__ if "command" not in kwargs else kwargs["command"]
        elif handler_type == SupportedHandlerTypes.MESSAGE:
            HandlerType = MessageHandler
            kwargs["filters"] = None if "filters" not in kwargs else kwargs["filters"]
        else:
            raise ValueError(f"Handler type {handler_type} not supported")

        handler = HandlerType(callback=callback, **kwargs)

        self.app.add_handler(handler)

    def start(self):
        self.app.run_polling()
