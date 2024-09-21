import os

from dotenv import find_dotenv, load_dotenv

from TelegramLangBot.bot_handlers import echo, help, joke, start
from TelegramLangBot.core.logger_config import logger_config
from TelegramLangBot.models.bot import SupportedHandlerTypes, TelegramApplication

if __name__ == "__main__":
    # Config
    load_dotenv(find_dotenv())
    logger_config("./TelegramLangBot/core/logger_conf.json")
    TOKEN: str = os.environ["BOT_TOKEN"]

    # Initialize and start the bot
    app = TelegramApplication(TOKEN)
    # Add handlers
    app.add_handler(start)
    app.add_handler(help)
    # app.add_handler(echo, handler_type=SupportedHandlerTypes.MESSAGE)
    app.add_handler(joke)
    # Start the bot
    app.start()
