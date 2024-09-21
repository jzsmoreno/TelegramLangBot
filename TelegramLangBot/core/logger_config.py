import json
import logging
import logging.config

logger = logging.getLogger(__name__)


def logger_config(config_path: str = "./logger_conf.json"):
    with open(config_path, "r") as file:
        logging_config = json.load(file)
    logging.config.dictConfig(logging_config)
