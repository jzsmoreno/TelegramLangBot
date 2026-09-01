import ast
from configparser import ConfigParser


def load_config(config_file):
    """Load the configuration file.

    Parameters
    ----------
    config_file : `str`
        The path to the configuration file.

    Returns
    -------
    config : `ConfigParser`
        A configuration object loaded from file.
    """

    config = ConfigParser()
    config.read(config_file)

    return config


def parse_config(config):
    """Parse the configuration file.

    Parameters
    ----------
    config : `ConfigParser`
        A configuration object loaded from file.

    Returns
    -------
    parsed_config : `dict`
        A dictionary of parsed configuration values.
    """

    parsed_config = {}

    for key, child in config.items():
        parsed_config[key] = {}
        for child_key, value in child.items():
            try:
                parsed_value = ast.literal_eval(value)
            except Exception:
                # If the value cannot be evaluated, keep it as a string
                parsed_value = value.strip()
            parsed_config[key][child_key] = parsed_value

    return parsed_config
