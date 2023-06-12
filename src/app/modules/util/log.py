"""
Module for logging utilities

This module contains functions to configure logging for the application.

Classes:
    CustomFormatter: Logging colored formatter

Functions:
    load_config: Load the config file from the given path
    configure_logging: Configure logging using the given config

Variables:
    LOG_CONFIG: Path to the logging config file

attributes:
    log: Logger object for the application
"""
import json
from pathlib import Path
import logging
from logging import Formatter, LogRecord
from logging.config import dictConfig

LOG_CONFIG = "app/config/log_config.json"
LOG_CONFIG = Path.cwd().joinpath(LOG_CONFIG)


class CustomFormatter(Formatter):
    """Logging colored formatter"""

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.formats = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset,
        }

    def format(self, record: LogRecord):
        """Format the given log
        Args:
            record (LogRecord): Log to format
        Returns:
            str: Formatted log
        """
        log_fmt = self.formats.get(record.levelno)
        formatter = Formatter(log_fmt)
        return formatter.format(record)


def load_config(config_path: str = LOG_CONFIG) -> dict:
    """Load the config file from the given path
    Args:
        config_path (str, optional): Path to the config file.
            Defaults to LOG_CONFIG.
    Returns:
        dict: Config dictionary
    """
    with open(config_path, "r", encoding="utf8") as file:
        config = json.load(file)
    return config


def configure_logging(formatter: Formatter = None, config: dict = None):
    """Configure logging using the given config
    Args:
        formatter (Formatter, optional): Formatter to use.
            Defaults to None.
        config (dict, optional): Config dictionary.
            Defaults to None.
    Returns:
        Logger: Logger object for the application
    """
    if config is None:
        config = load_config()

    dictConfig(config)

    logger = logging.getLogger("color")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    if formatter is None:
        formatter = CustomFormatter(config["formatters"]["default"]["format"])

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


log = configure_logging()
