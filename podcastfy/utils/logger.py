"""
Logger Module

This module provides a utility function to set up and configure a logger for the Podcastfy application.
It ensures consistent logging format and configuration across the application.
"""

import logging

from podcastfy.utils.config import load_app_config_model


def setup_logger(name: str) -> logging.Logger:
    """
    Set up and configure a logger.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: A configured logger instance.
    """
    app_config = load_app_config_model()
    logging_config = app_config.logging

    logger = logging.getLogger(name)
    logger.setLevel(logging_config.level)

    formatter = logging.Formatter(logging_config.format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger
