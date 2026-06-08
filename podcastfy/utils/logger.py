"""
Logger Module

This module provides a utility function to set up and configure a logger for the Podcastfy application.
It ensures consistent logging format and configuration across the application.
"""

import logging
import yaml

def setup_logger(name: str) -> logging.Logger:
    from podcastfy.utils.config import get_config_path
    """
    Set up and configure a logger.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: A configured logger instance.
    """
    config_path = get_config_path()
    raw_config = {}
    if config_path:
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
    logging_config = raw_config.get('logging', {})

    logger = logging.getLogger(name)
    logger.setLevel(logging_config.get('level', 'INFO'))
    
    formatter = logging.Formatter(logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger