import logging
from listener_effort_api import config

# Global LOGGER to store the logger instance
global LOGGER

def get_logger():
    # If the logger has already been created, return it
    if 'LOGGER' in globals():
        return LOGGER
    
    # Configure logging the first time get_logger is called
    logging_config = config.LoggingConfig()
    logging.basicConfig(
        level=logging_config.level,
        format=logging_config.format,
        datefmt=logging_config.datefmt
    )
    
    # Create and store the logger
    LOGGER = logging.getLogger(__name__)
    return LOGGER
