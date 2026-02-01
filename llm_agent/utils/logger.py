import logging
import os

def setup_logger(name=__name__, log_file='llm_activities.log', level=logging.INFO):
    """Function to setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Check if log file is absolute path, if not assume relative to CWD
    if not os.path.isabs(log_file):
        log_file = os.path.join(os.getcwd(), log_file)

    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(console_handler)

    return logger

# Create a default logger instance
logger = setup_logger("llm_agent")
