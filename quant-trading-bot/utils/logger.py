# utils/logger.py
import logging
import sys

def setup_logger(
    name="quantbot",
    level=logging.INFO,
    format_str="%(asctime)s - %(levelname)s - %(message)s"
):
    """
    Configure and return a logger.
    Call this once at program start (e.g. in main.py) if you want custom behavior.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if logger is already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Create and export a default root logger instance for easy import
# Most modules can just do: from utils.logger import logger
logger = setup_logger()