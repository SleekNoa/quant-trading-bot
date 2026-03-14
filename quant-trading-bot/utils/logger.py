# utils/logger.py
import logging
import os
import sys
from config.settings import SYMBOL



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

def setup_trade_logger(symbol=SYMBOL, level=logging.INFO, mode="w", force=False):
    """
    Creates a logger that writes to logs/<symbol>.csv
    Replaces previous file on each run.
    """
    logger = logging.getLogger("trade_logger")
    logger.setLevel(level)

    if force and logger.handlers:
        for handler in list(logger.handlers):
            try:
                handler.close()
            finally:
                logger.removeHandler(handler)

    if not symbol:
        return logger

    os.makedirs("logs", exist_ok=True)
    csv_path = f"logs/{symbol}.csv"

    if not logger.handlers:
        handler = logging.FileHandler(csv_path, mode=mode)
        handler.setLevel(level)

        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.info(f"[trade_logger] writing {csv_path}")

    return logger

def reset_trade_logger(symbol=SYMBOL, level=logging.INFO, mode="w"):
    """Force a fresh file for the given symbol (useful for backtests)."""
    return setup_trade_logger(symbol=symbol, level=level, mode=mode, force=True)

trade_logger = setup_trade_logger(symbol=None)
