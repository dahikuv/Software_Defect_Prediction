"""Project logging helpers."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a basic console logger.

    The logger is configured once using `basicConfig` and then reused by
    project scripts.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
