"""Logging setup: short progress on the console, full detail in a rotating log file (agent.log).

Every module logs through a child of the "agent" logger (agent.main, agent.tools, agent.jev).
Nothing is configured on import, so the tests and any other code stay silent unless they call setup_logging().
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "agent"
DEFAULT_LOG_FILE = Path(__file__).with_name("agent.log")  # next to the code, wherever the app is started from


class _StdoutHandler(logging.StreamHandler):
    """Writes to whatever sys.stdout is at the moment of logging, so redirecting it later still works."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value) -> None:  # the base class assigns it in __init__; the property ignores that
        pass


class _HideToolRecords(logging.Filter):
    """Tool failures are already visible in the [step N] output, so the console skips the duplicate."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name != "agent.tools"


def setup_logging(log_file: Path | str = DEFAULT_LOG_FILE) -> None:
    """Console: plain INFO messages. File: everything (DEBUG) with time, level, logger name and thread."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):  # calling this twice must not print every line twice
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console = _StdoutHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    console.addFilter(_HideToolRecords())

    file = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8")  # agent.log, agent.log.1, ...
    file.setLevel(logging.DEBUG)
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s [%(threadName)s] %(message)s"))

    logger.addHandler(console)
    logger.addHandler(file)
