"""
Structured logging. Keeps it simple but production-shaped: timestamp, level,
logger name, message. Swap formatter for JSON if you ship logs to a
collector later.
"""
import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.setLevel(level.upper())
    # Avoid duplicate handlers if called twice (e.g. uvicorn reload)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)

    # Quiet some chatty libs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
