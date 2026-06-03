"""Structured logging utilities for AlignAI."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from alignai.config import get_config


def setup_logging(name: Optional[str] = None, level: Optional[str] = None) -> logging.Logger:
    """Configure and return a named logger."""
    cfg = get_config()
    log_level = level or cfg.log_level
    logger = logging.getLogger(name or "alignai")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    return logger
