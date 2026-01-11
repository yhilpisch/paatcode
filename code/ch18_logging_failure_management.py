from __future__ import annotations

from dataclasses import dataclass  # configuration container
from pathlib import Path  # filesystem paths for log files
from typing import Callable, TypeVar  # generic types for wrappers

import logging  # standard logging framework
from logging.handlers import RotatingFileHandler  # log rotation
import time  # timestamps for retry backoff
import traceback  # text representations of tracebacks

"""
Python & AI for Algorithmic Trading
Chapter 18 -- Logging, Failure Management, and Resilience

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Logging and failure-handling helpers for small trading systems.

This script keeps failure management concrete without introducing
full-blown frameworks. It offers three building blocks that you can
reuse across collectors, strategy loops, and reporting jobs.

1. A compact logging configuration with rotating file handlers.
2. A ``safe_call`` helper that wraps arbitrary callables and records
   structured error messages when they fail.
3. A ``retry_with_backoff`` decorator for background jobs that should
   survive transient errors such as network hiccups or broker timeouts.
"""

LOGGER_NAME = "pyaialgo"

T = TypeVar("T")


@dataclass
class LoggingConfig:
    """Configuration for the rotating log file."""

    path: Path=Path("logs") / "pyaialgo.log"
    level: int=logging.INFO
    max_bytes: int=1_000_000
    backup_count: int=5


def configure_logging(cfg: LoggingConfig) -> logging.Logger:
    """Configure a rotating file logger and return it."""
    cfg.path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(cfg.level)
    logger.handlers.clear()

    handler = RotatingFileHandler(
        filename=str(cfg.path),
        maxBytes=cfg.max_bytes,
        backupCount=cfg.backup_count,
    )
    fmt = (
        "%(asctime)s | %(levelname)s | %(name)s | "
        "%(message)s"
    )
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(stream)

    return logger


def safe_call(func: Callable[..., T], *args, **kwargs) -> T | None:
    """Call ``func`` and log any exception with traceback text."""
    logger = logging.getLogger(LOGGER_NAME)
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover  # defensive clause
        tb_text = "".join(traceback.format_exception(exc))
        logger.error("uncaught exception in %s", func.__name__)
        logger.debug("traceback:\n%s", tb_text)
        return None


def retry_with_backoff(
    max_attempts: int=5,
    initial_delay: float=1.0,
    factor: float=2.0,
) -> Callable[[Callable[..., T]], Callable[..., T | None]]:
    """Decorator that retries a function with exponential backoff."""

    def decorator(func: Callable[..., T]) -> Callable[..., T | None]:
        def wrapper(*args, **kwargs) -> T | None:
            logger = logging.getLogger(LOGGER_NAME)
            delay = initial_delay
            attempt = 0

            while attempt < max_attempts:
                attempt += 1
                result = safe_call(func, *args, **kwargs)
                if result is not None:
                    if attempt > 1:
                        logger.info(
                            "recovered after %d attempts in %s",
                            attempt,
                            func.__name__,
                        )
                    return result

                logger.warning(
                    "attempt %d/%d failed in %s; "
                    "sleeping for %.1f seconds",
                    attempt,
                    max_attempts,
                    func.__name__,
                    delay,
                )
                time.sleep(delay)
                delay *= factor

            logger.error(
                "giving up after %d attempts in %s",
                max_attempts,
                func.__name__,
            )
            return None

        return wrapper

    return decorator


@retry_with_backoff(max_attempts=3, initial_delay=0.5, factor=2.0)
def flaky_demo_job() -> str:
    """Small demo job that fails on purpose sometimes.

    The function pretends to perform useful work but raises a
    ``RuntimeError`` on its first call in order to demonstrate how the
    retry logic behaves.
    """
    if not hasattr(flaky_demo_job, "_counter"):
        flaky_demo_job._counter = 0  # type: ignore[attr-defined]

    flaky_demo_job._counter += 1  # type: ignore[attr-defined]
    counter = flaky_demo_job._counter  # type: ignore[attr-defined]

    if counter == 1:
        raise RuntimeError("simulated transient failure in demo job")
    return f"demo job completed on attempt {counter}"


def main() -> None:
    """Run a small logging and retry demonstration."""
    logger = configure_logging(LoggingConfig())
    logger.info("starting logging demo")

    result = flaky_demo_job()
    if result is not None:
        logger.info("job result: %s", result)
    else:
        logger.error("job failed permanently")


if __name__ == "__main__":
    main()

