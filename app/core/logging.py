import logging
import structlog
from app.core.config import settings


def configure_logging() -> None:
    """
    Set up structlog for the application.
    - Development (LOG_LEVEL=INFO/DEBUG): coloured, human-readable console output.
    - Production (LOG_LEVEL=WARNING+): JSON output, one object per line.
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure the standard-library root logger so uvicorn / third-party
    # libraries that use logging also respect our level.
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
    )

    is_dev = settings.log_level.upper() in ("DEBUG", "INFO")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if is_dev
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Return a module-level structlog logger."""
    return structlog.get_logger(name)
