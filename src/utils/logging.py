"""
Logging utilities for the project.
"""

from pathlib import Path
from loguru import logger
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent


def setup_logging(
    log_dir: Path = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "30 days"
):
    """
    Configure project-wide logging.
    
    Args:
        log_dir: Directory for log files (default: logs/)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        rotation: Log rotation size
        retention: Log retention period
    """
    if log_dir is None:
        log_dir = PROJECT_ROOT / "logs"
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level
    )
    
    # Add file handler
    logger.add(
        log_dir / "app_{time}.log",
        rotation=rotation,
        retention=retention,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        encoding="utf-8"
    )
    
    return logger

