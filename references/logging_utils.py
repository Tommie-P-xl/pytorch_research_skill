"""
终端 + 文件日志示例。

目标：关键运行信息既显示在终端，也写入持久化日志。
"""

from __future__ import annotations

import logging
from pathlib import Path


def build_logger(
    name: str,
    log_path: Path,
) -> logging.Logger:
    """创建同时输出到终端和文件的 logger。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_path,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
