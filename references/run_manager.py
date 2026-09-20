"""
Run Directory 与实验环境记录示例。

每次正式实验创建独立目录，避免不同实验互相覆盖。
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


def create_run_directory(
    project_root: Path,
    config: dict[str, Any],
) -> Path:
    """创建本次实验独立目录及常用子目录。"""
    output_root = project_root / config["experiment"]["output_root"]
    experiment_name = config["experiment"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = output_root / f"{timestamp}_{experiment_name}"

    for subdir in (
        "logs",
        "checkpoints",
        "metrics",
        "figures",
    ):
        (run_dir / subdir).mkdir(parents=True, exist_ok=False)

    return run_dir


def save_config_snapshot(
    config: dict[str, Any],
    run_dir: Path,
) -> None:
    """保存本次实验实际使用的配置快照。"""
    output_path = run_dir / "resolved_config.yaml"
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            allow_unicode=True,
            sort_keys=False,
        )


def save_environment_info(run_dir: Path, seed: int) -> None:
    """保存 Python、PyTorch、CUDA、GPU、Git 等环境信息。"""
    info: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "seed": seed,
        "gpu": None,
        "git_commit": None,
    }

    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        info["git_commit"] = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        # Git 信息属于辅助元数据，获取失败不应阻止实验。
        pass

    output_path = run_dir / "environment.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(info, file, indent=2, ensure_ascii=False)
