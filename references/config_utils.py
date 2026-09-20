"""
配置读取与启动前合法性检查示例。

原则：
- YAML 是实验参数的唯一来源；
- 非法配置尽早报错；
- 不静默修正用户输入。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置。"""
    path = Path(config_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("配置文件顶层必须是 YAML mapping。")

    return config


def validate_config(config: dict[str, Any], project_root: Path) -> None:
    """
    在正式训练前检查关键配置。

    此函数只演示通用检查。项目应继续补充模型和任务特有规则。
    """
    seed = config["reproducibility"]["seed"]
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("reproducibility.seed 必须是非负整数。")

    batch_size = config["training"]["batch_size"]
    if batch_size <= 0:
        raise ValueError("training.batch_size 必须大于 0。")

    dataset_root = project_root / config["paths"]["dataset_root"]
    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_root}")

    split_cfg = config["split"]
    mode = split_cfg["mode"]

    if mode == "standard":
        total = (
            float(split_cfg["train_ratio"])
            + float(split_cfg["val_ratio"])
            + float(split_cfg["test_ratio"])
        )
        if abs(total - 1.0) > 1e-8:
            raise ValueError(
                "standard 模式下 train/val/test 比例之和必须等于 1。"
            )
    elif mode == "merged_eval":
        total = (
            float(split_cfg["train_ratio"])
            + float(split_cfg["val_ratio"])
        )
        if abs(total - 1.0) > 1e-8:
            raise ValueError(
                "merged_eval 模式下 train/eval 比例之和必须等于 1。"
            )
    else:
        raise ValueError(
            f"不支持的 split.mode: {mode!r}。"
        )

    model_cfg = config["model"]
    embed_dim = int(model_cfg["embed_dim"])
    num_heads = int(model_cfg["num_heads"])

    if embed_dim % num_heads != 0:
        raise ValueError(
            "model.embed_dim 必须能被 model.num_heads 整除。"
        )
