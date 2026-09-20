"""
完整 checkpoint 示例。

只有当 YAML 中 checkpoint.enabled=true 时才启用完整恢复训练功能。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_metric: float,
    config: dict,
) -> None:
    """保存能够恢复训练的完整状态。"""
    state = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "scaler_state_dict": (
            scaler.state_dict() if scaler is not None else None
        ),
        "config": config,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }

    torch.save(state, path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
) -> dict:
    """加载 checkpoint，不允许 checkpoint 缺失时静默从头训练。"""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint 不存在: {path}")

    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=True)

    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and state["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    if scaler is not None and state["scaler_state_dict"] is not None:
        scaler.load_state_dict(state["scaler_state_dict"])

    torch.set_rng_state(state["torch_rng_state"])

    if (
        torch.cuda.is_available()
        and state["cuda_rng_state_all"] is not None
    ):
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

    return state
