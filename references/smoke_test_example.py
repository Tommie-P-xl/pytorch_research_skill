"""
轻量级 Smoke Test 示例。

目的：正式训练前，用 1 个 batch 检查 Dataset、forward、loss 和 backward。
默认 CLI 仍只接收 --config。
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn


def run_smoke_test(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> None:
    """执行单 batch 前向与反向传播。"""
    model.train()
    criterion = nn.CrossEntropyLoss()

    inputs, labels = next(iter(dataloader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    logits = model(inputs)
    # 输入示例: (B, C, H, W)
    # 输出示例: (B, K)

    if logits.ndim != 2:
        raise RuntimeError(
            f"模型输出应为二维 logits，实际 shape={tuple(logits.shape)}"
        )

    if logits.size(0) != labels.size(0):
        raise RuntimeError(
            "模型输出 batch size 与标签数量不一致。"
        )

    loss = criterion(logits, labels)
    loss.backward()

    print("Smoke test 通过：Dataset / forward / loss / backward 正常。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML 配置文件路径。",
    )
    return parser.parse_args()
