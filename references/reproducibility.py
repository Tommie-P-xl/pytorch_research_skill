"""
实验可复现工具。

功能：
1. 统一设置 Python、NumPy、PyTorch 和 CUDA 随机种子；
2. 为 DataLoader worker 提供稳定随机种子；
3. 根据配置控制确定性行为。
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    设置全局随机种子。

    参数:
        seed: 全局随机种子。
        deterministic: 是否优先保证确定性。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False


def seed_worker(worker_id: int) -> None:
    """
    初始化 DataLoader worker 的随机种子。

    PyTorch 会为不同 worker 生成不同初始 seed，这里将该 seed 同步到
    NumPy 和 Python random，避免多进程数据增强出现不可复现行为。
    """
    del worker_id  # worker_id 仅用于表达接口语义

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_generator(seed: int) -> torch.Generator:
    """构建供 DataLoader 使用的固定随机数生成器。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
