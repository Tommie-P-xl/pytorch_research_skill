"""
Dataset / DataLoader 风格参考示例。

重点：
- 初始化时只扫描一次目录；
- 文件排序保证可复现；
- 默认 lazy loading；
- DataLoader worker 使用统一 seed；
- 大型 npy 数据可选择 mmap 读取。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .reproducibility import build_generator, seed_worker


class NpyClassificationDataset(Dataset):
    """按类别子目录组织的 NPY 分类数据集。"""

    def __init__(
        self,
        root: Path,
        transform: Callable | None = None,
        mmap_mode: str | None = "r",
    ) -> None:
        self.root = root
        self.transform = transform
        self.mmap_mode = mmap_mode

        class_dirs = sorted(
            path
            for path in root.iterdir()
            if path.is_dir()
        )

        self.class_to_idx = {
            path.name: index
            for index, path in enumerate(class_dirs)
        }

        self.samples: list[tuple[Path, int]] = []

        for class_dir in class_dirs:
            label = self.class_to_idx[class_dir.name]
            files = sorted(class_dir.glob("*.npy"))
            self.samples.extend(
                (file_path, label)
                for file_path in files
            )

        if not self.samples:
            raise RuntimeError(
                f"数据集为空或没有找到 .npy 文件: {root}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]

        array = np.load(
            path,
            mmap_mode=self.mmap_mode,
        )
        array = np.asarray(array, dtype=np.float32)

        x = torch.from_numpy(array.copy()).unsqueeze(0)
        # (H, W) -> (1, H, W)

        if self.transform is not None:
            x = self.transform(x)

        return x, label


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    drop_last: bool,
) -> DataLoader:
    """构建具备可复现 worker 随机性的 DataLoader。"""
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "worker_init_fn": seed_worker,
        "generator": build_generator(seed),
    }

    # num_workers=0 时 PyTorch 不允许设置以下多进程参数。
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**kwargs)
