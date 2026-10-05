---
name: pytorch-data-pipeline
description: Use when 已选择 PyTorch 科研项目的数据流程细则，并进入 Dataset、数据划分、DataLoader 或可选缓存的实现阶段。
---

# 数据流程与效率

仅在主入口记录启用本模块、或用户直接要求数据流程规范时读取。跳过本模块仍需遵守主入口的种子与数据泄漏边界。

## 输入与阶段输出

输入：数据格式、标签含义、划分协议、全局 seed、数据规模和缓存选择。
输出：可复现样本索引、明确 split、可取 batch 的 Dataset/DataLoader；缓存只在用户选择时实现。

## 数据与划分

- 初始化只扫描一次并排序；`__getitem__` 按索引 lazy loading，避免逐样本重新遍历目录。
- 样本清单保存可移植相对路径。训练增强只应用于训练；验证/测试不随机改变评估协议。
- 长期对比保存 split manifest 与类别映射，防止同 seed 因输入新增或排序变化改变划分。
- 随机划分使用统一 seed；时间序列、相邻帧或同一对象需按任务分组/时间隔离。先确认协议，不擅自将时间划分改成随机。
- 检查精确样本重叠，也区分同场景/同对象的相关性；标准化、特征统计等仅在许可训练集合拟合。
- `standard` 使用 train/val/test；`merged_eval` 使用 train/eval，并明确其承担共同评估职责。
- Open Set/OOD 分别声明 known/unknown 集合、训练许可与阈值来源；不假定每个任务都允许 unknown 参与训练。

## DataLoader 与缓存

- worker/generator 使用全局 seed 派生随机状态；Windows 多进程入口使用主入口保护，worker 回调放在模块顶层。
- `num_workers=0` 时不传 `prefetch_factor` 或开启 `persistent_workers`。验证/测试默认不 shuffle、不 drop_last。
- 根据设备配置 pin_memory 与 non_blocking，不盲目一次性载入全部大数据。
- DDP 训练使用 DistributedSampler 等显式分片；每 epoch 调用 sampler.set_epoch，不能让每张卡重复训练全部数据。验证/测试确保无补齐重复样本进入最终指标，按 sample_id 去重或采用无重复分片后正确汇总。
- 全局 batch 按并行策略分配，sampler 与 shuffle 不同时启用；下方单进程参考不能未经适配直接用于 DDP。
- 缓存是模块内独立开关；只有用户选择且计算昂贵、结果可安全复用时才实现。
- cache key 包含数据版本/指纹、预处理参数，以及特征缓存使用的模型权重和层；不复用不兼容缓存。
- 缓存写入输出目录，不修改原始数据；随机增强后的结果不能当作固定预处理缓存复用。
- 非法比例、空数据、缺失类别或无效缓存明确报错，不静默换 split 或忽略 unknown。

## 配置片段

与主配置合并；合并评估时使用 `eval_ratio`，校验实际使用的比例为非负且和为 1。

```yaml
split:
  mode: standard
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  save_manifest: true
dataloader:
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
  drop_last: true  # 仅训练；验证/测试固定 false
cache:
  enabled: false
  directory: outputs/cache
```

## 种子参考

目标项目可放在 `utils/reproducibility.py`；文件名是目标项目示例，本 Skill 无此 Python 附件。
设置必须在初始化模型和 DataLoader 之前执行。
运行时修改 PYTHONHASHSEED 不会改变当前进程已初始化的哈希种子；不要靠它弥补未排序索引。
不启用严格确定性算法检查；deterministic 只控制 cuDNN 设置，固定 seed 不等于全部算子逐位确定。
完整 resume 还需要验证模块的状态恢复。

```python
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed 必须是 [0, 2**32) 范围内的整数。")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)
```

## Dataset / DataLoader 参考

以下示例只适用于每个 NPY 存储一幅二维数组的分类任务；不是通用数据协议。
按上一段建立目标项目的种子工具后，再采用此片段。目标模块需正常包导入。
示例注释仅为说明接口；项目实际注释密度由已确认的注释档位决定。

```python
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

from utils.reproducibility import build_generator, seed_worker


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
```

## 阶段检查与下一步

检查一次索引扫描、split 无越界重叠、实际 batch 的 dtype/shape/标签、worker=0 分支、缓存开关及失效条件。
记录划分协议和数据接口，再按主入口进入模型/训练阶段；不提前加载展示或注释模块。
