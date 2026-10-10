---
name: pytorch-data-pipeline
description: 已选择数据流程模块，准备实现 PyTorch 的样本读取、训练验证测试划分、DataLoader 或可选数据缓存时读取。
---

# 数据流程与效率

在用户选择本模块、或直接要求处理数据流程时读取。即使跳过本模块，也要统一随机种子，避免把测试信息用于训练和调参。

## 输入与阶段输出

开始前要知道：数据文件怎么存、标签代表什么、怎么划分数据、使用哪个 seed、数据有多大、是否缓存。
完成后应得到：样本列表、每个样本属于哪部分数据的记录，以及能返回一批数据的 Dataset/DataLoader。只有用户选择缓存才实现它。

## 数据与划分

- Dataset 初始化时扫描并排序文件，只做一次。`__getitem__` 根据索引读取当前样本（按需读取，lazy loading），不要每取一个样本就重新扫描目录。
- 样本清单保存相对路径，方便换机器使用。随机裁剪等训练增强只用于训练；验证和测试按固定的评估方式处理。
- 长期比较实验时保存划分清单（split manifest）和类别编号。只固定 seed 不够：增加文件后，即使用同一 seed，也可能分出不同的训练集。
- 随机划分用统一 seed。同一患者、同一对象、相邻视频帧等相关样本，按任务要求分组或按时间隔离，不能只检查文件名不同。用户要求时间划分时，不改成随机划分。
- 检查训练、验证、测试有没有共用样本。均值、标准差等数据统计只在实验允许的训练数据上计算，再用于其他集合。
- `standard` 表示 train/val/test 三部分；`merged_eval` 表示 train/eval 两部分。后者的 eval 同时承担验证和测试用途，要说明它不是独立最终测试。
- Open Set/OOD 要说明已知与未知类别、哪些数据允许训练、在哪部分数据上选阈值；不能默认未知类别也参与训练。

例如，标准化时先用训练数据计算统计量，而不是把三部分数据合在一起算：

```python
# 假设 train_x、val_x、test_x 是 (样本数, 特征数) 的 NumPy 浮点数组。
mean = train_x.mean(axis=0)
std = train_x.std(axis=0).clip(min=1e-6)  # 防止常量特征导致除零。
train_x = (train_x - mean) / std
val_x = (val_x - mean) / std
test_x = (test_x - mean) / std
```

## DataLoader 与缓存

- worker 是负责读数据的子进程，generator 用于控制随机采样；都从全局 seed 设置随机状态。Windows 下启动多进程的代码放在 `if __name__ == "__main__":` 中，worker 回调函数定义在模块顶层。
- `num_workers=0` 表示在主进程读数据，此时不传 `prefetch_factor`，也不开 `persistent_workers`。验证/测试不打乱顺序（shuffle），不丢弃最后不足一批的样本（drop_last）。
- 根据设备选择 `pin_memory` 与 `non_blocking`，不要为了加速把放不下的大数据一次性读进内存。
- DDP 用 `DistributedSampler` 等方式给不同 GPU 分配数据，每轮调用 `sampler.set_epoch(epoch)` 更新随机顺序，不能让每张卡都重复训练全量数据。评估时，采样器可能为均分补齐重复样本，要按 sample_id 去重，或使用不重复的分配方式，再合并指标。
- 每张卡的 batch 大小遵守主入口的总 batch 规则。设置 sampler 后不要同时设置 `shuffle=True`；下面的单进程示例不能直接用于 DDP。
- 缓存单独开关控制；只有用户选择，而且重复计算昂贵、结果能复用时才实现。
- 缓存标识（cache key）记录数据版本或内容摘要、预处理参数；缓存模型特征时还要记录权重和取特征的层。任一项改变，就不能继续用旧缓存。
- 缓存写入输出目录，不改原始数据。随机增强每次可能不同，不能把增强后的某一次结果固定缓存，冒充普通预处理。
- 比例不合法、数据为空、缺少类别、缓存不匹配时明确报错，不偷偷换划分或忽略未知类别。

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

重点：初始化时只扫描一次并排序；取样时才读取文件；
读数据的子进程使用统一 seed。大型 NPY 可用 mmap 按需访问文件，
但转换成 Tensor 时仍会复制当前样本，不代表整个流程不占内存。
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
    """构建 DataLoader，并统一采样和读数据子进程的随机种子。"""
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

取一批数据，检查数据类型（dtype）、形状（shape）和标签是否符合模型要求；确认各集合没有不允许的样本重叠。再检查只扫描一次、worker=0 能用、缓存开关及更新条件正确。
记录数据如何划分、一个 batch 返回什么，再进入模型/训练阶段；不提前读取展示或注释模块。
