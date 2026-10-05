---
name: pytorch-model-training
description: Use when 已选择 PyTorch 科研项目的模型与训练细则，并进入核心网络、loss、训练器或评估器的实现阶段。
---

# 模型与训练细则

输入：模型/论文设计、数据 batch 契约、训练策略、设备与 YAML。
输出：透明模型实现、轻量入口、训练/评估循环及可验证的数值接口。

## 实现边界

- 默认手写 BasicBlock/Bottleneck、ResNet 主体、Patch Embedding、Attention、Transformer Block、MLP、ViT、DropPath 和项目创新模块；只实现任务用到的部分。
- PyTorch 基础层和张量运算可直接使用。未经用户指定，不用 `torchvision.models.resnet*`、`torchvision.models.vit*`、`timm.create_model`、`nn.MultiheadAttention`、`nn.TransformerEncoder*` 隐藏科研核心计算。
- 通用预处理/指标可使用成熟库。用户指定现成模型或预训练权重时优先遵循，说明权重与自定义结构的兼容条件。
- 构造参数从 YAML 传入；校验通道、patch、head 数、类别数等，不把配置静默修正成“能跑”的值。
- 模型只定义计算，不读取数据、不创建 logger、不解析 CLI。统一输出 logits/feature 的明确接口。
- 数据、训练、评估和模型职责分离；避免没有需求的注册框架或过深继承。
- 注释由 [代码学习与注释模块](code-annotation.md) 统一管理；本链接不要求现在读取，按主入口的注释时机加载。

## 训练与评估

- train 使用 `model.train()`，验证/测试使用 `model.eval()` 与无梯度上下文。
- 每步清梯度，完成 forward、loss、backward、optimizer step；scheduler 按明确的 batch/epoch 语义更新。
- AMP、梯度裁剪、优化器、调度器等通过 YAML 控制；启用 AMP 时裁剪前先 unscale。
- 聚合指标按样本数加权，避免简单平均不同大小 batch；评估默认保留最后不足一批的样本。
- 分类模型返回未 softmax 的 logits 给 CrossEntropyLoss；其他任务明确 loss 与输出约定。
- 显示阶段进度；如果选择展示模块，**在开始记录 epoch 指标前**按主入口读取它，不能训练完才发现未留 metrics。
- 配置 checkpoint/resume 时，在实现保存恢复接口前读取已选的验证模块。完整训练只在用户要求且资源允许时执行，不把 smoke test 当正式实验。
- `scripts/train.py` 只读/校验配置、设置 seed、构建数据与模型、调用 trainer 并报告产物路径；`test.py` 同理。

## 自动多 GPU / 单 GPU / CPU（基础能力）

这项要求来自主入口，即使本细则模块未启用也要满足；本节提供实现参考。

- 默认使用 CUDA 可见设备，多卡自动并行、单卡直接执行、没有 CUDA 时 CPU。用户可用 device.type/parallel 覆盖，不需默认手填卡数。
- 自动模式优先 DDP + NCCL；所需后端不可用（例如某些 Windows 构建）时采用 DataParallel。根据运行环境选择，不能依赖 NCCL 一定存在。
- batch 定义为全局值。batch 不能整除卡数时自动采用 DataParallel 保持语义；batch 小于可见卡数时只启用可分配到样本的卡，并说明原因，不暗改 batch。
- 入口先检测执行计划；需要 DDP 且还未被启动器启动时，内部通过当前解释器的 torch.distributed.run 启动同一入口，传入原 --config。使用 LOCAL_RANK/RANK/WORLD_SIZE 环境变量，在 __main__ 保护中启动，避免递归和 Windows 多进程导入副作用。
- 参数解析兼容启动器传入的 --local-rank/--local_rank，并与 LOCAL_RANK 环境值核对；这是内部系统参数，不增加用户业务参数。不得用忽略全部未知参数的方式兼容，以免吞掉配置拼写错误。
- 每个 DDP worker 设置本地 CUDA 设备、初始化进程组，再创建模型并用 DistributedDataParallel 包装；同一次初始化用同一 seed，数据/增强 RNG 按 rank 派生。退出时清理进程组。
- DDP 数据显式分片，每 epoch 更新 sampler；loss/指标按样本数归约，预测按 sample_id 正确合并且不重复；不把 padding 的重复样本算进测试结果。
- 只有 rank 0 创建共享 run、写模型/日志/manifest/汇总，其他 rank 通过广播得到相同路径。多卡启动器完整退出后，编排器才开始下一阶段。
- DataParallel 在主 CUDA 上构建模型，再按选定 device_ids 包装；代码不要依赖 .module 总是存在。权重保存/加载统一使用未包装模型接口，保持单卡、多卡、CPU 兼容。
- 设备发现失败、OOM 和分布式训练异常分别报告；不能用 catch-all 把训练失败当成 CPU 自动回退。
- 使用 DDP 的训练与分片评估保持 collective 调用顺序一致。无梯度评估需避免不同 shard 长度与 DDP forward 的 buffer 同步导致等待；可用同步完成的未包装模型执行评估，再统一汇总。

依据 [PyTorch DDP 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) 与 [DataParallel 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) 选择后端与分片方案。

### 执行计划参考

下面只返回执行计划，**不独自实现多卡训练**。调用方还必须完成上述启动、包装、分片与汇总。
将 YAML 的 batch_size、device.type、device.parallel 传入；默认无需用户改启动命令。

```python
import torch
import torch.distributed as distributed


def select_execution(
    global_batch_size: int,
    device_type: str = "auto",
    parallel: str = "auto",
) -> dict:
    if type(global_batch_size) is not int or global_batch_size <= 0:
        raise ValueError("training.batch_size 必须是正整数。")
    if device_type not in {"auto", "cpu", "cuda"}:
        raise ValueError("device.type 必须是 auto/cpu/cuda。")
    if parallel not in {"auto", "single", "ddp", "data_parallel"}:
        raise ValueError("不支持的 device.parallel。")
    visible_count = (
        torch.cuda.device_count() if torch.cuda.is_available() else 0
    )
    if device_type == "cuda" and visible_count == 0:
        raise RuntimeError("显式要求 CUDA，但 CUDA 不可用。")
    if device_type == "cpu" or visible_count == 0:
        if parallel not in {"auto", "single"}:
            raise ValueError("当前设备不能使用 CUDA 多卡策略。")
        return {
            "device": "cpu", "parallel": "single", "gpu_ids": [],
            "world_size": 1, "local_batch_size": global_batch_size,
        }

    count = min(visible_count, global_batch_size)
    nccl_ready = (
        distributed.is_available() and distributed.is_nccl_available()
    )
    if parallel == "single" or count == 1:
        mode, count = "single", 1
        if parallel in {"ddp", "data_parallel"}:
            raise ValueError("显式多卡策略需要至少两张可分配到样本的 GPU。")
    elif parallel == "auto":
        mode = (
            "ddp"
            if nccl_ready and global_batch_size % count == 0
            else "data_parallel"
        )
    else:
        mode = parallel
    if mode == "ddp" and (
        not nccl_ready or global_batch_size % count != 0
    ):
        raise ValueError("DDP 需要 NCCL，且全局 batch 能整除所选 GPU 数。")
    return {
        "device": "cuda:0", "parallel": mode,
        "gpu_ids": list(range(count)),
        "world_size": count if mode == "ddp" else 1,
        "local_batch_size": (
            global_batch_size // count if mode == "ddp"
            else global_batch_size
        ),
    }
```

内部启动器可采用如下命令数组；entry/config_path 由入口解析为可靠路径，
nproc 来自计划。用户仍执行 `python scripts/train.py --config ...`。
torchrun 子进程读 LOCAL_RANK 环境变量，兼容透传 rank 参数，不要求额外用户业务参数。
已有外部启动器环境时先校验一致性，不再次启动。

```python
import argparse
import subprocess
import sys
from pathlib import Path


def parse_entry_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--local-rank", "--local_rank", type=int, default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def launch_distributed(
    entry: Path,
    config_path: Path,
    nproc: int,
    project_root: Path,
) -> None:
    subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", f"--nproc-per-node={nproc}",
            str(entry), "--config", str(config_path),
        ],
        cwd=project_root,
        check=True,
    )
```

## 配置片段

将以下 `training` 键合并到主配置；不是再建一个竞争的配置源。

```yaml
model:
  name: vit_tiny
  in_channels: 1
  num_classes: 10
  image_size: 224
  patch_size: 16
  embed_dim: 192
  num_heads: 3
  num_layers: 12
  mlp_ratio: 4.0
  drop_rate: 0.0
  attn_drop_rate: 0.0
  drop_path_rate: 0.1
training:
  amp: false
  grad_clip: 5.0
  optimizer:
    type: adamw
    lr: 0.001
    weight_decay: 0.0001
  scheduler:
    type: cosine
    eta_min: 0.000001
```

## 手写计算参考

下面不是完整 ViT，不要求项目使用这两个类。教学注释展示张量流；生成项目按用户档位调整。
`H` 在 Attention 中表示 head 数，在图像输入中表示图像高度，作用域不同。

```python
"""
模型代码风格参考示例。

示例目的：
1. 展示科研核心模块手写方式；
2. 展示中文 docstring；
3. 展示参数换行风格；
4. 展示 forward 中的 Tensor Shape 注释。

注意：该文件只是代码风格示例，不代表具体项目必须使用这个模型。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    手写多头自注意力模块。

    设计说明：
    1. 显式构造 Q、K、V，避免使用 nn.MultiheadAttention 隐藏核心计算；
    2. 将 embedding 维度拆分为多个 attention head；
    3. 使用 scaled dot-product attention 计算 token 间关系。

    输入:
        x: (B, N, D)

    输出:
        x: (B, N, D)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if num_heads <= 0 or embed_dim <= 0 or embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim 必须能被 num_heads 整除。"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行多头自注意力计算。"""
        batch_size, num_tokens, embed_dim = x.shape

        qkv = self.qkv(x)  # (B, N, D) -> (B, N, 3D)
        qkv = qkv.reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )  # (B, N, 3D) -> (B, N, 3, H, Dh)

        qkv = qkv.permute(2, 0, 3, 1, 4)
        # (B, N, 3, H, Dh) -> (3, B, H, N, Dh)

        q, k, v = qkv.unbind(dim=0)
        # 每个张量: (B, H, N, Dh)

        attn = q @ k.transpose(-2, -1)
        # (B, H, N, Dh) @ (B, H, Dh, N) -> (B, H, N, N)

        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        # (B, H, N, N) @ (B, H, N, Dh) -> (B, H, N, Dh)

        x = x.transpose(1, 2)
        # (B, H, N, Dh) -> (B, N, H, Dh)

        x = x.reshape(batch_size, num_tokens, embed_dim)
        # (B, N, H, Dh) -> (B, N, D)

        x = self.proj(x)  # (B, N, D) -> (B, N, D)
        x = self.proj_drop(x)

        return x


class PatchEmbedding(nn.Module):
    """
    图像 Patch Embedding。

    将二维图像划分为不重叠 patch，并映射到 token embedding。
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()

        if patch_size <= 0 or image_size <= 0 or image_size % patch_size != 0:
            raise ValueError(
                "image_size 必须能被 patch_size 整除。"
            )

        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将图像转换为 token 序列。"""
        x = self.proj(x)
        # (B, C, H, W) -> (B, D, H/P, W/P)

        x = x.flatten(2)
        # (B, D, H/P, W/P) -> (B, D, N)

        x = x.transpose(1, 2)
        # (B, D, N) -> (B, N, D)

        return x
```

## 阶段检查与下一步

核验输入输出 shape、有限 loss、一次 backward、train/eval 行为和参数来自配置。
记录核心接口；已选实验展示则进入其阶段，之后按选择补注释。
