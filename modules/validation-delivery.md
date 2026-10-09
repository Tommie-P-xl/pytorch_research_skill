---
name: pytorch-validation-delivery
description: 已选择检查与交付模块，需要实现模型保存、继续训练、少量 batch 的运行检查，或检查修改是否影响其他文件时读取。
---

# 检查点、验证与交付

写模型保存代码前读取本模块，最后交付时再按它检查。
**选择本模块不等于开启所有功能**；保存模型（checkpoint）、继续训练（resume）和少量数据的运行检查（smoke test）分别按用户选择开启。

## 保存什么，才能怎样恢复

| 保存内容 | 可以做什么 | 不能据此声称什么 |
|---|---|---|
| 仅 `model.state_dict()` | 加载权重做测试，或用这些权重开始新的训练 | 不能说完整接续了上次训练 |
| 模型 + 优化器等训练状态 | 按保存的进度继续训练 | 不能仅凭权重和 epoch 保证完全重现后续随机过程 |

比如，AdamW 的优化器还保存历史梯度统计。只加载模型而新建优化器，会丢失这些统计，即使写上相同的 epoch，也不是完整续训。

- YAML 分别控制是否保存、保存最佳模型/最后模型/定期模型、保存间隔和续训路径。关闭时不创建对应训练状态文件。
- 显式 resume 文件缺失、格式不兼容或结构不匹配要报错，不偷偷随机初始化。
- 完整续训保存模型、优化器、学习率调度器、AMP 梯度缩放器、已完成轮数/步数、最佳指标、实际配置，以及 Python/NumPy/Torch/CUDA 的随机数生成状态（RNG）。
- 用到 DataLoader generator 或 sampler 时，还要保存并恢复它们的状态。在一轮中间续训还涉及读到了第几个样本、子进程预取的数据和随机增强，不能只保存 RNG 就保证完全重现。
- 默认建议从一轮结束处恢复，并写清支持范围。持续运行的数据子进程（persistent workers）的随机状态，不能直接从主进程拿到；要严格重现需要额外实现。
- 加载 RNG 用 CPU ByteTensor；不要将 RNG 状态随模型统一映射到 CUDA 后直接传给 CPU 的 set_rng_state。
- 只加载模型权重属于初始化或推理，不叫完整续训。保存“最佳模型”时写清用哪个验证集合、指标越大还是越小越好，不能用最终测试集选最佳模型。
- 载入文件使用适合其结构的安全加载策略；不为兼容任意来源文件默认关闭受限加载。
- 保存未包装模型的 `state_dict`，避免 DDP/DataParallel 增加的 `module.` 前缀影响单卡/CPU 加载。权重应能在不同 GPU 数量下使用；完整续训可能依赖原来的进程数、各进程随机状态和设备分配方式，要单独说明，不能因此限制普通测试加载。
- 批量 train→test 至少需要可供测试的模型权重，但不一定需要完整 resume checkpoint。若用户关闭所有模型保存，则不能宣称跨进程自动测试可用；在流程规划时指出依赖或采用明确的同进程方案，不能擅自重新打开开关。
- 需要批量实验时，成功保存的权重和后续要用的文件，按 [批量模块](batch-experiments.md) 登记到文件清单中。未选批量模块时不用读取它的全部正文。

### 只保存测试需要的模型权重

下面是调用片段，假设模型结构已创建，`weight_path` 已按项目根目录解析。只使用自己生成或可信来源的文件，不要求为了测试保存完整训练状态。

```python
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel


# 多卡包装会增加一层 .module，保存真正模型的权重，方便换设备加载。
raw_model = (
    model.module
    if isinstance(model, (DataParallel, DistributedDataParallel))
    else model
)
weight_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(raw_model.state_dict(), weight_path)

# 加载到同结构的未包装模型；不因为原来是多卡就要求测试也用相同卡数。
state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
raw_model.load_state_dict(state_dict)
raw_model.to(device).eval()
```

DDP 只由 rank 0 执行保存；加载和各进程同步由调用方按实际策略处理。这个片段不恢复优化器、epoch 或 RNG，不能作为完整 resume 的实现。

## 配置片段

```yaml
checkpoint:
  enabled: true
  save_best: true
  save_last: true
  save_periodic: false
  interval: 10
  resume: null
smoke_test:
  enabled: true
  num_batches: 1
```

## 状态打包参考

这两个函数只收集和恢复随机状态；完整续训还要保存前面列出的其他内容。
NumPy 状态转换成列表、整数等基础类型，方便使用受限的权重加载方式，不需要反序列化任意 NumPy 对象。

```python
import random

import numpy as np
import torch


def capture_rng_state(
    generator: torch.Generator | None = None,
) -> dict:
    algorithm, keys, position, has_gauss, cached_gaussian = (
        np.random.get_state()
    )
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": algorithm,
            "keys": keys.tolist(),
            "position": position,
            "has_gauss": has_gauss,
            "cached_gaussian": cached_gaussian,
        },
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "loader_generator": (
            generator.get_state() if generator is not None else None
        ),
    }


def restore_rng_state(
    state: dict,
    generator: torch.Generator | None = None,
) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["algorithm"],
        np.asarray(numpy_state["keys"], dtype=np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG 恢复要求 CUDA 可用。")
        if len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA 设备数量与保存状态不一致。")
        torch.cuda.set_rng_state_all(
            [item.cpu() for item in state["cuda"]]
        )
    if state["loader_generator"] is not None:
        if generator is None:
            raise ValueError("保存了 DataLoader generator，但未提供恢复目标。")
        generator.set_state(state["loader_generator"].cpu())
```

写入 `rng_state = capture_rng_state(generator)` 到 checkpoint。
加载可参考 `torch.load(path, map_location="cpu", weights_only=True)`，配置和附加信息用基础类型保存。
先创建并恢复模型、优化器等，**最后**恢复 RNG；否则模型初始化会消耗刚恢复的随机数，让后续随机过程发生变化。
调用方还要恢复 scheduler/scaler、epoch 和最佳指标，检查当前配置与保存配置的差异；本片段不是完整续训代码。

## 启动前校验与 Smoke Test

- 检查配置是否为字典、必需字段是否存在、类型是否正确，再检查路径、数据是否为空、划分比例、模型维度、设备，以及权重和缓存是否匹配。
- 按实际模型检查，例如 CNN 不必提供 ViT 的 patch 参数。seed 不能是 `true/false`；头数和 patch 大小先检查正数，再计算除法。
- 开启 smoke test 时，只取 1～2 个 batch，检查数据读取、前向计算、损失、反向计算和梯度是否为有限数。它用于尽早发现接线问题，不用于判断模型效果。
- 使用单独的临时模型和实验，避免检查过程中改变了正式模型的 BatchNorm 统计、梯度或随机状态，接着又拿它开始正式训练。
- 使用现有测试或最小验证；不为注释改动强制完整训练、下载数据或修改用户环境。

下面示例只适用于单标签分类；使用临时模型，调用方传入正确 device、dataloader，
并将 YAML 的 `smoke_test.num_batches` 传给 `num_batches`，不另加 CLI 参数。

```python
import torch
from torch import nn


def run_smoke_test(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_batches: int = 1,
) -> None:
    if type(num_batches) is not int or num_batches <= 0:
        raise ValueError("smoke_test.num_batches 必须是正整数。")
    model.to(device).train()
    batches = iter(dataloader)
    for batch_index in range(num_batches):
        try:
            inputs, labels = next(batches)
        except StopIteration as error:
            raise RuntimeError("可用 batch 少于 smoke_test.num_batches。") from error
        model.zero_grad(set_to_none=True)
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        if logits.ndim != 2 or logits.size(0) != labels.size(0):
            raise RuntimeError("分类 logits 应为 (B, K)，且 B 与标签一致。")
        loss = nn.CrossEntropyLoss()(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Smoke test loss 非有限值。")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients or not all(
            torch.isfinite(gradient).all().item()
            for gradient in gradients
        ):
            raise RuntimeError("缺少有效梯度或梯度含非有限值。")
        print(f"Smoke batch {batch_index + 1}/{num_batches} 通过。")
    model.zero_grad(set_to_none=True)
    print("Smoke test 通过：读取、forward、loss、backward。")
```

## 修改影响与交付

逐项判断是否需要同步，而不是不分任务全量修改：

| 改动 | 同步检查 |
|---|---|
| 新超参数/模型 | YAML、校验、构造接口、README、checkpoint、smoke |
| 新数据流程 | 数据划分、batch 内容、类别编号、缓存标识、训练/评估 |
| 新指标/图表 | 日志字段、预测数据的格式、输出文件名、README |
| 新依赖 | requirements、安装说明、可选依赖开关 |
| 仅学习注释 | 解释是否准确、语法能否解析、计算是否被误改；不重建配置和目录 |

交付说明启用/跳过模块、改动文件、运行命令、已运行检查与结果、未执行内容及原因。
没有实际运行就标“未运行”，无真实数据不宣称实验成功。主入口的基本校验始终适用。
