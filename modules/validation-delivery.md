---
name: pytorch-validation-delivery
description: Use when 已选择 PyTorch 检查点与交付细则，并需要实现可选 checkpoint/resume、轻量 smoke test 或核验代码变更影响。
---

# 检查点、验证与交付

本模块可在核心阶段的 checkpoint 接口实现前读取，在最后交付时再次应用。
**启用本模块不等于启用 checkpoint**；checkpoint、完整 resume 和 smoke test 分别遵守用户选择。

## Checkpoint 与恢复语义

- YAML 分别控制 enabled、save_best、save_last、save_periodic、interval 和 resume。关闭时不强行创建训练状态文件。
- 显式 resume 文件缺失、格式不兼容或结构不匹配要报错，不偷偷随机初始化。
- 完整恢复保存 model、optimizer、scheduler、AMP scaler、已完成 epoch/step、best metric、实际 config、Python/NumPy/Torch/CUDA RNG。
- 使用 DataLoader generator/sampler 时保存并恢复其状态。中途续训还涉及 sampler 游标、worker/prefetch 和随机增强，不能仅凭一个 RNG 快照承诺严格复现。
- 推荐明确支持 epoch 边界恢复；persistent workers 的内部随机状态无法从主进程快照直接恢复，严格重现需要额外设计。
- 加载 RNG 用 CPU ByteTensor；不要将 RNG 状态随模型统一映射到 CUDA 后直接传给 CPU 的 set_rng_state。
- 只加载 model weights 是权重初始化/推理，不称为完整 resume。save_best 的选优 split 与方向要明确，不用最终 test 选优。
- 载入文件使用适合其结构的安全加载策略；不为兼容任意来源文件默认关闭受限加载。
- 保存未包装模型的 state_dict，使 DDP/DataParallel 的 module. 前缀不影响单卡/CPU 加载；跨 GPU 数的权重加载必须可用。完整 resume 的逐 rank RNG 状态与拓扑限制单独说明，不能把其限制应用到普通测试加载。
- 批量 train→test 至少需要可供测试的模型权重，但不一定需要完整 resume checkpoint。若用户关闭所有模型保存，则不能宣称跨进程自动测试可用；在流程规划时指出依赖或采用明确的同进程方案，不能擅自重新打开开关。
- 成功保存的权重与其他下游产物按 [批量模块](batch-experiments.md) 的 manifest 契约登记；这里只定义接口，未选择批量模块时不要求读取其全部正文。

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

这两个函数只负责构造/恢复 RNG；完整 checkpoint 还需按上述清单保存组件。
NumPy 状态转换为基础类型，避免要求加载任意 NumPy pickle 对象。

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
加载的参考接口为 `torch.load(path, map_location="cpu", weights_only=True)`；
配置/metadata 保持基础类型。先构造并恢复 model/optimizer 等组件，**最后**恢复 RNG，
防止模型重新初始化消耗已恢复随机流。调用方还要恢复 scheduler/scaler、epoch、best metric，
并明确配置差异处理；本片段不独自构成完整恢复训练代码。

## 启动前校验与 Smoke Test

- 检查配置顶层 mapping、必需字段与类型、路径、非空数据、split 比例/模式、模型维度、device、checkpoint 和 cache 兼容性。
- 配置校验依据实际模型，不让 CNN 必须包含 ViT 字段。seed 的 bool 不视作 int；检查 head/patch 正值后再做除法。
- 启用 smoke test 时运行 1～2 batch 的读取、forward、loss、backward 与有限梯度检查，按实际任务选择 loss。
- 在独立模型/临时实验中执行，避免改变正式模型的 BatchNorm、梯度或 RNG 后直接启动训练。
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
| 新数据流程 | split、batch 接口、类别映射、cache key、训练/评估 |
| 新指标/图表 | 日志字段、预测契约、产物名称、README |
| 新依赖 | requirements、安装说明、可选依赖开关 |
| 仅学习注释 | 注释真实性、语法、可执行语义；不重建配置和目录 |

交付说明启用/跳过模块、改动文件、运行命令、已运行检查与结果、未执行内容及原因。
没有实际运行就标“未运行”，无真实数据不宣称实验成功。主入口的基本校验始终适用。
