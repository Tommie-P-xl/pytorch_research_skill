---
name: pytorch-experiment-reporting
description: Use when 已选择 PyTorch 实验记录与数据展示，并需要接入持久化日志、指标、实验快照或根据真实产物绘图。
---

# 实验记录与数据展示

输入：任务指标、训练/评估接口、图表选择与格式、输出根目录。
输出：独立实验目录、实际配置快照、日志、结构化指标，以及用户需要的图表。

## 接入时机与展示范围

- 在 epoch/评估记录接口实现前读取本模块，先确保原始指标可保存；最终图表在核心实现后生成。
- 耗时阶段显示配置读取、扫描、训练、提取、标定、测试、保存等状态；长循环用 tqdm 或已有进度工具。
- 终端关键信息也写入文件。训练默认 `logs/training.log`，测试 `logs/testing.log`；合并评估可用 `evaluation.log`。
- 每 epoch 保存 CSV/JSON，字段可含 epoch、split、样本数、loss、任务指标和 lr；NaN/Inf 明确记录或报错。
- 指标含义固定：不能同一字段一会儿是百分比、一会儿是 0～1 比例；未计算字段不填 0 冒充测量。
- 分类按需展示 loss、accuracy、confusion matrix；Open Set/OOD 按协议选择 ROC、PR、AUROC、AUPR-In/Out、FPR@TPR95、OSCR 或阈值图，不机械全选。
- 图表格式按用户选择，例如 SVG；如果未指定且影响交付，在本阶段询问。坐标、单位、类别与图例完整。
- 优先从已存 metrics/predictions 重建图表，无需重复推理；不把模拟数据包装为真实结果。
- 没有数据/权重时交付生成代码和输入契约，注明真实图表尚未生成。
- TensorBoard 默认关闭；明确选择后才加入依赖、SummaryWriter 与配置。
- 用户只要结果展示时不自动增加 checkpoint、缓存或详细注释。

## 实验目录与快照

每次正式实验独立目录，不覆盖先前结果；配置关闭的能力不创建对应产物。

```text
outputs/runs/<timestamp>_<name>_<unique_id>/
├── config.yaml             # 原始配置副本
├── resolved_config.yaml    # 实际生效配置
├── environment.json        # 环境记录
├── splits.json             # 启用划分归档时
├── logs/
├── metrics/
├── figures/                # 启用图表时
└── checkpoints/            # 启用 checkpoint 时
```

记录 Python、PyTorch、系统、seed、时间及 Git commit；CUDA/cuDNN 等按任务需要记录，环境辅助信息获取失败给 warning，不影响训练。
配置快照区分原始配置与实际配置；不得借“resolved”静默改变用户实验语义。环境记录不要求终端展示 GPU 信息。
批量编排指定 experiment.run_dir 时使用该目录，不再另建随机 run，避免下游找错产物。
多进程只有 rank 0 写共享指标和产物清单；选择批量模块后须提供其最小 artifacts.json/status 接口，即使未选择高级图表展示也不能缺失该接口。

## 配置片段

```yaml
logging:
  console: true
  save_file: true
  save_metrics: true
  save_figures: true
  figure_format: svg
  tensorboard: false
evaluation:
  save_predictions: true
  save_confusion_matrix: true
```

各保存开关必须影响实际代码路径；console=false 时仍应有用户可见的阶段进度，避免完全沉默。

## 可重建结果的输入契约

启用相应保存能力时，明确以下最小语义；可适配已有字段名，不另建重复格式：

- epoch 记录：epoch、split、num_samples、loss、任务指标；lr 只在训练时记录。说明指标单位与越大/越小越好的方向。
- 逐样本预测：稳定 sample_id、split、true_label、predicted_label；绘制曲线时还需相应概率/分数。sample_id 不保存机器专属绝对路径。
- 类别元数据：类别名称与索引映射、概率列的类别顺序；Open Set/OOD 额外说明 known/unknown 判定、分数方向、阈值及来源 split。
- 图表读取已确认的字段与单位；缺少必需列或出现不一致类别顺序时清晰报错，不猜测列含义。

## 目录与日志参考

以下辅助函数可放到目标项目 `utils/`；路径传入前按项目根目录解析。
调用方依据 YAML 只传需要的子目录；logger 使用 run 内唯一名称。

```python
import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def create_run_directory(
    output_root: Path,
    experiment_name: str,
    subdirectories: tuple[str, ...] = ("logs", "metrics"),
) -> Path:
    if (
        not experiment_name
        or experiment_name in {".", ".."}
        or "/" in experiment_name
        or "\\" in experiment_name
    ):
        raise ValueError("实验名必须是单一目录名。")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / (
        f"{timestamp}_{experiment_name}_{uuid4().hex[:8]}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in subdirectories:
        if name not in {"logs", "metrics", "figures", "checkpoints"}:
            raise ValueError(f"不支持的实验子目录: {name}")
        (run_dir / name).mkdir()
    return run_dir


def build_logger(
    name: str,
    log_path: Path | None,
    console: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    handlers = []
    if console:
        handlers.append(logging.StreamHandler())
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(log_path, mode="a", encoding="utf-8")
        )
    if not handlers:
        handlers.append(logging.NullHandler())
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
```

## 指标落盘参考

只演示追加记录，不代替任务特定的指标计算与绘图。

```python
import json
from pathlib import Path


def append_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 禁止把非有限数写入 JSON，避免绘图时悄悄跳过坏数据。
    record = json.dumps(metrics, ensure_ascii=False, allow_nan=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(record + "\n")
```

## 阶段检查与下一步

检查日志/指标存在、单位一致、样本总数正确、图表来自真实记录、开关与格式生效、旧实验未被覆盖。
记录输出路径；下一阶段按用户选择读取注释模块，不将本阶段扩展成整套源码教学。
