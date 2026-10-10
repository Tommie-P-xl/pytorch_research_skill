---
name: pytorch-experiment-reporting
description: 已选择实验记录与展示模块，需要保存 PyTorch 日志、指标、运行配置，或用真实结果生成图表时读取。
---

# 实验记录与数据展示

开始前要知道：记录哪些指标、训练/评估怎样返回结果、需要哪些图和格式、结果保存在哪。
完成后应得到：本次实验独立的目录、实际运行配置的副本、日志、可供程序读取的指标，以及用户需要的图表。

## 接入时机与展示范围

- 开始写每轮训练和评估的记录代码前读取本模块，先把真实指标保存下来，后面再用它们画图。
- 耗时步骤显示当前正在读配置、扫描数据、训练、提取特征、选阈值、测试或保存文件；长循环用 tqdm 或已有进度工具。
- 终端关键信息也写入文件。训练默认 `logs/training.log`，测试 `logs/testing.log`；合并评估可用 `evaluation.log`。
- 每轮保存 CSV、JSON 或 JSONL（一行一条 JSON 记录），包含轮数、数据集合、样本数、loss、任务指标和学习率。遇到 NaN/Inf 明确记录异常或报错，不当作正常数值。
- 指标含义固定：不能同一字段一会儿是百分比、一会儿是 0～1 比例；未计算字段不填 0 冒充测量。
- 分类按需展示 loss、accuracy、confusion matrix；Open Set/OOD 按协议选择 ROC、PR、AUROC、AUPR-In/Out、FPR@TPR95、OSCR 或阈值图，不机械全选。
- 图表格式按用户选择，例如 SVG；如果未指定且影响交付，在本阶段询问。坐标、单位、类别与图例完整。
- 优先读取已存的指标或逐样本预测来画图，不为重新画图再跑一次模型。不把演示数据说成真实实验结果。
- 没有数据或权重时，提供画图代码，说明需要哪些输入字段，并注明尚未生成真实实验图。
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

记录 Python、PyTorch、系统、seed、时间及 Git commit；CUDA/cuDNN 信息按任务需要记录。某项辅助环境信息取不到就提示，不让训练因此失败，也不要求在终端展示 GPU 详情。
`config.yaml` 保留原始参数，`resolved_config.yaml` 记录合并参数和解析路径后实际使用的配置；不能借此偷偷改 batch 或其他实验设置。
批量程序指定了 `experiment.run_dir` 时就用该目录，不另外新建随机目录。多进程仅由 rank 0 写共享指标和产物清单。
选择批量模块时，仍需保存它要求的最小文件清单和状态，即使用户不需要高级图表。

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

## 保存哪些字段，才能重新画图

启用对应保存功能时，要写清楚下面这些字段代表什么。已有项目可以沿用原字段名，不再保存一套重复格式：

- 每轮记录：`epoch`、`split`、`num_samples`、`loss` 和任务指标；`lr` 只在训练时记录。说明指标单位，以及越大还是越小越好。
- 每个样本的预测：固定的样本标识 `sample_id`、数据集合 `split`、真实标签、预测标签；画 ROC 等曲线时还需要对应概率或分数。sample_id 不使用机器专属绝对路径。
- 类别信息：类别名对应哪个数字、概率每一列对应哪个类别。Open Set/OOD 还要写清已知/未知判断、分数增大表示什么、阈值及其选取数据。
- 图表按这些字段和单位读取。缺列、类别顺序不一致时明确报错，不靠猜。

下面是**格式演示数据，不是真实实验结果**；数字要替换为实际测量值：

```json
{"epoch": 1, "split": "val", "num_samples": 100, "loss": 0.72, "accuracy": 0.81}
```

这里约定 `accuracy` 是 0～1 的比例，越大越好；`0.81` 表示 81%。画百分比坐标时统一乘 100，保存时不要有的轮次写 `0.81`、有的写 `81`。

单个预测记录的格式可以是：

```json
{
  "sample_id": "class_a/sample_001.npy",
  "split": "test",
  "true_label": 0,
  "predicted_label": 0,
  "probabilities": [0.8, 0.2]
}
```

同时保存类别顺序，例如 `["class_a", "class_b"]`，才能知道概率列对应什么类别。

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

## 把指标写入文件

只演示追加记录，不代替具体任务的指标计算与绘图。JSONL 每行一条记录，可在训练时逐轮追加。

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

检查日志和指标文件确实写出，指标单位一致，样本数正确，图表来自真实记录。确认保存开关和文件格式生效，没有覆盖旧实验。
记录结果路径；下一步按用户选择读取注释模块，不在此时额外增加整套代码教学。
