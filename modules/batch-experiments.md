---
name: pytorch-batch-experiments
description: 预设多组 YAML 参数，依次运行训练、测试或特征提取，并自动把本次模型和 NPZ 等文件路径传给后续步骤时读取。
---

# 批量实验与流程编排

这个模块解决反复“改学习率 → 训练 → 找模型文件 → 填测试路径 → 测试”的问题：提前列出参数，由程序逐组完成。

开始前要知道：基础配置、每组要改的参数、步骤顺序，以及上一步生成的哪个文件要填进下一步的哪个配置字段。
完成后应得到：一个启动入口、每组每步独立的配置、成功或失败记录、自动填写的文件路径和结果汇总。
下文代码用于在目标项目实现这些功能，不是本 Skill 自带的可执行脚本。

## 1. 先确定流程与依赖

- 用户明确要求批量运行即视为已选本模块；只询问缺失的参数范围、必要阶段与失败策略，不再问是否需要批量。
- 常见链为 train → test；需要 NPZ/特征/统计文件时为 train → extract_features/fit_statistics → test，按实际项目增加必要阶段。
- 每个步骤通过 `--config` 读取参数，成功返回退出码 0，失败返回非零。批量入口也只接受 `--config`，例如：
  `python scripts/run_experiments.py --config configs/experiments.yaml`。
- 复用已有 train/test 脚本，不重写模型。批量程序只负责生成配置、决定步骤顺序、启动脚本、记录状态和汇总结果。
- 默认一组做完再做下一组，每组仍可用多张 GPU。要同时跑多组，先由用户选择，并说明各组使用哪些设备，避免争抢同一批卡。
- 明确文件路径填进哪个配置字段。遵守项目已有的字段和类型约定（schema），不靠文件名猜“测试需要哪个 NPZ”。
- 跨进程测试需要训练权重产物；只保存推理权重即可，不强制完整 resume checkpoint。所有权重保存关闭且又要求分进程测试时，规划阶段说明依赖，不在后台打开被关闭功能。
- 即使不要高级图表，也要保存批量运行所需的状态、每步实际配置和最小文件清单，不因此强制开启其他模块。

## 2. 参数清单与独立配置

- 不改基础 YAML。每组先深拷贝配置（deepcopy），再覆盖这一组的参数（override）。例如 `training.optimizer.lr` 就是逐层找到 `training → optimizer → lr`。训练前检查未知字段、错误类型和非法组合。
- 可以用 `experiments` 逐组列参数，也可以用 `sweep` 网格列出所有组合，两种方式选一种。例如 2 个学习率 × 3 个 seed 是 6 组，不是把两个列表一一配对；先展示总组数和执行顺序。
- 同一个字段不能既由手填参数覆盖，又由上一步文件自动填写（binding）。父子字段冲突也报错，例如同时改 `evaluation` 和 `evaluation.checkpoint`，不偷偷决定谁优先。
- 比较参数时固定数据划分和类别编号。要比较多个 seed，就在参数组里明确列出它们。
- 每组有 `experiment_id`，每次执行/重试有 `attempt_id`，每步有 `stage_id`，组合起来唯一标识一次执行。保存基础配置和计划的内容摘要（hash），以及每步实际配置，供续跑时核对。
- 批量程序给每步分配 `experiment.run_dir`，脚本直接用它，不另建随机目录。这个字段要先在项目配置中声明，用户平时不用手填。
- 参数覆盖和文件路径自动填写都完成后，再保存最终 YAML。训练只读自己的这份配置，不依赖之后可能被修改的基础文件。
- 固定测试数据写进基础配置或 overrides；本次生成的模型、NPZ、阈值等通过 binding 传入。固定外部数据不是本次运行生成的文件。

## 3. YAML 示例

仅示范 train → 特征提取 → test；没有中间 NPZ 需求时删除 extract 阶段和对应 binding。
字段名按项目实际配置调整，不要求每个项目都实现特征提取脚本。
`base_config`、`entry`、`output_root` 从项目根目录算起，不从临时配置所在目录算起。

```yaml
batch:
  base_config: configs/config.yaml
  output_root: outputs/batches
  on_error: stop           # stop / continue_other_experiments
  max_parallel_experiments: 1
  reuse_completed: true
  dry_run: false

experiments:
  - name: lr_1e3
    overrides:
      training.optimizer.lr: 0.001  # 覆盖基础配置中的学习率。
      model.drop_rate: 0.0
  - name: lr_3e4_dropout
    overrides:
      training.optimizer.lr: 0.0003
      model.drop_rate: 0.1

stages:
  - name: train
    entry: scripts/train.py
    requires: []
    bindings: {}
    required_artifacts: [checkpoint.best]

  - name: extract
    entry: scripts/extract_features.py
    requires: [train]
    bindings:
      feature_extraction.checkpoint: train:checkpoint.best
    required_artifacts: [features.test]

  - name: test
    entry: scripts/test.py
    requires: [train, extract]
    bindings:
      evaluation.checkpoint: train:checkpoint.best  # 使用本组训练保存的最佳模型。
      evaluation.features: extract:features.test   # 使用本组提取的测试特征。
    required_artifacts: [metrics.test]
```

基础配置要先声明这些路径字段（初始可写 `null`）、模型保存方式和 `experiment.run_dir`。
使用 `checkpoint.best` 就要求本次训练确实按验证结果保存最佳模型；如果只保存最后模型，明确改为 `checkpoint.last`，不能偷偷替换。
`requires` 表示当前步骤必须等哪些步骤成功；`bindings` 表示取哪个文件、填到哪个字段。步骤关系不能形成循环（DAG，即有向无环图），步骤名唯一，引用必须存在，生成文件的步骤必须先执行。
无需特征提取时，test 直接依赖 train，仍自动填入模型路径。

读懂一个 binding：

| 写法 | 含义 |
|---|---|
| `evaluation.checkpoint` | 下一步配置中，要填入模型路径的字段 |
| `train:checkpoint.best` | 从 train 步骤的文件清单中，取名为 checkpoint.best 的文件 |
| `requires: [train]` | train 成功且需要的文件检查通过后，才能开始当前步骤 |

假设本组训练实际输出 `outputs/batches/demo/lr_1e3/train/checkpoints/best.pt`，自动生成的测试配置会包含：

```yaml
evaluation:
  checkpoint: outputs/batches/demo/lr_1e3/train/checkpoints/best.pt
```

这个路径只是说明结果格式；实际值必须从本组成功训练的清单里读取，不能照抄示例或全局查找最新文件。

## 4. 产物清单：下游只接本次成功产物

“产物”指模型权重、特征、指标等输出文件。每步在指定 `run_dir` 写 `artifacts.json` 清单，说明本步生成了哪些文件。
先保存并检查全部必需文件，再把临时清单一次性替换为正式清单（原子发布），避免下一步读到半写入内容。失败或没完成时，不能留下标记成功的清单。不能只从终端打印里猜输出路径。

```json
{
  "schema_version": 1,
  "batch_id": "batch_a",
  "experiment_id": "lr_1e3",
  "attempt_id": "attempt_001",
  "stage_id": "train",
  "config_sha256": "sha256-of-final-stage-config",
  "status": "success",
  "artifacts": {
    "checkpoint.best": {
      "path": "checkpoints/best.pt",
      "kind": "model_weights"
    }
  },
  "inputs": {}
}
```

- 清单里的 `path` 从**清单所在步骤目录**算起，不从项目根目录算起。输出文件不能用 `../` 跳到其他实验目录。
- 下一步先核对批次、参数组、执行次数、步骤和配置摘要是否符合计划，再检查状态成功、文件名存在、文件真实存在且能按所需格式读取。
- 进程退出码是 0，但缺少必需清单或文件，也算失败，不能继续运行依赖它的步骤。
- NPZ 还要记录并检查数组名、shape、dtype、样本 ID、数据集合和类别顺序。特征和统计文件要说明用哪个模型和预处理生成，不能换了模型还用旧特征。
- `inputs` 记录本步用了谁的清单、哪些文件和内容摘要，以便查来源。只比较配置 hash 不够：配置相同，文件内容仍可能被改过。
- 按明确名字读取本组本次的文件，不扫描整个 outputs 查找 newest/best，不猜“最近一次训练”。
- 路径不存在、同名文件不明确、字段缺失或来源不匹配时就报错，不另挑文件凑合。
- 要单独使用旧模型或旧特征时，要求用户明确指定来源，不说成当前训练结果。

## 5. 执行、失败与续跑

1. 列出全部参数组，检查配置、步骤有无循环和需要的模型是否会保存，再生成独立批次目录和计划。`dry_run` 只展示参数差异、步骤顺序和将读取的文件名，不训练。
2. 每组依次执行：复制基础配置 → 覆盖参数 → 检查上一步的成功文件 → 填写路径 → 保存最终 YAML → 启动脚本。
3. 用当前 Python 解释器和命令数组启动脚本，工作目录设为项目根。不使用 `shell=True` 或拼接 shell 命令字符串；设备选择沿用主入口规则。
4. 运行中显示当前实验序号、参数、阶段、进度和日志路径；子进程输出应实时可见且保存，不能等数小时结束才一次吐日志。
5. 每步记录状态：pending（等待）、running（运行中）、success（成功）、failed（失败）、skipped（跳过），以及退出码、时间、配置和清单路径。运行中不算成功；上一步失败后，依赖它的步骤标为跳过。
6. on_error=stop 在首个失败停止；continue_other_experiments 跳过该组剩余依赖、继续独立参数组。最后有失败则批量入口返回非零并输出汇总。
7. DDP 所有进程完成后才宣布当前步骤成功；rank 0 负责共享清单和模型输出，批量程序不能看到一个子进程结束就开始测试。
8. 续跑只跳过上次已成功、计划/配置/输入内容摘要相同、文件仍检查通过的步骤。输入文件变了就重跑使用它的步骤；上次运行中或失败的步骤不能当作已完成。
9. 重试失败步骤时创建新的执行目录，不覆盖前次文件。如果复用上次成功的前序步骤，要记录真实来源和检查结果，重新填入路径，不改清单里的执行身份冒充本次输出。
10. 汇总参数、状态、主要测试指标、耗时、输出目录与失败原因；跨 seed 的统计仅使用实际完成且协议可比的实验。

默认不自动无限重试，不自动降精度、减 batch 或换 CPU 来“完成”失败实验。

## 6. 配置与绑定核心参考

下面三个函数分别负责：按点分字段名改参数、从清单找到并检查文件、生成当前步骤的新配置。可放在目标项目的 `utils/experiment_pipeline.py`。
它们不是完整批量程序，还需实现启动入口、状态记录、步骤依赖检查和汇总。生成文件的步骤应有什么身份，由批量程序按计划提供，不能照着待检查清单抄一份再说检查通过。

```python
import copy
import json
import os
from pathlib import Path


def set_config_value(config: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    if not dotted_key or any(not part for part in parts):
        raise ValueError("配置点路径不能为空。")
    node = config
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"未知配置字段: {dotted_key}")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise KeyError(f"未知配置字段: {dotted_key}")
    node[parts[-1]] = copy.deepcopy(value)


def resolve_artifact(
    manifest_path: Path,
    artifact_name: str,
    expected_identity: dict,
) -> Path:
    required_identity = {
        "batch_id", "experiment_id", "attempt_id",
        "stage_id", "config_sha256",
    }
    if set(expected_identity) != required_identity:
        raise ValueError("批量程序必须提供生成文件的步骤的完整身份。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("不支持的产物清单版本。")
    if manifest.get("status") != "success":
        raise RuntimeError("上一步未成功，不能自动填入它的文件路径。")
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise ValueError(f"上游产物身份不匹配: {key}")
    record = manifest.get("artifacts", {}).get(artifact_name)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise KeyError(f"上游缺少产物: {artifact_name}")
    relative = Path(record["path"])
    if relative.is_absolute() or relative.drive or relative.root:
        raise ValueError("产物路径必须相对于清单所在目录。")
    run_dir = manifest_path.parent.resolve()
    path = (run_dir / relative).resolve()
    if not path.is_relative_to(run_dir) or not path.is_file():
        raise FileNotFoundError("产物越界或文件不存在。")
    return path


def materialize_config(
    base: dict,
    overrides: dict,
    bindings: dict,
    producers: dict,
    project_root: Path,
) -> dict:
    if any(
        override == binding
        or override.startswith(binding + ".")
        or binding.startswith(override + ".")
        for override in overrides
        for binding in bindings
    ):
        raise ValueError("同一字段不能既手动覆盖，又自动填入文件路径。")
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        set_config_value(result, key, value)
    for key, reference in bindings.items():
        stage, separator, artifact = reference.partition(":")
        if not separator or not stage or not artifact:
            raise ValueError(f"非法产物引用: {reference}")
        producer = producers[stage]
        path = resolve_artifact(
            producer["manifest"], artifact, producer["identity"]
        )
        # 生成配置的路径仍按项目根目录解析，不能变成按临时配置目录解析。
        try:
            relative_path = Path(os.path.relpath(path, project_root)).as_posix()
        except ValueError:
            # 用户指定的跨盘本地产物无法表示成相对路径。
            relative_path = path.as_posix()
        set_config_value(result, key, relative_path)
    return result
```

调用例子：先看参数如何覆盖，暂时没有文件路径需要自动填写，因此 bindings 和 producers 都为空。

```python
from pathlib import Path


base = {"training": {"optimizer": {"lr": 0.001}}}
run_config = materialize_config(
    base=base,
    overrides={"training.optimizer.lr": 0.0003},
    bindings={},
    producers={},
    project_root=Path.cwd(),  # 本演示从项目根运行；实际入口应解析固定项目根。
)
print(base["training"]["optimizer"]["lr"])        # 0.001，基础配置没有被改。
print(run_config["training"]["optimizer"]["lr"])  # 0.0003，本组配置使用新值。
```

生成配置后，还要按项目字段和类型约定检查，再保存 YAML。上述函数只检查文件存在和来源身份，文件内容 hash、格式、NPZ 信息和 inputs 要另外实现检查。
状态和清单先写到同目录临时文件，再替换正式文件；所有 `required_artifacts` 检查通过后才能标记 success。

## 7. 交付与轻量验证

- 不需跑整批训练来证明编排接线：临时假阶段可生成小权重占位/NPZ/指标文件，
  检查参数展开、配置隔离、路径绑定、顺序、失败跳过与续跑逻辑。假产物只能用于流程测试，不能当真实训练结果。
- 覆盖路径有空格、工作目录改变、单/多 GPU 启动边界、训练非零退出、
  “退出 0 但无产物”、错 experiment 身份、越界路径、override/binding 冲突。
- 检查 batch 配置和 base.yaml 均未被写回；最终测试配置模型/NPZ 来自本组已成功生产者。
- 交付可直接填写的批量 YAML 示例、一个运行命令、实际阶段字段映射、日志/汇总路径和验证结果。
- 实际完整训练/测试按用户授权范围执行；仅实现自动化功能时不擅自启动数小时实验。
