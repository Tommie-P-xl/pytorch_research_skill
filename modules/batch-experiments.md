---
name: pytorch-batch-experiments
description: Use when 用户需要预先配置多组 YAML 参数后批量训练、测试或特征提取，并自动传递本次模型、NPZ 等产物路径，减少重复手改配置。
---

# 批量实验与流程编排

输入：基础配置、实验参数组、阶段链、产物到下游配置字段的映射。
输出：一个批量入口、每组/每阶段独立配置、真实运行状态、自动产物绑定与汇总。
Skill 仅包含本 Markdown；下文代码指导 Agent 在目标项目实现编排功能，不是随 Skill 分发的脚本。

## 1. 先确定流程与依赖

- 用户明确要求批量运行即视为已选本模块；只询问缺失的参数范围、必要阶段与失败策略，不再问是否需要批量。
- 常见链为 train → test；需要 NPZ/特征/统计文件时为 train → extract_features/fit_statistics → test，按实际项目增加必要阶段。
- 每个阶段只接受 --config，并返回正确退出码。批量入口同样只接受 --config，例如：
  `python scripts/run_experiments.py --config configs/experiments.yaml`。
- 先读/复用现有 train/test 接口；不要重写模型。编排层只负责计划、配置、依赖、启动、状态和汇总。
- 批量默认串行执行不同参数组，每组训练可自动使用所有可分配的 GPU，避免多组实验同时争抢同一批卡。实验级并发需用户选择并明确资源隔离。
- 路径字段由用户或项目 schema 明确映射。自动传递指定产物，不靠文件名猜“测试需要哪个 NPZ”。
- 跨进程测试需要训练权重产物；只保存推理权重即可，不强制完整 resume checkpoint。所有权重保存关闭且又要求分进程测试时，规划阶段说明依赖，不在后台打开被关闭功能。
- 即使不启用高级实验展示，也实现批量必需的状态、配置快照和最小产物清单；不因此强制启用其余模块。

## 2. 参数清单与独立配置

- 保留基础 YAML 不变。每组 deepcopy 后应用点路径 override，按 schema 校验；未知字段、错误类型或非法组合在启动训练前报错。
- 支持显式 experiments 列表；也可实现 sweep 网格的笛卡尔积。两种方式在配置中择一，确定展开顺序并先展示总实验数，避免把列表错误配对。
- 同一个字段或其父/子路径同时被 override 与产物 binding 设置时报冲突；不要静默决定谁覆盖谁。
- 固定数据划分/类别映射用于跨参数比较；多 seed 重复实验以明确参数组表达。
- 每组 experiment_id、每次执行 attempt_id、每阶段 stage_id 唯一；保存基础配置摘要、计划摘要和最终阶段配置。
- 编排器分配各阶段 experiment.run_dir，单次入口使用它，不另建无法定位的随机目录。该字段属于已声明的运行 schema；用户平常不需要手填。
- 各阶段最终 YAML 在所有 overrides 和已解析 bindings 应用后写出；训练仍只读自己的配置文件，不依赖后来修改 base.yaml。
- 固定输入测试数据直接在基础配置/override 声明；上游生成模型、NPZ、阈值等由 binding 注入。不要把固定外部数据错误当成本轮输出。

## 3. YAML 示例

仅示范 train → 特征提取 → test；没有中间 NPZ 需求时删除 extract 阶段和对应 binding。
键名按项目实际 schema 调整并在规划时确认，示例不要求 every project 实现同名提取器。
base_config、entry、output_root 相对于项目根目录，不相对于临时阶段配置文件。

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
      training.optimizer.lr: 0.001
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
      evaluation.checkpoint: train:checkpoint.best
      evaluation.features: extract:features.test
    required_artifacts: [metrics.test]
```

基础配置须声明对应路径字段（初始可为 null）、模型保存策略与 experiment.run_dir。
例如选择 checkpoint.best 就要求该次训练实际保存 best，选优依据合法验证协议；
若项目只输出 last，显式把 binding 改为 checkpoint.last，不偷偷替换。
阶段 requires/bindings 构成 DAG：校验无环、生产者先执行、名称唯一、引用存在。
无需特征提取时，test 直接依赖 train，仍自动注入模型路径。

## 4. 产物清单：下游只接本次成功产物

每阶段在指定 run_dir 写 artifacts.json。成功清单只在产物写入、验证完成后原子发布，
失败/未完成不能留下可被当成成功的清单。单个阶段 stdout 不是产物定位依据。

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

- path 相对于**该 manifest 所在阶段目录**，不是项目根目录；输出产物不得通过 ../ 逃到别的实验。
- 下游校验 batch/experiment/attempt/stage/config 摘要与计划一致、status 成功、artifact 存在、文件实际存在且格式可用。
- 训练进程退出码 0 但缺少所需 manifest/产物也视为失败，不继续依赖阶段。
- NPZ 额外记录/验证数组键名、shape、dtype、样本 ID、split、类别顺序；特征/统计产物记录模型与预处理来源，避免模型更新后复用旧特征。
- 下游 inputs 登记所用生产者 manifest/产物与内容指纹，追溯来源；计划/配置摘要不能单独证明文件内容未变。
- 不扫描全 outputs 找 newest/best，也不猜“最近一次训练”。按显式逻辑名绑定本组本次的产物。
- 不存在的路径、重复/歧义产物、缺键或来源不匹配时报错；不挑另一个文件凑合测试。
- 单独使用旧产物时要求用户显式指定来源，不伪装成当前实验训练结果。

## 5. 执行、失败与续跑

1. 展开实验清单、校验阶段 DAG/schema/保存依赖，生成独立 batch 目录和计划；dry_run 只报告参数差异/阶段顺序/待绑定逻辑名，不启动训练。
2. 对每组按依赖顺序执行：基础配置克隆 → 参数覆盖 → 校验上游成功产物 → 路径绑定 → 保存最终 YAML → 启动阶段。
3. 通过当前 Python 解释器与命令数组调用入口，cwd 为项目根；不要 shell=True 或拼接带参数的 shell 字符串。阶段的 CUDA/多卡自动选择沿用基础规则。
4. 运行中显示当前实验序号、参数、阶段、进度和日志路径；子进程输出应实时可见且保存，不能等数小时结束才一次吐日志。
5. 每阶段记录 pending/running/success/failed/skipped、退出码、时间、配置与 manifest 路径。running 不算成功，失败上游的依赖标 skipped。
6. on_error=stop 在首个失败停止；continue_other_experiments 跳过该组剩余依赖、继续独立参数组。最后有失败则批量入口返回非零并输出汇总。
7. DDP 所有进程完成后再宣布阶段成功；rank 0 负责共享清单和模型输出，编排器不能在一个 worker 结束时就启动测试。
8. 续跑仅跳过状态成功、计划/配置/输入指纹匹配且产物验证通过的步骤；依赖内容改变则重跑消费者。不把上次 running/failed 当成可复用成功。
9. 重试失败阶段分配新的 attempt/stage 目录，不覆盖前次产物。需要跨 attempt 复用成功前序时显式记载来源与验证结果，重新绑定，不伪造 attempt 身份。
10. 汇总参数、状态、主要测试指标、耗时、输出目录与失败原因；跨 seed 的统计仅使用实际完成且协议可比的实验。

默认不自动无限重试，不自动降精度、减 batch 或换 CPU 来“完成”失败实验。

## 6. 配置与绑定核心参考

以下不是完整调度器：Agent 还需按上述流程实现入口、阶段状态机、DAG 校验与汇总。
示例支持已有字段的点路径赋值、无共享修改的配置克隆、受范围约束的 manifest 解析；
适合在目标项目的 utils/experiment_pipeline.py 中适配。
生产者身份由编排器根据实际计划提供，不从未知 manifest 反推期望身份。

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
        raise ValueError("编排器必须提供完整生产者身份。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("不支持的 artifact manifest 版本。")
    if manifest.get("status") != "success":
        raise RuntimeError("上游阶段未成功，不能绑定其产物。")
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise ValueError(f"上游产物身份不匹配: {key}")
    record = manifest.get("artifacts", {}).get(artifact_name)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise KeyError(f"上游缺少产物: {artifact_name}")
    relative = Path(record["path"])
    if relative.is_absolute() or relative.drive or relative.root:
        raise ValueError("产物路径必须相对于 manifest 目录。")
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
        raise ValueError("同一字段不能同时 override 和自动绑定。")
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

生成配置后还要调用项目 schema 校验并保存 YAML。产物的 hash/格式/NPZ 元数据与 inputs 验证
由任务适配层补齐，不能把上面的存在性/身份检查说成已完成内容验证。
状态/manifest 建议先写同目录临时文件再原子替换，只有所有 required_artifacts 验证通过才能标 success。

## 7. 交付与轻量验证

- 不需跑整批训练来证明编排接线：临时假阶段可生成小权重占位/NPZ/指标文件，
  检查参数展开、配置隔离、路径绑定、顺序、失败跳过与续跑逻辑。假产物只能用于流程测试，不能当真实训练结果。
- 覆盖路径有空格、工作目录改变、单/多 GPU 启动边界、训练非零退出、
  “退出 0 但无产物”、错 experiment 身份、越界路径、override/binding 冲突。
- 检查 batch 配置和 base.yaml 均未被写回；最终测试配置模型/NPZ 来自本组已成功生产者。
- 交付可直接填写的批量 YAML 示例、一个运行命令、实际阶段字段映射、日志/汇总路径和验证结果。
- 实际完整训练/测试按用户授权范围执行；仅实现自动化功能时不擅自启动数小时实验。
