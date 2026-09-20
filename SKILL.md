---
name: pytorch-research-code-style
description: 统一规范代码 Agent 编写、修改和维护 PyTorch 深度学习科研项目。强调可复现、YAML 单一配置源、核心模型手写、中文注释与 Tensor Shape、可移植路径、实验日志、Run Directory、数据效率、数据泄漏防护和修改影响检查。用户当前提示词始终可以覆盖本 Skill。
---

# PyTorch 科研代码统一工程规范

## 1. 适用范围与优先级

本 Skill 用于规范代码 Agent 在 PyTorch 深度学习科研项目中的：

- 新建项目；
- 新增模型、数据集、训练器、评估器；
- 修改既有代码；
- 复现实验；
- 重构项目目录；
- 添加开集识别、分类、特征提取、消融实验等科研功能。

### 最高优先级规则

1. **用户当前提示词优先于本 Skill。**
2. 本 Skill 是默认工程规范，不是不可覆盖的硬约束。
3. 若用户明确要求与本 Skill 不同的实现方式，应听从用户，并仅在必要时简要说明可能影响。
4. 不得因为本 Skill 的默认偏好而擅自改变用户要求的算法、模型、实验设计或输出格式。

---

# 2. 规范强度说明

本文使用以下关键词：

- **MUST**：默认必须遵守；除非用户明确要求覆盖。
- **SHOULD**：强烈推荐；在不适合当前任务时可以省略。
- **MUST NOT**：默认禁止；除非用户明确要求。

---

# 3. 实验可复现性

## 3.1 统一随机种子

项目 **MUST** 提供唯一的全局随机种子，例如：

```yaml
reproducibility:
  seed: 42
  deterministic: true
```

该随机种子 **MUST** 统一作用于：

- Python `random`；
- NumPy；
- PyTorch CPU；
- PyTorch CUDA；
- 多 GPU；
- 数据集划分；
- DataLoader worker；
- 模型参数初始化；
- 随机数据增强；
- 需要随机采样的评估过程。

不得在不同模块中随意写死不同的 seed。

## 3.2 数据划分可复现

数据集划分 **MUST** 使用固定随机种子。

若项目需要长期对比实验，**SHOULD** 将本次数据划分结果保存到 Run Directory，例如：

- `splits.json`
- `splits.csv`
- `split_manifest.json`

这样可以避免代码修改后，即使 seed 相同，也因为文件排序、数据新增等原因导致划分发生变化。

## 3.3 确定性与性能

若 `deterministic: true`：

- SHOULD 启用确定性设置；
- SHOULD 关闭会引入非确定性的 benchmark 行为；
- 日志中 SHOULD 提示确定性设置可能降低训练速度。

若用户更关心速度，可通过 YAML 关闭。

---

# 4. YAML 是实验参数的 Single Source of Truth

## 4.1 CLI 原则

所有训练、测试、评估、特征提取等入口脚本，默认 **只允许一个业务参数**：

```bash
python scripts/train.py --config configs/config.yaml
```

Agent **MUST NOT** 默认增加以下形式的业务参数：

```bash
--epochs
--batch-size
--lr
--resume
--checkpoint
--force-recompute
--num-workers
```

这些实验参数 **MUST** 放入 YAML。

除非用户明确要求，否则不得让 CLI 和 YAML 同时控制同一个实验参数。

## 4.2 可调参数全部进入 YAML

凡是科研人员可能调整的内容，原则上 **MUST** 配置化，包括：

- 数据路径；
- 数据划分比例；
- 随机种子；
- 模型类型；
- 模型结构超参数；
- batch size；
- epoch；
- 优化器；
- 学习率；
- scheduler；
- AMP；
- 梯度裁剪；
- DataLoader；
- checkpoint；
- cache；
- 评估阈值；
- 输出目录；
- 是否保存图像；
- 是否启用某个实验模块。

Agent **MUST NOT** 将实验参数散落硬编码在 Python 文件中。

## 4.3 配置读取后不得偷偷改写实验语义

Agent **MUST NOT** 静默覆盖用户 YAML 中的参数。

例如类别数可以从数据集自动推导，但应：

- 显式打印推导结果；
- 对冲突配置进行报错或明确警告；
- 不要静默修改后继续执行。

---

# 5. 核心模型结构必须透明可读

## 5.1 允许直接使用的基础积木

可以直接使用 PyTorch 基础层和张量运算，例如：

- `nn.Conv1d / nn.Conv2d / nn.Conv3d`
- `nn.Linear`
- `nn.BatchNorm*`
- `nn.LayerNorm`
- `nn.Dropout`
- `nn.MaxPool* / nn.AvgPool* / nn.AdaptiveAvgPool*`
- 激活函数
- `torch.matmul`
- `torch.softmax`
- `torch.cat`
- `reshape / view / permute / transpose / flatten`

这些属于基础积木，不要求重新实现 PyTorch 本身。

## 5.2 默认必须手写的科研核心模块

以下结构 **MUST** 自己实现核心逻辑：

- ResNet BasicBlock；
- ResNet Bottleneck；
- ResNet 主体；
- Patch Embedding；
- Multi-Head Self-Attention；
- Transformer Encoder Block；
- MLP / Feed Forward Block；
- ViT；
- Stochastic Depth / DropPath；
- 项目提出的新型特征融合模块；
- 论文中的关键创新模块。

## 5.3 默认禁止隐藏核心结构的高级封装

除非用户明确要求，**MUST NOT** 使用：

- `torchvision.models.resnet*`
- `torchvision.models.vit*`
- `timm.create_model(...)`
- 其他库中直接返回完整主干网络的高级 API；
- `nn.MultiheadAttention`
- `nn.TransformerEncoder`
- `nn.TransformerEncoderLayer`

原因：科研代码应让研究人员直接看到关键计算过程，方便理解、修改、插入模块和做消融实验。

## 5.4 第三方库的合理使用

数据预处理、通用数学工具、指标计算等可以合理使用成熟库。

禁止的重点是：**不要把需要研究和修改的核心网络结构隐藏在黑盒高级接口里。**

---

# 6. 代码可读性与中文注释规范

## 6.1 语言约定

默认：

- 变量名、函数名、类名：英文；
- docstring：中文；
- 解释性注释：中文；
- 常见数学或深度学习术语可以保留英文，例如 Attention、logits、feature、token、patch。

## 6.2 文件级说明

重要 Python 文件 **SHOULD** 在顶部说明：

- 文件功能；
- 输入；
- 输出；
- 在整体 pipeline 中的位置。

## 6.3 Class / Function docstring

核心类和核心函数 **MUST** 有清晰 docstring，至少说明：

- 功能；
- 参数；
- 返回值；
- 必要时说明输入输出 Tensor Shape；
- 特殊设计或对应论文思想。

## 6.4 注释写“为什么”，而不只是复述代码

避免：

```python
# x 加 1
x = x + 1
```

更应该解释：

- 为什么这样处理；
- 对应哪一步算法；
- 为什么需要 reshape；
- 为什么此处必须 detach；
- 为什么验证集不能参与训练。

## 6.5 行长度与排版

代码 **MUST** 优先可读性，避免一行塞入大量参数或复杂表达式。

建议：

- 单行尽量不超过约 100 个字符；
- 函数参数较多时换行；
- 字典、函数调用、条件表达式过长时换行；
- 使用空行分隔逻辑阶段。

---

# 7. Tensor Shape 注释规范

在以下位置，只要 Shape 可以确定，**MUST** 或 **SHOULD** 添加张量形状说明：

- `forward()`；
- reshape / view；
- permute / transpose；
- flatten；
- pooling；
- token 拼接；
- 多尺度特征融合；
- attention Q/K/V；
- concat；
- 特征提取接口。

推荐格式：

```python
x = torch.cat((x0, x1, x2), dim=2)  # (B, N, D) * 3 -> (B, N, 3D)
```

Shape 注释应表达**语义维度**，优先使用：

- `B`：batch size
- `C`：channel
- `H, W`：空间尺寸
- `N`：token / sequence length
- `D`：embedding dimension
- `K`：class count

---

# 8. 推荐项目结构

默认推荐：

```text
project/
├── configs/
│   └── config.yaml
├── datasets/
│   ├── __init__.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── backbones/
│   ├── blocks/
│   └── model_factory.py
├── trainers/
│   ├── __init__.py
│   └── trainer.py
├── evaluators/
│   ├── __init__.py
│   └── metrics.py
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── reproducibility.py
│   ├── logging.py
│   ├── run_manager.py
│   └── checkpoint.py
├── tools/
│   ├── verify_dataset.py
│   └── smoke_test.py
├── scripts/
│   ├── train.py
│   └── test.py
├── outputs/
├── README.md
└── requirements.txt
```

### 入口脚本职责

`scripts/train.py` / `scripts/test.py` **MUST** 保持轻量。

入口脚本主要负责：

1. 读取 config；
2. 校验 config；
3. 设置 seed；
4. 创建 Run Directory；
5. 构建数据；
6. 构建模型；
7. 调用 trainer / evaluator；
8. 输出最终路径与关键结果。

**MUST NOT** 把大量模型、数据、训练核心逻辑全部堆在入口脚本中。

---

# 9. 数据集、DataLoader 与缓存

## 9.1 数据读取原则

Dataset 初始化阶段 **SHOULD**：

- 只扫描一次目录；
- 建立样本索引；
- 避免每个 `__getitem__` 重复扫描文件系统；
- 对文件名排序，保证可复现。

大规模数据默认优先 lazy loading，而不是盲目一次性全部加载进内存。

## 9.2 DataLoader 性能

Agent **SHOULD** 根据任务提供 YAML 配置：

- `num_workers`
- `pin_memory`
- `persistent_workers`
- `prefetch_factor`
- `drop_last`

GPU 训练时 SHOULD 使用合理的 `non_blocking=True`。

不同平台的 DataLoader 行为可能不同，Windows / Ubuntu 迁移时不得依赖仅某个平台有效的写法。

## 9.3 缓存机制

当以下条件同时满足时，**SHOULD** 设计缓存：

1. 预处理或特征提取计算昂贵；
2. 输入数据和关键配置没有变化；
3. 结果可以安全复用。

例如：

- STFT / scattering 预处理；
- backbone feature；
- 固定模型下的中间特征；
- 大型索引文件。

缓存 **MUST** 能识别关键配置变化，避免读取过期缓存。

缓存功能 **MUST** 可通过 YAML 开关控制。

---

# 10. 数据泄漏防护与数据划分模式

## 10.1 默认模式：train / val / test

若项目区分三者：

- train：模型参数训练；
- val：模型选择、阈值标定、超参数选择；
- test：最终报告。

测试集 **MUST NOT** 反向影响训练、模型选择和超参数调整。

## 10.2 允许 val / test 不区分

用户允许某些项目不区分 val 和 test。

此时可以使用：

- `train / eval`
- 或 `train / val_test`

但 **MUST**：

1. 在 README 和日志中明确说明该集合同时承担验证与最终评估职责；
2. 不得将其描述为“完全独立、无偏的最终测试集”；
3. 不得在代码中虚构一个不存在的独立 test split；
4. 若后续论文需要严格最终测试，应支持重新切分。

## 10.3 Open Set / OOD 特别规则

必须明确：

- known classes；
- unknown classes；
- unknown 是否允许出现在训练阶段；
- threshold 使用哪个 split 标定；
- 最终指标使用哪个 split。

任何可能造成未知类泄漏的操作都必须显式处理。

---

# 11. 终端反馈、日志、指标与图像

## 11.1 运行过程不得长时间沉默

任何耗时阶段 **MUST** 在终端给出状态，例如：

- 加载配置；
- 扫描数据集；
- 构建 DataLoader；
- 构建模型；
- 加载 checkpoint；
- epoch 训练；
- 特征提取；
- 阈值标定；
- 测试；
- 保存结果。

长循环 SHOULD 使用 `tqdm`。

## 11.2 日志文件

训练默认 SHOULD 保存：

```text
training.log
```

测试默认 SHOULD 保存：

```text
testing.log
```

若 val/test 合并，可使用：

```text
evaluation.log
```

重要终端信息 SHOULD 同时写入日志文件。

## 11.3 指标记录

训练结果 SHOULD 记录结构化指标，例如：

- CSV；
- JSON；
- YAML。

不要只依赖终端历史。

## 11.4 图像保存

分类任务通常 SHOULD 包括：

- loss 曲线；
- accuracy 曲线；
- confusion matrix。

Open Set / OOD 任务根据实际需要增加：

- ROC；
- PR；
- AUROC；
- AUPR-In；
- AUPR-Out；
- FPR@TPR95；
- OSCR；
- threshold 相关图。

不得机械生成与任务无关的图。

## 11.5 TensorBoard 默认关闭

除非用户明确要求，Agent 默认 **MUST NOT**：

- 引入 TensorBoard；
- 创建 `SummaryWriter`；
- 增加 TensorBoard 依赖；
- 在项目结构中生成 runs 目录。

若用户要求使用 TensorBoard，再通过 YAML 配置启用。

---

# 12. Run Directory 与实验快照

每次正式训练或完整评估 **SHOULD** 创建独立 Run Directory。

推荐：

```text
outputs/
└── runs/
    └── 20260920_160000_resnet50_baseline/
        ├── config.yaml
        ├── resolved_config.yaml
        ├── environment.json
        ├── splits.json
        ├── logs/
        │   ├── training.log
        │   └── testing.log
        ├── checkpoints/
        ├── metrics/
        │   ├── train_metrics.csv
        │   └── test_metrics.json
        └── figures/
```

### 必须避免

Agent **MUST NOT** 默认把不同实验都覆盖写入：

```text
outputs/best_model.pth
outputs/results.json
```

除非项目本身明确只需要一次性实验。

---

# 13. Checkpoint 规范

Checkpoint 功能 **MUST** 由 YAML 开关控制，例如：

```yaml
checkpoint:
  enabled: true
  save_best: true
  save_last: true
  save_periodic: false
  resume: null
```

若 `checkpoint.enabled: false`，不得强行创建完整训练状态 checkpoint。

若启用“可恢复训练”的 checkpoint，则 **MUST** 尽量保存完整状态：

- model state；
- optimizer state；
- scheduler state；
- AMP scaler state（若使用）；
- epoch / step；
- best metric；
- config；
- 必要的随机状态；
- 关键 metadata。

不得把“只加载模型权重”称为完整 resume。

---

# 14. Device 与跨平台路径

## 14.1 Device 不得硬编码

Device SHOULD 通过 YAML 指定：

```yaml
device:
  type: auto
```

推荐支持：

- `auto`
- `cpu`
- `cuda`
- 必要时显式 GPU index

如果用户明确指定 CUDA，但 CUDA 不可用，**MUST** 报错或明确提示；不得静默切换 CPU。

`auto` 模式才可以自动选择。

## 14.2 路径优先使用 pathlib

代码 SHOULD 使用 `pathlib.Path`。

配置文件中优先使用相对路径，例如：

```yaml
dataset:
  root: data/uav_rf
```

**MUST NOT** 默认硬编码：

```text
C:\Users\xxx\...
/home/xxx/...
```

目标是让项目可以在 Windows 与 Ubuntu 之间迁移。

## 14.3 路径解析规则必须统一

项目必须统一约定相对路径相对于：

- 项目根目录；
- 或配置文件所在目录。

只能选择一种规则，并在 README 中写清楚。

默认推荐：**相对项目根目录解析。**

---

# 15. Fail Fast 与禁止静默降级

程序 SHOULD 在正式训练开始前完成配置和数据合法性检查。

常见检查包括：

- 路径是否存在；
- 数据集是否为空；
- 类别目录是否合法；
- unknown class 是否存在；
- train / val / test 比例是否合法；
- `patch_size` 是否能正确处理输入；
- `embed_dim % num_heads == 0`；
- batch size 是否大于 0；
- checkpoint 是否存在；
- cache 是否与当前配置匹配。

### 禁止静默降级

Agent **MUST NOT** 默认做以下行为：

- CUDA 不可用时偷偷改用 CPU；
- checkpoint 找不到时偷偷从头训练；
- unknown class 写错时自动忽略；
- 配置项非法时偷偷替换为默认值；
- cache 不兼容时仍继续使用；
- 依赖缺失时悄悄关闭重要功能。

应明确报错，或在确实安全时给出清晰 warning。

---

# 16. 接口、命名与类型提示

为保证项目长期维护：

- 类名 SHOULD 使用 `PascalCase`；
- 函数和变量 SHOULD 使用 `snake_case`；
- 常量 SHOULD 使用 `UPPER_SNAKE_CASE`；
- 公共函数 SHOULD 添加类型提示；
- 同类模块 SHOULD 保持一致的参数命名；
- train / eval 的 batch 数据结构 SHOULD 尽量统一。

不要在不同文件中交替使用：

```text
label / target / y / gt
```

除非语义确实不同。

---

# 17. 轻量级 Smoke Test

项目 SHOULD 提供一个快速检查入口，例如：

```bash
python tools/smoke_test.py --config configs/config.yaml
```

Smoke test 不要求完整训练，目标是快速发现：

- 配置错误；
- Dataset 读取错误；
- DataLoader 错误；
- 输入 Shape 错误；
- 模型 forward 错误；
- loss 计算错误；
- 单步 backward 错误；
- device 不兼容。

建议只运行 1~2 个 batch。

---

# 18. 实验环境元信息

为了复现，Run Directory SHOULD 保存：

```text
environment.json
```

建议包含：

- Python 版本；
- PyTorch 版本；
- CUDA 版本；
- cuDNN 版本（可获取时）；
- GPU 名称；
- 操作系统；
- 随机种子；
- Git commit hash（若项目位于 Git 仓库）；
- 时间戳。

环境记录失败不应影响训练本身，但 SHOULD 产生 warning。

---

# 19. 修改代码后的影响范围检查

Agent 在完成明显功能修改后 **MUST** 检查以下内容是否需要同步：

```text
实现代码
  ↓
config.yaml
  ↓
README.md
  ↓
requirements.txt
  ↓
train / test / evaluator
  ↓
tools / smoke test
  ↓
输出文件说明
```

例如新增一个模型参数后，需要检查：

- YAML 是否加入该参数；
- 默认值是否合理；
- config validator 是否支持；
- README 是否说明；
- checkpoint 是否受影响；
- 测试脚本是否受影响。

Agent 完成修改时 SHOULD 简要说明同步修改了哪些文件。

---

# 20. Agent 执行工作流

当用户要求新建或修改 PyTorch 科研项目时，Agent 默认按以下顺序工作：

1. 阅读当前项目结构与 README；
2. 找到 config、数据、模型、训练、评估入口；
3. 判断用户要求会影响哪些模块；
4. 优先修改 YAML schema / config；
5. 编写或修改核心模块；
6. 保持模型结构透明；
7. 补充中文 docstring 与 Shape 注释；
8. 检查路径、device、seed；
9. 检查数据泄漏；
10. 检查日志、metrics、figures；
11. 检查 checkpoint 与 cache 开关；
12. 运行或提供 smoke test；
13. 检查 README / requirements / config 是否需要同步；
14. 最后给出变更摘要和运行命令。

默认运行命令应尽量保持为：

```bash
python scripts/train.py --config configs/config.yaml
python scripts/test.py --config configs/config.yaml
```

---

# 21. 最终交付检查表

Agent 完成代码任务前，应自检：

- [ ] 是否只有 `--config` 作为默认业务 CLI 参数？
- [ ] 可调实验参数是否已经进入 YAML？
- [ ] 是否设置统一随机种子？
- [ ] 数据划分是否可复现？
- [ ] 是否避免数据泄漏？
- [ ] 核心模型是否没有使用被禁止的黑盒高级实现？
- [ ] 核心类与函数是否有中文 docstring？
- [ ] forward 等关键位置是否有 Tensor Shape 注释？
- [ ] 是否避免超长代码行？
- [ ] 数据加载是否避免重复扫描和明显低效操作？
- [ ] 是否需要缓存？缓存是否可关闭？
- [ ] 终端是否能看到关键运行状态？
- [ ] 是否保存必要日志、指标与图像？
- [ ] TensorBoard 是否仅在用户明确要求时加入？
- [ ] 正式实验是否使用独立 Run Directory？
- [ ] checkpoint 是否可通过 YAML 开关控制？
- [ ] device 是否没有硬编码？
- [ ] 路径是否跨 Windows / Ubuntu 可迁移？
- [ ] 是否 Fail Fast？
- [ ] 是否避免静默降级？
- [ ] 是否提供必要的 smoke test？
- [ ] 是否记录环境元信息？
- [ ] 修改后是否同步检查 config / README / requirements 等文件？
- [ ] 用户当前提示词是否得到最高优先级？

---

# 22. 参考附件

本 Skill 的 `references/` 目录包含参考实现，用于约束 Agent 的代码风格，而不是要求每个项目机械复制：

- `config.example.yaml`：统一 YAML 组织方式；
- `reproducibility.py`：统一随机种子；
- `config_utils.py`：配置读取与 Fail Fast；
- `run_manager.py`：Run Directory 和实验环境记录；
- `logging_utils.py`：终端 + 文件日志；
- `checkpoint.py`：可关闭的完整 checkpoint；
- `model_style_example.py`：手写模型、中文注释和 Shape 风格示例；
- `dataset_style_example.py`：高效 Dataset / DataLoader 风格；
- `smoke_test_example.py`：快速前向与反向检查；
- `project_structure.md`：推荐目录职责；
- `impact_checklist.md`：代码修改后的同步检查表。

参考附件只用于提供统一风格。若用户指定不同结构或实现，以用户要求为准。
