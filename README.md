# PyTorch Research Code Style Skill

> 面向代码 Agent 的 PyTorch 深度学习科研项目统一工程规范。

本项目提供一套用于约束和指导代码 Agent 编写、修改与维护 **PyTorch 深度学习科研代码** 的 Skill。  
目标不是搭建一个固定的训练框架，而是让 Agent 在不同科研项目中持续生成 **可复现、可阅读、可维护、可迁移、适合实验迭代** 的代码。

这套规范主要来自实际 PyTorch 科研项目开发过程中反复遇到的问题，并结合以下项目的工程实践进行整理：

- [OpenMax-PyTorch](https://github.com/Tommie-P-xl/OpenMax-PyTorch)
- [MulScatteringViT](https://github.com/Tommie-P-xl/MulScatteringViT)
- [ViM-Pytorch](https://github.com/Tommie-P-xl/ViM-Pytorch)

适用于图像分类、RF 信号识别、开集识别（Open Set Recognition）、OOD Detection、Transformer、CNN、特征提取、消融实验等常见 PyTorch 科研任务。

---

## 为什么需要这个 Skill？

代码 Agent 可以很快生成能够运行的深度学习代码，但“能运行”并不意味着“适合科研”。

实际使用中经常会出现：

- 随机种子只设置了一部分，实验无法真正复现；
- 大量参数散落在 Python 文件和命令行参数中；
- `train.py`、`test.py` 越写越大，项目难以维护；
- 直接调用 `torchvision`、`timm` 等完整模型，核心网络结构被隐藏；
- Transformer、Attention 等关键计算过程无法直观看到；
- 修改模型后忘记同步更新 `config.yaml`、README 或测试逻辑；
- 训练过程缺少日志、进度条、指标文件和可视化结果；
- 数据划分不严谨，甚至出现训练集、验证集或测试集数据泄漏；
- 每次实验直接覆盖 `best_model.pth`，后续很难追踪不同实验；
- Windows 上能运行的绝对路径，迁移到 Ubuntu 后全部失效；
- Agent 遇到配置异常时自动“兜底”，导致实验已经改变但用户并不知情；
- 为了方便而过度抽象，使科研人员反而难以理解模型内部逻辑。

本 Skill 希望解决这些问题。

它强调的不是复杂的“企业级架构”，而是：

> **让科研代码足够透明，让实验足够可靠，让项目能够长期迭代。**

---

## 核心设计原则

### 1. 实验必须可复现

所有随机行为使用统一的全局随机种子管理，包括：

- Python `random`
- NumPy
- PyTorch CPU
- PyTorch CUDA
- 多 GPU
- 数据集划分
- DataLoader Worker
- 模型初始化
- 随机数据增强
- 评估阶段的随机采样

对于长期对比实验，推荐保存数据划分 manifest，避免数据文件变化后仅依赖随机种子仍产生不同划分。

---

### 2. YAML 是实验配置的 Single Source of Truth

默认情况下，训练、测试和评估脚本只保留：

```bash
python scripts/train.py --config configs/config.yaml
```

除非用户明确要求，否则不应继续增加：

```bash
--epochs
--batch-size
--lr
--resume
--checkpoint
--num-workers
--force-recompute
```

等实验参数。

模型、数据集、训练策略、优化器、调度器、缓存、Checkpoint、评估阈值、输出路径等可调整内容，应统一放入 YAML。

这样一份配置文件就能够描述一次完整实验。

---

### 3. 科研核心模型默认手写

本 Skill 强调 **模型结构透明性**。

允许直接使用 PyTorch 的基础积木，例如：

```text
Conv2d
Linear
BatchNorm
LayerNorm
Dropout
Pooling
Softmax
Tensor reshape / permute / matmul
```

但默认要求手写科研核心模块，例如：

```text
ResNet BasicBlock
ResNet Bottleneck
ResNet
Patch Embedding
Multi-Head Self-Attention
Transformer Encoder Block
MLP
ViT
DropPath / Stochastic Depth
论文中的核心创新模块
```

除非用户明确要求，否则不直接使用：

```text
torchvision.models.resnet*
torchvision.models.vit*
timm.create_model(...)
nn.MultiheadAttention
nn.TransformerEncoder
nn.TransformerEncoderLayer
```

这样可以让科研人员直接看到模型内部的计算过程，也更方便修改结构、提取中间特征和进行消融实验。

---

### 4. 中文注释 + 清晰的 Tensor Shape

默认代码风格：

- 类名、函数名、变量名使用英文；
- docstring 使用中文；
- 解释性注释使用中文；
- Attention、logits、feature、token、patch 等常见术语可以保留英文。

对于核心模型的 `forward()`，尤其是以下操作：

```text
reshape
flatten
permute
transpose
concatenate
attention
feature fusion
pooling
patch embedding
```

如果 Shape 可以确定，应明确写出 Tensor Shape 的变化。

例如：

```text
(B, N, D) × 3 → (B, N, 3D)
```

目标是让研究人员阅读代码时能够直接跟踪数据在网络中的流动。

---

### 5. 代码修改必须检查影响范围

代码 Agent 不应只完成“局部修改”。

例如新增一个模型参数后，还应检查它是否影响：

```text
Python implementation
        ↓
config.yaml
        ↓
README
        ↓
requirements
        ↓
train / test
        ↓
checkpoint
        ↓
tools / smoke test
```

本项目提供了专门的修改影响检查清单，用于降低“代码已经改了，但其他文件没有同步”的问题。

---

### 6. 推荐清晰的科研项目结构

推荐项目逐渐组织为：

```text
project/
├── configs/
├── datasets/
├── models/
├── trainers/
├── evaluators/
├── utils/
├── tools/
├── scripts/
├── tests/
├── outputs/
├── README.md
└── requirements.txt
```

其中入口脚本只负责：

```text
读取配置
→ 构建数据集
→ 构建模型
→ 构建训练/评估组件
→ 启动流程
```

而不是把数据处理、模型定义、训练循环、指标计算全部塞进一个巨大的 `train.py`。

---

### 7. 运行过程必须可观察

对于耗时操作，终端不应长时间没有任何反馈。

根据任务性质，应合理使用：

- `print`
- `logging`
- `tqdm`
- epoch / batch 信息
- loss
- accuracy
- learning rate
- 当前阶段
- 数据集统计
- 模型信息
- 特征提取进度
- 测试进度

训练和测试过程应保存必要日志，例如：

```text
training.log
testing.log
```

实验完成后，根据任务保存必要的：

- 指标文件；
- Loss 曲线；
- Accuracy 曲线；
- Confusion Matrix；
- ROC / PR 曲线；
- Open Set / OOD 指标；
- 其他任务相关图表。

**TensorBoard 默认不启用。**

只有用户明确要求，或者当前项目本身已经依赖 TensorBoard 时，再添加相关代码和依赖。

---

### 8. 高效的数据读取与缓存

数据读取设计需要考虑实际数据规模。

Agent 应主动判断是否需要：

- 预先建立样本索引；
- 避免重复扫描目录；
- 合理设置 `num_workers`；
- `pin_memory`；
- `persistent_workers`；
- `prefetch_factor`；
- GPU 传输时使用 `non_blocking`；
- Memory Mapping；
- 特征缓存；
- 昂贵预处理结果缓存。

缓存不是强制功能。

原则是：

> **当预处理或特征计算昂贵，并且结果能够被安全复用时，优先设计缓存机制。**

缓存同时需要考虑失效条件，避免模型、数据或预处理已经变化，却继续读取旧缓存。

---

### 9. 防止数据泄漏

默认推荐区分：

```text
train
val
test
```

其中：

- `train`：模型训练；
- `val`：模型选择、阈值标定、超参数选择；
- `test`：最终结果报告。

对于 Open Set / OOD 任务，需要特别注意未知类是否允许出现在训练和验证阶段。

本 Skill **允许 val / test 不区分**。

某些研究或快速实验中，可以只使用：

```text
train
test
```

但此时必须在代码、README 或实验说明中明确指出：

> 测试集同时承担模型选择、阈值选择或最终评估等职责，因此它不再是严格意义上的独立无偏最终测试集。

避免对实验含义产生误解。

---

### 10. 每次正式实验使用独立 Run Directory

推荐一次正式实验对应一个独立目录，例如：

```text
outputs/
└── runs/
    └── 20260920_160000_resnet50/
        ├── config.yaml
        ├── resolved_config.yaml
        ├── training.log
        ├── testing.log
        ├── environment.json
        ├── checkpoints/
        ├── metrics/
        └── figures/
```

这样可以避免：

```text
outputs/best_model.pth
```

被下一次实验直接覆盖，也方便论文实验对比、回溯和复现。

---

### 11. Checkpoint 是可配置能力

Checkpoint 不要求所有任务必须启用。

是否保存、保存频率、是否允许恢复训练等，都应通过 YAML 控制。

当启用完整 Checkpoint 时，根据任务需要记录：

```text
model state
optimizer state
scheduler state
AMP scaler
epoch
best metric
config
必要的随机状态
实验 metadata
```

从而使 Resume 真正恢复训练状态，而不只是重新加载模型权重。

---

### 12. 跨平台与路径可移植性

项目应尽量避免写死：

```text
C:\Users\xxx\...
/home/xxx/...
```

等机器相关绝对路径。

推荐：

- 使用相对项目根目录的路径；
- 优先使用 `pathlib.Path`；
- 统一处理路径解析；
- Device 通过配置或运行环境选择；
- 避免在业务逻辑中写死某张 GPU。

目标是让项目更容易在：

```text
Windows
Ubuntu
本地工作站
实验室服务器
```

之间迁移。

---

### 13. Fail Fast，禁止静默降级

程序应在真正开始长时间训练前验证：

- 配置字段是否合法；
- 数据路径是否存在；
- 数据划分比例是否合法；
- Unknown Class 是否存在；
- 图像尺寸是否符合模型要求；
- `embed_dim` 是否能被 `num_heads` 整除；
- Checkpoint 是否存在；
- Cache 是否有效；
- 设备配置是否可用。

如果用户配置了 CUDA，但 CUDA 不可用，不应该悄悄切换到 CPU 然后继续几个小时的训练。

如果用户指定的 Checkpoint 不存在，也不应该静默改成随机初始化。

**实验语义发生改变时必须让用户知道。**

---

### 14. 轻量级 Smoke Test

复杂科研项目推荐提供轻量级 Smoke Test，用来快速检查：

```text
Dataset 能否读取
DataLoader 能否取出 Batch
模型能否完成 Forward
输出 Shape 是否正确
Loss 能否计算
Backward 是否正常
配置能否解析
```

Smoke Test 的目标不是替代完整单元测试，而是在正式训练前尽早发现低级错误。

---

### 15. 记录实验环境

为了提高实验可复现性，正式 Run 推荐记录必要环境信息，例如：

```text
Python version
PyTorch version
CUDA version
GPU model
操作系统
关键依赖版本
```

这些信息可以保存到：

```text
environment.json
```

或实验日志中。

---

## 用户 Prompt 始终具有最高优先级

这套 Skill 是默认工程规范，而不是不可覆盖的框架。

例如用户明确要求：

> 这次只是快速做一个 baseline，直接使用 torchvision 的 ResNet50。

那么 Agent 应该按照用户要求执行。

用户也可以明确关闭：

- Run Directory；
- Checkpoint；
- Cache；
- Smoke Test；
- 确定性训练；
- 某些日志；
- 某些图像保存；
- 其他默认行为。

核心原则是：

> **Skill 负责提供高质量默认行为，而不是限制科研人员的实验自由。**

---

## 项目结构

```text
pytorch_research_agent_skill/
├── SKILL.md
├── README.md
└── references/
    ├── config.example.yaml
    ├── reproducibility.py
    ├── config_utils.py
    ├── run_manager.py
    ├── logging_utils.py
    ├── checkpoint.py
    ├── model_style_example.py
    ├── dataset_style_example.py
    ├── smoke_test_example.py
    ├── project_structure.md
    └── impact_checklist.md
```

### `SKILL.md`

整个项目的核心。

包含 Agent 在编写或修改 PyTorch 科研项目时应遵守的完整规范，并通过：

```text
MUST
SHOULD
MUST NOT
```

区分不同规则的强度。

### `references/`

这里的代码是 **参考实现，而不是要求 Agent 每次机械复制的模板**。

| 文件 | 作用 |
|---|---|
| `config.example.yaml` | 推荐的 YAML 配置组织方式 |
| `reproducibility.py` | 随机种子与实验可复现性参考实现 |
| `config_utils.py` | 配置读取、校验与路径解析参考 |
| `run_manager.py` | 独立 Run Directory 管理参考 |
| `logging_utils.py` | 终端 + 文件日志参考 |
| `checkpoint.py` | Checkpoint 保存与恢复参考 |
| `model_style_example.py` | 手写模型、中文注释、Tensor Shape 风格示例 |
| `dataset_style_example.py` | Dataset / DataLoader 编写风格示例 |
| `smoke_test_example.py` | 轻量级 Smoke Test 示例 |
| `project_structure.md` | 推荐科研项目文件结构 |
| `impact_checklist.md` | 修改代码后的影响范围检查清单 |

---

## 如何使用

将整个 Skill 目录放入你所使用的、支持 Skills 或自定义 Agent Instructions 的代码 Agent 环境中，并让 Agent 在处理 PyTorch 深度学习科研项目时读取 `SKILL.md`。

具体安装目录与加载方式取决于你所使用的 Agent。

核心文件是：

```text
SKILL.md
```

`references/` 中的文件主要用于在 Agent 需要具体实现时提供参考。

### 推荐使用方式

新建项目时，可以直接要求：

```text
请按照 pytorch-research-code-style Skill 创建这个 PyTorch 项目。
```

修改已有项目时：

```text
请按照 pytorch-research-code-style Skill 检查并重构当前项目，
保持算法逻辑不变。
```

新增模型时：

```text
按照项目 Skill 新增 ViT 模型。
核心 Attention 和 Transformer Block 不调用高级封装，
保持完整 Tensor Shape 注释。
```

复现实验时：

```text
按照 Skill 检查当前项目的随机种子、数据划分、
配置文件、Run Directory 和日志机制是否满足可复现要求。
```

---

## 适用场景

这套规范尤其适合：

- PyTorch 科研代码；
- 论文复现；
- 图像分类；
- RF / 时频信号识别；
- CNN / ResNet；
- ViT / Transformer；
- Open Set Recognition；
- OOD Detection；
- 多模态模型；
- 特征提取与后处理算法；
- 消融实验；
- 硕士 / 博士科研项目；
- 需要长期持续迭代的实验代码。

它并不特别适合直接约束：

- 大型商业后端项目；
- Web 服务；
- 移动端应用；
- 只追求极简 Demo 的一次性代码；
- 已经拥有成熟内部工程框架的团队项目。

---

## 设计取向

这套规范刻意选择：

**科研可读性 > 代码极度简洁**

**实验可复现性 > 临时方便**

**显式实现 > 黑盒高级封装**

**配置集中管理 > 大量命令行参数**

**长期可维护 > 一次运行成功**

但这不意味着所有项目都必须使用完全相同的结构。

Agent 应根据项目规模合理调整，不为了“规范”而过度设计。

---

## 后续计划

后续可以继续补充和迭代：

- [ ] CNN / ResNet 完整手写参考
- [ ] ViT 完整手写参考
- [ ] Open Set / OOD 项目模板
- [ ] 分类任务 Metrics 模板
- [ ] 混合精度与梯度累积参考
- [ ] DistributedDataParallel 科研项目规范
- [ ] 大规模数据集读取与缓存参考
- [ ] 实验自动化与消融实验管理规范
- [ ] 更完整的单元测试规范

如果实际使用过程中发现 Agent 经常出现某一类问题，可以继续将其抽象成新的规则加入 `SKILL.md`。

---

## 核心理念

最终希望代码 Agent 生成的不是：

> “一段能够跑起来的 PyTorch 代码”

而是：

> **“一个科研人员几个月后重新打开，仍然能够理解、修改、复现和继续实验的 PyTorch 项目。”**
