# PyTorch Research Skill

面向 PyTorch 深度学习科研的模块化 Agent Skill，支持项目构建、批量实验和源码学习。

通过一个主入口与六个按需加载的模块，将工程规范分配到不同阶段：先确定项目架构，再完成核心实现、实验展示与代码注释，减少一次性加载全部要求带来的上下文负担。

## 主要功能

- **项目构建**：统一目录职责、YAML 配置、随机种子、路径、命名和代码排版。
- **自动设备选择**：默认优先 CUDA，自动适配多 GPU、单 GPU 和 CPU。
- **实验记录与展示**：保存配置快照、日志和结构化指标，按需生成实验图表。
- **批量实验**：预设多组参数，串联训练、特征提取与测试，自动传递模型和 NPZ 等产物路径。
- **代码学习与注释**：为已有代码补充中文解释、算法说明与 Tensor Shape 注释，支持三种详细程度。

本仓库以 Markdown 分发。Python 与 YAML 示例直接嵌入模块文档，由 Agent 按实际项目实现。

## 快速开始

### 1. 获取仓库

**ZIP 安装（推荐）**：从 [最新 Release](https://github.com/Tommie-P-xl/pytorch_research_skill/releases/latest) 下载 [pytorch-research-code-style.zip](https://github.com/Tommie-P-xl/pytorch_research_skill/releases/latest/download/pytorch-research-code-style.zip)。支持 ZIP 导入的环境可直接上传；采用目录安装的环境先解压，再将 `pytorch-research-code-style/` 放入技能目录。

ZIP 按 [Agent Skills 目录规范](https://agentskills.io/specification) 打包，根目录名与 `SKILL.md` 中的 `name` 一致，包含主文件与全部模块。请下载 Release 中的 Skill ZIP，而不是 GitHub 自动生成的源码压缩包。

也可以克隆仓库：

```bash
git clone https://github.com/Tommie-P-xl/pytorch_research_skill.git
```

### 2. 加载 Skill

将仓库完整目录提供给支持本地 Skill 的 Agent，或放入该 Agent 的技能目录。具体安装位置以使用环境为准。

Skill 名称为 `pytorch-research-code-style`。如果运行环境要求目录名与 Skill 名称一致，可将目录命名为 `pytorch-research-code-style`，并保留 `modules/` 的相对位置。

也可以直接指定入口：

```text
请读取 pytorch_research_skill/SKILL.md，
按照其中的阶段流程与模块选择规则处理当前 PyTorch 项目。
```

### 3. 选择任务与模块

主入口会识别项目构建、批量实验或源码学习任务，并询问尚未明确的模块选项。已指定的选择不会重复询问，各模块仅在对应阶段读取。

```text
请按照 pytorch-research-code-style 构建 PyTorch 分类项目。
先确定整体架构，再询问需要启用哪些模块。
实验展示保存日志、指标和 SVG 曲线，最后添加标准程度的中文注释。
```

## 模块说明

[SKILL.md](SKILL.md) 定义项目基础规范、任务路由、模块选择与阶段加载规则。

| 模块 | 内容 | 加载时机 |
|---|---|---|
| [数据流程与效率](modules/data-pipeline.md) | 数据划分、样本索引、DataLoader、worker 种子与可选缓存 | 数据接口实现前 |
| [模型与训练](modules/model-training.md) | 核心网络、训练与评估、自动并行策略 | 核心实现前 |
| [实验记录与展示](modules/experiment-reporting.md) | 独立实验目录、配置快照、日志、指标与图表 | 指标接口实现前及图表阶段 |
| [检查点、验证与交付](modules/validation-delivery.md) | 可选 checkpoint/resume、smoke test 与修改影响检查 | 保存恢复接口实现前及交付阶段 |
| [批量实验与流程编排](modules/batch-experiments.md) | 参数组、阶段依赖、产物绑定、失败处理与续跑 | 实验流程和产物接口确定前 |
| [代码学习与注释](modules/code-annotation.md) | 源码追踪、中文 docstring、算法解释与 Tensor Shape | 注释阶段或独立学习任务 |

模块通过主入口中的相对链接加载，无需单独安装。模块选择时仅使用主文件中的摘要，随后按阶段读取正文。

## 使用示例

### 构建科研项目

```text
按照这个 Skill 创建 PyTorch 项目。
启用数据流程、模型训练、实验展示和验证模块。
不使用数据缓存，不保存完整 resume checkpoint。
最后添加标准程度的中文注释。
```

默认流程：

```text
整体架构 → 数据与模型实现 → 实验展示 → 添加注释 → 验证与交付
```

日志与指标接口在训练实现时接入，图表在后续展示阶段生成；checkpoint 接口同样提前规划。

### 批量训练与测试

```text
给当前项目加入批量实验模块。
预设多组 learning rate 和模型参数后，一次完成训练与测试。
训练模型和中间 NPZ 路径自动传入测试配置，保留基础 YAML。
一组实验失败时跳过其依赖步骤，继续其他独立实验。
```

批量流程：

```text
基础配置 + 参数清单
  → 独立实验与阶段配置
  → 训练 → 可选特征或统计提取 → 测试
  → 结果汇总
```

通过各阶段的 `artifacts.json` 传递已验证成功的产物，避免反复手动填写模型、特征和统计文件路径。不同参数组默认串行执行，每组仍可自动使用多 GPU。

基础 YAML 保持不变；失败阶段的下游任务跳过，历史实验产物不会被自动当作本次输出。续跑需核验配置、输入和产物，重试保留独立目录。

Agent 为目标项目实现编排入口后，可使用：

```bash
python scripts/run_experiments.py --config configs/experiments.yaml
```

参数清单、阶段链与字段绑定示例见 [批量实验模块](modules/batch-experiments.md)。

### 学习与注释已有代码

```text
给现有 models/attention.py 添加教学程度的中文学习注释。
重点解释 Q/K/V、reshape、softmax 的轴和 Tensor Shape。
保持算法、接口和配置不变。
```

也支持只在对话中解释：

```text
解释现有训练器的调用链和梯度流，不修改文件。
```

独立学习任务仅加载注释模块，不要求调整原项目架构。

| 注释程度 | 说明 |
|---|---|
| 简洁 | 关键用途、约束、易误读分支与必要 Shape |
| 标准 | 核心接口说明、参数与返回值、设计原因及关键 Shape |
| 教学 | 在标准内容上增加分步算法、符号含义、张量变换和梯度语义 |

默认使用中文解释，保留英文标识符与常见技术术语。

## 项目基础约定

- 实验参数集中在 YAML，默认业务 CLI 仅保留 `--config`。
- 使用统一随机种子；不启用严格确定性算法检查，`deterministic` 仅控制 cuDNN 设置。
- 默认自动选择多 GPU → 单 GPU → CPU。多卡优先 DDP，后端不支持时采用 DataParallel。
- 用户无需手填 GPU 数或额外启动 torchrun；保持全局 batch 语义，batch 较小时仅使用能分配到样本的设备。
- 使用可移植路径，统一按项目根目录解析，避免机器专属路径成为共享默认值。
- 核心科研模型默认显式实现，保持数据、模型、训练和评估职责清晰。
- 明确数据划分和模型选择边界，避免测试信息泄漏。
- 用户指定的算法、实验协议、模块选择和输出格式优先于默认约定。

自动设备选择侧重兼容性与合理默认策略，实际加速效果取决于模型、batch 和硬件。

## 仓库结构

```text
pytorch_research_skill/
├── .github/workflows/release-skill.yml
├── .gitignore
├── SKILL.md
├── README.md
└── modules/
    ├── data-pipeline.md
    ├── model-training.md
    ├── experiment-reporting.md
    ├── validation-delivery.md
    ├── batch-experiments.md
    └── code-annotation.md
```

本仓库提供 Skill 指令与参考代码。上文的训练、测试和批量入口属于 Agent 按 Skill 构建的目标 PyTorch 项目，不是本仓库附带的可执行程序。

## 自动构建与发布

[GitHub Actions](https://github.com/Tommie-P-xl/pytorch_research_skill/actions/workflows/release-skill.yml) 自动校验 Skill 元数据与本地文档链接，再构建 ZIP 和 `SHA256SUMS.txt`。发布包只包含 `SKILL.md` 与 `modules/`，构建代码直接保存在 workflow 中。

| 触发方式 | 结果 |
|---|---|
| 推送到 `main` 或提交 Pull Request | 校验并上传 `skill-package` 构建产物 |
| Actions 页面手动运行 | 构建当前选择的分支；选择版本标签时也会发布 |
| 推送 `v1.0.0` 等版本标签 | 构建后自动创建 GitHub Release，上传 ZIP 与校验文件 |

维护者发布新版本：

```bash
git tag v1.1.0
git push origin v1.1.0
```

发布前应将待发布改动提交并推送。带 `-rc.1` 等后缀的标签发布为预发布版本；同一标签的 workflow 重跑可重新上传该版本产物。

安装包结构：

```text
pytorch-research-code-style.zip
└── pytorch-research-code-style/
    ├── SKILL.md
    └── modules/
        └── *.md
```

下载 ZIP 和同一 Release 的 `SHA256SUMS.txt` 后，可使用 SHA-256 工具核对文件；Linux/macOS 可运行 `sha256sum --check SHA256SUMS.txt`（需已安装该工具）。
