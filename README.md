# PyTorch Research Skill

当前版本：**1.2.0** · [Release 与下载](https://github.com/Tommie-P-xl/pytorch_research_skill/releases/tag/v1.2.0)

这是一套给 AI 编程助手使用的 PyTorch 科研说明。它让助手按统一规则编写项目、批量运行实验，或帮你读懂已有代码。

先读一个主文件，再根据任务读取需要的模块。例如，只想给 Attention 加注释，就只读注释模块；要批量训练和测试，就在设计流程时读批量模块。不必一次读完全部规则。

## 主要功能

- **项目构建**：统一目录职责、YAML 配置、随机种子、路径、命名和代码排版。
- **自动设备选择**：默认优先 CUDA，自动适配多 GPU、单 GPU 和 CPU。
- **实验记录与展示**：保存配置快照、日志和结构化指标，按需生成实验图表。
- **批量实验**：预设多组参数，串联训练、特征提取与测试，自动传递模型和 NPZ 等产物路径。
- **代码学习与注释**：为已有代码补充中文解释、算法说明与 Tensor Shape 注释，支持三种详细程度。

Python 和 YAML 科研示例放在文档里，助手会根据它们在你的项目中编写代码。Skill 内容位于 `skills/`；仓库另外提供独立的维护工具，不依赖 PyTorch。

## 快速开始

### 1. 安装 Skill

按使用方式选择：

| 使用方式 | 推荐安装方法 | 后续更新 |
|---|---|---|
| 使用 CC Switch 管理多个 Agent | 从 GitHub 仓库安装 | 在 CC Switch 中检查并更新，同步到已启用的应用 |
| 直接放入 Agent 的技能目录 | 下载并解压 Release Skill ZIP | 使用管理器或仓库级独立更新工具 |
| 平台只支持上传 ZIP | 上传 Release Skill ZIP | 使用平台的更新或重新导入功能 |
| 开发和修改这套 Skill | Git 克隆仓库 | 通过 Git 管理源码 |

#### 使用 CC Switch

1. 打开 Skills 页面的“仓库管理”，添加以下仓库信息。
2. 在“发现”中找到 `pytorch-research-code-style`，从这个仓库安装，并选择需要启用的应用。
3. 以后在“已安装”中检查更新，有新版时点击这一项的“更新”或批量更新。

```text
仓库：Tommie-P-xl/pytorch_research_skill
分支：main
技能路径：skills
```

从仓库安装才能保留远端来源信息；本地导入或 ZIP 导入不等于已经关联这个 GitHub 仓库。操作细节见 [更新说明](docs/updates.md#cc-switch-仓库安装)。

#### 直接安装或上传 ZIP

从 [最新 Release](https://github.com/Tommie-P-xl/pytorch_research_skill/releases/latest) 下载 [pytorch-research-code-style.zip](https://github.com/Tommie-P-xl/pytorch_research_skill/releases/latest/download/pytorch-research-code-style.zip)。支持 ZIP 导入的环境可直接上传；采用目录安装的环境先解压，再将 `pytorch-research-code-style/` 放入技能目录。

ZIP 按 [Agent Skills 目录规范](https://agentskills.io/specification) 打包，根目录名与 `SKILL.md` 中的 `name` 一致，包含主文件、全部模块和文件校验清单；更新工具单独下载。请下载 Release 中的 Skill ZIP，而不是 GitHub 自动生成的源码压缩包。

#### 开发这套 Skill

```bash
git clone https://github.com/Tommie-P-xl/pytorch_research_skill.git
```

### 2. 加载 Skill

通过 CC Switch 安装后，在对应应用开启即可。直接安装时，将解压后的完整 Skill 目录提供给 Agent，或放入它的技能目录；具体位置以使用环境为准。

Skill 名称为 `pytorch-research-code-style`。如果运行环境要求目录名与 Skill 名称一致，可将目录命名为 `pytorch-research-code-style`，并保留 `modules/` 的相对位置。

也可以直接指定入口：

```text
请读取 pytorch_research_skill/skills/pytorch-research-code-style/SKILL.md，
按照其中的阶段流程与模块选择规则处理当前 PyTorch 项目。
```

### 3. 选择任务与模块

主入口会识别项目构建、批量实验或源码学习任务，并询问尚未明确的模块选项。已指定的选择不会重复询问，各模块仅在对应阶段读取。

```text
请按照 pytorch-research-code-style 构建 PyTorch 分类项目。
先确定整体架构，再询问需要启用哪些模块。
实验展示保存日志、指标和 SVG 曲线，最后添加标准程度的中文注释。
```

## 安装与更新由仓库工具负责

参照 [superpowers 的分层结构](https://github.com/obra/superpowers)，本仓库将技能内容与安装维护分开：

| 内容 | 位置 | 谁使用 |
|---|---|---|
| PyTorch 工作规则、代码示例 | `skills/pytorch-research-code-style/` | Agent 按需读取 |
| 版本号、安装和更新说明 | `VERSION`、`docs/` | 用户和管理器 |
| 检查、更新和打包工具 | `tools/` | 用户、管理器或外部任务 |

`SKILL.md` 和六个模块只说明 PyTorch 工作，不含版本号、版本检查步骤或更新工具链接。加载 Skill 不会要求 Agent 发起更新，也不会读入更新日志、备份说明和发布流程。

使用 CC Switch 时，从仓库的 `skills` 目录发现并安装，由它检查更新和同步应用。直接安装的副本可在 Skill 目录外保存独立工具，再从终端运行：

```bash
# 在本仓库根目录运行；路径换成已安装的 Skill 目录，需要 Python 3.10+。
python tools/update_skill.py --skill-dir "<已安装Skill目录>" --check
python tools/update_skill.py --skill-dir "<已安装Skill目录>" --update
```

独立工具也提供 `--auto`（缓存成功检查结果 24 小时，有新版则更新），可供用户已配置的外部任务调用。Skill 不触发它，本仓库也不会自动创建系统定时任务。

**从 1.1.0 迁移**：通过 CC Switch 更新，或下载本版独立 `update_skill.py`，在技能目录外运行并显式指定 `--skill-dir`。独立工具支持读取旧包清单，更新时移除旧包附带的更新脚本和说明，并备份旧副本。旧版内置更新器不支持新版纯 Skill 包，不使用它执行这次迁移。若管理器仍记录旧仓库根路径，从新的 `skills` 路径重新安装一次。

详细操作见 [仓库级更新说明](docs/updates.md)。

## 模块说明

[SKILL.md](skills/pytorch-research-code-style/SKILL.md) 说明基础规则、不同任务怎么开始、有哪些模块，以及什么时候读取它们。各模块采用“要知道什么 → 具体怎么做 → 示例 → 检查”的写法。

| 模块 | 内容 | 加载时机 |
|---|---|---|
| [数据流程与效率](skills/pytorch-research-code-style/modules/data-pipeline.md) | 数据划分、样本列表、批量读取、子进程种子与可选缓存 | 编写数据读取代码前 |
| [模型与训练](skills/pytorch-research-code-style/modules/model-training.md) | 核心网络、训练与评估、自动并行策略 | 核心实现前 |
| [实验记录与展示](skills/pytorch-research-code-style/modules/experiment-reporting.md) | 独立实验目录、配置副本、日志、指标与图表 | 编写指标保存代码前及画图时 |
| [检查点、验证与交付](skills/pytorch-research-code-style/modules/validation-delivery.md) | 保存模型、继续训练、少量数据检查、修改影响检查 | 编写模型保存/恢复代码前及交付时 |
| [批量实验与流程编排](skills/pytorch-research-code-style/modules/batch-experiments.md) | 多组参数、步骤顺序、自动传递文件路径、失败处理和续跑 | 决定实验流程和文件传递方式时 |
| [代码学习与注释](skills/pytorch-research-code-style/modules/code-annotation.md) | 源码追踪、中文 docstring、算法解释与 Tensor Shape | 注释阶段或独立学习任务 |

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

写训练代码时就接入日志和指标保存，后面再用记录画图；模型保存也要提前规划。

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

每个步骤用 `artifacts.json` 记录实际生成的模型、特征或指标文件。下一步先检查这些文件，再自动填写配置中的路径，省去手动复制路径。默认一组完成后再做下一组，每组仍可自动使用多 GPU。

基础 YAML 保持不变，每组使用单独的配置。某一步失败后，依赖它的步骤跳过；旧实验文件不会自动当作本次结果。续跑前检查配置、输入和输出是否仍然匹配，重试使用新目录。

Agent 为目标项目实现编排入口后，可使用：

```bash
python scripts/run_experiments.py --config configs/experiments.yaml
```

参数清单、步骤顺序和自动填写文件路径的示例见 [批量实验模块](skills/pytorch-research-code-style/modules/batch-experiments.md)。

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
| 简洁 | 说明用途，只标容易看错的点和重要形状变化 |
| 标准 | 说明输入输出、主要步骤、这样写的原因和关键形状 |
| 教学 | 再展开算法步骤、符号含义、维度怎样变化，以及梯度传到哪里 |

默认使用中文解释，保留英文标识符与常见技术术语。

想直观看到三档差别，阅读 [同一段 Attention 的三种注释示例](skills/pytorch-research-code-style/modules/code-annotation.md#4-三种详细程度的同一段代码)。三段的计算完全一样，只改变说明的详细程度。

其他短代码示例：

- [配置与路径](skills/pytorch-research-code-style/SKILL.md#6-配置与路径示例)：从 YAML 取学习率，相对路径统一从项目根目录算起。
- [数据流程](skills/pytorch-research-code-style/modules/data-pipeline.md)：用训练集计算标准化参数，再用于验证和测试。
- [模型与训练](skills/pytorch-research-code-style/modules/model-training.md)：一步训练、一轮验证，以及如何按样本数算平均 loss。
- [实验记录](skills/pytorch-research-code-style/modules/experiment-reporting.md)：指标和逐样本预测应保存哪些字段。
- [检查与交付](skills/pytorch-research-code-style/modules/validation-delivery.md)：只保存模型权重与完整续训有什么不同。
- [批量实验](skills/pytorch-research-code-style/modules/batch-experiments.md)：覆盖当前组参数但不修改基础配置，自动填写本组模型路径。

## 项目基础约定

- 实验参数集中在 YAML，默认业务 CLI 仅保留 `--config`。
- 使用统一随机种子；不启用严格确定性算法检查，`deterministic` 仅控制 cuDNN 设置。
- 默认自动选择多 GPU → 单 GPU → CPU。多卡优先 DDP，后端不支持时采用 DataParallel。
- 用户无需手填 GPU 数或额外启动 torchrun；batch 是所有卡合计的样本数，样本较少时只用能分到样本的设备。
- 使用可移植路径，统一按项目根目录解析，避免机器专属路径成为共享默认值。
- 核心科研模型默认显式实现，保持数据、模型、训练和评估职责清晰。
- 用训练集更新权重、验证集选模型、测试集做最终评估，不根据测试结果调参。
- 用户指定的算法、实验协议、模块选择和输出格式优先于默认约定。

自动设备选择侧重兼容性与合理默认策略，实际加速效果取决于模型、batch 和硬件。

## 仓库结构

```text
pytorch_research_skill/
├── README.md
├── VERSION                      # 仓库版本，不进入 Skill 指令
├── skills/
│   └── pytorch-research-code-style/
│       ├── SKILL.md             # PyTorch 主入口
│       └── modules/             # 六个 PyTorch 模块
├── docs/updates.md              # 给用户看的安装、更新说明
├── tools/
│   ├── update_skill.py          # 独立维护工具
│   └── build_skill.py           # 构建安装包与独立工具
├── tests/test_update_skill.py
└── .github/workflows/release-skill.yml
```

本仓库提供 Skill 指令与参考代码。上文的训练、测试和批量入口属于 Agent 按 Skill 构建的目标 PyTorch 项目，不是本仓库附带的可执行程序。

## 自动构建与发布

[GitHub Actions](https://github.com/Tommie-P-xl/pytorch_research_skill/actions/workflows/release-skill.yml) 先运行更新工具测试，再校验元数据、版本号和本地链接，构建 ZIP 与 `SHA256SUMS.txt`。构建实现位于 `tools/build_skill.py`；Skill ZIP 只包含主文件、六个模块和文件摘要清单；更新工具作为独立 Release 产物发布。

本地检查和打包：

```bash
python -m pip install PyYAML
python -m unittest discover -s tests -v
python tools/build_skill.py
```

PyYAML 仅用于维护时打包；独立更新工具只使用标准库。

| 触发方式 | 结果 |
|---|---|
| 推送到 `main` 或提交 Pull Request | 校验并上传 `skill-package` 构建产物 |
| Actions 页面手动运行 | 构建当前选择的分支；选择版本标签时也会发布 |
| 推送 `v1.2.0` 等版本标签 | 构建后自动创建 GitHub Release，上传 Skill ZIP、独立更新工具与校验文件 |

维护者发布新版本：

```bash
git tag v1.3.0
git push origin v1.3.0
```

上面以未来版本 `1.3.0` 为例：发布前将仓库根目录的 `VERSION` 改为同一版本，提交并推送。版本标签必须与这个字段一致；`1.3.0-rc.1` 等版本可发布为预发布版，但自动更新只下载安装稳定 Release。版本内容改变时使用新版本号和新标签。

安装包结构：

```text
pytorch-research-code-style.zip
└── pytorch-research-code-style/
    ├── SKILL.md
    ├── modules/*.md
    └── package-manifest.json
```

同一 Release 另提供独立 `update_skill.py`。下载所需文件和 `SHA256SUMS.txt` 后，可使用 SHA-256 工具核对文件；Linux/macOS 可运行 `sha256sum --check SHA256SUMS.txt`（需已安装该工具）。
