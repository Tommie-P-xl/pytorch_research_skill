---
name: pytorch-code-annotation
description: 学习已有 PyTorch 代码、补中文注释或解释张量形状时读取；项目构建后选择了代码注释，也使用本模块。
---

# 代码学习与注释

本模块既能用于给新项目加注释，也能单独用于读懂已有代码。**只学习代码时，不需要读其余五个模块。**

## 1. 先确定注释方式

从用户要求中找出下面四项，只问还不明确的部分：

| 要确认什么 | 例子与处理方式 |
|---|---|
| 是否修改文件 | “解释这个函数”只在对话中讲；“给它加注释”修改原文件；“做学习副本”写到指定副本 |
| 注释哪些代码 | 一个文件、一个函数、函数之间的调用过程，或整个项目。整个项目要先列出实际文件，再分批处理 |
| 讲多细 | 简洁、标准、教学。“详细学习注释”按教学理解，不重复询问 |
| 用什么语言 | 默认中文解释，保留英文变量名和 Attention、logits 等常用术语；用户指定其他语言时遵从 |

用户只说“加注释”时，先询问程度；没有答案可以先读代码，但不自行选择档位写入注释。用户说“采用推荐程度”时，可选择标准并说明。

| 程度 | 要写什么 | 适合什么情况 |
|---|---|---|
| 简洁 | 一句话说明用途，标关键限制和容易看错的形状变化 | 熟悉 PyTorch，只想快速看懂关键点 |
| 标准 | 说明输入、输出、主要步骤、这样写的原因和关键 Shape | 能读基本代码，需要理解模块怎样工作 |
| 教学 | 在标准基础上，展开算法步骤、符号、维度变化和梯度影响 | 初次接触该算法，希望跟着代码一步步学 |

三档采用同一套正确性要求，只改变解释的多少。不要为了“教学”逐行解释赋值、循环等基础语法。

## 2. 读懂以后再写

1. 看目标代码，以及必要的调用它的代码、它调用的函数、配置和数据格式。有 CodeGraph 时优先用它查看结构与调用关系，否则使用当前可用的搜索和阅读工具。
2. 沿着“输入 → 中间张量 → 输出”检查计算。例如，先确认 `x` 是图像还是 token，再解释 `flatten` 合并了哪些维度。
3. 只写能确认的形状。动态维度用 `B`、`N` 等符号；依赖配置时写明条件，不凭空写成固定数字。
4. 只有核对过论文或实现来源，才写“对应论文中的某个公式”。推测就注明是推测，不把它说成作者的设计意图。
5. 发现 bug 时说明位置和依据，修复单独处理，不在加注释时顺便改算法。

## 3. 怎么写注释

- 重要文件说明它做什么、接收和返回什么、在哪一步被调用。已有正确说明就保留，不重复写一份。
- 核心类和函数可用 docstring（定义下方的三引号说明）解释用途、参数、返回值和限制，不整段删除原有英文技术文档。
- 多解释“为什么”，少复述代码。例如：

  ```python
  # 信息太少：把 labels 转成 long。
  # 更有用：此处使用类别索引标签，CrossEntropyLoss 要求标签为整数类型。
  labels = labels.long()
  ```

- 在 `forward`、`reshape/view`、`permute/transpose`、`flatten`、池化、Q/K/V、拼接和特征输出处标出关键 Shape。
- `softmax` 说明沿哪个轴归一化；广播说明哪一项被扩展；`detach/no_grad` 说明哪些计算不记录梯度。只解释实际发生的事，不因函数名叫 attention 就套用某个算法。

常用形状符号如下；首次出现时说明，不要求每处重复：

| 符号 | 含义 | 例子 |
|---|---|---|
| B | 一批样本数 | B=2 表示一次处理 2 个样本 |
| C、H、W | 通道数、图像高、图像宽 | 图像 `(B, C, H, W)` |
| N、D | token 数、每个 token 的特征维度 | token 序列 `(B, N, D)` |
| A、Dh | 注意力头数、每个头的特征维度 | 多头张量 `(B, A, N, Dh)`；`D = A * Dh` |
| K | 类别数 | 分类输出 `(B, K)` |

例如，只知道输入是 `(B, N, D)` 时这样标注，不猜 `N=196`：

```python
x = x.transpose(1, 2)  # (B, N, D) -> (B, D, N)
```

只加注释时必须保留原代码的计算和接口：

- 不改函数签名、默认值、返回结构、运算顺序、配置、路径或实验安排；不添加打印、assert、类型提示或重命名。
- 保留文件头、编码声明、许可证、`from __future__` 导入位置，以及 `type: ignore`、`noqa` 等工具指令。模块 docstring 应放在 future import 之前，并保留已有文件头。
- docstring 能被程序读取。若代码通过 `__doc__` 注册对象或解析信息，先检查其用途，必要时只加 `#` 注释。

## 4. 三种详细程度的同一段代码

下面三段使用**完全相同的计算**，只改变注释。每段都是独立示例，选一段使用即可，不需要复制三份函数到项目。

示例计算简化的多头自注意力：`q/k/v` 都是 `(B, A, N, Dh)`，已由上游代码生成；这里不展示生成 Q/K/V 的线性层、mask、dropout 和最后的输出投影。实际项目仍保留原有步骤。

例如 `B=2, A=4, N=8, Dh=16`，输出为 `(2, 8, 64)`，因为 `D = 4 * 16 = 64`。

### 4.1 简洁：只标用途和关键点

```python
def attention_context(q, k, v):
    """合并各注意力头的输出，返回 (B, N, A * Dh)。"""
    batch_size, num_heads, num_tokens, head_dim = q.shape
    scale = head_dim ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale
    weights = scores.softmax(dim=-1)  # 沿 key 位置归一化。
    context = weights @ v
    return context.transpose(1, 2).reshape(
        batch_size, num_tokens, num_heads * head_dim,
    )  # (B, A, N, Dh) -> (B, N, A * Dh)
```

### 4.2 标准：说明输入输出和主要步骤

```python
def attention_context(q, k, v):
    """计算各 token 的注意力特征，并合并多个头。

    参数：
        q/k/v：形状均为 (B, A, N, Dh) 的 query、key、value 张量。
            本示例要求序列长度相同，Dh > 0，dtype 和 device 兼容。
    返回：
        (B, N, D) 的特征，其中 D = A * Dh。
    """
    batch_size, num_heads, num_tokens, head_dim = q.shape
    scale = head_dim ** -0.5

    # 计算 query 与 key 的相似度，并用 1/sqrt(Dh) 缩放。
    scores = (q @ k.transpose(-2, -1)) * scale  # (B, A, N, N)
    weights = scores.softmax(dim=-1)  # 每个 query 对所有 key 的权重和为 1。
    context = weights @ v  # 按权重汇总 value，得到 (B, A, N, Dh)。

    # 把 token 放到头数之前，再合并 A 与 Dh，恢复每个 token 的完整特征。
    return context.transpose(1, 2).reshape(
        batch_size, num_tokens, num_heads * head_dim,
    )  # (B, A, N, Dh) -> (B, N, A, Dh) -> (B, N, D)
```

### 4.3 教学：展开每一步为什么这样做

```python
def attention_context(q, k, v):
    """让每个 token 根据与其他 token 的相关程度，汇总它们的 value 特征。

    参数：
        q/k/v：形状均为 (B, A, N, Dh)，要求序列长度相同，Dh > 0，
            dtype 和 device 兼容。
        q（query）用于发起匹配，k（key）用于计算匹配分数，
        v（value）提供最后被加权汇总的特征。
        B 是样本数，A 是头数，N 是 token 数，Dh 是每个头的特征维度。
    返回：
        (B, N, D) 的特征，其中 D = A * Dh。
    """
    batch_size, num_heads, num_tokens, head_dim = q.shape

    # 1. Dh 越大，点积的数值幅度可能越大。
    # 用 1/sqrt(Dh) 缩放，缓和 softmax 权重过于集中的问题。
    scale = head_dim ** -0.5

    # 2. transpose(-2, -1) 交换 k 的 token 维和特征维。
    # (B, A, N, Dh) @ (B, A, Dh, N) -> (B, A, N, N)
    # 最后两个 N 分别是 query 位置和 key 位置。
    scores = (q @ k.transpose(-2, -1)) * scale

    # 3. 最后一维对应所有 key；每个 query 在这一维上的权重和为 1。
    # dim=-1 指最后一维，不是 batch 维或 head 维。
    weights = scores.softmax(dim=-1)

    # 4. 按权重对 value 做加权求和。
    # (B, A, N, N) @ (B, A, N, Dh) -> (B, A, N, Dh)
    # 每个 query 位置得到一个汇总后的 Dh 维特征。
    context = weights @ v

    # 5. 先变成 (B, N, A, Dh)，让同一 token 的各头特征排在一起，
    # 再合并成 (B, N, A * Dh)。直接合并原张量会混淆头与 token 的顺序。
    # transpose 后内存可能不连续；reshape 必要时会复制，而 view 要求布局兼容。
    # 这里没有 detach 或 no_grad：若输入需要梯度，梯度会沿这些运算传回上游。
    return context.transpose(1, 2).reshape(
        batch_size, num_tokens, num_heads * head_dim,
    )
```

选择档位后，按这个密度处理实际代码；示例不是要求把已有 Attention 改写成上述函数。

## 5. 检查和交付

- 看修改前后的 diff，确认只变了注释或 docstring，没有误改导入、计算、接口或配置。
- 可比较注释前后的语法树（AST）。只忽略模块、类、函数开头真正的 docstring，不忽略其他字符串，因为字符串也可能参与计算。
- AST 相同只能说明代码结构没变；仍需检查读取 `__doc__` 的代码，以及供工具使用的注释。
- 使用已有相关检查，不为注释启动完整训练。缺依赖时先做语法检查，并明确说明没有运行代码。
- 核对每处 Shape、轴和条件，无法确认的内容写明依赖或待核实。
- 交付时说明处理了哪些文件、采用哪档、覆盖哪些范围、检查结果和仍不确定的内容。只要求注释时不额外生成新架构或图表；项目构建任务则回主入口完成剩余交付。
