# 推荐项目目录职责

```text
project/
├── configs/        # 所有实验可调参数
├── datasets/       # Dataset、split、DataLoader
├── models/         # 手写 backbone、block、完整模型
├── trainers/       # train/validate 流程
├── evaluators/     # metrics、测试、OOD/OSR 算法
├── utils/          # 通用工具，不承载任务核心逻辑
├── tools/          # dataset check、smoke test 等辅助入口
├── scripts/        # train/test 等轻量级主入口
├── outputs/        # 所有运行产物
├── README.md
└── requirements.txt
```

## 目录边界

- `models/` 不负责读取数据集。
- `datasets/` 不负责保存模型 checkpoint。
- `trainers/` 不实现具体 backbone。
- `evaluators/` 不修改模型训练参数。
- `scripts/` 只负责组装流程，不堆积大量核心逻辑。
- `utils/` 只放真正可复用的通用工具，避免成为“杂物文件夹”。

## val/test 合并项目

如果项目不区分 val 与 test，推荐命名为：

```text
train_loader
eval_loader
```

而不是伪造：

```text
val_loader
test_loader
```

并在 README 中明确 eval 集同时用于模型选择与最终评估。
