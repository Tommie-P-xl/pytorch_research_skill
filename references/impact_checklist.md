# 代码修改后的影响范围检查表

当 Agent 修改科研项目时，逐项判断是否需要同步修改：

- [ ] `configs/*.yaml`
- [ ] 配置合法性检查
- [ ] Dataset / DataLoader
- [ ] 模型构造接口
- [ ] Trainer
- [ ] Evaluator / Metrics
- [ ] Checkpoint 兼容性
- [ ] Cache key / cache invalidation
- [ ] 日志字段
- [ ] 指标输出文件
- [ ] 图像输出
- [ ] Smoke test
- [ ] README
- [ ] requirements.txt
- [ ] 运行命令示例

## 常见例子

### 新增模型超参数

例如新增 `drop_path_rate`：

1. YAML 增加配置；
2. config validator 检查合法范围；
3. 模型构造函数增加参数；
4. README 参数表补充；
5. checkpoint 中保存 resolved config；
6. smoke test 覆盖该模型。

### 新增第三方依赖

1. 更新 `requirements.txt`；
2. README 环境安装说明同步；
3. 若依赖是可选功能，缺失时必须明确 warning 或报错；
4. 不得静默关闭关键算法逻辑。
