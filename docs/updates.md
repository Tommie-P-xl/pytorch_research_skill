# 安装与更新：仓库维护说明

本文件给用户和管理器使用，不由 Skill 加载。PyTorch 工作说明仅位于 `skills/pytorch-research-code-style/`，不会在对话开始时检查版本。

## 选择更新方式

| 安装方式 | 更新方式 |
|---|---|
| CC Switch 从 GitHub 仓库安装 | 由 CC Switch 检查更新、下载并同步各应用 |
| 直接解压 Release Skill ZIP | 从终端运行独立更新工具，或重新安装新版 ZIP |
| Git 克隆的开发仓库 | 正常通过 Git 管理源码，不用安装更新工具覆盖 |
| 平台只接受上传 ZIP | 使用平台提供的更新或重新导入操作 |

## CC Switch 仓库安装

在 Skills 的“仓库管理”中添加：

```text
仓库：Tommie-P-xl/pytorch_research_skill
分支：main
技能路径：skills
```

然后从“发现”安装 `pytorch-research-code-style`，选择要启用的应用。后续在“已安装”中检查更新并点击更新，由管理器同步。
CC Switch 按仓库文件的内容摘要检测变化，不需要每次都发布版本标签。仓库来源安装才能跟踪远端；本地或 ZIP 导入不等于自动关联这个仓库。

详见 [CC Switch 官方说明](https://github.com/farion1231/cc-switch/blob/main/docs/user-manual/zh/3-extensions/3.3-skills.md)。

## 独立更新工具

需要 Python 3.10+，无需 PyTorch、Git 或第三方 Python 库。工具在仓库 `tools/update_skill.py`，也单独发布为 Release 资产 `update_skill.py`。

将工具保存在 Skill 目录之外；从终端运行时显式指定已安装的 Skill 目录：

```bash
# 从仓库根目录运行。
python tools/update_skill.py --skill-dir "<已安装Skill目录>" --check
python tools/update_skill.py --skill-dir "<已安装Skill目录>" --update

# 单独下载工具后，可在你自己的工具目录运行。
python update_skill.py --skill-dir "<已安装Skill目录>" --check
python update_skill.py --skill-dir "<已安装Skill目录>" --update
```

`--check` 只报告最新稳定版，`--update` 发现新版后校验、备份并更新直接安装的副本。`--auto` 可供外部任务调用：24 小时内复用成功检查结果，发现新版则更新。它不由 Skill 触发，不常驻后台，也不会创建系统定时任务。
CC Switch 管理的副本仍由管理器更新；自定义路径无法自动识别时，加 `--managed` 明确禁止直接覆盖。

输出为一行 JSON：

| status | 含义 |
|---|---|
| `up_to_date` / `local_ahead` | 已是最新稳定版或本地版本更高，不降级 |
| `cached` | 24 小时内已经检查过，`last_result` 是上次结果 |
| `update_available` | 只检查时发现新版，可另运行 `--update` |
| `updated` | 更新完成，`backup` 是旧副本 ZIP 的路径 |
| `managed_by_cc_switch` | 交给 CC Switch 更新和同步 |
| `development_checkout` | 开发仓库保持原样，通过 Git 更新 |
| `check_or_update_failed` | 网络、校验、权限或本地改动导致失败，详情在 message |

手动检查/更新失败返回非零；`--auto` 失败返回 0，但 JSON 仍标记失败，方便外部调度记录。`--check` 和 `--update` 不受缓存间隔限制。

## 校验、备份和恢复

工具只使用这个仓库的最新稳定 Release。先检查 `SHA256SUMS.txt`、包内版本和逐文件摘要，拒绝预发布包、降级、不合法路径和混入维护工具的包。
本地包文件被修改、缺失，或新文件与用户额外文件冲突时停止。正常更新保留额外文件，只移除旧清单里被新版淘汰的文件；替换前保存并校验完整旧副本，替换失败时尝试恢复旧目录。

仓库版本来自根目录 `VERSION`。安装包版本和文件摘要位于自动生成的 `package-manifest.json`，供外部工具读取，不写进 SKILL.md，也不包含 Agent 指令。不要手工编辑清单。

缓存和备份位于 Windows 的 `%LOCALAPPDATA%/pytorch-research-code-style/updater/`；其他系统使用 `$XDG_CACHE_HOME/pytorch-research-code-style/updater/`，未配置时使用 `~/.cache/`。按安装目录分开保存，不写进技能目录。
恢复时保留旧副本 ZIP，停止使用该 Skill，将备份解压回原安装目录；备份根目录直接是 `SKILL.md`，没有多套一层。

## 从旧版迁移

- **1.1.0**：通过 CC Switch 更新，或下载本版独立更新工具，在技能目录外运行 `--update --skill-dir "<已安装Skill目录>"`。工具兼容旧清单，更新后删除旧包拥有的 `scripts/update_skill.py` 和 `references/updates.md`，并移除 SKILL.md 中的更新指令。旧内容完整保留在备份 ZIP 中。旧版内置工具不支持新版纯 Skill 包，不使用它完成本次迁移。
- **1.0.x 或没有包清单的副本**：通过管理器更新，或重新安装本版 Release Skill ZIP。
- **旧仓库根路径**：若管理器仍按旧根目录查找，重新从 `skills` 路径安装。直接指定入口的使用方式也改为 `skills/pytorch-research-code-style/SKILL.md`。

本次不会替换用户机器上已经安装的副本；上面说明的是用户自行升级时的流程。

## 发布新版本

修改根目录 `VERSION`，运行测试和打包，提交并推送到 main，再推送同版本标签。例如下一版：

```bash
python -m unittest discover -s tests -v
python tools/build_skill.py
git tag v1.3.0
git push origin v1.3.0
```

`VERSION` 应先填写 `1.3.0`。GitHub Actions 校验标签与版本一致，发布三项资产：

- `pytorch-research-code-style.zip`：PyTorch Skill 主文件、六个模块、文件摘要清单。
- `update_skill.py`：独立维护工具，不在 Skill ZIP 内。
- `SHA256SUMS.txt`：以上两个文件的 SHA-256。

CC Switch 跟踪 main 分支内容；独立工具跟踪稳定 Release。发布新版本不会改变已经发布的旧版本资产。
