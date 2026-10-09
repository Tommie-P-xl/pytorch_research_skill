# 版本检查与更新

## 先看安装方式

| 你怎样安装 | 怎样更新 |
|---|---|
| 在 CC Switch 中从 GitHub 仓库安装 | 用 CC Switch 的检查更新和更新按钮，它负责同步各 Agent |
| 解压本仓库的 Release Skill ZIP 到技能目录 | 使用随包附带的 Python 更新脚本，可在首次使用时自动更新 |
| Git 克隆的开发仓库 | 正常使用 Git 管理，不让更新脚本覆盖未提交修改 |
| 只能上传文件、不能执行命令的 Agent | 使用该平台的更新或重新导入功能，Skill 文字本身不能后台运行 |

## CC Switch：从仓库安装，才能跟踪更新

在 CC Switch 的 Skills 页面打开“仓库管理”，添加：

```text
仓库：Tommie-P-xl/pytorch_research_skill
分支：main
技能路径：留空（SKILL.md 在仓库根目录）
```

然后在“发现”里从这个仓库安装 Skill，并开启需要使用的应用。之后在“已安装”里“检查更新”，有更新时点这一项的“更新”或批量更新。

CC Switch 通过文件内容的 SHA-256 检测远端变化，不要求每次改动都发布版本标签。**自动检测和自动安装是两回事**：当前官方说明提供检查和更新按钮，不能靠 Skill 元数据让它无人值守安装。
从本地目录或 ZIP 导入，不等于已经关联本仓库；要使用管理器的仓库更新能力，应从仓库安装。更新时保留的备份和各应用同步由管理器处理。

详见 [CC Switch 官方 Skills 管理说明](https://github.com/farion1231/cc-switch/blob/main/docs/user-manual/zh/3-extensions/3.3-skills.md)。

## 直接安装：三个命令

需要 Python 3.10 或更高版本，不需要 PyTorch、Git 或第三方 Python 库。将下方路径替换为**已安装 Skill** 中的脚本路径，不是你正在编写 PyTorch 项目的路径。

```bash
# 立即检查版本，不改 Skill 文件。
python "<Skill目录>/scripts/update_skill.py" --check

# 立即检查，有更新就下载、校验、备份并更新。
python "<Skill目录>/scripts/update_skill.py" --update

# 适合 Agent 使用：24 小时内复用检查结果，发现新版自动更新。
python "<Skill目录>/scripts/update_skill.py" --auto
```

脚本输出一行 JSON，Agent 可以根据 `status` 判断下一步：

| status | 含义与处理 |
|---|---|
| `up_to_date` / `local_ahead` | 已是最新稳定版，或本地版本更高；不降级 |
| `cached` | 24 小时内已经检查过；`last_result` 保留上次结果，通常不用重复提示 |
| `update_available` | 只检查时发现新版；运行 `--update` 可更新 |
| `updated` | 已更新；重新读 SKILL.md 和当前用到的模块，`backup` 是旧副本 ZIP 路径 |
| `managed_by_cc_switch` | 交给 CC Switch 更新，不改其管理的目录 |
| `development_checkout` | 当前目录是开发仓库，未覆盖源码 |
| `check_or_update_failed` | 网络、校验、权限或本地改动等原因导致失败；说明原因并继续原任务 |

`--check` 和 `--update` 会立即联网，不受每天一次的缓存限制。`--auto` 失败时返回 0，避免挡住科研任务，但 JSON 仍明确说明失败；手动模式失败返回非零。
若管理器使用自定义路径、自动检测无法确认，可加 `--managed` 明确禁止直接覆盖。

## 自动更新会怎样处理文件

1. 查询这个仓库最新的稳定 GitHub Release，比较版本号，不安装预发布版本或旧版本。
2. 下载 Skill ZIP 和同一 Release 的 `SHA256SUMS.txt`，检查整个包以及包内文件清单的摘要、版本和路径。
3. 检查本地包文件是否被修改；有修改、缺文件或新版文件与用户额外文件冲突时停止，不覆盖它们。
4. 在 Skill 目录外保存完整旧副本 ZIP，再准备新版目录。保留用户额外添加的文件，只删除旧包清单中已被新版移除的文件。
5. 更换目录；更换失败时恢复旧目录。成功后返回备份路径，不自动清理备份。

版本号在 `SKILL.md` 的 `metadata.version`。`package-manifest.json` 在打包时生成，记录包中每个文件的 SHA-256；不手工编辑它。直接更新需要这份清单，因此旧版或 GitHub 自动源码 ZIP 首次迁移时，请安装带清单的新 Release Skill ZIP。

缓存和备份在 Windows 的 `%LOCALAPPDATA%/pytorch-research-code-style/updater/`，其他系统的 `$XDG_CACHE_HOME/pytorch-research-code-style/updater/`（未设置时用 `~/.cache/`），按安装目录分开保存，不写入 Skill 目录，不干扰管理器的内容哈希。
需要恢复时，保留备份 ZIP，退出正在使用 Skill 的进程，把备份解压回原 Skill 目录即可；备份内容从 `SKILL.md` 开始，没有多套一层目录。

## 发布下一版

CC Switch 跟踪的是仓库分支；直接更新脚本跟踪的是稳定 Release。要让两种安装方式都获取新版：

1. 修改 Skill，更新 `SKILL.md` 中的版本号，例如下一版 `1.2.0`。
2. 运行检查，提交并推送改动到 `main`。
3. 创建与版本号一致的标签，例如：

   ```bash
   git tag v1.2.0
   git push origin v1.2.0
   ```

GitHub Actions 会校验版本号和标签是否一致，生成包含更新脚本、说明和清单的 ZIP，再上传 Release 和 `SHA256SUMS.txt`。标签与文件里的版本不一致时会停止打包，避免用户下载“标签是新版、内容还是旧版”的文件。
在 GitHub 上发布之前，本地修改不会让其他 Agent 获得新版。
