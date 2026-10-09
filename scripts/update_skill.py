#!/usr/bin/env python3
"""检查稳定 Release，并更新直接安装的 Skill；只使用 Python 标准库。"""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4
import zipfile


SKILL_NAME = "pytorch-research-code-style"
REPOSITORY = "Tommie-P-xl/pytorch_research_skill"
ARCHIVE_NAME = SKILL_NAME + ".zip"
MANIFEST_NAME = "package-manifest.json"
MAX_DOWNLOAD = 8 * 1024 * 1024
CHECK_INTERVAL = 24 * 60 * 60


class UpdateError(Exception):
    """可向用户说明的更新失败。"""


def version_tuple(version: str, allow_prerelease: bool = False) -> tuple[int, int, int]:
    pattern = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    if allow_prerelease:
        pattern += r"(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?"
    if not isinstance(version, str) or not re.fullmatch(
        pattern, version
    ):
        raise UpdateError(f"需要稳定版本号，例如 1.1.0；实际为 {version!r}。")
    return tuple(int(part) for part in version.split("-", 1)[0].split("."))


def read_version(skill_text: str) -> str:
    """只读取本仓库约定的 name 和 metadata.version，不需要安装 YAML 库。"""
    frontmatter = re.match(r"\A---\r?\n(.*?)\r?\n---", skill_text, re.DOTALL)
    if frontmatter is None:
        raise UpdateError("缺少 SKILL.md 元数据。")
    content = frontmatter.group(1).replace("\r\n", "\n")
    if not re.search(rf"(?m)^name: {re.escape(SKILL_NAME)}\s*$", content):
        raise UpdateError("目标目录不是本 Skill，停止更新。")
    metadata = re.search(r"(?m)^metadata:\n((?:[ \t]+[^\n]*\n?)+)", content)
    match = re.search(r'(?m)^  version: "([0-9A-Za-z.-]+)"\s*$', metadata.group(1)) if metadata else None
    if match is None:
        raise UpdateError("未找到版本号，请先安装带更新工具的新版 Release ZIP。")
    version_tuple(match.group(1), allow_prerelease=True)
    return match.group(1)


def request_bytes(url: str, limit: int, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": SKILL_NAME + "-updater"})
    with urlopen(request, timeout=timeout) as response:
        final_url = urlsplit(response.geturl())
        host = final_url.hostname or ""
        if final_url.scheme != "https" or not (
            host in {"github.com", "api.github.com"}
            or host.endswith(".githubusercontent.com")
        ):
            raise UpdateError("下载跳转到非 GitHub HTTPS 地址，停止更新。")
        data = response.read(limit + 1)
    if len(data) > limit:
        raise UpdateError("下载内容超过大小限制。")
    return data


def latest_release(timeout: float) -> dict:
    url = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
    release = json.loads(request_bytes(url, 1024 * 1024, timeout))
    tag = release.get("tag_name", "")
    if not isinstance(tag, str) or not tag.startswith("v"):
        raise UpdateError("Release 标签格式不正确。")
    version_tuple(tag[1:])
    if release.get("draft") or release.get("prerelease"):
        raise UpdateError("自动更新只使用稳定 Release。")
    names = {asset.get("name") for asset in release.get("assets", [])}
    if not {ARCHIVE_NAME, "SHA256SUMS.txt"}.issubset(names):
        raise UpdateError("新 Release 缺少 Skill ZIP 或 SHA256SUMS.txt。")
    return {"version": tag[1:], "tag": tag}


def safe_relative(name: str) -> Path:
    if not isinstance(name, str) or not name:
        raise UpdateError("包内文件名为空或类型错误。")
    parts = PurePosixPath(name).parts
    if (
        "/".join(parts) != name
        or any(part in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts)
        or any(part.endswith(".") for part in parts)
    ):
        raise UpdateError(f"包内文件路径不合法：{name!r}。")
    if name not in {"SKILL.md", MANIFEST_NAME, "scripts/update_skill.py"} and not (
        len(parts) >= 2 and parts[0] in {"modules", "references"} and name.endswith(".md")
    ):
        raise UpdateError(f"包中包含不支持的文件：{name}。")
    return Path(*parts)


def validate_manifest(manifest: dict, version: str) -> dict[str, str]:
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("name") != SKILL_NAME
        or manifest.get("version") != version
        or not isinstance(manifest.get("files"), dict)
    ):
        raise UpdateError("包内文件清单或版本不匹配。")
    files = manifest["files"]
    if not {"SKILL.md", "scripts/update_skill.py", "references/updates.md"}.issubset(files):
        raise UpdateError("包内文件清单缺少必需文件。")
    if not any(name.startswith("modules/") for name in files):
        raise UpdateError("包内文件清单缺少模块。")
    folded = set()
    for name, digest in files.items():
        safe_relative(name)
        if name == MANIFEST_NAME or name.casefold() in folded:
            raise UpdateError("文件清单包含重复文件或自身。")
        folded.add(name.casefold())
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise UpdateError("文件清单的 SHA-256 格式错误。")
    return files


def unpack_verified(archive: bytes, version: str) -> dict[str, bytes]:
    """读取到内存，检查路径和摘要后才允许写入安装目录。"""
    payload = {}
    folded = set()
    with zipfile.ZipFile(io.BytesIO(archive)) as package:
        entries = package.infolist()
        if len(entries) > 512 or sum(item.file_size for item in entries) > MAX_DOWNLOAD:
            raise UpdateError("解压后内容超过限制。")
        for item in entries:
            prefix = SKILL_NAME + "/"
            if not item.filename.startswith(prefix) or item.is_dir():
                raise UpdateError("ZIP 根目录或文件结构不符合 Skill 包格式。")
            name = item.filename[len(prefix):]
            safe_relative(name)
            if stat.S_IFMT(item.external_attr >> 16) not in {0, stat.S_IFREG}:
                raise UpdateError("更新包不能包含软链接或特殊文件。")
            if name.casefold() in folded:
                raise UpdateError("ZIP 包含重复文件。")
            folded.add(name.casefold())
            payload[name] = package.read(item)
    if MANIFEST_NAME not in payload or "SKILL.md" not in payload:
        raise UpdateError("更新包缺少元数据或文件清单。")
    actual_version = read_version(payload["SKILL.md"].decode("utf-8"))
    if actual_version != version:
        raise UpdateError("ZIP 版本和 Release 标签不一致。")
    files = validate_manifest(json.loads(payload[MANIFEST_NAME]), version)
    if set(payload) != {*files, MANIFEST_NAME}:
        raise UpdateError("ZIP 内容和包内清单不一致。")
    for name, digest in files.items():
        if hashlib.sha256(payload[name]).hexdigest() != digest:
            raise UpdateError(f"包内文件校验失败：{name}。")
    return payload


def download_package(release: dict, timeout: float) -> dict[str, bytes]:
    base = f"https://github.com/{REPOSITORY}/releases/download/{release['tag']}"
    checksums = request_bytes(base + "/SHA256SUMS.txt", 64 * 1024, timeout).decode("utf-8")
    matches = re.findall(
        rf"(?m)^([a-fA-F0-9]{{64}})\s+\*?{re.escape(ARCHIVE_NAME)}\s*$", checksums
    )
    if len(matches) != 1:
        raise UpdateError("找不到唯一的 ZIP 校验值。")
    archive = request_bytes(base + "/" + ARCHIVE_NAME, MAX_DOWNLOAD, timeout)
    if hashlib.sha256(archive).hexdigest() != matches[0].lower():
        raise UpdateError("ZIP 的 SHA-256 校验失败，未修改安装目录。")
    return unpack_verified(archive, release["version"])


def managed_by_cc_switch(root: Path) -> bool:
    """只读检测常见主副本、软链接目标和数据库记录，不修改管理器。"""
    home = Path.home()
    config_dir = Path(os.environ.get("CC_SWITCH_CONFIG_DIR", str(home / ".cc-switch")))
    if root.parent == (config_dir / "skills").resolve():
        return True
    app_dirs = [home / name / "skills" for name in (
        ".agents", ".claude", ".codex", ".gemini", ".grok", ".hermes", ".minimax",
    )]
    app_dirs += [home / ".config/opencode/skills", home / ".pi/agent/skills"]
    if root.parent not in {path.resolve() for path in app_dirs}:
        return False
    database = config_dir / "cc-switch.db"
    if not database.is_file():
        return False
    try:
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            return connection.execute(
                "SELECT 1 FROM skills WHERE directory = ? LIMIT 1", (root.name,)
            ).fetchone() is not None
    except sqlite3.Error as error:
        raise UpdateError("无法确认 CC Switch 的管理状态，请使用管理器检查更新。") from error


def cache_directory(root: Path) -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    identity = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
    # Windows 商店应用可能把 AppData 重定向到另一块盘，先使用实际绝对路径。
    return (base / SKILL_NAME / "updater" / identity).resolve()


def check_local_files(root: Path, version: str, new_files: dict[str, bytes]) -> dict:
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise UpdateError("此副本没有安装包清单，请先从新版 Release ZIP 安装；未覆盖任何文件。")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    old_files = validate_manifest(manifest, version)
    for name, digest in old_files.items():
        local = root / safe_relative(name)
        if not local.is_file() or hashlib.sha256(local.read_bytes()).hexdigest() != digest:
            raise UpdateError(f"本地文件被修改或缺失：{name}；保留本地内容，停止自动更新。")
    for name, data in new_files.items():
        local = root / safe_relative(name)
        if name not in old_files and name != MANIFEST_NAME and local.exists():
            if not local.is_file() or local.read_bytes() != data:
                raise UpdateError(f"新增上游文件和本地文件冲突：{name}。")
    return old_files


def remove_temporary(path: Path, parent: Path, prefix: str) -> None:
    """删除前验证绝对路径，避免 Windows 上清理到安装目录之外。"""
    if path.resolve().parent != parent.resolve() or not path.name.startswith(prefix):
        raise UpdateError("临时目录范围校验失败，未清理。")
    if path.exists():
        shutil.rmtree(path)


def directory_snapshot(root: Path) -> dict[str, str]:
    snapshot = {}
    for path in root.rglob("*"):
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise UpdateError("安装目录内包含链接，请使用宿主管理器更新。")
        if path.is_file():
            snapshot[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def install_update(root: Path, version: str, payload: dict[str, bytes], cache: Path) -> Path:
    if cache.resolve().is_relative_to(root):
        raise UpdateError("缓存和备份不能放在 Skill 安装目录内。")
    snapshot = directory_snapshot(root)
    old_files = check_local_files(root, version, payload)
    backups = cache / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = backups / f"{version}-{timestamp}-{uuid4().hex[:8]}.zip"
    with zipfile.ZipFile(backup, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                package.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(backup) as package:
        if package.testzip() is not None:
            raise UpdateError("备份校验失败，未修改安装目录。")

    parent = root.parent.resolve()
    prefix = "." + SKILL_NAME + "-update-"
    staging = Path(tempfile.mkdtemp(prefix=prefix, dir=parent)).resolve()
    previous = parent / (prefix + "previous-" + uuid4().hex)
    switched = False
    try:
        shutil.copytree(root, staging, dirs_exist_ok=True)
        if directory_snapshot(staging) != snapshot:
            raise UpdateError("复制期间本地文件发生变化，停止更新。")
        # 只移除旧包明确拥有的文件；用户额外添加的文件保留。
        for name in set(old_files) - set(payload):
            (staging / safe_relative(name)).unlink()
        for name, data in payload.items():
            path = staging / safe_relative(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        new_version = read_version((staging / "SKILL.md").read_text(encoding="utf-8"))
        check_local_files(staging, new_version, payload)
        # 替换前再检查一次，发现编辑中的文件就停止。
        check_local_files(root, version, payload)
        if directory_snapshot(root) != snapshot:
            raise UpdateError("准备更新期间本地文件发生变化，停止更新。")
        root.rename(previous)
        try:
            staging.rename(root)
        except OSError:
            previous.rename(root)
            raise
        switched = True
    finally:
        remove_temporary(staging, parent, prefix)
        if switched:
            # 完整旧副本仍保留在已校验的 backup ZIP 中。
            try:
                remove_temporary(previous, parent, prefix)
            except OSError:
                # Windows 可能暂时占用旧目录；不把已经成功的更新说成失败。
                pass
    return backup


def run(root: Path, mode: str, timeout: float = 10, managed: bool = False) -> dict:
    root = root.resolve()
    current = read_version((root / "SKILL.md").read_text(encoding="utf-8"))
    result = {"current_version": current, "skill_dir": str(root)}
    cache = cache_directory(root)
    cache_file = cache / "last-check.json"
    release = None
    if mode == "auto" and cache_file.is_file():
        try:
            saved = json.loads(cache_file.read_text(encoding="utf-8"))
            age = time.time() - saved["checked_at"]
            if 0 <= age < CHECK_INTERVAL and saved["version"] == current:
                if saved["result"]["status"] == "update_available":
                    # --check 刚发现新版，随后 --auto 应执行更新，而不是等一天。
                    latest = saved["result"]["latest_version"]
                    version_tuple(latest)
                    release = {"version": latest, "tag": "v" + latest}
                else:
                    return {**result, "status": "cached", "last_result": saved["result"]}
        except (OSError, ValueError, KeyError, TypeError):
            pass
    release = release or latest_release(timeout)
    result["latest_version"] = release["version"]
    latest_number = version_tuple(release["version"])
    current_number = version_tuple(current, allow_prerelease=True)
    comparison = latest_number > current_number or (latest_number == current_number and "-" in current)
    if not comparison:
        result["status"] = "up_to_date" if release["version"] == current else "local_ahead"
    elif mode == "check":
        result["status"] = "update_available"
    elif (root / ".git").exists():
        result.update(status="development_checkout", message="开发仓库不会自动覆盖，请发布或通过 Git 更新。")
    elif managed or managed_by_cc_switch(root):
        result.update(status="managed_by_cc_switch", message="请在 CC Switch 中检查更新并更新此 Skill，由管理器同步各应用。")
    else:
        payload = download_package(release, timeout)
        cache.mkdir(parents=True, exist_ok=True)
        lock = cache / "update.lock"
        try:
            with lock.open("x", encoding="utf-8") as stream:
                stream.write(str(os.getpid()))
        except FileExistsError as error:
            raise UpdateError(f"已有更新锁：{lock}；确认没有更新进程后再处理。") from error
        try:
            backup = install_update(root, current, payload, cache)
        finally:
            lock.unlink()
        result.update(status="updated", current_version=release["version"], backup=str(backup))
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache / ("last-check-" + uuid4().hex + ".json")
    temporary.write_text(json.dumps({
        "checked_at": time.time(), "version": result["current_version"], "result": result,
    }, ensure_ascii=False), encoding="utf-8")
    temporary.replace(cache_file)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="立即检查，只报告版本，不修改 Skill。")
    group.add_argument("--update", action="store_true", help="立即检查并更新直接安装的副本。")
    group.add_argument("--auto", action="store_true", help="每 24 小时检查一次，有新版则更新。")
    parser.add_argument("--skill-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--managed", action="store_true", help="此副本由 CC Switch 管理，只提示管理器更新。")
    args = parser.parse_args()
    if not 0 < args.timeout <= 60:
        parser.error("timeout 必须大于 0 且不超过 60 秒。")
    mode = "auto" if args.auto else "update" if args.update else "check"
    try:
        result = run(args.skill_dir, mode, args.timeout, args.managed)
    except (UpdateError, OSError, ValueError, KeyError, TypeError, HTTPError, URLError, zipfile.BadZipFile) as error:
        result = {"status": "check_or_update_failed", "message": str(error)}
        print(json.dumps(result, ensure_ascii=True))
        return 0 if mode == "auto" else 1
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
