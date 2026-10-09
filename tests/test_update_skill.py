"""使用临时安装目录和模拟下载测试更新，不接触真实 Agent 的 Skill。"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from urllib.error import URLError
import zipfile


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/update_skill.py"
SPEC = importlib.util.spec_from_file_location("update_skill", MODULE_PATH)
updater = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(updater)


def make_payload(version, extras=None):
    skill = f'---\nname: {updater.SKILL_NAME}\nmetadata:\n  version: "{version}"\n---\n'
    files = {
        "SKILL.md": skill.encode(),
        "scripts/update_skill.py": b"# updater fixture\n",
        "references/updates.md": b"# update instructions\n",
        "modules/core.md": ("# module " + version + "\n").encode(),
        **(extras or {}),
    }
    manifest = {
        "schema_version": 1, "name": updater.SKILL_NAME, "version": version,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    files[updater.MANIFEST_NAME] = json.dumps(manifest).encode()
    return files


def archive_bytes(payload):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as package:
        for name, data in payload.items():
            package.writestr(updater.SKILL_NAME + "/" + name, data)
    return stream.getvalue()


class UpdaterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pytorch-skill-test-")
        self.parent = Path(self.temporary.name).resolve()
        self.root = self.parent / "installed skill"
        self.cache = self.parent / "cache"
        self.root.mkdir()
        self.old = make_payload("1.1.0", {"modules/removed.md": b"old owned module"})
        self.new = make_payload("1.2.0")
        for name, data in self.old.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def tearDown(self):
        # TemporaryDirectory 只清理本测试创建的确定目录。
        self.assertTrue(self.parent.name.startswith("pytorch-skill-test-"))
        self.assertEqual(Path(self.temporary.name).resolve(), self.parent)
        self.temporary.cleanup()

    def test_versions_are_compared_numerically_and_prereleases_rejected(self):
        self.assertGreater(updater.version_tuple("1.10.0"), updater.version_tuple("1.2.0"))
        self.assertEqual(updater.read_version(self.old["SKILL.md"].decode()), "1.1.0")
        for version in ("1.2.0-rc.1", "v1.2.0", "01.2.0", "../1.2.0"):
            with self.assertRaises(updater.UpdateError):
                updater.version_tuple(version)

    def test_valid_package_round_trip(self):
        self.assertEqual(updater.unpack_verified(archive_bytes(self.new), "1.2.0"), self.new)

    def test_tampered_file_and_wrong_release_version_are_rejected(self):
        changed = {**self.new, "modules/core.md": b"tampered"}
        with self.assertRaises(updater.UpdateError):
            updater.unpack_verified(archive_bytes(changed), "1.2.0")
        with self.assertRaises(updater.UpdateError):
            updater.unpack_verified(archive_bytes(self.new), "1.3.0")

    def test_unsafe_archive_paths_are_rejected(self):
        for name in ("modules/../escape.md", "/absolute.md", "modules/a:b.md", "modules\\file.md"):
            with self.subTest(name=name), self.assertRaises(updater.UpdateError):
                updater.unpack_verified(archive_bytes({**self.new, name: b"bad"}), "1.2.0")

    def test_duplicate_case_and_symlink_are_rejected(self):
        with self.assertRaises(updater.UpdateError):
            updater.unpack_verified(archive_bytes({**self.new, "modules/CORE.md": b"duplicate"}), "1.2.0")
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as package:
            entry = zipfile.ZipInfo(updater.SKILL_NAME + "/modules/link.md")
            entry.create_system = 3
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            package.writestr(entry, "elsewhere")
        with self.assertRaises(updater.UpdateError):
            updater.unpack_verified(stream.getvalue(), "1.2.0")

    def test_zip_checksum_failure_prevents_install(self):
        checksum = ("0" * 64 + "  " + updater.ARCHIVE_NAME + "\n").encode()
        with mock.patch.object(updater, "request_bytes", side_effect=[checksum, archive_bytes(self.new)]):
            with self.assertRaises(updater.UpdateError):
                updater.download_package({"version": "1.2.0", "tag": "v1.2.0"}, 10)
        self.assertEqual((self.root / "SKILL.md").read_bytes(), self.old["SKILL.md"])

    def test_install_preserves_extras_removes_obsolete_owned_file_and_backs_up(self):
        (self.root / "my-notes.txt").write_text("keep me", encoding="utf-8")
        backup = updater.install_update(self.root, "1.1.0", self.new, self.cache)
        self.assertEqual(updater.read_version((self.root / "SKILL.md").read_text()), "1.2.0")
        self.assertEqual((self.root / "my-notes.txt").read_text(), "keep me")
        self.assertFalse((self.root / "modules/removed.md").exists())
        with zipfile.ZipFile(backup) as package:
            self.assertEqual(package.read("SKILL.md"), self.old["SKILL.md"])
            self.assertEqual(package.read("my-notes.txt"), b"keep me")
        self.assertFalse(any(path.name.startswith("." + updater.SKILL_NAME + "-update-") for path in self.parent.iterdir()))

    def test_local_edits_are_preserved_and_stop_update(self):
        changed = self.root / "modules/core.md"
        changed.write_bytes(b"my edits")
        with self.assertRaises(updater.UpdateError):
            updater.install_update(self.root, "1.1.0", self.new, self.cache)
        self.assertEqual(changed.read_bytes(), b"my edits")
        self.assertEqual((self.root / "SKILL.md").read_bytes(), self.old["SKILL.md"])

    def test_new_upstream_file_cannot_overwrite_user_file(self):
        (self.root / "modules/local.md").write_bytes(b"my extra file")
        with self.assertRaises(updater.UpdateError):
            updater.install_update(self.root, "1.1.0", make_payload("1.2.0", {"modules/local.md": b"upstream"}), self.cache)
        self.assertEqual((self.root / "modules/local.md").read_bytes(), b"my extra file")

    def test_failed_directory_switch_restores_original(self):
        original_rename = Path.rename

        def fail_staging_switch(path, destination):
            if path.name.startswith("." + updater.SKILL_NAME + "-update-") and "previous-" not in path.name:
                raise PermissionError("simulated locked destination")
            return original_rename(path, destination)

        with mock.patch.object(Path, "rename", fail_staging_switch), self.assertRaises(PermissionError):
            updater.install_update(self.root, "1.1.0", self.new, self.cache)
        self.assertEqual(updater.directory_snapshot(self.root), {
            name: hashlib.sha256(data).hexdigest() for name, data in self.old.items()
        })

    def test_cleanup_rejects_non_temporary_directory(self):
        with self.assertRaises(updater.UpdateError):
            updater.remove_temporary(self.root, self.parent, "." + updater.SKILL_NAME + "-update-")
        self.assertTrue(self.root.is_dir())

    def test_development_and_managed_installs_are_not_overwritten(self):
        release = {"version": "1.2.0", "tag": "v1.2.0"}
        with mock.patch.object(updater, "cache_directory", return_value=self.cache), mock.patch.object(updater, "latest_release", return_value=release), mock.patch.object(updater, "download_package") as download:
            (self.root / ".git").mkdir()
            self.assertEqual(updater.run(self.root, "update")["status"], "development_checkout")
            (self.root / ".git").rmdir()
            self.assertEqual(updater.run(self.root, "update", managed=True)["status"], "managed_by_cc_switch")
            download.assert_not_called()
        self.assertEqual((self.root / "SKILL.md").read_bytes(), self.old["SKILL.md"])

    def test_cc_switch_source_and_copied_app_directory_are_detected_read_only(self):
        home = self.parent / "home"
        master = home / ".cc-switch/skills" / updater.SKILL_NAME
        installed = home / ".codex/skills" / updater.SKILL_NAME
        master.mkdir(parents=True)
        installed.mkdir(parents=True)
        database = home / ".cc-switch/cc-switch.db"
        with contextlib.closing(sqlite3.connect(database)) as connection, connection:
            connection.execute("CREATE TABLE skills (directory TEXT)")
            connection.execute("INSERT INTO skills VALUES (?)", (updater.SKILL_NAME,))
        before = database.read_bytes()
        with mock.patch.object(Path, "home", return_value=home), mock.patch.dict("os.environ", {"CC_SWITCH_CONFIG_DIR": str(home / ".cc-switch")}):
            self.assertTrue(updater.managed_by_cc_switch(master.resolve()))
            self.assertTrue(updater.managed_by_cc_switch(installed.resolve()))
            self.assertFalse(updater.managed_by_cc_switch(self.root))
        self.assertEqual(database.read_bytes(), before)

    def test_check_then_auto_updates_without_a_second_release_query(self):
        release = {"version": "1.2.0", "tag": "v1.2.0"}
        with mock.patch.object(updater, "cache_directory", return_value=self.cache), mock.patch.object(updater, "latest_release", return_value=release) as fetch, mock.patch.object(updater, "managed_by_cc_switch", return_value=False), mock.patch.object(updater, "download_package", return_value=self.new) as download:
            self.assertEqual(updater.run(self.root, "check")["status"], "update_available")
            self.assertEqual(updater.run(self.root, "auto")["status"], "updated")
            self.assertEqual(updater.run(self.root, "auto")["status"], "cached")
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(download.call_count, 1)

    def test_no_downgrade(self):
        release = {"version": "1.0.1", "tag": "v1.0.1"}
        with mock.patch.object(updater, "cache_directory", return_value=self.cache), mock.patch.object(updater, "latest_release", return_value=release), mock.patch.object(updater, "download_package") as download:
            self.assertEqual(updater.run(self.root, "update")["status"], "local_ahead")
            download.assert_not_called()

    def test_installed_prerelease_can_update_to_same_stable_version(self):
        beta = make_payload("1.2.0-rc.1")
        for name, data in beta.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        release = {"version": "1.2.0", "tag": "v1.2.0"}
        with mock.patch.object(updater, "cache_directory", return_value=self.cache), mock.patch.object(updater, "latest_release", return_value=release):
            self.assertEqual(updater.run(self.root, "check")["status"], "update_available")

    def test_auto_network_failure_is_reported_without_blocking_task(self):
        with mock.patch("sys.argv", [str(MODULE_PATH), "--auto", "--skill-dir", str(self.root)]), mock.patch.object(updater, "cache_directory", return_value=self.cache), mock.patch.object(updater, "latest_release", side_effect=URLError("offline")), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(updater.main(), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "check_or_update_failed")
        self.assertEqual((self.root / "SKILL.md").read_bytes(), self.old["SKILL.md"])

    def test_real_release_package_can_update_its_own_loaded_script(self):
        repository = MODULE_PATH.parents[1]
        output = self.parent / "build-output"
        environment = {**os.environ, "BUILD_OUTPUT_DIR": str(output), "GITHUB_REF_TYPE": "branch"}
        build = subprocess.run(
            [sys.executable, str(repository / "tools/build_skill.py")],
            cwd=repository, env=environment, capture_output=True, text=True,
        )
        self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
        version = updater.read_version((repository / "SKILL.md").read_text(encoding="utf-8"))
        package_path = output / updater.ARCHIVE_NAME
        payload = updater.unpack_verified(package_path.read_bytes(), version)
        installed = self.parent / "real installed skill"
        for name, data in payload.items():
            path = installed / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (installed / "personal-notes.txt").write_bytes(b"preserved")

        major, minor, patch = updater.version_tuple(version, allow_prerelease=True)
        newer = f"{major}.{minor}.{patch + 1}"
        future = {name: data for name, data in payload.items() if name != updater.MANIFEST_NAME}
        future["SKILL.md"] = future["SKILL.md"].replace(
            f'version: "{version}"'.encode(), f'version: "{newer}"'.encode(),
        )
        manifest = {
            "schema_version": 1, "name": updater.SKILL_NAME, "version": newer,
            "files": {name: hashlib.sha256(data).hexdigest() for name, data in future.items()},
        }
        future[updater.MANIFEST_NAME] = json.dumps(manifest).encode()
        archive = archive_bytes(future)
        checksum = (hashlib.sha256(archive).hexdigest() + "  " + updater.ARCHIVE_NAME + "\n").encode()
        api = json.dumps({
            "tag_name": "v" + newer, "draft": False, "prerelease": False,
            "assets": [{"name": updater.ARCHIVE_NAME}, {"name": "SHA256SUMS.txt"}],
        }).encode()
        spec = importlib.util.spec_from_file_location("installed_updater", installed / "scripts/update_skill.py")
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        with mock.patch.object(loaded, "cache_directory", return_value=self.cache), mock.patch.object(loaded, "request_bytes", side_effect=[api, checksum, archive]), mock.patch.object(loaded, "managed_by_cc_switch", return_value=False):
            result = loaded.run(installed, "auto")
        self.assertEqual(result["status"], "updated")
        self.assertEqual(result["current_version"], newer)
        self.assertEqual((installed / "personal-notes.txt").read_bytes(), b"preserved")
        self.assertEqual(loaded.read_version((installed / "SKILL.md").read_text(encoding="utf-8")), newer)
        with zipfile.ZipFile(result["backup"]) as backup:
            self.assertEqual(backup.read("scripts/update_skill.py"), payload["scripts/update_skill.py"])

    def test_release_tag_must_match_package_version(self):
        repository = MODULE_PATH.parents[1]
        environment = {
            **os.environ, "BUILD_OUTPUT_DIR": str(self.parent / "mismatched-build"),
            "GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "v999.0.0",
        }
        build = subprocess.run(
            [sys.executable, str(repository / "tools/build_skill.py")],
            cwd=repository, env=environment, capture_output=True, text=True,
        )
        self.assertNotEqual(build.returncode, 0)
        self.assertIn("Release tag must match", build.stderr)


if __name__ == "__main__":
    unittest.main()
