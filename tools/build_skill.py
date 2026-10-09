import hashlib
import json
import os
from pathlib import Path
import re
import stat
from urllib.parse import unquote, urlsplit
import zipfile

import yaml

root = Path.cwd().resolve()
if os.environ.get("GITHUB_REF_TYPE") == "tag":
    tag = os.environ["GITHUB_REF_NAME"]
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", tag):
        raise ValueError(f"Use a version tag such as v1.0.0: {tag}")

def read_skill(path):
    if path.is_symlink():
        raise ValueError(f"Symlinks are not packaged: {path}")
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---(?:\n|$)", text, re.DOTALL)
    if match is None:
        raise ValueError(f"Missing YAML frontmatter: {path}")
    metadata = yaml.safe_load(match.group(1))
    if not isinstance(metadata, dict):
        raise ValueError(f"Frontmatter must be a mapping: {path}")
    name = metadata.get("name")
    if (
        not isinstance(name, str) or len(name) > 64
        or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
    ):
        raise ValueError(f"Invalid Skill name: {path}")
    description = metadata.get("description")
    if (
        not isinstance(description, str) or not description.strip()
        or len(description) > 1024
    ):
        raise ValueError(f"Invalid Skill description: {path}")
    return metadata, text

modules = sorted((root / "modules").rglob("*.md"))
if not modules or (root / "modules").is_symlink():
    raise ValueError("The modules directory must contain Markdown files.")
sources = [root / "SKILL.md", *modules]
metadata, _ = read_skill(sources[0])
skill_name = metadata["name"]
version = metadata.get("metadata", {}).get("version")
if not isinstance(version, str) or not re.fullmatch(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?", version
):
    raise ValueError("metadata.version must be a version such as 1.1.0 or 1.1.0-rc.1.")
if os.environ.get("GITHUB_REF_TYPE") == "tag" and os.environ["GITHUB_REF_NAME"] != "v" + version:
    raise ValueError("Release tag must match SKILL.md metadata.version.")
files = {}
for source in sources:
    _, text = read_skill(source)
    if not source.resolve().is_relative_to(root):
        raise ValueError(f"Source escapes the repository: {source}")
    relative = source.relative_to(root).as_posix()
    files[relative] = text.encode("utf-8")

# Only ship runtime resources; build tools and tests stay in the repository.
resources = [root / "scripts/update_skill.py", *sorted((root / "references").rglob("*.md"))]
if not (root / "references/updates.md").is_file():
    raise ValueError("Missing references/updates.md.")
for source in resources:
    if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(root):
        raise ValueError(f"Invalid runtime resource: {source}")
    text = source.read_text(encoding="utf-8")
    if source.suffix == ".py":
        compile(text, str(source), "exec")
    files[source.relative_to(root).as_posix()] = text.encode("utf-8")

manifest = {
    "schema_version": 1,
    "name": skill_name,
    "version": version,
    "files": {relative: hashlib.sha256(content).hexdigest() for relative, content in sorted(files.items())},
}
files["package-manifest.json"] = (
    json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
).encode("utf-8")

# Validate document links outside fenced code; every local target
# must be included in the package, not merely exist in the checkout.
for relative, content in files.items():
    if not relative.endswith(".md"):
        continue
    text = content.decode("utf-8")
    prose = re.sub(r"(?ms)^\x60\x60\x60.*?^\x60\x60\x60\s*$", "", text)
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", prose):
        target = target.strip().strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        destination = (
            root / relative
        ).parent.joinpath(unquote(parsed.path)).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"Link escapes the package: {relative}: {target}")
        linked_file = destination.relative_to(root).as_posix()
        if linked_file not in files:
            raise ValueError(f"Missing packaged link target: {relative}: {target}")

output_dir = Path(os.environ.get("BUILD_OUTPUT_DIR", "dist")).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
archive = output_dir / f"{skill_name}.zip"
expected = {}
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
    for relative, content in sorted(files.items()):
        member = f"{skill_name}/{relative}"
        info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        package.writestr(info, content, compresslevel=9)
        expected[member] = content

with zipfile.ZipFile(archive) as package:
    if package.testzip() is not None or set(package.namelist()) != set(expected):
        raise ValueError("Invalid ZIP contents.")
    for member, content in expected.items():
        if package.read(member) != content:
            raise ValueError(f"ZIP content mismatch: {member}")
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
checksum = output_dir / "SHA256SUMS.txt"
checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
print(f"Validated Skill {skill_name} {version}: {len(files)} packaged files.")
print(f"Built {archive.name}: {archive.stat().st_size} bytes")
print(f"SHA256: {digest}")
