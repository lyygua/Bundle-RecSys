#!/usr/bin/env python3
"""Audit tracked and unignored files before committing or publishing."""

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BLOCKED_DIRS = {
    "dataset", "data", "saved", "log", "logs", "log_tensorboard", "outputs",
    "results", "wandb", ".venv", "venv", "__pycache__", ".pytest_cache",
}
DATA_SUFFIXES = {".inter", ".item", ".user", ".kg", ".link"}
SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{40,}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
]


def main():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    )
    paths = sorted(set(p for p in result.stdout.decode().split("\0") if p))
    failures = []
    total = 0
    for name in paths:
        path = Path(name)
        parts = path.parts
        fixture = len(parts) > 1 and parts[0] == "tests"
        if parts[0] in BLOCKED_DIRS or name.startswith("recbole/dataset_example/"):
            failures.append(f"Excluded directory: {name}")
        if path.suffix in DATA_SUFFIXES and not fixture:
            failures.append(f"Dataset outside test fixtures: {name}")
        if name.endswith((".jsonl", ".jsonl.gz", ".pth", ".pt", ".ckpt", ".log", ".pyc")):
            failures.append(f"Runtime artifact: {name}")
        if name.endswith("_top20_top50.md"):
            failures.append(f"Recommendation export: {name}")
        if path.name.startswith(".env") and path.name != ".env.example":
            failures.append(f"Environment file: {name}")
        source = ROOT / name
        if source.is_symlink():
            failures.append(f"Symlink: {name}")
            continue
        if not source.is_file():
            failures.append(f"Missing file: {name}")
            continue
        data = source.read_bytes()
        total += len(data)
        if len(data) > 10 * 1024 * 1024:
            failures.append(f"File larger than 10 MiB: {name}")
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            failures.append(f"Possible credential: {name}")
        if ("/home/" + "ubuntu/") in content:
            failures.append(f"Machine-specific absolute path: {name}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"PASS: {len(paths)} files, {total / 1024 / 1024:.2f} MiB; no real datasets, "
          "weights, logs, recommendation exports, machine paths or detected credentials.")
    print("Small test fixtures under tests/ are allowed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
