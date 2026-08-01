"""Local filesystem helpers for Colab experiment artifacts."""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

EXCLUDED_SOURCE_DIRS = {
    ".git",
    ".venv",
    ".ruff_cache",
    ".pytest_cache",
    ".vscode",
    "__pycache__",
    "outputs",
    "results",
    "multirun",
    "wandb",
    "dist",
    "build",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".npy"}
EXCLUDED_SOURCE_FILES = {".env"}


def make_source_archive(repo_root: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    root = repo_root.resolve()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            dirs[:] = [name for name in dirs if name not in EXCLUDED_SOURCE_DIRS]
            for filename in files:
                if filename in EXCLUDED_SOURCE_FILES:
                    continue
                path = current_path / filename
                rel = path.relative_to(root)
                if any(part in EXCLUDED_SOURCE_DIRS for part in rel.parts):
                    continue
                if path.suffix in EXCLUDED_SUFFIXES:
                    continue
                archive.write(path, rel.as_posix())
    return destination


def unpack_archive(archive_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive_path), str(destination))
    return destination
