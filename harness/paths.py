# -*- coding: utf-8 -*-
"""工作区路径解析与穿越防护（显式传入 workdir，避免全局耦合）。"""
from __future__ import annotations

from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def get_workdir() -> Path:
    """返回规范化的当前工作目录（真实绝对路径）。"""
    return Path.cwd().resolve()


def resolve_path(workdir, raw_path: PathLike) -> Path:
    """相对路径解析到 workdir；绝对路径保留其绝对含义。"""
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(workdir) / candidate).resolve()


def is_within_workspace(workdir, path: PathLike) -> bool:
    """判断规范化后的路径是否仍落在工作区内。"""
    try:
        return resolve_path(workdir, path).is_relative_to(Path(workdir).resolve())
    except (OSError, ValueError):
        return False


def safe_path(workdir, raw_path: PathLike) -> Path:
    """解析并强制路径落在工作区内，否则抛 ValueError。"""
    resolved = resolve_path(workdir, raw_path)
    if not resolved.is_relative_to(Path(workdir).resolve()):
        raise ValueError(f"path escapes workspace: {raw_path}")
    return resolved
