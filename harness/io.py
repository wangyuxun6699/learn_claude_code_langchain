"""统一编码、截断标记、超时与返回码的文件 / shell 工具。"""
from __future__ import annotations

import glob as _glob
import subprocess
from pathlib import Path

from harness import paths
from harness.security import check_deny_list

UTF8 = "utf-8"
DEFAULT_TIMEOUT = 120
MAX_OUTPUT = 50_000
MAX_GLOB_RESULTS = 2_000


def truncate(text: str, limit: int = MAX_OUTPUT, marker: str = "...(truncated)") -> str:
    """硬截断并追加标记；未超限时原样返回。"""
    if len(text) <= limit:
        return text
    return text[:limit] + marker


def read_text_file(path: Path) -> str:
    """以 UTF-8 读取文本，无法解码的字节用替换符，不抛异常。"""
    return path.read_text(encoding=UTF8, errors="replace")


def run_bash(workdir, command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """执行 shell 命令，统一编码 / 返回码 / 超时 / 截断。"""
    denied = check_deny_list(command)
    if denied:
        return f"Blocked: {denied}"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=Path(workdir),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout} seconds"
    except OSError as exc:
        return f"Error: {exc}"
    output = truncate((result.stdout + result.stderr).strip()) or "(no output)"
    if result.returncode != 0:
        output = f"Exit code: {result.returncode}\n{output}"
    return output


def run_read(workdir, raw_path: paths.PathLike, limit: int | None = None) -> str:
    """读取工作区内的 UTF-8 文本文件，可选限制返回行数。"""
    try:
        lines = read_text_file(paths.safe_path(workdir, raw_path)).splitlines()
    except (OSError, ValueError) as exc:
        return f"Error: {exc}"
    if limit is not None and 0 <= limit < len(lines):
        lines = [*lines[:limit], f"...({len(lines) - limit} more lines)"]
    return "\n".join(lines)


def run_write(workdir, raw_path: paths.PathLike, content: str) -> str:
    """向工作区内文件写入 UTF-8 文本（覆盖写）。"""
    try:
        file_path = paths.safe_path(workdir, raw_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding=UTF8, newline="")
        return f"Wrote {len(content.encode(UTF8))} bytes to {raw_path}"
    except (OSError, ValueError) as exc:
        return f"Error: {exc}"


def run_edit(
    workdir, raw_path: paths.PathLike, old_text: str, new_text: str
) -> str:
    """替换文件中首次出现的 old_text。"""
    try:
        file_path = paths.safe_path(workdir, raw_path)
        current = read_text_file(file_path)
        if old_text not in current:
            return f"Error: old_text was not found in {raw_path}"
        file_path.write_text(
            current.replace(old_text, new_text, 1), encoding=UTF8, newline=""
        )
        return f"Edited {raw_path}"
    except (OSError, ValueError) as exc:
        return f"Error: {exc}"


def run_glob(workdir, pattern: str, max_results: int = MAX_GLOB_RESULTS) -> str:
    """在工作区内按 glob 查找文件，结果有数量上限。"""
    try:
        results: list = []
        for match in _glob.glob(pattern, root_dir=Path(workdir), recursive=True):
            if paths.is_within_workspace(workdir, match):
                results.append(Path(match).as_posix())
            if len(results) >= max_results:
                results.append(f"...({max_results}+ matches, truncated)")
                break
        return "\n".join(sorted(results)) if results else "(no matches)"
    except (OSError, ValueError) as exc:
        return f"Error: {exc}"
