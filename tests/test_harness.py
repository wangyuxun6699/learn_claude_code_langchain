"""harness 公共内核单元测试（纯标准库，无需 LLM / 网络）。"""
import tempfile
from pathlib import Path

import pytest

from harness.io import run_bash, run_read, run_write, truncate
from harness.paths import is_within_workspace, resolve_path, safe_path
from harness.security import check_deny_list, normalize


def test_normalize_collapses_case_and_space():
    assert normalize("  SUDO   Apt  ") == "sudo apt"


def test_deny_list_is_case_insensitive():
    assert check_deny_list("sudo apt update") is not None
    assert check_deny_list("SUDO apt update") is not None
    assert check_deny_list("rm -rf /*") is not None
    assert check_deny_list("dd if=/dev/sda of=/dev/sdb") is not None


def test_deny_list_word_boundary_avoids_false_positive():
    assert check_deny_list("echo pseudo") is None
    assert check_deny_list("echo resume the task") is None
    assert check_deny_list("ls -la") is None


def test_safe_path_blocks_traversal():
    with tempfile.TemporaryDirectory() as d:
        wd = Path(d)
        assert safe_path(wd, "notes.txt") == (wd / "notes.txt").resolve()
        with pytest.raises(ValueError):
            safe_path(wd, "../etc/passwd")


def test_resolve_and_within():
    with tempfile.TemporaryDirectory() as d:
        wd = Path(d).resolve()
        assert is_within_workspace(wd, "a/b.txt")
        assert not is_within_workspace(wd, "../outside.txt")
        assert resolve_path(wd, "a/b.txt").parent == wd / "a"


def test_truncate_adds_marker():
    assert truncate("hello", limit=3) == "hel...(truncated)"
    assert truncate("hi", limit=10) == "hi"


def test_read_write_roundtrip_utf8():
    with tempfile.TemporaryDirectory() as d:
        wd = Path(d)
        run_write(wd, "子目录/笔记.txt", "你好，世界")
        assert "你好，世界" in run_read(wd, "子目录/笔记.txt")


def test_run_bash_echo():
    assert run_bash(Path.cwd(), "echo hello").strip() == "hello"


def test_run_bash_enforces_deny_list():
    assert run_bash(Path.cwd(), "sudo id").startswith("Blocked:")
