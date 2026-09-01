# -*- coding: utf-8 -*-
"""harness —— 各章节共用的安全 / 路径 / IO 内核。

本包把 s01~s17 中反复复制粘贴的基础设施抽成单一实现，避免：
- 同一个编码 bug、黑名单漏洞在章节间修复程度不一致；
- 想改一个工具的签名要同步十几处。

所有模块只依赖 Python 标准库，因此任何章节都能安全导入。
"""
from harness import config, io, paths, security
from harness.io import run_bash, run_edit, run_glob, run_read, run_write, truncate
from harness.paths import is_within_workspace, resolve_path, safe_path
from harness.security import (
    CONFIRM_PHRASES,
    DANGEROUS_PHRASES,
    DANGEROUS_WORDS,
    check_deny_list,
    check_permission,
)

__all__ = [
    "config",
    "io",
    "paths",
    "security",
    "run_bash",
    "run_edit",
    "run_glob",
    "run_read",
    "run_write",
    "truncate",
    "is_within_workspace",
    "resolve_path",
    "safe_path",
    "CONFIRM_PHRASES",
    "DANGEROUS_PHRASES",
    "DANGEROUS_WORDS",
    "check_deny_list",
    "check_permission",
]
