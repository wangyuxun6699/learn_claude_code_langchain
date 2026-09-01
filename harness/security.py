"""统一、大小写不敏感、可读的命令权限策略。

背景：shell=True 是把模型输出直接交给 shell，等价于把 shell 的权限下放给
模型（以及它能读到的任意内容）。下面的 deny-list 只是教学意义上的启发式，
它减少误操作，但绝不是安全边界。真正的隔离需要沙箱/容器/独立进程 +
默认拒绝的权限中间件。
"""
from __future__ import annotations

import re

DANGEROUS_PHRASES: list[str] = [
    "rm -rf /",
    "rm -fr /",
    "rm -rf --no-preserve-root",
    "mkfs",
    "dd if=",
    "dd if =",
    "> /dev/sd",
    ">/dev/sd",
    "> /dev/hd",
    "> /dev/disk",
    "shutdown -h",
    "shutdown -r",
    "shutdown -p",
    "systemctl poweroff",
    "systemctl reboot",
    "systemctl halt",
    "init 0",
    "init 6",
    "poweroff --",
    "reboot --",
    "chmod 777 /",
    "chmod -r 777 /",
    ":(){:|:&};:",
    ":(){ :|:& };:",
]

DANGEROUS_WORDS: list[str] = [
    "sudo",
    "su",
    "doas",
    "shutdown",
    "reboot",
    "poweroff",
    "halt",
]

CONFIRM_PHRASES: list[str] = [
    "rm ",
    "del ",
    "> /etc/",
    "chmod 777",
    "chmod -r",
]


def normalize(command) -> str:
    """小写并合并连续空白，让大小写与多空格变体失效。"""
    return re.sub(r"\s+", " ", str(command or "")).strip().lower()


def _segment_leading_words(command) -> set:
    """返回位于管道段开头的单词集合，用于短危险词的词边界匹配。"""
    text = str(command or "").lower()
    return set(re.findall(r"(?:^|[;&|()]\s*)([a-z][a-z0-9_-]*)", text))


def check_deny_list(command) -> str | None:
    """返回禁止原因；命令安全时返回 None。"""
    normalized = normalize(command)
    for phrase in DANGEROUS_PHRASES:
        if phrase in normalized:
            return f"'{phrase}' is on the deny list"
    leading = _segment_leading_words(command)
    for word in DANGEROUS_WORDS:
        if word in leading:
            return f"'{word}' is on the deny list"
    return None


_BASH_TOOL_NAMES = {"bash", "run_bash", "execute_bash", "run_shell"}


def check_confirmation(tool_name: str, args: dict) -> str | None:
    """返回需要用户确认的原因；无需确认时返回 None。"""
    if tool_name in _BASH_TOOL_NAMES:
        command = str(args.get("command", ""))
        normalized = normalize(command)
        if normalized.strip().startswith("del "):
            return "potentially destructive command: del"
        for phrase in CONFIRM_PHRASES:
            if phrase in normalized:
                return f"potentially destructive command: {phrase.strip()}"
    return None


def check_permission(
    tool_name: str,
    args: dict,
    *,
    ask=None,
) -> bool:
    """统一权限入口：硬拒绝优先，其次按需请求用户确认（默认拒绝）。"""
    command = str(args.get("command", "")) if tool_name in _BASH_TOOL_NAMES else ""
    if check_deny_list(command):
        return False
    reason = check_confirmation(tool_name, args)
    if reason and ask is not None:
        return ask(tool_name, args, reason)
    return reason is None
