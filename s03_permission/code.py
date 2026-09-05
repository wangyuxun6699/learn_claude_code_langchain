#!/usr/bin/env python3
"""s03_permission.py - 用 middleware 在工具执行前完成权限审核。

本章完整保留 s02 的五个工具和 ``create_agent`` 循环，只新增三道权限闸门：

1. 硬拒绝列表：危险命令直接拒绝；
2. 规则匹配：识别工作区外访问和破坏性命令；
3. 用户审批：规则命中后默认拒绝，只有明确确认才执行。

``PermissionMiddleware`` 把权限管线包在所有工具调用外层。拒绝时返回合法的
``ToolMessage``，因此模型仍能看到本次工具调用的结果并继续推理。

Usage:
    pip install -r requirements.txt
    OPENAI_API_KEY=... BASE_URL=... MODEL_ID=... python s03_permission/code.py
"""

import glob as glob_module
import os
import re
import subprocess
from pathlib import Path
from threading import Lock

try:
    import readline

    # 修复 macOS libedit 下中文退格等 UTF-8 输入问题。
    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    pass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain.messages import AIMessageChunk, ToolMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

WORKDIR = Path.cwd().resolve()
MODEL = os.environ["MODEL_ID"]
SYSTEM = f"You are a coding agent at {WORKDIR}. All destructive operations require user approval."


# -- From s02: five focused tools --


def resolve_path(path: str) -> Path:
    """把相对路径解析到工作目录；越界访问由权限 middleware 审核。"""
    return (WORKDIR / path).resolve()


def print_tool_result(name: str, detail: str, output: str) -> str:
    """显示已获准的工具调用和结果预览，同时把完整结果返回给 Agent。"""
    print(f"\n\033[33m> {name}({detail})\033[0m")
    print(output[:200])
    return output


@tool
def bash(command: str) -> str:
    """Run a shell command in the workspace and return its combined output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        output = output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        output = "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as error:
        output = f"Error: {error}"

    return print_tool_result("bash", command, output)


@tool
def read_file(path: str, limit: int | None = None) -> str:
    """Read a UTF-8 text file, optionally limiting the number of returned lines."""
    try:
        if limit is not None and limit < 1:
            raise ValueError("limit must be a positive integer")
        lines = resolve_path(path).read_text(encoding="utf-8").splitlines()
        if limit is not None and limit < len(lines):
            omitted = len(lines) - limit
            lines = lines[:limit] + [f"... ({omitted} more lines)"]
        output = "\n".join(lines)
    except (OSError, UnicodeError, ValueError) as error:
        output = f"Error: {error}"
    return print_tool_result("read_file", path, output)


@tool
def write_file(path: str, content: str) -> str:
    """Write UTF-8 text to a file, replacing it and creating parent directories."""
    try:
        file_path = resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8", newline="")
        output = f"Wrote {len(content)} characters to {path}"
    except (OSError, UnicodeError, ValueError) as error:
        output = f"Error: {error}"
    return print_tool_result("write_file", path, output)


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace the first exact occurrence of old_text in a UTF-8 file."""
    try:
        file_path = resolve_path(path)
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            output = f"Error: text not found in {path}"
        else:
            file_path.write_text(
                text.replace(old_text, new_text, 1),
                encoding="utf-8",
                newline="",
            )
            output = f"Edited {path}"
    except (OSError, UnicodeError, ValueError) as error:
        output = f"Error: {error}"
    return print_tool_result("edit_file", path, output)


@tool
def glob(pattern: str) -> str:
    """Find workspace files matching a glob pattern; use ** for recursive matching."""
    try:
        matches = set()
        for match in glob_module.glob(pattern, root_dir=WORKDIR, recursive=True):
            resolved = (WORKDIR / match).resolve()
            if resolved.is_relative_to(WORKDIR) and resolved.is_file():
                matches.add(resolved.relative_to(WORKDIR).as_posix())
        shown = sorted(matches)[:200]
        if len(matches) > 200:
            shown.append("... (more matches omitted; narrow the pattern)")
        output = "\n".join(shown) if shown else "(no matches)"
    except (OSError, ValueError) as error:
        output = f"Error: {error}"
    return print_tool_result("glob", pattern, output)


TOOLS = [bash, read_file, write_file, edit_file, glob]


# -- New in s03: three-gate permission pipeline --


# Gate 1: commands containing these fragments can never be approved.
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]


def check_deny_list(command: str) -> str | None:
    """返回硬拒绝原因；未命中时返回 ``None``。"""
    normalized = command.lower()
    for pattern in DENY_LIST:
        if pattern in normalized:
            return f"Blocked: '{pattern}' is on the deny list"
    return None


# Gate 2: rules identify calls that require an explicit user decision.
DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)


def contains_destructive_command(command: str) -> bool:
    """识别位于命令段开头的 ``rm`` / ``del``，避免误判 model、delimiter。"""
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))


PERMISSION_RULES = [
    {
        "tools": ["read_file", "write_file", "edit_file"],
        "check": lambda args: not resolve_path(args.get("path", "")).is_relative_to(WORKDIR),
        "message": "Access outside workspace",
    },
    {
        "tools": ["bash"],
        "check": lambda args: contains_destructive_command(args.get("command", ""))
        or any(
            keyword in args.get("command", "").lower()
            for keyword in ["rm ", "> /etc/", "chmod 777"]
        ),
        "message": "Potentially destructive command",
    },
]


def check_rules(tool_name: str, args: dict) -> str | None:
    """返回第一条命中的审批规则原因；无需审批时返回 ``None``。"""
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None


def ask_user(tool_name: str, args: dict, reason: str) -> str:
    """显示待审核调用；只有 y/yes 明确允许，其余输入均拒绝。"""
    print(f"\n\033[33m[permission] {reason}\033[0m")
    print(f"   Tool: {tool_name}({args})")
    try:
        choice = input("   Allow? [y/N] ").strip().lower()
    except EOFError:
        choice = ""
    return "allow" if choice in ("y", "yes") else "deny"


# 同一轮的工具可能并发执行；锁保证多个审批提示不会争用终端输入。
APPROVAL_LOCK = Lock()


def check_permission(tool_name: str, args: dict) -> bool:
    """依次执行硬拒绝、规则匹配和用户审批三道闸门。"""
    if tool_name == "bash":
        reason = check_deny_list(args.get("command", ""))
        if reason:
            print(f"\n\033[31m[blocked] {reason}\033[0m")
            return False

    reason = check_rules(tool_name, args)
    if reason:
        with APPROVAL_LOCK:
            return ask_user(tool_name, args, reason) == "allow"
    return True


class PermissionMiddleware(AgentMiddleware):
    """在 LangChain 分发每个工具调用前执行权限审核。"""

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        tool_call = request.tool_call
        tool_name = tool_call["name"]
        args = tool_call.get("args", {})

        if not check_permission(tool_name, args):
            return ToolMessage(
                content="Permission denied.",
                tool_call_id=tool_call["id"],
                name=tool_name,
                status="error",
            )
        return handler(request)


# 与 s02 一样延迟创建，导入模块时不连接模型。
agent = None


def get_agent():
    """创建并复用带权限 middleware 的五工具 LangChain Agent。"""
    global agent
    if agent is None:
        model = ChatOpenAI(
            model=MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("BASE_URL") or None,
            max_completion_tokens=8000,
            temperature=0,
        )
        agent = create_agent(
            model=model,
            tools=TOOLS,
            system_prompt=SYSTEM,
            middleware=[PermissionMiddleware()],
        )
    return agent


def agent_loop(messages: list) -> None:
    """流式运行 Agent；工具调用会先经过 ``PermissionMiddleware``。"""
    final_messages = messages
    for chunk in get_agent().stream(
        {"messages": messages},
        stream_mode=["messages", "values"],
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if isinstance(token, AIMessageChunk) and metadata.get("langgraph_node") == "model" and token.text:
                print(token.text, end="", flush=True)
        elif chunk["type"] == "values":
            final_messages = chunk["data"]["messages"]

    messages[:] = final_messages
    print()


if __name__ == "__main__":
    print("s03: Permission (LangChain middleware)")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            # \001/\002 告诉 Readline：ANSI 转义序列不占显示宽度。
            query = input("\001\033[36m\002s03 >> \001\033[0m\002")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        print()
