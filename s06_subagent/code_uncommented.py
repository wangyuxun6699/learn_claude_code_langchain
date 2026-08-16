from __future__ import annotations

import glob
import os
import subprocess
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    TodoListMiddleware,
    after_agent,
    before_agent,
    wrap_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.runtime import Runtime
from langgraph.types import Command


load_dotenv(override=True)

WORKDIR = Path.cwd().resolve()
MODEL_ID = os.getenv("MODEL_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL")

if not MODEL_ID:
    raise RuntimeError("Missing MODEL_ID in .env")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

AGENT_SCOPE: ContextVar[str] = ContextVar("agent_scope", default="parent")

HOOKS: dict[str, list[Callable[..., Any]]] = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": [],
}


def register_hook(event: str, callback: Callable[..., Any]) -> None:
    if event not in HOOKS:
        raise ValueError(f"Unknown hook event: {event}")
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args: Any) -> Any | None:
    for callback in HOOKS.get(event, []):
        result = callback(*args)
        if result is not None:
            return result
    return None


@before_agent
def user_prompt_submit(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    messages = state.get("messages", [])
    if not messages:
        return None

    last_message = messages[-1]
    if isinstance(last_message, dict):
        content = last_message.get("content")
    else:
        content = getattr(last_message, "content", None)

    trigger_hooks("UserPromptSubmit", content)
    return None


@wrap_tool_call
def tool_hook(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})
    tool_call_id = request.tool_call["id"]

    blocked_reason = trigger_hooks("PreToolUse", tool_name, tool_args)
    if blocked_reason:
        return ToolMessage(
            content=str(blocked_reason),
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )

    result = handler(request)
    trigger_hooks("PostToolUse", tool_name, tool_args, result)
    return result


@after_agent
def stop_hook(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    trigger_hooks("Stop", state.get("messages", []))
    return None


DANGEROUS_COMMANDS = [
    "rm -rf /",
    "sudo",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=",
    "> /dev/sda",
]

POTENTIALLY_DESTRUCTIVE_COMMANDS = [
    "rm ",
    "> /etc/",
    "chmod 777",
]


def resolve_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (WORKDIR / candidate).resolve()


def check_deny_list(command: str) -> str | None:
    normalized = command.lower()
    for pattern in DANGEROUS_COMMANDS:
        if pattern.lower() in normalized:
            return f"Blocked: '{pattern}' is on the deny list"
    return None


def check_rules(tool_name: str, args: dict[str, Any]) -> str | None:
    if tool_name == "run_bash":
        normalized = str(args.get("command", "")).lower()
        if normalized.strip().startswith("del "):
            return "Potentially destructive shell command: del "
        for pattern in POTENTIALLY_DESTRUCTIVE_COMMANDS:
            if pattern.lower() in normalized:
                return f"Potentially destructive shell command: {pattern}"

    if tool_name in {"run_read", "run_write", "run_edit"}:
        raw_path = str(args.get("path", ""))
        try:
            target = resolve_path(raw_path)
        except (OSError, RuntimeError, ValueError) as exc:
            return f"Invalid path: {exc}"
        if not target.is_relative_to(WORKDIR):
            return f"Operation accesses outside workspace: {target}"

    return None


def ask_user(tool_name: str, args: dict[str, Any], reason: str) -> bool:
    scope = AGENT_SCOPE.get()
    print(f"\nWarning: [{scope}] Permission required")
    print(f"Reason: {reason}")
    print(f"Tool: {tool_name}")
    print(f"Arguments: {args}")
    choice = input("Allow? [y/N] ").strip().lower()
    return choice in {"y", "yes"}


def check_permission(tool_name: str, args: dict[str, Any]) -> bool:
    if tool_name == "run_bash":
        denied_reason = check_deny_list(str(args.get("command", "")))
        if denied_reason:
            print(f"\nBlocked: {denied_reason}")
            return False

    confirmation_reason = check_rules(tool_name, args)
    if confirmation_reason:
        return ask_user(tool_name, args, confirmation_reason)
    return True


def on_user_prompt_submit(content: Any) -> None:
    print(f"[UserPromptSubmit] {content}")


def on_pre_tool_use(
    tool_name: str,
    tool_args: dict[str, Any],
) -> str | None:
    scope = AGENT_SCOPE.get()
    print(f"[{scope} PreToolUse] {tool_name}")
    print(f"Arguments: {tool_args}")
    if not check_permission(tool_name, tool_args):
        return "Permission denied"
    return None


def on_post_tool_use(
    tool_name: str,
    tool_args: dict[str, Any],
    result: ToolMessage | Command,
) -> None:
    scope = AGENT_SCOPE.get()
    print(f"[{scope} PostToolUse] {tool_name}")
    preview = str(getattr(result, "content", result))
    if len(preview) > 500:
        preview = preview[:500] + "...(truncated)"
    print(f"Result: {preview}")


def on_stop(messages: list[Any]) -> None:
    tool_call_count = 0
    for message in messages:
        if isinstance(message, AIMessage):
            tool_call_count += len(message.tool_calls or [])
    print(f"[Stop] messages={len(messages)}, tool_calls={tool_call_count}")


register_hook("UserPromptSubmit", on_user_prompt_submit)
register_hook("PreToolUse", on_pre_tool_use)
register_hook("PostToolUse", on_post_tool_use)
register_hook("Stop", on_stop)


@tool
# 安全边界：shell=True 仅为教学演示，黑名单/路径检查不等于安全边界；生产请使用权限中间件 + 沙箱。
def run_bash(command: str) -> str:
    """Execute a shell command in the current workspace."""
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
        output = (result.stdout + result.stderr).strip() or "(no output)"
        if result.returncode != 0:
            output = f"Exit code: {result.returncode}\n{output}"
        return output[:50000]
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120 seconds"
    except OSError as exc:
        return f"Error: {exc}"


@tool
def run_read(path: str, limit: int | None = None) -> str:
    """Read a UTF-8 text file, optionally limiting the returned line count."""
    try:
        lines = resolve_path(path).read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        if limit is not None and 0 <= limit < len(lines):
            lines = [*lines[:limit], f"...({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except (OSError, ValueError) as exc:
        return f"Error: {exc}"


@tool
def run_write(path: str, content: str) -> str:
    """Write UTF-8 content to a file, replacing its existing content."""
    try:
        file_path = resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}"
    except (OSError, ValueError) as exc:
        return f"Error: {exc}"


@tool
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace the first exact occurrence of old_text in a UTF-8 file."""
    try:
        file_path = resolve_path(path)
        current_content = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
        if old_text not in current_content:
            return f"Error: old_text was not found in {path}"
        file_path.write_text(
            current_content.replace(old_text, new_text, 1),
            encoding="utf-8",
        )
        return f"Edited {path}"
    except (OSError, ValueError) as exc:
        return f"Error: {exc}"


@tool
def run_glob(pattern: str) -> str:
    """Find workspace files matching a glob pattern."""
    try:
        results: list[str] = []
        for match in glob.glob(pattern, root_dir=WORKDIR, recursive=True):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(sorted(results)) if results else "(no matches)"
    except (OSError, ValueError) as exc:
        return f"Error: {exc}"


BASE_TOOLS = [run_bash, run_read, run_write, run_edit, run_glob]

MODEL = ChatOpenAI(
    model=MODEL_ID,
    max_completion_tokens=8000,
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

SUB_SYSTEM = f"""
You are an isolated coding subagent working in:

{WORKDIR}

Complete the exact task given by the parent agent.

Rules:

1. Work independently and use the available tools when necessary.
2. You do not have access to the parent's conversation history.
3. Do not assume information that was not included in the task description.
4. Do not delegate or attempt to create another agent.
5. When finished, return a concise but complete summary.
6. Include relevant file paths, findings, changes, verification results,
   and unresolved problems in the final summary.
"""

SUB_AGENT = create_agent(
    model=MODEL,
    tools=BASE_TOOLS,
    system_prompt=SUB_SYSTEM,
    middleware=[tool_hook],
    name="worker",
)


def extract_final_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue

        content = message.content
        if isinstance(content, str):
            if content.strip():
                return content.strip()
            continue
        if not isinstance(content, list):
            continue

        texts: list[str] = []
        for block in content:
            text: str | None = None
            if isinstance(block, str):
                text = block
            elif isinstance(block, dict):
                possible_text = block.get("text")
                if isinstance(possible_text, str):
                    text = possible_text
            else:
                possible_text = getattr(block, "text", None)
                if isinstance(possible_text, str):
                    text = possible_text
            if text and text.strip():
                texts.append(text.strip())

        if texts:
            return "\n".join(texts)
    return ""


@tool("task")
def task(description: str) -> str:
    """Launch an isolated subagent and return only its final conclusion."""
    print("\n\033[35m[Subagent spawned]\033[0m")
    print(f"Task: {description}")
    scope_token = AGENT_SCOPE.set("sub")
    try:
        result = SUB_AGENT.invoke(
            {"messages": [{"role": "user", "content": description}]},
            config={"recursion_limit": 128},
        )
        summary = extract_final_text(result.get("messages", []))
        return summary or "Subagent finished without a textual conclusion."
    except GraphRecursionError:
        return "Subagent stopped because it reached the execution limit."
    except Exception as exc:
        return f"Subagent failed: {type(exc).__name__}: {exc}"
    finally:
        AGENT_SCOPE.reset(scope_token)
        print("\033[35m[Subagent done]\033[0m")


PARENT_SYSTEM = f"""
You are a coding agent working in:

{WORKDIR}

You must use write_todos for every non-trivial request.

Before using run_bash, run_read, run_write, run_edit, run_glob, or task,
create or update the todo list.

Use task when a subproblem is:

- complex and self-contained
- likely to require reading many files
- likely to require several tool calls
- useful to solve in an isolated context

The task subagent cannot see this conversation. Every task description must
therefore contain:

- the precise objective
- relevant files and directories
- necessary background information
- constraints
- expected output
- whether files may be modified

The task tool returns only the subagent's final conclusion. Its intermediate
messages are deliberately discarded.

After receiving a task result, evaluate it, verify it when necessary, and
continue working on the parent request.

For this demonstration, you MUST use task for every request that requires
reading, writing, editing, or executing files. The parent agent must not
perform those operations directly.
"""

PARENT_TOOLS = [*BASE_TOOLS, task]

PARENT_MIDDLEWARE = [
    user_prompt_submit,
    TodoListMiddleware(
        system_prompt="""
You must call write_todos before calling run_bash, run_read, run_write,
run_edit, run_glob, or task. Create at least one todo for every non-trivial
request, keep one relevant item in_progress, and update statuses as work
progresses.
""",
        tool_description="""
Create or update the current task list. Call this mandatory planning tool
before run_bash, run_read, run_write, run_edit, run_glob, or task.
""",
    ),
    tool_hook,
    stop_hook,
]

agent = create_agent(
    model=MODEL,
    tools=PARENT_TOOLS,
    system_prompt=PARENT_SYSTEM,
    middleware=PARENT_MIDDLEWARE,
    name="parent",
)


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    texts: list[str] = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                texts.append(text)
        else:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts)


def print_message(message: Any) -> None:
    if isinstance(message, AIMessage):
        if message.tool_calls:
            print("\n模型调用工具：")
            for tool_call in message.tool_calls:
                print(f"工具名：{tool_call['name']}")
                print(f"参数：{tool_call.get('args', {})}")
        text = content_to_text(message.content)
        if text.strip():
            print("\n模型回复：")
            print(text)
        return

    if isinstance(message, ToolMessage):
        print("\n工具返回结果：")
        print(f"工具名：{message.name}")
        print(f"状态：{getattr(message, 'status', 'success')}")
        print(f"内容：{message.content}")
        return

    print("\n消息：")
    print(content_to_text(getattr(message, "content", message)))


def print_todos(todos: list[dict[str, Any]]) -> None:
    print("\n当前 Todo：")
    for index, todo_item in enumerate(todos, start=1):
        status = todo_item.get("status", "pending")
        content = todo_item.get("content", "")
        print(f"{index}. [{status}] {content}")


def agent_loop(session_state: dict[str, Any]) -> None:
    seen_message_count = len(session_state.get("messages", []))
    last_todos = session_state.get("todos")
    final_state: dict[str, Any] | None = None

    for state in agent.stream(session_state, stream_mode="values"):
        final_state = state
        todos = state.get("todos")
        if todos is not None and todos != last_todos:
            print_todos(todos)
            last_todos = todos

        current_messages = state.get("messages", [])
        for message in current_messages[seen_message_count:]:
            print_message(message)
        seen_message_count = len(current_messages)

    if final_state is not None:
        session_state.clear()
        session_state.update(final_state)


def main() -> None:
    print("s06: LangChain Subagent — isolated context, summary only")
    print("输入问题并回车。输入 q 或 exit 退出。\n")
    session_state: dict[str, Any] = {"messages": []}

    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.strip().lower() in {"", "q", "exit"}:
            break

        session_state.setdefault("messages", []).append(
            {"role": "user", "content": query}
        )
        try:
            agent_loop(session_state)
        except GraphRecursionError:
            print("\nAgent stopped because it reached the execution limit.")
        except Exception as exc:
            print(f"\nAgent error: {type(exc).__name__}: {exc}")
        print()


if __name__ == "__main__":
    main()
