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
from langchain.agents.middleware import(
    AgentState,
    TodoListMiddleware,
    after_agent,
    before_agent,
    wrap_tool_call
)

from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.runtime import Runtime
from langgraph.types import Command

import yaml


load_dotenv(override=True)

WORKDIR = Path.cwd().resolve()
SKILL_DIR = WORKDIR/"skills"

SKILL_REGISTRY: dict[str, dict[str, str]] = {}


def _parse_frontmatter(raw: str) -> tuple[dict[str,Any], str]:

    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw

    try:
        end_index = next(
            index
            for index, lines in enumerate(lines[1:], start=1)
            if lines.strip() == "---"
        )
    except StopIteration:
        return {}, raw


    yaml_text = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index+1:])

    try:
        metadata = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError:
        metadata = {}

    if not isinstance(metadata, dict):
           metadata = {}

    return metadata, body


def _scan_skills() -> None:

    if not SKILL_DIR.exists():
        return

    for manifest in sorted(SKILL_DIR.glob("*/SKILL.md")):
        raw = manifest.read_text(
            encoding="utf-8",
            errors="replace",
        )


        metadata, body = _parse_frontmatter(raw)


        name = str(
            metadata.get("name")
            or manifest.parent.name
        )

        description = metadata.get("description")

        if not description:
            description = next(
                (
                    line.lstrip("#").strip()
                    for line in body.splitlines()
                    if line.lstrip().startswith("#")
                ),
                name,
            )


        description = " ".join(
            str(description).split()
        )

        if name in SKILL_REGISTRY:
            raise ValueError(
                f"Duplicate skill name: {name}"
            )

        skill_root = manifest.parent.relative_to(
            WORKDIR
        ).as_posix()


        SKILL_REGISTRY[name] = {
            "name": name,
            "description": description,
            "content": raw,
            "root": skill_root,
        }

def list_skills() ->str:

    if not SKILL_REGISTRY:
        return "- (no skills found)"

    return "\n".join(
        f"- {skill['name']}: {skill['description']}"
        for skill in SKILL_REGISTRY.values()
    )


_scan_skills()

SKILL_CATALOG = list_skills()

MODEL_ID = os.getenv("MODEL_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("BASE_URL")


SKILL_SYSTEM_PROMPT = f"""
Available skills:

{SKILL_CATALOG}

Skills contain specialized instructions that should be loaded only when
relevant.

When a request clearly matches a skill description:

1. Call load_skill using the exact skill name.
2. Read and follow the returned instructions before doing the task.
3. Do not guess a skill's full instructions from its description.
4. Load only skills relevant to the current request.
"""


AGENT_SCOPE: ContextVar[str] = ContextVar(
    "agent_scope",
    default="parent"
)


HOOKS: dict[str, list[Callable[..., Any]]] = {
    "UserPromptSubmit": [],
    "PreToolUse" : [],
    "PostToolUse": [],
    "Stop": [],
}

def register_hook(event: str,callback: Callable[..., Any]) -> None:


    if event not in HOOKS:
        raise ValueError(f"unknown hook event:{event}")

    HOOKS[event].append(callback)


def trigger_hook(event:str, *args: Any)-> Any|None:

    for callable in HOOKS.get(event,[]):
        result = callable(*args)

        if result is not None:
            return result
    return None

@before_agent
def user_prompt_submit(state:AgentState, runtime:Runtime) -> dict[str, Any] |None:


    messages = state.get("messages", [])

    if not messages:
        return None

    last_messages = messages[-1]

    if isinstance(last_messages, dict):
        content = last_messages.get("content")

    else:
        content = getattr(last_messages,"content",None)

    trigger_hook("UserPromptSubmit", content)

    return None


@wrap_tool_call
def tool_hook(
    request: ToolCallRequest,
    handler: Callable[
        [ToolCallRequest],
        ToolMessage | Command,
    ],
) -> ToolMessage | Command:
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})
    tool_call_id = request.tool_call["id"]

    blocked_reason = trigger_hook(
        "PreToolUse",
        tool_name,
        tool_args,
    )

    if blocked_reason:
        return ToolMessage(
            content=str(blocked_reason),
            tool_call_id = tool_call_id,
            name = tool_name,
            status = "error",
        )

    result = handler(request)

    trigger_hook(
        "PostToolUse",
        tool_name,
        tool_args,
        result,
    )

    return result


@after_agent
def stop_hook(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:

    trigger_hook(
        "Stop",
        state.get("messages", [])
    )

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

def resolve_path(raw_path:str)-> Path:


    candidate = Path(raw_path)

    if candidate.is_absolute():
        return candidate.resolve()

    return (WORKDIR/candidate).resolve()


def check_deny_list(command:str) ->str | None:
    normalized = command.lower()
    for pattern in DANGEROUS_COMMANDS:
        if pattern.lower() in normalized:
            return f"Blocked:{pattern} is in the deny list"


    return None

def ask_user(tool_name:str, args:dict[str, Any],reason: str) ->bool:

    scope = AGENT_SCOPE.get()

    print(f"\nWarning: [{scope}] Permission required")
    print(f"Reason: {reason}")
    print(f"Tool: {tool_name}")
    print(f"Arguments: {args}")

    choice = input("Allow? [y/N] ").strip().lower()

    return choice in {"y", "yes"}


def  check_rules(
        tool_name: str,
        args: dict[str, Any]
) -> str| None:


    if tool_name == "run_bash":
        command = str(args.get("command", ""))
        normalized = command.lower()

        if normalized.strip().startswith("del "):
            return "Potentially destructive shell command: del "
        for pattern in POTENTIALLY_DESTRUCTIVE_COMMANDS:
            if pattern.lower() in normalized:
                return(
                    "Potentially destructive shell command: "
                    f"{pattern}"
                )

    if tool_name in {"run_read","run_write","run_edit"}:
        raw_path = str(args.get("path", ""))

        try:
            target = resolve_path(raw_path)
        except(OSError, RuntimeError, ValueError) as exc:
            return f"Invalid path:{exc}"

        if not target.is_relative_to(WORKDIR):
            return f"Operation accesses outside workspace: {target}"

    return None


def check_permission(
        tool_name: str,
        args: dict[str, Any]
)-> bool:


    if tool_name == "run_bash":
        command = str(args.get("command", ""))

        denied_reason = check_deny_list(command)

        if denied_reason:
            print(f"\nBlocked: {denied_reason}")
            return False

    confirmation_reason = check_rules(
        tool_name,
        args,
    )

    if confirmation_reason:
        return ask_user(
            tool_name,
            args,
            confirmation_reason,
        )

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
) ->None:
    scope = AGENT_SCOPE.get()

    print(f"[{scope} PostToolUse] {tool_name}")

    content = getattr(result, "content", result)
    preview = str(content)

    if len(preview) >500:
        preview = preview[:500] + "...(truncated)"

    print(f"Result: {preview}")


def on_stop(messages: list[Any]) ->None:
    tool_call_count = 0
    for message in messages:
        if isinstance(message, AIMessage):
            tool_call_count += len(message.tool_calls or [])

    print(
        f"[Stop] messages={len(messages)}, "
        f"tool_calls={tool_call_count}"
    )


register_hook(
    "UserPromptSubmit",
    on_user_prompt_submit,
)

register_hook(
    "PreToolUse",
    on_pre_tool_use,
)

register_hook(
    "PostToolUse",
    on_post_tool_use,
)

register_hook(
    "Stop",
    on_stop,
)


@tool
def load_skill(name: str) -> str:
    """Load the full instructions for a skill.

    Args:
        name: Exact skill name shown in the available-skills catalog.
    """

    skill = SKILL_REGISTRY.get(name)

    if skill is None:
        available = ", ".join(SKILL_REGISTRY)
        return (
            f"Skill not found: {name}. "
            f"Available skills: {available or '(none)'}"
        )

    return (
        f"Loaded skill: {skill['name']}\n"
        f"Skill root: {skill['root']}\n"
        "Resolve relative paths mentioned by this skill "
        "against the skill root above.\n\n"
        f"{skill['content']}"
    )

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

        output = (
            result.stdout
            + result.stderr
        ).strip()

        if not output:
            output = "(no output)"

        if result.returncode != 0:
            output = (
                f"Exit code: {result.returncode}\n"
                f"{output}"
            )

        return output[:50000]

    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120 seconds"

    except OSError as exc:
        return f"Error: {exc}"


@tool
def run_read(
    path: str,
    limit: int | None = None,
) -> str:
    """Read a UTF-8 text file.

    Args:
        path: File path, normally relative to the workspace.
        limit: Optional maximum number of lines to return.
    """

    try:
        file_path = resolve_path(path)

        lines = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()

        if limit is not None and limit >= 0 and limit < len(lines):
            remaining = len(lines) - limit

            lines = [
                *lines[:limit],
                f"...({remaining} more lines)",
            ]

        return "\n".join(lines)

    except (OSError, ValueError) as exc:
        return f"Error: {exc}"


@tool
def run_write(
    path: str,
    content: str,
) -> str:
    """Write UTF-8 content to a file, replacing existing content.

    Args:
        path: Target file path.
        content: Complete new file content.
    """

    try:
        file_path = resolve_path(path)

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_path.write_text(
            content,
            encoding="utf-8",
        )

        return f"Wrote {len(content)} characters to {path}"

    except (OSError, ValueError) as exc:
        return f"Error: {exc}"


@tool
def run_edit(
    path: str,
    old_text: str,
    new_text: str,
) -> str:
    """Replace the first exact occurrence of text in a UTF-8 file.

    Args:
        path: Target file path.
        old_text: Exact text that should be replaced.
        new_text: Replacement text.
    """

    try:
        file_path = resolve_path(path)

        current_content = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        if old_text not in current_content:
            return f"Error: old_text was not found in {path}"

        updated_content = current_content.replace(
            old_text,
            new_text,
            1,
        )

        file_path.write_text(
            updated_content,
            encoding="utf-8",
        )

        return f"Edited {path}"

    except (OSError, ValueError) as exc:
        return f"Error: {exc}"


@tool
def run_glob(pattern: str) -> str:
    """Find workspace files matching a glob pattern.

    Args:
        pattern: Pattern such as "*.py" or "src/**/*.py".
    """

    try:
        results: list[str] = []

        for match in glob.glob(
            pattern,
            root_dir=WORKDIR,
            recursive=True,
        ):
            full_path = (WORKDIR / match).resolve()

            if full_path.is_relative_to(WORKDIR):
                results.append(match)

        if not results:
            return "(no matches)"

        return "\n".join(sorted(results))

    except (OSError, ValueError) as exc:
        return f"Error: {exc}"

BASE_TOOLS = [
    run_bash,
    run_read,
    run_write,
    run_edit,
    run_glob,
    load_skill,
]


MODEL = ChatOpenAI(
    model=MODEL_ID,
    max_completion_tokens=8000,
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

SUB_SYSTEM = f"""
You are an isolated coding subagent working in:

{WORKDIR}

{SKILL_SYSTEM_PROMPT}

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
            text:str | None =None

            if isinstance(block, str):
                text = block

            elif isinstance(block,dict):
                possible_text = block.get("text")

                if isinstance(possible_text, str):
                    text = possible_text

            else:
                possible_text = getattr(
                    block,
                    "text",
                    None
                )

                if isinstance(possible_text, str):
                    text = possible_text

            if text and text.strip():
                texts.append(text.strip())

        if texts:
            return "\n".join(texts)

    return ""

@tool("task")
def task(description: str) -> str:
    """Launch an isolated subagent for a complex subtask.

    The subagent receives only this description, not the parent conversation.
    Include the complete objective, paths, constraints and expected output.
    Only the final textual conclusion is returned.
    """

    print("\n\033[35m[Subagent spawned]\033[0m")
    print(f"Task: {description}")

    scope_token = AGENT_SCOPE.set("sub")

    try:
        result = SUB_AGENT.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": description,
                    }
                ]
            },
            config={"recursion_limit": 128},
        )

        summary = extract_final_text(
            result.get("messages", [])
        )

        return (
            summary
            or "Subagent finished without a textual conclusion."
        )

    except GraphRecursionError:
        return (
            "Subagent stopped because it reached "
            "the execution limit."
        )

    except Exception as exc:
        return (
            "Subagent failed: "
            f"{type(exc).__name__}: {exc}"
        )

    finally:
        AGENT_SCOPE.reset(scope_token)
        print("\033[35m[Subagent done]\033[0m")

PARENT_SYSTEM = f"""
You are a coding agent working in:

{WORKDIR}
{SKILL_SYSTEM_PROMPT}

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


PARENT_TOOLS = [
    *BASE_TOOLS,
    task,
]

PARENT_MIDDLEWARE = [
    user_prompt_submit,

    TodoListMiddleware(
        system_prompt="""
        You must call write_todos before calling run_bash, run_read, run_write,
        run_edit, run_glob, or task.

        For every non-empty, non-trivial request:

        1. Create at least one todo item before using another tool.
        2. Keep exactly one relevant item in_progress while working.
        3. Update todo statuses as work progresses.
        4. Mark items completed only after the work is actually complete.
        """,
        tool_description="""
        Create or update the current task list. This is a mandatory planning tool.
        Call it before using run_bash, run_read, run_write, run_edit, run_glob,
        or task.
        """,
    ),

    tool_hook,
    stop_hook,
]

agent = create_agent(
    model= MODEL,
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
            continue

        if isinstance(block, dict):
            text = block.get("text")

            if isinstance(text, str):
                texts.append(text)

            continue

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
                print(
                    "参数："
                    f"{tool_call.get('args', {})}"
                )

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

    content = getattr(message, "content", message)

    print("\n消息：")
    print(content_to_text(content))


def print_todos(todos: list[dict[str, Any]]) -> None:


    print("\n当前 Todo：")

    for index, todo_item in enumerate(
        todos,
        start=1,
    ):
        status = todo_item.get(
            "status",
            "pending",
        )

        content = todo_item.get(
            "content",
            "",
        )

        print(
            f"{index}. [{status}] {content}"
        )


def agent_loop(
    session_state: dict[str, Any],
) -> None:


    existing_messages = session_state.get(
        "messages",
        [],
    )

    seen_message_count = len(existing_messages)
    last_todos = session_state.get("todos")
    final_state: dict[str, Any] | None = None

    for state in agent.stream(
        session_state,
        stream_mode="values",
    ):
        final_state = state

        todos = state.get("todos")

        if todos is not None and todos != last_todos:
            print_todos(todos)
            last_todos = todos

        current_messages = state.get(
            "messages",
            [],
        )

        new_messages = current_messages[
            seen_message_count:
        ]

        for message in new_messages:
            print_message(message)

        seen_message_count = len(
            current_messages
        )

    if final_state is not None:

        session_state.clear()
        session_state.update(final_state)


def main() -> None:
    print("s07: Skill Loading — catalog in SYSTEM, content on demand")
    print("输入问题，回车发送。输入 q 退出。\n")


    session_state: dict[str, Any] = {
        "messages": [],
    }

    while True:
        try:
            query = input(
                "\033[36ms07 >> \033[0m"
            )

        except (
            EOFError,
            KeyboardInterrupt,
        ):
            print()
            break

        if query.strip().lower() in {
            "",
            "q",
            "exit",
        }:
            break

        session_state.setdefault(
            "messages",
            [],
        ).append(
            {
                "role": "user",
                "content": query,
            }
        )

        try:
            agent_loop(session_state)

        except GraphRecursionError:
            print(
                "\nAgent stopped because it reached "
                "the execution limit."
            )

        except Exception as exc:
            print(
                "\nAgent error: "
                f"{type(exc).__name__}: {exc}"
            )

        print()


if __name__ == "__main__":
    main()
