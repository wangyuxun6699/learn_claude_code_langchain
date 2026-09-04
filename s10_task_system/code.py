#!/usr/bin/env python3
"""
s10_task_system.py - Task System

    .tasks/
      task_a1b2c3d4.json  {status: completed, blockedBy: []}
      task_e5f6a7b8.json  {status: pending, blockedBy: [task_a1b2c3d4]}
      task_11223344.json  {status: pending, blockedBy: [task_e5f6a7b8]}

    Dependency graph:

    +-----------+      +-----------+      +-----------+
    | schema    | ---> | API       | ---> | tests     |
    | completed |      | pending   |      | pending   |
    +-----------+      +-----------+      +-----------+

    can_start(API) is true because schema is completed.

    Task lifecycle:

    pending --claim_task--> in_progress --complete_task--> completed
"""

import glob
import json
import os
import re
import secrets
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    pass

# -- 本章内置的 LangChain 消息适配（直接展开，便于单文件阅读） --
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI


@dataclass(slots=True)
class TextBlock:
    """课程循环读取的文本内容块。"""

    text: str
    type: str = field(default="text", init=False)


@dataclass(slots=True)
class ToolUseBlock:
    """课程循环读取的工具调用内容块。"""

    id: str
    name: str
    input: dict[str, Any]
    type: str = field(default="tool_use", init=False)


@dataclass(slots=True)
class Usage:
    """统一暴露工作流和 Goal 统计所需的 token 字段。"""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(slots=True)
class MessageResponse:
    """课程侧需要的最小模型响应。"""

    content: list[TextBlock | ToolUseBlock]
    stop_reason: str
    usage: Usage
    raw: AIMessage


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _block_type(block: Any) -> str | None:
    return _value(block, "type")


def _text_content(content: Any) -> str:
    """把 provider 内容块、工具结果或普通对象安全地转成文本。"""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        return str(content)

    texts: list[str] = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
            continue
        block_text = _value(block, "text")
        if isinstance(block_text, str):
            texts.append(block_text)
            continue
        nested = _value(block, "content")
        if nested is not None:
            texts.append(_text_content(nested))
    return "\n".join(text for text in texts if text)


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    return _text_content(system)


def _assistant_message(content: Any) -> AIMessage:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    blocks = content if isinstance(content, list) else [content]

    for block in blocks:
        kind = _block_type(block)
        if kind == "tool_use":
            tool_calls.append(
                {
                    "id": str(_value(block, "id", "")),
                    "name": str(_value(block, "name", "")),
                    "args": dict(_value(block, "input", {}) or {}),
                    "type": "tool_call",
                }
            )
            continue
        text = _value(block, "text")
        if isinstance(text, str) and text:
            text_parts.append(text)

    return AIMessage(content="\n".join(text_parts), tool_calls=tool_calls)


def _user_messages(content: Any) -> list[BaseMessage]:
    if isinstance(content, str):
        return [HumanMessage(content=content)]
    if not isinstance(content, list):
        return [HumanMessage(content=str(content))]

    results: list[BaseMessage] = []
    user_text: list[str] = []
    for block in content:
        if _block_type(block) == "tool_result":
            if user_text:
                results.append(HumanMessage(content="\n".join(user_text)))
                user_text.clear()
            results.append(
                ToolMessage(
                    content=_text_content(_value(block, "content", "")),
                    tool_call_id=str(_value(block, "tool_use_id", "")),
                    status="error" if bool(_value(block, "is_error", False)) else "success",
                )
            )
            continue
        text = _value(block, "text")
        if isinstance(text, str):
            user_text.append(text)
        elif isinstance(block, str):
            user_text.append(block)

    if user_text or not results:
        results.append(HumanMessage(content="\n".join(user_text)))
    return results


def _to_langchain_messages(messages: list[Any]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, BaseMessage):
            converted.append(message)
            continue
        role = _value(message, "role")
        content = _value(message, "content", "")
        if role == "assistant":
            converted.append(_assistant_message(content))
        elif role == "system":
            converted.append(SystemMessage(content=_text_content(content)))
        else:
            converted.extend(_user_messages(content))
    return converted


def _openai_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for item in tools or []:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": item["name"],
                    "description": item.get("description", ""),
                    "parameters": item.get("input_schema", {"type": "object"}),
                },
            }
        )
    return converted


def _response_blocks(message: AIMessage) -> list[TextBlock | ToolUseBlock]:
    blocks: list[TextBlock | ToolUseBlock] = []
    text = _text_content(message.content)
    if text:
        blocks.append(TextBlock(text=text))
    for call in message.tool_calls:
        blocks.append(
            ToolUseBlock(
                id=str(call.get("id", "")),
                name=str(call.get("name", "")),
                input=dict(call.get("args", {}) or {}),
            )
        )
    return blocks


def _usage(message: AIMessage) -> Usage:
    metadata = message.usage_metadata or {}
    return Usage(
        input_tokens=int(metadata.get("input_tokens", 0) or 0),
        output_tokens=int(metadata.get("output_tokens", 0) or 0),
    )


class _MessagesAPI:
    def __init__(self, owner: "LangChainMessagesClient") -> None:
        self.owner = owner

    def create(
        self,
        *,
        model: str,
        messages: list[Any],
        system: Any = "",
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8000,
        temperature: float = 0,
        **_: Any,
    ) -> MessageResponse:
        llm = ChatOpenAI(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("BASE_URL") or self.owner.base_url,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            timeout=self.owner.timeout,
            max_retries=self.owner.max_retries,
        )
        openai_tools = _openai_tools(tools)
        runnable = llm.bind_tools(openai_tools) if openai_tools else llm
        request = [SystemMessage(content=_system_text(system))]
        request.extend(_to_langchain_messages(messages))
        raw = runnable.invoke(request)
        if not isinstance(raw, AIMessage):
            raw = AIMessage(content=str(getattr(raw, "content", raw)))

        finish_reason = str((raw.response_metadata or {}).get("finish_reason", ""))
        if raw.tool_calls:
            stop_reason = "tool_use"
        elif finish_reason in {"length", "max_tokens"}:
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"
        return MessageResponse(
            content=_response_blocks(raw),
            stop_reason=stop_reason,
            usage=_usage(raw),
            raw=raw,
        )


class LangChainMessagesClient:
    """课程统一模型边界；真实网络调用延迟到 ``create``。"""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int = 2,
        **_: Any,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.messages = _MessagesAPI(self)

# -- 本章的 Agent / Harness 机制 --
from dotenv import load_dotenv

load_dotenv(override=True)
WORKDIR = Path.cwd()
client = LangChainMessagesClient(base_url=os.getenv("BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use task tools to track dependencies and progress. Create all task nodes "
    "first. After create_task returns runtime-generated IDs, use update_task "
    "with those exact IDs to add dependencies."
)


# -- New in s10: persistent task records --

TASKS_DIR = WORKDIR / ".tasks"
TASK_ID_PATTERN = re.compile(r"^task_[0-9a-f]{8}$")


@dataclass
class Task:
    id: str
    subject: str
    description: str
    status: str
    owner: str | None
    blockedBy: list[str]


class TaskStore:
    def __init__(self, directory: Path):
        self.directory = directory

    def _root(self, create: bool = False) -> Path:
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
        root = self.directory.resolve()
        if not root.is_relative_to(WORKDIR.resolve()):
            raise ValueError("Task store escapes the workspace")
        return root

    def _path(self, task_id: str, create_root: bool = False) -> Path:
        if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError(f"Invalid task ID: {task_id!r}")
        root = self._root(create=create_root)
        path = (root / f"{task_id}.json").resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Invalid task ID: {task_id!r}")
        return path

    def exists(self, task_id: str) -> bool:
        return self._path(task_id).is_file()

    def create(self, subject: str, description: str = "") -> Task:
        subject = subject.strip()
        if not subject:
            raise ValueError("Task subject cannot be empty")

        self._root(create=True)
        for _ in range(100):
            task = Task(
                id=f"task_{secrets.token_hex(4)}",
                subject=subject,
                description=description,
                status="pending",
                owner=None,
                blockedBy=[],
            )
            try:
                with self._path(task.id, create_root=True).open(
                    "x", encoding="utf-8"
                ) as handle:
                    json.dump(asdict(task), handle, indent=2)
                return task
            except FileExistsError:
                continue
        raise RuntimeError("Could not allocate a unique task ID")

    def _depends_on(self, task_id: str, target_id: str) -> bool:
        """Return whether task_id transitively depends on target_id."""
        pending = [task_id]
        visited = set()
        while pending:
            current = pending.pop()
            if current == target_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(self.load(current).blockedBy)
        return False

    def update_dependencies(self, task_id: str,
                            add_blocked_by: list[str]) -> Task:
        if not isinstance(add_blocked_by, list):
            raise ValueError("addBlockedBy must be a list of task IDs")

        task = self.load(task_id)
        if task.status != "pending" or task.owner is not None:
            raise ValueError(
                f"Task {task_id} dependencies can only be updated while "
                "pending and unowned"
            )

        dependencies = list(dict.fromkeys(add_blocked_by))
        for dependency in dependencies:
            if dependency == task_id:
                raise ValueError("Task cannot depend on itself")
            if not self.exists(dependency):
                raise ValueError(f"Dependency not found: {dependency}")
            if dependency not in task.blockedBy and self._depends_on(
                dependency, task_id
            ):
                raise ValueError(
                    f"Dependency cycle detected: {task_id} -> {dependency}"
                )

        task.blockedBy.extend(
            dependency for dependency in dependencies
            if dependency not in task.blockedBy
        )
        self.save(task)
        return task

    def save(self, task: Task) -> None:
        self._path(task.id, create_root=True).write_text(
            json.dumps(asdict(task), indent=2),
            encoding="utf-8",
        )

    def load(self, task_id: str) -> Task:
        data = json.loads(self._path(task_id).read_text(encoding="utf-8"))
        task = Task(**data)
        if task.id != task_id:
            raise ValueError(f"Task file ID does not match {task_id}")
        if task.status not in ("pending", "in_progress", "completed"):
            raise ValueError(f"Invalid task status: {task.status}")
        return task

    def list(self) -> list[Task]:
        if not self.directory.exists():
            return []
        root = self._root()
        return [self.load(path.stem)
                for path in sorted(root.glob("task_*.json"))]


TASKS = TaskStore(TASKS_DIR)


def create_task(subject: str, description: str = "") -> Task:
    return TASKS.create(subject, description)


def update_task(task_id: str, addBlockedBy: list[str]) -> Task:
    return TASKS.update_dependencies(task_id, addBlockedBy)


def load_task(task_id: str) -> Task:
    return TASKS.load(task_id)


def list_tasks() -> list[Task]:
    return TASKS.list()


def get_task(task_id: str) -> str:
    return json.dumps(asdict(load_task(task_id)), indent=2)


def incomplete_dependencies(task: Task) -> list[str]:
    incomplete = []
    for dependency in task.blockedBy:
        try:
            if load_task(dependency).status != "completed":
                incomplete.append(dependency)
        except (FileNotFoundError, ValueError):
            incomplete.append(dependency)
    return incomplete


def can_start(task_id: str) -> bool:
    return not incomplete_dependencies(load_task(task_id))


def claim_task(task_id: str, owner: str = "agent") -> str:
    task = load_task(task_id)
    if task.status != "pending":
        return f"Task {task_id} is {task.status}, cannot claim"
    dependencies = incomplete_dependencies(task)
    if dependencies:
        return f"Blocked by: {dependencies}"
    task.owner = owner
    task.status = "in_progress"
    TASKS.save(task)
    print(f"  [claim] {task.subject} -> in_progress (owner: {owner})")
    return f"Claimed {task.id} ({task.subject})"


def complete_task(task_id: str, owner: str = "agent") -> str:
    task = load_task(task_id)
    if task.status != "in_progress":
        return f"Task {task_id} is {task.status}, cannot complete"
    if task.owner != owner:
        return f"Task {task_id} is owned by {task.owner}, not {owner}"
    ready_before = {
        candidate.id
        for candidate in list_tasks()
        if candidate.status == "pending"
        and candidate.blockedBy
        and can_start(candidate.id)
    }
    task.status = "completed"
    TASKS.save(task)
    unblocked = [candidate.subject for candidate in list_tasks()
                 if candidate.status == "pending"
                 and candidate.blockedBy
                 and candidate.id not in ready_before
                 and can_start(candidate.id)]
    print(f"  [complete] {task.subject}")
    message = f"Completed {task.id} ({task.subject})"
    if unblocked:
        message += f"\nUnblocked: {', '.join(unblocked)}"
        print(f"  [unblocked] {', '.join(unblocked)}")
    return message


# -- From s04: tool implementations --

def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True, errors="replace",
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = (WORKDIR / path).resolve().read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as error:
        return f"Error: {error}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8", newline="")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as error:
        return f"Error: {error}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8", newline="")
        return f"Edited {path}"
    except Exception as error:
        return f"Error: {error}"


def run_glob(pattern: str) -> str:
    try:
        matches = sorted({
            Path(match).as_posix()
            for match in glob.glob(pattern, root_dir=WORKDIR, recursive=True)
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR)
        })
        shown = matches[:200]
        if len(matches) > 200:
            shown.append("... (more matches omitted; narrow the pattern)")
        return "\n".join(shown) if shown else "(no matches)"
    except Exception as error:
        return f"Error: {error}"


def run_create_task(subject: str, description: str = "") -> str:
    task = create_task(subject, description)
    print(f"  [create] {task.subject}")
    return f"Created {task.id}: {task.subject}"


def run_update_task(task_id: str, addBlockedBy: list[str]) -> str:
    task = update_task(task_id, addBlockedBy)
    dependencies = ", ".join(task.blockedBy) or "(none)"
    print(f"  [update] {task.subject} blockedBy: {dependencies}")
    return f"Updated {task.id} blockedBy: {dependencies}"


def run_list_tasks() -> str:
    tasks = list_tasks()
    if not tasks:
        return "No tasks. Use create_task to add some."
    lines = []
    for task in tasks:
        marker = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "completed": "[x]",
        }.get(task.status, "[?]")
        dependencies = (
            f" (blockedBy: {', '.join(task.blockedBy)})"
            if task.blockedBy else ""
        )
        owner = f" [{task.owner}]" if task.owner else ""
        lines.append(
            f"{marker} {task.id}: {task.subject} "
            f"[{task.status}]{owner}{dependencies}"
        )
    return "\n".join(lines)


def run_get_task(task_id: str) -> str:
    return get_task(task_id)


def run_claim_task(task_id: str) -> str:
    return claim_task(task_id, owner="agent")


def run_complete_task(task_id: str) -> str:
    return complete_task(task_id, owner="agent")


TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern; ** matches recursively.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "create_task", "description": "Create a task and return its runtime-generated ID.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"], "additionalProperties": False}},
    {"name": "update_task", "description": "Add dependencies using IDs returned by create_task.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string", "pattern": "^task_[0-9a-f]{8}$"}, "addBlockedBy": {"type": "array", "items": {"type": "string", "pattern": "^task_[0-9a-f]{8}$"}, "minItems": 1}}, "required": ["task_id", "addBlockedBy"], "additionalProperties": False}},
    {"name": "list_tasks", "description": "List tasks with status, owner, and dependencies.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_task", "description": "Get a task by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}},
    {"name": "claim_task", "description": "Claim a pending task whose dependencies are complete.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}},
    {"name": "complete_task", "description": "Complete the task claimed by this agent.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}},
]

TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
    "create_task": run_create_task,
    "update_task": run_update_task,
    "list_tasks": run_list_tasks,
    "get_task": run_get_task,
    "claim_task": run_claim_task,
    "complete_task": run_complete_task,
}


# -- From s04: hooks and permission checks --

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}


def register_hook(event: str, callback):
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def contains_destructive_command(command: str) -> bool:
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))


def permission_hook(block):
    if block.name == "bash":
        command = block.input.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                print(f"\n\033[31m[blocked] '{pattern}'\033[0m")
                return "Permission denied by deny list"
        if contains_destructive_command(command) or any(
            keyword in command for keyword in DESTRUCTIVE
        ):
            print("\n\033[33m[permission] Potentially destructive command\033[0m")
            print(f"   Tool: {block.name}({block.input})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"

    if block.name in ("read_file", "write_file", "edit_file"):
        path = block.input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print("\n\033[33m[permission] Access outside workspace\033[0m")
            print(f"   Tool: {block.name}({block.input})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    return None


def log_hook(block):
    preview = str(list(block.input.values())[:2])[:60]
    print(f"\033[90m[HOOK] {block.name}({preview})\033[0m")
    return None


def large_output_hook(block, output):
    if len(str(output)) > 100000:
        print(
            f"\033[33m[HOOK] Large output from {block.name}: "
            f"{len(str(output))} chars\033[0m"
        )
    return None


def context_hook(query: str):
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None


def summary_hook(messages: list):
    tool_count = sum(
        1
        for message in messages
        for block in (
            message.get("content")
            if isinstance(message.get("content"), list)
            else []
        )
        if isinstance(block, dict) and block.get("type") == "tool_result"
    )
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None


register_hook("UserPromptSubmit", context_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


def execute_tool(block) -> str:
    blocked = trigger_hooks("PreToolUse", block)
    if blocked:
        return str(blocked)

    handler = TOOL_HANDLERS.get(block.name)
    try:
        output = handler(**block.input) if handler else f"Unknown: {block.name}"
    except Exception as error:
        output = f"Error: {error}"

    trigger_hooks("PostToolUse", block, output)
    return str(output)


# -- Agent loop --

def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        if not tool_calls:
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return

        results = []
        for block in tool_calls:
            output = execute_tool(block)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("s10: Task System - dependencies and task state")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            # \001/\002 tell Readline the ANSI escapes have zero display width.
            query = input("\001\033[36m\002s10 >> \001\033[0m\002")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        agent_loop(history)
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(block.text)
        print()
