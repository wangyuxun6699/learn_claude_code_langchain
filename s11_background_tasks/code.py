#!/usr/bin/env python3
"""
s11_background_tasks.py - Background Tasks

    Main thread                              Background thread
    +------------------------------+         +----------------------+
    | bash(run_in_background=True) | ------> | run command          |
    | return bg_id                 |         | queue result         |
    | continue agent loop          | <------ +----------------------+
    | next turn: collect           |
    +------------------------------+
"""

import atexit
import glob
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    pass

# -- Local LangChain message adapter (expanded for single-file reading) --
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
    """Text content block consumed by the lesson loop."""

    text: str
    type: str = field(default="text", init=False)


@dataclass(slots=True)
class ToolUseBlock:
    """Tool-call content block consumed by the lesson loop."""

    id: str
    name: str
    input: dict[str, Any]
    type: str = field(default="tool_use", init=False)


@dataclass(slots=True)
class Usage:
    """Token usage fields used by workflow and goal accounting."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(slots=True)
class MessageResponse:
    """Minimal model response consumed by the lessons."""

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
    """Convert provider blocks, tool results, or plain objects to text."""
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
    """Model boundary; network requests are deferred until ``create``."""

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

# -- Agent / Harness mechanisms for this lesson --
# -- Cross-platform Bash and process-tree cleanup --
import shutil


def _git_bash() -> Path | None:
    """Prefer the Bash installation bundled with Git for Windows."""
    if os.name != "nt":
        return None

    candidates: list[Path] = []
    git = shutil.which("git")
    if git:
        git_root = Path(git).resolve().parent.parent
        candidates.append(git_root / "bin" / "bash.exe")

    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(variable)
        if root:
            candidates.append(Path(root) / "Git" / "bin" / "bash.exe")

    return next((path for path in candidates if path.is_file()), None)


def shell_invocation(command: str) -> tuple[str | list[str], bool]:
    """Return the platform-specific ``Popen`` command and ``shell`` flag."""
    bash = _git_bash()
    if bash is not None:
        return [str(bash), "-lc", command], False
    return command, True


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL


def attach_kill_on_close_job(process: subprocess.Popen[Any]) -> bool:
    """Attach a Windows child to a kill-on-close job object."""
    if os.name != "nt":
        return False

    job = _kernel32.CreateJobObjectW(None, None)
    if not job:
        return False

    information = _ExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    configured = _kernel32.SetInformationJobObject(
        job,
        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(information),
        ctypes.sizeof(information),
    )
    assigned = configured and _kernel32.AssignProcessToJobObject(
        job, wintypes.HANDLE(process._handle)
    )
    if not assigned:
        _kernel32.CloseHandle(job)
        return False

    process._kill_on_close_job = job
    return True


def terminate_process_tree(process: subprocess.Popen[Any]) -> bool:
    """Terminate the attached Windows job; return whether it was handled."""
    if os.name != "nt":
        return False
    job = getattr(process, "_kill_on_close_job", None)
    if not job:
        return False
    _kernel32.TerminateJobObject(job, 1)
    return True


def close_process_job(process: subprocess.Popen[Any]) -> None:
    """Close the job handle and clean up remaining descendants."""
    if os.name != "nt":
        return
    job = getattr(process, "_kill_on_close_job", None)
    if not job:
        return
    _kernel32.CloseHandle(job)
    process._kill_on_close_job = None

from dotenv import load_dotenv

load_dotenv(override=True)
WORKDIR = Path.cwd()
client = LangChainMessagesClient(base_url=os.getenv("BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. "
    "Set run_in_background to true only for independent Bash commands."
)


# -- From s04: tool implementations --

_shell_processes: set[subprocess.Popen] = set()
_shell_process_lock = threading.RLock()


def _stop_process_group(process: subprocess.Popen):
    """Stop processes that remain in the command's original process group."""
    if os.name == "nt":
        if terminate_process_tree(process):
            return
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=0.05)
        except subprocess.TimeoutExpired:
            process.kill()
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, OSError):
            return
        time.sleep(0.05)


def _stop_all_shell_processes():
    with _shell_process_lock:
        processes = list(_shell_processes)
    for process in processes:
        _stop_process_group(process)


def _handle_termination_signal(signum, _frame):
    _stop_all_shell_processes()
    raise SystemExit(128 + signum)


atexit.register(_stop_all_shell_processes)
signal.signal(signal.SIGTERM, _handle_termination_signal)


def _run_bash_process(command: str) -> tuple[str, int | None]:
    process = None
    try:
        invocation, use_shell = shell_invocation(command)
        process = subprocess.Popen(
            invocation,
            shell=use_shell,
            cwd=WORKDIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, errors="replace",
            start_new_session=True,
        )
        attach_kill_on_close_job(process)
        with _shell_process_lock:
            _shell_processes.add(process)
        stdout, stderr = process.communicate(timeout=120)
        output = (stdout + stderr).strip()
        return (output[:50000] if output else "(no output)"), process.returncode
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)", None
    except OSError as error:
        return f"Error: {type(error).__name__}: {error}", None
    finally:
        if process is not None:
            _stop_process_group(process)
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
            close_process_job(process)
            with _shell_process_lock:
                _shell_processes.discard(process)


def _format_bash_result(output: str, exit_code: int | None) -> str:
    if exit_code in (0, None):
        return output
    return f"Error: command exited with status {exit_code}\n{output}"


def run_bash(command: str, run_in_background: bool = False) -> str:
    return _format_bash_result(*_run_bash_process(command))


def run_read(path: str, limit: int | None = None) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        lines = file_path.read_text(encoding="utf-8").splitlines()
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


TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object",
                      "properties": {
                          "command": {"type": "string"},
                          "run_in_background": {"type": "boolean"}},
                      "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "limit": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "old_text": {"type": "string"},
                                     "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern; ** matches recursively.",
     "input_schema": {"type": "object",
                      "properties": {"pattern": {"type": "string"}},
                      "required": ["pattern"]}},
]

TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
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


def context_inject_hook(query: str):
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


register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


def call_tool(block) -> str:
    handler = TOOL_HANDLERS.get(block.name)
    try:
        output = handler(**block.input) if handler else f"Unknown: {block.name}"
    except Exception as error:
        output = f"Error: {error}"
    return str(output)


# -- New in s11: background execution --

class BackgroundManager:
    def __init__(self):
        self.tasks: dict[str, dict] = {}
        self.results: dict[str, str] = {}
        self._ready: list[str] = []
        self._counter = 0
        self._lock = threading.Lock()

    def start(self, block) -> str:
        if block.name != "bash":
            raise ValueError("Only Bash commands can run in the background")
        command = block.input.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("Bash command cannot be empty")

        with self._lock:
            self._counter += 1
            task_id = f"bg_{self._counter:04d}"
            self.tasks[task_id] = {
                "tool_use_id": block.id,
                "command": command,
                "status": "running",
            }

        thread = threading.Thread(
            target=self._run,
            args=(task_id, command),
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            with self._lock:
                self.tasks.pop(task_id, None)
            raise
        print(f"  [background] started {task_id}: {command[:60]}")
        return task_id

    def _run(self, task_id: str, command: str):
        try:
            output, exit_code = _run_bash_process(command)
            result = _format_bash_result(output, exit_code)
            status = "completed" if exit_code == 0 else "failed"
        except Exception as error:
            result = f"Error: {type(error).__name__}: {error}"
            status = "failed"

        with self._lock:
            task = self.tasks.get(task_id)
            if task is None:
                return
            task["status"] = status
            self.results[task_id] = result
            self._ready.append(task_id)

    def collect(self) -> list[str]:
        with self._lock:
            ready = []
            for task_id in self._ready:
                task = self.tasks.pop(task_id, None)
                result = self.results.pop(task_id, "")
                if task is not None:
                    ready.append((task_id, task, result))
            self._ready.clear()

        notifications = []
        for task_id, task, result in ready:
            notifications.append(
                f"<task_notification>\n"
                f"  <task_id>{task_id}</task_id>\n"
                f"  <status>{task['status']}</status>\n"
                f"  <command>{task['command']}</command>\n"
                f"  <summary>{result[:500]}</summary>\n"
                f"</task_notification>"
            )
            print(f"  [background] collected {task_id}: {task['status']}")
        return notifications


BACKGROUND = BackgroundManager()
background_tasks = BACKGROUND.tasks
background_results = BACKGROUND.results


def should_run_background(tool_name: str, tool_input: dict) -> bool:
    return (
        tool_name == "bash"
        and tool_input.get("run_in_background") is True
    )


def start_background_task(block) -> str:
    return BACKGROUND.start(block)


def collect_background_results() -> list[str]:
    return BACKGROUND.collect()


def inject_background_results(messages: list) -> int:
    notifications = collect_background_results()
    if not notifications:
        return 0

    blocks = [{"type": "text", "text": item} for item in notifications]
    if messages and messages[-1].get("role") == "user":
        content = messages[-1].get("content", "")
        if isinstance(content, list):
            content.extend(blocks)
        else:
            messages[-1]["content"] = [
                {"type": "text", "text": str(content)},
                *blocks,
            ]
    else:
        messages.append({"role": "user", "content": blocks})
    return len(notifications)


def execute_tool(block) -> str:
    blocked = trigger_hooks("PreToolUse", block)
    if blocked is not None:
        return str(blocked)

    if should_run_background(block.name, block.input):
        try:
            task_id = start_background_task(block)
            output = (
                f"[Background task {task_id} started] "
                "The result will be collected on a later turn."
            )
        except Exception as error:
            output = f"Error: {error}"
    else:
        output = call_tool(block)

    trigger_hooks("PostToolUse", block, output)
    return output


# -- Agent loop --

def agent_loop(messages: list):
    while True:
        inject_background_results(messages)
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
    print("s11: Background Tasks - explicit background Bash execution")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            # \001/\002 tell Readline the ANSI escapes have zero display width.
            query = input("\001\033[36m\002s11 >> \001\033[0m\002")
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
