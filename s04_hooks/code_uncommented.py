import glob as glob_module
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

try:
    import readline

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

def resolve_path(path: str) -> Path:
    return (WORKDIR / path).resolve()

def print_tool_result(name: str, detail: str, output: str) -> str:
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

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}

def register_hook(event: str, callback) -> None:
    HOOKS[event].append(callback)

def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None

@dataclass(slots=True)
class ToolUseBlock:

    id: str
    name: str
    input: dict[str, Any]

DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]

def check_deny_list(command: str) -> str | None:
    normalized = command.lower()
    for pattern in DENY_LIST:
        if pattern in normalized:
            return f"Blocked: '{pattern}' is on the deny list"
    return None

DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)

def contains_destructive_command(command: str) -> bool:
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
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None

def ask_user(tool_name: str, args: dict, reason: str) -> str:
    print(f"\n\033[33m[permission] {reason}\033[0m")
    print(f"   Tool: {tool_name}({args})")
    try:
        choice = input("   Allow? [y/N] ").strip().lower()
    except EOFError:
        choice = ""
    return "allow" if choice in ("y", "yes") else "deny"

APPROVAL_LOCK = Lock()

def permission_hook(block: ToolUseBlock) -> str | None:
    if block.name == "bash":
        reason = check_deny_list(block.input.get("command", ""))
        if reason:
            print(f"\n\033[31m[blocked] {reason}\033[0m")
            return reason

    reason = check_rules(block.name, block.input)
    if reason:
        with APPROVAL_LOCK:
            if ask_user(block.name, block.input, reason) != "allow":
                return "Permission denied by user"
    return None

def log_hook(block: ToolUseBlock) -> None:
    args_preview = str(list(block.input.values())[:2])[:60]
    print(f"\033[90m[HOOK] {block.name}({args_preview})\033[0m")

def large_output_hook(block: ToolUseBlock, output: Any) -> None:
    size = len(str(output))
    if size > 100000:
        print(f"\033[33m[HOOK] Large output from {block.name}: {size} chars\033[0m")

def context_inject_hook(query: str) -> None:
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")

def _tool_result_count(messages: list) -> int:
    count = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            count += 1
            continue
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            count += sum(
                isinstance(block, ToolMessage)
                or (isinstance(block, dict) and block.get("type") == "tool_result")
                for block in content
            )
    return count

def summary_hook(messages: list) -> None:
    print(f"\033[90m[HOOK] Stop: session used {_tool_result_count(messages)} tool calls\033[0m")

register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)

def _tool_output(result: Any) -> Any:
    return result.content if isinstance(result, ToolMessage) else result

class HookMiddleware(AgentMiddleware):

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        tool_call = request.tool_call
        block = ToolUseBlock(
            id=tool_call["id"],
            name=tool_call["name"],
            input=dict(tool_call.get("args") or {}),
        )

        blocked = trigger_hooks("PreToolUse", block)
        if blocked is not None:
            return ToolMessage(
                content=str(blocked),
                tool_call_id=block.id,
                name=block.name,
                status="error",
            )

        result = handler(request)
        trigger_hooks("PostToolUse", block, _tool_output(result))
        return result

agent = None

def get_agent():
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
            middleware=[HookMiddleware()],
        )
    return agent

def agent_loop(messages: list) -> None:
    while True:
        final_messages = messages
        for chunk in get_agent().stream(
            {"messages": messages},
            stream_mode=["messages", "values"],
            version="v2",
        ):
            if chunk["type"] == "messages":
                token, metadata = chunk["data"]
                if (
                    isinstance(token, AIMessageChunk)
                    and metadata.get("langgraph_node") == "model"
                    and token.text
                ):
                    print(token.text, end="", flush=True)
            elif chunk["type"] == "values":
                final_messages = chunk["data"]["messages"]

        messages[:] = final_messages
        continuation = trigger_hooks("Stop", messages)
        if continuation is None:
            break
        messages.append({"role": "user", "content": str(continuation)})
    print()

if __name__ == "__main__":
    print("s04: Hooks (LangChain middleware + lifecycle registry)")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:

            query = input("\001\033[36m\002s04 >> \001\033[0m\002")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        agent_loop(history)
        print()
