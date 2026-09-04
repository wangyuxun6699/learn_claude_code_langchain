"""
s14: MCP Tools - discover external tools and add them to the agent loop.

Run:  python s14_mcp_plugin/code.py
Need: pip install -r requirements.txt + .env with OPENAI_API_KEY / BASE_URL / MODEL_ID

    connect_mcp("docs")
              |
              v
    +------------------+     tools/list     +------------------+
    | Agent Harness    | <----------------- | MCP server       |
    |                  |                    | docs             |
    | built-in tools   |     tools/call     |                  |
    | + MCP tools      | -----------------> | search           |
    +--------+---------+                    | get_version      |
             |                              +------------------+
             v
    +-----------------------------------------------+
    | bash | read | write | edit | glob | connect  |
    | mcp__docs__search | mcp__docs__get_version   |
    +-----------------------------------------------+
"""

import glob
import os
import re
import subprocess
from pathlib import Path

try:
    import readline
    readline.parse_and_bind("set bind-tty-special-chars off")
except ImportError:
    pass

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

from dotenv import load_dotenv

load_dotenv(override=True)
WORKDIR = Path.cwd()
client = LangChainMessagesClient(base_url=os.getenv("BASE_URL"))
MODEL = os.environ["MODEL_ID"]

BASE_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Use built-in and connected MCP "
    "tools to solve tasks. Call connect_mcp before using a server."
)

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
        output = output[:50000] if output else "(no output)"
        if result.returncode:
            return f"Error: command exited with status {result.returncode}\n{output}"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except OSError as exc:
        return f"Error: {type(exc).__name__}: {exc}"

def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = (WORKDIR / path).resolve().read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as exc:
        return f"Error: {exc}"

def run_write(path: str, content: str) -> str:
    try:
        target = (WORKDIR / path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"Error: {exc}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        target = (WORKDIR / path).resolve()
        content = target.read_text(encoding="utf-8")
        count = content.count(old_text)
        if count != 1:
            return f"Error: Expected 1 occurrence, found {count}"
        target.write_text(content.replace(old_text, new_text), encoding="utf-8", newline="")
        return f"Edited {path}"
    except Exception as exc:
        return f"Error: {exc}"

def run_glob(pattern: str) -> str:
    try:
        matches = sorted({
            Path(match).as_posix()
            for match in glob.glob(pattern, root_dir=WORKDIR, recursive=True)
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR.resolve())
        })
        shown = matches[:200]
        if len(matches) > 200:
            shown.append("... (more matches omitted; narrow the pattern)")
        return "\n".join(shown) if shown else "(no matches)"
    except Exception as exc:
        return f"Error: {exc}"

BASE_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object",
                      "properties": {"command": {"type": "string"}},
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
    {"name": "edit_file", "description": "Replace exact text once.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "old_text": {"type": "string"},
                                     "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files by glob pattern; ** matches recursively.",
     "input_schema": {"type": "object",
                      "properties": {"pattern": {"type": "string"}},
                      "required": ["pattern"]}},
]

BASE_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}

class MCPClient:
    """Small in-process stand-in for MCP tools/list and tools/call."""

    def __init__(self, name: str):
        self.name = name
        self.tools: list[dict] = []
        self._handlers: dict[str, callable] = {}

    def register(self, tool_defs: list[dict], handlers: dict[str, callable]):
        names = [tool.get("name") for tool in tool_defs]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Every MCP tool needs a non-empty name")
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate MCP tool name on server {self.name!r}")
        missing = [name for name in names if name not in handlers]
        if missing:
            raise ValueError(f"Missing MCP handlers: {', '.join(missing)}")
        self.tools = list(tool_defs)
        self._handlers = dict(handlers)

    def call_tool(self, tool_name: str, args: dict) -> str:
        handler = self._handlers.get(tool_name)
        if not handler:
            return f"MCP error: unknown tool '{tool_name}'"
        try:
            return str(handler(**args))
        except Exception as exc:
            return f"MCP error: {type(exc).__name__}: {exc}"

mcp_clients: dict[str, MCPClient] = {}
mcp_tool_policies: dict[str, str] = {}
_DISALLOWED_CHARS = re.compile(r"[^a-zA-Z0-9_-]")

MCP_HOST_POLICY = {
    ("docs", "search"): "allow",
    ("docs", "get_version"): "allow",
    ("deploy", "status"): "allow",
    ("deploy", "trigger"): "confirm",
}

def normalize_mcp_name(name: str) -> str:
    """Replace characters outside the model tool-name alphabet."""
    normalized = _DISALLOWED_CHARS.sub("_", name)
    if not normalized:
        raise ValueError("MCP names cannot normalize to an empty string")
    return normalized

def _mock_server_docs() -> MCPClient:
    server = MCPClient("docs")
    server.register(
        tool_defs=[
            {
                "name": "search",
                "description": "Search the documentation.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "annotations": {"readOnlyHint": True},
            },
            {
                "name": "get_version",
                "description": "Get the documentation API version.",
                "inputSchema": {"type": "object", "properties": {}},
                "annotations": {"readOnlyHint": True},
            },
        ],
        handlers={
            "search": lambda query: f"[docs] Found 3 results for '{query}'",
            "get_version": lambda: "[docs] API v2.1.0",
        },
    )
    return server

def _mock_server_deploy() -> MCPClient:
    server = MCPClient("deploy")
    server.register(
        tool_defs=[
            {
                "name": "trigger",
                "description": "Trigger a deployment.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                },
                "annotations": {"destructiveHint": True},
            },
            {
                "name": "status",
                "description": "Check deployment status.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                },
                "annotations": {"readOnlyHint": True},
            },
        ],
        handlers={
            "trigger": lambda service: f"[deploy] Triggered: {service}",
            "status": lambda service: f"[deploy] {service}: running (v1.4.2)",
        },
    )
    return server

MOCK_SERVERS = {
    "docs": _mock_server_docs,
    "deploy": _mock_server_deploy,
}

def connect_mcp(name: str) -> str:
    if name in mcp_clients:
        return f"MCP server '{name}' already connected"
    factory = MOCK_SERVERS.get(name)
    if not factory:
        return f"Unknown server '{name}'. Available: {', '.join(MOCK_SERVERS)}"
    server = factory()
    mcp_clients[name] = server
    names = ", ".join(tool["name"] for tool in server.tools)
    print(f"  [mcp] connected: {name} -> {names}")
    return (
        f"Connected to MCP server '{name}'. "
        f"Discovered {len(server.tools)} tools: {names}"
    )

def run_connect_mcp(name: str) -> str:
    return connect_mcp(name)

CONNECT_TOOL = {
    "name": "connect_mcp",
    "description": "Connect to an MCP server and discover its tools.",
    "input_schema": {
        "type": "object",
        "properties": {"name": {"type": "string", "enum": ["docs", "deploy"]}},
        "required": ["name"],
    },
}

BUILTIN_TOOLS = [*BASE_TOOLS, CONNECT_TOOL]
BUILTIN_HANDLERS = {**BASE_HANDLERS, "connect_mcp": run_connect_mcp}

def assemble_tool_pool() -> tuple[list[dict], dict[str, callable]]:
    """Combine built-in tools with every connected server tool."""
    global mcp_tool_policies
    tools = list(BUILTIN_TOOLS)
    handlers = dict(BUILTIN_HANDLERS)
    policies: dict[str, str] = {}
    origins = {
        tool["name"]: f"built-in tool {tool['name']!r}"
        for tool in tools
    }

    for server_name, server in mcp_clients.items():
        safe_server = normalize_mcp_name(server_name)
        for tool_def in server.tools:
            raw_name = tool_def["name"]
            safe_tool = normalize_mcp_name(raw_name)
            prefixed = f"mcp__{safe_server}__{safe_tool}"
            if len(prefixed) > 64:
                raise ValueError(f"MCP tool name is longer than 64 characters: {prefixed}")
            origin = f"MCP tool {server_name!r}/{raw_name!r}"
            if prefixed in origins:
                raise ValueError(
                    "MCP tool name collision after normalization: "
                    f"{prefixed!r} maps both {origins[prefixed]} and {origin}"
                )
            schema = tool_def.get("inputSchema", {})
            if not isinstance(schema, dict) or schema.get("type", "object") != "object":
                raise ValueError(f"Invalid input schema for {origin}")
            origins[prefixed] = origin
            tools.append({
                "name": prefixed,
                "description": tool_def.get("description", ""),
                "input_schema": schema,
            })
            handlers[prefixed] = (
                lambda *, client=server, tool=raw_name, **kwargs:
                client.call_tool(tool, kwargs)
            )
            policies[prefixed] = MCP_HOST_POLICY.get(
                (server_name, raw_name), "confirm"
            )

    mcp_tool_policies = policies
    return tools, handlers

def assemble_system_prompt() -> str:
    if not mcp_clients:
        return BASE_SYSTEM
    return BASE_SYSTEM + "\n\nConnected MCP servers: " + ", ".join(mcp_clients)

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]

def contains_destructive_command(command: str) -> bool:
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))

def register_hook(event: str, callback):
    HOOKS[event].append(callback)

def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None

def permission_hook(block):
    if block.name == "bash":
        command = block.input.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                return f"Permission denied by deny list: {pattern}"
        if contains_destructive_command(command) or any(
            keyword in command for keyword in DESTRUCTIVE
        ):
            print(f"\n[permission] {block.name}({block.input})")
            if input("Allow? [y/N] ").strip().lower() not in {"y", "yes"}:
                return "Permission denied by user"

    if block.name in {"read_file", "write_file", "edit_file"}:
        raw_path = block.input.get("path", "")
        if not (WORKDIR / raw_path).resolve().is_relative_to(WORKDIR.resolve()):
            print(f"\n[permission] {block.name}({block.input})")
            if input("Allow? [y/N] ").strip().lower() not in {"y", "yes"}:
                return "Permission denied by user"

    if block.name.startswith("mcp__"):
        policy = mcp_tool_policies.get(block.name, "confirm")
        if policy != "allow":
            print(f"\n[permission] External tool {block.name}({block.input})")
            if input("Allow? [y/N] ").strip().lower() not in {"y", "yes"}:
                return "Permission denied by user"
    return None

def log_hook(block):
    preview = str(list(block.input.values())[:2])[:60]
    print(f"[hook] {block.name}({preview})")
    return None

def large_output_hook(block, output):
    if len(str(output)) > 100000:
        print(f"[hook] Large output from {block.name}: {len(str(output))} chars")
    return None

def context_hook(query: str):
    print(f"[hook] UserPromptSubmit: working in {WORKDIR}")
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
    print(f"[hook] Stop: session used {tool_count} tool calls")
    return None

register_hook("UserPromptSubmit", context_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)

def execute_tool(block, handlers: dict[str, callable]) -> str:
    blocked = trigger_hooks("PreToolUse", block)
    if blocked:
        return str(blocked)
    handler = handlers.get(block.name)
    if not handler:
        return f"Unknown tool: {block.name}"
    try:
        output = str(handler(**block.input))
    except Exception as exc:
        output = f"Error: {type(exc).__name__}: {exc}"
    trigger_hooks("PostToolUse", block, output)
    return output

def agent_loop(messages: list):
    while True:
        try:
            tools, handlers = assemble_tool_pool()
            response = client.messages.create(
                model=MODEL,
                system=assemble_system_prompt(),
                messages=messages,
                tools=tools,
                max_tokens=8000,
            )
        except Exception as exc:
            messages.append({
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": f"[Error] {type(exc).__name__}: {exc}",
                }],
            })
            trigger_hooks("Stop", messages)
            return

        messages.append({"role": "assistant", "content": response.content})
        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        if not tool_calls:
            trigger_hooks("Stop", messages)
            return

        results = []
        for block in tool_calls:
            print(f"> {block.name}")
            output = execute_tool(block, handlers)
            print(output[:300])
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        messages.append({"role": "user", "content": results})

if __name__ == "__main__":
    print("s14: MCP tools")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history = []

    while True:
        try:
            query = input("s14 >> ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in {"q", "exit", ""}:
            break
        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        agent_loop(history)
        for block in history[-1].get("content", []):
            if getattr(block, "type", None) == "text":
                print(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                print(block.get("text", ""))
        print()
