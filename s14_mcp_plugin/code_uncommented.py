"""s14: MCP Tools -- discover and invoke external tools (uncommented)."""
import functools
import re
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, TypedDict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv(override=True)

MODEL_ID = os.getenv("MODEL_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("BASE_URL")
WORKDIR = Path.cwd()

SYSTEM = (
    f"you are a coding agent at {WORKDIR}. Use tools to solve tasks. "
    "External tools are available through connect_mcp; connect first, then use them. "
    "Act, don't explain."
)

HOOKS = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": [],
}


def register_hook(event: str, callback):
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]


def check_deny_list(command: str) -> str | None:
    for pattern in dangerous:
        if pattern in command:
            return f"blocked:{pattern} is on the deny list"
    return None


def resolve_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (WORKDIR / candidate).resolve()


def check_rules(tool_name: str, args: dict) -> str | None:
    if tool_name == "run_bash":
        command = args.get("command", "")
        if command.strip().lower().startswith("del ") or any(kw in command for kw in ["rm ", "> /etc/", "chmod 777"]):
            return "potentially destructive command"
    if tool_name in ("run_write", "run_edit", "run_read"):
        path = args.get("path", "")
        if not resolve_path(path).is_relative_to(WORKDIR):
            return "Working outside workspace"
    return None


def ask_user(tool_name: str, args: dict, reason: str) -> bool:
    print(f"\nWarning: {reason}")
    print(f"Tool: {tool_name}({args})")
    choice = input("Allow? [y/N] ").strip().lower()
    return choice in ("y", "yes")


MCP_HOST_POLICY = {
    "mcp__docs__search": "allow",
    "mcp__docs__get_version": "allow",
    "mcp__deploy__status": "allow",
    "mcp__deploy__trigger": "confirm",
}


def check_mcp_permission(tool_name: str) -> bool:
    decision = MCP_HOST_POLICY.get(tool_name, "confirm")
    if decision == "allow":
        return True
    if decision == "deny":
        print(f"\nBlocked: MCP tool {tool_name} denied by host policy")
        return False
    return ask_user(tool_name, {}, "external MCP tool requires host confirmation")


def check_permission(tool_name: str, args: dict) -> bool:
    if tool_name.startswith("mcp__"):
        return check_mcp_permission(tool_name)
    if tool_name == "run_bash":
        reason = check_deny_list(args.get("command", ""))
        if reason:
            print(f"\nBlocked:{reason}")
            return False
    reason = check_rules(tool_name, args)
    if reason:
        return ask_user(tool_name, args, reason)
    return True


def on_user_prompt_submit(content):
    print("[UserPromptSubmit]", content)


def on_pre_tool_use(tool_name, tool_args):
    print("[PreToolUse]", tool_name, tool_args)
    if not check_permission(tool_name, tool_args):
        return "Permission denied"


def on_post_tool_use(tool_name, tool_args, result):
    print("[PostToolUse]", tool_name, tool_args)
    print("result:", getattr(result, "content", result))


def on_stop(messages):
    print("[Stop]", len(messages))


register_hook("UserPromptSubmit", on_user_prompt_submit)
register_hook("PreToolUse", on_pre_tool_use)
register_hook("PostToolUse", on_post_tool_use)
register_hook("Stop", on_stop)


@tool
def run_bash(command: str) -> str:
    """Execute a shell command in the current workspace."""
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout(120s)"
    except OSError as e:
        return f"Error: {e}"


@tool
def run_read(path: str, limit: int | None = None) -> str:
    """Read a UTF-8 text file, optionally limiting the returned line count."""
    try:
        lines = resolve_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"...({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@tool
def run_write(path: str, content: str) -> str:
    """Write UTF-8 content to a file, replacing its existing content."""
    try:
        file_path = resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"write {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace the first exact occurrence of old_text in a UTF-8 file."""
    try:
        file_path = resolve_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"edit {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def run_glob(pattern: str) -> str:
    """Find workspace files matching a glob pattern."""
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


def normalize_mcp_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


class MCPClient:
    """An in-process stand-in for a connected MCP server's tool list and call boundary."""

    def __init__(self):
        self.tools = []
        self._handlers = {}

    def register(self, tool_defs, handlers):
        self.tools = list(tool_defs)
        self._handlers = dict(handlers)

    def call_tool(self, tool_name, args) -> str:
        handler = self._handlers.get(tool_name)
        if not handler:
            return f"MCP error: unknown tool '{tool_name}'"
        try:
            return str(handler(**args))
        except Exception as error:
            return f"MCP error: {type(error).__name__}: {error}"


def docs_server() -> MCPClient:
    client = MCPClient()

    def search(query: str, limit: int = 10) -> str:
        """Search the product documentation."""
        slug = query.replace(" ", "-")
        hits = [f"docs/{slug}/overview", f"docs/{slug}/getting-started", f"docs/{slug}/agent-hooks"][:limit]
        return "found:\n" + "\n".join(hits)

    def get_version() -> str:
        """Return the documentation API version."""
        return "docs API version 1.4.2"

    client.register(
        [
            {"name": "search", "description": "Search the product documentation."},
            {"name": "get_version", "description": "Return the documentation API version."},
        ],
        {"search": search, "get_version": get_version},
    )
    return client


def deploy_server() -> MCPClient:
    client = MCPClient()

    def status(service: str = "web") -> str:
        """Return the deployment status of a service."""
        return f"{service} service: healthy (replica 3/3)"

    def trigger(service: str) -> str:
        """Trigger a new deployment for a service (destructive)."""
        return f"triggered deployment for {service}"

    client.register(
        [
            {"name": "status", "description": "Return the deployment status of a service."},
            {"name": "trigger", "description": "Trigger a new deployment for a service (destructive)."},
        ],
        {"status": status, "trigger": trigger},
    )
    return client


MOCK_SERVERS: dict[str, Callable[[], MCPClient]] = {
    "docs": docs_server,
    "deploy": deploy_server,
}
mcp_clients: dict[str, MCPClient] = {}


@tool
def connect_mcp(name: str) -> str:
    """Connect to an MCP server by name ('docs' or 'deploy'), discovering its tools."""
    if name in mcp_clients:
        return f"MCP server '{name}' already connected"
    factory = MOCK_SERVERS.get(name)
    if not factory:
        return f"Unknown server '{name}'. Available: {', '.join(MOCK_SERVERS)}"
    client = factory()
    mcp_clients[name] = client
    discovered = ", ".join(f"mcp__{normalize_mcp_name(name)}__{normalize_mcp_name(t['name'])}" for t in client.tools)
    return f"Connected '{name}', discovered tools: {discovered}"


def _make_mcp_tool(client: MCPClient, raw_name: str, prefixed: str, description: str) -> StructuredTool:
    handler = client._handlers[raw_name]

    @functools.wraps(handler)
    def _runner(**kwargs):
        return client.call_tool(raw_name, kwargs)

    _runner.__name__ = prefixed.replace(".", "_").replace("-", "_")
    return StructuredTool.from_function(func=_runner, name=prefixed, description=description)


CURRENT_TOOL_MAP: dict[str, StructuredTool] = {}


def assemble_tool_pool() -> list:
    tools = [connect_mcp, run_bash, run_read, run_write, run_edit, run_glob]
    seen = {t.name for t in tools}
    for server_name, client in mcp_clients.items():
        safe_server = normalize_mcp_name(server_name)
        for raw in client.tools:
            safe_tool = normalize_mcp_name(raw["name"])
            prefixed = f"mcp__{safe_server}__{safe_tool}"
            if prefixed in seen:
                raise ValueError(f"MCP tool name collision after normalization: {prefixed}")
            if len(prefixed) > 64:
                raise ValueError(f"MCP tool name exceeds 64 chars: {prefixed}")
            seen.add(prefixed)
            tools.append(_make_mcp_tool(client, raw["name"], prefixed, raw["description"]))
    CURRENT_TOOL_MAP.clear()
    CURRENT_TOOL_MAP.update({t.name: t for t in tools})
    return tools


MODEL = ChatOpenAI(
    model=MODEL_ID,
    max_completion_tokens=8000,
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def model_node(state: AgentState) -> dict[str, Any]:
    tools = assemble_tool_pool()
    response = MODEL.bind_tools(tools).invoke(state["messages"])
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict[str, Any]:
    last = state["messages"][-1]
    outputs = []
    for tool_call in last.tool_calls:
        name = tool_call["name"]
        args = tool_call.get("args", {})
        blocked = trigger_hooks("PreToolUse", name, args)
        if blocked:
            outputs.append(ToolMessage(content=str(blocked), tool_call_id=tool_call["id"], name=name, status="error"))
            continue
        tool = CURRENT_TOOL_MAP.get(name)
        if tool is None:
            output = f"Error: unknown tool {name}"
        else:
            try:
                output = tool.invoke(args)
            except Exception as e:
                output = f"Error: {e}"
        result = ToolMessage(content=str(output), tool_call_id=tool_call["id"], name=name)
        trigger_hooks("PostToolUse", name, args, result)
        outputs.append(result)
    return {"messages": outputs}


def route(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"


graph_builder = StateGraph(AgentState)
graph_builder.add_node("model", model_node)
graph_builder.add_node("tools", tools_node)
graph_builder.add_edge(START, "model")
graph_builder.add_conditional_edges("model", route, {"tools": "tools", "end": END})
graph_builder.add_edge("tools", "model")
agent = graph_builder.compile()


def print_assistant_message(message: AIMessage) -> None:
    content = message.content
    if isinstance(content, str):
        print(content)
        return
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    print(block.get("text", ""))
            elif hasattr(block, "text"):
                print(block.text)


def print_tool_activity(message) -> None:
    if isinstance(message, AIMessage):
        for tool_call in message.tool_calls:
            name = tool_call["name"]
            args = tool_call.get("args", {})
            if name == "run_bash":
                print(f"\033[33m$ {args.get('command', '')}\033[0m")
            else:
                print(f"\033[33m{name}{args}\033[0m")
        return
    if isinstance(message, ToolMessage):
        content = str(message.content)
        if getattr(message, "status", None) == "error":
            print(f"(blocked/error) {content[:200]}")
        else:
            print(content[:200])


def agent_loop(messages: list) -> None:
    if messages:
        trigger_hooks("UserPromptSubmit", getattr(messages[-1], "content", messages[-1]))
    result = agent.invoke({"messages": messages})
    for message in result["messages"][len(messages):]:
        print_tool_activity(message)
    trigger_hooks("Stop", result["messages"])
    messages[:] = result["messages"]


if __name__ == "__main__":
    print("s14: MCP Tools")
    print("输入问题，回车发送。输入 q 退出。\n")
    history = []
    while True:
        try:
            query = input("\033[36ms14 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append(HumanMessage(content=query))
        agent_loop(history)
        last_message = history[-1]
        if isinstance(last_message, AIMessage):
            print_assistant_message(last_message)
        print()
