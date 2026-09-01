"""s14: MCP Tools -- 外部工具按需发现并接入同一个工具池。

主线：
1. 复用 s04 内核：五个基础工具（bash/read/write/edit/glob）+ Hook + 权限。
2. 增加 MCPClient，代表“某个 MCP server 暴露的工具列表和调用边界”。
3. 增加 connect_mcp 工具，按名字连接一个（mock）server 并发现它的工具。
4. assemble_tool_pool() 每轮重新组装：基础工具 + 所有已连接 server 的动态工具。
5. 因为工具池会在同一轮里变化（connect_mcp 之后再出现 mcp__docs__search），
   这里不再用 create_agent 一次性编译静态图，而是手写一个 LangGraph 循环：
   model 节点每次都用 assemble_tool_pool() 绑定最新工具，再决定是否继续调用工具。

这样正好展现 s01 里那句“create_agent = 编译好的 LangGraph runtime”的另一面：
当你需要在运行时改工具集合、或者接管返回边界时，也可以自己把这张图铺开。
"""

# functools.wraps 用于把 mock 工具的签名复制到转发包装器上，
# 让 Pydantic 能推断出正确的参数 schema（例如 search(query, limit)）。
import functools

# re 用于把任意 server/tool 名规范化为模型工具名允许的字符。
import re

# Callable 给 handler 类型做标注。
from collections.abc import Callable

# Annotated/TypedDict 用来声明 LangGraph 的状态结构。
from typing import Annotated, TypedDict, Any

# Path 处理文件路径；os/subprocess 执行命令和读环境变量。
from pathlib import Path
import os, subprocess

# load_dotenv 读取 .env，override=True 让 .env 覆盖已有同名环境变量。
from dotenv import load_dotenv
load_dotenv(override=True)

# ChatOpenAI：兼容 OpenAI 接口的 ChatModel。
from langchain_openai import ChatOpenAI

# @tool 把普通函数注册成工具；StructuredTool.from_function 可动态造工具。
from langchain_core.tools import tool, StructuredTool

# 消息类型：HumanMessage 用户消息、ToolMessage 工具返回、AIMessage 模型回复。
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

# LangGraph 手动建图所需的元件。add_messages 是消息列表的合并 reducer。
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

# s04 用 os.getenv，避免缺少环境变量时在导入阶段直接 KeyError。
MODEL_ID = os.getenv("MODEL_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("BASE_URL")

# 程序启动时所在的工作目录；文件工具和 bash 都在这里执行。
WORKDIR = Path.cwd()

# system prompt：告诉模型它是一个 coding agent，能用工具、需要时可连接 MCP。
SYSTEM = (
    f"you are a coding agent at {WORKDIR}. Use tools to solve tasks. "
    "External tools are available through connect_mcp; connect first, then use them. "
    "Act, don't explain."
)

# ---------------------------------------------------------------------------
# Hook 系统（与 s04 相同）
# ---------------------------------------------------------------------------

HOOKS = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": [],
}


def register_hook(event: str, callback):
    """把一个回调注册到指定生命周期事件。"""
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    """依次触发某事件下所有回调；第一个返回非 None 的结果会中断后续回调并返回。"""
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# 权限（s04 内核 + 新增 MCP 主机策略）
# ---------------------------------------------------------------------------

# 统一走 harness.security 的大小写不敏感、覆盖更广的拒绝策略。
from harness.security import check_deny_list


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


# MCP 主机策略：server 自带的描述不是授权，真正的决定权在 host 手里。
# 键是规范化后的工具名。未配置的外部工具默认走 "confirm"（要用户确认）。
MCP_HOST_POLICY = {
    "mcp__docs__search": "allow",
    "mcp__docs__get_version": "allow",
    "mcp__deploy__status": "allow",
    "mcp__deploy__trigger": "confirm",
}


def check_mcp_permission(tool_name: str) -> bool:
    """按主机策略决定一个 MCP 工具能否执行。"""
    # 未配置的默认 confirm：外部能力默认不信任。
    decision = MCP_HOST_POLICY.get(tool_name, "confirm")

    if decision == "allow":
        return True

    if decision == "deny":
        print(f"\nBlocked: MCP tool {tool_name} denied by host policy")
        return False

    # confirm：交给用户在终端确认。
    return ask_user(tool_name, {}, "external MCP tool requires host confirmation")


def check_permission(tool_name: str, args: dict) -> bool:
    """统一权限入口：MCP 工具走主机策略，本地工具走 denylist + 规则。"""
    # mcp__ 前缀的工具进入 MCP 主机策略分支。
    if tool_name.startswith("mcp__"):
        return check_mcp_permission(tool_name)

    # bash 风险最高，先过硬拦截列表。
    if tool_name == "run_bash":
        reason = check_deny_list(args.get("command", ""))
        if reason:
            print(f"\nBlocked:{reason}")
            return False

    # 普通规则可能需要问用户。
    reason = check_rules(tool_name, args)
    if reason:
        return ask_user(tool_name, args, reason)

    return True


# ---------------------------------------------------------------------------
# Hook 回调（与 s04 相同，on_pre_tool_use 里做权限检查）
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 基础工具（s04 内核）
# ---------------------------------------------------------------------------

@tool
def run_bash(command: str) -> str:
    """Execute a shell command in the current workspace."""
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=120,
        )
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
        lines = resolve_path(path).read_text(encoding="utf-8").splitlines()
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
        file_path.write_text(content, encoding="utf-8")
        return f"write {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace the first exact occurrence of old_text in a UTF-8 file."""
    try:
        file_path = resolve_path(path)
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
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


# ---------------------------------------------------------------------------
# MCP 部分：MCPClient + mock servers + connect_mcp
# ---------------------------------------------------------------------------

# 规范化工具名：模型工具名只允许字母数字和少量符号，其余统一替换成下划线。
def normalize_mcp_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


class MCPClient:
    """代表一个已连接的 MCP server。

    tools      是“发现”到的工具定义列表（name + description）。
    _handlers  是原始工具名 -> 实现函数的映射。
    call_tool  是调用边界：把错误转成字符串，绝不把异常抛回 agent 循环。
    """

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


# mock docs server：用进程内数据模拟 tools/list 和 tools/call。
# 真正的 MCP transport（JSON-RPC、OAuth、资源订阅）不在本章范围内。
def docs_server() -> MCPClient:
    client = MCPClient()

    def search(query: str, limit: int = 10) -> str:
        """Search the product documentation."""
        slug = query.replace(" ", "-")
        hits = [
            f"docs/{slug}/overview",
            f"docs/{slug}/getting-started",
            f"docs/{slug}/agent-hooks",
        ][:limit]
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


# mock deploy server。
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


# 可连接的 server 工厂。未知名字会返回错误，而不是抛异常。
MOCK_SERVERS: dict[str, Callable[[], MCPClient]] = {
    "docs": docs_server,
    "deploy": deploy_server,
}

# 已连接的 server 实例。connect_mcp 成功后写进来，assemble_tool_pool 从这里读。
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


# 把一个已发现的外部工具包成一个 LangChain 工具：
# - 参数 schema 从原始 typed handler 的签名推断（functools.wraps 复制 annotations）；
# - 真正的调用走 client.call_tool，把错误停在调用边界上。
def _make_mcp_tool(client: MCPClient, raw_name: str, prefixed: str, description: str) -> StructuredTool:
    handler = client._handlers[raw_name]

    @functools.wraps(handler)
    def _runner(**kwargs):
        return client.call_tool(raw_name, kwargs)

    _runner.__name__ = prefixed.replace(".", "_").replace("-", "_")
    return StructuredTool.from_function(func=_runner, name=prefixed, description=description)


# 当前工具池的名字 -> 工具对象映射。model 节点和 tools 节点共用。
CURRENT_TOOL_MAP: dict[str, StructuredTool] = {}


def assemble_tool_pool() -> list:
    """每轮调用前组装当前可用的全部工具：基础工具 + 已连接 server 的动态工具。

    connect_mcp 执行后，mcp_clients 更新；下一轮 model 节点再次调用本函数，
    就会把 mcp__docs__search 这样的新工具一起放进模型输入。
    """
    tools = [connect_mcp, run_bash, run_read, run_write, run_edit, run_glob]
    seen = {t.name for t in tools}

    for server_name, client in mcp_clients.items():
        safe_server = normalize_mcp_name(server_name)
        for raw in client.tools:
            safe_tool = normalize_mcp_name(raw["name"])
            prefixed = f"mcp__{safe_server}__{safe_tool}"

            # 规范化可能把两个不同的名字折叠成同一个，必须拦下来。
            if prefixed in seen:
                raise ValueError(f"MCP tool name collision after normalization: {prefixed}")
            if len(prefixed) > 64:
                raise ValueError(f"MCP tool name exceeds 64 chars: {prefixed}")

            seen.add(prefixed)
            tools.append(_make_mcp_tool(client, raw["name"], prefixed, raw["description"]))

    CURRENT_TOOL_MAP.clear()
    CURRENT_TOOL_MAP.update({t.name: t for t in tools})
    return tools


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

MODEL = ChatOpenAI(
    model=MODEL_ID,
    max_completion_tokens=8000,
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


# ---------------------------------------------------------------------------
# 手写 LangGraph 循环（替代 create_agent 的静态编译图）
# ---------------------------------------------------------------------------

# Agent 状态只保存 messages，用 add_messages 作为合并 reducer。
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def model_node(state: AgentState) -> dict[str, Any]:
    """模型节点：每次都用最新的工具池绑定工具，再调用模型。"""
    tools = assemble_tool_pool()
    response = MODEL.bind_tools(tools).invoke(state["messages"])
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict[str, Any]:
    """工具节点：执行模型上一轮请求的所有工具调用，插入 Hook 和权限。"""
    last = state["messages"][-1]
    outputs = []

    for tool_call in last.tool_calls:
        name = tool_call["name"]
        args = tool_call.get("args", {})

        # PreToolUse：打印 + 权限检查。返回字符串表示被拦截。
        blocked = trigger_hooks("PreToolUse", name, args)
        if blocked:
            outputs.append(ToolMessage(
                content=str(blocked),
                tool_call_id=tool_call["id"],
                name=name,
                status="error",
            ))
            continue

        # 从当前工具池找工具并执行；找不到就返回错误，不中断循环。
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
    """决定下一步：模型还想调用工具就走 tools，否则结束本轮。"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"


# 建图：model -> (有 tool_calls ? tools : END) -> tools -> model。
graph_builder = StateGraph(AgentState)
graph_builder.add_node("model", model_node)
graph_builder.add_node("tools", tools_node)
graph_builder.add_edge(START, "model")
graph_builder.add_conditional_edges("model", route, {"tools": "tools", "end": END})
graph_builder.add_edge("tools", "model")
agent = graph_builder.compile()


# ---------------------------------------------------------------------------
# 打印与交互
# ---------------------------------------------------------------------------

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
    # UserPromptSubmit：本轮开始前触发一次。
    if messages:
        last = messages[-1]
        content = getattr(last, "content", last)
        trigger_hooks("UserPromptSubmit", content)

    result = agent.invoke({"messages": messages})
    new_messages = result["messages"][len(messages):]

    for message in new_messages:
        print_tool_activity(message)

    # Stop：本轮结束后触发一次。
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
