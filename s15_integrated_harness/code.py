"""s15: Integrated Harness -- 很多机制，一个循环。

本章不再引入某个孤立的新机制，而是把前面章节的机制挂回同一条循环上：
基础工具 + Hook + 权限 + todo + skill + 记忆 + 一次性子 agent + 后台 bash + cron + MCP，
外加 system prompt 动态组装、上下文压缩和错误恢复。

因为要在运行时注入后台/定时通知、动态组装 MCP 工具，主循环仍然是手写的
（同 s14），用 MODEL.bind_tools(assemble_tool_pool()) 每次取当前工具。
一次性子 agent（task 工具）则复用 create_agent，体现“隔离上下文”的委托。
"""

# functools：MCP 动态工具签名复制；threading/time/queue：后台与定时任务。
import functools
import re
import threading
import time
import queue
import json

# Callable / Any 做类型标注。
from collections.abc import Callable
from typing import Any

# Path 处理路径；os/subprocess 执行命令。
from pathlib import Path
import os, subprocess

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain.agents import create_agent

load_dotenv(override=True)

MODEL_ID = os.getenv("MODEL_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("BASE_URL")
WORKDIR = Path.cwd()
MEMORY_DIR = WORKDIR / ".memory"


def build_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_ID,
        max_completion_tokens=8000,
        temperature=0,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )


MODEL = build_chat_model()

# ---------------------------------------------------------------------------
# Hook 系统（s04）
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 权限（s04 + MCP 主机策略）
# ---------------------------------------------------------------------------

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


def on_pre_tool_use(tool_name, tool_args):
    print("[PreToolUse]", tool_name, tool_args)
    if not check_permission(tool_name, tool_args):
        return "Permission denied"


def on_post_tool_use(tool_name, tool_args, result):
    print("[PostToolUse]", tool_name)


register_hook("UserPromptSubmit", lambda c: print("[UserPromptSubmit]", c))
register_hook("PreToolUse", on_pre_tool_use)
register_hook("PostToolUse", on_post_tool_use)
register_hook("Stop", lambda msgs: print("[Stop]", len(msgs)))


# ---------------------------------------------------------------------------
# 基础工具（s04）
# ---------------------------------------------------------------------------

@tool
def run_bash(command: str, run_in_background: bool = False) -> str:
    """Execute a shell command. Set run_in_background=True to run async and return a placeholder."""
    if run_in_background:
        return start_background_task(command)
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


# ---------------------------------------------------------------------------
# 计划（s05 todo_write 的轻量版）
# ---------------------------------------------------------------------------

# 当前会话的待办列表，元素形如 {"content": str, "status": str}。
TODO: list[dict] = []


@tool
def todo_write(todos: list) -> str:
    """Replace the current todo list with [{content, status}] items (status: pending|in_progress|completed)."""
    global TODO
    if not isinstance(todos, list):
        return "Error: todos must be a list of {content, status}"
    cleaned = []
    for item in todos:
        if isinstance(item, dict) and "content" in item:
            cleaned.append({"content": item["content"], "status": item.get("status", "pending")})
    TODO = cleaned
    return f"todo updated: {len(TODO)} item(s)"


# ---------------------------------------------------------------------------
# Skills（s07 的按需加载）
# ---------------------------------------------------------------------------

def _parse_skill(path: Path) -> tuple[str, str]:
    """解析一个 SKILL.md：返回 (name, description)。

    Skill 的 frontmatter 格式为 --- 开头，里面有 name: 和 description: 字段。
    description 可能是一行，也可能是 | 多行块；多行块只取第一行作为目录摘要。
    """
    name = path.parent.name
    desc = ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return name, desc

    # 定位 frontmatter（首行是 --- 时，切出中间的名字段和后面的正文）。
    if lines and lines[0].strip() == "---":
        end = 1
        while end < len(lines) and lines[end].strip() != "---":
            end += 1
        fm, after = lines[1:end], lines[end + 1:]
    else:
        fm, after = [], lines

    in_multiline_desc = False
    for ln in fm:
        s = ln.strip()
        if s.startswith("name:") and len(s) > 5:
            name = s[5:].strip()
        elif s.startswith("description:"):
            rest = s[len("description:"):].strip()
            if rest == "|":
                # 多行块：后续缩进的正文行由下面统一兜底。
                in_multiline_desc = True
            elif rest:
                desc = rest
        elif in_multiline_desc and s and not s.startswith("name:") and not desc:
            desc = s

    if not desc:
        # description 缺失时，取正文里第一个非标题行作为摘要。
        for ln in after:
            s = ln.strip()
            if s and not s.startswith("#"):
                desc = s
                break
    return name, desc


def skills_catalog() -> str:
    """扫描 skills/*/SKILL.md，只把目录（名称 + 摘要）放进目录，内容按需再加载。"""
    entries = []
    for p in sorted(Path("skills").glob("*/SKILL.md")):
        name, desc = _parse_skill(p)
        entries.append(f"- {name}:" + (f" {desc[:90]}" if desc else ""))
    if not entries:
        return "Available skills: (none)"
    return "Available skills (load the full content with load_skill):\n" + "\n".join(entries)


@tool
def load_skill(name: str) -> str:
    """Load the full content of an installed skill by directory name."""
    skill = Path("skills") / name / "SKILL.md"
    if not skill.exists():
        return f"Error: unknown skill '{name}'"
    try:
        return skill.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# 记忆（s09 的目录读取 + 简单相关性选择）
# ---------------------------------------------------------------------------

def memory_section(query_text: str = "") -> str:
    """把 .memory/MEMORY.md 里相关的记录注入 system prompt。"""
    mem_file = MEMORY_DIR / "MEMORY.md"
    if not mem_file.exists():
        return "Long-term memory: (empty)"
    try:
        lines = mem_file.read_text(encoding="utf-8").splitlines()
    except Exception:
        return "Long-term memory: (unreadable)"

    if query_text:
        kws = [w for w in query_text.lower().split() if len(w) > 3]
        relevant = [ln for ln in lines if kws and any(k in ln.lower() for k in kws)]
        selected = relevant if relevant else lines
    else:
        selected = lines
    return "Long-term memory (relevant):\n" + "\n".join(selected[:50])


# ---------------------------------------------------------------------------
# 一次性子 agent（s06 的隔离上下文委托）
# ---------------------------------------------------------------------------

@tool
def task(prompt: str) -> str:
    """Dispatch an isolated one-shot subagent that investigates and returns only a final summary."""
    sub = create_agent(
        model=build_chat_model(),
        tools=[run_read, run_glob],
        system_prompt=(
            "You are a focused subagent. Investigate the workspace with read/glob, "
            "then return a concise summary of your findings. Do not ask questions."
        ),
    )
    try:
        result = sub.invoke({"messages": [HumanMessage(content=prompt)]})
        last = result["messages"][-1]
        return str(getattr(last, "content", last))
    except Exception as e:
        return f"subagent error: {e}"


# ---------------------------------------------------------------------------
# 后台 bash（s11 的后台执行 + 完成通知）
# ---------------------------------------------------------------------------

BACKGROUND_RESULTS: "queue.Queue[str]" = queue.Queue()
BG_COUNTER = [0]


def start_background_task(command: str) -> str:
    """在守护线程里跑命令；完成后把 <task_notification> 放进队列，由循环注入。"""
    BG_COUNTER[0] += 1
    task_id = BG_COUNTER[0]

    def worker():
        try:
            r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=300)
            out = (r.stdout + r.stderr).strip() or "(no output)"
            status = "completed" if r.returncode == 0 else f"failed (exit {r.returncode})"
        except Exception as e:
            out = f"Error: {e}"
            status = "failed"
        BACKGROUND_RESULTS.put(
            f"<task_notification>[background #{task_id}] {status}:\n{out[:4000]}</task_notification>"
        )

    threading.Thread(target=worker, daemon=True).start()
    return f"started background task #{task_id}; its result will arrive as a notification"


# ---------------------------------------------------------------------------
# Cron（s12 的定时触发，简化版：一次性相对延迟）
# ---------------------------------------------------------------------------

CRON_JOBS: list[dict] = []
CRON_COUNTER = [0]
CRON_QUEUE: "queue.Queue[str]" = queue.Queue()


@tool
def schedule_cron(delay_seconds: int, prompt: str) -> str:
    """Schedule a one-shot reminder prompt delivered after delay_seconds."""
    CRON_COUNTER[0] += 1
    job_id = CRON_COUNTER[0]
    CRON_JOBS.append({"id": job_id, "due": time.time() + max(1, delay_seconds), "prompt": prompt, "delivered": False})
    return f"scheduled cron #{job_id} in {delay_seconds}s: {prompt}"


@tool
def list_crons() -> str:
    """List scheduled cron jobs and how long until each fires."""
    if not CRON_JOBS:
        return "(no cron jobs)"
    now = time.time()
    return "\n".join(
        f"#{j['id']} due_in={int(j['due'] - now)}s delivered={j['delivered']} {j['prompt']}"
        for j in CRON_JOBS
    )


@tool
def cancel_cron(job_id: int) -> str:
    """Cancel a scheduled cron job by id."""
    for j in CRON_JOBS:
        if j["id"] == job_id:
            j["delivered"] = True
            j["due"] = 0
            return f"cancelled cron #{job_id}"
    return f"unknown cron #{job_id}"


def cron_worker():
    """守护线程，每秒扫描一次到期任务，把 prompt 放进 CRON_QUEUE。"""
    while True:
        time.sleep(1)
        now = time.time()
        for j in CRON_JOBS:
            if not j["delivered"] and j["due"] <= now:
                j["delivered"] = True
                CRON_QUEUE.put(j["prompt"])


threading.Thread(target=cron_worker, daemon=True).start()


# ---------------------------------------------------------------------------
# MCP（s14 的动态工具，内联到本集成章）
# ---------------------------------------------------------------------------

def normalize_mcp_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


class MCPClient:
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


MOCK_SERVERS: dict[str, Callable[[], MCPClient]] = {"docs": docs_server, "deploy": deploy_server}
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


BASE_TOOLS = [
    run_bash, run_read, run_write, run_edit, run_glob,
    todo_write, load_skill, task,
    connect_mcp,
    schedule_cron, list_crons, cancel_cron,
]
CURRENT_TOOL_MAP: dict[str, Any] = {}


def assemble_tool_pool() -> list:
    tools = list(BASE_TOOLS)
    seen = {t.name for t in tools}
    for server_name, client in mcp_clients.items():
        safe_server = normalize_mcp_name(server_name)
        for raw in client.tools:
            safe_tool = normalize_mcp_name(raw["name"])
            prefixed = f"mcp__{safe_server}__{safe_tool}"
            if prefixed in seen:
                raise ValueError(f"MCP tool name collision after normalization: {prefixed}")
            seen.add(prefixed)
            tools.append(_make_mcp_tool(client, raw["name"], prefixed, raw["description"]))
    CURRENT_TOOL_MAP.clear()
    CURRENT_TOOL_MAP.update({t.name: t for t in tools})
    return tools


def mcp_state_section() -> str:
    if not mcp_clients:
        return "Connected MCP servers: (none)"
    return f"Connected MCP servers: {', '.join(mcp_clients)}"


# ---------------------------------------------------------------------------
# system prompt 组装（记忆 + skills + MCP 状态）
# ---------------------------------------------------------------------------

SYSTEM_BASE = (
    f"you are a coding agent at {WORKDIR}. Use the provided tools to solve tasks. "
    "Plan with todo_write, load skills on demand, delegate isolated investigation with task, "
    "and schedule reminders with schedule_cron. Act, don't explain."
)


def assemble_system_prompt(query_text: str = "") -> str:
    return "\n\n".join([
        SYSTEM_BASE,
        skills_catalog(),
        memory_section(query_text),
        mcp_state_section(),
    ])


# ---------------------------------------------------------------------------
# 上下文压缩（简化：剪掉过长的工具结果）
# ---------------------------------------------------------------------------

MAX_TOOL_CHARS = 20000
MAX_SINGLE_TOOL_CHARS = 4000


def compact_messages(messages: list) -> list:
    """把过长的单个工具结果截断，并给累计工具结果设一个总预算。"""
    out: list = []
    tool_chars = 0
    for m in messages:
        if isinstance(m, ToolMessage):
            c = str(m.content)
            if len(c) > MAX_SINGLE_TOOL_CHARS:
                c = c[:MAX_SINGLE_TOOL_CHARS] + "\n...(snipped)"
            tool_chars += len(c)
            if tool_chars > MAX_TOOL_CHARS:
                c = c[:500] + "\n...(further compacted)"
            out.append(ToolMessage(content=c, tool_call_id=m.tool_call_id, name=m.name))
        else:
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# 错误恢复（对模型调用做重试/退避/上下文裁剪）
# ---------------------------------------------------------------------------

def call_model_with_recovery(system: str, messages: list, tools: list):
    """模型调用包裹恢复逻辑：临时失败指数退避重试，上下文过长则裁剪重试。"""
    llm = MODEL.bind_tools(tools)
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            return llm.invoke([SystemMessage(content=system)] + messages)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "context" in msg or "maximum context" in msg or "too long" in msg:
                messages = compact_messages(messages)
                print("[recover] trimmed context and retrying once")
            else:
                wait = 2 ** attempt
                print(f"[recover] attempt {attempt + 1} failed ({e}); retrying in {wait}s")
                time.sleep(wait)
    raise last_err


# ---------------------------------------------------------------------------
# 主循环：很多机制，一个 while True（手写版 LangGraph 循环）
# ---------------------------------------------------------------------------

def inject_pending(messages: list) -> None:
    """把已完成的后台任务通知和到期 cron prompt 注入消息流。"""
    count = 0
    while True:
        try:
            item = BACKGROUND_RESULTS.get_nowait()
        except queue.Empty:
            break
        messages.append(HumanMessage(content=item))
        count += 1
    while True:
        try:
            item = CRON_QUEUE.get_nowait()
        except queue.Empty:
            break
        messages.append(HumanMessage(content=f"<cron reminder>{item}</cron reminder>"))
        count += 1
    if count:
        print(f"[inject] {count} pending notification(s)")


def execute_tool_calls(response: AIMessage) -> list:
    """执行一条模型消息里的所有工具调用，插入 Hook 与权限。"""
    outputs = []
    for tool_call in response.tool_calls:
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
    return outputs


def agent_loop(messages: list) -> None:
    if messages:
        trigger_hooks("UserPromptSubmit", getattr(messages[-1], "content", messages[-1]))

    while True:
        # 1. 注入后台/定时通知。
        inject_pending(messages)

        # 2. 组装 system prompt（记忆 + skills + MCP 状态）。
        query_text = str(getattr(messages[-1], "content", ""))
        system = assemble_system_prompt(query_text)

        # 3. 取当前工具池，带恢复地调用模型。
        tools = assemble_tool_pool()
        response = call_model_with_recovery(system, messages, tools)
        messages.append(response)

        # 4. 没有工具调用 → Stop → 结束本轮。
        if not response.tool_calls:
            trigger_hooks("Stop", messages)
            return

        # 5. 执行工具，把结果追加回消息流，继续循环。
        messages.extend(execute_tool_calls(response))


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
                bg = " (background)" if args.get("run_in_background") else ""
                print(f"\033[33m$ {args.get('command', '')}{bg}\033[0m")
            else:
                print(f"\033[33m{name}{args}\033[0m")
        return
    if isinstance(message, ToolMessage):
        content = str(message.content)
        if getattr(message, "status", None) == "error":
            print(f"(blocked/error) {content[:200]}")
        else:
            print(content[:200])


def last_assistant_message(messages: list) -> AIMessage | None:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m
    return None


if __name__ == "__main__":
    print("s15: Integrated Harness")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []
    while True:
        try:
            query = input("\033[36ms15 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append(HumanMessage(content=query))
        start = len(history)
        agent_loop(history)

        for message in history[start:]:
            print_tool_activity(message)
        last = last_assistant_message(history)
        if last is not None:
            print_assistant_message(last)
        print()
