"""
s06：基于 LangChain create_agent 的同步 Subagent 示例。

整体结构：

    用户输入
       │
       ▼
    父 Agent ──调用 task 工具──▶ 子 Agent
       ▲                         │
       │                         ├─ 独立 messages
       │                         ├─ 独立 Agent 循环
       │                         ├─ 共用基础文件/命令工具
       │                         └─ 没有 task，不能继续委派
       │
       └──────── 只接收最终摘要 ──┘

子 Agent 与父 Agent 共享工作目录，所以子 Agent 写入或修改的文件会保留；
但子 Agent 的消息历史不会合并进父 Agent，避免探索过程污染父上下文。
"""

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


# ============================================================
# 1. 环境配置
# ============================================================

# override=True 表示 .env 中的值覆盖当前进程里同名的环境变量。
load_dotenv(override=True)

# 所有文件工具和 shell 工具都以程序启动目录作为工作区。
WORKDIR = Path.cwd().resolve()

# 使用标准 OPENAI_API_KEY，并兼容 BASE_URL / OPENAI_BASE_URL 两种地址变量名。
MODEL_ID = os.getenv("MODEL_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL")

# 尽早检查必要配置，避免运行到第一次模型请求时才得到难理解的错误。
if not MODEL_ID:
    raise RuntimeError("Missing MODEL_ID in .env")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")


# ============================================================
# 2. 父/子 Agent 执行范围标记
# ============================================================

# ContextVar 类似“跟随当前执行上下文的局部全局变量”。
# 默认值 parent；进入 task() 后临时改成 sub。
# Hook 读取这个值，就能在日志中显示工具调用来自父 Agent 还是子 Agent。
#
# 与普通全局字符串相比，ContextVar 更适合异步或并发执行：
# 不同执行上下文修改自己的值时，不容易互相覆盖。
AGENT_SCOPE: ContextVar[str] = ContextVar("agent_scope", default="parent")


# ============================================================
# 3. 最小 Hook 注册与分发系统
# ============================================================

# 每个事件可以注册多个回调函数，按照列表顺序依次执行。
HOOKS: dict[str, list[Callable[..., Any]]] = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": [],
}


def register_hook(event: str, callback: Callable[..., Any]) -> None:
    """把 callback 注册到指定事件。"""

    # 提前拒绝拼错的事件名，否则回调会被放进一个永远不会触发的列表。
    if event not in HOOKS:
        raise ValueError(f"Unknown hook event: {event}")

    # append 保留注册顺序；权限检查依赖这一顺序可预测。
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args: Any) -> Any | None:
    """触发事件；第一个非 None 返回值会终止后续 Hook。"""

    # get(..., []) 让没有回调的已知事件自然成为 no-op。
    for callback in HOOKS.get(event, []):
        result = callback(*args)

        # PreToolUse 利用非 None 返回值表达“拦截原因”。
        # 其他事件通常只记录日志，因此返回 None。
        if result is not None:
            return result

    return None


# ============================================================
# 4. 把自定义 Hook 接入 LangChain Middleware
# ============================================================

@before_agent
def user_prompt_submit(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    """父 Agent 开始执行前，把最新输入交给 UserPromptSubmit Hook。"""

    # before_agent 收到的是本次图运行的完整 state，而非单独一条用户消息。
    messages = state.get("messages", [])
    if not messages:
        return None

    # CLI 在 invoke/stream 前刚追加 HumanMessage，因此最后一条就是本轮输入。
    last_message = messages[-1]

    # 输入既可能是用户传入的 dict，也可能已被 LangChain 转成 BaseMessage。
    if isinstance(last_message, dict):
        content = last_message.get("content")
    else:
        content = getattr(last_message, "content", None)

    trigger_hooks("UserPromptSubmit", content)

    # before_agent 返回 None 表示不修改 AgentState。
    return None


@wrap_tool_call
def tool_hook(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """包裹每次工具执行，在执行前后触发权限检查和日志 Hook。"""

    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})
    tool_call_id = request.tool_call["id"]

    # 工具尚未执行。PreToolUse 可以返回字符串说明拒绝原因。
    blocked_reason = trigger_hooks("PreToolUse", tool_name, tool_args)

    if blocked_reason:
        # 不调用 handler，因此真实工具不会执行。
        # 仍然返回对应 tool_call_id 的 ToolMessage，让模型知道调用失败，
        # 否则消息协议会缺少工具结果。
        return ToolMessage(
            content=str(blocked_reason),
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )

    # handler 是 LangChain 提供的真实工具分发函数。
    result = handler(request)

    # 工具完成后记录结果。这里不会改变返回给模型的真实结果。
    trigger_hooks("PostToolUse", tool_name, tool_args, result)
    return result


@after_agent
def stop_hook(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    """父 Agent 完成一轮请求后触发 Stop Hook。"""

    # Stop Hook 只观察最终状态；返回 None 表示不再修改 messages/todos。
    trigger_hooks("Stop", state.get("messages", []))
    return None


# ============================================================
# 5. 权限策略
# ============================================================

# 这些模式直接拒绝，不向用户询问。
from harness.security import check_deny_list

# 这些模式不直接拒绝，但必须由用户确认。
POTENTIALLY_DESTRUCTIVE_COMMANDS = [
    "rm ",
    "> /etc/",
    "chmod 777",
]


def resolve_path(raw_path: str) -> Path:
    """把相对路径解析到 WORKDIR；绝对路径保持其绝对含义。"""

    candidate = Path(raw_path)
    # 权限层需要知道用户/模型真正请求的目标，因此绝对路径不能偷偷重定位到 WORKDIR。
    if candidate.is_absolute():
        return candidate.resolve()

    # resolve 会折叠 ..，后续 is_relative_to 检查看到的是规范化后的真实位置。
    return (WORKDIR / candidate).resolve()


# check_deny_list 已由上方 harness.security import 提供。


def check_rules(tool_name: str, args: dict[str, Any]) -> str | None:
    """返回需要用户确认的原因；无需确认时返回 None。"""

    # shell 规则检查的是 command；文件工具规则检查的是 path。
    if tool_name == "run_bash":
        normalized = str(args.get("command", "")).lower()
        if normalized.strip().startswith("del "):
            return "Potentially destructive shell command: del "
        for pattern in POTENTIALLY_DESTRUCTIVE_COMMANDS:
            if pattern.lower() in normalized:
                return f"Potentially destructive shell command: {pattern}"

    # glob 只以 root_dir=WORKDIR 搜索，因此无需走这里的单路径确认流程。
    if tool_name in {"run_read", "run_write", "run_edit"}:
        raw_path = str(args.get("path", ""))
        try:
            # 先解析再比较，避免 path 中的 .. 造成字符串前缀误判。
            target = resolve_path(raw_path)
        except (OSError, RuntimeError, ValueError) as exc:
            return f"Invalid path: {exc}"

        # 工作区外的读写不是直接拒绝，而是交给用户决定。
        if not target.is_relative_to(WORKDIR):
            return f"Operation accesses outside workspace: {target}"

    return None


def ask_user(tool_name: str, args: dict[str, Any], reason: str) -> bool:
    """在当前终端请求用户批准。子 Agent 的请求也会冒泡到这里。"""

    # scope 只用于提示来源；父子 Agent 最终都由同一个人类终端批准。
    scope = AGENT_SCOPE.get()
    print(f"\nWarning: [{scope}] Permission required")
    print(f"Reason: {reason}")
    print(f"Tool: {tool_name}")
    print(f"Arguments: {args}")

    # 默认拒绝；只有明确输入 y/yes 才允许。
    choice = input("Allow? [y/N] ").strip().lower()
    return choice in {"y", "yes"}


def check_permission(tool_name: str, args: dict[str, Any]) -> bool:
    """先执行硬拒绝规则，再执行需要确认的规则。"""

    # deny list 优先级最高：命中后不提供“仍然批准”的机会。
    if tool_name == "run_bash":
        denied_reason = check_deny_list(str(args.get("command", "")))
        if denied_reason:
            print(f"\nBlocked: {denied_reason}")
            return False

    # 只有未硬拒绝的调用才会进入 ask_user。
    confirmation_reason = check_rules(tool_name, args)
    if confirmation_reason:
        return ask_user(tool_name, args, confirmation_reason)

    return True


# ============================================================
# 6. 具体 Hook 回调
# ============================================================

def on_user_prompt_submit(content: Any) -> None:
    """记录用户本轮提交的内容。"""

    print(f"[UserPromptSubmit] {content}")


def on_pre_tool_use(
    tool_name: str,
    tool_args: dict[str, Any],
) -> str | None:
    """记录工具调用，并执行统一权限检查。"""

    # ContextVar 让同一回调无需分别为父、子 Agent 注册两份。
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
    """记录工具结果；只打印前 500 个字符，避免终端被大文件刷屏。"""

    scope = AGENT_SCOPE.get()
    print(f"[{scope} PostToolUse] {tool_name}")

    # 正常 ToolMessage 从 content 取值；Command 等特殊返回则直接字符串化。
    preview = str(getattr(result, "content", result))
    if len(preview) > 500:
        preview = preview[:500] + "...(truncated)"
    print(f"Result: {preview}")


def on_stop(messages: list[Any]) -> None:
    """统计本次父会话累积的消息和工具调用数量。"""

    tool_call_count = 0
    # 一个 AIMessage 可以一次发起多个并行 tool_calls，因此不能只按消息条数统计。
    for message in messages:
        if isinstance(message, AIMessage):
            tool_call_count += len(message.tool_calls or [])

    # print 必须位于 for 循环外。如果缩进到循环内部，messages 有多少条，
    # Stop 行就会重复打印多少次，并显示逐步累积的 tool_calls 数量。
    print(f"[Stop] messages={len(messages)}, tool_calls={tool_call_count}")


# 将回调挂到事件上。注册顺序就是同一事件的执行顺序。
register_hook("UserPromptSubmit", on_user_prompt_submit)
register_hook("PreToolUse", on_pre_tool_use)
register_hook("PostToolUse", on_post_tool_use)
register_hook("Stop", on_stop)


# ============================================================
# 7. 父子 Agent 共用的基础工具
# ============================================================

@tool
# 安全边界：shell=True 仅为教学演示，黑名单/路径检查不等于安全边界；生产请使用权限中间件 + 沙箱。
def run_bash(command: str) -> str:
    """Execute a shell command in the current workspace."""

    try:
        # shell=True 让模型可使用管道和重定向；安全性由前面的 PreToolUse 规则兜底。
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
        )

        # stdout 和 stderr 都需要返回给模型，否则模型无法判断失败原因。
        output = (result.stdout + result.stderr).strip() or "(no output)"
        if result.returncode != 0:
            output = f"Exit code: {result.returncode}\n{output}"

        # 控制单次工具结果大小，降低上下文被超长终端输出占满的风险。
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

        # limit 只截断工具返回，不修改真实文件。
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
        # mkdir 允许模型直接写入新目录；工作区外路径已经在权限 Hook 中要求确认。
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
        # 先完整读取再做一次精确替换，适合教学示例的小型文本文件。
        current_content = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        # 精确替换可以避免模型因模糊匹配误改其他位置。
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

        # root_dir 使结果保持为工作区相对路径，便于后续传给 read/edit。
        # recursive=True 使 ** 生效；普通 * 的行为不受影响。
        for match in glob.glob(pattern, root_dir=WORKDIR, recursive=True):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)

        return "\n".join(sorted(results)) if results else "(no matches)"

    except (OSError, ValueError) as exc:
        return f"Error: {exc}"


# BASE_TOOLS 是父子 Agent 都能使用的工具集合。
# 这里故意不包含 task：子 Agent 因此不能继续创建子 Agent。
BASE_TOOLS = [run_bash, run_read, run_write, run_edit, run_glob]


# ============================================================
# 8. 模型与子 Agent
# ============================================================

# 父子 Agent 可以安全地复用同一个 ChatOpenAI 模型对象；
# 它们的上下文是否隔离，取决于各自 invoke 时传入的 state/messages，
# 而不是是否复用 Python 模型实例。
MODEL = ChatOpenAI(
    model=MODEL_ID,
    max_completion_tokens=8000,
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


# 子 Agent 的提示词明确强调：上下文是独立的、不得继续委派、
# 最终需要给父 Agent 一份自包含的结论。
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


# 子 Agent 只装 tool_hook：它的文件和 shell 操作仍经过统一权限策略。
# 它不装 TodoListMiddleware，所以没有 write_todos；也不装 task。
# 不配置 checkpointer，因此每次 invoke 都从传入的新 state 开始。
SUB_AGENT = create_agent(
    model=MODEL,
    tools=BASE_TOOLS,
    system_prompt=SUB_SYSTEM,
    middleware=[tool_hook],
    name="worker",
)


def extract_final_text(messages: list[Any]) -> str:
    """从子 Agent 历史中倒序提取最后一条非空 AI 文本。"""

    # 倒序查找很重要：子 Agent 历史包含多轮 AIMessage，
    # 父 Agent 只应该收到最后结论，而不是第一轮工具规划。
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue

        content = message.content

        # 大多数 OpenAI 兼容模型直接返回字符串。
        if isinstance(content, str):
            if content.strip():
                return content.strip()
            continue

        # 某些模型/API 返回 content block 列表，需要逐块提取 text。
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


# ============================================================
# 9. task 工具：父 Agent 进入子 Agent 的唯一入口
# ============================================================

@tool("task")
def task(description: str) -> str:
    """Launch an isolated subagent and return only its final conclusion."""

    print("\n\033[35m[Subagent spawned]\033[0m")
    print(f"Task: {description}")

    # set() 返回 token，稍后用 reset(token) 恢复进入子 Agent 前的值。
    scope_token = AGENT_SCOPE.set("sub")

    try:
        # 这是上下文隔离的核心：
        # 只传入 description 形成一条全新 user message，绝不传父 messages。
        result = SUB_AGENT.invoke(
            {"messages": [{"role": "user", "content": description}]},
            # LangGraph 统计图节点步骤；约 60 多步可以限制约 30 轮
            # model/tool 循环，防止子 Agent 无限运行。
            config={"recursion_limit": 128},
        )

        # result 内部包含子 Agent 的完整历史，但这里只提取最终文本。
        # task 返回 str 后，父 Agent 的 ToolNode 会把它包装成一条 ToolMessage。
        summary = extract_final_text(result.get("messages", []))
        return summary or "Subagent finished without a textual conclusion."

    except GraphRecursionError:
        return "Subagent stopped because it reached the execution limit."
    except Exception as exc:
        # 把子 Agent 故障转换成工具结果，使父 Agent 有机会解释或恢复。
        return f"Subagent failed: {type(exc).__name__}: {exc}"
    finally:
        # 无论成功、失败还是提前 return，都恢复为进入 task 前的 scope。
        AGENT_SCOPE.reset(scope_token)
        print("\033[35m[Subagent done]\033[0m")


# ============================================================
# 10. 父 Agent
# ============================================================

# 父提示词告诉模型何时委派，并强调子 Agent 看不到父会话，
# 因此 description 必须包含完成任务所需的全部信息。
# 最后的 demonstration 规则是当前 s06.py 新增的强制测试开关：
# 只要请求涉及文件读写、编辑或执行，父 Agent 就必须调用 task，
# 而不能自己直接调用基础工具。正式使用时可以删除最后三行强制规则，
# 恢复成“只在复杂、自包含任务中委派”的按需模式。
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


# *BASE_TOOLS 会把基础工具逐个展开，得到一个扁平工具列表。
# 如果写成 [BASE_TOOLS, task]，第一个元素会是 list，LangChain 会报
# “Got <class 'list'>”。
PARENT_TOOLS = [*BASE_TOOLS, task]


# Middleware 按职责组合：
# 1. before_agent 记录用户输入；
# 2. TodoListMiddleware 增加 write_todos 工具和 todos 状态；
# 3. tool_hook 统一保护所有工具（包括 task）；
# 4. after_agent 输出会话统计。
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


# 父 Agent 是 CLI 主循环实际调用的 Agent。
agent = create_agent(
    model=MODEL,
    tools=PARENT_TOOLS,
    system_prompt=PARENT_SYSTEM,
    middleware=PARENT_MIDDLEWARE,
    name="parent",
)


# ============================================================
# 11. 消息与 Todo 的终端显示
# ============================================================

def content_to_text(content: Any) -> str:
    """兼容字符串和多种 content block 表示，统一返回纯文本。"""

    # 最常见路径：普通聊天模型直接把 content 设为 str。
    if isinstance(content, str):
        return content
    # 对未知但可打印的 provider 类型保留诊断信息，而不是静默丢弃。
    if not isinstance(content, list):
        return str(content)

    texts: list[str] = []
    # 列表内容兼容三种表示：字符串、OpenAI 字典块、LangChain 对象块。
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
    """根据消息类型打印模型工具调用、工具结果或普通文本。"""

    if isinstance(message, AIMessage):
        # AIMessage 可能同时包含自然语言 content 和结构化 tool_calls。
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
    """显示 TodoListMiddleware 写入 AgentState 的 todos。"""

    print("\n当前 Todo：")
    # start=1 与用户看到的自然任务编号一致，内部列表仍保持零基索引。
    for index, todo_item in enumerate(todos, start=1):
        status = todo_item.get("status", "pending")
        content = todo_item.get("content", "")
        print(f"{index}. [{status}] {content}")


# ============================================================
# 12. 父 Agent 流式循环与状态持久化
# ============================================================

def agent_loop(session_state: dict[str, Any]) -> None:
    """流式运行父 Agent，并把最终 messages/todos 写回会话状态。"""

    # 输入消息已经由之前轮次打印过；只打印 seen_message_count 之后的新消息。
    seen_message_count = len(session_state.get("messages", []))
    last_todos = session_state.get("todos")
    final_state: dict[str, Any] | None = None

    # values 模式每次产出当前完整状态，而不是单独 token。
    for state in agent.stream(session_state, stream_mode="values"):
        final_state = state

        # TodoListMiddleware 只在列表变化时写新值，避免每个 state 快照重复打印。
        todos = state.get("todos")
        if todos is not None and todos != last_todos:
            print_todos(todos)
            last_todos = todos

        current_messages = state.get("messages", [])
        for message in current_messages[seen_message_count:]:
            print_message(message)
        seen_message_count = len(current_messages)

    if final_state is not None:
        # 保存完整 state 而不只是 messages，因此 todos 可以跨用户轮次保留。
        session_state.clear()
        session_state.update(final_state)


# ============================================================
# 13. 命令行入口
# ============================================================

def main() -> None:
    """运行交互式命令行会话。"""

    print("s06: LangChain Subagent — isolated context, summary only")
    print("输入问题并回车。输入 q 或 exit 退出。\n")

    # 该字典会在每轮调用后被 agent_loop 更新。
    session_state: dict[str, Any] = {"messages": []}

    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.strip().lower() in {"", "q", "exit"}:
            break

        # setdefault 兼容外部调用方传入缺少 messages 的普通 dict。
        session_state.setdefault("messages", []).append(
            {"role": "user", "content": query}
        )

        try:
            agent_loop(session_state)
        # 图递归上限通常意味着模型/工具循环没有收敛，单独给出更清晰提示。
        except GraphRecursionError:
            print("\nAgent stopped because it reached the execution limit.")
        except Exception as exc:
            print(f"\nAgent error: {type(exc).__name__}: {exc}")

        print()


if __name__ == "__main__":
    main()
