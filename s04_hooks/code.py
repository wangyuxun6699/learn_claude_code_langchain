"""s04: 演示如何给 LangChain Agent 加 Hook 和工具权限检查。

这个文件的主线是：
1. 读取 .env 里的模型配置。
2. 定义一个简单的 Hook 系统，模拟 UserPromptSubmit / PreToolUse / PostToolUse / Stop。
3. 定义几个可被 Agent 调用的本地工具，比如执行命令、读文件、写文件、搜索文件。
4. 在 Agent 调用工具前后插入 Hook，用来打印日志和做权限检查。
5. 在命令行里循环接收用户输入，把对话交给 Agent 处理。
"""

# Callable 用来给 handler 这类“可调用对象”做类型标注。
from collections.abc import Callable

# Any 表示任意类型，这里用于标注 Agent 状态返回值。
from typing import Any

# load_dotenv 用来读取当前目录下的 .env 文件，把其中的变量加载到环境变量里。
from dotenv import load_dotenv

# override=True 表示如果系统环境变量里已经有同名变量，也用 .env 里的值覆盖它。
load_dotenv(override=True)

# 下面几个装饰器来自 LangChain Agent 的 middleware 机制。
# before_agent：Agent 开始执行前触发。
# after_agent：Agent 完成执行后触发。
# wrap_tool_call：包裹每一次工具调用，可以在调用前后插入自己的逻辑。
from langchain.agents.middleware import (
    AgentState,
    before_agent,
    after_agent,
    wrap_tool_call,
)

# ToolCallRequest 表示一次工具调用请求，里面包含工具名、参数、调用 id 等信息。
from langchain.tools.tool_node import ToolCallRequest

# ToolMessage 是工具调用返回给 Agent 的消息类型。
from langchain_core.messages import ToolMessage

# Runtime 是 LangGraph 运行时对象；当前代码里只做类型标注，没有直接使用它。
from langgraph.runtime import Runtime

# Command 是 LangGraph 中用于控制图执行流的返回类型之一，这里用于 middleware 类型标注。
from langgraph.types import Command

# Path 用来处理文件路径，比直接拼字符串更安全、更跨平台。
from pathlib import Path

# @tool 装饰器会把普通 Python 函数注册成 Agent 可以调用的工具。
from langchain_core.tools import tool

# os 用来读取环境变量和当前工作目录；subprocess 用来执行 shell 命令。
import os, subprocess

# 这里重复导入了一次 Path，不影响运行；为了尽量不改原代码逻辑，先保留。

# AIMessage 是模型回复消息的类型，print_assistant_message 会用它做类型标注。
from langchain_core.messages import AIMessage

# ChatOpenAI 是 LangChain 里兼容 OpenAI 接口的聊天模型封装。
from langchain_openai import ChatOpenAI

# create_agent 用来创建一个带工具和 middleware 的 Agent。
from langchain.agents import create_agent


# HOOKS 是一个“事件名 -> 回调函数列表”的注册表。
# 每个 key 代表一个生命周期事件，每个 value 保存这个事件触发时要依次执行的函数。
HOOKS = {
    # 用户提交 prompt 后、Agent 真正执行前触发。
    "UserPromptSubmit": [],
    # Agent 调用工具之前触发，适合做权限检查、拦截危险操作。
    "PreToolUse": [],
    # Agent 调用工具之后触发，适合记录工具返回结果。
    "PostToolUse": [],
    # Agent 本轮执行结束后触发。
    "Stop": [],
}


def register_hook(event: str, callback):
    """把一个回调函数注册到指定事件上。

    event 是 HOOKS 里的事件名，比如 "PreToolUse"。
    callback 是事件触发时要执行的函数。
    """
    # 把 callback 追加到对应事件列表末尾；触发时会按照注册顺序执行。
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    """触发某个事件下注册的所有 Hook。

    如果某个 Hook 返回了非 None 的值，就立刻停止后续 Hook，并把该值返回给调用者。
    这个设计让 PreToolUse 可以通过返回错误原因来拦截工具调用。
    """
    # 依次取出这个事件下所有已注册的回调函数。
    for callback in HOOKS[event]:
        # 把外部传进来的参数原样交给回调函数。
        result = callback(*args)

        # 只要某个回调明确返回了内容，就把它当成“有处理结果”。
        if result is not None:
            return result

    # 所有回调都没有返回内容时，表示没有拦截、没有特殊处理。
    return None


@before_agent
def user_prompt_submit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Agent 运行前触发 UserPromptSubmit Hook。

    LangChain 会把当前对话状态放在 state["messages"] 里。
    这里取最后一条消息作为用户刚提交的内容，然后交给 Hook 系统处理。
    """
    # 从 Agent 状态里取出历史消息；如果没有 messages，就使用空列表避免报错。
    message = state.get("messages", [])

    # 只有存在消息时才需要触发 UserPromptSubmit。
    if message:
        # 最后一条消息通常就是用户这次刚输入的问题。
        last = message[-1]

        # messages 里的元素可能是 dict，也可能是 LangChain Message 对象，所以要分情况取 content。
        content = last.get("content") if isinstance(last, dict) else getattr(last, "content", None)

        # 把用户输入内容传给所有注册到 UserPromptSubmit 的回调。
        trigger_hooks("UserPromptSubmit", content)

    # 返回 None 表示不修改 Agent 的 state，只做旁路监听。
    return None


@wrap_tool_call
def tool_hook(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """包裹工具调用，在真正执行工具前后插入 Hook。

    request 保存这次工具调用的信息。
    handler 是 LangChain 原本要执行工具的函数；调用 handler(request) 才会真正运行工具。
    """
    # 从工具调用请求里拿到工具名，比如 run_bash、run_read。
    tool_name = request.tool_call["name"]

    # 获取工具参数；如果没有 args，就用空字典兜底。
    tool_args = request.tool_call.get("args", {})

    # 在工具运行前触发 PreToolUse。回调可以返回字符串来表示“禁止调用的原因”。
    blockd = trigger_hooks("PreToolUse", tool_name, tool_args)

    # 如果 PreToolUse 返回了内容，就不再真正执行工具，而是伪造一个失败的 ToolMessage。
    if blockd:
        return ToolMessage(
            # content 会被 Agent 看到，相当于告诉模型工具调用被拒绝的原因。
            content=str(blockd),
            # tool_call_id 必须对应原始调用 id，这样 LangChain 才能把结果和调用配对。
            tool_call_id=request.tool_call["id"],
            # name 保留原工具名，方便日志和模型理解是哪一个工具失败。
            name=tool_name,
            # status="error" 明确告诉 Agent 这次工具调用失败了。
            status="error",
        )

    # 如果没有被拦截，就调用 handler 真正执行工具。
    result = handler(request)

    # 工具执行完成后触发 PostToolUse，常用于打印日志、记录审计信息。
    trigger_hooks("PostToolUse", tool_name, tool_args, result)

    # 把真实工具结果返回给 Agent。
    return result


@after_agent
def stop_hook(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Agent 本轮运行结束后触发 Stop Hook。"""
    # 把完整消息列表传给 Stop 回调，方便统计本轮对话产生了多少消息。
    trigger_hooks("Stop", state.get("messages", []))

    # 同样不修改 Agent state，只做监听。
    return None


# WORKDIR 表示程序启动时所在的工作目录，后面会用它限制文件读写范围。
WORKDIR = Path.cwd()


# 从 .env 或系统环境变量中读取模型名称。
MODEL_ID = os.getenv("MODEL_ID")

# 这里读取 OPENAI_API_KEY，并作为 api_key 传给 ChatOpenAI。
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 系统提示词告诉模型：它是当前目录下的 coding agent，要通过工具解决任务。
SYSTEM = f"you are a coding agent at {WORKDIR}. Use tools to solve tasks. Act dont explain"

# 从环境变量读取兼容 OpenAI 接口的 base_url，例如 DeepSeek 或其他代理服务地址。
OPENAI_BASE_URL = os.getenv("BASE_URL")


"""第一层检查"""

# 统一走 harness.security 的大小写不敏感、覆盖更广的拒绝策略。
from harness.security import check_deny_list


def resolve_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (WORKDIR / candidate).resolve()


def check_rules(tool_name: str, args: dict) -> str | None:
    """执行较温和的规则检查。

    这类规则不一定直接拒绝，而是返回原因，让 ask_user 再询问用户是否允许。
    """
    # 针对执行命令工具做检查。
    if tool_name == "run_bash":
        # 从工具参数里拿到命令文本。
        command = args.get("command", "")

        # 这些关键词可能会破坏文件或修改系统权限，因此需要用户二次确认。
        if command.strip().lower().startswith("del ") or any(kw in command for kw in ["rm ", "> /etc/", "chmod 777"]):
            return "potentially destructive command"

    # 对文件读写编辑工具做路径限制，避免 Agent 操作工作区之外的文件。
    if tool_name in ("run_write", "run_edit", "run_read"):
        # 从工具参数中取文件路径；没有 path 时用空字符串兜底。
        path = args.get("path", "")

        # resolve_path(path) 会把相对路径转成绝对路径并消解 ..。
        # is_relative_to(WORKDIR) 用来确认最终路径仍然在工作区里面。
        if not resolve_path(path).is_relative_to(WORKDIR):
            return "Working outside workspace"

    # 没有触发任何规则，表示可以继续执行。
    return None


def ask_user(tool_name: str, args: dict, reason: str) -> bool:
    """遇到需要确认的操作时，在命令行里询问用户是否允许。"""
    # 打印触发确认的原因，让用户知道为什么要问。
    print(f"\nWarning: {reason}")

    # 打印工具名和参数，方便用户判断这次调用是否可信。
    print(f"Tool: {tool_name}({args})")

    # 默认是拒绝：只有输入 y 或 yes 才允许执行。
    choice = input("Allow? [y/N] ").strip().lower()
    return choice in ("y", "yes")


def check_permission(tool_name: str, args: dict) -> bool:
    """统一的权限检查入口。

    返回 True 表示允许工具继续执行。
    返回 False 表示拒绝工具调用。
    """
    # run_bash 风险最高，所以先经过硬拦截列表。
    if tool_name == "run_bash":
        # 从参数里取 command；没有 command 时用空字符串，避免 KeyError。
        reason = check_deny_list(args.get("command", ""))

        # 命中硬拦截就直接拒绝，不再询问用户。
        if reason:
            print(f"\nBlocked:{reason}")
            return False

    # 再执行普通规则检查，这些规则可能需要用户确认。
    reason = check_rules(tool_name, args)
    if reason:
        return ask_user(tool_name, args, reason)

    # 没有任何风险命中时，默认允许。
    return True


def on_user_prompt_submit(content):
    """UserPromptSubmit 事件的回调：打印用户刚提交的内容。"""
    # 这里主要用于观察 Hook 是否在 Agent 运行前被触发。
    print("[UserPromptSubmit]", content)


def on_pre_tool_use(tool_name, tool_args):
    """PreToolUse 事件的回调：打印工具调用信息，并做权限检查。"""
    # 打印即将调用的工具名和参数，方便调试 Agent 行为。
    print("[PreToolUse]", tool_name, tool_args)

    # 如果权限检查不通过，返回字符串。
    # trigger_hooks 会把这个字符串传回 tool_hook，从而阻止工具真正执行。
    if not check_permission(tool_name, tool_args):
        return "Permission denied"


def on_post_tool_use(tool_name, tool_args, result):
    """PostToolUse 事件的回调：打印工具调用完成后的结果。"""
    # 记录工具名和调用参数。
    print("[PostToolUse]", tool_name, tool_args)

    # 有些结果是 ToolMessage，内容在 .content；没有 .content 时就直接打印对象本身。
    print("result:", getattr(result, "content", result))


def on_stop(messages):
    """Stop 事件的回调：打印本轮结束时消息列表的长度。"""
    # len(messages) 可以粗略看出这一轮 Agent 产生了多少中间消息。
    print("[Stop]", len(messages))


# 把上面定义好的回调函数注册到对应事件上。
# 注册后，trigger_hooks("事件名", ...) 才能找到这些函数并执行。
register_hook("UserPromptSubmit", on_user_prompt_submit)
register_hook("PreToolUse", on_pre_tool_use)
register_hook("PostToolUse", on_post_tool_use)
register_hook("Stop", on_stop)


@tool
# 安全边界：shell=True 仅为教学演示，黑名单/路径检查不等于安全边界；生产请使用权限中间件 + 沙箱。
def run_bash(command: str) -> str:
    """Execute a shell command in the current workspace."""

    try:
        # 执行模型传入的 shell 命令。
        # shell=True 允许执行字符串命令；cwd=WORKDIR 限定命令运行目录。
        # capture_output=True 会同时捕获 stdout 和 stderr，避免直接刷屏。
        # text=True 表示用字符串形式返回输出，而不是 bytes。
        # timeout=120 限制最长运行 120 秒，避免命令卡死。
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # 合并标准输出和错误输出，方便模型统一读取执行结果。
        # strip() 去掉首尾多余空白，让返回内容更干净。
        out = (r.stdout + r.stderr).strip()

        # 限制工具返回长度，避免一次命令输出过长把上下文撑爆。
        # 如果没有任何输出，就明确返回 "(no output)"。
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        # 如果命令执行超过 120 秒，就返回超时提示。
        return "Error: Timeout(120s)"
    except OSError as e:
        # 捕获常见系统级异常，并把错误信息返回给模型。
        return f"Error: {e}"


@tool
def run_read(path: str, limit: int | None = None) -> str:
    """Read a UTF-8 text file, optionally limiting the returned line count."""
    try:
        # 读取整个文件并按行拆开，方便后面做行数限制。
        lines = resolve_path(path).read_text(encoding="utf-8").splitlines()

        # 如果传入 limit，并且文件行数超过 limit，就只返回前 limit 行。
        if limit and limit < len(lines):
            # 末尾追加一行提示，告诉调用者还有多少行没有展示。
            lines = lines[:limit] + [f"...({len(lines) - limit} more lines)"]

        # 再把行列表合并成一个字符串，作为工具结果返回。
        return "\n".join(lines)
    except Exception as e:
        # 任何读取错误都转成字符串返回，避免工具异常直接中断 Agent。
        return f"Error: {e}"


@tool
def run_write(path: str, content: str) -> str:
    """Write UTF-8 content to a file, replacing its existing content."""

    try:
        # 把传入路径转换成 Path 对象，后续创建目录和写文件更方便。
        file_path = resolve_path(path)

        # 如果父目录不存在，就自动创建；parents=True 表示多级目录也会一起创建。
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入文件内容；如果文件已存在，会被覆盖。
        file_path.write_text(content, encoding="utf-8")

        # 返回写入字节数/字符数提示，方便模型确认操作结果。
        return f"write {len(content)} bytes to {path}"
    except Exception as e:
        # 写入失败时，把异常信息返回给 Agent。
        return f"Error: {e}"


@tool
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace the first exact occurrence of old_text in a UTF-8 file."""

    try:
        # 先把文件路径包装成 Path 对象。
        file_path = resolve_path(path)

        # 读取原文件内容。
        text = file_path.read_text(encoding="utf-8")

        # 如果找不到 old_text，就返回错误，避免误以为编辑成功。
        if old_text not in text:
            return f"Error: text not found in {path}"

        # 只替换第一处匹配内容，避免一次工具调用误改太多地方。
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")

        # 返回简短成功提示。
        return f"edit {path}"
    except Exception as e:
        # 编辑过程中任何异常都转成字符串返回。
        return f"Error: {e}"


@tool
def run_glob(pattern: str) -> str:
    """Find workspace files matching a glob pattern."""
    # 在函数内部导入 glob，只有调用这个工具时才加载它。
    import glob as g

    try:
        # results 用来收集所有通过安全检查的匹配结果。
        results = []

        # root_dir=WORKDIR 表示匹配范围从工作目录开始。
        for match in g.glob(pattern, root_dir=WORKDIR):
            # 再次确认匹配到的路径没有跳出 WORKDIR。
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)

        # 如果有匹配结果，就逐行返回；否则返回固定提示。
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        # glob 过程出错时，把错误信息返回给 Agent。
        return f"Error: {e}"


def print_assistant_message(message: AIMessage) -> None:
    """把模型回复打印到终端。

    不同模型/适配器返回的 message.content 结构可能不一样：
    有时是普通字符串，有时是多个内容块组成的列表。
    """
    # AIMessage.content 可能是字符串，也可能是内容块列表。
    content = message.content

    # 如果是普通字符串，直接打印。
    if isinstance(content, str):
        print(content)
        return

    # 如果是内容块列表，就逐块提取文本内容。
    if isinstance(content, list):
        for block in content:
            # OpenAI/Anthropic 不同适配器可能返回 dict 格式的内容块。
            if isinstance(block, dict):
                # type == "text" 时，真正的文本通常在 text 字段。
                if block.get("type") == "text":
                    print(block.get("text", ""))

            # 有些内容块可能是对象，并通过 .text 保存文本。
            elif hasattr(block, "text"):
                print(block.text)


# TOOLS 是提供给 Agent 使用的工具列表。
# 顺序不一定决定模型选择哪个工具，但会影响工具注册展示的顺序。
TOOLS = [run_bash, run_edit, run_glob, run_write, run_read]

# MIDDLEWARE 是 Agent 生命周期钩子列表。
# user_prompt_submit 负责运行前监听；tool_hook 负责工具前后监听；stop_hook 负责结束后监听。
MIDDLEWARE = [user_prompt_submit, tool_hook, stop_hook]


# 创建模型对象。
# ChatOpenAI 只要求接口兼容 OpenAI，不一定必须连接 OpenAI 官方服务。
MODEL = ChatOpenAI(
    # 模型名称来自环境变量 MODEL_ID。
    model=MODEL_ID,
    # 限制模型最多输出 8000 token。
    max_completion_tokens=8000,
    # temperature=0 表示尽量稳定、确定性输出。
    temperature=0,
    # API key 来自 OPENAI_API_KEY 环境变量。
    api_key=OPENAI_API_KEY,
    # base_url 来自 BASE_URL 环境变量。
    base_url=OPENAI_BASE_URL,
)

# 创建 Agent，把模型、工具、系统提示词和 middleware 组合起来。
agent = create_agent(
    model=MODEL,
    tools=TOOLS,
    system_prompt=SYSTEM,
    middleware=MIDDLEWARE,
)


def agent_loop(messages: list) -> None:
    """执行一轮 Agent 对话，并把新增消息打印出来。"""
    # 把当前完整历史消息交给 Agent。
    # Agent 可能会在内部多次调用工具，最后返回更新后的 messages。
    result = agent.invoke({"messages": messages})

    # 只取本轮新产生的消息，避免每轮都重复打印完整历史。
    new_messages = result["messages"][len(messages):]

    # 逐条打印本轮新增消息。
    for message in new_messages:
        # 如果消息里有 tool_calls，说明这是模型发起工具调用的中间消息。
        if hasattr(message, "tool_calls") and message.tool_calls:
            print("模型调用工具:")

            # 一个模型消息里可能包含多个工具调用，所以这里逐个打印。
            for tool_call in message.tool_calls:
                print("工具名：", tool_call["name"])
                print("参数：", tool_call.get("args", {}))

        # ToolMessage 表示某个工具执行完成后返回的结果。
        elif message.__class__.__name__ == "ToolMessage":
            print("工具返回结果：")
            print("工具名", getattr(message, "name", None))
            print("内容:", message.content)

        # 其他情况通常就是模型给用户的最终回复。
        else:
            print("模型回复:")
            print(getattr(message, "content", message))

        # 每条消息之间空一行，终端输出更清楚。
        print()

    # 用 Agent 返回的新消息列表覆盖原列表，实现历史对话持续累积。
    # 这里用切片赋值，是为了保留外部传入的 messages 这个列表对象。
    messages[:] = result["messages"]


# 只有直接运行 python s04.py 时，下面这段命令行交互才会执行。
# 如果这个文件被其他模块 import，则不会启动交互循环。
if __name__ == "__main__":
    print("s04: Hook")
    print("输入问题，回车发送。输入 q 退出。\n")

    # history 保存用户和 Agent 的完整对话历史。
    history = []

    # 用 while True 持续读取用户输入，直到用户主动退出或终端中断。
    while True:
        try:
            # \033[36m 和 \033[0m 是 ANSI 颜色码，用来把提示符显示成青色。
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D / Ctrl+Z 或 Ctrl+C 时退出循环。
            break

        # 输入 q、exit 或空字符串时退出程序。
        if query.strip().lower() in ("q", "exit", ""):
            break

        # 把用户输入追加到历史消息中，role="user" 表示这条消息来自用户。
        history.append({"role": "user", "content": query})

        # 交给 Agent 执行一轮，并在函数内部更新 history。
        agent_loop(history)
