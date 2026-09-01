# 导入 os，用来读取环境变量和当前工作目录。
import os

# readline 可以让命令行输入体验更好，例如支持方向键、历史输入等。
# Windows 环境里可能没有 readline，所以这里用 try/except 做兼容。
try:
    import readline

    # 修复 macOS libedit 下 UTF-8 退格等输入问题。
    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
    readline.parse_and_bind("set enable-meta-keybindings on")
except ImportError:
    # 如果当前系统没有 readline，就跳过，不影响主程序运行。
    pass

# subprocess 用来执行 shell 命令，后面会封装成 bash 工具给智能体调用。
import subprocess
from pathlib import Path

# load_dotenv 用来从 .env 文件加载模型、key、base_url 等配置。
from dotenv import load_dotenv
from harness.security import check_deny_list

# 旧版本使用 ChatAnthropic，这里保留注释，方便对比，不再实际导入。
# from langchain_anthropic import ChatAnthropic

# create_agent 用来创建 LangChain 智能体，它会自动处理模型调用和工具调用循环。
from langchain.agents import create_agent

# ChatOpenAI 是 OpenAI 兼容格式的 ChatModel，可对接 OpenAI、DeepSeek 等兼容接口。
from langchain_openai import ChatOpenAI

# HumanMessage 表示用户消息，ToolMessage 表示工具返回，AIMessage 表示模型回复。
# SystemMessage 当前只在旧写法注释中保留，用来说明之前的 system prompt 传法。
from langchain_core.messages import HumanMessage,ToolMessage,AIMessage

# StructuredTool 可以把普通 Python 函数包装成模型可调用的工具。
from langchain_core.tools import StructuredTool

# 读取 .env 文件，并允许 .env 覆盖当前环境里已有的同名变量。
load_dotenv(override=True)


# 从环境变量读取模型名称，例如 deepseek-chat、gpt-4o-mini 等。
MODEL = os.environ["MODEL_ID"]

# 读取 OpenAI 兼容接口需要的 API key。
# 使用标准的 OPENAI_API_KEY 环境变量。
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# 读取 OpenAI 兼容接口的 base_url。
# 例如 DeepSeek 的 OpenAI 兼容地址通常类似 https://api.deepseek.com/v1。
OPENAI_BASE_URL = os.getenv("BASE_URL")

# 记录程序启动时所在的工作目录，后续 bash 工具都在这个目录里执行命令。
WORKDIR = Path.cwd()

# system prompt 用来告诉模型它是一个代码智能体，以及应该在哪个目录里工作。
SYSTEM = f"you are a coding agent at {WORKDIR}. Use bash to solve tasks. Act dont explain"

# 把执行 shell 命令的能力封装成普通函数，后面会注册成智能体工具。
# 安全边界：shell=True 仅为教学演示，黑名单/路径检查不等于安全边界；生产请使用权限中间件 + 沙箱。
def run_bash(command:str)->str:
    """Execute a shell command in the current workspace."""
    # 统一走 harness.security 的大小写不敏感、覆盖更广的拒绝策略。
    denied = check_deny_list(command)
    if denied:
        return f"Blocked: {denied}"

    try:
        # 执行模型传入的 shell 命令。
        # shell=True 允许执行字符串命令；cwd=WORKDIR 限定命令运行目录。
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # 合并标准输出和错误输出，方便模型统一读取执行结果。
        out = (r.stdout + r.stderr).strip()

        # 限制工具返回长度，避免一次命令输出过长把上下文撑爆。
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        # 如果命令执行超过 120 秒，就返回超时提示。
        return "Error: Timeout(120s)"
    except OSError as e:
        # 捕获常见系统级异常，并把错误信息返回给模型。
        return f"Error: {e}"

# 把 run_bash 函数包装成 LangChain 工具，工具名叫 bash。
bash_tool = StructuredTool.from_function(
    func=run_bash,
    name="bash",
    description="Run a shell command",
)


# 当前智能体可用的工具列表。
TOOLS=[bash_tool]

# 工具名到工具对象的映射，主要用于旧版手写 agent_loop，对新写法不是必需。
#TOOL_MAP = {tool.name: tool for tool in TOOLS}


# 构建 OpenAI 兼容格式的 ChatModel。
def build_chat_model():
    # ChatOpenAI 的基础参数：模型名、最大输出 token 数和温度。
    kwargs = {
        "model":MODEL,
        "max_completion_tokens":8000,
        "temperature":0,
    }

    # 如果配置了 API key，就传给 ChatOpenAI。
    if OPENAI_API_KEY:
        kwargs["api_key"] = OPENAI_API_KEY

    # 如果配置了 base_url，就让 ChatOpenAI 请求这个 OpenAI 兼容接口。
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL

    # 返回 LangChain 可调用的 ChatModel 实例。
    return ChatOpenAI(**kwargs)


# 创建 ChatModel 实例。
chat_model = build_chat_model()

# 基于 ChatModel、工具列表和 system prompt 创建智能体。
# create_agent 会自动完成“模型决定是否调用工具 -> 执行工具 -> 把结果交回模型”的循环。
agent = create_agent(
    model=chat_model,
    tools=TOOLS,
    system_prompt=SYSTEM,
)


# 旧版本写法：创建 ChatAnthropic、绑定工具，然后手动处理 tool_calls。
# 当前版本已经改成 ChatOpenAI + create_agent，所以工具调用循环由 agent 自动处理。
#
# def build_llm():
#     # 构造 Anthropic 模型参数。
#     kwargs = {
#         "model":MODEL,
#         "max_tokens":8000,
#         "temperature":0,
#     }
#
#
#     # 返回 Anthropic ChatModel。
#     return ChatAnthropic(**kwargs)
#
#
# # 创建旧版 llm，并手动绑定工具。
# llm = build_llm()
# llm_with_tools = llm.bind_tools(TOOLS)


# 打印模型最终回复内容。
def print_assistant_message(message:AIMessage) -> None:
    # AIMessage.content 可能是字符串，也可能是内容块列表。
    content = message.content

    # 如果是普通字符串，直接打印。
    if isinstance(content, str):
        print(content)
        return

    # 如果是内容块列表，就逐块提取文本内容。
    if isinstance(content,list):
        for block in content:
            # OpenAI/Anthropic 不同适配器可能返回 dict 格式的内容块。
            if isinstance(block, dict):
                if block.get("type") == "text":
                    print(block.get("text", ""))
            # 有些内容块可能是对象，并通过 .text 保存文本。
            elif hasattr(block, "text"):
                print(block.text)

# 打印工具调用过程，方便在终端看到模型执行了哪些命令。
def print_tool_activity(message:AIMessage|ToolMessage) -> None:
    # 如果当前消息是模型消息，就检查它是否请求调用工具。
    if isinstance(message, AIMessage):
        for tool_call in message.tool_calls:
            # 取出工具名、工具参数和 bash 命令。
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            command = tool_args.get("command", "")

            # 如果模型调用的是 bash 工具，就用黄色打印命令。
            if tool_name == "bash":
                print(f"\033[33m$ {command}\033[0m")
        return

    # 如果当前消息是工具返回，就打印前 200 个字符，避免终端刷太多内容。
    if isinstance(message, ToolMessage):
        print(str(message.content)[:200])


# 执行一次智能体循环：传入历史消息，调用 agent，并把结果写回 history。
def agent_loop(messages:list) ->None:
    # create_agent 接收 {"messages": messages} 这样的输入格式。
    result = agent.invoke({"messages": messages})

    # 找出本轮新增的消息，用来打印工具调用过程。
    new_messages = result["messages"][len(messages):]

    # 逐条打印本轮发生的工具调用和工具返回。
    for message in new_messages:
        print_tool_activity(message)

    # 用 agent 返回的完整消息列表更新原 history。
    messages[:] = result["messages"]


# 旧版本的手写 agent_loop，保留作对照。
#
# def agent_loop(messages:list) ->None:
#     # 不断调用模型，直到模型不再请求工具调用。
#     while True:
#         # 调用绑定了工具的 llm。
#         response = llm_with_tools.invoke(messages)
#         # 把模型回复加入历史。
#         messages.append(response)
#
#
#         # 如果模型没有请求工具调用，本轮 agent_loop 结束。
#         if not response.tool_calls:
#             return
#
#         # 遍历模型请求的所有工具调用。
#         for tool_call in response.tool_calls:
#             # 解析工具名、参数和 tool_call_id。
#             tool_name = tool_call["name"]
#             tool_args = tool_call.get("args", {})
#             tool_id = tool_call["id"]
#
#             # 根据工具名找到对应工具对象。
#             tool = TOOL_MAP.get(tool_name)
#
#
#             # 如果模型请求了不存在的工具，就返回错误信息。
#             if tool is None:
#                 output = f"Error: unknown tool: {tool_name}"
#             else:
#                 # 取出 bash 命令文本。
#                 command = tool_args.get("command","")
#
#                 # 如果是 bash 工具，就先把命令打印出来。
#                 if tool_name == "bash":
#                     print(f"\033[33m$ {command}\033[0m")
#
#                 # 真正执行工具，并捕获执行异常。
#                 try:
#                     output = tool.invoke(tool_args)
#                 except Exception as e:
#                     output = f"Error: {e}"
#
#             # 只打印工具输出前 200 个字符，避免刷屏。
#             print(output[:200])
#
#             # 把工具返回封装成 ToolMessage，再交回模型继续推理。
#             messages.append(
#                 ToolMessage(
#                     content=output,
#                     tool_call_id=tool_id,
#                     name=tool_name
#                 )
#             )

# 只有直接运行这个文件时，才进入交互式命令行。
if __name__=="__main__":
    # system prompt 已经交给 create_agent 注入，这里只保留用户和模型的对话历史。
    # 旧版本写法如下：把 SystemMessage 手动放进 history。
    # history = [
    #     SystemMessage(content=SYSTEM),
    # ]

    # 初始化对话历史。
    history = []

    # 进入命令行循环，持续读取用户输入。
    while True:
        try:
            # 读取一行用户输入，并用青色显示提示符。
            query = input("\033[36ms01 >> \033[0m")

        except (EOFError,KeyboardInterrupt):
            # 用户按 Ctrl+D 或 Ctrl+C 时退出循环。
            break

        # 输入 q、exit 或空字符串时退出程序。
        if query.strip().lower() in ("q","exit",""):
            break

        # 把用户输入包装成 HumanMessage，加入历史。
        history.append(HumanMessage(content=query))
#        print(history)
        # 调用智能体处理本轮用户请求。
        agent_loop(history)

        # 取出智能体本轮结束后的最后一条消息。
        last_message = history[-1]

        # 如果最后一条是模型回复，就打印给用户看。
        if isinstance(last_message,AIMessage):
            print_assistant_message(last_message)

        # 每轮对话后打印空行，让终端输出更清楚。
        print()
