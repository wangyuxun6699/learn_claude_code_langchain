from dotenv import load_dotenv
load_dotenv(override=True)


from langchain_core.tools import tool
import os,subprocess
from pathlib import Path
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent



WORKDIR = Path.cwd()
MODEL_ID = os.getenv("MODEL_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SYSTEM = f"you are a coding agent at {WORKDIR}. Use tools to solve tasks. Act dont explain"
OPENAI_BASE_URL = os.getenv("BASE_URL")


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


@tool
# 安全边界：shell=True 仅为教学演示，黑名单/路径检查不等于安全边界；生产请使用权限中间件 + 沙箱。
def run_bash(command:str)->str:
    """Execute a shell command in the current workspace."""
    # 简单拦截一些明显危险的命令，避免模型误删系统文件或关机。
    dangerous = ["rm -rf /","sudo","shutdown","reboot",">/dev/"]

    # 只要命令里包含危险片段，就直接返回错误，不真正执行。
    if any(d in command for d in dangerous):
        return "Error : Dangerous command blocked"

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



@tool
def run_read(path: str, limit: int | None = None) -> str:
    """Read a UTF-8 text file, optionally limiting the returned line count."""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"...({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"
    
@tool
def run_write(path: str,content: str)-> str:
    """Write UTF-8 content to a file, replacing its existing content."""
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"write {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

@tool
def run_edit(path: str, old_text: str, new_text: str) ->str:
    """Replace the first exact occurrence of old_text in a UTF-8 file."""
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"edit {path}"
    except Exception as e:
        return f"Error: {e}"

@tool
def run_glob(pattern: str) ->str:
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


TOOLS = [run_bash,run_edit,run_glob,run_write,run_read]


MODEL = ChatOpenAI(
    model = MODEL_ID,
    max_completion_tokens = 8000,
    temperature = 0,
    api_key = OPENAI_API_KEY,
    base_url = OPENAI_BASE_URL,
)

agent = create_agent(
    model=MODEL,
    tools=TOOLS,
    system_prompt=SYSTEM,
)

def agent_loop(messages: list) -> None:
    result = agent.invoke({"messages": messages})
    new_messages = result["messages"][len(messages):]
    for message in new_messages:
        if hasattr(message, "tool_calls") and message.tool_calls:
            print("模型调用工具:")
            for tool_call in message.tool_calls:
                print("工具名：", tool_call["name"])
                print("参数：", tool_call.get("args", {}))
        elif message.__class__.__name__=="ToolMessage":
            print("工具返回结果：")
            print("工具名", getattr(message, "name", None))
            print("内容:", message.content)

        else:
            print("模型回复:")
            print(getattr(message, "content", message))

    messages[:] = result["messages"]



if __name__ == "__main__":
    print("s02: Tool Use — 在 s01 基础上加了 4 个工具")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
#        print(history)

#        last_message = history[-1]
#        print_assistant_message(last_message)
        print()