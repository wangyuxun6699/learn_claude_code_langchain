from dotenv import load_dotenv
load_dotenv(override=True)

from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool
import os,subprocess
from pathlib import Path
from langchain_core.messages import AIMessage,ToolMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


WORKDIR = Path.cwd()
MODEL_ID = os.getenv("MODEL_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SYSTEM = f"you are a coding agent at {WORKDIR}. Use tools to solve tasks. Act dont explain"
OPENAI_BASE_URL = os.getenv("BASE_URL")


"""第一层检查：本文件直接实现大小写不敏感拒绝策略。"""
# -- 本文件的命令拒绝策略 --
import re

DANGEROUS_PHRASES: list[str] = [
    "rm -rf /",
    "rm -fr /",
    "rm -rf --no-preserve-root",
    "mkfs",
    "dd if=",
    "dd if =",
    "> /dev/sd",
    ">/dev/sd",
    "> /dev/hd",
    "> /dev/disk",
    "shutdown -h",
    "shutdown -r",
    "shutdown -p",
    "systemctl poweroff",
    "systemctl reboot",
    "systemctl halt",
    "init 0",
    "init 6",
    "poweroff --",
    "reboot --",
    "chmod 777 /",
    "chmod -r 777 /",
    ":(){:|:&};:",
    ":(){ :|:& };:",
]

DANGEROUS_WORDS: list[str] = [
    "sudo",
    "su",
    "doas",
    "shutdown",
    "reboot",
    "poweroff",
    "halt",
]

CONFIRM_PHRASES: list[str] = [
    "rm ",
    "del ",
    "> /etc/",
    "chmod 777",
    "chmod -r",
]


def normalize(command) -> str:
    """小写并合并连续空白，让大小写与多空格变体失效。"""
    return re.sub(r"\s+", " ", str(command or "")).strip().lower()


def _segment_leading_words(command) -> set:
    """返回位于管道段开头的单词集合，用于短危险词的词边界匹配。"""
    text = str(command or "").lower()
    return set(re.findall(r"(?:^|[;&|()]\s*)([a-z][a-z0-9_-]*)", text))


def check_deny_list(command) -> str | None:
    """返回禁止原因；命令安全时返回 None。"""
    normalized = normalize(command)
    for phrase in DANGEROUS_PHRASES:
        if phrase in normalized:
            return f"'{phrase}' is on the deny list"
    leading = _segment_leading_words(command)
    for word in DANGEROUS_WORDS:
        if word in leading:
            return f"'{word}' is on the deny list"
    return None


_BASH_TOOL_NAMES = {"bash", "run_bash", "execute_bash", "run_shell"}


def check_confirmation(tool_name: str, args: dict) -> str | None:
    """返回需要用户确认的原因；无需确认时返回 None。"""
    if tool_name in _BASH_TOOL_NAMES:
        command = str(args.get("command", ""))
        normalized = normalize(command)
        if normalized.strip().startswith("del "):
            return "potentially destructive command: del"
        for phrase in CONFIRM_PHRASES:
            if phrase in normalized:
                return f"potentially destructive command: {phrase.strip()}"
    return None


def check_permission(
    tool_name: str,
    args: dict,
    *,
    ask=None,
) -> bool:
    """统一权限入口：硬拒绝优先，其次按需请求用户确认（默认拒绝）。"""
    command = str(args.get("command", "")) if tool_name in _BASH_TOOL_NAMES else ""
    if check_deny_list(command):
        return False
    reason = check_confirmation(tool_name, args)
    if reason and ask is not None:
        return ask(tool_name, args, reason)
    return reason is None



def resolve_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (WORKDIR / candidate).resolve()


def check_rules(tool_name: str,args: dict) -> str|None:
    if tool_name == "run_bash":
        command = args.get("command","")
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



def check_permission(tool_name: str,args: dict) ->bool:
    if tool_name == "run_bash":
        reason = check_deny_list(args.get("command", ""))
        if reason:
            print(f"\nBlocked: {reason}")
            return False
        
    reason = check_rules(tool_name, args)
    if reason:
        return ask_user(tool_name, args, reason)
    

    return True



class permission_check(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        tool_name = request.tool.name if request.tool else request.tool_call["name"]
        args = request.tool_call.get("args",{})
        tool_call_id =request.tool_call["id"]

        if not check_permission(tool_name, args):
            return ToolMessage(
                content = "Permission denied",
                tool_call_id = tool_call_id,
                name = tool_name,
                status = "error"
            )
        
        return handler(request)
    



@tool
# 安全边界：shell=True 仅为教学演示，黑名单/路径检查不等于安全边界；生产请使用权限中间件 + 沙箱。
def run_bash(command:str)->str:
    """Execute a shell command in the current workspace."""

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
        lines = resolve_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"...({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


 
@tool
def run_write(path: str,content: str)-> str:
    """Write UTF-8 content to a file, replacing its existing content."""

    try:
        file_path = resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"write {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"



@tool
def run_edit(path: str, old_text: str, new_text: str) ->str:
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
    middleware=[permission_check()],
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
        print()
    messages[:] = result["messages"]

if __name__ == "__main__":
    print("s03.5: Permission before agent run tools")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)

#        print()

