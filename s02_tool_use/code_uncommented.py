"""s02_tool_use.py - 在 s01 的 ``create_agent`` 上增加结构化工具。

Agent Loop、模型配置和流式输出与 s01 保持一致。本章只新增四个文件工具，
并把它们和 bash 一起注册到 LangChain Agent。

Usage:
    pip install -r requirements.txt
    OPENAI_API_KEY=... BASE_URL=... MODEL_ID=... python s02_tool_use/code.py
"""

import glob as glob_module
import os
import subprocess
from pathlib import Path

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
from langchain.messages import AIMessageChunk
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

WORKDIR = Path.cwd().resolve()
MODEL = os.environ["MODEL_ID"]
SYSTEM = f"You are a coding agent at {WORKDIR}. Use the available tools to solve tasks. Act, don't explain."


def safe_path(path: str) -> Path:
    """把用户提供的路径解析到工作区内，越界时拒绝。"""
    resolved = (WORKDIR / path).resolve()
    if not resolved.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


def print_tool_result(name: str, detail: str, output: str) -> str:
    """显示工具调用和结果预览，同时把完整结果返回给 Agent。"""
    print(f"\n\033[33m> {name}({detail})\033[0m")
    print(output[:200])
    return output


@tool
def bash(command: str) -> str:
    """Run a shell command in the workspace and return its combined output."""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(fragment in command for fragment in dangerous):
        output = "Error: Dangerous command blocked"
    else:
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
    """Read a UTF-8 text file in the workspace, optionally limiting the number of lines."""
    try:
        if limit is not None and limit < 1:
            raise ValueError("limit must be a positive integer")
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit is not None and limit < len(lines):
            omitted = len(lines) - limit
            lines = lines[:limit] + [f"... ({omitted} more lines)"]
        output = "\n".join(lines)
    except (OSError, UnicodeError, ValueError) as error:
        output = f"Error: {error}"
    return print_tool_result("read_file", path, output)


@tool
def write_file(path: str, content: str) -> str:
    """Write UTF-8 text to a workspace file, replacing it and creating parent directories."""
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8", newline="")
        output = f"Wrote {len(content)} characters to {path}"
    except (OSError, UnicodeError, ValueError) as error:
        output = f"Error: {error}"
    return print_tool_result("write_file", path, output)


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace the first exact occurrence of old_text in a UTF-8 workspace file."""
    try:
        file_path = safe_path(path)
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

agent = None


def get_agent():
    """创建并复用注册了五个工具的 LangChain Agent。"""
    global agent
    if agent is None:
        model = ChatOpenAI(
            model=MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("BASE_URL") or None,
            max_completion_tokens=8000,
            temperature=0,
        )
        agent = create_agent(model=model, tools=TOOLS, system_prompt=SYSTEM)
    return agent


def agent_loop(messages: list) -> None:
    """流式运行完整 Agent Loop，并把最终消息写回会话历史。"""
    final_messages = messages
    for chunk in get_agent().stream(
        {"messages": messages},
        stream_mode=["messages", "values"],
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if isinstance(token, AIMessageChunk) and metadata.get("langgraph_node") == "model" and token.text:
                print(token.text, end="", flush=True)
        elif chunk["type"] == "values":
            final_messages = chunk["data"]["messages"]

    messages[:] = final_messages
    print()


if __name__ == "__main__":
    print("s02: Tool Use (five LangChain tools)")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            query = input("\001\033[36m\002s02 >> \001\033[0m\002")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        print()
