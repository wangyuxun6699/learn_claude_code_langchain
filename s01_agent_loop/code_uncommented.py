"""s01_agent_loop.py - 用 LangChain ``create_agent`` 实现最小 Agent Loop。

参考实现的核心能力保持不变：模型可以调用一个 bash 工具，并根据工具结果
继续推理，直到给出最终回答。模型、工具调用和结果回传的循环由 LangChain 管理。

Usage:
    pip install -r requirements.txt
    OPENAI_API_KEY=... BASE_URL=... MODEL_ID=... python s01_agent_loop/code.py
"""

import os
import subprocess

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

MODEL = os.environ["MODEL_ID"]
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."


@tool
def bash(command: str) -> str:
    """Run a shell command in the current working directory."""
    print(f"\n\033[33m$ {command}\033[0m")

    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(fragment in command for fragment in dangerous):
        output = "Error: Dangerous command blocked"
    else:
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=os.getcwd(),
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

    print(output[:200])
    return output


agent = None


def get_agent():
    """创建并复用 LangChain Agent。"""
    global agent
    if agent is None:
        model = ChatOpenAI(
            model=MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("BASE_URL") or None,
            max_completion_tokens=8000,
            temperature=0,
        )
        agent = create_agent(model=model, tools=[bash], system_prompt=SYSTEM)
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
            if (
                isinstance(token, AIMessageChunk)
                and metadata.get("langgraph_node") == "model"
                and token.text
            ):
                print(token.text, end="", flush=True)
        elif chunk["type"] == "values":
            final_messages = chunk["data"]["messages"]

    messages[:] = final_messages
    print()


if __name__ == "__main__":
    print("s01: Agent Loop (LangChain create_agent)")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            query = input("\001\033[36m\002s01 >> \001\033[0m\002")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        print()
