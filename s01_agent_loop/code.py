#!/usr/bin/env python3
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while True:
        response = LLM(messages, tools)
        if response contains no tool_use:
            break
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Later chapters add policy,
hooks, and lifecycle controls around it.

Usage:
    pip install -r requirements.txt
    OPENAI_API_KEY=... BASE_URL=... python s01_agent_loop/code.py
"""

import os
import subprocess

try:
    import readline
    # #143 UTF-8 backspace fix for macOS libedit
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

# -- 本章内置的 LangChain 消息适配（直接展开，便于单文件阅读） --
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI


@dataclass(slots=True)
class TextBlock:
    """课程循环读取的文本内容块。"""

    text: str
    type: str = field(default="text", init=False)


@dataclass(slots=True)
class ToolUseBlock:
    """课程循环读取的工具调用内容块。"""

    id: str
    name: str
    input: dict[str, Any]
    type: str = field(default="tool_use", init=False)


@dataclass(slots=True)
class Usage:
    """统一暴露工作流和 Goal 统计所需的 token 字段。"""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(slots=True)
class MessageResponse:
    """课程侧需要的最小模型响应。"""

    content: list[TextBlock | ToolUseBlock]
    stop_reason: str
    usage: Usage
    raw: AIMessage


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _block_type(block: Any) -> str | None:
    return _value(block, "type")


def _text_content(content: Any) -> str:
    """把 provider 内容块、工具结果或普通对象安全地转成文本。"""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        return str(content)

    texts: list[str] = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
            continue
        block_text = _value(block, "text")
        if isinstance(block_text, str):
            texts.append(block_text)
            continue
        nested = _value(block, "content")
        if nested is not None:
            texts.append(_text_content(nested))
    return "\n".join(text for text in texts if text)


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    return _text_content(system)


def _assistant_message(content: Any) -> AIMessage:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    blocks = content if isinstance(content, list) else [content]

    for block in blocks:
        kind = _block_type(block)
        if kind == "tool_use":
            tool_calls.append(
                {
                    "id": str(_value(block, "id", "")),
                    "name": str(_value(block, "name", "")),
                    "args": dict(_value(block, "input", {}) or {}),
                    "type": "tool_call",
                }
            )
            continue
        text = _value(block, "text")
        if isinstance(text, str) and text:
            text_parts.append(text)

    return AIMessage(content="\n".join(text_parts), tool_calls=tool_calls)


def _user_messages(content: Any) -> list[BaseMessage]:
    if isinstance(content, str):
        return [HumanMessage(content=content)]
    if not isinstance(content, list):
        return [HumanMessage(content=str(content))]

    results: list[BaseMessage] = []
    user_text: list[str] = []
    for block in content:
        if _block_type(block) == "tool_result":
            if user_text:
                results.append(HumanMessage(content="\n".join(user_text)))
                user_text.clear()
            results.append(
                ToolMessage(
                    content=_text_content(_value(block, "content", "")),
                    tool_call_id=str(_value(block, "tool_use_id", "")),
                    status="error" if bool(_value(block, "is_error", False)) else "success",
                )
            )
            continue
        text = _value(block, "text")
        if isinstance(text, str):
            user_text.append(text)
        elif isinstance(block, str):
            user_text.append(block)

    if user_text or not results:
        results.append(HumanMessage(content="\n".join(user_text)))
    return results


def _to_langchain_messages(messages: list[Any]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, BaseMessage):
            converted.append(message)
            continue
        role = _value(message, "role")
        content = _value(message, "content", "")
        if role == "assistant":
            converted.append(_assistant_message(content))
        elif role == "system":
            converted.append(SystemMessage(content=_text_content(content)))
        else:
            converted.extend(_user_messages(content))
    return converted


def _openai_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for item in tools or []:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": item["name"],
                    "description": item.get("description", ""),
                    "parameters": item.get("input_schema", {"type": "object"}),
                },
            }
        )
    return converted


def _response_blocks(message: AIMessage) -> list[TextBlock | ToolUseBlock]:
    blocks: list[TextBlock | ToolUseBlock] = []
    text = _text_content(message.content)
    if text:
        blocks.append(TextBlock(text=text))
    for call in message.tool_calls:
        blocks.append(
            ToolUseBlock(
                id=str(call.get("id", "")),
                name=str(call.get("name", "")),
                input=dict(call.get("args", {}) or {}),
            )
        )
    return blocks


def _usage(message: AIMessage) -> Usage:
    metadata = message.usage_metadata or {}
    return Usage(
        input_tokens=int(metadata.get("input_tokens", 0) or 0),
        output_tokens=int(metadata.get("output_tokens", 0) or 0),
    )


class _MessagesAPI:
    def __init__(self, owner: "LangChainMessagesClient") -> None:
        self.owner = owner

    def create(
        self,
        *,
        model: str,
        messages: list[Any],
        system: Any = "",
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8000,
        temperature: float = 0,
        **_: Any,
    ) -> MessageResponse:
        llm = ChatOpenAI(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("BASE_URL") or self.owner.base_url,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            timeout=self.owner.timeout,
            max_retries=self.owner.max_retries,
        )
        openai_tools = _openai_tools(tools)
        runnable = llm.bind_tools(openai_tools) if openai_tools else llm
        request = [SystemMessage(content=_system_text(system))]
        request.extend(_to_langchain_messages(messages))
        raw = runnable.invoke(request)
        if not isinstance(raw, AIMessage):
            raw = AIMessage(content=str(getattr(raw, "content", raw)))

        finish_reason = str((raw.response_metadata or {}).get("finish_reason", ""))
        if raw.tool_calls:
            stop_reason = "tool_use"
        elif finish_reason in {"length", "max_tokens"}:
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"
        return MessageResponse(
            content=_response_blocks(raw),
            stop_reason=stop_reason,
            usage=_usage(raw),
            raw=raw,
        )


class LangChainMessagesClient:
    """课程统一模型边界；真实网络调用延迟到 ``create``。"""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int = 2,
        **_: Any,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.messages = _MessagesAPI(self)

# -- 本章的 Agent / Harness 机制 --
from dotenv import load_dotenv

load_dotenv(override=True)

client = LangChainMessagesClient(base_url=os.getenv("BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# -- Tool definition: just bash --
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


# -- Tool execution --
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, errors="replace", timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        # If the model didn't call a tool, we're done
        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        if not tool_calls:
            return

        # Execute each tool call, collect results
        results = []
        for block in tool_calls:
            print(f"\033[33m$ {block.input['command']}\033[0m")
            output = run_bash(block.input["command"])
            print(output[:200])
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })

        # Feed tool results back, loop continues
        messages.append({"role": "user", "content": results})


# -- Entry point --
if __name__ == "__main__":
    print("s01: Agent Loop")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            # \001/\002 tell Readline the ANSI escapes have zero display width.
            query = input("\001\033[36m\002s01 >> \001\033[0m\002")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        # Print the model's final text response
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()
