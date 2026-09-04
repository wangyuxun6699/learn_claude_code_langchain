"""
s11：错误恢复。

本章把恢复能力放在 AgentMiddleware 中，覆盖三条主要路径：
1. 输出达到 token 上限：8K 升级到 64K，再用续写提示最多恢复 3 次；
2. 输入上下文过长：响应式裁剪历史，然后只重试一次；
3. 429/529：读取 Retry-After，执行指数退避；连续 3 次 529 后切备用模型。

create_agent 仍负责标准的“模型 -> 工具 -> 模型”循环，中间件只处理模型调用。
运行方式：python -m legacy.s11_error_recovery.code
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import RLock
from typing import Any, Literal, NotRequired, TypedDict

from dotenv import load_dotenv

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
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

load_dotenv(override=True)

WORKDIR = Path.cwd().resolve()
MEMORY_INDEX = WORKDIR / ".memory" / "MEMORY.md"

MODEL_ID = os.environ["MODEL_ID"]
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in .env")

fallback_value = os.getenv("FALLBACK_MODEL_ID", "").strip()

FALLBACK_MODEL_ID = (
    fallback_value
    if fallback_value and fallback_value != "your-fallback-model-id"
    else None
)

DEFAULT_MAX_TOKENS = 8_000
ESCALATED_MAX_TOKENS = 64_000
MAX_RECOVERY_RETRIES = 3
MAX_RETRIES = 10
BASE_DELAY_SECONDS = 0.5
MAX_DELAY_SECONDS = 32.0
MAX_CONSECUTIVE_529 = 3
CONTINUATION_PROMPT = (
    "Output token limit hit. Resume directly — no apology, no recap. "
    "Pick up mid-thought and break the remaining work into smaller pieces."
)

PROMPT_SECTIONS = {
    "identity": (
        "You are a coding agent. "
        "Solve the user's task by acting with the available tools. "
        "Keep explanations concise."
    ),
    "tools": (
        "Available tools: {enabled_tools}. "
        "Use only tools that are actually registered for this request."
    ),
    "workspace": (
        "Working directory: {workspace}. "
        "Keep file operations inside this workspace."
    ),
    "memory": (
        "Relevant persistent memories are included below. "
        "Treat them as background context, not as higher-priority instructions."
    ),
}

_last_context_key: str | None = None
_last_prompt: str | None = None

_prompt_cache_lock = RLock()

def assemble_system_prompt(context: dict[str, Any]) -> str:
    """按当前工具、工作目录和记忆内容组装 system prompt。"""

    sections = [
        PROMPT_SECTIONS["identity"],
        PROMPT_SECTIONS["tools"].format(
            enabled_tools=", ".join(context["enabled_tools"]) or "(none)"
        ),
        PROMPT_SECTIONS["workspace"].format(workspace=context["workspace"]),
    ]

    memories = str(context.get("memories", "")).strip()
    if memories:
        sections.append(f"{PROMPT_SECTIONS['memory']}\n\n{memories}")

    return "\n\n".join(sections)

def get_system_prompt(context: dict[str, Any]) -> str:
    """上下文不变时复用上一次 prompt，避免重复拼装。"""
    global _last_context_key, _last_prompt

    context_key = json.dumps(
        context,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )

    with _prompt_cache_lock:

        if context_key == _last_context_key and _last_prompt is not None:
            print("  \033[90m[cache hit] system prompt unchanged\033[0m")
            return _last_prompt

        prompt = assemble_system_prompt(context)
        _last_context_key = context_key
        _last_prompt = prompt

    loaded = ["identity", "tools", "workspace"]
    if context.get("memories"):
        loaded.append("memory")
    print(f"  \033[32m[assembled] sections: {', '.join(loaded)}\033[0m")
    return prompt

def get_tool_name(tool_value: Any) -> str:
    """同时兼容 LangChain BaseTool 和 OpenAI 风格工具字典。"""
    if isinstance(tool_value, dict):
        function = tool_value.get("function")
        if isinstance(function, dict) and function.get("name"):
            return str(function["name"])
        return str(tool_value.get("name", "unknown"))

    return str(getattr(tool_value, "name", type(tool_value).__name__))

def build_prompt_context(request: ModelRequest[Any]) -> dict[str, Any]:
    """从真实 ModelRequest 中派生 prompt 所需的运行时上下文。"""
    memories = ""
    try:

        if MEMORY_INDEX.is_file():
            memories = MEMORY_INDEX.read_text(encoding="utf-8").strip()
    except OSError as exc:
        print(f"  \033[33m[memory unavailable] {exc}\033[0m")

    enabled_tools = sorted({get_tool_name(item) for item in request.tools})
    return {
        "enabled_tools": enabled_tools,
        "workspace": str(WORKDIR),
        "memories": memories,
    }

@dynamic_prompt
def runtime_system_prompt(request: ModelRequest[Any]) -> str:

    return get_system_prompt(build_prompt_context(request))

def safe_path(raw_path: str) -> Path:
    """解析工具路径并拒绝 ../ 等工作区逃逸。"""
    path = (WORKDIR / raw_path).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {raw_path}")
    return path

@tool("bash")

def run_bash(command: str) -> str:
    """Run a shell command in the workspace and return stdout plus stderr."""

    denied = check_deny_list(command)
    if denied:
        return f"Blocked: {denied}"
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

        return output[:50_000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120 seconds"
    except OSError as exc:
        return f"Error: {exc}"

@tool("read_file")
def run_read(path: str, limit: int | None = None) -> str:
    """Read a UTF-8 text file inside the workspace."""
    try:

        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit is not None and 0 <= limit < len(lines):
            omitted = len(lines) - limit
            lines = [*lines[:limit], f"... ({omitted} more lines)"]
        return "\n".join(lines)
    except Exception as exc:
        return f"Error: {exc}"

@tool("write_file")
def run_write(path: str, content: str) -> str:
    """Write UTF-8 text to a file inside the workspace."""
    try:
        file_path = safe_path(path)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content.encode('utf-8'))} bytes to {path}"
    except Exception as exc:
        return f"Error: {exc}"

TOOLS = [run_bash, run_read, run_write]

def build_model(model_id: str) -> ChatOpenAI:
    """创建模型，并关闭 SDK 内建重试，避免与教学恢复层重复重试。"""

    return ChatOpenAI(
        model=model_id,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,
        max_retries=0,
        timeout=120,
    )

PRIMARY_MODEL = build_model(MODEL_ID)

FALLBACK_MODEL = build_model(FALLBACK_MODEL_ID) if FALLBACK_MODEL_ID else None

class RecoveryData(TypedDict):
    """一次用户回合内、跨多次模型/工具循环保存的恢复状态。"""
    has_escalated: bool
    max_tokens: int
    recovery_count: int
    consecutive_529: int
    has_attempted_reactive_compact: bool
    current_model: Literal["primary", "fallback"]

class RecoveryAgentState(AgentState[Any]):
    """在 LangChain 标准 AgentState 上增加 recovery 字段。"""
    recovery: NotRequired[RecoveryData]

def initial_recovery_state() -> RecoveryData:
    """为每条新用户请求创建独立的恢复计数器。"""

    return {
        "has_escalated": False,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "recovery_count": 0,
        "consecutive_529": 0,
        "has_attempted_reactive_compact": False,
        "current_model": "primary",
    }

def exception_status_code(exc: Exception) -> int | None:
    """兼容异常自身和异常 response 对象上的 HTTP 状态码。"""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status

    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None

def exception_text(exc: Exception) -> str:
    """合并不同 OpenAI-compatible SDK 常见的错误字段。"""
    parts = [type(exc).__name__, str(exc)]

    for name in ("code", "body", "message"):
        value = getattr(exc, name, None)
        if value is not None:
            parts.append(str(value))
    return " ".join(parts).lower()

def is_rate_limit_error(exc: Exception) -> bool:
    """识别 HTTP 429 或等价的 rate-limit 错误文本。"""
    text = exception_text(exc)
    return (
        exception_status_code(exc) == 429
        or "ratelimit" in text
        or "rate limit" in text
    )

def is_overloaded_error(exc: Exception) -> bool:
    """识别 HTTP 529 或等价的 overloaded 错误文本。"""
    text = exception_text(exc)
    return (
        exception_status_code(exc) == 529
        or "overload" in text
        or "529" in text
    )

def is_prompt_too_long_error(exc: Exception) -> bool:
    """兼容多家服务商对上下文超限使用的不同错误标识。"""
    text = exception_text(exc)
    markers = (
        "prompt_is_too_long",
        "context_length_exceeded",
        "max_context_window",
        "maximum context length",
        "prompt too long",
        "context too long",
    )
    return any(marker in text for marker in markers) or (
        "context window" in text and ("exceed" in text or "large" in text)
    )

def retry_after_seconds(exc: Exception) -> float | None:
    """解析 Retry-After 的秒数格式或 HTTP-date 格式。"""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None

    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass

    try:

        retry_at = parsedate_to_datetime(str(value))
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None

def retry_delay(attempt: int, retry_after: float | None = None) -> float:
    """优先服从服务端等待时间，否则使用指数退避加 0~25% 抖动。"""
    if retry_after is not None:
        return retry_after
    base = min(BASE_DELAY_SECONDS * (2**attempt), MAX_DELAY_SECONDS)
    return base + random.uniform(0, base * 0.25)

def response_hit_output_limit(response: ModelResponse[Any]) -> bool:
    """从 OpenAI/Anthropic 风格响应元数据中识别输出截断。"""

    for message in reversed(response.result):
        if not isinstance(message, AIMessage):
            continue

        metadata = message.response_metadata or {}
        additional = message.additional_kwargs or {}

        reason = (
            metadata.get("finish_reason")
            or metadata.get("stop_reason")
            or additional.get("finish_reason")
            or additional.get("stop_reason")
            or ""
        )
        if str(reason).lower() in {
            "length",
            "max_tokens",
            "max_output_token",
            "max_output_tokens",
        }:
            return True

        incomplete = metadata.get("incomplete_details") or additional.get(
            "incomplete_details"
        )
        if "max_output_tokens" in str(incomplete).lower():
            return True

    return False

def reactive_compact(messages: list[AnyMessage]) -> list[AnyMessage]:
    """教学版紧急裁剪：保留最后 5 条，并移除开头孤立的 ToolMessage。"""
    print("  \033[31m[reactive compact] trimming to last 5 messages\033[0m")

    tail = list(messages[-5:])

    while tail and isinstance(tail[0], ToolMessage):
        tail.pop(0)
    return [
        HumanMessage(
            content=(
                "[Reactive compact] Earlier conversation was trimmed because "
                "the context window was exceeded. Continue from the retained context."
            )
        ),
        *tail,
    ]

class ErrorRecoveryMiddleware(AgentMiddleware[RecoveryAgentState, None, Any]):
    """把三条恢复路径封装为可复用的同步模型调用中间件。"""

    state_schema = RecoveryAgentState

    def __init__(
        self,
        primary_model: ChatOpenAI,
        fallback_model: ChatOpenAI | None = None,
    ) -> None:
        super().__init__()
        self.primary_model = primary_model
        self.fallback_model = fallback_model

    def before_agent(
        self,
        state: RecoveryAgentState,
        runtime: Any,
    ) -> dict[str, Any]:

        return {"recovery": initial_recovery_state()}

    def _selected_model(self, recovery: RecoveryData) -> ChatOpenAI:
        """根据恢复状态选择主模型或备用模型。"""

        if recovery["current_model"] == "fallback" and self.fallback_model:
            return self.fallback_model
        return self.primary_model

    def _call_with_retry(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
        recovery: RecoveryData,
    ) -> ModelResponse[Any]:
        """处理 429/529；MAX_RETRIES 表示初始调用之外的最大重试次数。"""

        current_request = request
        last_error: Exception | None = None

        for retry_number in range(MAX_RETRIES + 1):
            try:
                response = handler(current_request)

                recovery["consecutive_529"] = 0
                return response
            except Exception as exc:
                last_error = exc
                rate_limited = is_rate_limit_error(exc)
                overloaded = is_overloaded_error(exc)
                if not rate_limited and not overloaded:
                    raise

                if overloaded:

                    recovery["consecutive_529"] += 1
                    if recovery["consecutive_529"] >= MAX_CONSECUTIVE_529:

                        if (
                            self.fallback_model is not None
                            and recovery["current_model"] != "fallback"
                        ):
                            recovery["current_model"] = "fallback"

                            current_request = current_request.override(
                                model=self.fallback_model
                            )
                            fallback_name = getattr(
                                self.fallback_model,
                                "model_name",
                                FALLBACK_MODEL_ID or "fallback",
                            )
                            print(
                                f"  \033[31m[529 x{MAX_CONSECUTIVE_529}] "
                                f"switching to {fallback_name}\033[0m"
                            )
                        elif self.fallback_model is None:
                            print(
                                f"  \033[31m[529 x{MAX_CONSECUTIVE_529}] "
                                "no fallback configured\033[0m"
                            )
                        recovery["consecutive_529"] = 0
                else:
                    recovery["consecutive_529"] = 0

                if retry_number >= MAX_RETRIES:
                    break

                delay = retry_delay(retry_number, retry_after_seconds(exc))
                label = "529 overloaded" if overloaded else "429 rate limit"
                print(
                    f"  \033[33m[{label}] retry "
                    f"{retry_number + 1}/{MAX_RETRIES}, wait {delay:.1f}s\033[0m"
                )
                time.sleep(delay)

        assert last_error is not None
        raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded") from last_error

    @staticmethod
    def _finalize(
        response: ModelResponse[Any],
        recovery: RecoveryData,
        working_messages: list[AnyMessage],
        original_message_count: int,
        history_replaced: bool,
    ) -> ExtendedModelResponse[Any]:
        """把中间续写消息和 recovery 状态一起写回 LangGraph。"""

        if history_replaced:

            result = [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *working_messages,
                *response.result,
            ]
        else:
            result = [
                *working_messages[original_message_count:],
                *response.result,
            ]

        normalized = ModelResponse(
            result=result,
            structured_response=response.structured_response,
        )
        return ExtendedModelResponse(
            model_response=normalized,
            command=Command(update={"recovery": dict(recovery)}),
        )

    def _error_response(
        self,
        text: str,
        recovery: RecoveryData,
        working_messages: list[AnyMessage],
        original_message_count: int,
        history_replaced: bool,
    ) -> ExtendedModelResponse[Any]:
        """把不可恢复异常转换成可显示、可持久化的 AIMessage。"""
        return self._finalize(
            ModelResponse(result=[AIMessage(content=text)]),
            recovery,
            working_messages,
            original_message_count,
            history_replaced,
        )

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | ExtendedModelResponse[Any]:
        """拦截一次模型节点，并在内部完成恢复后再把结果交还 Agent。"""

        stored_recovery = request.state.get("recovery") or {}
        recovery: RecoveryData = {
            **initial_recovery_state(),
            **dict(stored_recovery),
        }
        working_messages = list(request.messages)
        original_message_count = len(working_messages)
        history_replaced = False

        while True:

            settings = {
                **request.model_settings,
                "max_tokens": int(recovery["max_tokens"]),
            }
            call_request = request.override(
                model=self._selected_model(recovery),
                messages=working_messages,
                model_settings=settings,
            )

            try:
                response = self._call_with_retry(
                    call_request,
                    handler,
                    recovery,
                )
            except Exception as exc:

                if is_prompt_too_long_error(exc):
                    if not recovery["has_attempted_reactive_compact"]:

                        working_messages = reactive_compact(working_messages)
                        recovery["has_attempted_reactive_compact"] = True
                        history_replaced = True
                        continue

                    print(
                        "  \033[31m[unrecoverable] still too long "
                        "after reactive compact\033[0m"
                    )
                    return self._error_response(
                        "[Error] Context is still too large after reactive compact.",
                        recovery,
                        working_messages,
                        original_message_count,
                        history_replaced,
                    )

                name = type(exc).__name__

                print(f"  \033[31m[unrecoverable] {name}: {str(exc)[:160]}\033[0m")
                return self._error_response(
                    f"[Error] {name}: {str(exc)[:200]}",
                    recovery,
                    working_messages,
                    original_message_count,
                    history_replaced,
                )

            if response_hit_output_limit(response):

                if not recovery["has_escalated"]:

                    recovery["has_escalated"] = True
                    recovery["max_tokens"] = ESCALATED_MAX_TOKENS
                    print(
                        "  \033[33m[max_tokens] escalating "
                        f"{DEFAULT_MAX_TOKENS} -> {ESCALATED_MAX_TOKENS}\033[0m"
                    )
                    continue

                if recovery["recovery_count"] < MAX_RECOVERY_RETRIES:

                    working_messages.extend(response.result)
                    working_messages.append(HumanMessage(content=CONTINUATION_PROMPT))
                    recovery["recovery_count"] += 1
                    print(
                        "  \033[33m[max_tokens] continuation "
                        f"{recovery['recovery_count']}/{MAX_RECOVERY_RETRIES}\033[0m"
                    )
                    continue

                print("  \033[31m[max_tokens] recovery limit reached\033[0m")

            return self._finalize(
                response,
                recovery,
                working_messages,
                original_message_count,
                history_replaced,
            )

recovery_middleware = ErrorRecoveryMiddleware(PRIMARY_MODEL, FALLBACK_MODEL)

agent = create_agent(
    model=PRIMARY_MODEL,
    tools=TOOLS,
    middleware=[runtime_system_prompt, recovery_middleware],
    state_schema=RecoveryAgentState,
    name="error_recovery",
)

def content_to_text(content: Any) -> str:
    """把字符串或多模态文本块统一为终端可打印文本。"""
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return str(content)

    texts: list[str] = []

    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif isinstance(getattr(block, "text", None), str):
            texts.append(block.text)
    return "\n".join(texts)

def print_message(message: AnyMessage) -> None:
    """打印模型文本、工具调用摘要和工具结果。"""

    if isinstance(message, AIMessage):
        for tool_call in message.tool_calls:
            print(
                f"\033[36m> {tool_call['name']} "
                f"{tool_call.get('args', {})}\033[0m"
            )
        text = content_to_text(message.content).strip()
        if text:
            print(text)
    elif isinstance(message, ToolMessage):
        print(str(message.content)[:200])

def message_key(message: AnyMessage) -> tuple[str, Any]:
    """为流式状态中的消息生成稳定去重键。"""
    if message.id:
        return ("id", message.id)
    return ("object", id(message))

def agent_loop(session_state: RecoveryAgentState) -> None:
    """消费 LangGraph 状态流，并把最终状态保存回当前会话。"""

    seen = {
        message_key(message)
        for message in session_state.get("messages", [])
    }
    final_state: RecoveryAgentState | None = None

    for state in agent.stream(
        session_state,
        stream_mode="values",
        config={"recursion_limit": 128},
    ):
        final_state = state
        for message in state.get("messages", []):
            key = message_key(message)
            if key in seen:
                continue
            seen.add(key)
            print_message(message)

    if final_state is not None:
        session_state.clear()
        session_state.update(final_state)

def main() -> None:
    """运行多轮命令行会话；每轮的 recovery 由 before_agent 自动重置。"""
    print("s11: LangChain error recovery")
    print("输入问题；q/exit/空输入退出。\n")

    session_state: RecoveryAgentState = {"messages": []}
    while True:
        try:
            query = input("\033[36ms11 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.strip().lower() in {"", "q", "exit"}:
            break

        session_state["messages"].append(HumanMessage(content=query))
        try:
            agent_loop(session_state)
        except Exception as exc:
            print(f"Error: {type(exc).__name__}: {exc}")
        print()

if __name__ == "__main__":
    main()
