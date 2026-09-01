"""s08：上下文工程与自动压缩。

长时间运行的 Agent 会不断累积用户消息、模型回复和工具结果。若不治理，上下文会变贵、
变慢，最终超过模型窗口。本章在 s07 基础上加入三层互补策略：
1. 每次调用模型前做无损/低损整理：大工具输出落盘、中段裁剪、旧工具结果占位；
2. 估算 token 达到阈值时，保存完整 transcript，并让模型把历史压成可继续工作的摘要；
3. 若服务端仍返回 prompt-too-long，截获异常并执行一次应急压缩后重试。

此外还提供 compact 工具，让模型可以主动压缩。所有压缩都特别维护 AIMessage.tool_calls 与
ToolMessage 的配对关系，因为破坏这套协议会在到达模型前就触发消息校验错误。后半部分沿用
s07 的 Skills、权限 Hook、父/子 Agent 和 CLI，展示压缩中间件如何接入完整运行时。"""

from __future__ import annotations
import json
import re
import time
from typing import NotRequired

import glob
import os
import subprocess
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any


from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import(
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    AgentState,
    TodoListMiddleware,
    after_agent,
    before_agent,
    wrap_tool_call
)
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import(
    HumanMessage,
    AIMessage, 
    ToolMessage,
    AnyMessage,
    RemoveMessage,
    message_to_dict,
    SystemMessage,
) 
from langchain_core.messages.utils import(
    count_tokens_approximately,
    get_buffer_string,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.runtime import Runtime
from langgraph.types import Command

import yaml


# ------------------------------------------------------------------
# 章节说明：1. 压缩预算与持久化目录
# 主动阈值 = 上下文窗口 - 最大输出 - 安全缓冲，必须给下一次回复预留空间。
# ------------------------------------------------------------------
load_dotenv(override=True)

WORKDIR = Path.cwd().resolve()
TRANSCRIPT_DIR = WORKDIR / ".transcripts"

TOOL_RESULTS_DIR = (
    WORKDIR / ".task_outputs" / "tool-results"
)
KEEP_RECENT_TOOL_RESULTS = 3
TOOL_RESULT_BUDGET_BYTES =200_000
PERSIST_THRESHOLD_BYTES = 30_000

CONTEXT_WINDOW_TOKENS = int(
    os.getenv("CONTEXT_WINDOW_TOKENS", "128000")
)
MAX_OUTPUT_TOKENS = 8000
AUTOCOMPACT_BUFFER_TOKENS = 13_000

# 例如 128k 窗口减去 8k 输出和 13k 缓冲，约在 107k token 时主动压缩。
AUTO_COMPACT_TOKENS = (
    CONTEXT_WINDOW_TOKENS
    - MAX_OUTPUT_TOKENS
    - AUTOCOMPACT_BUFFER_TOKENS
)

MAX_COMPACT_FAILURES = 3

SKILL_DIR = WORKDIR/"skills"

SKILL_REGISTRY: dict[str, dict[str, str]] = {}


# ------------------------------------------------------------------
# 章节说明：2. 第一层：每次模型调用前的便宜压缩
# 三步固定顺序为 tool_result_budget → snip_compact → micro_compact。
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 函数 _safe_tail_start
# 修正尾部切片起点，避免只保留 ToolMessage 却丢掉发起调用的 AIMessage。
# 工具协议是一组 AI(tool_calls) + 若干 ToolMessage；若 start 落在结果组中间，就回退到 AI。
# 如果向前找不到带 tool_calls 的 AIMessage，则保留原起点，避免误删更多无关消息。
# ------------------------------------------------------------------
def _safe_tail_start(
        messages:list[AnyMessage],
        start: int,
) -> int :
    """避免切开一组toolmessage"""
    if start >= len(messages):
        return start
    
    if not isinstance(messages[start],ToolMessage):
        return start
    

    original = start
    

    # 连续向前越过 ToolMessage，尝试找到这一批结果对应的 AIMessage。
    while start > 0 and isinstance(messages[start], ToolMessage):
        start -=1

    if isinstance(messages[start], AIMessage):
        if messages[start].tool_calls:
            return start
        
    return original

# ------------------------------------------------------------------
# 函数 snip_compact
# 在消息很多时保留开头三条和最近一段，用一条 HumanMessage 标记被裁掉的中段。
# 头尾边界都会照顾工具调用组；若两段重叠则放弃裁剪，优先保证消息协议合法。
# 这是一种便宜但有损的压缩，适合先清掉久远的过程性对话。
# ------------------------------------------------------------------
def snip_compact(
    messages: list[AnyMessage],
    max_messages: int = 50,
) -> list[AnyMessage]:
    """裁掉无关的旧对话"""
    if len(messages) <= max_messages:
        return list(messages)

    # 固定保留前三条通常能留下最初目标；其余配额优先给最近消息。
    head_end = 3
    tail_start = len(messages) - (max_messages - 3)

    if (
        head_end > 0
        and isinstance(
            messages[head_end - 1],
            AIMessage,
        )
        and messages[head_end - 1].tool_calls
    ):
        while (
            head_end < len(messages)
            and isinstance(
                messages[head_end],
                ToolMessage,
            )
        ):
            head_end += 1

    # 无论头部是否遇到工具组，尾部边界都必须单独修正。
    # 注意：这部分必须在上面的 if 外面。
    tail_start = _safe_tail_start(
        messages,
        tail_start,
    )

    if head_end >= tail_start:
        return list(messages)

    snipped = tail_start - head_end

    return [
        *messages[:head_end],
        HumanMessage(
            content=(
                f"[snipped {snipped} messages "
                "from conversation middle]"
            )
        ),
        *messages[tail_start:],
    ]

# ------------------------------------------------------------------
# 函数 _message_content_text
# 把 ToolMessage.content 稳定序列化为文本，供字节计数、预览和落盘共同使用。
# 非字符串内容用 JSON 表示，ensure_ascii=False 保留中文可读性。
# ------------------------------------------------------------------
def _message_content_text(message: AnyMessage) -> str:
    # 字节预算、预览和落盘必须使用同一种序列化，否则阈值会前后不一致。
    content = message.content
    if isinstance(content,str):
        return content
    
    return json.dumps(
        content,
        ensure_ascii=False,
        default= str,
    )

# ------------------------------------------------------------------
# 函数 micro_compact
# 保留最近 KEEP_RECENT_TOOL_RESULTS 条工具结果，把更旧且较长的结果替换为提示占位符。
# 使用 model_copy 而非原地改对象，并清空 artifact，避免隐藏的大对象继续占内存。
# response_metadata 标记 context_compacted，便于调试或后续中间件识别。
# ------------------------------------------------------------------
def micro_compact(
        messages: list[AnyMessage],
)-> list[AnyMessage]:
    """旧工具结果占位"""
    result = list(messages)

    # 先收集索引而不是直接删除列表元素，避免删除一个元素后其他索引整体左移。
    tool_result_indexes = [
        index
        for index, message in enumerate(result)
        if isinstance (message,ToolMessage)
    ]

    # 切片为负或结果少于保留数时自然得到空列表，不需要额外分支。
    old_indexes = tool_result_indexes[:-KEEP_RECENT_TOOL_RESULTS]


    # 只处理旧结果；最近 KEEP_RECENT_TOOL_RESULTS 条保留原文以保证短期可追溯。
    for index in old_indexes:
        message = result[index]
        content = _message_content_text(message)

        # 很短的旧结果本就占用很少，保留原文通常比占位符更有价值。
        if len(content) <=120:
            continue

        result[index] = message.model_copy(
            update={
                "content": (
                    "[Earlier tool result compacted. "
                    "Re-run the tool if needed.]"
                ),
                "artifact": None,
                "response_metadata": {
                    **message.response_metadata,
                    "context_compacted": True,
                },
            }
        )
    return result
    

# ------------------------------------------------------------------
# 函数 _content_bytes
# 按 UTF-8 编码后的字节数计算工具结果体积；字节预算比 Python 字符数更接近存储成本。
# ------------------------------------------------------------------
def _content_bytes(message: ToolMessage) -> int:
    return len(
        _message_content_text(message).encode("utf_8")
    )

# ------------------------------------------------------------------
# 章节说明：2.1 超大工具输出落盘：保留可恢复性，而不是直接丢弃
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 函数 persist_large_output
# 把单条超大工具结果保存到磁盘，上下文中只留下路径、标签和 2000 字符预览。
# tool_call_id 先过滤为安全文件名；同一 id 重复出现时始终覆盖写，以最新结果为准（幂等）。
# 返回的 XML 风格标记会明确告诉模型：完整内容仍可通过文件工具重新读取。
# ------------------------------------------------------------------
def persist_large_output(
        tool_call_id: str,
        output: str
)->str:
    """
    模型一次读了 5 个大文件，单条 user 消息里所有 tool_result 加起来 500KB。

    统计最后一条 user 消息里所有 tool_result 的总大小。
    超过 200KB → 按大小排序，从最大的开始落盘到/".task_outputs"/"tool-results",
    上下文里只留 <persisted-output> 标记 + 前 2000 字符预览。
    模型看到标记后知道完整内容在磁盘上，需要时可以重新读。
"""

    # mkdir(exist_ok=True) 让首次调用创建目录，之后重复调用保持幂等。
    TOOL_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    # tool_call_id 来自模型，不能直接当作文件名；过滤斜杠和特殊字符防止路径逃逸。
    safe_id = re.sub(
        r"[^a-zA-Z0-9_.-]",
        "_",
        tool_call_id or "unknown",
    )

    path = TOOL_RESULTS_DIR / f"{safe_id}.txt"

    # 总是覆盖写，保证返回给模型的预览与落盘内容一致（同一 tool_call_id 的多次结果以最新为准）。
    path.write_text(
        output,
        encoding="utf_8",
    )

    relative_path = path.relative_to(WORKDIR).as_posix()

    return (
        "<persisted-output>\n"
        f"Full output: {relative_path}\n"
        f"Preview:\n{output[:2000]}\n"
        "</persisted-output>"
    )

# ------------------------------------------------------------------
# 函数 tool_result_budget
# 只统计消息尾部连续的一组 ToolMessage；当总字节超预算时，优先落盘最大的结果。
# 小于单条持久化阈值的结果保持原样，因此总量可能无法降到目标值，这是刻意的保守策略。
# 它必须在 micro_compact 前运行，否则原始大内容被占位后就失去可恢复的落盘机会。
# ------------------------------------------------------------------
def tool_result_budget(
        messages: list[AnyMessage],
        max_bytes:int = TOOL_RESULT_BUDGET_BYTES,
) -> list[AnyMessage]:
    # 复制列表后只替换消息对象，调用方的原始 state 不会被原地修改。
    result = list(messages)
    # 从末尾回溯连续 ToolMessage，当前工具批次通常正是这段消息。
    start = len(result)

    # 只处理末尾连续结果组，因为它通常对应刚完成的一次并行工具调用。
    while(
        start>0
        and isinstance(result[start-1],ToolMessage)
    ):
        start-=1

    indexes = [
        index
        for index in range(start, len(result))
        if isinstance(result[index], ToolMessage)
    ]

    total = sum(
        _content_bytes(result[index])
        for index in indexes
    )

    if total <= max_bytes:
        return result
    

    # 从最大项开始替换，通常用最少文件操作就能让总量快速下降。
    # 按体积从大到小处理，通常最少的落盘操作就能显著降低上下文体积。
    ranked_indexes = sorted(
        indexes,
        key = lambda index: _content_bytes(result[index]),
        reverse=True
    )

    for index in ranked_indexes:
        if total <= max_bytes:
            break
        message = result[index]
        output = _message_content_text(message)

        # 低于 30KB 的单项不落盘，避免生成大量小文件；因此预算是软上限。
        if len(output.encode("utf_8")) <= PERSIST_THRESHOLD_BYTES:
            continue

        marker = persist_large_output(
            message.tool_call_id,
            output,
        )

        result[index] = message.model_copy(
            update={
                "content": marker,
                "artifact": None,
            }
        )

        total = sum(
            _content_bytes(result[i])
            for i in indexes
        )

    return result

# 一定保持这个执行顺序：tool_result_budget -> snip_compact -> micro_compact。
# micro_compact 如果先运行，原始大输出会被占位符替换，后续就没有内容可落盘。


# ------------------------------------------------------------------
# 章节说明：3. 第二层：用模型摘要替换长历史
# ------------------------------------------------------------------
SUMMARY_SYSTEM_PROMPT = """
CRITICAL: Respond with TEXT ONLY.
Do not call tools.

Summarize the coding-agent conversation so that another agent can
continue the work without seeing the original history.

Preserve:

1. Current user objective
2. Important findings and decisions
3. Files read, created or modified
4. Completed work and verification results
5. Remaining work
6. Errors and unsuccessful approaches
7. User constraints and preferences

Be concise, but preserve concrete paths, commands, names and decisions.
""".strip()

# ------------------------------------------------------------------
# 函数 write_transcript
# 把压缩前的每条 LangChain 消息序列化成一行 JSONL，生成带纳秒时间戳的恢复记录。
# default=str 让少数非 JSON 原生字段仍可记录；transcript 不会自动重新注入模型上下文。
# ------------------------------------------------------------------
def write_transcript(
        messages: list[AnyMessage],
)->Path:
    # transcript 是压缩前的恢复副本；即使摘要模型失败，原始历史仍然可查。
    TRANSCRIPT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    path = TRANSCRIPT_DIR/(f"transcript_{time.time_ns()}.jsonl")

    # JSONL 一条消息一行，既能流式读取，也便于某行损坏时保留其他记录。
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        # 一行一条消息让文件可增量读取，也便于定位某条损坏记录。
        for message in messages:
            data = message_to_dict(message)

            file.write(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    default=str,
                )
                +"\n"
            )
    return path

# ------------------------------------------------------------------
# 函数 summarize_history
# 把消息格式化为 XML 风格对话，交给同一聊天模型生成可供后续 Agent 接手的工作摘要。
# 可选 focus 告诉摘要器重点保留什么；空摘要被视为失败，防止用空消息替换完整历史。
# ------------------------------------------------------------------
def summarize_history(
        messages: list[AnyMessage],
        model: ChatOpenAI,
        focus: str =""
) -> str:
    if not messages:
        return "no messages"
    
    # get_buffer_string 把不同 BaseMessage 统一成可读的 role/content 文本。
    history = get_buffer_string(
        messages,
        format="xml",
    )

    # focus 为空时不向摘要器注入额外偏好，避免无意义的提示词变化。
    focus_instruction = (
        f"\nCompaction focus:{focus}\n"
        if focus.strip()
        else ""
    )

    # 摘要请求不挂载工具，并由 system prompt 强制纯文本，降低摘要器产生 tool_calls 的可能。
    response = model.invoke(
        [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=focus_instruction+"\nConversation to summarize:\n"+history),
        ]
    )

    # 摘要模型的 AIMessage 仍可能是多内容块，因此复用统一文本提取器。
    summary = content_to_text(response.content).strip()
    if not summary:
        raise ValueError("Summary model return empty content")
    
    return summary

# ------------------------------------------------------------------
# 函数 compact_history
# 完整压缩事务：先保存 transcript，再生成摘要，最后用一条带元数据的 HumanMessage 替换历史。
# 只有摘要成功后才返回新消息；调用者负责用 RemoveMessage(REMOVE_ALL_MESSAGES) 更新 state。
# ------------------------------------------------------------------
def compact_history(
        messages: list[AnyMessage],
        model: ChatOpenAI,
        label: str = "Compacted",
        focus: str =""
)->list[AnyMessage]:
    """
    保存 transcript：完整对话写入 .transcripts/，JSONL 格式。transcript 保留了可恢复记录，但模型的活跃上下文里只剩摘要。对模型当下推理来说，细节已经不在上下文中了。教学代码没有提供 transcript 检索工具。
    LLM 生成摘要：把对话历史发给 LLM，要求保留当前目标、重要发现、已改文件、剩余工作、用户约束等关键信息。
    替换消息列表：所有旧消息被替换为一条摘要。教学版只保留摘要；真实 Claude Code 会在 compact 后重新附加部分最近文件、计划、agent/skill/tool 等上下文。"""

    # 先保存再摘要：即使模型摘要失败，原始历史仍有 transcript 可供排查和恢复。
    # 先落盘再调用模型，保证摘要失败时至少有完整 transcript。
    transcript_path = write_transcript(messages)

    print(f"[Transcript saved: {transcript_path}]")

    # 摘要结果只需支持后续 Agent 继续工作，不追求逐字还原原始对话。
    summary = summarize_history(
        messages,
        model,
        focus,
    )

    # 压缩后的单条 HumanMessage 带 transcript 路径元数据，正文只放模型继续工作所需摘要。
    # 调用方会用 RemoveMessage + 该列表替换旧 state；这里只构造新消息。
    return [
        HumanMessage(
            content=f"[{label}]\n\n{summary}",
            additional_kwargs={
                "context_compacted": True,
                "transcript": str(transcript_path),
            }
        )
    ]

# ------------------------------------------------------------------
# 函数 is_prompt_too_long
# 把不同兼容 API 的上下文超限错误归一化：既检查 HTTP 413，也检查常见错误文本。
# 仅识别确定的长度错误；认证、网络等异常必须继续抛出，不能被错误地当成压缩问题。
# ------------------------------------------------------------------
def is_prompt_too_long(exc: Exception) -> bool:
    """判断是否提示词过长"""
    # 不同兼容端点可能只给文本、不提供 status_code，因此两类信号都检查。
    text = (f"{type(exc).__name__}:{exc}").lower()

    status_code = getattr(exc,"status_code",None)

    return (
        status_code == 413
        or "prompt_too_long" in text
        or "context_length_exceeded" in text
        or "too many tokens" in text
        or "maximum context length" in text
    )


# ------------------------------------------------------------------
# 函数 reactive_compact
# 服务端拒绝请求后的应急方案：摘要较旧消息，但原样保留最近五条附近的完整上下文。
# 尾部起点仍通过 _safe_tail_start 修正，确保工具调用与工具结果不会被拆开。
# ------------------------------------------------------------------
def reactive_compact(
        messages: list[AnyMessage],
        model: ChatOpenAI,
) -> list[AnyMessage]:
    # 应急路径仍先保存完整对话；这是一次服务端拒绝后的最后可追溯副本。
    transcript_path = write_transcript(messages)

    # 应急模式保留最近约五条，比主动全量摘要更重视触发错误前的局部工作现场。
    tail_start = max(0, len(messages)-5)
    tail_start = _safe_tail_start(messages,tail_start)

    # 旧段交给摘要模型，最近段原样保留以维持当前工具现场。
    old_messages = messages[:tail_start]
    recent_messages = messages[tail_start:]

    summary = summarize_history(old_messages,model)

    print(
        f"[reactive transcript saved: "
        f"{transcript_path}]"
    )

    return [
        HumanMessage(content = f"[Reactive] compact\n\n{summary}"),
        *recent_messages,
    ]

# ------------------------------------------------------------------
# 章节说明：4. 压缩中间件：主动阈值压缩 + 被动超限补救
# ------------------------------------------------------------------
#core part

# ------------------------------------------------------------------
# 类 CompactState
# 在标准 AgentState 上增加 compact_failures，可跨模型调用记录连续自动压缩失败次数。
# NotRequired 表示旧 state 没有该键也合法，读取时使用默认值 0。
# ------------------------------------------------------------------
class CompactState(AgentState):
    compact_failures: NotRequired[int]

# ------------------------------------------------------------------
# 类 ContentCompactionMiddleware
# 上下文治理核心中间件，同时实现调用前主动整理和异常后的应急重试。
# state_schema 告诉 LangGraph 合并自定义失败计数字段；中间件本身不保存会话外状态。
# ------------------------------------------------------------------
class ContentCompactionMiddleware(AgentMiddleware):
    state_schema = CompactState

    # ----------------------------------------------------------
    # 函数 ContentCompactionMiddleware.__init__
    # 注入摘要模型、完整 system prompt 和主动压缩阈值，便于复用与测试。
    # ----------------------------------------------------------
    def __init__(
            self,
            summary_model: ChatOpenAI,
            system_prompt: str,
            trigger_token: int,
        ) -> None:
        self.summary_model = summary_model
        self.system_prompt = system_prompt
        self.trigger_token =  trigger_token

    # ----------------------------------------------------------
    # 函数 ContentCompactionMiddleware._count_token
    # 估算 system prompt 加当前消息的总 token；阈值判断必须包含每次都会发送的系统提示。
    # ----------------------------------------------------------
    def _count_token(self, messages: list[AnyMessage]) -> int:
        return count_tokens_approximately(
            [SystemMessage(content=self.system_prompt),*messages]
        )

    # ----------------------------------------------------------
    # 函数 ContentCompactionMiddleware.before_model
    # 每次模型调用前依次执行：大输出预算 → 中段裁剪 → 旧结果占位 → token 阈值摘要。
    # 压缩失败只记录并放行，达到上限后不再反复调用摘要模型；上下文变小会把计数清零。
    # 消息改变时用 RemoveMessage 清空旧列表再整体写回，避免 LangGraph 默认追加语义造成重复。
    # ----------------------------------------------------------
    # before_model 会在每次工具循环后再次运行，因此低成本整理能持续控制上下文增长。
    def before_model(self, state:CompactState, runtime:Runtime) ->dict[str, Any] | None:
        # state["messages"] 是 LangGraph 当前完整历史；先复制，避免中间件直接改变 reducer 输入。
        orignal = list(state["messages"])

        # 顺序不可交换：先落盘原始大结果，再裁消息，最后才把旧结果替换为占位。
        messages = tool_result_budget(orignal)
        messages = snip_compact(messages)
        messages = micro_compact(messages)

        # compact_failures 是连续摘要失败计数；成功压缩或低于阈值时都会归零。
        failures = state.get("compact_failures",0)
        token_count = self._count_token(messages)

        # 只有超过阈值且失败次数未达上限才调用昂贵的摘要模型。
        if(token_count > self.trigger_token and failures < MAX_COMPACT_FAILURES):
            print(f"[auto compact: {token_count} tokens]")

            try:
                messages = compact_history(
                    messages,
                    self.summary_model,
                    label="Auto compact"
                )
                failures = 0

            # 摘要失败不立即终止主 Agent；递增计数可以防止每个工具循环都重复失败。
            except Exception as exc:
                failures += 1

                print(
                    "[auto compact failed "
                        f"{failures}/{MAX_COMPACT_FAILURES}: "
                        f"{exc}]"
                    )

        elif token_count <= self.trigger_token:
            failures = 0

        # 不要无条件返回 messages update：空 update 会让图产生额外、难以追踪的状态事件。
        update: dict[str, Any] = {}

        # 只在确有变化时返回 state update，减少无意义的图状态写入。
        if messages != orignal:
            update["messages"] = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]

        if failures != state.get("compact_failures",0):
            update["compact_failures"] = failures

        return update or None
    

    # ----------------------------------------------------------
    # 函数 ContentCompactionMiddleware.wrap_model_call
    # 包裹真实模型请求；普通异常原样抛出，只有上下文超限才执行一次 reactive_compact 重试。
    # ExtendedModelResponse 同时返回模型结果和 state 更新，使重试使用的压缩历史成为后续真实状态。
    # ----------------------------------------------------------
    # wrap_model_call 位于真正网络请求边界，因此能捕获服务端对 token 的最终判断。
    def wrap_model_call(self, request:ModelRequest, handler: Callable[[ModelRequest],ModelRequest]) -> ModelRequest | ExtendedModelResponse:
        # 这里位于真实 provider 请求边界，只有服务端最终拒绝后才能确认上下文确实过长。
        try:
            return handler(request)
        except Exception as exc:
            # 认证、网络、业务错误不能靠压缩解决，必须原样抛给上层处理。
            if not is_prompt_too_long(exc):
                raise

            print("[reactive compact]")

            # 重试只发生一次；若第二次仍失败，异常会自然向外传播，避免无限递归。
            # reactive_compact 返回“摘要 + 最近现场”；只重写本次 request，不修改原 request。
            compacted = reactive_compact(list(request.messages), self.summary_model)
            #应急压缩最多压缩一次
            response = handler(request.override(messages=compacted))

            # Command 把重试所用消息和本次模型响应一起写回，保持内存 state 与模型实际所见一致。
            return ExtendedModelResponse(
                model_response=response,
                command=Command(
                    update={
                        "messages":[
                            RemoveMessage(
                                id=REMOVE_ALL_MESSAGES
                            ),
                            *compacted,
                            *response.result
                        ],
                        "compact_failures": 0,
                    }
                )
            )




# ------------------------------------------------------------------
# 章节说明：5. 以下复用 s07：Skills 渐进加载
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 函数 _parse_frontmatter
# 解析 SKILL.md 顶部由 --- 包围的 YAML 元数据，并同时返回正文。
# 没有完整 frontmatter 或 YAML 格式错误时采用空字典，单个坏技能不会让扫描器崩溃。
# 正文与原始文本分开返回，目录阶段只取元数据，真正 load_skill 时仍能交付完整文件。
# ------------------------------------------------------------------
def _parse_frontmatter(raw: str) -> tuple[dict[str,Any], str]:
    """解析skill.md开头的yaml frontmatter"""
    # s08 沿用 s07 的渐进式技能加载：启动时只解析轻量元数据。
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    
    try:
        # start=1 让找到的闭合 --- 索引仍对应原始 lines。
        end_index = next(
            index
            for index, lines in enumerate(lines[1:], start=1)
            if lines.strip() == "---"
        )
    except StopIteration:
        return {}, raw
    

    yaml_text = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index+1:])

    try: 
        # safe_load 避免 YAML 构造任意 Python 对象。
        metadata = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError:
        metadata = {}

    if not isinstance(metadata, dict):
           metadata = {} 

    return metadata, body
    

# ------------------------------------------------------------------
# 函数 _scan_skills
# 扫描 skills/*/SKILL.md，把技能名称、简介、完整内容和根目录登记到内存注册表。
# 这里故意只做本地文件发现，不调用模型；重复名称直接报错，避免模型加载到含糊的技能。
# description 被压成单行以节省 system prompt，content 则原样保留，供按需加载使用。
# ------------------------------------------------------------------
def _scan_skills() -> None:
    """启动的时候自动扫描skills"""
    # skills 目录不存在不是错误，catalog 会退化为明确的空目录提示。
    if not SKILL_DIR.exists():
        return
    
    # 排序确保提示词顺序和 token 估算稳定。
    for manifest in sorted(SKILL_DIR.glob("*/SKILL.md")):
        raw = manifest.read_text(
            encoding="utf-8",
            errors="replace",
        )


        metadata, body = _parse_frontmatter(raw)


        name = str(
            metadata.get("name")
            or manifest.parent.name
        )

        description = metadata.get("description")

        if not description:
            description = next(
                (
                    line.lstrip("#").strip()
                    for line in body.splitlines()
                    if line.lstrip().startswith("#")
                ),
                name,
            )
            # YAML 的 description 可能使用 | 写成多行，
            # catalog 中应压缩为一行，避免浪费上下文。
        description = " ".join(
            str(description).split()
        )

        # 重名不能静默覆盖，否则 load_skill 的实际内容不确定。
        if name in SKILL_REGISTRY:
            raise ValueError(
                f"Duplicate skill name: {name}"
            )
        
        skill_root = manifest.parent.relative_to(
            WORKDIR
        ).as_posix()


        SKILL_REGISTRY[name] = {
            "name": name,
            "description": description,
            "content": raw,
            "root": skill_root,
        }

# ------------------------------------------------------------------
# 函数 list_skills
# 把注册表压缩成“名称 + 一行简介”的目录文本，供 system prompt 做能力发现。

# ------------------------------------------------------------------
def list_skills() ->str:
    """生成注入system_prompt的轻量技能目录"""
    if not SKILL_REGISTRY:
        return "- (no skills found)"
    
    return "\n".join(
        f"- {skill['name']}: {skill['description']}"
        for skill in SKILL_REGISTRY.values()
    )



_scan_skills()

SKILL_CATALOG = list_skills()

MODEL_ID = os.getenv("MODEL_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("BASE_URL")


SKILL_SYSTEM_PROMPT = f"""
Available skills:

{SKILL_CATALOG}

Skills contain specialized instructions that should be loaded only when
relevant.

When a request clearly matches a skill description:

1. Call load_skill using the exact skill name.
2. Read and follow the returned instructions before doing the task.
3. Do not guess a skill's full instructions from its description.
4. Load only skills relevant to the current request.
"""

# ------------------------------------------------------------------
# 章节说明：6. 以下复用 s07：生命周期 Hook 与父/子作用域
# ------------------------------------------------------------------
#用来区分hook日志来自子agent还是父agent
AGENT_SCOPE: ContextVar[str] = ContextVar(
    "agent_scope",
    default="parent"
)


HOOKS: dict[str, list[Callable[..., Any]]] = {
    "UserPromptSubmit": [],
    "PreToolUse" : [],
    "PostToolUse": [],
    "Stop": [],
}

# ------------------------------------------------------------------
# 函数 register_hook
# 向指定生命周期事件注册回调。先校验事件名，可尽早发现拼写错误。
# 同一事件允许多个回调，保存顺序就是稍后触发顺序。
# ------------------------------------------------------------------
def register_hook(event: str,callback: Callable[..., Any]) -> None:
    """注册一个hook"""
    # 事件名先校验，避免回调被注册到永远不会触发的拼写错误键。

    if event not in HOOKS:
        raise ValueError(f"unknown hook event:{event}")
    
    HOOKS[event].append(callback)


# ------------------------------------------------------------------
# 函数 trigger_hook
# 依注册顺序执行回调，并返回第一个非 None 结果。
# PreToolUse 把非 None 当作拦截原因；纯日志 Hook 通常返回 None，让后续回调继续运行。
# ------------------------------------------------------------------
def trigger_hook(event:str, *args: Any)-> Any|None:
    """按照顺序执行hook"""
    # 第一个非 None 返回值具有短路语义，PreToolUse 用它表达拒绝原因。
    for callable in HOOKS.get(event,[]):
        result = callable(*args)

        if result is not None:
            return result
    return None

# ------------------------------------------------------------------
# 函数 user_prompt_submit
# 把 LangChain 的 before_agent 生命周期适配成自定义 UserPromptSubmit 事件。
# 消息可能是普通 dict，也可能是 LangChain 消息对象，因此分别读取 content。
# ------------------------------------------------------------------
@before_agent
def user_prompt_submit(state:AgentState, runtime:Runtime) -> dict[str, Any] |None:
    """父agent运行的时候触发"""

    messages = state.get("messages", [])

    if not messages:
        return None
    
    last_messages = messages[-1]

    if isinstance(last_messages, dict):
        content = last_messages.get("content")

    else:
        content = getattr(last_messages,"content",None)
    
    trigger_hook("UserPromptSubmit", content)

    return None


# ------------------------------------------------------------------
# 函数 tool_hook
# 包裹每次工具调用：先运行 PreToolUse，获准后调用真实 handler，最后运行 PostToolUse。
# 被拒绝时返回带原 tool_call_id 的错误 ToolMessage，保证模型侧工具调用协议仍然闭合。
# ------------------------------------------------------------------
@wrap_tool_call
def tool_hook(
    request: ToolCallRequest,
    handler: Callable[
        [ToolCallRequest],
        ToolMessage | Command,
    ],
) -> ToolMessage | Command:
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})
    tool_call_id = request.tool_call["id"]

    blocked_reason = trigger_hook(
        "PreToolUse",
        tool_name,
        tool_args,
    )

    if blocked_reason:
        return ToolMessage(
            content=str(blocked_reason),
            tool_call_id = tool_call_id,
            name = tool_name,
            status = "error",
        )
    
    result = handler(request)

    trigger_hook(
        "PostToolUse",
        tool_name,
        tool_args,
        result,
    )

    return result


# ------------------------------------------------------------------
# 函数 stop_hook
# 在一次 Agent invoke/stream 完成后触发 Stop 回调，用于统计或清理。
# 返回 None 表示不修改 LangGraph state。
# ------------------------------------------------------------------
@after_agent
def stop_hook(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    """父agent结束时触发"""
    trigger_hook(
        "Stop",
        state.get("messages", [])
    )

    return None

# ------------------------------------------------------------------
# 章节说明：7. 以下复用 s07：权限规则
# ------------------------------------------------------------------
from harness.security import check_deny_list

POTENTIALLY_DESTRUCTIVE_COMMANDS = [
    "rm ",
    "> /etc/",
    "chmod 777",
]

# ------------------------------------------------------------------
# 函数 resolve_path
# 把工具参数中的相对路径锚定到 WORKDIR，并解析为规范化绝对路径。
# 它只负责标准化；是否越出工作区由权限规则在执行工具前判断。
# ------------------------------------------------------------------
def resolve_path(raw_path:str)-> Path:
    """相对路径转换为绝对"""

    candidate = Path(raw_path)

    # 绝对路径保持真实含义，权限层才能正确识别工作区外目标。
    if candidate.is_absolute():
        return candidate.resolve()
    
    return (WORKDIR/candidate).resolve()


# ------------------------------------------------------------------
# 函数 check_deny_list
# 对 shell 命令做不可绕过的高危模式检查。命中后直接返回原因，不询问用户。
# 这是教学版字符串匹配，不等价于完整 shell 解析器，生产环境需要更严格的隔离。
# ------------------------------------------------------------------
# check_deny_list 已由上方 harness.security import 提供。
        
    
    return None

# ------------------------------------------------------------------
# 函数 ask_user
# 在终端显示调用来源、工具名、参数和风险原因，并要求用户明确输入 y/yes。
# 默认答案为拒绝；ContextVar 让提示能区分父 Agent 与子 Agent。
# ------------------------------------------------------------------
def ask_user(tool_name:str, args:dict[str, Any],reason: str) ->bool:

    scope = AGENT_SCOPE.get()

    print(f"\nWarning: [{scope}] Permission required")
    print(f"Reason: {reason}")
    print(f"Tool: {tool_name}")
    print(f"Arguments: {args}")

    # 仅 y/yes 放行，回车默认拒绝。
    choice = input("Allow? [y/N] ").strip().lower()

    return choice in {"y", "yes"}



# ------------------------------------------------------------------
# 函数 check_rules
# 检查需要人工确认的软规则：潜在破坏性命令，以及读写编辑越出工作区。
# 返回字符串代表需要确认，返回 None 代表没有命中软规则；它本身不执行任何操作。
# ------------------------------------------------------------------
def  check_rules(
        tool_name: str,
        args: dict[str, Any]
) -> str| None:
    """检查用户的操作"""
    
    if tool_name == "run_bash":
        command = str(args.get("command", ""))
        normalized = command.lower()

        if normalized.strip().startswith("del "):
            return "Potentially destructive shell command: del "
        for pattern in POTENTIALLY_DESTRUCTIVE_COMMANDS:
            if pattern.lower() in normalized:
                return(
                    "Potentially destructive shell command: "
                    f"{pattern}"
                )
            
    if tool_name in {"run_read","run_write","run_edit"}:
        raw_path = str(args.get("path", ""))

        try:
            target = resolve_path(raw_path)
        except(OSError, RuntimeError, ValueError) as exc:
            return f"Invalid path:{exc}"
        
        if not target.is_relative_to(WORKDIR):
            return f"Operation accesses outside workspace: {target}"
        
    return None


# ------------------------------------------------------------------
# 函数 check_permission
# 统一编排权限决策：先执行绝对拒绝列表，再执行需要确认的规则。
# 最终只返回布尔值，供 PreToolUse Hook 转换为允许或 Permission denied。
# ------------------------------------------------------------------
def check_permission(
        tool_name: str,
        args: dict[str, Any]
)-> bool:
    "执行deny list 和权限规则检查"

    # 硬拒绝优先，命中后不允许用户意外放行。
    if tool_name == "run_bash":
        command = str(args.get("command", ""))

        denied_reason = check_deny_list(command)

        if denied_reason:
            print(f"\nBlocked: {denied_reason}")
            return False
        
    # 软规则只要求确认，未命中则自动允许。
    confirmation_reason = check_rules(
        tool_name,
        args,
    )

    if confirmation_reason:
        return ask_user(
            tool_name,
            args,
            confirmation_reason,
        )
    
    return True

# ------------------------------------------------------------------
# 函数 on_user_prompt_submit
# UserPromptSubmit 的示例回调，仅记录本轮输入，不改变 Agent 状态。
# ------------------------------------------------------------------
def on_user_prompt_submit(content: Any) -> None:
    print(f"[UserPromptSubmit] {content}")

# ------------------------------------------------------------------
# 函数 on_pre_tool_use
# 记录工具调用并执行权限检查；返回拒绝文本即可阻止 handler 真正运行。
# ------------------------------------------------------------------
def on_pre_tool_use(
        tool_name: str,
        tool_args: dict[str, Any],
) -> str | None:
    scope = AGENT_SCOPE.get()

    print(f"[{scope} PreToolUse] {tool_name}")
    print(f"Arguments: {tool_args}")

    if not check_permission(tool_name, tool_args):
        return "Permission denied"
    
    return None

# ------------------------------------------------------------------
# 函数 on_post_tool_use
# 记录工具结果的短预览。限制为 500 字符，避免日志被大文件内容淹没。
# ------------------------------------------------------------------
def on_post_tool_use(
        tool_name: str,
        tool_args: dict[str, Any],
        result: ToolMessage | Command,
) ->None:
    scope = AGENT_SCOPE.get()

    print(f"[{scope} PostToolUse] {tool_name}")

    content = getattr(result, "content", result)
    preview = str(content)

    if len(preview) >500:
        preview = preview[:500] + "...(truncated)"

    print(f"Result: {preview}")


# ------------------------------------------------------------------
# 函数 on_stop
# 遍历最终消息，统计 AIMessage 中的工具调用数量，输出一次回合级摘要。
# ------------------------------------------------------------------
def on_stop(messages: list[Any]) ->None:
    tool_call_count = 0
    for message in messages:
        if isinstance(message, AIMessage):
            tool_call_count += len(message.tool_calls or [])

    print(
        f"[Stop] messages={len(messages)}, "
        f"tool_calls={tool_call_count}"
    )
        

register_hook(
    "UserPromptSubmit",
    on_user_prompt_submit,
)

register_hook(
    "PreToolUse",
    on_pre_tool_use,
)

register_hook(
    "PostToolUse",
    on_post_tool_use,
)

register_hook(
    "Stop",
    on_stop,
)



# ------------------------------------------------------------------
# 函数 load_skill
# 按精确名称取出完整技能说明，这是渐进式披露的第二阶段。
# 返回值同时声明 skill root，提醒模型据此解析 SKILL.md 内的相对引用。
# 该函数由 @tool 暴露；它的 docstring 会成为模型可见的工具描述，因此不能随意删除。
# ------------------------------------------------------------------
@tool

# ------------------------------------------------------------------
# 章节说明：8. 以下复用 s07：基础文件与命令工具
# ------------------------------------------------------------------
def load_skill(name: str) -> str:
    """Load the full instructions for a skill.

    Args:
        name: Exact skill name shown in the available-skills catalog.
    """

    # 精确匹配让模型必须使用 catalog 中看到的名字，避免加载错误技能。
    skill = SKILL_REGISTRY.get(name)

    if skill is None:
        available = ", ".join(SKILL_REGISTRY)
        return (
            f"Skill not found: {name}. "
            f"Available skills: {available or '(none)'}"
        )

    # root 先于正文返回，技能中出现相对脚本路径时已有明确解析基准。
    return (
        f"Loaded skill: {skill['name']}\n"
        f"Skill root: {skill['root']}\n"
        "Resolve relative paths mentioned by this skill "
        "against the skill root above.\n\n"
        f"{skill['content']}"
    )

# ------------------------------------------------------------------
# 函数 run_bash
# 在 WORKDIR 中运行 shell 命令，合并 stdout/stderr，并把超时和系统错误转成文本。
# shell=True 能演示完整命令，但风险较高，所以调用前必须经过 tool_hook 权限层。
# 输出截到 50000 字符，避免一次命令无限膨胀上下文。
# ------------------------------------------------------------------
@tool
# 安全边界：shell=True 仅为教学演示，黑名单/路径检查不等于安全边界；生产请使用权限中间件 + 沙箱。
def run_bash(command: str) -> str:
    """Execute a shell command in the current workspace."""

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

        output = (
            result.stdout
            + result.stderr
        ).strip()

        if not output:
            output = "(no output)"

        if result.returncode != 0:
            output = (
                f"Exit code: {result.returncode}\n"
                f"{output}"
            )

        return output[:50000]

    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120 seconds"

    except OSError as exc:
        return f"Error: {exc}"


# ------------------------------------------------------------------
# 函数 run_read
# 以 UTF-8 读取文本，可用 limit 限制返回行数；剩余行数会通过占位文本告知模型。
# 路径标准化与工作区权限检查分层完成，文件错误统一作为字符串返回给 Agent。
# ------------------------------------------------------------------
@tool
def run_read(
    path: str,
    limit: int | None = None,
) -> str:
    """Read a UTF-8 text file.

    Args:
        path: File path, normally relative to the workspace.
        limit: Optional maximum number of lines to return.
    """

    try:
        file_path = resolve_path(path)

        lines = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()

        if limit is not None and limit >= 0 and limit < len(lines):
            remaining = len(lines) - limit

            lines = [
                *lines[:limit],
                f"...({remaining} more lines)",
            ]

        return "\n".join(lines)

    except (OSError, ValueError) as exc:
        return f"Error: {exc}"


# ------------------------------------------------------------------
# 函数 run_write
# 创建缺失的父目录并以 UTF-8 覆盖写入完整内容。
# 这是有副作用的工具；是否允许写入由 PreToolUse 在函数执行前决定。
# ------------------------------------------------------------------
@tool
def run_write(
    path: str,
    content: str,
) -> str:
    """Write UTF-8 content to a file, replacing existing content.

    Args:
        path: Target file path.
        content: Complete new file content.
    """

    try:
        file_path = resolve_path(path)

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_path.write_text(
            content,
            encoding="utf-8",
        )

        return f"Wrote {len(content)} characters to {path}"

    except (OSError, ValueError) as exc:
        return f"Error: {exc}"


# ------------------------------------------------------------------
# 函数 run_edit
# 执行一次精确文本替换，只替换第一个匹配，便于模型做可预测的小范围编辑。
# 找不到 old_text 时不写文件，而是返回明确错误，避免静默产生错误修改。
# ------------------------------------------------------------------
@tool
def run_edit(
    path: str,
    old_text: str,
    new_text: str,
) -> str:
    """Replace the first exact occurrence of text in a UTF-8 file.

    Args:
        path: Target file path.
        old_text: Exact text that should be replaced.
        new_text: Replacement text.
    """

    try:
        file_path = resolve_path(path)

        current_content = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        if old_text not in current_content:
            return f"Error: old_text was not found in {path}"

        updated_content = current_content.replace(
            old_text,
            new_text,
            1,
        )

        file_path.write_text(
            updated_content,
            encoding="utf-8",
        )

        return f"Edited {path}"

    except (OSError, ValueError) as exc:
        return f"Error: {exc}"


# ------------------------------------------------------------------
# 函数 run_glob
# 在工作区内递归匹配 glob，并只保留 resolve 后仍位于 WORKDIR 的路径。
# 结果排序保证相同文件集得到稳定输出，有利于模型推理和测试复现。
# ------------------------------------------------------------------
@tool
def run_glob(pattern: str) -> str:
    """Find workspace files matching a glob pattern.

    Args:
        pattern: Pattern such as "*.py" or "src/**/*.py".
    """

    try:
        results: list[str] = []

        for match in glob.glob(
            pattern,
            root_dir=WORKDIR,
            recursive=True,
        ):
            full_path = (WORKDIR / match).resolve()

            if full_path.is_relative_to(WORKDIR):
                results.append(match)

        if not results:
            return "(no matches)"

        return "\n".join(sorted(results))

    except (OSError, ValueError) as exc:
        return f"Error: {exc}"
    


# ------------------------------------------------------------------
# 章节说明：9. 模型主动调用的 compact 工具
# 手动压缩必须独占调用，避免同时运行的其他工具结果在重写 state 时丢失。
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 函数 compact
# 模型可主动调用的手动压缩工具。它从 ToolRuntime 取得当前 state 和本次 tool_call_id。
# 摘要时排除最后一条发起 compact 的 AIMessage，随后重建 AI 调用与 ToolMessage 配对。
# Command 先删除所有旧消息，再写入摘要和配对消息，既释放上下文又满足工具协议校验。
# ------------------------------------------------------------------
@tool("compact")
def compact(
    runtime: ToolRuntime,
    focus: str ="",
)-> Command[Any] | ToolMessage:
    """Summarize conversation history to free context space.

    Call this tool alone, never in parallel with other tools.

    Args:
        focus: Optional information the summary should emphasize.
    """

    # ToolRuntime 提供调用时的完整 state；普通工具参数拿不到这份图状态。
    # ToolRuntime.state 包含当前图状态；普通工具参数本身看不到这份历史。
    messages = list(runtime.state["messages"])

    # compact 必须是当前最后一条 AIMessage 发起的工具调用，否则无法重建合法配对。
    last_ai = (messages[-1] if messages and isinstance(messages[-1],AIMessage) else None)

    if last_ai is None:
        return ToolMessage(
            content="Compaction failed: no tool-call message found.",
            tool_call_id=runtime.tool_call_id,
            status="error",
        )

    # 从最后一条 AIMessage 中精确找到当前 compact 调用，避免并行 tool_call id 混淆。
    # 通过 tool_call_id 精确匹配，避免一个 AIMessage 同时调用多个工具时拿错调用。
    compact_call = next(
        (
            call
            for call in last_ai.tool_calls
            if call["id"] == runtime.tool_call_id
        ),
        None,
    )

    if compact_call is None:
        return ToolMessage(
            content="Compaction failed: tool call not found.",
            tool_call_id=runtime.tool_call_id,
            status="error",
        )
    
    # 手动压缩也保留完整 transcript，和自动压缩共享同一恢复语义。
    transcript_path = write_transcript(messages)

    # messages[:-1] 排除包含 compact tool_call 的 AIMessage，摘要内容更干净。
    # 排除最后一条尚未执行完成的 compact AIMessage，摘要只包含此前真实对话。
    summary = summarize_history(messages[:-1],MODEL,focus)

    # 压缩后仍重建空 AIMessage + compact ToolMessage 对，满足 API 的工具消息配对要求。
    # 即使历史被清空，仍重建 AI(tool_calls) + ToolMessage 配对，满足 provider 消息协议。
    compact_pair  = AIMessage(content="", tool_calls=[compact_call])

    result_message = ToolMessage(
        content=(
            "[Compacted. Conversation history "
            "has been summarized.]"
        ),
        tool_call_id = runtime.tool_call_id,
        name = "compact"
    )

    # 返回 Command 而不是普通字符串，因为该工具需要原子地重写整个消息 state。
    # Command 让“清空旧消息 + 写摘要 + 写工具结果”作为一次状态更新提交。
    return Command(
        update={
            "messages": [
                RemoveMessage(
                    id=REMOVE_ALL_MESSAGES
                ),
                HumanMessage(
                    content=(
                        "[Manual compact]\n\n"
                        f"{summary}\n\n"
                        "Full transcript: "
                        f"{transcript_path}"
                    )
                ),
                compact_pair,
                result_message,
            ],
            "compact_failures": 0,
        }
    )


# ------------------------------------------------------------------
# 章节说明：10. 模型、子 Agent 与父 Agent 组装
# ------------------------------------------------------------------
#这个列表不包含task防止循环调用
BASE_TOOLS = [
    run_bash,
    run_read,
    run_write,
    run_edit,
    run_glob,
    load_skill,
]


# 父、子和摘要流程复用同一模型实例；生产系统也可注入更便宜的专用摘要模型。
MODEL = ChatOpenAI(
    model=MODEL_ID,
    max_completion_tokens=8000,
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

SUB_SYSTEM = f"""
You are an isolated coding subagent working in:

{WORKDIR}

{SKILL_SYSTEM_PROMPT}

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


# 子 Agent 继续使用权限 Hook。
# 不传入 TodoListMiddleware，因为 s06 的子 Agent 只有基础工具。
# SUB_MIDDLEWARE = [
#     tool_hook,
# ]

SUB_AGENT = create_agent(
    model=MODEL,
    tools=BASE_TOOLS,
    system_prompt=SUB_SYSTEM,
    middleware=[tool_hook],
    name="worker",
)

# ------------------------------------------------------------------
# 函数 extract_final_text
# 从后向前寻找最后一条有文本的 AIMessage，兼容纯字符串和多内容块两种响应格式。
# 父 Agent 只需要子 Agent 的最终结论，因此不会把子 Agent 的完整消息历史合并回来。
# ------------------------------------------------------------------
def extract_final_text(messages: list[Any]) -> str:
    """读取最后一条aimessage的中文文本内容"""

    # 倒序跳过早期工具规划，只提取子 Agent 最终结论。
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue


        content = message.content

        if isinstance(content, str):
            if content.strip():
                return content.strip()
            

            continue

        if not isinstance(content, list):
            continue

        texts: list[str] = []

        # provider 可能使用字符串块、字典块或带 .text 的对象块。
        for block in content:
            text:str | None =None

            if isinstance(block, str):
                text = block

            elif isinstance(block,dict):
                possible_text = block.get("text")

                if isinstance(possible_text, str):
                    text = possible_text

            else:
                possible_text = getattr(
                    block,
                    "text",
                    None
                )

                if isinstance(possible_text, str):
                    text = possible_text

            if text and text.strip():
                texts.append(text.strip())

        if texts:
            return "\n".join(texts)
        
    return ""

# ------------------------------------------------------------------
# 函数 task
# 创建一次隔离的子 Agent 调用：只传 description，不传父对话，从而控制上下文污染。
# 调用期间用 ContextVar 标记 sub；finally 无论成功或异常都恢复旧值，避免作用域泄漏。
# 递归上限、一般异常和无文本结论都被转换为父 Agent 能继续处理的工具返回值。
# ------------------------------------------------------------------
@tool("task")
def task(description: str) -> str:
    """Launch an isolated subagent for a complex subtask.

    The subagent receives only this description, not the parent conversation.
    Include the complete objective, paths, constraints and expected output.
    Only the final textual conclusion is returned.
    """

    print("\n\033[35m[Subagent spawned]\033[0m")
    print(f"Task: {description}")

    # token 记录进入子 Agent 前的精确 ContextVar 值，finally 中可无损恢复。
    scope_token = AGENT_SCOPE.set("sub")

    try:
        # 只传 description，不传父 messages；这就是上下文隔离边界。
        result = SUB_AGENT.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": description,
                    }
                ]
            },
            config={"recursion_limit": 128},
        )

        # 父 Agent 只获得最终摘要，子 Agent 的探索历史不会占父上下文。
        summary = extract_final_text(
            result.get("messages", [])
        )

        return (
            summary
            or "Subagent finished without a textual conclusion."
        )

    except GraphRecursionError:
        return (
            "Subagent stopped because it reached "
            "the execution limit."
        )

    except Exception as exc:
        return (
            "Subagent failed: "
            f"{type(exc).__name__}: {exc}"
        )

    finally:
        # 成功、异常、提前 return 都会恢复 scope，防止后续日志误标。
        AGENT_SCOPE.reset(scope_token)
        print("\033[35m[Subagent done]\033[0m")

# ------------------------------------------------------------------
# 章节说明：父 Agent 的 system prompt 除规划/委托外，还说明何时可主动 compact。
# ------------------------------------------------------------------
PARENT_SYSTEM = f"""
You are a coding agent working in:

{WORKDIR}
{SKILL_SYSTEM_PROMPT}

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
You have a compact tool. Call it only when conversation history should be
summarized to free context space. Always call compact alone, never together
with another tool.
"""


PARENT_TOOLS = [
    *BASE_TOOLS,
    task,
    compact,
]

# ------------------------------------------------------------------
# 章节说明：ContentCompactionMiddleware 放在工具 Hook 之前，负责每次模型调用的消息治理。
# ------------------------------------------------------------------
# 压缩中间件使用完整 PARENT_SYSTEM 计数，避免低估固定系统提示所占 token。
PARENT_MIDDLEWARE = [
    user_prompt_submit,


    ContentCompactionMiddleware(
        summary_model=MODEL,
        system_prompt=PARENT_SYSTEM,
        trigger_token=AUTO_COMPACT_TOKENS,
    ),

    TodoListMiddleware(
        system_prompt="""
        You must call write_todos before calling run_bash, run_read, run_write,
        run_edit, run_glob, or task.

        For every non-empty, non-trivial request:

        1. Create at least one todo item before using another tool.
        2. Keep exactly one relevant item in_progress while working.
        3. Update todo statuses as work progresses.
        4. Mark items completed only after the work is actually complete.
        """,
        tool_description="""
        Create or update the current task list. This is a mandatory planning tool.
        Call it before using run_bash, run_read, run_write, run_edit, run_glob,
        or task.
        """,
    ),

    tool_hook,
    stop_hook,
]

agent = create_agent(
    model= MODEL,
    tools=PARENT_TOOLS,
    system_prompt=PARENT_SYSTEM,
    middleware=PARENT_MIDDLEWARE,
    name="parent",
)


# ------------------------------------------------------------------
# 章节说明：11. 流式终端输出与 CLI
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 函数 content_to_text
# 把模型内容统一转换成终端文本，兼容字符串、字典内容块和带 text 属性的对象。
# 未知非列表类型使用 str 兜底，使日志展示不会因供应商响应格式差异而中断。
# ------------------------------------------------------------------
def content_to_text(content: Any) -> str:
    """把 LangChain 消息内容转换成适合终端输出的文本。"""

    if isinstance(content, str):
        return content

    # 未知对象保留字符串表示，调试时不会静默丢失供应商特有内容。
    if not isinstance(content, list):
        return str(content)

    texts: list[str] = []

    # 多内容块逐个提取文本，图片等非文本块在本终端示例中忽略。
    for block in content:
        if isinstance(block, str):
            texts.append(block)
            continue

        if isinstance(block, dict):
            text = block.get("text")

            if isinstance(text, str):
                texts.append(text)

            continue

        text = getattr(block, "text", None)

        if isinstance(text, str):
            texts.append(text)

    return "\n".join(texts)


# ------------------------------------------------------------------
# 函数 print_message
# 按消息类型打印新增事件：AIMessage 展示工具调用和回复，ToolMessage 展示状态与结果。
# 只负责界面输出，不修改 state；因此可与 stream_mode='values' 的增量游标配合。
# ------------------------------------------------------------------
def print_message(message: Any) -> None:
    """打印一条新产生的 Agent 消息。"""

    if isinstance(message, AIMessage):
        # AIMessage 可以同时包含自然语言和 tool_calls，两部分都要展示。
        if message.tool_calls:
            print("\n模型调用工具：")

            for tool_call in message.tool_calls:
                print(f"工具名：{tool_call['name']}")
                print(
                    "参数："
                    f"{tool_call.get('args', {})}"
                )

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

    content = getattr(message, "content", message)

    print("\n消息：")
    print(content_to_text(content))


# ------------------------------------------------------------------
# 函数 print_todos
# 把 TodoListMiddleware 维护的结构化列表格式化为带状态的编号列表。
# ------------------------------------------------------------------
def print_todos(todos: list[dict[str, Any]]) -> None:
    """打印 Todo 状态。"""

    print("\n当前 Todo：")

    for index, todo_item in enumerate(
        todos,
        start=1,
    ):
        status = todo_item.get(
            "status",
            "pending",
        )

        content = todo_item.get(
            "content",
            "",
        )

        print(
            f"{index}. [{status}] {content}"
        )


# ============================================================
# 父 Agent 主循环
# ============================================================

# ------------------------------------------------------------------
# 函数 agent_loop
# 消费 Agent 的状态流，只打印上次游标之后的新消息以及发生变化的 Todo。
# 流结束后用 final_state 覆盖 session_state，使 messages、todos 和扩展状态跨用户轮次保留。
# ------------------------------------------------------------------
def agent_loop(
    session_state: dict[str, Any],
) -> None:
    """运行父 Agent，并把最终状态保存到 session_state。"""

    existing_messages = session_state.get(
        "messages",
        [],
    )

    seen_message_count = len(existing_messages)
    last_todos = session_state.get("todos")
    final_state: dict[str, Any] | None = None

    # values 模式每一步返回完整状态，因此需要 seen_message_count 做显示去重。
    for state in agent.stream(
        session_state,
        stream_mode="values",
    ):
        final_state = state

        todos = state.get("todos")

        # 仅 Todo 结构真正变化时重绘列表。
        if todos is not None and todos != last_todos:
            print_todos(todos)
            last_todos = todos

        current_messages = state.get(
            "messages",
            [],
        )

        new_messages = current_messages[
            seen_message_count:
        ]

        for message in new_messages:
            print_message(message)

        seen_message_count = len(
            current_messages
        )

    if final_state is not None:
        # 不只保存 messages，也保存 TodoListMiddleware 的 todos 状态。
        session_state.clear()
        session_state.update(final_state)


# ============================================================
# CLI
# ============================================================

# ------------------------------------------------------------------
# 函数 main
# 提供交互式 CLI：收集用户输入、追加 Human 消息、运行 Agent，并处理退出与异常。
# session_state 在 while 循环外创建，所以同一进程中的多轮对话共享完整状态。
# ------------------------------------------------------------------
def main() -> None:
    print("s08: Context Compact — four-layer compaction pipeline")

    # 保存完整状态，而不只是 messages，
    # 这样 todos 也能跨用户轮次保留。
    session_state: dict[str, Any] = {
        "messages": [],
    }

    while True:
        try:
            query = input(
                "\033[36ms08 >> \033[0m"
            )

        except (
            EOFError,
            KeyboardInterrupt,
        ):
            print()
            break

        if query.strip().lower() in {
            "",
            "q",
            "exit",
        }:
            break

        # 同一个 session_state 在 while 外创建，messages/todos/失败计数可跨回合保留。
        session_state.setdefault(
            "messages",
            [],
        ).append(
            {
                "role": "user",
                "content": query,
            }
        )

        try:
            agent_loop(session_state)

        # 循环未收敛与普通 API 异常分开提示，便于判断问题位于图控制流。
        except GraphRecursionError:
            print(
                "\nAgent stopped because it reached "
                "the execution limit."
            )

        except Exception as exc:
            print(
                "\nAgent error: "
                f"{type(exc).__name__}: {exc}"
            )

        print()


if __name__ == "__main__":
    main()
