from __future__ import annotations

import json
import sys
from pathlib import Path
from threading import RLock
from typing import Annotated, Any, Callable, Literal, NotRequired


from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage,AnyMessage,HumanMessage,ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain.agents import AgentState, create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from s11_background_tasks import code as base

MODEL = base.model
WORKDIR = base.WORKDIR

MessageType = Literal[
    "handoff_request",
    "handoff_response",
    "result",
    "message",
    "shutdown_request",
    "shutdown_response"
]

MAILBOXES: dict[str,list[dict[str,Any]]] = {"lead": []}
MAILBOX_LOCK = RLock()

PENDING_REQUESTS: dict[str, dict[str,Any]] = {}
_next_message_id = 0

def _new_id(prefix: str = "msg") -> str:
    global _next_message_id
    with MAILBOX_LOCK:
        _next_message_id+=1
        return f"{prefix}_{_next_message_id}"

def post_envelope(
        from_agent: str,
        to_agent: str,
        msg_type: str,
        payload: dict[str,Any],
) -> dict[str,Any]:
    """把一条协议信封投递到 to_agent 的收件箱，返回完整信封。"""
    envelope = {
        "id": _new_id("msg"),
        "type": msg_type,
        "from": from_agent,
        "to": to_agent,
        "payload": payload,
    }

    with MAILBOX_LOCK:
        MAILBOXES.setdefault(to_agent,[]).append(envelope)

    return envelope

def consume_inbox(agent:str) -> list[dict[str, Any]]:
    """取出并清空 agent 收件箱（每条只交付一次）。"""
    with MAILBOX_LOCK:
        inbox = MAILBOXES.setdefault(agent,[])
        messages = list(inbox)
        inbox.clear()
        return messages


def envelopes_to_text(envelopes: list[dict[str,Any]]) -> str:
    """把信封列表格式化为模型可读的单行 JSON 文本。"""
    return "\n".join(json.dumps(e, ensure_ascii=False) for e in envelopes)


class TeamState(AgentState):
    messages: Annotated[list[AnyMessage], add_messages]

    active_agent: NotRequired[str]
    teammate_name: NotRequired[str]
    teammate_role: NotRequired[str]
    teammate_task: NotRequired[str]

def last_ai_message(state: TeamState) -> AIMessage:
    """当前工具调用即由最近一条 AIMessage 中的 tool_call 触发。"""
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            return message
    raise ValueError("当前状态没有 AIMessage")

def active_agent_name(state: TeamState) -> str:
    """解析当前发言者的邮箱名：lead 固定叫 lead，队友用其名字。"""

    if state.get("active_agent") == "teammate":
        return state.get("teammate_name","teammate")
    return "lead"


class ProtocolInboxMiddleware(AgentMiddleware):
    """
    等价于参考主循环的"检查收件箱 (shutdown_request 等)"：
    每次模型调用前把当前 agent 收件箱里的协议信封注入为 HumanMessage。
    与 s11 的 BackgroundNotificationMiddleware 同构。
    """

    def __init__(self,kind: str) -> None:
        self.kind = kind

    def before_model(
            self,
            state:dict[str,Any],
            runtime: Any,
    ) -> dict[str, Any] | None:
        agent = "lead" if self.kind == "lead" else state.get("teammate_name","teammate")

        envelopes = consume_inbox(agent)
        if not envelopes:
            return None

        content = (
            "以下是你的收件箱里的协议信封（每条是一个 JSON），请严格按协议处理：\n"
            + envelopes_to_text(envelopes)
        )
        print(f"\033[32m[{agent} inbox] 注入 {len(envelopes)} 条协议信封\033[0m")
        return {"messages": [HumanMessage(content=content)]}

lead_inbox = ProtocolInboxMiddleware("lead")
teammate_inbox = ProtocolInboxMiddleware("teammate")


def _json_text(value:Any) -> str:
    try:
        return json.dumps(value,ensure_ascii=False,indent=2,default=str)
    except(TypeError,ValueError):
        return str(value)

def _trace_tool_call(
        agent_name: str,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest],ToolMessage | Command],
) -> ToolMessage | Command:
    print(f"\n\033[33m[{agent_name} tool_call]\033[0m")
    print(_json_text({
        "id": request.tool_call.get("id"),
        "name": request.tool_call.get("name"),
        "args": request.tool_call.get("args", {}),
    }))
    try:
        result = handler(request)
    except Exception as exc:
        print(f"\033[31m[{agent_name} tool error]\033[0m {type(exc).__name__}: {exc}")
        raise

    if isinstance(result, Command):
        update = result.update
        print(f"\033[35m[{agent_name} Command]\033[0m")
        print(_json_text({
            "graph": result.graph,
            "goto": result.goto,
            "update_keys": sorted(update) if isinstance(update, dict) else None,
        }))
    else:
        print(f"\033[35m[{agent_name} ToolMessage]\033[0m")
        print(_json_text({"name": result.name, "content": result.content}))
    return result

@wrap_tool_call
def trace_lead(request,handler):
    return _trace_tool_call("Lead",request, handler)

@wrap_tool_call
def trace_teammate(request,handler):
    return _trace_tool_call("Teammate",request,handler)

@tool("assign_teammate")
def assign_teammate(
    name: Annotated[str, "Short unique name of the teammate, e.g. alice"],
    role: Annotated[str, "Role of the teammate, e.g. backend developer"],
    plan: Annotated[str, "Complete plan + task the teammate must approve"],
    state: Annotated[TeamState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Literal["teammate"]]:
    """
    召唤一个teammate同时发生handoff_request.
    teammate必须回复request_id + approve做之前.
    """
    clean_name = name.strip()
    clean_role =  role.strip()
    clean_plan = plan.strip()
    if not (clean_name and clean_role and clean_plan):
        raise ValueError("name/role/plan不为空")

    current_ai_message = last_ai_message(state)
    request_id = _new_id("req")
    transfer_message = ToolMessage(
        content=f"已向 {clean_name} 投递 handoff_request {request_id}",
        tool_call_id=tool_call_id,
        name="assign_teammate",
    )
    PENDING_REQUESTS[request_id] = {
        "from": "lead",
        "to": clean_name,
        "plan": clean_plan,
        "status": "pending",
    }

    post_envelope(
        "lead", clean_name, "handoff_request",
        {"request_id": request_id, "plan": clean_plan},
    )
    print(f"\n\033[36m[handoff] lead -> {clean_name} handoff_request {request_id}\033[0m")

    return Command(
        graph=Command.PARENT,
        goto="teammate",
        update={
            "active_agent": "teammate",
            "teammate_name": clean_name,
            "teammate_role": clean_role,
            "teammate_task": clean_plan,
            "messages": [current_ai_message, transfer_message],
        },
    )
@tool("shutdown_team")
def shutdown_team(
    state: Annotated[TeamState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Literal["teammate"]] | str:
    """发送一个shutdown_request给active teammate并且掌握控制"""
    teammate_name = state.get("teammate_name")
    if not teammate_name:
        return "当前没有活跃队友，无需关机。"

    current_ai_message = last_ai_message(state)
    post_envelope("lead", teammate_name, "shutdown_request", {"note": "work complete"})
    tool_result = ToolMessage(
        content=f"已向 {teammate_name} 发送 shutdown_request",
        tool_call_id=tool_call_id,
        name="shutdown_team",
    )
    print(f"\n\033[36m[protocol] lead -> {teammate_name} shutdown_request\033[0m")
    return Command(
        graph=Command.PARENT,
        goto="teammate",
        update={
            "active_agent": "teammate",
            "messages": [current_ai_message, tool_result],
        },
    )


@tool("respond_handoff")
def respond_handoff(
    request_id: Annotated[str, "The request_id from the lead's handoff_request"],
    approve: Annotated[bool, "True to accept and execute the plan, False to reject"],
    state: Annotated[TeamState, InjectedState],
) -> str:
    """
    回复lead的handoff_request用相同的request_id+approve.
    两者都是必需的（这与参考合约的行为一致——缺少它们时调用会报错）。
    返回一个普通字符串可保持队友（teammate）处于活跃状态：随后它要么执行（已批准），要么报告拒绝（已拒绝）。
    """
    pending = PENDING_REQUESTS.get(request_id)
    if pending is None:
        raise ValueError(f"未知的 request_id：{request_id}")

    teammate_name = state.get("teammate_name", "teammate")
    pending["status"] = "approved" if approve else "rejected"

    post_envelope(
        teammate_name, "lead", "handoff_response",
        {"request_id": request_id, "approve": approve},
    )
    print(
        f"\n\033[32m[protocol] {teammate_name} -> lead "
        f"handoff_response request_id={request_id} approve={approve}\033[0m"
    )
    if approve:
        return f"已批准 handoff_request {request_id}，现在开始执行计划。"
    return f"已拒绝 handoff_request {request_id}。"


@tool("report_result")
def report_result(
    summary: Annotated[str, "Complete work summary; files, findings, verification"],
    state: Annotated[TeamState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Literal["lead"]]:
    """发送最后的结果给lead同时夺回控制权"""
    teammate_name = state.get("teammate_name", "teammate")
    current_ai_message = last_ai_message(state)

    post_envelope(teammate_name, "lead", "result", {"summary": summary.strip()})
    tool_result = ToolMessage(
        content="结果已发回 Lead",
        tool_call_id=tool_call_id,
        name="report_result",
    )
    print(f"\n\033[32m[protocol] {teammate_name} -> lead result\033[0m")

    return Command(
        graph=Command.PARENT,
        goto="lead",
        update={
            "active_agent": "lead",
            "messages": [current_ai_message, tool_result],
        },
    )


@tool("shutdown_response")
def shutdown_response(
    state: Annotated[TeamState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Literal["lead"]]:
    """确认 shutdown_request 并把控制权交还 Lead。"""
    teammate_name = state.get("teammate_name", "teammate")
    current_ai_message = last_ai_message(state)

    post_envelope(teammate_name, "lead", "shutdown_response", {"note": "ack"})
    tool_result = ToolMessage(
        content="已确认关机",
        tool_call_id=tool_call_id,
        name="shutdown_response",
    )
    print(f"\n\033[32m[protocol] {teammate_name} -> lead shutdown_response\033[0m")

    return Command(
        graph=Command.PARENT,
        goto="lead",
        update={
            "active_agent": "lead",
            "messages": [current_ai_message, tool_result],
        },
    )


@tool("send_message")
def send_message(
    to: Annotated[str, "Recipient mailbox name, e.g. lead or a teammate name"],
    content: Annotated[str, "Message text"],
    state: Annotated[TeamState, InjectedState],
) -> str:
    """向指定收件箱投递一条自由格式消息（异步邮箱投递）。"""
    sender = active_agent_name(state)
    post_envelope(sender, to, "message", {"content": content.strip()})
    print(f"\n\033[33m[message] {sender} -> {to}\033[0m")
    return f"消息已投递到 {to} 的收件箱（其激活后才会读取）"


LEAD_TOOLS = [
    base.run_bash,
    base.run_read,
    base.run_write,
    base.run_create_task,
    base.run_list_tasks,
    base.run_get_task,
    base.run_claim_task,
    base.run_complete_task,
    assign_teammate,
    shutdown_team,
    send_message,
]

TEAMMATE_TOOLS = [
    base.run_bash,
    base.run_read,
    base.run_write,
    respond_handoff,
    report_result,
    shutdown_response,
    send_message,
]


@dynamic_prompt
def lead_system_prompt(request: ModelRequest[Any]) -> str:
    return f"""
You are the Lead coding agent working in:

{WORKDIR}

You coordinate teammates through a fixed request-response mailbox protocol.
The protocol inbox (handoff_response / result / shutdown_response) is injected
into your context before every model call. Rules:

1. Assign substantial, isolated work with assign_teammate. Its `plan` argument is
   the handoff_request you are asking the teammate to approve; make it
   self-contained and unambiguous.
2. A teammate replies to your handoff_request with the SAME request_id plus
   approve. Do not treat a request as accepted until you see
   handoff_response with approve=true.
3. Do not claim work succeeded before receiving a result envelope.
4. Verify important results yourself when necessary. Use send_message for short
   notes, and the task tools for multi-step work.
5. When the overall request is complete, call shutdown_team before giving the
   user the final answer.
6. Give the final answer only when the whole request is actually complete.
""".strip()


@dynamic_prompt
def teammate_system_prompt(request: ModelRequest[Any]) -> str:
    state = request.state
    teammate_name = state.get("teammate_name", "teammate")
    teammate_role = state.get("teammate_role", "coding specialist")
    teammate_task = state.get("teammate_task", "")

    return f"""
You are teammate "{teammate_name}".

Role:
{teammate_role}

Working directory:
{WORKDIR}

Assigned task:
{teammate_task}

You communicate only through the fixed request-response mailbox protocol. Rules:

1. Check your injected inbox every turn and obey the envelope type:
   - handoff_request: call respond_handoff(request_id, approve) — approve=true to
     accept and execute, approve=false to reject. Do NOT execute before you have
     approved the request.
   - shutdown_request: call shutdown_response() and do nothing else.
2. After respond_handoff(approve=true), execute the plan the lead sent.
3. After respond_handoff(approve=false), call report_result with your rejection
   reason instead of doing the work.
4. Use bash / read_file / write_file only inside the workspace. Do not delegate
   or create another teammate; do not answer the user directly.
5. When finished, call report_result(summary) with files changed, findings,
   verification results, and unresolved issues.
""".strip()


lead_agent = create_agent(
    model=MODEL,
    tools=LEAD_TOOLS,
    state_schema=TeamState,
    middleware=[
        base.BackgroundNotificationMiddleware(),
        lead_inbox,
        trace_lead,
        lead_system_prompt,
    ],
    name="lead",
)

teammate_agent = create_agent(
    model=MODEL,
    tools=TEAMMATE_TOOLS,
    state_schema=TeamState,
    middleware=[
        teammate_inbox,
        trace_teammate,
        teammate_system_prompt,
    ],
    name="teammate",
)

builder = StateGraph(TeamState)
builder.add_node("lead", lead_agent)
builder.add_node("teammate", teammate_agent)
builder.add_edge(START, "lead")
checkpointer = InMemorySaver()
team_graph = builder.compile(checkpointer=checkpointer)


def _stream_agent_name(namespace: tuple[str, ...]) -> str:
    for part in reversed(namespace):
        node_name = part.split(":", 1)[0]
        if node_name == "lead":
            return "Lead"
        if node_name == "teammate":
            return "Teammate"
    return "Team"


def print_new_messages(
    namespace: tuple[str, ...],
    state: dict[str, Any],
    seen: set[tuple[str, Any]],
) -> None:
    agent_name = _stream_agent_name(namespace)
    for message in state.get("messages", []):
        key = base.message_key(message)
        if key in seen:
            continue
        seen.add(key)
        if not isinstance(message, AIMessage):
            continue
        text = base.content_to_text(message.content).strip()
        if text:
            print(f"\033[36m[{agent_name} AIMessage]\033[0m")
            print(text)


def run_turn(query: str, thread_id: str = "s16-main") -> dict[str, Any]:
    seen: set[tuple[str, Any]] = set()
    final_state: dict[str, Any] = {}

    for namespace, state in team_graph.stream(
        {
            "messages": [HumanMessage(content=query)],
            "active_agent": "lead",
        },
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 128,
        },
        stream_mode="values",
        subgraphs=True,
    ):
        final_state = state
        print_new_messages(namespace, state, seen)

    return final_state


def main() -> None:
    print("s16: LangChain Team Protocols (request-response mailbox + shutdown)")
    print("输入问题后回车发送；输入 q 退出。\n")

    while True:
        try:
            query = input("\033[36ms16 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        query = query.strip()
        if query.lower() in {"", "q", "exit"}:
            break

        try:
            run_turn(query)
        except Exception as exc:
            print(f"执行失败：{type(exc).__name__}：{exc}")

        print()


if __name__ == "__main__":
    main()