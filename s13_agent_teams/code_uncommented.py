from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

from typing_extensions import NotRequired
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt, wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from s11_background_tasks import code as base


MODEL = base.model
WORKDIR = base.WORKDIR

class TeamState(AgentState):
    """Lead和Teammate共享LangGraph状态"""
    messages: Annotated[list[AnyMessage], add_messages]

    active_agent: NotRequired[str]
    teammate_name: NotRequired[str]
    teammate_role: NotRequired[str]
    teammate_task: NotRequired[str]

def extract_last_ai_message(messages: list[AnyMessage]):
    for message in reversed(messages):
        if not isinstance(message,AIMessage):
            continue

        text = base.content_to_text(message.content).strip()
        if text:
            return text

    return ""

def last_ai_message(state: TeamState)->AIMessage:
    for message in reversed(state["messages"]):
        if isinstance(message,AIMessage):
            return message

    raise ValueError("当前状态没有AImessage")



def _json_text(value: Any) -> str:
    """把工具参数和结果完整格式化为可读 JSON。"""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    except (TypeError, ValueError):
        return str(value)


def _print_tool_message(
    agent_name: str,
    message: ToolMessage,
) -> None:
    """完整打印一条返回给模型的 ToolMessage。"""
    print(f"\033[35m[{agent_name} ToolMessage]\033[0m")
    print(
        _json_text(
            {
                "type": "tool_message",
                "name": message.name,
                "tool_call_id": message.tool_call_id,
                "status": getattr(message, "status", None),
                "content": message.content,
                "artifact": getattr(message, "artifact", None),
            }
        )
    )


def _print_tool_result(
    agent_name: str,
    result: ToolMessage | Command,
) -> None:
    """打印普通工具结果，或 handoff 工具返回的 Command。"""
    if isinstance(result, ToolMessage):
        _print_tool_message(agent_name, result)
        return

    update = result.update
    print(f"\033[35m[{agent_name} Command]\033[0m")
    print(
        _json_text(
            {
                "graph": result.graph,
                "goto": result.goto,
                "resume": result.resume,
                "update_keys": (
                    sorted(update) if isinstance(update, dict) else None
                ),
            }
        )
    )

    if not isinstance(update, dict):
        print(_json_text(update))
        return

    for message in update.get("messages", []):
        if isinstance(message, ToolMessage):
            _print_tool_message(agent_name, message)

    other_updates = {
        key: value
        for key, value in update.items()
        if key != "messages"
    }
    if other_updates:
        print(f"\033[35m[{agent_name} Command state update]\033[0m")
        print(_json_text(other_updates))


def _trace_tool_call(
    agent_name: str,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """记录一次完整的 tool_call -> ToolMessage/Command 交互。"""
    print(f"\n\033[33m[{agent_name} tool_call]\033[0m")
    print(
        _json_text(
            {
                "id": request.tool_call.get("id"),
                "name": request.tool_call.get("name"),
                "args": request.tool_call.get("args", {}),
                "type": request.tool_call.get("type", "tool_call"),
            }
        )
    )

    try:
        result = handler(request)
    except Exception as exc:
        print(f"\033[31m[{agent_name} tool error]\033[0m")
        print(f"{type(exc).__name__}: {exc}")
        raise

    _print_tool_result(agent_name, result)
    return result


@wrap_tool_call
def trace_lead_tool_call(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """打印 Lead Agent 的全部工具交互。"""
    return _trace_tool_call("Lead", request, handler)


@wrap_tool_call
def trace_teammate_tool_call(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """打印 Teammate Agent 的全部工具交互。"""
    return _trace_tool_call("Teammate", request, handler)

@tool("assign_teammate")
def assign_teammate(
    name: Annotated[
        str,
        "Short unique name of the teammate, e.g. alice or database_dev",
    ],
    role: Annotated[
        str,
        "Role of the teammate, e.g. backend developer",
    ],
    task: Annotated[
        str,
        "Complete task description including paths, constraints, expected results and verification requirements",
    ],
    state: Annotated[
        TeamState,
        InjectedState,
    ],
    tool_call_id: Annotated[
        str,
        InjectedToolCallId,
    ],
) -> Command[Literal["teammate"]]:
    """
    Hand off the current task to a teammate agent.

    state and tool_call_id are injected by LangGraph and do not
    appear in the model-visible tool parameter schema.
    """
    clean_name = name.strip()
    clean_role = role.strip()
    clean_task = task.strip()
    if not clean_name:
        raise ValueError("队友名称不能为空。")
    if not clean_role:
        raise ValueError("队友角色不能为空。")
    if not clean_task:
        raise ValueError("队友任务不能为空。")
    current_ai_message = last_ai_message(state)
    transfer_message = ToolMessage(
        content=(
            f"任务已交给{clean_name}"
            "等待队友汇报"
        ),
        tool_call_id = tool_call_id,
        name = "assign_teammate"
    )
    assignment_message = HumanMessage(
        name="lead",
        content=(
            "<teammate-assignment>\n"
            f"  <name>{clean_name}</name>\n"
            f"  <role>{clean_role}</role>\n"
            f"  <task>{clean_task}</task>\n"
            "</teammate-assignment>"
        ),
    )
    print(
        f"\n\033[36m[handoff] "
        f"lead -> {clean_name} ({clean_role})\033[0m"
    )

    return Command(
        graph=Command.PARENT,
        goto="teammate",
        update={
            "active_agent": "teammate",
            "teammate_name": clean_name,
            "teammate_role": clean_role,
            "teammate_task": clean_task,
            "messages": [
                current_ai_message,
                transfer_message,
                assignment_message,
            ],
        },
    )


@tool("report_to_lead")
def report_to_lead(
    summary: Annotated[
        str,
        "Complete work summary including changes, findings, verification results and open issues",
    ],
    state: Annotated[
        TeamState,
        InjectedState,
    ],
    tool_call_id: Annotated[
        str,
        InjectedToolCallId,
    ],
) -> Command[Literal["lead"]]:
    """Return the completed work summary to the lead agent."""
    teammate_name = state.get("teammate_name","teammate")
    current_ai_message = last_ai_message(state)

    tool_result = ToolMessage(
        content="工作总结已发给lead",
        tool_call_id = tool_call_id,
        name = "report_to_lead"
    )

    result_message = HumanMessage(
        name = str(teammate_name),
        content=(
            "<teammate-result>\n"
            f"  <from>{teammate_name}</from>\n"
            f"  <summary>{summary}</summary>\n"
            "</teammate-result>"
        ),
    )
    print(
        f"\n\033[32m[handoff] "
        f"{teammate_name} -> lead\033[0m"
    )

    return Command(
        graph=Command.PARENT,
        goto="lead",
        update={
            "active_agent": "lead",
            "messages": [
                current_ai_message,
                tool_result,
                result_message,
            ],
        },
    )

@dynamic_prompt
def lead_system_prompt(request: ModelRequest[Any],) -> str:
    return f"""
You are the Lead coding agent working in:

{WORKDIR}

Available capabilities:

- Read, write, and inspect workspace files.
- Run shell commands.
- Assign isolated work to a teammate with assign_teammate.

Team rules:

1. Use assign_teammate for substantial, isolated work.
2. The teammate receives the shared message history, but the task argument
   must still be self-contained and unambiguous.
3. Include the complete objective, paths, constraints, expected output, and
   verification requirements in the task argument.
4. Only one teammate is active at a time in this handoff implementation.
5. After a teammate reports, examine the result and continue working.
6. You may assign another teammate after receiving the previous result.
7. Do not claim teammate work succeeded before receiving <teammate-result>.
8. Verify important teammate results yourself when necessary.
9. Give the final answer to the user only when the overall request is complete.
""".strip()


@dynamic_prompt
def teammate_system_prompt(request: ModelRequest[Any],) -> str:
    state = request.state

    teammate_name = state.get(
        "teammate_name",
        "teammate",
    )
    teammate_role = state.get(
        "teammate_role",
        "coding specialist",
    )
    teammate_task = state.get(
        "teammate_task",
        "",
    )

    return f"""
You are teammate "{teammate_name}".

Role:
{teammate_role}

Working directory:
{WORKDIR}

Assigned task:
{teammate_task}

Rules:

1. Complete the assigned task independently.
2. Use read_file, write_file, and bash when needed.
3. Work only inside the current workspace.
4. Do not delegate or create another teammate.
5. Do not answer the user directly.
6. Verify your work before reporting.
7. When finished, you MUST call report_to_lead.
8. The report must include:
   - files read or changed
   - important implementation details
   - commands executed
   - verification results
   - unresolved issues
""".strip()

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
]

TEAMMATE_TOOLS = [
    base.run_bash,
    base.run_read,
    base.run_write,
    report_to_lead,
]


lead_agent = create_agent(
    model=MODEL,
    tools=LEAD_TOOLS,
    state_schema=TeamState,
    middleware=[
        base.BackgroundNotificationMiddleware(),
        trace_lead_tool_call,
        lead_system_prompt,
    ],
    name="lead",
)


teammate_agent = create_agent(
    model=MODEL,
    tools=TEAMMATE_TOOLS,
    state_schema=TeamState,
    middleware=[
        trace_teammate_tool_call,
        teammate_system_prompt,
    ],
    name="teammate",
)

builder = StateGraph(TeamState)

builder.add_node("lead",lead_agent)
builder.add_node("teammate",teammate_agent)
builder.add_edge(START, "lead")
checkpointer = InMemorySaver()

team_graph = builder.compile(checkpointer=checkpointer)


def _stream_agent_name(namespace: tuple[str, ...]) -> str:
    """从 LangGraph 子图命名空间识别当前 Agent。"""
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
    """
    打印 Lead/Teammate 子图产生的 AI 文本。

    工具调用和 ToolMessage 由工具追踪中间件完整打印，避免这里重复输出。
    """
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

def run_turn(query: str,thread_id: str = "s13-main"):
    """
    执行一个用户回合。

    checkpointer 根据 thread_id 保留多轮状态；
    每次只需要传入新用户消息。
    """
    seen: set[tuple[str, Any]] = set()
    final_state: dict[str, Any] = {}

    for namespace, state in team_graph.stream(
        {
            "messages": [
                HumanMessage(content=query),
            ],
            "active_agent": "lead",
        },
        config={
            "configurable": {
                "thread_id": thread_id,
            },
            "recursion_limit": 128,
        },
        stream_mode="values",
        subgraphs=True,
    ):
        final_state = state
        print_new_messages(namespace, state, seen)

    return final_state

def main() -> None:
    print("s15: LangChain Agent Teams with Annotated handoffs")
    print("输入问题后回车发送；输入 q 退出。\n")

    while True:
        try:
            query = input("\033[36ms15 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        query = query.strip()

        if query.lower() in {"", "q", "exit"}:
            break

        try:
            run_turn(query)
        except Exception as exc:
            print(
                "执行失败："
                f"{type(exc).__name__}：{exc}"
            )

        print()


if __name__ == "__main__":
    main()

