"""
s15：Agent Teams。

这一章用 LangChain 的 create_agent 构造两个可独立运行的 Agent 子图，再用
LangGraph StateGraph 把它们组织成 Lead <-> Teammate 的顺序 handoff 工作流：

    用户 -> Lead
              |
              | assign_teammate 返回 Command(goto="teammate")
              v
          Teammate
              |
              | report_to_lead 返回 Command(goto="lead")
              v
            Lead -> 用户

重点不是“多开几个模型”，而是处理三个工程问题：谁拥有当前控制权、交接时传递
哪些上下文、工具调用消息如何保持合法。本实现一次只激活一个队友，因此展示的是
可解释的 handoff，而不是多个队友并行执行；并行 fan-out/fan-in 留给更完整的团队
调度器。

运行方式：python -m s13_agent_teams.code
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

# LangChain 提供 Agent 工厂、消息、工具与 middleware；它负责每个节点内部的
# “模型 -> 工具 -> 模型”循环。
from typing_extensions import NotRequired
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt, wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool

# LangGraph 提供跨 Agent 的共享状态、图路由、handoff Command 与 checkpoint。
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

# 兼容 `python s13_agent_teams/code.py` 直接运行。以 `python -m` 启动时
# 仓库根目录通常已经在 sys.path 中，这个判断不会重复插入。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from s11_background_tasks import code as base


# 复用 s13 已配置好的 ChatOpenAI 模型、工作目录和文件/任务/后台工具。这样本章
# 只实现团队编排，不复制此前章节的基础设施。
MODEL = base.model
WORKDIR = base.WORKDIR


# ============================================================
# 1. 团队共享状态与 handoff 消息辅助函数
# ============================================================

class TeamState(AgentState):
    """Lead和Teammate共享LangGraph状态"""
    # add_messages 是消息 reducer：节点只返回新增消息，LangGraph 负责按消息 ID
    # 追加或覆盖；工具不需要复制整段 history。
    messages: Annotated[list[AnyMessage], add_messages]

    # NotRequired 允许首次进入图时只提供 messages。第一次 handoff 后才写入队友
    # 元数据，动态 prompt 会从这些字段生成队友身份和任务边界。
    active_agent: NotRequired[str]
    teammate_name: NotRequired[str]
    teammate_role: NotRequired[str]
    teammate_task: NotRequired[str]

def extract_last_ai_message(messages: list[AnyMessage]):
    # 这个文本辅助函数适合日志/UI；handoff 本身必须保留完整 AIMessage，因为其中
    # 还包含 tool_calls、id 和 provider 元数据，不能只传纯文本。
    for message in reversed(messages):
        if not isinstance(message,AIMessage):
            continue

        text = base.content_to_text(message.content).strip()
        if text:
            return text

    return ""

def last_ai_message(state: TeamState)->AIMessage:
    # 当前工具就是由最近一条 AIMessage 中的 tool call 触发的。交接时把这条消息
    # 和对应 ToolMessage 成对带回父图，才能维持合法的消息协议。
    for message in reversed(state["messages"]):
        if isinstance(message,AIMessage):
            return message

    raise ValueError("当前状态没有AImessage")


# ============================================================
# 工具调用追踪
# ============================================================

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

    # handoff 工具不返回普通字符串，而是返回同时包含“状态更新 + 下一节点”的
    # Command。这里把 Command 展开，方便学习时观察路由发生了什么。
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

    # handler 是 LangChain middleware 链中的下游工具执行器。包装它而不替代它，
    # 就能同时记录普通 ToolMessage 和 handoff Command。
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
    # name/role/task 是模型可见的参数；state/tool_call_id 由运行时注入，不会暴露
    # 在工具 JSON Schema 中，也不需要模型自己伪造。
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

    # LLM 发出工具调用后，消息历史必须出现相同 tool_call_id 的 ToolMessage。
    # 即使真正作用是“切换节点”，也要补这个人工工具响应来闭合调用。
    transfer_message = ToolMessage(
        content=(
            f"任务已交给{clean_name}"
            "等待队友汇报"
        ),
        tool_call_id = tool_call_id,
        name = "assign_teammate"
    )
    # 父图会把共享 messages 传给下游队友，但 Lead 仍应把完整任务压缩成显式
    # 交接消息：共享历史可能很长、含糊或包含与本任务无关的信息。XML 标签只是
    # 提示结构，不是 LangGraph 的路由协议。
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

    # assign_teammate 在 Lead 的 create_agent 子图内部执行；Command.PARENT 表示
    # 跳出最近的子图，到父 StateGraph 的 teammate 节点，而不是在 Lead 子图中
    # 查找同名节点。
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

    # 与正向交接相同，先补齐 Teammate 发出的 report_to_lead 工具调用。
    tool_result = ToolMessage(
        content="工作总结已发给lead",
        tool_call_id = tool_call_id,
        name = "report_to_lead"
    )

    # 只把高层总结交还给 Lead，而不是无条件复制队友的全部内部轨迹。这是最重要
    # 的 context engineering 选择之一：控制 token、隔离噪声、保留可行动结论。
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

    # 返回父图的 lead 节点，恢复 Lead 的控制权。
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
    # dynamic_prompt 在每次模型调用前运行。Lead 的规则虽是静态文本，但使用
    # middleware 与 s13 的动态上下文组装方式保持一致，也方便以后按状态扩展。
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
    # 同一个 teammate_agent 图可以服务不同名字/角色/任务；身份由 TeamState 在
    # handoff 时写入，不必为 Alice、Bob 各编译一份图。
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

# ============================================================
# 2. Agent 能力边界与子图构造
# ============================================================

# Lead 拥有工作区工具、共享任务系统和“分配队友”能力。它没有 report_to_lead，
# 因为汇报只属于 Teammate -> Lead 方向。
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

# 队友只获得完成隔离任务所需的最小工具集，不继承任务管理或再次委派能力。
TEAMMATE_TOOLS = [
    base.run_bash,
    base.run_read,
    base.run_write,
    report_to_lead,
]


# create_agent 返回的不是一个简单函数，而是已经编译好的 LangGraph Agent：内部
# 会反复进行 model node -> tools node，直到模型产生不含工具调用的 AIMessage，
# 或工具返回 Command 跳出子图。
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


# Lead 与 Teammate 共用同一个底层模型对象，但 system prompt、工具集合和
# middleware 各自独立，因此表现为两个不同角色。
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

# ============================================================
# 3. 父级团队图：用 Command 实现双向路由
# ============================================================

builder = StateGraph(TeamState)

# 两个 create_agent 图作为父图中的两个节点，也就是“子图作为节点”。
builder.add_node("lead",lead_agent)
builder.add_node("teammate",teammate_agent)
builder.add_edge(START, "lead")
# 不添加 lead -> teammate 或 teammate -> lead 的固定边。
# 路由完全由 assign_teammate/report_to_lead 返回的 Command 控制。
#
# 如果 Lead 直接生成最终回答，没有 Command，图自然结束。
# 如果 Teammate 没调用 report_to_lead，图也会结束，因此 prompt 强制要求汇报。
# InMemorySaver 按 thread_id 保存 checkpoint，让多个用户回合能延续同一团队状态。
# 它不跨 Python 进程持久化；生产环境应换 SQLite/Postgres 等 saver。
checkpointer = InMemorySaver()

team_graph = builder.compile(checkpointer=checkpointer)


def _stream_agent_name(namespace: tuple[str, ...]) -> str:
    """从 LangGraph 子图命名空间识别当前 Agent。"""
    # subgraphs=True 时事件携带命名空间，例如 ('lead:<task-id>',)。从内向外查找
    # 节点名，比依赖固定层级更稳健。
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

    # values 模式会反复返回累计状态，所以使用 message_key 去重；否则每次事件都
    # 会把历史 AIMessage 再打印一遍。
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

    # 每次只提交本轮新增 HumanMessage。checkpointer 会用 thread_id 恢复已有状态，
    # add_messages reducer 再把新消息并入历史。
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
            # 一次交接可能包含多个模型/工具节点，默认递归上限偏小；这里放宽上限，
            # 但它仍是防止 Lead/Teammate 相互无限 handoff 的最后保险。
            "recursion_limit": 128,
        },
        # values 便于教学输出完整状态；subgraphs=True 才能看到两个 Agent 子图的
        # 中间事件与命名空间。
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
