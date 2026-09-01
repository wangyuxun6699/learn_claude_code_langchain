# s13: Agent Teams — 用 LangGraph Handoff 组织多 Agent 协作

[s12](../s12_cron_scheduler/) → **s13** → [s14](../s14_mcp_plugin/)

> 一个 Agent 负责统筹，一个 Agent 负责执行；通过显式状态和工具调用完成控制权交接。

本章参考 [`shareAI-lab/learn-claude-code/s13_agent_teams`](https://github.com/shareAI-lab/learn-claude-code/tree/main/s13_agent_teams)，使用本项目锁定的 **LangChain 1.3.11 + LangGraph 1.2.7** 重新实现 Agent Teams 的核心概念。

参考仓库用“队友线程 + 文件收件箱”模拟真实 Claude Code 的异步团队；本章代码采用另一条更贴近 LangChain/LangGraph 的路线：用两个 `create_agent` 子图分别表示 Lead 和 Teammate，再由父级 `StateGraph` 和 `Command` 完成顺序 handoff。

这两种实现都在解决“一个 Agent 的上下文和专业能力有限”这个问题，但执行模型并不相同：

| 维度 | 参考仓库 | 本章实现 |
|---|---|---|
| 协作机制 | 文件收件箱传消息 | 共享图状态 + `Command` |
| 执行方式 | 多个线程可并行 | 同一图内顺序 handoff |
| 队友数量 | Lead + 多个命名队友 | Lead + 一个可动态改名/改角色的队友节点 |
| 上下文 | 每个线程独立消息历史 | 父图共享 `messages`，任务再显式交接 |
| 持久化 | JSON/JSONL 邮箱文件 | `InMemorySaver` checkpoint |
| 教学重点 | 收件箱、线程、异步通信 | 状态、子图、工具协议、路由与上下文工程 |

因此，本章更准确的定位是：**Agent Teams 的最小 handoff 内核**。它展示了团队协作最关键的控制面，但还不是一个真正并行、长期驻留的完整 Agent Team。

## 问题

复杂工作往往同时包含规划、编码、测试、审查和资料分析。把所有职责塞进单个 Agent 会出现几个问题：

- system prompt 和工具越来越多，模型每一步都要在大量无关能力中选择；
- 长任务持续累积消息和工具输出，重要信息被噪声淹没；
- 不同专业角色的工作方式互相干扰；
- 主 Agent 既要统筹又要执行，很容易失去对整体目标的关注；
- 简单的“调用一次子 Agent 工具”只能返回结果，无法清楚表达控制权正在谁手里。

本章把职责拆成两个角色：

- **Lead**：理解用户目标、使用基础工具、管理任务，并决定何时委派；
- **Teammate**：根据 Lead 给出的名字、角色和完整任务执行具体工作，完成后必须汇报。

Lead 与 Teammate 不是两个互相调用的普通 Python 函数。它们各自都是一个能循环调用模型和工具的 LangChain Agent，外层再由 LangGraph 管理状态和跳转。

## 总体架构

![s13: Agent Teams 总览](images/agent-teams-overview.svg)

```text
                         Parent StateGraph(TeamState)

用户消息 ──START──> [ lead_agent 子图 ]
                          │
                          │ assign_teammate(...)
                          │ Command.PARENT + goto="teammate"
                          ▼
                    [ teammate_agent 子图 ]
                          │
                          │ report_to_lead(summary)
                          │ Command.PARENT + goto="lead"
                          ▼
                    [ lead_agent 子图 ] ──> 最终回答

共享内容：messages、active_agent、teammate_name/role/task
会话保存：InMemorySaver + thread_id
```

父图只声明一条固定边：

```python
builder.add_edge(START, "lead")
```

没有固定的 `lead -> teammate` 或 `teammate -> lead` 边。两个方向都由工具返回的 `Command` 在运行时决定：模型不调用 handoff 工具，当前 Agent 子图正常结束，父图也随之结束；模型调用工具，控制权才切换到另一个节点。

## LangChain 和 LangGraph 分别负责什么

LangChain 与 LangGraph 经常一起出现，但在本章中职责很清楚。

### LangChain：负责单个 Agent 内部的执行循环

`create_agent` 把模型、工具、system prompt 和 middleware 组合成一个已经编译的 Agent 图。每个 Agent 内部大致执行：

```text
动态 system prompt
       ↓
调用模型
       ↓
有 tool_calls？──否──> 返回 AIMessage，当前 Agent 结束
       │
       是
       ↓
执行工具 ─────────────> 把 ToolMessage 送回模型继续判断
```

本章通过 LangChain 完成：

- `create_agent`：分别构造 Lead 和 Teammate；
- `@tool`：声明 `assign_teammate` 与 `report_to_lead`；
- `@dynamic_prompt`：每次模型调用前生成角色 prompt；
- `@wrap_tool_call`：在不改变工具行为的前提下打印调用和结果；
- `AgentState`：提供 Agent 所需的标准消息状态；
- `AIMessage`、`HumanMessage`、`ToolMessage`：表达不同来源的会话事件。

### LangGraph：负责多个 Agent 之间的状态与控制流

外层 `StateGraph(TeamState)` 把两个 LangChain Agent 当作子图节点。LangGraph 负责：

- `START`：规定用户请求先进入 Lead；
- `TeamState`：定义多个节点共同读写的数据结构；
- `add_messages`：合并节点产生的新消息；
- `Command`：一次返回“更新什么状态”和“下一步去哪里”；
- `Command.PARENT`：从当前 Agent 子图跳回父团队图；
- `InMemorySaver`：按 `thread_id` 保存多轮 checkpoint；
- `stream(..., subgraphs=True)`：输出父图和两个子图的中间事件。

可以把二者理解为：**LangChain 管一个 Agent 怎么工作，LangGraph 管多个 Agent 何时交接、共享什么状态、下一步去哪里。**

## 共享状态：`TeamState`

```python
class TeamState(AgentState):
    messages: Annotated[list[AnyMessage], add_messages]
    active_agent: NotRequired[str]
    teammate_name: NotRequired[str]
    teammate_role: NotRequired[str]
    teammate_task: NotRequired[str]
```

### `messages` 与 reducer

`messages` 不是普通列表字段。`Annotated[..., add_messages]` 给它绑定消息 reducer：节点只需返回新增消息，LangGraph 会按消息 ID 追加或覆盖，而不是让后一次更新直接替换整个历史。

这对 handoff 很重要。Lead 工具返回三条消息时：

```python
"messages": [
    current_ai_message,
    transfer_message,
    assignment_message,
]
```

它表达的是“把这几条合并到父状态”，不是“丢弃原会话，只留下这三条”。

### `active_agent`

本实现会在交接时更新 `active_agent`，便于观察和未来扩展，但当前路由并不读取它。真正决定下一节点的是 `Command.goto`。

如果以后希望每个新用户回合自动回到上次活跃角色，可以把 `START -> lead` 改为基于 `active_agent` 的条件路由；当前教学版始终让新回合从 Lead 开始。

### 队友元数据

`teammate_name`、`teammate_role` 和 `teammate_task` 在第一次委派前可以不存在，所以使用 `NotRequired`。`teammate_system_prompt` 每次从状态读取这些值：同一个 Teammate 子图可以先扮演测试工程师，下一次再扮演数据库工程师，而不需要重新编译图。

要注意，这只是**动态角色复用**，不是多个命名队友实例同时驻留。新的委派会覆盖上一组队友元数据。

## Handoff 的核心：工具返回 `Command`

普通工具通常返回字符串或结构化数据，LangChain 再把结果包装为 `ToolMessage`。Handoff 工具还必须改变外层图的执行位置，因此返回 `Command`：

```python
return Command(
    graph=Command.PARENT,
    goto="teammate",
    update={
        "active_agent": "teammate",
        "teammate_name": clean_name,
        "teammate_role": clean_role,
        "teammate_task": clean_task,
        "messages": [...],
    },
)
```

三个参数各自回答一个问题：

- `graph=Command.PARENT`：在哪一层图中寻找目标节点；
- `goto="teammate"`：接下来执行哪个节点；
- `update={...}`：跳转前向共享状态合并哪些数据。

Lead 和 Teammate 本身都是 `create_agent` 生成的子图。若省略 `Command.PARENT`，运行时会尝试在当前 Agent 子图里寻找 `teammate` 或 `lead`，而这些节点只存在于父团队图中。

## 为什么必须传 `AIMessage + ToolMessage`

模型调用 `assign_teammate` 时，最近一条 `AIMessage` 中包含类似下面的工具调用：

```text
AIMessage(tool_calls=[{"id": "call_123", "name": "assign_teammate", ...}])
```

对大模型 API 来说，每个工具调用都必须有使用相同 `tool_call_id` 的工具结果。虽然 handoff 的真正效果是切换图节点，消息协议仍然需要闭合：

```python
transfer_message = ToolMessage(
    content="任务已交给 alice，等待队友汇报",
    tool_call_id=tool_call_id,
    name="assign_teammate",
)
```

所以 `assign_teammate` 会把以下内容一起写回父图：

1. 发起工具调用的完整 `AIMessage`；
2. 确认交接的 `ToolMessage`；
3. 发给 Teammate 的结构化 `HumanMessage`。

`report_to_lead` 使用相同原则：保留 Teammate 发起汇报工具调用的 `AIMessage`，补齐对应 `ToolMessage`，再用新的 `HumanMessage` 把总结交给 Lead。

如果只复制 AI 文本、丢掉 `tool_calls`，或者保留工具调用却没有对应 `ToolMessage`，历史就可能被模型服务判定为格式非法，也可能让下游 Agent 误以为工具仍未完成。

## 上下文工程：共享历史不等于任务清晰

本章的两个子图使用同一个 `TeamState.messages` 字段。因此 Teammate 会收到父图已有的共享消息历史，并非真正的空白上下文。

但 Lead 仍然必须通过 `task` 参数提供完整目标、路径、限制、预期输出和验证要求，原因有三点：

- 原始会话可能很长，队友不应自行猜测哪些内容与当前任务相关；
- 结构化交接能形成明确的工作契约，便于 Lead 检查汇报；
- 未来若改为严格隔离的 per-agent 消息通道，完整任务仍可直接复用。

任务使用 `<teammate-assignment>`，汇报使用 `<teammate-result>`。这些 XML 标签只是给模型看的语义边界，不参与 LangGraph 路由。真正路由仍由 `Command` 完成。

当前实现选择共享历史，优点是简单且连续；代价是上下文会继续增长，队友也可能看到无关内容。更强的隔离方案可以为每个 Agent 建立独立消息字段，或者在进入子图前显式筛选/摘要消息。

## 动态 Prompt 和能力隔离

Lead 的工具集：

```text
bash, read_file, write_file,
create_task, list_tasks, get_task, claim_task, complete_task,
assign_teammate
```

Teammate 的工具集：

```text
bash, read_file, write_file, report_to_lead
```

Teammate 没有 `assign_teammate`，因此不能递归创建队友；也没有共享任务管理工具，避免它擅自改变 Lead 的协调计划。这种**按角色裁剪工具**比只在 prompt 里说“不要调用某工具”更可靠，因为不可用工具根本不会进入模型看到的 schema。

`@dynamic_prompt` 则解决同一个图在不同状态下如何获得不同身份：

- Lead prompt 强调委派质量、结果核验和最终责任；
- Teammate prompt 注入动态名称、角色、工作目录和任务；
- Teammate 被要求结束前调用 `report_to_lead`，否则它直接输出普通 AIMessage 时父图会自然结束，Lead 无法继续接管。

## Middleware：横切逻辑不进入业务工具

两个 `@wrap_tool_call` middleware 会打印：

- 模型请求的工具名、参数和调用 ID；
- 普通工具返回的完整 `ToolMessage`；
- handoff 工具返回的 `Command`、目标节点和状态更新字段；
- 工具异常类型与消息。

Lead 还复用 s11 的 `BackgroundNotificationMiddleware`，在模型调用前收集已完成的后台命令结果。Teammate 没有这个 middleware，因为本章把后台结果的统筹责任留给 Lead。

Middleware 适合日志、重试、权限检查、动态 prompt、上下文压缩等横切逻辑；业务工具则应专注于实际动作。两者分离后，不需要在每个工具里重复写追踪代码。

## Checkpoint 与多轮会话

```python
checkpointer = InMemorySaver()
team_graph = builder.compile(checkpointer=checkpointer)
```

`run_turn` 每次只传入新的 `HumanMessage`，并固定使用 `thread_id="s13-main"`。`InMemorySaver` 根据这个 ID 找回先前 checkpoint，因此 Lead 能在下一轮继续看到前面的委派和汇报。

需要区分三种“状态保存”：

- 同一 Python 进程、同一 `thread_id`：保留；
- 同一进程、不同 `thread_id`：隔离；
- Python 进程重启：丢失。

生产环境若需要跨重启恢复，应换成 SQLite 或 Postgres checkpointer，并为不同用户/会话生成稳定且隔离的 thread ID。

## Streaming 与输出去重

```python
team_graph.stream(
    ...,
    stream_mode="values",
    subgraphs=True,
)
```

`subgraphs=True` 让流事件携带 Lead/Teammate 子图命名空间，终端才能标出当前消息来自哪个 Agent。`values` 模式返回每一步的累计状态，因此同一历史消息会反复出现；`print_new_messages` 使用 s11 的 `message_key` 去重，只打印首次观察到的 AI 文本。

工具调用与 `ToolMessage` 已由 middleware 完整打印，所以流处理函数只打印 AI 文本，避免重复输出。

## 一次完整交接

假设用户输入：

```text
请让 alice 作为测试工程师检查 schema.sql，并把发现汇报给你。
```

执行顺序如下：

1. 父图从 `START` 进入 Lead；
2. Lead 模型决定调用 `assign_teammate(name="alice", role="tester", task="...")`；
3. LangChain 注入当前 `state` 和 `tool_call_id`；
4. 工具返回 `Command.PARENT`，更新队友元数据和消息，并跳到 `teammate`；
5. Teammate 的动态 prompt 从状态读取 Alice 的身份和任务；
6. Teammate 可调用 `read_file`、`bash` 等工具完成检查；
7. Teammate 调用 `report_to_lead(summary="...")`；
8. 第二个 `Command.PARENT` 把总结写入状态并跳回 `lead`；
9. Lead 读取 `<teammate-result>`，必要时自行验证，然后向用户给出最终回答。

整个过程可能包含多次模型调用和工具调用，但控制权在任一时刻只属于一个 Agent。

## 相对 s12 的变化

本章复用的是 s11 的模型、文件工具、任务系统和后台通知 middleware，没有把 s12 的 Cron 工具带入团队图。这样可以把注意力集中在 handoff，而不是继续扩大工具集合。

| 组件 | s12 | s13 |
|---|---|---|
| Agent 数量 | 单个 Agent | Lead + Teammate 两个 Agent 子图 |
| 外层编排 | 单 Agent 回合 + 调度线程 | `StateGraph(TeamState)` |
| 控制权 | 始终属于同一 Agent | `Command.goto` 在节点间切换 |
| 新状态 | Cron 注册表和队列 | 当前角色与队友元数据 |
| 新工具 | Cron 创建、列出、取消 | `assign_teammate`、`report_to_lead` |
| Prompt | 一个动态 system prompt | Lead/Teammate 两套动态 prompt |
| 观测 | 消息输出 | 子图命名空间 + 工具/Command 追踪 |

## 本章文件

- `code.py`：带注释教学版（可直接运行）；
- `code_uncommented.py`：去掉教学行注释的完整版本，适合快速通读和自行练习；

两个文件的运行逻辑一致。

## 运行

先在仓库根目录准备环境：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# 编辑 .env，至少填写 MODEL_ID 和 OPENAI_API_KEY
```

运行任一版本：

```powershell
python -m s13_agent_teams.code
python -m s13_agent_teams.code_uncommented
```

也支持直接运行文件：

```powershell
python s13_agent_teams/code.py
```

可以尝试以下 prompt：

1. `请把当前目录下的 Python 文件清单交给 alice，她是代码审查员，让她总结每章的文件命名规律。`
2. `让 bob 作为测试工程师检查 schema.sql 的 SQL 语法和约束设计，完成后你再复核他的结论。`
3. `先自己读取 README.md，再把 s13_agent_teams/code.py 的架构分析交给一名 LangGraph 专家。收到汇报后比较两者。`

观察终端中的四类信号：

- `[Lead tool_call]`：Lead 发起委派；
- `[handoff] lead -> alice`：父图控制权切到 Teammate；
- `[Teammate tool_call]`：队友独立使用工具；
- `[handoff] alice -> lead`：队友汇报后 Lead 恢复控制权。

## 验证代码结构

不调用模型也可以先做静态验证：

```powershell
python -m py_compile s13_agent_teams/code.py
python -m py_compile s13_agent_teams/code_uncommented.py

python -c "import s13_agent_teams.code as c; print(type(c.team_graph).__name__)"
```

预期最后输出 `CompiledStateGraph`。

## 教学版边界

为了让 handoff 机制保持可读，本章有意省略或简化以下能力：

- **不并行**：一次只有一个 Teammate 节点运行，不支持多个队友 fan-out/fan-in；
- **不驻留多个身份**：名字和角色只是状态字段，下一次委派会覆盖；
- **共享消息历史**：没有 per-agent 私有上下文，长任务可能继续膨胀；
- **内存 checkpoint**：进程重启后会话消失；
- **固定 thread ID**：示例 CLI 只维护一个主会话，不适合多用户服务；
- **没有权限冒泡**：队友工具不会向 Lead 发起审批请求；
- **没有取消/关机协议**：顺序 handoff 不需要终止后台队友，但也没有优雅停止语义；
- **没有业务级循环保护**：`recursion_limit=128` 只是运行时保险，不等于明确的最大委派次数；
- **共享工作目录缺少隔离**：若扩展为并行队友，需要文件锁、worktree 或任务所有权约束。

## 从本章扩展到真正的 Agent Team

如果要从顺序 handoff 升级为并行团队，可以按下面的顺序演进：

1. 把单个 `teammate_*` 字段改成按 agent ID 索引的成员注册表；
2. 为每个队友维护独立消息通道或独立子图 checkpoint；
3. 用 `Send` 或任务队列进行并行 fan-out，再由 Lead 汇总 fan-in；
4. 引入结构化邮箱消息，区分任务、结果、权限、空闲和关机事件；
5. 给共享文件写入增加锁、worktree 隔离或冲突检测；
6. 把 `InMemorySaver` 换成持久化 checkpointer；
7. 增加最大队友数、超时、取消、重试、权限审批和可观测性。

如果主要需求是“主 Agent 调用临时专家并收集结果”，LangChain 的 subagents/supervisor 模式通常更直接；如果需要角色轮流直接接管对话，本章的 handoff 模式更合适；如果需要多个专家同时工作后统一综合，则应使用 router 或自定义 LangGraph fan-out/fan-in 工作流。

## 参考资料

- [参考仓库 s13：Agent Teams](https://github.com/shareAI-lab/learn-claude-code/tree/main/s13_agent_teams)
- [LangChain 官方文档：Multi-agent patterns](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [LangChain 官方文档：Handoffs](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs)
- [LangChain 官方文档：Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain 官方文档：Middleware](https://docs.langchain.com/oss/python/langchain/middleware/overview)
- [LangGraph API：Command](https://reference.langchain.com/python/langgraph/types/Command)

## 接下来

s13 已经能在 Lead 和 Teammate 之间交接控制权。新版 17 章编排中，旧版的 s16 Team Protocols、s17 Autonomous Agents、s18 Worktree Isolation 已并入本章，对应材料见 [legacy](../legacy/)。

[s14: MCP & Plugin](../s14_mcp_plugin/) 将把外部工具接入同一个工具池。

<details>
<summary>深入 CC 源码</summary>

> 本章为机制级对照：Claude Code 真实实现里并没有“命名队友 + 文件收件箱”这套东西，参考仓库和本 LangChain 版本都是为教学把“多 Agent 协作”拆成可读的抽象。以下不逐行对应 CC 源码，只讲清真实 CC 怎么做、教学版各自简化和改写了什么。

### 一、CC 的委派本质是“一次性 subagent”，不是“持久队友”

Claude Code 里最接近“委派”的是 Task/subagent 工具（本仓库 s06 的对应物）：主 agent 每次调用都新建一个隔离的上下文，跑完只回传最终结果，然后销毁；没有跨任务保留身份的“队友线程”，也没有 WORK/IDLE 生命周期。参考仓库引入“持久队友”，是为了给“结果与空闲分开表达”“空闲队友自动认领 ready task”“队友与 Lead 之间有显式关机 / 审批协议”这些概念一个落点；它们都是教学抽象，不是 CC 源码里的某个组件。

### 二、“MessageBus 文件邮箱”是教学版的异步通道模拟

CC 的 subagent 结果作为工具结果同步回传主循环，不存在演员之间通过 JSONL 邮箱互发消息的机制。参考仓库用文件收件箱演示“通信放在模型上下文之外”；本 LangChain 版改用一张父级 `StateGraph` 里的两个子图，通过 `Command` 在父子图之间切换控制权。三者目标一致（不让一个队友的工具结果污染另一个的推理），实现分别是文件收件箱、类型化邮箱消息、图状态 + `Command`。

### 三、任务认领与依赖图，对应 CC 的 TodoWrite 与任务系统

“空闲队友原子认领 ready task”是团队版的教学简化。CC 有 TodoWrite（会话内清单）和持久化任务系统（s10）；多 agent 竞争同一任务时的跨进程锁、高水位 ID，都属于 s10 任务系统的课题，而不是“队友抢任务”这个教学场景本身。

### 四、worktree 隔离才是 CC 真实存在的并行机制

CC 确实用 git worktree 给并行任务分隔工作目录（本仓库 legacy 里 s18 Worktree Isolation 的来源）。参考仓库把它并入 s13 作为“任务绑定的 worktree”；本 LangChain 版为聚焦 handoff 没有实现 worktree，只保留“共享工作目录”这一最小假设。

### 五、本 LangChain 版的取舍

本仓库最终做的是“Agent Teams 的最小 handoff 内核”：Lead / Teammate 是同一张 `StateGraph` 里顺序切换的两个 `create_agent` 子图，共享 `messages`，用 `Command.PARENT` + `goto` 交接控制权。它刻意省略了并行队友、驻留多身份、每-agent 私有上下文、持久化 checkpointer 与 worktree——这些在参考仓库 / 真实 CC 里分别对应线程、命名队友、线程收件箱、会话状态与 git worktree。
</details>

