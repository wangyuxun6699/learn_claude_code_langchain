# s16: Team Protocols — 用请求-响应协议协调队友

[s13: Agent Teams](../../s13_agent_teams/) → **s16** → [s17](../s17_autonomous_agents/)

> 队友之间要有约定。在 s13 的 handoff 之上，给队友通信加一层结构化协议：把“谁发请求、谁回响应、怎么关联、怎么优雅关机”固化成一套固定格式。

本章参考 [`shareAI-lab/learn-claude-code/s13_agent_teams`](https://github.com/shareAI-lab/learn-claude-code/tree/main/s13_agent_teams)，使用本项目锁定的 **LangChain 1.3.11 + LangGraph 1.2.7** 重新实现团队协议的核心概念。

参考仓库用“队友线程 + 文件收件箱（`INBOX_DIR/<name>`）”模拟真实 Claude Code 的异步团队协议；本章沿用 s13 已经搭好的 LangGraph 路线：Lead 与 Teammate 仍是两个 `create_agent` 子图，控制权仍由 `Command` 在父图里路由，但**消息不再直接写进共享历史，而是投递到每个 Agent 的收件箱**，收件箱再由 middleware 在每次模型调用前注入上下文。

| 维度 | 参考仓库 | 本章实现 |
|---|---|---|
| 协作机制 | 文件收件箱（`INBOX_DIR/<name>`） | 进程内 dict 收件箱（`MAILBOXES`） |
| 执行方式 | 多个队友线程可以各自轮询 | 同一图内顺序 handoff（单活跃队友） |
| 协议状态 | `ProtocolState` dataclass | `PENDING_REQUESTS` dict |
| 消息路由 | `dispatch_message` 按类型分发 | 模型读注入的信封，调用对应协议工具 |
| 响应匹配 | `match_response` 校验类型 | `respond_handoff` 查 `request_id` 并校验 |
| 关机 | 队友 idle loop 轮询 | 模型读 `shutdown_request` 后调 `shutdown_response` |
| 消息格式 | type + metadata 字典 | 结构化 JSON 信封 |

## 问题

s15 的队友能干活了，但协调是松散的：Lead 发消息、队友回复，既没有固定格式，也无法把一次“请求”和它的“回复”可靠地关联起来。两个场景会立刻暴露问题：

**计划审批**：Lead 想把一个高风险重构交给队友。s15 里队友接到 `assign_teammate` 就直接开干，没有“先确认计划再执行”的环节。应该让队友先看到完整计划，明确回答“同意/拒绝”，通过之后才动手。

**优雅关机**：Lead 想说“工作完成了，你可以退出了”。s15 里队友只是自然结束当前回合，没有任何收尾语义；如果以后有多个并行队友，直接“杀线程”会让写了一半的文件留在磁盘上。需要握手：Lead 发请求，队友确认收尾后关机。

这两个场景结构完全一样：**一方发请求，另一方回响应，请求和响应通过同一个 ID 关联**，并且有一条状态机追踪 `pending → approved / rejected`。

## 总体架构

```text
                      Parent StateGraph(TeamState)

用户消息 ──START──> [ lead_agent 子图 ]
                       │ assign_teammate(name, role, plan)
                       │   → post_envelope("lead","alice","handoff_request",...)
                       │   → Command(goto="teammate")
                       ▼
                     [ teammate_agent 子图 ]
                       │ ProtocolInboxMiddleware 注入 handoff_request
                       │ respond_handoff(request_id, approve)
                       │   → post_envelope("alice","lead","handoff_response",...)
                       │ → （approve=true 时）bash/read/write 干活
                       │ report_result(summary)
                       │   → post_envelope(... "result" ...) + Command(goto="lead")
                       ▼
                     [ lead_agent 子图 ] 读 result
                       │ shutdown_team()
                       │   → post_envelope(... "shutdown_request" ...) + Command(goto="teammate")
                       ▼
                     [ teammate_agent 子图 ] shutdown_response() + Command(goto="lead")
                       ▼
                     [ lead_agent 子图 ] → 最终回答

信封：进程内邮箱 MAILBOXES（投递/消费，deliver-once）
控制权：工具返回的 Command（graph=PARENT, goto=...）
```

父图依然只有一条固定边 `builder.add_edge(START, "lead")`。所有 `lead ↔ teammate` 的跳转都由协议工具返回的 `Command` 在运行时决定；模型不调用这些工具，当前 Agent 子图自然结束。

## 协议信封

信封是一条约定的最小消息结构，投递到收件箱后由 middleware 序列化成单行 JSON 交给模型：

```json
{
    "id": "msg_3",
    "type": "handoff_request",
    "from": "lead",
    "to": "alice",
    "payload": {"request_id": "req_1", "plan": "..."},
}
```

本章定义了六个消息类型：

| 消息类型 | 方向 | 含义 |
|---|---|---|
| `handoff_request` | Lead → 队友 | 委派 + 请求审批：带上 `request_id` 和完整 `plan` |
| `handoff_response` | 队友 → Lead | 审批结果：带上同一个 `request_id` 和 `approve` |
| `result` | 队友 → Lead | 工作完成的总结 `summary` |
| `message` | 双向 | 自由格式短消息（`send_message`） |
| `shutdown_request` | Lead → 队友 | 请求队友收尾后关机 |
| `shutdown_response` | 队友 → Lead | 确认关机并把控制权交还 |

`request_id` 是贯穿全链路的关联键：`assign_teammate` 生成它，`handoff_request` 带着它出去，`handoff_response` 带着它回来，`PENDING_REQUESTS` 用它在内存里把状态从 `pending` 翻转到 `approved/rejected`。

## 邮箱：投递与消费

```python
MAILBOXES: dict[str, list[dict[str, Any]]] = {"lead": []}
MAILBOX_LOCK = RLock()
```

- `post_envelope(from_agent, to_agent, msg_type, payload)`：生成信封并追加到 `to_agent` 的收件箱；
- `consume_inbox(agent)`：取出并清空该 Agent 的收件箱，保证**每条只交付一次**（deliver-once）。

Lead 的收件箱固定叫 `"lead"`，队友的收件箱用队友名字（如 `"alice"`）。这里用进程内 `dict` + `RLock`，等价于参考仓库的文件收件箱，但更轻；数据不会跨进程持久化。

## 请求状态：PENDING_REQUESTS

`assign_teammate` 创建请求时登记一条记录：

```python
PENDING_REQUESTS[request_id] = {
    "from": "lead",
    "to": "alice",
    "plan": clean_plan,
    "status": "pending",
}
```

`respond_handoff` 收到响应时按 `request_id` 找到这条记录，更新 `status` 为 `approved` 或 `rejected`；如果 `request_id` 不存在则直接抛错。这就是参考实现里 `ProtocolState` + `match_response` 类型的简化版：它既做关联，又做“响应必须对应已知请求”的校验。

## 收件箱中间件：ProtocolInboxMiddleware

真正的“检查收件箱”动作不写成显式函数调用，而是挂在 middleware 上——等价于参考主循环里那句 “Check inbox for protocol messages”：

```python
class ProtocolInboxMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        agent = "lead" if self.kind == "lead" else state.get("teammate_name", "teammate")
        envelopes = consume_inbox(agent)
        if not envelopes:
            return None
        return {"messages": [HumanMessage(content=envelopes_to_text(envelopes))]}
```

它在**每次模型调用之前**运行，把当前 Agent 收件箱里积压的信封注入为一条 `HumanMessage`。与 s13 的 `BackgroundNotificationMiddleware` 同构：模型既不需要知道收件箱是 dict 还是文件，也不需要自己轮询；上下文到点就会出现在它面前。

Lead 和 Teammate 各自持有一个实例（`lead_inbox`、`teammate_inbox`），因为两者读取的收件箱名不同：Lead 固定读 `"lead"`，Teammate 读自己的名字。

## 协议工具

### Lead 侧

**`assign_teammate(name, role, plan)`** 做了三件事：

1. 用 `_new_id("req")` 生成 `request_id`，登记进 `PENDING_REQUESTS`；
2. 把 `handoff_request{request_id, plan}` 投递到队友收件箱；
3. 返回 `Command(graph=PARENT, goto="teammate")`，并写入 `active_agent`、`teammate_name/role/task` 和闭合工具调用的 `AIMessage + ToolMessage`。

关键点在于第 3 步仍然遵守 s15 的规矩：handoff 工具虽然真正的效果是“切节点”，也必须补一条相同 `tool_call_id` 的 `ToolMessage`，否则消息协议不闭合。

**`shutdown_team()`** 向活跃队友投递 `shutdown_request`，再 `Command(goto="teammate")` 把控制权交给队友去处理确认。没有活跃队友时返回普通字符串，不做跳转。

### Teammate 侧

**`respond_handoff(request_id, approve)`** 是审批握手的核心。它返回**普通字符串**而不是 `Command`，因此队友保持活跃：`approve=true` 时随即开始执行计划，`approve=false` 时向 Lead 报告拒绝。`request_id` 和 `approve` 都是必需参数——缺了会抛错，这与参考合约的行为一致。

**`report_result(summary)`** 把结果投递为 `result` 信封，再 `Command(goto="lead")` 交还控制权。

**`shutdown_response()`** 认可 `shutdown_request`，投递 `shutdown_response` 信封后 `Command(goto="lead")`。

**`send_message(to, content)`** 是双向通用工具：Lead 和队友都能用它发一条自由格式 `message`。发送方身份由 `active_agent_name(state)` 解析——Lead 固定叫 `"lead"`，队友用当前 `teammate_name`。

## 工具集合

Lead 的工具：

```text
bash, read_file, write_file,
create_task, list_tasks, get_task, claim_task, complete_task,
assign_teammate, shutdown_team, send_message
```

Teammate 的工具：

```text
bash, read_file, write_file,
respond_handoff, report_result, shutdown_response, send_message
```

Teammate 没有 `assign_teammate`（不能递归创建队友），也没有共享任务管理工具（不能擅自改变 Lead 的协调计划）。协议工具按方向裁剪：`assign_teammate`、`shutdown_team` 只属于 Lead，`respond_handoff`、`report_result`、`shutdown_response` 只属于 Teammate。

## 一次完整流程

假设用户输入：

```text
让 alice 作为后端工程师创建 config.py，完成后向我汇报。
```

执行顺序：

1. 父图从 `START` 进入 Lead；
2. Lead 调用 `assign_teammate(name="alice", role="backend", plan="创建 config.py ...")`；
3. 工具登记 `PENDING_REQUESTS`、投递 `handoff_request`、`Command(goto="teammate")`；
4. Teammate 的 `ProtocolInboxMiddleware` 在模型调用前注入 `handoff_request`；
5. Teammate 读信封后调用 `respond_handoff(request_id, approve=true)`，投递 `handoff_response`；
6. Teammate 继续执行计划，用 `write_file` 创建 `config.py`；
7. Teammate 调用 `report_result(summary="创建了 config.py ...")`，投递 `result` 并 `Command(goto="lead")`；
8. Lead 的 middleware 注入 `handoff_response` 和 `result` 两条信封；
9. Lead 核验结果，必要时自行验证，再调用 `shutdown_team()` 触发关机握手；
10. Teammate 收到 `shutdown_request` 后调用 `shutdown_response()`；
11. Lead 看到 `shutdown_response` 后向用户给出最终回答。

两段握手（审批、关机）都走“请求 → 响应 → 状态翻转”，每步都有 `request_id` 可追溯。

## 相对 s15 的变化

| 组件 | s15 | s16 |
|---|---|---|
| 协调方式 | 松散文本 + XML 标签 | 结构化 JSON 信封 |
| 请求追踪 | 无 | `PENDING_REQUESTS` + `request_id` |
| 消息通道 | 直接写共享历史 | 每个 Agent 自己的收件箱 |
| 审批 | 无 | `handoff_request/response` + `approve` |
| 关机 | 无 | `shutdown_request/response` 握手 |
| 新消息类型 | — | `handoff_*`、`result`、`message`、`shutdown_*` |
| 收件箱注入 | 仅 `BackgroundNotificationMiddleware` | 新增 `ProtocolInboxMiddleware` |
| Lead 工具 | 9（原生能力） | 11（新增 `shutdown_team`、`send_message`） |
| Teammate 工具 | 4 | 7（新增 `respond_handoff`、`shutdown_response`、`send_message`） |

## 试一下

```sh
cd learn-claude-code
python -m legacy.s16_team_protocols.code
```

试试这些 prompt：

1. `让 alice 作为后端工程师创建 config.py，完成后向我汇报。`
2. `把 s16 的 README 架构分析交给一名 LangChain 专家，收到汇报后你再复核它。`

观察终端里的信号：

- `[handoff] lead -> alice handoff_request`：Lead 投递请求；
- `[alice inbox] 注入 1 条协议信封`：middleware 把信封交给模型；
- `[protocol] alice -> lead handoff_response ... approve=true`：队友审批；
- `[protocol] alice -> lead result`：队友汇报；
- `[protocol] lead -> alice shutdown_request` / `alice -> lead shutdown_response`：关机握手。

重点观察：信封里的 `request_id` 在请求与响应之间是否一致？`PENDING_REQUESTS` 的状态是否从 `pending` 翻转到 `approved`？关机握手是否完整？

## 教学版边界

为了让协议机制保持可读，本章有意省略或简化以下能力：

- **无真实线程 / 无 idle loop**：队友“保持活跃”是靠在当前子图节点里继续跑模型循环实现的，不是后台线程轮询；`respond_handoff` 返回普通字符串即停留在当前节点，`report_result` / `shutdown_response` 返回 `Command` 才跳回 Lead。
- **无执行门控**：`approve` 是消息级的握手，靠 system prompt 约束模型“先批准再执行”，并没有在 `bash/write_file` 工具层拦截未批准的操作。
- **单活跃队友**：同一时刻只有一个 `teammate_*` 身份，新委派会覆盖上一组元数据；
- **进程内邮箱**：`MAILBOXES` 是内存 dict，进程重启即丢失；
- **共享工作目录无隔离**：多队友并行前需要文件锁或 worktree；
- **没有自动清理**：教学版关机只发消息，不清理 pane、任务或团队配置（那是真实 CC 的行为）。

## 接下来

s15-s16 里，Lead 必须给每个队友分配任务。能不能让队友自己看板、自己认领？Lead 只需要创建任务，队友自己发现、自己认领、自己完成。

[s17: Autonomous Agents](../s17_autonomous_agents/) → 队友自组织，不需要领导分配。



## 本章文件

- `code.py`：带注释教学版（可直接运行）；
- `code_uncommented.py`：去掉教学行注释的完整版本，适合快速通读和自行练习。

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
python -m legacy.s16_team_protocols.code
python -m legacy.s16_team_protocols.code_uncommented
```

也支持直接运行文件：

```powershell
python legacy/s16_team_protocols/code.py
```

## 验证代码结构

不调用模型也可以先做静态验证：

```powershell
python -m py_compile legacy/s16_team_protocols/code.py
python -m py_compile legacy/s16_team_protocols/code_uncommented.py

python -c "import legacy.s16_team_protocols.code as c; print(type(c.team_graph).__name__)"
```

预期最后输出 `CompiledStateGraph`。

## 参考资料

- [参考仓库 s13：Agent Teams（团队协议已并入该章）](https://github.com/shareAI-lab/learn-claude-code/tree/main/s13_agent_teams)
- [LangChain 官方文档：Multi-agent patterns](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [LangChain 官方文档：Handoffs](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs)
- [LangChain 官方文档：Middleware](https://docs.langchain.com/oss/python/langchain/middleware/overview)
- [LangGraph API：Command](https://reference.langchain.com/python/langgraph/types/Command)

<!-- upstream-cc-source:start -->
## 深入 CC 源码

> 原文：[s16_team_protocols](https://github.com/shareAI-lab/learn-claude-code/blob/67a9126c6435a8654ba7a6f68c0fd2130f00a462/s16_team_protocols/README.md)。以下折叠块保持原文，文中的章号与源码行号沿用该版本。

<details>
<summary>深入 CC 源码</summary>

CC 的团队协议实现（`teammateMailbox.ts`，1184 行）和教学版在核心结构上一致：request_id + approve/reject 的请求-响应模式。差异在于：

**关机协议**：CC 的 shutdown 是三向通信（`teammateMailbox.ts:720-763`、`SendMessageTool.ts:268-430`）。Lead 发 `shutdown_request`，队友回复 `shutdown_approved`（或 `shutdown_rejected` 附原因），系统发送 `teammate_terminated` 通知所有相关方。关机确认后系统自动清理 pane（tmux/iTerm2）、unassign 任务、从 team config 移除成员（`useInboxPoller.ts:677-800`）。教学版用 `shutdown_response` 统一命名，真实源码拆成 approved/rejected 两种独立消息。

**计划审批**：真实源码里 plan approval request 由 `ExitPlanModeV2Tool.ts:263-312` 在 plan-mode-required 队友退出 plan mode 时产生。`useInboxPoller.ts:599-661` 当前会自动回写 approval，并把请求交给 Lead 作为上下文（regular message）。`SendMessageTool.ts:434-518` 仍保留显式 approve/reject response 能力，审批时可同时设置 `permissionMode`（如"批准但以 plan mode 运行"），响应中可包含 `feedback` 字符串供队友修正后重新提交。不是简单的"Lead 手动 review_plan 工具"流程。

**消息格式**：CC 的协议消息是结构化的 JSON（有 Zod schema 验证），教学版用简单的 type + metadata 字典。字段名也不统一：permission 用 `request_id`（`teammateMailbox.ts:453-462`），shutdown 和 plan approval 用 `requestId`（`teammateMailbox.ts:684-763`）。

**执行门控**：CC 的队友有完整的 permission gating。未获批准的高风险操作会被拦截，不是可选的。教学版只演示了消息流程，没有实现执行拦截。

**通用性**：教学版的一个 FSM（pending → approved | rejected）对应两种协议，这个简化完全正确。CC 的所有协议消息共用同一个 request id 关联机制。

</details>

<!-- upstream-cc-source:end -->
