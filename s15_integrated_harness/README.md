# s15: Integrated Harness — 很多机制，一个循环

> LangChain / LangGraph 教学改编版。章节结构参考
> [shareAI-lab/learn-claude-code 的 s15](https://github.com/shareAI-lab/learn-claude-code/blob/main/s15_integrated_harness/README.md)。
>
> *"Many mechanisms, one loop"* — 工具、权限、记忆、任务、团队、插件全部挂在同一条循环上。
>
> **Harness 层**：集成 — 把前面章节的机制放进一个可运行的系统。

[s14](../s14_mcp_plugin/) → **s15** → [s16](../s16_workflow_runtime/)

---

## 问题

前几章把机制都放在各自独立的脚本里。这一章把它们接成一个集成 runtime。一个长期运行的 coding agent 需要同时具备：工具分发与权限边界、Hook 扩展点、规划与任务图、skills / 记忆 / system prompt 组装、压缩与错误恢复、后台任务与 cron、团队与 worktree、MCP 外部工具。

s15 不引入新机制，而是展示这些机制**在模型循环里的接入位置**，以及它们的事件如何回到同一条对话。

---

## 解决方案

```text
用户输入
  → UserPromptSubmit hook
  → 后台/cron 通知注入
  → system prompt 组装（记忆 + skills + MCP 状态）
  → LLM（bind_tools(assemble_tool_pool())，带恢复重试）
  → 是否还有 tool_calls？
      否 → Stop hook → 返回
      是 → PreToolUse hook + 权限
           → 内置工具 / MCP 工具 / 后台分发
           → PostToolUse hook
           → tool_result / task_notification 回到 messages
           → 下一轮
```

循环本身和 s01 一样：调模型、看有没有工具调用、执行工具、把结果追加回 messages。有没有 `tool_calls` 决定工具执行是否继续。

---

## 各机制在 LangChain 版本里的位置

| 位置 | 组件 | 作用 |
|---|---|---|
| 用户输入前后 | UserPromptSubmit hook | 打印/审计 |
| 模型调用前 | 后台/定时通知 | 把 `<task_notification>` / `<cron reminder>` 注入 messages |
| 模型调用前 | 记忆 / skills / MCP 状态 | 组装 system prompt |
| 模型调用 | 错误恢复 | 指数退避重试，上下文过长则裁剪重试 |
| 工具执行前 | PreToolUse + 权限 | 拦截危险命令、越界写入、破坏性 MCP 工具 |
| 工具分发 | assemble_tool_pool | 内置工具 + 动态 MCP 工具 |
| 工具执行中 | 后台分发 | `run_in_background=true` 的命令进守护线程，返回占位结果 |
| 工具执行后 | PostToolUse | 日志 |
| 回到循环 | tool_result | 每个 tool_use 一个 ToolMessage，进入下一轮 |

> 说明：本章把团队（s13）与 worktree 的完整协议保留在 s13，未在此重复；后台 bash 与 cron 用简化版演示"注入式唤醒"。完整 5 字段 cron 解析、任务绑定 worktree 的多 agent 协作请回到 s12 / s13。

---

## 本章包含的工具

```
run_bash (run_in_background 可选)   run_read   run_write   run_edit   run_glob
todo_write                          load_skill
task (一次性子 agent)                connect_mcp (+ 动态 mcp__server__tool)
schedule_cron   list_crons   cancel_cron
```

### 关键机制

- **动态工具池**：与 s14 相同，`assemble_tool_pool()` 每轮组装内置 + MCP 工具；因为要注入通知并且工具在运行时变化，主循环用 `MODEL.bind_tools(tools)` 手写（同 s14 的手写 LangGraph 思路）。
- **一次性子 agent**：`task(prompt)` 用 `create_agent`（只读 run_read/run_glob）在隔离上下文里调查，只把最终摘要带回。
- **后台命令**：`run_bash(command, run_in_background=True)` 在守护线程里执行，完成后把通知放进队列，由 `inject_pending()` 注入消息流。
- **cron**：`schedule_cron(delay_seconds, prompt)` 是一次性相对延迟；守护线程每秒扫描到期任务放进队列。
- **记忆**：`.memory/MEMORY.md` 作为长期记忆目录，`memory_section()` 做关键词相关性选择后注入 system prompt。
- **skills**：`skills_catalog()` 只给目录摘要，`load_skill(name)` 按需加载全文。
- **压缩**：`compact_messages()` 截断过长工具结果，并给累计工具结果设总预算。
- **恢复**：`call_model_with_recovery()` 对临时错误指数退避，对上下文超长裁剪重试。

---

## 运行

```sh
cd learn-claude-code
python -m s15_integrated_harness.code
```

可试：

1. `Inspect this repository and tell me which Python files matter most.`
2. `Search the connected documentation for agent loop guidance.`（先让它 connect_mcp("docs")）
3. `Install the dependencies in the background while you read README.md.`
4. `Remind me to stand up in 30 seconds.`（schedule_cron）

观察：

- 每个工具调用是否经过 Hook / 权限；
- connect_mcp 后下一轮是否出现 `mcp__docs__search`；
- `run_in_background=true` 是否返回占位、稍后注入通知；
- cron 到点是否把 reminder 注入消息流；
- 记忆/skills/MCP 状态是否进入 system prompt。

---

## 与 s14 的对比

| 范围 | s14 MCP | s15 Integrated Harness |
|---|---|---|
| 内置工具 | 6 | 12（含 MCP 时动态增加） |
| 外部工具 | 连接后的 MCP 工具 | 同 s14 的动态 MCP + 主机策略 |
| 本地机制 | s04 工具、Hook、权限、MCP | 加 todo、skills、记忆、子 agent、后台、cron |
| 事件来源 | 用户输入 + 工具结果 | 用户输入、工具结果、后台通知、cron 提示 |

---

## 下一步

[s16 Workflow Runtime](../s16_workflow_runtime/) 给这个 host 加一个 `Workflow` 工具：把固定编排路径写进代码，用 journal 记录进度，中断后能续跑。
