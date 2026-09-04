# s06: Subagent — 给子任务一段独立上下文

s01 → s02 → s03 → s04 → s05 → `s06` → [s07](../s07_skill_loading/) → s08 → ... → s16 → s17

> Subagent 从全新的 `messages[]` 开始。最终文本返回父循环，中间对话不会进入父上下文。
>
> **Harness 层**: 委派 — 在另一段对话上下文中处理一个明确的子任务。

---

## 问题

Agent 在修一个 bug。为了追踪调用链，它读取了许多文件；每次工具调用和结果都会留在父循环的 `messages[]` 中。调用链已经弄清以后，多数中间细节不再需要，却仍然占用上下文。

---

## 解决方案

![Subagent Overview](images/subagent-overview.svg)

调用 `task` 时，会同步运行一个使用全新 `messages[]` 的嵌套 Agent Loop。循环结束后，它的最终文本会成为父对话中的工具结果。

这里隔离的是消息，不是进程或文件系统。父 Agent 与子 Agent 共享 `WORKDIR`，写文件和命令仍会影响同一个工作区。子 Agent 拥有五个基础工具，但没有 `task`；它的工具调用与父 Agent 使用同一组权限和生命周期 Hooks。

---

## 工作原理

**run_subagent** 创建新的消息列表，运行嵌套循环，并返回最终文本：

```python
SUB_TOOLS = list(BASE_TOOLS)  # no task tool

def run_subagent(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]

    for _ in range(30):
        response = client.messages.create(
            model=MODEL, system=SUB_SYSTEM,
            messages=messages, tools=SUB_TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        if not tool_calls:
            return extract_text(response.content) or "(no summary)"

        results = []
        for block in tool_calls:
            output = execute_tool(block, SUB_HANDLERS)
            results.append({... "content": output})
        messages.append({"role": "user", "content": results})

    return "Subagent stopped after 30 turns without a final answer."
```

主 Agent 调用时，跟调其他工具一样：

```python
TASK_TOOL = {
    "name": "task",
    "description": "Run a subagent with fresh conversation context and return its final text.",
    "input_schema": {
        "type": "object",
        "properties": {"prompt": {"type": "string"}},
        "required": ["prompt"],
    },
}

TOOLS = [*BASE_TOOLS, TASK_TOOL]
TOOL_HANDLERS = {**BASE_HANDLERS, "task": run_subagent}
```

实际边界如下：

| 决策 | 选择 | 原因 |
|------|------|------|
| 对话 | 全新的 `messages[]` | 不把父对话复制给子 Agent |
| 执行 | 同一进程和 `WORKDIR` | 两个循环都能看到文件系统修改 |
| 返回值 | 只返回最终文本 | 子 Agent 的工具调用和结果不进入父消息列表 |
| 委派深度 | `SUB_TOOLS` 中没有 `task` | 本章只允许一层委派 |
| 工具策略 | 共享 Hooks | 父子循环使用相同的权限检查 |

父 Agent 与其他工具一样，通过 handler map 分发 `task`。子 Agent 使用 `SUB_SYSTEM`、`SUB_TOOLS` 和自己的局部 `messages` 列表。

---

## 结合本章代码理解 Subagent

本章采用最典型的 supervisor 模式：父 Agent 把子任务包装成一个名为 `task` 的工具。模型调用它时，`run_subagent(prompt)` 启动另一条独立 Agent loop，最后只把文本总结返回父上下文。

### 上下文隔离在代码中如何发生

[`code.py`](code.py) 的关键不是另一个 `while`，而是这行：

```python
messages = [{"role": "user", "content": prompt}]
```

子 Agent 不复制父 Agent 的完整历史，只收到父 Agent 为它整理的 prompt；它有独立的 `SUB_SYSTEM`，只能使用 `SUB_TOOLS`，其中没有 `task` 工具，因此不能递归创建更多子 Agent。循环最多执行 30 轮，最终文本通过普通工具结果回到父 Agent。

这种隔离同时解决两个问题：

- 父上下文不会塞入子任务的每条命令和文件输出。
- 子 Agent 不会被父对话里的无关讨论干扰，但父 Agent 必须提供足够完整的任务说明。

文件和 shell 工具仍经过 s04 的 `PreToolUse` / `PostToolUse` 边界，所以“上下文隔离”不等于“权限绕过”。

### 与 LangChain subagents 的对应关系

LangChain 官方的 subagent 模式同样把子 Agent 暴露为主 Agent 的工具：主 Agent 负责选择子 Agent、构造输入并整合结果。常见实现是对子 Agent 调用 `create_agent(...).invoke()`，再把最后一条消息作为工具返回值。

| 本章实现 | 框架化实现 |
|---|---|
| `run_subagent()` 内手写循环 | 子 `create_agent()` 或 LangGraph subgraph |
| `TASK_TOOL` | `@tool` 包装的子 Agent 调用 |
| fresh `messages` | per-invocation subgraph / 无共享 thread state |
| 最终文本返回父循环 | ToolMessage 返回 supervisor |
| 30 轮硬上限 | model/tool call limit middleware 或 recursion limit |

如果多个子 Agent 需要并行执行，可以让模型一次产生多个工具调用，并由宿主异步调度；如果子 Agent 需要暂停等待用户，则应使用带 checkpointer 的 subgraph 和 `interrupt()`，而不是在子线程里直接读 stdin。

### 设计子 Agent 工具描述

父模型只根据工具名、描述和参数决定何时委派，因此工具描述就是路由策略。应说明子 Agent 擅长什么、需要什么输入、返回什么结果；prompt 还应包含目标、范围、可用文件和完成标准。上下文隔离的收益来自“有选择地传递”，不是简单地把信息全部删掉。

官方概念：[Subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents) · [Context engineering](https://docs.langchain.com/oss/python/langchain/context-engineering)

---

## 试一下

```sh
cd learn-claude-code
python s06_subagent/code.py
```

试试这些 prompt：

1. `Use a subtask to find what testing framework this project uses`（子 Agent 去读文件，主 Agent 只收结论）
2. `Delegate: read all .py files in agents/ and summarize what each one does`
3. `Use a task to create s06_subagent/example/string_tools.py with a slugify(text: str) function, then verify it from the parent agent`

观察重点：是否出现 `[Subagent started]` / `[Subagent done]`？子 Agent 的工具调用是否以 `[sub] ...` 输出？父 Agent 是否只接收到 `task` 返回的最终文本？

---

## 接下来

Agent 现在能拆任务了。但每个任务需要的知识不一样：改前端组件需要知道 React 规范，写 SQL 需要知道表结构。这些知识全塞进 system prompt，上下文直接爆了。

s07 Skill Loading → 技能按需注入，不在 system prompt 里堆文档。用到的时候才加载，和读文件一样自然。
---

<!-- upstream-cc-source:start -->
## 深入 CC 源码

> 原文：[s06_subagent](https://github.com/shareAI-lab/learn-claude-code/blob/67a9126c6435a8654ba7a6f68c0fd2130f00a462/s06_subagent/README.md)。以下折叠块保持原文，文中的章号与源码行号沿用该版本。

<details>
<summary>深入 CC 源码</summary>

> 以下基于 CC 源码 `AgentTool.tsx`、`runAgent.ts`、`forkSubagent.ts`、`forkedAgent.ts` 的完整分析。

### 一、不是一种模式，是三种

教学版只讲了"全新的 messages[]"。CC 实际有三种执行模式：

| 模式 | 触发条件 | 上下文 |
|------|---------|--------|
| **Normal Subagent** | 指定了 `subagent_type`（normal path） | 全新 messages[]，只有 prompt |
| **Fork Subagent** | 没指定 `subagent_type`，fork gate 开启 | 通过 `buildForkedMessages()` 构造 cache-friendly 前缀，共享 prompt cache |
| **General-Purpose** | 没指定 `subagent_type`，fork gate 关闭 | 同 Normal |

### 二、Fork 模式：为了共享 Prompt Cache

这是教学版没有的核心概念。Fork 模式（`forkSubagent.ts:60-71`）不创建全新上下文，而是通过 `buildForkedMessages()`（`forkSubagent.ts:107-168`）构造 cache-friendly 消息前缀，保留父 assistant message 并生成 placeholder tool results。目的不是隔离，而是让 Anthropic API 的 prompt cache 命中：父子 Agent 的 system prompt、tools、messages 前缀完全一致，API 端不需要重算。

缓存命中的五个关键组件（`forkedAgent.ts:57-68`）：system prompt、tools、model、messages 前缀、thinking config，必须字节级一致。

### 三、Context Isolation 的精确粒度

`createSubagentContext()`（`forkedAgent.ts:345-462`）创建子 Agent 的 `ToolUseContext`：

| 字段 | 行为 |
|------|------|
| `abortController` | 新的 child controller，父 abort 向下传播 |
| `setAppState` | 默认 no-op；但 sync agent 通过 `shareSetAppState` 共享（`runAgent.ts:697-714`） |
| `readFileState` | **从父克隆**（避免重复读相同文件） |
| `queryTracking` | 新 chainId，`depth = parentDepth + 1` |

子 Agent 不是完全隔离的：文件读取状态是共享的。UI 和通知的隔离程度取决于执行路径（sync/async/fork/teammate 各不同）。

### 四、递归 Fork 防护

教学版用"子 Agent 不给 task 工具"表达递归保护。真实实现更精细：`isInForkChild()`（`forkSubagent.ts:78-89`）检查对话历史中是否有 `FORK_BOILERPLATE_TAG`，有就拒绝。但 `constants/tools.ts:36-46` 中 `Agent` 工具默认在所有 agent 的禁用集合里，`USER_TYPE === 'ant'` 时例外；`forkSubagent.ts:73-89` 针对 fork child 有专门的递归保护；`agentToolUtils.ts:100-110` 在 teammate 场景下有特殊放行。不是简单的"禁止新的子 Agent"。

### 五、Permission Bubbling

Fork Agent 的 `permissionMode: 'bubble'`（`forkSubagent.ts:67`）意味着子 Agent 的权限弹窗冒泡到父终端，用户在主终端里审批子 Agent 的操作。

### 六、Async vs Sync

教学版只展示了同步子 Agent（父等着子跑完）。CC 还支持异步路径（`AgentTool.tsx:686-764`）：`run_in_background: true` 时异步启动，返回 `{ status: 'async_launched' }` 立即给父 Agent，子 Agent 完成后通过通知机制告知父 Agent。实际触发条件不止 `run_in_background`，还有 auto-background、assistant force async、coordinator/proactive 等路径。

### 教学版的简化是刻意的

- 三种模式 → 一种（fresh messages）：概念清晰
- Prompt cache 共享 → 省略：教学版不涉及 API 层优化
- 递归 fork 防护 → 简化为"子 Agent 无 task 工具"
- Async → 省略（留给 s13）：s06 先理解同步模型

</details>

<!-- upstream-cc-source:end -->
