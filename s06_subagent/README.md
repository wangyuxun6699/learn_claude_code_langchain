# s06: Subagent — 给子任务一段独立上下文

> **对齐状态**：本章 `code.py` 对齐上游 `s06_subagent` 的结构；模型适配与本章机制在 `code.py` 中直接实现，使用 LangChain OpenAI-compatible 调用。
[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md)

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

## 本项目保留的 LangChain / LangGraph 教学补充

> 以下内容来自本仓库对齐前的 README，作为上游课程之外的本地教学补充完整保留。

<!-- local-langchain-additions:start -->
<details>
<summary>展开本仓库原有的 LangChain / LangGraph 教学说明</summary>

# s06: Subagent — 分而治之

> LangChain 教学改编版。章节结构与“深入 CC 源码”部分主要参考 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)。
>
> **Harness 层**：上下文隔离 — 父 Agent 委派，子 Agent 独立执行。

[s05](../s05_todo_write/) → **s06** → [s07](../s07_skill_loading/)

---

## 问题

复杂子任务会占用主会话大量 token，也可能把探索噪声带回主 Agent。

---

## 解决方案

![s06: Subagent — 分而治之](images/subagent-overview.svg)

创建一个只拥有基础工具的 `SUB_AGENT`，再把它包装成父 Agent 的 `task` 工具。子 Agent 看不到父对话，只返回最终摘要。

---

## 工作原理：LangChain 版本

```python
SUB_AGENT = create_agent(
    model=MODEL,
    tools=BASE_TOOLS,
    system_prompt=SUB_SYSTEM,
    middleware=[tool_hook],
    name="worker",
)

@tool("task")
def task(description: str) -> str:
    """Launch an isolated subagent and return only its final conclusion."""
    result = SUB_AGENT.invoke(
        {"messages": [{"role": "user", "content": description}]},
        config={"recursion_limit": 64},
    )
    return extract_final_text(result.get("messages", []))

PARENT_AGENT = create_agent(
    model=MODEL, tools=[*BASE_TOOLS, task],
    system_prompt=PARENT_SYSTEM, middleware=PARENT_MIDDLEWARE,
)
```

`ContextVar` 标记当前执行属于 parent 还是 sub，使 Hook 日志与权限提示保持可辨识。

---

## 本章文件

- `code.py`：带注释教学版（可直接运行）。
- `code_uncommented.py`：无教学注释的精简版。

---

## 试一下

先在仓库根目录准备环境，然后从根目录按模块运行：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m s06_subagent.code
```

> 这些教学 Agent 可以执行命令和修改文件。建议先在测试目录中试用，并认真阅读每次权限提示。

---

## 接下来

s07 把领域说明放进 Skill，先发现目录，命中时再加载正文。

</details>
<!-- local-langchain-additions:end -->

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
