# s01: Agent Loop — 一个循环就够了

`s01` → [s02](../s02_tool_use/) → s03 → s04 → ... → s16 → s17
> *"One loop & Bash is all you need"* — 一个工具 + 一个循环 = 一个 Agent。
>
> **Harness 层**: 循环 — 模型与真实世界的第一道连接。

---

## 问题

你提出了一个问题给大模型：“帮我读取下我的目录下有哪些文件，并且执行XXX.py”。

模型能输出一条 bash 命令，但输出完了就停了，它不会自己跑，也不会看到结果后继续推理。

你可以手动跑一遍，把输出粘贴回对话框，让它接着干。下一个命令出来，你再跑一遍、再贴回去。

每一个来回，你都在做中间层。而把它自动化，就是这一章要做的事。

---

## 解决方案

![Agent Loop](images/agent-loop.svg)

一个 `while True` 循环，模型调用工具就继续，不调用就停。循环直接检查响应里的内容块：

| 信号 | 含义 | 循环动作 |
|------|------|---------|
| 包含 `tool_use` block | 模型要求调用工具 | 执行 → 结果喂回去 → 继续 |
| 不包含 `tool_use` block | 模型没有调用工具 | 退出循环 |

---

## 工作原理

将这个过程翻译成代码。分步来看：

**第 1 步**：把用户的问题作为第一条消息。

```python
messages = [{"role": "user", "content": query}]
```

**第 2 步**：将消息和工具定义一起发给 LLM。

```python
response = client.messages.create(
    model=MODEL, system=SYSTEM, messages=messages,
    tools=TOOLS, max_tokens=8000,
)
```

**第 3 步**：追加模型回答，检查它是否调了工具。没调 → 结束。

```python
messages.append({"role": "assistant", "content": response.content})
tool_calls = [
    block for block in response.content if block.type == "tool_use"
]
if not tool_calls:
    return
```

只有实际存在的 `tool_use` block 才会进入执行阶段，因此不会追加空的工具结果消息。

**第 4 步**：执行模型要求的工具，收集结果。

```python
results = []
for block in tool_calls:
    output = run_bash(block.input["command"])
    results.append({
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": output,
    })
```

**第 5 步**：把工具结果作为新消息追加，回到第 2 步。

```python
messages.append({"role": "user", "content": results})
```

组装为一个完整函数：

```python
def agent_loop(messages):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        if not tool_calls:
            return

        results = []
        for block in tool_calls:
            output = run_bash(block.input["command"])
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        messages.append({"role": "user", "content": results})
```

三十多行，这就是最小可运行的 agent harness 内核。它为模型提供持续行动的最小运行框架：模型负责决策（要不要调工具、调哪个），harness 负责执行（调用工具，把结果作为新消息追加）。后面 16 个章节都在这个循环上叠加机制，循环本身始终不变。

---

## 结合 `code.py` 理解 LangChain / LangGraph

本章没有调用 `create_agent()`，也没有创建 `StateGraph`。这是刻意的：先把框架通常替你完成的工作摊开，才能看清 Agent 的最小闭环。

### 模型边界实际做了什么

从 [`code.py`](code.py) 的 `_MessagesAPI.create()` 开始读，调用链是：

1. `ChatOpenAI(...)` 从 `MODEL_ID`、`OPENAI_API_KEY` 和 `BASE_URL` 创建 OpenAI-compatible 聊天模型。
2. `_openai_tools()` 把课程使用的 `name / description / input_schema` 转成 OpenAI function schema。
3. `llm.bind_tools(openai_tools)` 把 `bash` 的 schema 绑定给模型。绑定只表示“允许模型请求这个工具”，不会自动执行命令。
4. `_to_langchain_messages()` 把课程消息转换为 `SystemMessage`、`HumanMessage`、`AIMessage` 和 `ToolMessage`。
5. `runnable.invoke(request)` 发起一次模型调用；返回的 `AIMessage.tool_calls` 再被转换成课程循环读取的 `ToolUseBlock`。
6. `usage_metadata` 被整理为 `Usage`，供后续章节做预算、恢复和 Goal 统计。

这里最重要的协议约束是 `tool_call_id`：模型产生的调用 ID 会进入 `ToolUseBlock.id`，执行结果必须用同一个 ID 构造 `ToolMessage`。如果 ID 丢失，模型就无法判断结果属于哪次调用。

### 手写循环与 LangChain Agent 的对应关系

| 本章代码 | LangChain / LangGraph 中的抽象 |
|---|---|
| `messages` 列表 | Agent state 中的 `messages` 通道 |
| `client.messages.create()` | 模型节点（model node） |
| 查找 `tool_use` | 根据 `AIMessage.tool_calls` 进行条件路由 |
| `run_bash()` | 工具执行节点 |
| 追加 `tool_result` 后继续 | 从工具节点回到模型节点 |
| 没有工具调用时 `return` | 路由到 `END` |

`create_agent()` 会提供同一种“模型 → 工具 → 模型”的循环，并运行在 LangGraph 之上。本章保留显式 `while True`，因此消息何时追加、工具何时执行、何时退出都能在一个函数里观察到。若改写成 `StateGraph`，通常会建立 `model` 与 `tools` 两个节点，再用条件边判断继续还是结束；语义没有变化，只是状态、流式输出和持久化交给图运行时管理。

### 阅读和调试建议

- 在 `_MessagesAPI.create()` 后观察 `raw.tool_calls`，确认提供商返回了标准工具调用。
- 在 `agent_loop()` 中观察 `messages[-2:]`，理解 `AIMessage` 与 `ToolMessage` 必须成对出现。
- 尝试让模型一次请求两个命令。本章会按列表顺序串行执行，说明“模型可生成并行工具调用”不等于“宿主已经并行调度”。
- 不要把 `stop_reason` 当作唯一依据；本项目最终以是否真实存在 `tool_use` 内容块决定是否继续。

官方概念：[Models 与 `bind_tools`](https://docs.langchain.com/oss/python/langchain/models) · [Tools](https://docs.langchain.com/oss/python/langchain/tools) · [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview)

---

## 试一下

> **安全提示**：代码会执行模型生成的 shell 命令。建议在一个临时测试目录中运行，避免影响你的项目文件。s03 会加入权限控制。

**准备**（首次运行）：

```sh
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY、BASE_URL 和 MODEL_ID
```

**运行**：

```sh
python s01_agent_loop/code.py
```

试试这些 prompt：

1. `Create a file called hello.py that prints "Hello, World!"`
2. `List all Python files in this directory`
3. `What is the current git branch?`

观察重点：模型什么时候调用工具（循环继续），什么时候不调用（循环结束）？

---

## 接下来

现在模型手里只有 bash 一个工具，读文件要 `cat`，写文件要 `echo ... >`，找个文件要 `find`，又丑又容易出错。

s02 Tool Use → 给它 5 个真正的工具，会发生什么？模型会不会一次调用多个工具？几个工具同时跑会不会互相踩？
---

<!-- upstream-cc-source:start -->
## 深入 CC 源码

> 原文：[s01_agent_loop](https://github.com/shareAI-lab/learn-claude-code/blob/67a9126c6435a8654ba7a6f68c0fd2130f00a462/s01_agent_loop/README.md)。以下折叠块保持原文，文中的章号与源码行号沿用该版本。

<details>
<summary>深入 CC 源码</summary>

> 以下内容基于 CC 源码 `src/query.ts`（1729 行）的核查。核心差异就两个：CC 不看 `stop_reason` 字段而是检查内容里有没有 tool_use 块（因为流式响应中 stop_reason 不可靠）；CC 有更多的退出路径和恢复策略做生产级保护。

**教学版的 30 行 `while True` 就是 CC 1729 行的核心。** 下面每一项都是在这个核心上叠加的保护机制。

<details>
<summary>一、循环结构差异</summary>

教学版检查 `response.stop_reason`。CC 不把它作为循环继续的唯一依据——流式响应中 `stop_reason` 可能还没更新但内容里已经有 `tool_use` 块了。CC 用 `needsFollowUp` 标志：接收到流式消息时（`query.ts:830-834`），只要检测到 `tool_use` 块就设为 `true`；`QueryEngine.ts` 会从 `message_delta` 捕获真实 `stop_reason` 用于其他逻辑，但 query loop 本身靠 `needsFollowUp` 决定是否继续。

```typescript
// query.ts:554-558
// stop_reason === 'tool_use' is unreliable.
// Set during streaming whenever a tool_use block arrives.
let needsFollowUp = false
```

</details>

<details>
<summary>二、State 对象 10 字段（教学版只用 messages）</summary>

| # | 字段 | 用途 | 对应章节 |
|---|------|------|---------|
| 1 | `messages` | 当前迭代的消息数组 | s01 |
| 2 | `toolUseContext` | 工具、信号、权限上下文 | s02 |
| 3 | `autoCompactTracking` | 压缩状态追踪 | s08 |
| 4 | `maxOutputTokensRecoveryCount` | token 恢复尝试次数（上限 3） | s11 |
| 5 | `hasAttemptedReactiveCompact` | 本轮是否已尝试响应式压缩 | s08 |
| 6 | `maxOutputTokensOverride` | 8K→64K 的升级覆盖 | s11 |
| 7 | `pendingToolUseSummary` | 后台 Haiku 生成的 tool use 摘要 | s08 |
| 8 | `stopHookActive` | 停止钩子是否产生阻塞错误 | s04 |
| 9 | `turnCount` | 轮次计数（maxTurns 检查） | s01 |
| 10 | `transition` | 上一次继续原因 | s11 |

> 注：`taskBudgetRemaining`（`query.ts:291`）是 loop-local 局部变量，不在 State 上。源码注释明确写了 "Loop-local (not on State)"。

</details>

<details>
<summary>三、多条退出和继续路径</summary>

教学版只有 1 条退出路径（模型不调工具就结束）。生产版有多条退出和继续路径，覆盖 blocking limit、prompt too long、model error、abort、hook stop、max turns、token budget continuation、reactive compact retry 等场景。每种场景都有对应的恢复或退出策略。

</details>

<details>
<summary>四、流式工具执行和 QueryEngine</summary>

CC 的 `StreamingToolExecutor`（`query.ts:561`）让工具在模型还在生成时就开始并行执行（根据工具是否 concurrency-safe 决定并发或独占）。`QueryEngine.ts` 额外加了费用超限、结构化输出验证失败等保护。教学版不实现这些——目标是概念清晰，不是性能极致。

</details>

**一句话**：1729 行的 query.ts 核心就是 30 行 `while True`。所有复杂字段和退出路径都是保护机制。先理解核心循环，后面的一切自然展开。

</details>

<!-- upstream-cc-source:end -->
