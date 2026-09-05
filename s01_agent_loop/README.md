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

## 解决方案：用 `create_agent` 创建智能体

![Agent Loop](images/agent-loop.svg)

参考项目用一个显式 `while True` 实现“模型 → 工具 → 模型”的循环。本章使用 LangChain 的 `create_agent()` 创建同等功能的智能体：模型调用工具时继续运行，没有工具调用时返回最终回答。

| 参考实现 | 本章的 LangChain 实现 |
|---|---|
| 手工维护 `messages` | Agent state 自动维护消息 |
| 手工检查 `tool_use` | `create_agent` 根据 `AIMessage.tool_calls` 路由 |
| 手工执行并回填 `tool_result` | 工具节点执行并生成 `ToolMessage` |
| `while True` 控制继续或结束 | LangGraph 执行图自动循环和结束 |
| 最终一次性打印回答 | `stream(..., stream_mode="messages")` 按 token 输出 |

---

## 核心代码片段

完整实现见 [`code.py`](code.py)，主体只包含模型、一个 Bash 工具、`create_agent` 和流式消费四部分。

### 1. 配置模型

传入 `ChatOpenAI` 实例，可以继续使用 `.env` 中的 OpenAI-compatible 地址和模型名：

```python
model = ChatOpenAI(
    model=os.environ["MODEL_ID"],
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL") or None,
    max_completion_tokens=8000,
    temperature=0,
)
```

### 2. 把普通函数声明成工具

函数签名生成参数 schema，docstring 告诉模型工具的用途。命令执行、输出截断、超时和基础危险命令拦截都直接放在同一个工具函数中，便于第一章完整阅读。

```python
@tool
def bash(command: str) -> str:
    """Run a shell command in the current working directory."""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(fragment in command for fragment in dangerous):
        return "Error: Dangerous command blocked"

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return (result.stdout + result.stderr).strip() or "(no output)"
```

### 3. 创建智能体

```python
agent = create_agent(
    model=model,
    tools=[bash],
    system_prompt=SYSTEM,
)
```

这一行会在 LangGraph 上创建预构建执行图。它完成参考实现中的关键闭环：调用模型、识别工具调用、执行 Bash、将结果作为 `ToolMessage` 放回状态，再次调用模型，直到模型给出最终回答。

### 4. 使用 `stream` 实时输出

```python
for chunk in agent.stream(
    {"messages": messages},
    stream_mode=["messages", "values"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if metadata.get("langgraph_node") == "model" and token.text:
            print(token.text, end="", flush=True)
    elif chunk["type"] == "values":
        final_messages = chunk["data"]["messages"]
```

两个 stream mode 各有职责：

- `messages` 提供模型 token，因此用户不用等整轮 Agent Loop 结束才看到回答。
- `values` 在图节点完成后提供完整 state，本章用最后一次 state 更新会话历史，以便下一轮继续对话。

`bash` 工具自己打印命令和一小段输出，因此运行时可以同时观察“模型正在回答什么”和“Agent 执行了什么”。

---

## 结合 `code.py` 理解 LangChain 的特点

- **代码短**：`create_agent` 封装标准的模型/工具循环，第一章不用自建消息适配器、工具路由和 `while True`。
- **模型接口统一**：本章使用 `ChatOpenAI`，也可以替换为其他 LangChain chat model；Agent 主体无需跟着重写。
- **工具协议统一**：`@tool` 从 Python 类型标注和 docstring 生成工具定义，LangChain 自动维护工具调用 ID 与 `ToolMessage` 的对应关系。
- **图运行时**：`create_agent` 构建的是 LangGraph compiled graph，因此天然支持 stream、state、checkpoint 和后续 middleware 扩展。
- **流式可观察**：`messages` 模式输出 LLM token，`updates` 或 `values` 模式输出 Agent 执行进度/状态；多个模式可以组合使用。
- **渐进扩展**：后续可以通过 middleware 加权限、Hook、动态提示词与错误处理，而不必改写核心循环。

这里的“代码更少”不是删除 Agent Loop：循环仍然存在，只是从本章的业务代码下沉到了 LangChain/LangGraph 的预构建运行时。模型仍负责决定是否调用工具，Harness 仍负责可靠地执行工具和回传结果。

官方文档：[Agents 与 `create_agent`](https://docs.langchain.com/oss/python/langchain/agents) · [Streaming](https://docs.langchain.com/oss/python/langchain/streaming) · [Tools](https://docs.langchain.com/oss/python/langchain/tools)

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
