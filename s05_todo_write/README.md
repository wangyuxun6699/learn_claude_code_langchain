# s05: TodoWrite — 没有计划的 Agent，做着做着就偏了

s01 → s02 → s03 → s04 → `s05` → [s06](../s06_subagent/) → s07 → ... → s16 → s17

> *"没有计划的 agent 走哪算哪"* — 先列步骤再动手，长任务更不容易漏项。
>
> **Harness 层**: 规划 — 让 Agent 在动手之前先想清楚。

---

## 问题

给 Agent 一个复杂任务："把所有 Python 文件改成 snake_case 命名，然后跑测试，修好失败。"

Agent 开始干活，改了 3 个文件，跑了个测试，发现 2 个失败，开始修。修着修着，它忘了最初是"改成 snake_case"，测试失败把注意力全吸走了。

对话越长越严重：工具结果不断填满上下文，系统提示的影响力被稀释。一个 10 步重构，做完 1-3 步就开始即兴发挥，因为 4-10 步已经被挤出注意力了。

---

## 解决方案

![Todo Overview](images/todo-overview.svg)

S05 保留 S04 的工具分发、权限检查和 Hooks，再加入 `todo_write` 与 reminder 计数器。`todo_write` 只更新计划状态，实际工作仍由原有工具完成。

新工具仍通过 `TOOL_HANDLERS[block.name]` 分发。连续三个工具调用轮次没有使用 `todo_write` 时，Harness 会把 reminder 追加到第三轮的工具结果中。

---

## 工作原理

**TodoManager** 持有内存中的任务列表，负责校验更新，并把渲染结果返回给模型。`run_todo_write` 同时把这份状态打印到终端：

```python
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, todos: list | str) -> str:
        # Parse and validate before replacing the current list.
        validated = []
        ...
        self.items = validated
        return self.render()

    def render(self) -> str:
        # [ ] pending, [>] in progress, [x] completed
        ...


TODO = TodoManager()

def run_todo_write(todos: list | str) -> str:
    output = TODO.update(todos)
    print(output)
    return output
```

一次更新最多包含 20 项；每项都必须有非空的 `content`；同一时间只能有一个 `in_progress`。字符串输入可以是 JSON，也可以是 Python 列表表示，解析过程不使用 `eval`。

工具定义和其他 5 个工具一起加入 dispatch map：

```python
TOOLS = [
    {"name": "bash",       ...},
    {"name": "read_file",  ...},
    {"name": "write_file", ...},
    {"name": "edit_file",  ...},
    {"name": "glob",       ...},
    # s05: 新增一条
    {"name": "todo_write", "description": "Create and manage a task list ...",
     "input_schema": {
         "type": "object",
         "properties": {
             "todos": {
                 "type": "array",
                 "items": {
                     "type": "object",
                     "properties": {
                         "content": {"type": "string"},
                         "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                     },
                 },
             },
         },
     },
    },
]

TOOL_HANDLERS["todo_write"] = run_todo_write
```

**Reminder**：连续三个工具调用轮次没有使用 `todo_write` 时，reminder 会追加到第三轮的结果中，随后计数器清零：

```python
rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
if rounds_since_todo >= 3:
    results.append({
        "type": "text",
        "text": "<reminder>Update your todos.</reminder>",
    })
    rounds_since_todo = 0
```

Agent 收到任务后的典型流程：先调 `todo_write` 列出所有步骤（全 `pending`）→ 做一个步骤，改成 `in_progress` → 做完改成 `completed` → 看下一个 `pending` → 继续。

**关键洞察**：todo_write 不给 Agent 增加任何**执行能力**。它增加的是**规划能力**。

---

## 相对 s04 的变更

| 组件 | 之前 (s04) | 之后 (s05) |
|------|-----------|-----------|
| 工具数量 | 5 (bash, read, write, edit, glob) | 6 (+todo_write) |
| 规划能力 | 无 | 带状态的 TODO 列表 + reminder |
| SYSTEM 提示 | 通用提示 | 加入 "先计划再执行" 引导 |
| 循环 | 工具分发与 Hooks | 保留分发路径，加入 rounds_since_todo 和 reminder 注入 |

---

## 结合本章代码理解 Agent State 与 Todo Middleware

[`code.py`](code.py) 的 `TodoManager` 是一个显式的会话内状态容器。它不把计划写进自然语言历史后就不管，而是把每一项约束为结构化记录：`content` 表示任务，`status` 只能是 `pending / in_progress / completed`，`activeForm` 表示当前进行时描述。

### 一次 `todo_write` 如何更新状态

1. `run_todo_write()` 接收列表；为兼容部分模型，也接受 JSON 字符串或 Python list 字面量。
2. 字符串只通过 `json.loads()` 和 `ast.literal_eval()` 解析，不使用危险的 `eval()`。
3. `TodoManager.update()` 先完整校验候选列表，再一次性替换旧状态；失败不会留下半更新数据。
4. 状态约束要求最多一个 `in_progress`，防止模型同时声称正在做多个串行步骤。
5. `render()` 把结构化状态转成模型和用户容易阅读的进度视图。
6. Agent 连续三批工具结果未更新 Todo 时，运行时追加一次提醒，而不是每轮重复污染上下文。

这里的关键是“先验证、后提交”。Todo 是控制状态，不应因为模型少传一个字段就把原计划破坏掉。

### 与 LangChain `TodoListMiddleware` 的关系

LangChain 的内置 Todo middleware 会为 Agent 增加写入 Todo 的工具，并把 Todo 保存到 Agent state。本章手写 `TodoManager`，因此可以直接看到状态校验、提醒注入和渲染行为；代价是它只存在于当前 Python 进程，退出后不会自动恢复。

仓库中的 [`code_streaming.py`](code_streaming.py) 展示了更框架化的版本：

- `create_agent()` 创建运行在 LangGraph 上的 Agent。
- `TodoListMiddleware` 管理结构化 Todo state。
- `before_agent`、`wrap_tool_call`、`after_agent` 承接 s04 的 hooks。
- `agent.stream(..., stream_mode="values")` 在每个图步骤后给出当前 state，因此可以逐步显示工具调用、Todo 变化和最终回复。

### Todo、Task 和 Graph state 不相同

| 概念 | 生命周期 | 用途 |
|---|---|---|
| 本章 Todo | 当前进程 / 当前会话 | 给模型一张短期执行清单 |
| s10 Task | 文件持久化、带依赖和 owner | 协调可认领的工作单元 |
| LangGraph state | 一个 thread 的图执行状态 | 在节点间传值，可由 checkpointer 持久化 |

如果把本章迁移到 LangGraph，Todo 最自然地成为 state 的一个字段，并由 reducer 或 middleware 定义更新规则；需要跨进程恢复时，再给编译后的图配置 checkpointer。

官方概念：[Built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in) · [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)

---

## 试一下

```sh
cd learn-claude-code
python s05_todo_write/code.py
```

试试这些 prompt：

1. `Refactor s05_todo_write/example/hello.py: add type hints, docstrings, and a main guard`（先列 3 步再执行）
2. `Create a Python package under s05_todo_write/example/demo_pkg with __init__.py, utils.py, and tests/test_utils.py`
3. `Review Python files under s05_todo_write/example and fix any style issues`

观察重点：第一次工具调用是不是 `todo_write`？TODO 列了几步？执行过程中状态有没有从 `pending` 变成 `in_progress` / `completed`？

---

## 接下来

Agent 能计划了。但如果一个任务太大，比如"重构整个认证模块"，光靠 TODO 列表不够。这个任务本身就是几十个小任务的集合，放在同一个对话里会被上下文淹没。

s06 Subagent → 把大任务拆成子任务，每个子任务派一个独立的 Agent。它们有自己的干净上下文，不会互相污染。
---

<!-- upstream-cc-source:start -->
## 深入 CC 源码

> 原文：[s05_todo_write](https://github.com/shareAI-lab/learn-claude-code/blob/67a9126c6435a8654ba7a6f68c0fd2130f00a462/s05_todo_write/README.md)。以下折叠块保持原文，文中的章号与源码行号沿用该版本。

<details>
<summary>深入 CC 源码</summary>

CC 中有两套任务系统并存（`tasks.ts:133-139`）：

- **TodoWrite（V1）**：一个简单的列表工具，数据在内存 AppState 中维护（`TodoWriteTool.ts:65-103`）。教学版也保存在进程内存里，退出后清空
- **Task System（V2 = s12）**：文件持久化、依赖图、并发锁、ownership

切换由 `isTodoV2Enabled()` 控制。当前源码的实现逻辑：交互式会话中 V2 默认启用，非交互式会话（SDK）中 V1 默认启用；设置 `CLAUDE_CODE_ENABLE_TASKS` 环境变量可强制启用 V2。注意源码注释 "Force-enable tasks in non-interactive mode" 描述的是 env var 路径的用途，和默认分支的返回值语义不同，阅读时需区分。

教学版省略了真实源码中的 `activeForm` 字段（`utils/todo/types.ts:8-15`）。CC 用它给 UI spinner 展示"正在做什么"，教学版只有终端输出，不需要这个字段。

教学版的 nag reminder（3 轮未更新就注入提醒）是教学机制。CC 源码中没有固定的"3 轮"逻辑，更接近的是 `TodoWriteTool.ts:72-107` 中当 3 个以上 todo 全部完成但没有 verification 项时，追加 verification nudge。

Task System 相比 TodoWrite 的核心增量：
- 文件持久化（Claude 配置目录下 `tasks/{taskListId}/{taskId}.json`）而非内存列表
- `blockedBy` 依赖图而非平铺列表
- `proper-lockfile` 并发安全而非无锁
- 四个独立工具（Create/Get/Update/List）而非一个
- TaskCreated / TaskCompleted hooks（`TaskCreateTool.ts:80-129`、`TaskUpdateTool.ts:231-260`）供外部系统集成

</details>

<!-- upstream-cc-source:end -->
