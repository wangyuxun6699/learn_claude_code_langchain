# s02: Tool Use — 循环不变，只扩展工具

s01 → `s02` → [s03](../s03_permission/) → s04 → ... → s16 → s17
> *"工具自己描述自己"* — Agent Loop 不动，新增一个 `@tool` 函数并加入 `TOOLS` 即可。
>
> **Harness 层**: 工具分发 — 扩展模型能触达的边界。

---

## 只有 bash 一个工具

s01 的 Agent 只有一个 bash 工具。读文件要 `cat`，写文件要 `echo "..." > file.py`，改文件要 `sed`。

模型想的是"读这个文件"，却要拼出 `cat path/to/file`。多了一层翻译，浪费 token，还容易拼错。

---

## 全局视角：工具注册

![Tool Dispatch](images/tool-dispatch.svg)

s01 的模型配置、`create_agent()`、流式消费和会话历史更新全部保留。第二章只增加四个 `@tool` 函数，并把传给 Agent 的工具列表从一个扩展为五个。

给 Agent 加一个工具只需要做两件事：

1. **定义工具**：给带类型标注和 docstring 的 Python 函数加上 `@tool`
2. **注册工具**：把生成的工具对象加入 `TOOLS`

工具 schema、调用分发、`ToolMessage` 回填和循环继续都由 LangChain/LangGraph 负责。

---

## 从 1 个工具到 5 个工具

s01 只有一个 `bash` 工具：

```python
@tool
def bash(command: str) -> str:
    """Run a shell command in the current working directory."""
    ...

TOOLS = [bash]
```

s02 保留它，再封装四个意图更明确的文件工具：

```python
@tool
def read_file(path: str, limit: int | None = None) -> str:
    """Read a UTF-8 text file in the workspace, optionally limiting lines."""
    lines = safe_path(path).read_text(encoding="utf-8").splitlines()
    if limit is not None:
        lines = lines[:limit]
    return "\n".join(lines)

@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace the first exact occurrence of old_text in a workspace file."""
    text = safe_path(path).read_text(encoding="utf-8")
    if old_text not in text:
        return "Error: text not found"
    safe_path(path).write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Edited {path}"
```

`write_file` 负责覆盖写入并自动创建父目录；`glob` 支持 `**` 递归匹配，并把输出限制在 200 项。完整实现见 [`code.py`](code.py)。

---

## 工具注册

```python
TOOLS = [bash, read_file, write_file, edit_file, glob]

agent = create_agent(
    model=model,
    tools=TOOLS,
    system_prompt=SYSTEM,
)
```

`@tool` 根据函数名、类型标注和 docstring 生成模型可见的 schema。`create_agent` 内部的工具节点按名称找到 Python 工具、校验参数、执行调用，并用相同的调用 ID 生成 `ToolMessage`。业务代码不再维护一份容易与 schema 失配的 `TOOL_HANDLERS`。

---

## 多个工具调用

模型经常一次返回多个工具调用："读一下 a.py 和 b.py，然后列出所有 .py 文件"。

`create_agent` 会把这些调用统一交给工具节点执行，再将全部结果送回模型。需要注意：预构建工具节点可以并行执行同一批调用，因此不要假定同一轮里的 `write_file` 与 `read_file` 存在先后关系；有数据依赖时，模型应分成两轮调用。并发安全分区和权限判断会在后续章节逐步展开。

---

## 速查

| 概念 | 一句话 |
|------|--------|
| `@tool` | 从函数名、签名和 docstring 生成工具对象与 JSON Schema |
| `TOOLS` | 传给 `create_agent` 的五个工具对象 |
| 多工具调用 | LangGraph 工具节点执行并回填与调用 ID 对应的结果 |
| 循环不变 | s01 的 `create_agent` 和 stream 消费逻辑保持一致 |

---

## 相对 s01 的变更

| 组件 | 之前 (s01) | 之后 (s02) |
|------|-----------|-----------|
| 工具数量 | 1 (bash) | 5 (+read, write, edit, glob) |
| 工具声明 | 一个 `@tool` 函数 | 五个 `@tool` 函数 + `TOOLS` 列表 |
| 路径安全 | bash 无工作区路径约束 | 四个文件工具由 `safe_path()` 限制在工作区 |
| Agent Loop | `create_agent` + stream | 完全一致 |

---

## 结合 `code.py` 理解 LangChain / LangGraph

本章的重点不是“多写四个 Python 函数”，而是让工具的声明、实现与运行时注册保持在同一个抽象里。

### 工具从声明到执行的完整路径

[`code.py`](code.py) 中的完整调用路径是：

```text
Python 函数 + @tool → StructuredTool（名称、描述、参数 schema）
→ create_agent(tools=TOOLS) → 模型产生 AIMessage.tool_calls
→ LangGraph ToolNode 执行工具 → ToolMessage → 模型继续推理
```

工具函数本身同时提供 schema 和执行入口，避免了旧实现中 `TOOLS` JSON 与 `TOOL_HANDLERS` 两处注册发生漂移。工具结果中的调用 ID 也由框架维护。

### 五个工具各自承担的边界

| 工具 | 本章实现中的关键约束 | 价值 |
|---|---|---|
| `bash` | 120 秒超时、输出截断、危险片段初步拦截 | 提供通用命令能力 |
| `read_file` | UTF-8、可选行数上限 | 避免 shell 引号和平台差异 |
| `write_file` | 自动创建父目录 | 明确覆盖语义和返回值 |
| `edit_file` | 只替换第一次精确匹配 | 减少对无关内容的破坏 |
| `glob` | 支持 `**`、限制工作区内、最多展示 200 项 | 控制路径和上下文规模 |

`safe_path()` 使用 `resolve()` 与 `is_relative_to(WORKDIR)` 建立工作区边界。它是宿主安全边界，不是提示词约定；真正的安全约束必须在工具实现或权限层执行。

### 与 LangChain 工具系统的关系

本章使用 LangChain 的常规写法：`@tool` 从 Python 类型标注生成 schema，`create_agent` 创建的 LangGraph 执行图负责模型节点与工具节点之间的路由。

- `@tool`：把一个普通函数变成有 schema 的 `StructuredTool`。
- `TOOLS`：显式列出本章开放给模型的能力边界。
- `create_agent`：绑定工具并建立“模型 → 工具 → 模型”执行图。
- `stream`：输出模型 token，并用最终 state 延续多轮会话。

本章仍然只做基础路径约束。`bash` 可以执行任意未被简单危险片段命中的命令，文件覆盖也不会询问用户；真正的 allow / deny / ask 权限策略属于 s03。

官方概念：[Tools](https://docs.langchain.com/oss/python/langchain/tools) · [Agents 与 `create_agent`](https://docs.langchain.com/oss/python/langchain/agents) · [Streaming](https://docs.langchain.com/oss/python/langchain/streaming)

---

## 试一下

```sh
cd learn-claude-code
python s02_tool_use/code.py
```

试试这些 prompt：

1. `Read the file README.md and tell me what this project is about`
2. `Create a file called test.py that prints "hello", then read it back`
3. `Find all Python files in this directory`
4. `Read both README.md and requirements.txt, then create a summary file`

观察重点：模型什么时候只调一个工具，什么时候一次调多个？存在数据依赖时，它是否会等结果返回后再发起下一轮调用？

---

## 接下来

现在 Agent 有 5 个工具。文件工具受 `safe_path` 保护，但 bash 仍有很大的操作范围，写入与编辑也会直接执行。

s03 Permission → 为了看清“工具执行之前”的控制点，下一章会展开底层调用循环并加入 allow / deny / ask 权限门。
---

<!-- upstream-cc-source:start -->
## 深入 CC 源码


<details>
<summary>深入 CC 源码</summary>

> 以下基于 CC 源码 `Tool.ts`、`tools.ts`、`toolOrchestration.ts`、`toolExecution.ts`、`StreamingToolExecutor.ts` 的核查。

### 一、工具定义方式

**教学版**：`TOOLS` 数组 + `TOOL_HANDLERS` 字典。定义和实现分开。
**CC**：每个工具是 `buildTool()` 创建的独立对象，包含 schema、验证、权限、执行。`getAllBaseTools()` 汇总所有工具。

教学版的分离方式对教学更清晰——读者一眼看到"加一个工具 = 两条定义"。

### 二、并发安全判断：isConcurrencySafe()

![Tool Concurrency](images/concurrency-comparison.svg)

教学版按原始顺序逐个执行，不做并发。CC 用 `isConcurrencySafe(input)` 判断能否并发——注意这不是简单的"只读 vs 写"，而是按具体输入判断：

| | isReadOnly | isConcurrencySafe |
|---|---|---|
| FileRead | true | true |
| Glob | true | true |
| Bash `ls` | true | **true** ← 关键差异 |
| Bash `rm` | false | false |
| TaskCreate | false | **true** ← 改状态但可并发（TaskCreate 在 s12 介绍） |

CC 的 Bash tool 的 `isConcurrencySafe` 等于 `isReadOnly`——只读命令可并发，写命令不可。TaskCreate 虽然改了任务文件，但每次都写不同的文件，所以可以并发。

### 三、分区算法

CC 的 `partitionToolCalls()`（`toolOrchestration.ts:91-115`）不是分两组，而是把工具调用**按连续块分批**：

```
[read A, read B, glob *.py, bash "rm x", read C]
  → batch1(并发): [read A, read B, glob *.py]
  → batch2(串行): [bash "rm x"]
  → batch3(并发): [read C]
```

并发安全的连续块编入同一个 batch，batch 内真正并发执行（`toolOrchestration.ts:152-176`，有并发上限）。遇到非并发安全的就开新 batch 串行执行。batch 之间严格顺序。

### 四、验证管线

CC 的每个工具调用经过严格的 5 步验证（`toolExecution.ts`）：

1. **Zod schema 验证**（`614-680`，教学版用 JSON Schema 替代）：参数类型/结构检查
2. **工具级 validateInput()**（`682-733`）：参数值验证（如路径是否在工作区内）
3. **PreToolUse hooks**（`800-862`，s04 详细介绍）：钩子可以返回消息、修改输入、阻止执行
4. **权限检查**（`921-931`，s03 的核心内容）：canUseTool + checkPermissions → allow/deny/ask
5. **执行 tool.call()**（`1207-1222`）

教学版省略了 Zod（用 JSON Schema）、省略了 validateInput（用安全函数）、保留了权限检查和钩子概念。

### 五、流式工具执行

CC 的 `StreamingToolExecutor`（`StreamingToolExecutor.ts`）让工具在模型还在生成时就启动——不等模型说完。`read_file` 可能在模型还在输出"我来分析"的时候就跑完了。教学版不实现这个，目标和 s01 一致——概念清晰，不追求性能极致。

### 六、工具结果持久化

每个工具有一个 `maxResultSizeChars` 字段。结果超过这个值就落盘，模型看到的是预览 + 文件路径。FileRead 特殊——设为 `Infinity`，防止读文件的输出又被当成文件落盘。具体来说，如果 FileRead 的结果超过阈值被落盘，模型下次读那个落盘文件时又会触发落盘 → 无限循环（读文件 → 落盘 → 再读 → 再落盘 → ...）。

</details>

<!-- upstream-cc-source:end -->
