# s02: Tool Use — 多加一个工具，只加一行

s01 → `s02` → [s03](../s03_permission/) → s04 → ... → s16 → s17
> *"加一个工具, 只加一个 handler"* — 循环不用动, 新工具注册进 dispatch map 就行。
>
> **Harness 层**: 工具分发 — 扩展模型能触达的边界。

---

## 只有 bash 一个工具

s01 的 Agent 只有一个 bash 工具。读文件要 `cat`，写文件要 `echo "..." > file.py`，改文件要 `sed`。

模型想的是"读这个文件"，却要拼出 `cat path/to/file`。多了一层翻译，浪费 token，还容易拼错。

---

## 全局视角：工具分发

![Tool Dispatch](images/tool-dispatch.svg)

s01 的循环完全保留（LLM 调用、`tool_use` block 判断、消息追加）。唯一的变动在工具执行那 1 行：`run_bash()` 替换为 `TOOL_HANDLERS[block.name]()` 查表分发。

给 Agent 加一个工具只需要做两件事：

1. **定义工具**：在 `TOOLS` 数组里加一条描述
2. **注册处理函数**：在 `TOOL_HANDLERS` 字典里加一个映射

---

## 从 1 个工具到 5 个工具

s01 只有一个 bash：

```python
TOOLS = [{"name": "bash", ...}]

def run_bash(command): ...
```

s02 加到 5 个，每个工具都是独立定义：

```python
TOOLS = [
    {"name": "bash",       "description": "Run a shell command.", ...},
    {"name": "read_file",  "description": "Read file contents.",  ...},
    {"name": "write_file", "description": "Write content to file.", ...},
    {"name": "edit_file",  "description": "Replace text in file once.", ...},
    {"name": "glob",       "description": "Find files by pattern.", ...},
]
```

每个工具有自己的实现函数：

```python
def run_read(path, limit=None):
    lines = safe_path(path).read_text(encoding="utf-8").splitlines()
    if limit:
        lines = lines[:limit]
    return "\n".join(lines)

def run_write(path, content):
    safe_path(path).write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"

def run_edit(path, old_text, new_text):
    text = safe_path(path).read_text(encoding="utf-8")
    if old_text not in text:
        return "Error: text not found"
    safe_path(path).write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Edited {path}"

def run_glob(pattern):
    import glob as g
    matches = sorted(set(g.glob(
        pattern, root_dir=WORKDIR, recursive=True)))
    shown = matches[:200]
    if len(matches) > 200:
        shown.append("... (more matches omitted; narrow the pattern)")
    return "\n".join(shown)
```

---

## 工具分发

```python
TOOL_HANDLERS = {
    "bash":       run_bash,
    "read_file":  run_read,
    "write_file": run_write,
    "edit_file":  run_edit,
    "glob":       run_glob,
}

# 循环里只改了一行——从硬编码 run_bash 变成查表：
for block in tool_calls:
    handler = TOOL_HANDLERS[block.name]    # 查表
    output = handler(**block.input)         # 调用
    results.append(...)
```

加一个工具 = 在 `TOOLS` 数组加一条 + 在 `TOOL_HANDLERS` 字典加一行。循环不变。

---

## 多个工具调用

模型经常一次返回多个 tool_use："读一下 a.py 和 b.py，然后列出所有 .py 文件"。

这些调用按照 `response.content` 中的原始顺序逐个执行。

---

## 速查

| 概念 | 一句话 |
|------|--------|
| TOOL_HANDLERS | 工具名 → 处理函数的字典。加工具 = 加一行映射 |
| 工具定义 | 告诉模型"我能做什么"的 JSON schema |
| 多工具调用 | 模型可一次返回多个 tool_use，并按原始顺序逐个执行 |
| 循环不变 | s01 的 `while True` 循环一行都没改 |

---

## 相对 s01 的变更

| 组件 | 之前 (s01) | 之后 (s02) |
|------|-----------|-----------|
| 工具数量 | 1 (bash) | 5 (+read, write, edit, glob) |
| 工具执行 | 硬编码 `run_bash()` | TOOL_HANDLERS 查表分发 |
| 路径安全 | 无 | safe_path 校验（仅 file tools） |
| 循环 | `while True` + `tool_use` block | 与 s01 完全一致 |

---

## 结合 `code.py` 理解 LangChain / LangGraph

本章的重点不是“多写四个 Python 函数”，而是把工具拆成两个必须保持一致的部分：给模型看的 schema，以及宿主真正执行的 handler。

### 工具从声明到执行的完整路径

[`code.py`](code.py) 中的 `TOOLS` 是模型可见的能力目录，`TOOL_HANDLERS` 是运行时分发表：

```text
TOOLS.input_schema → _openai_tools() → ChatOpenAI.bind_tools()
→ AIMessage.tool_calls → ToolUseBlock(name, input, id)
→ TOOL_HANDLERS[name](**input) → tool_result(tool_use_id=id)
```

名称是两侧的连接键。schema 声明了 `read_file`，分发表就必须存在同名 handler；参数字段也必须能被 `handler(**block.input)` 接收。未知工具不会直接崩溃，而是作为工具结果回给模型，让模型有机会修正。

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

本章直接写 JSON Schema，是为了展示 provider 无关的底层协议。LangChain 更常见的写法是用 `@tool` 或 `StructuredTool` 从 Python 类型标注生成 schema，再交给 `bind_tools()`。如果使用预构建 Agent，`ToolNode` 会负责读取 `AIMessage.tool_calls`、调用工具并生成带正确调用 ID 的 `ToolMessage`。

- `bind_tools()`：把工具定义告诉模型，不执行工具。
- `TOOL_HANDLERS` 或 `ToolNode`：执行模型提出的调用。
- Agent loop：决定执行完后是否再次调用模型。
- LangGraph：在需要持久化、分支、并行或人工中断时管理这些节点。

虽然模型一次响应可以给出多个调用，本章的 `for block in tool_calls` 仍是串行调度。只有明确使用异步任务、支持并行的工具节点或图分支，才获得真正的并行执行。

官方概念：[Tools](https://docs.langchain.com/oss/python/langchain/tools) · [Models 中的工具调用](https://docs.langchain.com/oss/python/langchain/models)

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

观察重点：模型什么时候只调一个工具，什么时候一次调多个？多个工具调用的顺序和结果是否正确？

---

## 接下来

现在 Agent 有 5 个专用工具。file tools 受 `safe_path` 保护，但 bash 不受限制，`rm -rf /` 还是能跑。

s03 Permission → 在工具执行之前加一道门：这个操作安全吗？需要用户批准吗？
---

<!-- upstream-cc-source:start -->
## 深入 CC 源码

> 原文：[s02_tool_use](https://github.com/shareAI-lab/learn-claude-code/blob/67a9126c6435a8654ba7a6f68c0fd2130f00a462/s02_tool_use/README.md)。以下折叠块保持原文，文中的章号与源码行号沿用该版本。

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
