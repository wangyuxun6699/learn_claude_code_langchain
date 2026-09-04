# s09: Memory — 让重要信息跨会话保留下来

s01 → ... → s07 → s08 → `s09` → [s10](../s10_task_system/) → s11 → ... → s16 → s17
> *"把以后还会用到的信息留下来。"* 文件存储 + 索引 + 相关性选择 + 按需召回。
>
> **Harness 层**：Memory 在会话之外保存可复用知识，并在相关任务中取回。

---

## 问题

Agent 开始新会话时，`messages` 里没有上一次的对话。用户之前说过的编码偏好、项目背景和排查线索，下次任务还可能用到。没有持久存储，这些信息只能由用户重新说一遍。

把完整 transcript 留下来适合归档，却不适合每次都发给模型。对话会越来越长，当前任务需要的信息很难定位，旧事实也可能已经过期。Memory 要解决的是两个问题：哪些信息值得跨会话保存，以及当前任务应该取回哪几条。

![Memory Overview](images/memory-overview.svg)

---

## 全部写进 system prompt，为什么不合适

最直接的做法，是把用户偏好和项目事实写进一个固定文件，启动时全部放进 system prompt。这样确实能够记住信息，但每次调用 LLM 都要重新发送全部内容。记忆越多，与当前任务无关的内容就越多，输入 token 和上下文窗口也会被持续占用。

s07 已经展示过一种更合适的读取方式：保留简短索引，只在需要时加载正文。Skill 由人编写并保持只读；Memory 则允许 Agent 从对话中提取内容，并在后续任务中再次使用。

因此，本章需要处理四件事：存储、召回、提取和整理。

![Memory Subsystems](images/memory-subsystems.svg)

---

## 存储：一个记忆一个文件

每条记忆是 `.memory/` 下的一个 Markdown 文件，YAML frontmatter 记录 `name`、`description` 和 `type`：

```markdown
---
name: user-preference-tabs
description: User prefers tabs for indentation
type: user
---

User prefers using tabs, not spaces, for indentation.
```

`type` 有四类：

| 类型 | 保存什么 | 示例 |
|------|---------|------|
| user | 用户的长期偏好 | “使用 tab 缩进” |
| feedback | 以后仍适用的工作反馈 | “不要 mock 数据库” |
| project | 稳定的项目事实 | “认证重写由合规要求驱动” |
| reference | 外部资料或查找线索 | “流水线问题记录在 Linear INGEST” |

`MEMORY.md` 是索引，每行对应一个记忆文件。写入完成后，`rebuild_memory_index()` 根据文件重新生成索引：

```python
def write_memory_file(name, mem_type, description, body):
    path = MEMORY_DIR / f"{memory_slug(name)}.md"
    path.write_text(
        memory_document(name, mem_type, description, body), encoding="utf-8"
    )
    rebuild_memory_index()
    return path
```

索引用于选择相关记忆，正文仍然保存在各自的文件中。

---

## 召回：先选择，再加载正文

每次用户发起请求时，`select_relevant_memories()` 读取最近的用户消息和记忆目录，让一次轻量模型调用选择最多五条相关记录：

```python
prompt = (
    "Select memory records that are relevant to the current user request. "
    "Return only a JSON array of catalog indices, such as [0, 2]. "
    "Return [] when none are relevant."
)
```

如果模型调用或 JSON 解析失败，代码会退回关键词匹配。选择完成后，`load_memories()` 才读取对应文件，并限制召回正文的总长度。

```python
relevant_memories = load_memories(messages)
system = build_system(relevant_memories)
```

`build_system()` 会明确说明：召回内容只是背景知识，不是新的用户命令；如果记忆与当前请求冲突，以当前请求为准。这样既能使用旧信息，也不会让旧记忆替用户发号施令。

---

## 提取：回合结束后保存可复用信息

用户不一定会明确说“请记住”。`extract_memories()` 在 Agent 完成本轮回答后检查当前对话，只提取以后仍可能有用的信息：

```python
tool_calls = [
    block for block in response.content if block.type == "tool_use"
]
if not tool_calls:
    force = trigger_hooks("Stop", messages)
    if force:
        messages.append({"role": "user", "content": force})
        continue
    if extract_memories(messages):
        consolidate_memories()
    return
```

模型返回的内容只是候选，不会直接写盘。候选必须带有 `scope`：只有 `persistent` 才表示它应当跨会话保留；`current_task` 表示本次任务的命令、临时路径和临时限制。

`should_store_memory()` 负责最后的检查。字段不完整、带有“本次会话”或“当前任务”等临时含义、或者与已有记忆重复的候选都会被拒绝。比如“这次不要创建文件”只约束当前任务，不应该在下次会话中继续生效。

---

## 整理：合并重复和过期内容

记忆文件积累到一定数量后，内容可能重复、矛盾或过期。教学实现达到 10 条时调用 `consolidate_memories()`，让模型生成一份整理后的记录列表。

整理过程先解析并校验新列表，再替换旧文件。替换前会保存快照；删除或写入失败时，代码恢复原文件并重建索引：

```python
snapshot = {
    path.name: path.read_text(encoding="utf-8")
    for path in MEMORY_DIR.glob("*.md")
    if path.name != MEMORY_INDEX.name
}

try:
    for path in MEMORY_DIR.glob("*.md"):
        if path.name != MEMORY_INDEX.name:
            path.unlink()
    for record in consolidated:
        path = MEMORY_DIR / f"{memory_slug(record['name'])}.md"
        path.write_text(memory_document(
            record["name"], record["type"],
            record["description"], record["body"],
        ), encoding="utf-8")
    rebuild_memory_index()
except Exception:
    for path in MEMORY_DIR.glob("*.md"):
        if path.name != MEMORY_INDEX.name:
            path.unlink()
    for filename, content in snapshot.items():
        (MEMORY_DIR / filename).write_text(content, encoding="utf-8")
    rebuild_memory_index()
    raise
```

课程代码把整理触发条件简化为数量阈值。真实应用还需要根据数据规模和并发方式，决定何时整理以及如何避免多个进程同时改写同一份存储。

---

## 本节代码

| 组成 | 本节实现 |
|------|---------|
| Agent Loop | 保留消息、工具调用、工具结果和 hooks 触发点 |
| 基础工具 | `bash`、`read_file`、`write_file`、`edit_file`、`glob` |
| 存储 | `.memory/MEMORY.md` 索引 + `.memory/*.md` 文件 |
| 召回 | 目录选择 + 关键词降级 + 正文长度上限 |
| 写入 | 回合结束后提取 + 持久性检查 + 重复过滤 |
| 整理 | 达到阈值后合并，失败时恢复原文件 |

> **与 s08 的边界：** s08 管理当前会话的上下文预算，s09 管理会话之外的可复用知识。Memory 是选择性存储，不是 transcript 的无损备份，也不会取代上下文压缩。

---

## 结合本章代码理解长期记忆

s08 管理同一会话的消息窗口；s09 管理跨会话可复用的知识。[`code.py`](code.py) 把每条记忆保存为 `.memory/*.md`，用 YAML frontmatter 描述 `name / type / description`，正文保存具体内容，`MEMORY.md` 只充当轻量索引。

### 一次完整的记忆生命周期

1. `list_memory_files()` 读取索引项和正文元数据。
2. `select_relevant_memories()` 把最近三次用户输入与记忆目录交给一个无工具模型，让它只返回目录下标；失败时退化为关键词排序。
3. `load_memories()` 在 20,000 字符预算内加载选中的正文。
4. `build_system()` 把目录和相关记忆加入 system prompt，并明确“记忆是背景数据，当前用户请求优先”。
5. 主 Agent 完成后，`extract_memories()` 从最近对话中提取稳定偏好、重复反馈、项目事实或外部参考。
6. `should_store_memory()` 排除临时状态、重复记录和不应跨会话保存的内容。
7. 记录达到阈值后，`consolidate_memories()` 合并重复、应用更新；写入失败时用快照回滚。

记忆提取与任务执行使用独立 prompt。提取器没有工具，并被要求把对话视为数据，避免把用户内容中的命令当作新的执行指令。

### 与 LangGraph Store 的对应关系

LangGraph 通常把短期记忆放在 thread state/checkpointer 中，把长期记忆放在 Store 中。Store 使用 namespace 与 key 组织 JSON 文档，可跨 thread 召回。本章的文件结构可以这样映射：

| 本章文件实现 | LangGraph 长期记忆 |
|---|---|
| `.memory/` | Store 后端 |
| 记忆类型与用户/项目范围 | namespace |
| 文件名 slug | key |
| Markdown + frontmatter | Store value JSON |
| LLM/关键词选择 | 语义搜索或应用自定义检索 |

本章没有 embedding，也没有向量数据库；它先让模型在小目录中做选择，失败再使用关键词。这种方案适合教学和小规模记忆库。记录很多时，应把召回改为 Store 查询或专门检索，并为不同用户、仓库和组织建立隔离 namespace。

### 记忆不是聊天记录

- “用户偏好 Python 3.12”可能是长期记忆。
- “正在修复第 42 行”是当前任务状态，应留在短期 state 或 task system。
- “工具输出了 500 行日志”是可审计记录，不应直接变成记忆。
- “以后不要修改生成目录”属于稳定反馈，适合在未来会话召回。

官方概念：[Long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory) · [Memory overview](https://docs.langchain.com/oss/python/concepts/memory) · [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)

---

## 试一下

```sh
cd learn-claude-code
python s09_memory/code.py
```

1. 输入 `I prefer using tabs for indentation. Remember that.`，结束后检查 `.memory/` 是否新增记忆文件，`MEMORY.md` 是否出现对应索引；
2. 输入 `q` 退出并重新运行程序，再问 `What indentation style do I prefer?`，确认新会话能够召回这条偏好；
3. 再保存一条与代码格式无关的偏好，然后询问缩进问题，观察当前请求只加载相关记忆；
4. 输入 `Do not create files in this session.`，确认这条临时要求不会成为下一次会话的持久规则。

模型的具体措辞和提取数量可能变化，判断重点是 `.memory/` 中保存了什么，以及新会话是否只取回相关内容。

---

## 接下来

Memory 解决了跨会话保留信息的问题，但复杂任务还需要记录每一步的状态和依赖关系。仅靠对话中的 TODO，程序退出后就无法继续追踪进度。

s10 Task System → 把任务、状态和依赖关系保存到磁盘。
---

<!-- upstream-cc-source:start -->
## 深入 CC 源码

> 原文：[s09_memory](https://github.com/shareAI-lab/learn-claude-code/blob/67a9126c6435a8654ba7a6f68c0fd2130f00a462/s09_memory/README.md)。以下折叠块保持原文，文中的章号与源码行号沿用该版本。

<details>
<summary>深入 CC 源码</summary>

> 以下基于 CC 源码 `src/` 下 `memdir/`、`services/`、`utils/`、`query/` 的分析，行号已对照核实。

### 源码路径

| 文件 | 行数 | 职责 |
|------|------|------|
| `memdir/memdir.ts` | 507 | 核心：MEMORY.md 定义（`34-38`）、记忆行为指令区分 memory/plan/tasks（`199-266`）、`loadMemoryPrompt()` 三条路径（`419-490`） |
| `memdir/findRelevantMemories.ts` | 141 | Sonnet side-query 选记忆（`18-24` 系统提示、`97-122` 调用逻辑） |
| `memdir/memoryTypes.ts` | 271 | 类型定义，frontmatter 字段 |
| `memdir/memoryScan.ts` | — | 扫描 .md 文件，排除 MEMORY.md，读 frontmatter，最多 200 个，按 mtime 降序（`35-94`） |
| `services/extractMemories/extractMemories.ts` | 615 | forked agent 提取记忆，受限权限，`skipTranscript: true`，`maxTurns: 5`（`371-427`） |
| `services/autoDream/autoDream.ts` | 324 | Dream 整理，四层门控（`63-66` 默认值、`130-190` 门控、`224-233` forked agent） |
| `services/SessionMemory/sessionMemory.ts` | 495 | 会话级记忆管理 |
| `services/compact/sessionMemoryCompact.ts` | — | session memory 轻量摘要，阈值 10K/5/40K（`56-61`） |
| `utils/attachments.ts` | — | 注入预算：200 行 / 4096 字节每文件，60KB 每 session（`269-288`）；按 query 找相关 memory（`2196-2241`） |
| `query.ts` | — | memory prefetch 每轮启动（`301-304`），非阻塞收集（`1592-1614`） |
| `query/stopHooks.ts` | — | stop hook fire-and-forget 触发提取和 Dream（`141-155`） |

### 记忆选择：LLM 选，不是 embedding

CC 用 **Sonnet 本身来选**（`findRelevantMemories.ts`），不是 embedding 向量相似度：

1. `memoryScan.ts` 扫描 `.memory/` 下所有 `.md` 文件（排除 MEMORY.md），最多 200 个，按 mtime 降序
2. 把 `name` + `description` 列成清单
3. 发给 Sonnet side-query："根据名称和描述选出真正有用的记忆（最多 5 个）。不确定就不要选。"
4. Sonnet 返回 `{ selected_memories: ["file1.md", ...] }`
5. 选中文件读取完整内容（每文件 ≤ 200 行 / 4096 字节），注入上下文。单 session 总预算 60KB

每轮用户 turn 开始时，`query.ts:301-304` 启动 memory prefetch（异步）；工具执行后 `1592-1614` 非阻塞收集结果，不卡主流程。

### 提取时机：stop hook，不是 autoCompact 后

触发位置（`stopHooks.ts:141-155`）：在 `handleStopHooks()` 中，fire-and-forget 触发提取和 Dream。教学版把提取放在 `stop_reason != "tool_use"` 分支里，方向一致。

CC 的提取通过 forked agent 执行（`extractMemories.ts:371-427`）：受限权限、`skipTranscript: true`、`maxTurns: 5`。还有重叠保护：如果主 Agent 已经写入了记忆文件，跳过提取。

### 记忆文件格式

CC 用 Markdown + YAML frontmatter，和教学版一致。四种类型：`user`、`feedback`、`project`、`reference`。

`memdir.ts:34-38` 定义索引约束：`MEMORY.md` 最多 200 行 / 25KB。`memdir.ts:199-266` 构建记忆行为指令，明确区分 memory、plan、tasks。存储位置：`~/.claude/projects/<sanitized-git-root>/memory/`。

### Dream：四层门控

不是"空闲时触发"或"数量够了就合并"，而是四层门控（`autoDream.ts`，默认值 `63-66`，门控逻辑 `130-190`）：

1. **时间门控**：距上次合并 ≥ 24 小时
2. **扫描节流**：避免频繁扫描文件系统
3. **会话门控**：自上次合并以来修改了 ≥ 5 个会话 transcript
4. **锁门控**：没有其他进程正在合并（`.consolidate-lock` 文件）

合并本身通过 forked agent 执行（`224-233`）：定位 → 收集近期信号 → 合并写文件 → 剪枝更新索引。锁文件 mtime 就是 lastConsolidatedAt。崩溃恢复：1 小时后锁自动过期。

### User Memory vs Session Memory

| | User Memory | Session Memory |
|---|---|---|
| 持久性 | 跨会话 | 单会话 |
| 存储 | `memory/` 下多个 .md 文件 | `session-memory/<id>/memory.md` |
| 加载到 | system prompt | compact 摘要 |
| 用途 | 跨会话的知识积累 | 跨 compact 的上下文连续性 |

sessionMemoryCompact（s08 中提到的机制）正是使用了 Session Memory：autoCompact 前先读 session memory 文件，如果内容足够（≥ 10K token、≥ 5 条文本消息、≤ 40K token，`sessionMemoryCompact.ts:56-61`），就用它做摘要，不调 LLM。

### 真实实现比教学版复杂的地方

- **Feature flags**：记忆相关功能有多层 feature gate 控制
- **Team memory**：团队共享记忆，`loadMemoryPrompt()` 有专门路径（教学版未涉及）
- **KAIROS**：时机感知的记忆提取策略，`loadMemoryPrompt()` 中 daily-log 模式
- **Prompt cache**：记忆注入需要考虑 prompt cache 的 TTL，避免每次都重写 system prompt 的大段内容
- **文件锁**：多进程并发时的锁机制
- **Memory prefetch**：异步预取，不阻塞主流程

### 教学版的简化是刻意的

- LLM side-query → LLM side-query + 关键词降级：教学版保留了 LLM 选择，加了降级路径
- 记忆 JSON → Markdown + frontmatter：教学版与 CC 一致
- stop hook 触发 → `stop_reason != "tool_use"` 分支：方向一致
- 四层门控 → 文件数阈值：教学版没有 transcript 系统和多会话概念
- forked agent + 受限权限 → 直接调用：教学版没有子进程隔离

</details>

<!-- upstream-cc-source:end -->
