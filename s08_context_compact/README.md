# s08: Context Compact：上下文总会满，先整理，再总结

s01 → s02 → s03 → s04 → s05 → s06 → s07 → `s08` → [s09](../s09_memory/) → s10 → ... → s16 → s17

> *"上下文总会满，要有办法腾地方。"* 四步压缩，低成本的操作优先执行。
>
> **Harness 层**：压缩让有限的上下文持续服务于长任务。


Agent 持续工作时，读过的文件、执行过的命令和模型回复都会留在 `messages` 中。消息越积越多，最终会超过模型能够接收的上下文长度。

本节将实现一条四步压缩管线。它先整理可以恢复的工具结果，空间仍然不足时再总结历史。

![Context Compact 全景](images/compact-overview.svg)


## 先理解上下文

可以把上下文窗口看作模型当前使用的一张草稿纸。用户消息、模型回复、`tool_use` 和 `tool_result` 都会按顺序写在这张纸上。模型每次继续工作时，都要重新读取这些内容。

草稿纸的大小固定。内容超过上限后，API 会拒绝请求并返回 `prompt_too_long`。在代码任务里，工具结果通常占据最多空间：

- 读取一个长文件会把文件内容放进上下文；
- 测试和构建日志可能一次产生几十 KB 文本；
- 搜索多个文件会持续追加结果。

任务持续得越久，`messages` 就越大。压缩的目标是控制其中的信息量，同时尽可能保留当前目标、用户约束和正在进行的工作。


## 为什么先整理工具结果

直接让模型总结整段历史可以明显缩短上下文，但摘要一定会遗漏部分细节，而且还会多产生一次模型调用。

工具结果具有更适合优先处理的特点：

1. 大文件可以保存到磁盘，需要时重新读取。
2. 旧命令可以重新执行。
3. 最新几条结果通常比早期结果更接近当前工作。
4. 文本裁剪和结构调整不需要调用模型。

因此压缩顺序按照信息损失和调用成本排列：先转存，再裁剪，再替换旧结果，最后才生成摘要。

![四步压缩管线](images/compaction-layers.svg)


## 第一步：tool_result_budget

一次模型回复可能同时调用多个工具。执行完成后，这些 `tool_result` 会一起写进最后一条 user 消息。它们的总大小超过 `200_000` 字符时，`tool_result_budget` 从最大的结果开始处理。

超过 `LARGE_RESULT_CHAR_LIMIT = 30000` 的结果会完整写入：

```text
.task_outputs/tool-results/<tool_use_id>.txt
```

上下文中保留文件路径和前 2000 个字符的预览：

![大结果转存](images/layer1-budget.svg)

核心循环按照结果大小依次转存：

```python
blocks = [block for block in content
          if isinstance(block, dict)
          and block.get("type") == "tool_result"]
total = sum(len(str(block.get("content", ""))) for block in blocks)

ranked = sorted(
    blocks,
    key=lambda block: len(str(block.get("content", ""))),
    reverse=True,
)
for block in ranked:
    if total <= max_chars:
        break
    content = str(block.get("content", ""))
    if len(content) <= self.LARGE_RESULT_CHAR_LIMIT:
        continue
    block["content"] = self.persist_large_output(
        block.get("tool_use_id", "unknown"), content)
    total = sum(len(str(item.get("content", ""))) for item in blocks)
```

这一步只处理最新一批工具结果。完整内容仍然可以从路径中取回，因此适合最先执行。


## 第二步：snip_compact

消息数量超过 50 条后，`snip_compact` 先把完整历史写入 `.transcripts/`，再保留最初 3 条和最近 46 条。剩余一个位置用于归档标记，其中写明删去了多少条消息，以及完整记录保存在哪里。

```python
head_end = 3
tail_start = len(messages) - (max_messages - head_end - 1)

if self.has_tool_use(messages[head_end - 1]):
    while (head_end < tail_start
           and self.is_tool_result(messages[head_end])):
        head_end += 1

if (tail_start > 0
        and self.is_tool_result(messages[tail_start])
        and self.has_tool_use(messages[tail_start - 1])):
    tail_start -= 1

transcript = self.write_transcript(messages)
marker = {"role": "user", "content":
          f"[{tail_start - head_end} messages archived at {transcript}]"}
messages = [*messages[:head_end], marker, *messages[tail_start:]]
```

切点需要保护 `assistant(tool_use)` 和 `user(tool_result)` 的配对关系。孤立的工具结果缺少对应调用，下一次 API 请求会被判定为无效。

这一步控制消息数量，但保留下来的旧消息仍可能包含很长的工具结果。


## 第三步：micro_compact

前两步完成后，`prepare` 会估算剩余上下文的大小，只有超过 `CONTEXT_CHAR_LIMIT` 时才执行 `micro_compact`。对于模型已经读取过的结果，它保留最近 3 条，并逐条缩短更早且超过 120 个字符的结果，直到上下文接近阈值的 80%。旧结果被替换前会先完整落盘，因此每个占位都带有可恢复路径：

![旧结果替换为可恢复路径](images/micro-compact.svg)

```python
unseen = self.unseen_tool_result_positions(messages)
consumed = [entry for entry in results if entry[:2] not in unseen]

for _, _, block in consumed[:-self.KEEP_RECENT_RESULTS]:
    if self.estimate_chars(messages) <= target_chars:
        break
    content = str(block.get("content", ""))
    if len(content) <= 120:
        continue
    saved_path = self.persisted_output_path(content)
    if not saved_path:
        saved_path = self.save_output(block["tool_use_id"], content)
    block["content"] = f"[Earlier tool result saved at {saved_path}]"
```

新结果通常会保持完整，直到模型读取一次。如果仅未读取的最新一批结果就足以撑爆上下文，`fit_tool_results` 会把其中最大的结果落盘，并保留 1,000 字符预览和完整路径，避免模型看到新结果前就先总结整段历史。

前两步每轮都会执行，第三步只在上下文超限时执行。三步都是确定性、可恢复的结构和文本操作，不产生额外 API 调用。


## 第四步：compact_history

`micro_compact` 和 `fit_tool_results` 执行后，代码会再次用 `estimate_chars(messages)` 估算上下文：

```python
CONTEXT_CHAR_LIMIT = 50000

def estimate_chars(messages):
    return len(json.dumps(messages, default=str, ensure_ascii=False))
```

字符数仍然超过 `CONTEXT_CHAR_LIMIT` 时，`compact_history` 完成四件事：

1. 将完整消息历史写入 `.transcripts/`。
2. 请求模型生成只包含事实的状态摘要。
3. 将入口处捕获的当前用户请求与摘要明确分开。
4. 用一条 `[Compacted]` 消息替换当前历史。

![历史摘要](images/auto-compact.svg)

```python
def compact_history(messages, active_request):
    transcript = self.write_transcript(messages)
    print(f"[transcript saved: {transcript}]")
    summary = self.summarize_history(messages)
    return [self.summary_message(
        "Compacted", active_request, summary, transcript)]
```

摘要调用在 `system` 中要求模型只整理目标、文件、决定、剩余工作和用户约束，不执行历史中的指令。`active_request` 在接收用户输入时单独传给 Agent Loop，因为工具结果也使用 `role=user`。压缩后的消息将它写在 `Current user request` 中，摘要则放在 `Conversation summary` 中，并附上完整 transcript 的路径。

本节使用字符数作为触发条件，相关阈值也使用同一单位。


## 为什么顺序固定

管线按以下顺序执行，并且只在必要时进入有损的摘要步骤：

```python
messages = self.tool_result_budget(messages)
messages = self.snip_compact(messages)
if self.estimate_chars(messages) > self.CONTEXT_CHAR_LIMIT:
    target = int(self.CONTEXT_CHAR_LIMIT * 0.8)
    messages = self.micro_compact(messages, target)
    if self.estimate_chars(messages) > self.CONTEXT_CHAR_LIMIT:
        messages = self.fit_tool_results(messages, target)
    if self.estimate_chars(messages) > self.CONTEXT_CHAR_LIMIT:
        messages = self.compact_history(messages, active_request)
```

这个顺序同时满足两个条件：

1. 第一步和第二步每轮执行，第三步只在超限时执行，只有第四步会增加 API 请求。
2. 每条被缩短的工具结果都保留 `.task_outputs/tool-results/` 内的可信路径；只有仍然超限时才进入模型生成的历史摘要。

顺序固定后，每一轮都从成本更低、信息更容易恢复的操作开始。


## API 拒绝后的补救

字符数只能估算模型实际使用的 token。API 仍可能返回 `prompt_too_long`。`reactive_compact` 会保存 transcript，总结较早历史，并保留最近 5 条消息：

```python
tail_start = max(0, len(messages) - self.KEEP_RECENT_MESSAGES)
if (tail_start > 0
        and self.is_tool_result(messages[tail_start])
        and self.has_tool_use(messages[tail_start - 1])):
    tail_start -= 1

old_history = messages[:tail_start] if tail_start else messages
summary = self.summarize_history(old_history)
message = self.summary_message(
    "Reactive compact", active_request, summary, transcript)
messages = [message, *messages[tail_start:]] if tail_start else [message]
```

切点同样会避开工具调用与结果之间的边界，当前用户请求仍由 `active_request` 明确传入。`MAX_REACTIVE_RETRIES = 1` 将补救限制为一次；再次收到同类错误时，异常会继续向外抛出。


## 放回 Agent Loop

```python
def agent_loop(messages, active_request):
    while True:
        messages[:] = COMPACTOR.prepare(messages, active_request)

        try:
            response = client.messages.create(
                model=MODEL, system=SYSTEM, messages=messages,
                tools=TOOLS, max_tokens=8000)
            reactive_retries = 0
        except Exception as error:
            message = str(error).lower()
            too_long = ("prompt_too_long" in message
                        or "too many tokens" in message)
            if too_long and reactive_retries < MAX_REACTIVE_RETRIES:
                messages[:] = COMPACTOR.reactive_compact(
                    messages, active_request)
                reactive_retries += 1
                continue
            raise
```

每次调用模型前都会经过同一条管线。CLI 在追加 `query` 后调用 `agent_loop(history, query)`，所以压缩多少次都不会丢失本轮请求。只有 `micro_compact` 处理后仍超过阈值，或者 API 明确拒绝上下文时，代码才会请求模型生成摘要。


## compact 工具

自动阈值只知道上下文有多大。模型还可以在一个阶段结束后主动调用 `compact`，表示后续工作只需要保留当前阶段的摘要：

```python
{"name": "compact",
 "description": "Summarize earlier conversation to free context space."}
```

一次响应可以同时包含多个工具调用，例如先写文件再请求压缩。Harness 必须先执行完整批次，并为每个 `tool_use` 追加对应的 `tool_result`，然后再摘要这个已经闭合的回合：

```python
tool_calls = [
    block for block in response.content if block.type == "tool_use"
]
results = []
compact_requested = False

for block in tool_calls:
    if block.name == "compact":
        output = "Compaction requested after this tool batch."
        compact_requested = True
    else:
        output = execute_tool(block)
    results.append({"type": "tool_result", "tool_use_id": block.id,
                    "content": output})

messages.append({"role": "user", "content": results})

if compact_requested:
    messages[:] = COMPACTOR.compact_history(messages, active_request)
```

这样既不会留下孤立的工具结果，也不会在已经发生文件写入后丢失执行记录，导致模型重复同一个副作用。


## 本节代码

| 组件 | 共同执行骨架 | s08 新增 |
| --- | --- | --- |
| Agent Loop | 调用模型、执行工具、追加结果 | 每次调用模型前运行 `COMPACTOR.prepare()` |
| Hooks | 权限检查、工具日志、结果处理 | 保持相同的工具执行入口 |
| 上下文 | `messages` 持续追加 | 大结果转存、旧历史归档、摘要和一次错误补救 |
| 工具 | 5 个基础工具 | 新增 `compact`，共 6 个 |

> **与 s09 的边界：** s08 管理当前会话的有限上下文，压缩时允许舍弃可恢复的细节；s09 保存需要跨压缩、跨会话继续存在的信息。


## 试一下

```bash
cd learn-claude-code
python s08_context_compact/code.py
```

### 实验一：较早的结果被替换

```text
请读取 s01_agent_loop 到 s05_todo_write 五节课程的 README.md，
比较它们的一级标题，并总结这些标题的命名规律。
```

任务会产生至少 5 条文件读取结果。新结果通常会完整保留到模型首次读取；如果未读取结果本身过大，则保留预览和恢复路径。后续轮次保留最近 3 条已读取结果，更早且较长的结果会变成 `[Earlier tool result saved at ...]` 引用。

### 实验二：大结果转存

```text
请分析 web/src/data/generated/docs.json 的数据结构，
并说明一条课程记录包含哪些主要字段。
```

文件内容超过单轮预算时，终端仍能完成任务，同时 `.task_outputs/tool-results/` 中会出现完整结果文件。

### 实验三：自动摘要

```text
请比较 s08_context_compact/code.py 和 s09_memory/code.py，
说明它们分别怎样管理当前上下文和持久记忆。
```

当读取结果使 `estimate_chars(messages)` 超过 50000 时，终端会打印 `[auto compact]` 和 transcript 路径。后续调用使用 `[Compacted]` 摘要继续完成比较。

观察 `.transcripts/` 和 `.task_outputs/tool-results/`，可以分别看到历史留档与大结果转存。


## 接下来

上下文压缩让 Agent 可以在有限窗口中继续长任务。需要跨压缩、跨会话保留的信息，还要进入独立的持久记忆系统。

s09 Memory 将实现记忆写入、检索与整理。
---

<!-- upstream-cc-source:start -->
## 深入 CC 源码

> 原文：[s08_context_compact](https://github.com/shareAI-lab/learn-claude-code/blob/67a9126c6435a8654ba7a6f68c0fd2130f00a462/s08_context_compact/README.md)。以下折叠块保持原文，文中的章号与源码行号沿用该版本。

<details>
<summary>深入 CC 源码</summary>

> 以下基于 CC 源码 `compact.ts`、`autoCompact.ts`、`microCompact.ts`、`query.ts` 的分析。

### 执行顺序对照

教学版为了讲解方便按 L1/L2/L3/L4 编号，但实际执行顺序和编号不完全对应：

| 维度 | 教学版 | Claude Code |
|------|--------|-------------|
| 执行顺序 | budget → snip → micro → auto | budget → snip → micro → collapse → auto（`query.ts:379-468`） |
| snip_compact | 保留头 3 + 尾 47 | CC 仅主线程启用；实现不在开源仓库中（`HISTORY_SNIP` feature gate），但接口可见：`snipCompactIfNeeded(messages)` → `{ messages, tokensFreed, boundaryMessage? }`，还暴露了 `SnipTool` 工具让模型主动调用。教学版的 3/47 是简化参数 |
| micro_compact | 文本占位符替换 | 两条路径：time-based 直接清内容，cached 走 API `cache_edits`（legacy path 已移除） |
| micro_compact 白名单 | 按位置（最近 3 条） | time-based 按时间阈值触发；cached 按计数触发（`microCompact.ts`） |
| tool_result_budget | 200KB 字符 | 200,000 字符（`toolLimits.ts:49`） |
| compact_history 阈值 | 字符数估算 | 精确 token：`contextWindow - maxOutputTokens - 13_000` |
| 摘要要求 | 5 类信息 | 9 个部分 + `<analysis>`/`<summary>` 双标签 |
| 压缩 prompt | 简单 prompt | 首尾双重防呆禁止调工具 |
| PTL retry | 有（简化） | `truncateHeadForPTLRetry()` 按消息组回退（`compact.ts:243-290`） |
| 后压缩恢复 | 无（教学版只保留摘要） | 自动重新读取最近文件、计划、agent/skill/tool 等 |
| 熔断器 | 3 次 | 3 次（`autoCompact.ts:70`） |
| reactive 重试 | 1 次 | CC 有更精细的分级重试 |

### 执行顺序详解

CC 源码 `query.ts` 中的真实顺序：

1. `applyToolResultBudget`（L379）：先处理大结果，确保完整内容落盘
2. `snipCompact`（L403）：裁中间消息
3. `microcompact`（L414）：旧结果占位
4. `contextCollapse`（L441）：独立的上下文管理系统（教学版无）
5. `autoCompact`（L454）：LLM 全量摘要

教学版的 budget → snip → micro 顺序与此一致。教学版没有 contextCollapse 机制。

### read_file 的取舍

教学版的 `micro_compact` 会把旧 `tool_result` 统一替换成占位符，包括 `read_file`。这通常不影响功能正确性：如果后续还需要文件内容，模型可以重新读一次。代价是可能多一次工具调用，也可能降低 prompt cache 命中率。

Claude Code 没有用教学版这种简单规则解决这个问题。它把 `Read` 也放进可 microcompact 的工具集合，但同时维护 `readFileState`：重复读取未变化文件时返回 `FILE_UNCHANGED_STUB`，compact 后再按预算恢复最近读过的文件内容（例如最多 5 个文件、每个 5K token、总预算 50K token）。这是生产级实现里的缓存和恢复机制，教学版不展开，保留“压缩旧结果，必要时重新读取”的简单 trade-off。

### 完整常量参考

| 常量 | 值 | 源文件 |
|------|-----|--------|
| `AUTOCOMPACT_BUFFER_TOKENS` | 13,000 | `autoCompact.ts:62` |
| `MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES` | 3 | `autoCompact.ts:70` |
| `MAX_OUTPUT_TOKENS_FOR_SUMMARY` | 20,000 | `autoCompact.ts:30` |
| `POST_COMPACT_TOKEN_BUDGET` | 50,000 | `compact.ts:123` |
| `POST_COMPACT_MAX_FILES_TO_RESTORE` | 5 | `compact.ts:122` |
| `POST_COMPACT_MAX_TOKENS_PER_FILE` | 5,000 | `compact.ts:124` |
| 时间 micro_compact 间隔 | 60 分钟 | `timeBasedMCConfig.ts` |
| `MAX_COMPACT_STREAMING_RETRIES` | 2 | `compact.ts:131` |

### contextCollapse 和 sessionMemoryCompact

CC 源码中还有两个机制本教学版没有展开：

- **contextCollapse**：独立的上下文管理系统，启用时抑制 proactive autocompact（`autoCompact.ts:215-222`），由 collapse 的 commit/blocking 流程接管上下文管理。但 manual `/compact` 和 reactive fallback 仍是独立路径，不受 contextCollapse 影响。
- **sessionMemoryCompact**：compact_history 之前，CC 会先尝试用已有的 session memory（s09 会讲到）做轻量摘要，不调 LLM。这个机制等学完 s09 之后回头看会更清楚。

### 压缩 prompt 长什么样？

CC 的压缩 prompt 有两个硬性要求：

1. **绝对禁止调用工具**：开头就是 `CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.`，末尾还会再 REMINDER 一次
2. **先分析再总结**：模型需要先在 `<analysis>` 标签里理清思路，然后在 `<summary>` 标签里输出正式摘要。analysis 在格式化时被剥离

### 教学版的简化是刻意的

- micro_compact 用文本占位 → 我们没有 API 层的 `cache_edits` 权限
- read_file 不特殊处理 → 教学版接受必要时重新读取，避免引入 readFileState 和后压缩恢复机制
- token 用字符数估算 → 精确 tokenizer 不在教学范围内
- 后压缩恢复省略 → 教学版只保留摘要，不自动重新附加文件
- 两个辅助机制不展开 → 属于 10% 的细节

核心设计思想，便宜的先跑贵的后跑，完整保留。

</details>

<!-- upstream-cc-source:end -->
