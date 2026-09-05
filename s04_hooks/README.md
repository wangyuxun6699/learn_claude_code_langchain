# s04: Hooks — 挂在循环上，不写进循环里

s01 → s02 → s03 → `s04` → [s05](../s05_todo_write/) → s06 → ... → s16 → s17

> *"挂在循环上, 不写进循环里"* — hook 在工具执行前后注入扩展逻辑。
>
> **Harness 层**: hook — 扩展点不侵入循环。

---

## 问题

s03 已用 `PermissionMiddleware.wrap_tool_call()` 隔离权限检查。但如果继续为日志、大输出告警、输入处理和停止守卫各写一个专用 middleware 或循环分支，扩展点仍然是分散的，也缺少统一的注册与顺序语义。

循环很快就变成了这样：

```python
def agent_loop(messages):
    while True:
        # ... LLM call ...
        for block in response.content:
            if block.type != "tool_use":
                continue
            log_to_file(block)          # 加一行
            check_permission(block)     # 加一行
            notify_slack(block)         # 又加一行
            output = execute(block)
            auto_git_add(block)         # 再加一行
            # ... 很快循环就认不出来了
```

你想扩展的是 Agent 的行为，而不是不断增加专用控制流。循环和 middleware handler 应该是稳定核心，扩展逻辑通过统一的生命周期注册表挂在边界上。

---

## 解决方案

![Hooks Overview](images/hooks-overview.svg)

s04 保留 s03 的五个 `@tool`、`create_agent()`、流式输出和三道权限规则。结构上的变化是：`PermissionMiddleware` 升级为通用 `HookMiddleware`，在工具 handler 前后触发 `PreToolUse` / `PostToolUse`；CLI 在提交输入前触发 `UserPromptSubmit`；`agent_loop()` 在一次 LangGraph Agent 运行结束后触发 `Stop`。

四个事件，覆盖一个完整的 agent cycle：

| 事件 | 触发时机 | 典型用途 |
|------|---------|---------|
| UserPromptSubmit | 用户输入提交后、进入 LLM 前 | 输入验证、注入上下文 |
| PreToolUse | 工具执行前 | 权限检查、日志记录 |
| PostToolUse | 工具执行后 | 副作用（自动 git add 等）、输出检查 |
| Stop | 循环即将退出时 | 收尾清理、决定是否继续循环 |

扩展通过 `register_hook()` 添加。工具边界只负责触发 hook，不再知道权限、日志或告警的具体实现。

---

## 工作原理

**hook 注册表**：一个字典，事件名映射到回调列表。

```python
HOOKS = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": [],
}

def register_hook(event: str, callback):
    HOOKS[event].append(callback)

def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:   # 返回值 ≠ None → hook 说"停"
            return result
    return None
```

`PreToolUse` 返回非 `None` 时，本次工具执行被阻止；`Stop` 返回非 `None` 时，把该值作为新用户消息并再次运行 Agent。`UserPromptSubmit` 和 `PostToolUse` 的返回值不改变本章控制流。无论事件类型如何，非 `None` 都会短路同一事件中排在后面的 callback。

**UserPromptSubmit** 在用户输入提交后、进入 LLM 前触发。以下 hook 记录当前工作目录：

```python
def context_inject_hook(query: str) -> None:
    """在输入进入 Agent 前记录当前工作目录。"""
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")

register_hook("UserPromptSubmit", context_inject_hook)
```

在主循环中，用户输入后立即触发：

```python
query = input("s04 >> ")
trigger_hooks("UserPromptSubmit", query)   # ← 进入 LLM 之前
history.append({"role": "user", "content": query})
agent_loop(history)
```

**PreToolUse / PostToolUse** 位于 LangChain 工具 handler 的两侧。`ToolUseBlock` 把 LangChain 的 tool call 转为稳定的 `id / name / input` 视图；s03 的三道权限闸门成为 `permission_hook()`，另外注册日志和大输出提醒。

```python
class HookMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request: ToolCallRequest, handler):
        tool_call = request.tool_call
        block = ToolUseBlock(
            id=tool_call["id"],
            name=tool_call["name"],
            input=dict(tool_call.get("args", {})),
        )

        blocked = trigger_hooks("PreToolUse", block)
        if blocked is not None:
            return ToolMessage(
                content=str(blocked),
                tool_call_id=block.id,
                name=block.name,
                status="error",
            )

        result = handler(request)
        trigger_hooks("PostToolUse", block, _tool_output(result))
        return result
```

拒绝结果仍带原始 `tool_call_id` 返回模型，因此不会破坏 `AIMessage.tool_calls → ToolMessage` 协议。只有通过 `PreToolUse` 且 handler 已返回的调用会进入 `PostToolUse`；handler 抛出的异常仍按 LangChain 的原有方式传播。由于默认注册顺序是权限在前、日志在后，被权限 hook 阻止的调用不会再进入日志 hook。

**Stop** 在循环即将退出时触发。以下 hook 打印收尾统计：

```python
def summary_hook(messages: list) -> None:
    print(f"\033[90m[HOOK] Stop: session used {_tool_result_count(messages)} tool calls\033[0m")

register_hook("Stop", summary_hook)
```

`agent_loop()` 保留 s03 的 `stream()` 消费方式；每次 LangGraph Agent 运行结束后更新完整消息历史，再触发 Stop：

```python
while True:
    final_messages = messages
    for chunk in get_agent().stream(
        {"messages": messages},
        stream_mode=["messages", "values"],
        version="v2",
    ):
        # 输出 token，并从 values 事件取得完整 messages
        ...

    messages[:] = final_messages
    continuation = trigger_hooks("Stop", messages)
    if continuation is None:
        break
    messages.append({"role": "user", "content": str(continuation)})
```

四个 hook 覆盖 Agent cycle 的关键节点：输入 → 执行前 → 执行后 → 退出。LangChain 仍负责模型与工具循环，本章注册表只负责课程要观察的生命周期扩展。

---

## 相对 s03 的变更

| 组件 | 之前 (s03) | 之后 (s04) |
|------|-----------|-----------|
| 扩展方式 | 专用 `PermissionMiddleware` | `HOOKS` 注册表 + 通用 `HookMiddleware` |
| 权限逻辑 | `check_permission()` 由 middleware 直接调用 | s03 三道闸门由 `permission_hook()` 组合 |
| 新函数/类型 | — | `register_hook`、`trigger_hooks`、`ToolUseBlock` |
| hook callback | — | `context_inject_hook`、`permission_hook`、`log_hook`、`large_output_hook`、`summary_hook` |
| Agent 创建 | middleware 列表注册权限组件 | middleware 列表注册 hook 分发组件 |
| 工具循环 | `create_agent()` / LangGraph 管理 | 保持不变，Pre/Post 位于 handler 两侧 |
| 外层循环 | 一次 `stream()` 完成后返回 | Stop 可注入消息并再次 `stream()` |

---

## 结合本章代码理解 Middleware

本章把 s03 的专用权限 middleware 提升为一个最小生命周期系统。[`code.py`](code.py) 中的 `HOOKS` 保存四类 callback，`register_hook()` 负责注册，`trigger_hooks()` 按顺序执行，并在 callback 返回非 `None` 时短路。

### 四个事件分别位于哪里

| 事件 | 触发位置 | 本章用途 | 可改变什么 |
|---|---|---|---|
| `UserPromptSubmit` | CLI 把用户消息写入 history 前 | 输出当前工作目录 | 本章只观察输入，不改写 |
| `PreToolUse` | handler 执行前 | 权限检查、日志 | 阻止工具执行 |
| `PostToolUse` | handler 返回后 | 大输出告警 | 本章只观察结果，不改写 |
| `Stop` | 一次 `create_agent` 图运行完成后 | 统计工具调用数 | 返回内容时可要求再次运行 |

`HookMiddleware.wrap_tool_call()` 是关键边界：它先把 `request.tool_call` 转成 `ToolUseBlock`，触发 `PreToolUse`，再调用 `handler(request)`，最后把工具输出交给 `PostToolUse`。权限拒绝会直接返回合法的错误 `ToolMessage`；Agent loop 不知道权限和日志细节。

### 与 LangChain middleware 的一一对应

LangChain 的 `create_agent()` 运行在 LangGraph 之上，middleware hook 会成为已编译 Agent 图的一部分：

- `UserPromptSubmit` 在本章由 CLI 显式触发，位置接近 `before_agent`；若要让所有调用入口都生效，可迁移为框架 node-style hook。
- `PreToolUse` / `PostToolUse` 真实落在 `wrap_tool_call` 的 handler 前后，可检查、阻断和观察一次工具执行。
- 模型调用前后还可用 `before_model`、`after_model` 或 `wrap_model_call`；本章尚未拆出这些事件。
- `Stop` 由外层 `agent_loop()` 在图运行结束后触发；非 `None` 返回值会作为用户消息开启下一次图运行，因此它也承担终止守卫的作用。

与框架 middleware 相比，本章注册表没有 typed state、runtime context、流式 writer 和图级跳转能力；优点是执行顺序完全透明。到 s15 时，权限、上下文、恢复、后台通知等机制仍沿用“主循环稳定，机制挂在边界上”的原则。

### 编写 hook 的约束

- hook 可能被多次执行，日志和持久化操作应尽量幂等。
- `PreToolUse` 返回的拒绝原因会成为工具结果，应让模型能据此修正行动。
- 本章 `PostToolUse` 的返回值不会替换 handler 结果；要改写结果，需要显式扩展 `HookMiddleware` 的返回协议。
- 多个 hook 的顺序有语义；新增 hook 时要明确它是否应在拒绝动作上运行。

官方概念：[Middleware overview](https://docs.langchain.com/oss/python/langchain/middleware/overview) · [Built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in) · [Runtime context](https://docs.langchain.com/oss/python/langchain/runtime)

---

## 试一下

```sh
cd learn-claude-code
python s04_hooks/code.py
```

试试这些 prompt：

1. `Read the file README.md`（应该直接通过，观察 hook 日志）
2. `Create a file called test.txt`（通过后观察 PostToolUse 是否触发）
3. `Delete all temporary files in /tmp`（bash + rm 触发权限 hook）

观察重点：每次工具执行前，是否出现了 `[HOOK]` 日志？权限被拒时，是 hook 拦截的还是循环里硬编码的？

---

## 接下来

Agent 现在能安全执行操作了。但它有没有停下来想过"我应该先做什么，再做什么"？给它一个复杂任务，它是一上来就动手，还是先列个计划？

s05 TodoWrite → 给 Agent 一个计划工具。先列清单，再做。
---

<!-- upstream-cc-source:start -->
## 深入 CC 源码

> 原文：[s04_hooks](https://github.com/shareAI-lab/learn-claude-code/blob/67a9126c6435a8654ba7a6f68c0fd2130f00a462/s04_hooks/README.md)。以下折叠块保持原文，文中的章号与源码行号沿用该版本。

<details>
<summary>深入 CC 源码</summary>

> 以下基于 CC 源码 `toolHooks.ts`（650 行）、`hooks.ts`、`stopHooks.ts`、`coreTypes.ts` 的完整分析。

### 一、Hook 事件：不止这 4 个，而是 27 个

教学版只讲了 PreToolUse 和 PostToolUse。CC 实际有 27 个 hook 事件（`coreTypes.ts:25-53`）：

| 类别 | 事件 |
|------|------|
| 工具相关 | `PreToolUse`, `PostToolUse`, `PostToolUseFailure` |
| 会话相关 | `SessionStart`, `SessionEnd`, `Stop`, `StopFailure`, `Setup` |
| 用户交互 | `UserPromptSubmit`, `Notification`, `PermissionRequest`, `PermissionDenied` |
| 子 Agent | `SubagentStart`, `SubagentStop` |
| 压缩相关 | `PreCompact`, `PostCompact` |
| 团队相关 | `TeammateIdle`, `TaskCreated`, `TaskCompleted` |
| 其他 | `Elicitation`, `ElicitationResult`, `ConfigChange`, `WorktreeCreate`, `WorktreeRemove`, `InstructionsLoaded`, `CwdChanged`, `FileChanged` |

教学版只讲 4 个核心事件（UserPromptSubmit、PreToolUse、PostToolUse、Stop），因为它们覆盖了一个完整 agent cycle 的关键节点。其他 23 个都是同样的模式。

### 二、HookResult 常用字段摘录

CC 的 `HookResult`（`types/hooks.ts:260-275`）有 14 个字段，以下是常用字段：

| 字段 | 类型 | 用途 |
|------|------|------|
| `message` | Message | 可选 UI 消息 |
| `blockingError` | HookBlockingError | 阻塞错误 → 注入对话让模型自纠 |
| `outcome` | success/blocking/non_blocking_error/cancelled | 执行结果 |
| `preventContinuation` | boolean | 阻止后续执行 |
| `stopReason` | string | 停止原因描述 |
| `permissionBehavior` | allow/deny/ask/passthrough | hook 返回权限决策 |
| `updatedInput` | Record | 修改工具输入 |
| `additionalContext` | string | 附加上下文 |
| `updatedMCPToolOutput` | unknown | MCP 工具输出修改 |

### 三、关键不变式：Hook 'allow' 不能绕过 deny/ask 规则

这是 CC 权限系统最重要的安全设计（`toolHooks.ts:325-331`）：**hook 返回 allow 时，仍然要检查 settings.json 的 deny/ask 规则**。即使用户的 hook 脚本说"允许"，如果在 settings.json 中禁用了这个工具，操作仍然会被阻止。

教学版没有这个层次，只把 PreToolUse 的非 None 返回值解释为阻止本次工具执行。这在教学场景中够了，但在生产环境中会形成安全漏洞。

### 四、stopHookActive 机制

CC 的 Stop hooks 有一个防无限循环机制（`query.ts:212,1300`）：`stopHookActive` 状态字段。当 stop hooks 产生 blockingError 时，循环带 `stopHookActive: true` 重入下一轮。后续迭代中 stop hooks 看到这个标志就不会再次触发。这防止了一个永不停机的 bug：模型自纠后 stop hook 再次报错 → 模型再自纠 → stop hook 再报错...

### 五、hook_stopped_continuation

PostToolUse hooks 返回 `preventContinuation: true` 时，会产生一个 `hook_stopped_continuation` 附件（`toolHooks.ts:117-130`）。query.ts（L1388-1393）检测到后设置 `shouldPreventContinuation = true`，循环退出。这是 "hook 优雅地让 Agent 停机" 的机制，不是崩溃，是完成。

### 教学版的简化是刻意的

- 27 个事件 → 4 个（UserPromptSubmit/PreToolUse/PostToolUse/Stop）：覆盖 agent cycle 关键节点
- 14 个字段 → 简单的返回值（None = 继续，非 None = 阻止/续跑）：心智负担降到最低
- Hook allow vs deny/ask 不变式 → 省略：教学版没有 settings.json 层
- stopHookActive → 省略：教学版 Stop hook 只做简单续跑，不涉及防无限循环机制

</details>

<!-- upstream-cc-source:end -->
