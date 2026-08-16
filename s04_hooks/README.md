# s04: Hooks — 挂在循环上，不写进循环里

> LangChain / LangGraph 教学改编版。章节结构与“深入 CC 源码”主要参考
> [shareAI-lab/learn-claude-code 的 s04](https://github.com/shareAI-lab/learn-claude-code/blob/main/s04_hooks/README.md)。
>
> *“挂在循环上，不写进循环里”*——横切逻辑通过 middleware 接入 Agent 生命周期。
>
> **Harness 层**：扩展点不侵入核心模型—工具循环。

[s03](../s03_permission/) → `s04` → [s05](../s05_todo_write/)

---

## 问题

s03 已经有权限检查，但日志、权限、审计、统计和收尾逻辑如果继续写进每个工具或 `agent_loop()`，核心流程会迅速膨胀：

```python
def agent_loop(messages):
    result = agent.invoke({"messages": messages})

    log_user_input(messages)        # 新增日志
    check_permission(result)        # 新增权限
    notify_external_system(result)  # 新增通知
    auto_git_add(result)            # 新增副作用
    write_audit_record(result)      # 新增审计
```

这些逻辑都不是“调用模型和工具”本身，却不断迫使我们修改核心循环。它们属于横切关注点：每次工具调用都可能需要，但不应该复制到每个工具里。

LangChain 的 `create_agent()` 已经把标准 Agent 循环编译成 LangGraph：

```text
model → tools → model → ... → final answer
```

如果为了加日志重新手写这条循环，就失去了框架提供的消息配对、并行工具执行、状态 reducer 和 middleware 组合能力。

---

## 解决方案

![s04: Hooks — 挂在循环上，不写进循环里](images/hooks-overview.svg)

本章使用两层扩展机制：

```text
LangChain middleware 生命周期
    ├─ @before_agent
    ├─ @wrap_tool_call
    └─ @after_agent
             ↓
本地 HOOKS 注册表
    ├─ UserPromptSubmit callbacks
    ├─ PreToolUse callbacks
    ├─ PostToolUse callbacks
    └─ Stop callbacks
```

第一层由 LangChain/LangGraph 决定“什么时候执行”；第二层由本章的 `register_hook()` 决定“这个时机要执行哪些业务回调”。

这样做的好处是：

- `create_agent` 的核心循环不需要修改；
- 工具函数不需要知道日志、审计和权限系统；
- 同一个生命周期点可以注册多个回调；
- 新增回调时不必重新装配整个 Agent；
- LangChain middleware 仍然可以与重试、HITL、摘要等其他 middleware 组合。

---

## 四个教学事件如何映射到 LangChain

| 教学事件 | 当前 LangChain 接入点 | 实际触发次数 | 当前用途 |
|---|---|---:|---|
| `UserPromptSubmit` | `@before_agent` | 每次 `agent.invoke()` 一次 | 打印最新输入 |
| `PreToolUse` | `@wrap_tool_call` 调用 `handler` 前 | 每个工具调用一次 | 日志、权限拒绝 |
| `PostToolUse` | `@wrap_tool_call` 调用 `handler` 后 | 每个正常返回的工具调用一次 | 结果日志、审计 |
| `Stop` | `@after_agent` | Agent 正常结束时一次 | 收尾统计 |

它们覆盖了本章命令行程序的一次完整调用：

```text
用户输入
   ↓
agent.invoke(...)
   ↓
before_agent
   └─ UserPromptSubmit
   ↓
模型调用
   ↓
wrap_tool_call
   ├─ PreToolUse
   ├─ handler(request) → 真实工具
   └─ PostToolUse
   ↓
模型可能继续调用其他工具
   ↓
after_agent
   └─ Stop
```

这里的名字是在模拟 CC hook 语义，并不是 LangChain 原生事件名。例如，LangChain 的 `before_agent` 并不知道什么叫 `UserPromptSubmit`；是本章在该生命周期点主动调用了：

```python
trigger_hooks("UserPromptSubmit", content)
```

---

## LangChain middleware 的两种 hook 风格

LangChain middleware 分成 node-style 和 wrap-style 两类。

### Node-style：在固定节点顺序执行

| Hook | 时机 |
|---|---|
| `before_agent` | Agent 开始前，每次 invocation 一次 |
| `before_model` | 每次模型调用前 |
| `after_model` | 每次模型返回后 |
| `after_agent` | Agent 完成后，每次 invocation 一次 |

Node-style hook 适合日志、验证和状态更新。返回字典时，LangGraph 会通过对应 reducer 把内容合并进 Agent state；返回 `None` 表示只观察、不更新状态。

### Wrap-style：包住一次真实调用

| Hook | 包住的对象 |
|---|---|
| `wrap_model_call` | 每次模型 API 调用 |
| `wrap_tool_call` | 每次工具执行 |

Wrap-style hook 控制 `handler` 何时被调用：

```text
调用 0 次 → 短路、拒绝或返回缓存
调用 1 次 → 正常执行
调用多次 → 重试或多路尝试
```

本章的工具 hook 在拒绝时不调用 `handler`，允许时调用一次。官方说明参见
[LangChain Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)
和 [`wrap_tool_call` API](https://reference.langchain.com/python/langchain/agents/middleware/types/AgentMiddleware/wrap_tool_call)。

---

## 工作原理：本地 Hook 注册表

LangChain middleware 负责接入图生命周期，但 `HOOKS` 字典、`register_hook()` 和 `trigger_hooks()` 都是本章自己实现的轻量注册表：

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

        if result is not None:
            return result

    return None
```

回调按注册顺序执行。第一个返回非 `None` 的回调会终止同一事件后面的回调。

这对 `PreToolUse` 很重要：

```text
permission hook 返回 "Permission denied"
    ↓
trigger_hooks 立即返回
    ↓
后续 PreToolUse callbacks 不再执行
    ↓
真实工具不执行
```

但也要注意：这个“遇到非 `None` 就停止”的规则会应用到所有事件。如果将来希望所有 PostToolUse 日志回调都执行，即使其中某个返回了值，就应该为不同事件设计不同聚合策略，而不是共享当前的短路语义。

注册过程：

```python
register_hook(
    "UserPromptSubmit",
    on_user_prompt_submit,
)
register_hook(
    "PreToolUse",
    on_pre_tool_use,
)
register_hook(
    "PostToolUse",
    on_post_tool_use,
)
register_hook(
    "Stop",
    on_stop,
)
```

---

## UserPromptSubmit：使用 before_agent

当前代码通过 `@before_agent` 在每次 `agent.invoke()` 开始前读取最后一条消息：

```python
@before_agent
def user_prompt_submit(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    messages = state.get("messages", [])

    if messages:
        last = messages[-1]
        content = (
            last.get("content")
            if isinstance(last, dict)
            else getattr(last, "content", None)
        )
        trigger_hooks(
            "UserPromptSubmit",
            content,
        )

    return None
```

命令行程序每收到一次用户输入，就调用一次 `agent.invoke()`，所以这个映射在当前程序中等价于“用户提交输入后触发一次”。

不过这不是严格的消息类型检查。函数只是读取最后一条消息，没有验证它一定是 `HumanMessage`。如果其他调用方以 AIMessage 或 ToolMessage 作为最后一条输入，hook 仍会触发。

当前 `on_user_prompt_submit()` 只打印日志：

```python
def on_user_prompt_submit(content):
    print("[UserPromptSubmit]", content)
```

而且 `user_prompt_submit()` 忽略了 `trigger_hooks()` 的返回值。因此本章目前不能通过这个本地 callback 修改用户输入。要真正注入上下文，应由 middleware 返回 state update，例如：

```python
from langchain_core.messages import HumanMessage


@before_agent
def inject_context(state, runtime):
    return {
        "messages": [
            HumanMessage(
                content=f"Current workspace: {WORKDIR}"
            )
        ]
    }
```

`messages` 字段使用 LangGraph 的消息 reducer，因此这里是追加消息，而不是覆盖历史。

---

## PreToolUse / PostToolUse：使用 wrap_tool_call

`wrap_tool_call` 是本章最核心的 LangChain 适配点：

```python
@wrap_tool_call
def tool_hook(
    request: ToolCallRequest,
    handler: Callable[
        [ToolCallRequest],
        ToolMessage | Command,
    ],
) -> ToolMessage | Command:
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get(
        "args",
        {},
    )

    blocked = trigger_hooks(
        "PreToolUse",
        tool_name,
        tool_args,
    )

    if blocked:
        return ToolMessage(
            content=str(blocked),
            tool_call_id=request.tool_call["id"],
            name=tool_name,
            status="error",
        )

    result = handler(request)

    trigger_hooks(
        "PostToolUse",
        tool_name,
        tool_args,
        result,
    )

    return result
```

`ToolCallRequest` 提供：

| 字段 | 内容 |
|---|---|
| `tool_call` | 工具名、参数、tool call ID |
| `tool` | 解析后的 `BaseTool`，可能为 `None` |
| `state` | 当前 Agent state |
| `runtime` | 当前 Tool runtime |

### PreToolUse 如何阻止工具

权限回调复用了 s03 的 `check_permission()`：

```python
def on_pre_tool_use(tool_name, tool_args):
    print("[PreToolUse]", tool_name, tool_args)

    if not check_permission(
        tool_name,
        tool_args,
    ):
        return "Permission denied"
```

一旦返回字符串，`tool_hook` 就直接构造与原始 tool call ID 配对的 error `ToolMessage`，不会调用 `handler(request)`。模型能够看到拒绝原因，并选择其他方案。

### PostToolUse 能看到什么

允许执行时，`handler(request)` 返回 `ToolMessage` 或 `Command`。当前回调打印工具名、参数以及结果内容：

```python
def on_post_tool_use(
    tool_name,
    tool_args,
    result,
):
    print(
        "[PostToolUse]",
        tool_name,
        tool_args,
    )
    print(
        "result:",
        getattr(result, "content", result),
    )
```

当前实现只有在 `handler()` 正常返回时才触发 `PostToolUse`。如果工具抛出未处理异常，异常会向外传播，后置 callback 不会执行。生产实现通常显式区分成功和失败：

```python
try:
    result = handler(request)
except Exception as exc:
    trigger_hooks(
        "PostToolUseFailure",
        tool_name,
        tool_args,
        exc,
    )
    raise
else:
    trigger_hooks(
        "PostToolUse",
        tool_name,
        tool_args,
        result,
    )
    return result
```

---

## Stop：使用 after_agent

当前实现把 `Stop` 映射到 `@after_agent`：

```python
@after_agent
def stop_hook(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    trigger_hooks(
        "Stop",
        state.get("messages", []),
    )
    return None
```

`on_stop()` 只统计当前消息数量：

```python
def on_stop(messages):
    print("[Stop]", len(messages))
```

这与上游手写循环版存在一个重要差异：

| 能力 | 上游手写循环 | 当前 LangChain 版 |
|---|---|---|
| 结束前执行收尾逻辑 | 支持 | 支持 |
| Stop callback 返回字符串 | 可注入消息并继续循环 | 返回值当前被忽略 |
| 防止 Stop 无限续跑 | 需要 `stopHookActive` | 当前没有续跑能力 |

`after_agent` 发生在 Agent 循环已经结束之后，所以不能简单地把本地 callback 的字符串当作“继续模型循环”。如果需要从 middleware 控制流程，应使用支持 `jump_to` 的 node-style hook、返回 `Command`，或者在外层 `StateGraph` 中显式建边；不能只修改 `on_stop()` 的返回值。

---

## 两层 Hook 为什么都有必要

可以只写多个 LangChain middleware：

```python
middleware=[
    permission_middleware,
    audit_middleware,
    metrics_middleware,
]
```

也可以像本章一样，在一个生命周期 middleware 内再维护本地 callback 注册表。两种方式定位不同：

| 层次 | 更适合 |
|---|---|
| 独立 LangChain middleware | 需要自定义 state、Runtime context、图跳转、复用到多个 Agent |
| 本地 `HOOKS` callback | 简单同步日志、小型教学扩展、运行时注册多个轻量回调 |

当前组合的价值是：用 LangChain middleware 获得稳定的生命周期边界，再用本地注册表演示 CC 风格的事件订阅。

---

## middleware 的装配与执行顺序

三个装饰器都会生成可注册的 middleware 对象：

```python
MIDDLEWARE = [
    user_prompt_submit,
    tool_hook,
    stop_hook,
]

agent = create_agent(
    model=MODEL,
    tools=TOOLS,
    system_prompt=SYSTEM,
    middleware=MIDDLEWARE,
)
```

LangChain 的组合规则是：

- `before_*` 按列表从前到后执行；
- `after_*` 按列表从后到前执行；
- `wrap_*` 像函数一样嵌套，第一个 middleware 在最外层。

本章三个 middleware 分属不同阶段，所以彼此嵌套关系不复杂。以后加入权限、人审、重试和审计 middleware 时，顺序会直接影响“谁先看到请求”和“谁先看到结果”。

Middleware 不是脱离 LangGraph 的另一套运行时。`create_agent()` 把这些 hook 一起编译进图；即使整个 Agent 被放进更大的 StateGraph 作为节点，middleware 仍然随节点执行。参见
[LangChain middleware overview](https://docs.langchain.com/oss/python/langchain/middleware/overview)。

---

## 相对 s03 的变化

| 组件 | s03 | s04 |
|---|---|---|
| 权限位置 | 工具函数或单个权限 middleware | 注册为 `PreToolUse` callback |
| 生命周期扩展 | 主要关注工具执行前 | 输入、工具前后、Agent 结束 |
| 核心 Agent 循环 | `create_agent` | 仍由 `create_agent` 负责 |
| 新增注册表 | 无 | `HOOKS` |
| 新增函数 | 无 | `register_hook()`、`trigger_hooks()` |
| 工具实现 | 可能包含横切逻辑 | 不直接知道 hook |
| 拒绝方式 | 返回权限错误 | PreToolUse 短路并返回 error ToolMessage |
| 收尾行为 | 无统一入口 | `after_agent` → `Stop` |

核心变化不是“重新写一个循环”，而是把 s03 的权限判断接入 `wrap_tool_call`，再补上 Agent 前后生命周期点。

---

## 本章文件

- `code.py`：带注释教学版（可直接运行），包含较完整的中文解释；
- `code_uncommented.py`：保留必要 docstring 的精简完整版本；

三个入口使用同一套 `HOOKS`、工具和 middleware 装配。

---

## 试一下

在仓库根目录准备环境：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

运行任一版本：

```powershell
python -m s04_hooks.code
python -m s04_hooks.code_uncommented
```

建议测试：

1. `Read the file README.md`
2. `Create a file called test.txt`
3. `Delete all temporary files in /tmp`
4. `Read a file outside the current workspace`

观察重点：

- 每次 invocation 开始时是否打印 `[UserPromptSubmit]`；
- 每个工具执行前是否打印 `[PreToolUse]`；
- 权限被拒绝后真实工具是否没有执行；
- 工具成功后是否打印 `[PostToolUse]`；
- Agent 正常结束时是否打印 `[Stop]`；
- 被拒绝的工具是否收到 `status="error"` 的 ToolMessage。

> 这些教学 Agent 可以执行命令和修改文件。请在测试工作区中运行，并认真检查权限提示。

---

## 教学版边界

- `HOOKS` 是进程内全局可变字典，没有持久化和跨进程同步；
- callback 是同步函数，耗时 callback 会阻塞当前 Agent；
- `trigger_hooks()` 对所有事件都采用“第一个非 None 返回值即短路”；
- UserPromptSubmit、PostToolUse 和 Stop 的 callback 返回值目前没有被利用；
- 未实现 `PostToolUseFailure`；
- 未实现 hook 超时、重试、隔离和错误降级；
- `ask_user()` 使用阻塞式终端输入，不是可恢复的 LangGraph interrupt；
- Stop 只做收尾，不能强制 Agent 续跑；
- 权限 allow 不能覆盖 deny/ask 的生产安全不变式尚未实现。

---

## 接下来

Agent 已经有了稳定扩展点。接下来需要让它面对复杂目标时先列计划，再逐项推进。

s05 TodoWrite → 给 Agent 一个计划工具。先列清单，再做。

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

