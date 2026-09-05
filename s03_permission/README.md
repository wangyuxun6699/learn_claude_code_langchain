# s03: Permission — 执行前做权限判断

s01 → s02 → `s03` → [s04](../s04_hooks/) → s05 → ... → s16 → s17
> *"工具执行前先做权限判断"* — 权限管线决定哪些操作需要审批。
>
> **Harness 层**: 权限 — 在工具执行前加一道门。

---

## 问题

s02 的 Agent 有 5 个工具。file tools 受 `safe_path` 保护，但 bash 不受限制。让它"清理一下项目"，可能执行 `rm -rf /`。

安全边界由代码负责，判断发生在工具执行之前。

---

## 解决方案

![Permission Overview](images/permission-overview.svg)

s02 的 `create_agent()` 与流式循环完全保留。唯一的结构性变动是注册 `PermissionMiddleware`，由它在每个工具真正执行前调用 `check_permission()`。每个工具调用依次经过三道闸门：硬拒绝优先，软询问次之，都没命中就放行。

三道闸门对应三种决策：

| 闸门 | 作用 | 命中后 |
|------|------|--------|
| 1. 拒绝列表 | 永远禁止的操作（`rm -rf /`、`sudo`） | 直接拒绝，不执行 |
| 2. 规则匹配 | 取决于上下文的操作（读/写工作区外、`rm` 文件） | 交给闸门 3 |
| 3. 用户审批 | 闸门 2 命中后，暂停等用户确认 | 用户决定允许或拒绝 |

三道都没命中 → 直接执行。大部分日常操作走这条路。

---

## 工作原理

![Permission Pipeline](images/permission-pipeline.svg)

**闸门 1**：一张硬拒绝表，先查，命中就返回阻止信息。这张表使用简单字符串匹配来说明权限闸门的位置，不能视为完整的安全边界。

```python
DENY_LIST = [
    "rm -rf /", "sudo", "shutdown", "reboot",
    "mkfs", "dd if=", "> /dev/sda",
]

def check_deny_list(command: str) -> str | None:
    normalized = command.lower()
    for pattern in DENY_LIST:
        if pattern in normalized:
            return f"Blocked: '{pattern}' is on the deny list"
    return None
```

**闸门 2**负责规则匹配，用来描述"什么时候需要问用户"。每条规则指定工具和检查条件。

```python
import re

DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)

def contains_destructive_command(command: str) -> bool:
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))

PERMISSION_RULES = [
    {
        "tools": ["read_file", "write_file", "edit_file"],
        "check": lambda args: not resolve_path(args.get("path", "")).is_relative_to(WORKDIR),
        "message": "Access outside workspace",
    },
    {
        "tools": ["bash"],
        "check": lambda args: contains_destructive_command(args.get("command", "")) or any(
            kw in args.get("command", "").lower()
            for kw in ["rm ", "> /etc/", "chmod 777"]
        ),
        "message": "Potentially destructive command",
    },
]

def check_rules(tool_name: str, args: dict) -> str | None:
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None
```

**闸门 3**：规则命中后，暂停等用户输入。

```python
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    print(f"\n⚠  {reason}")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"
```

**三道闸门串在一起**，由 middleware 插在工具执行之前：

```python
def check_permission(tool_name: str, args: dict) -> bool:
    # 闸门 1: 硬拒绝
    if tool_name == "bash":
        reason = check_deny_list(args.get("command", ""))
        if reason:
            print(f"\n⛔ {reason}")
            return False

    # 闸门 2 + 3: 规则匹配 → 用户审批
    reason = check_rules(tool_name, args)
    if reason:
        decision = ask_user(tool_name, args, reason)
        if decision == "deny":
            return False

    return True

class PermissionMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request: ToolCallRequest, handler):
        tool_call = request.tool_call
        if not check_permission(tool_call["name"], tool_call.get("args", {})):
            return ToolMessage(
                content="Permission denied.",
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
                status="error",
            )
        return handler(request)

agent = create_agent(
    model=model,
    tools=TOOLS,
    system_prompt=SYSTEM,
    middleware=[PermissionMiddleware()],
)
```

`handler(request)` 是真正进入工具执行节点的边界：允许时调用它；拒绝时不调用，并直接构造与原调用 ID 配对的 `ToolMessage`。这样权限拒绝仍是一次完整的工具结果，不会破坏 `AIMessage.tool_calls → ToolMessage` 协议。

---

## 相对 s02 的变更

| 组件 | 之前 (s02) | 之后 (s03) |
|------|-----------|-----------|
| 安全模型 | 文件工具仅由 `safe_path()` 硬限制 | deny / ask / allow 三道闸门 |
| 新组件 | — | `PermissionMiddleware` + 四个权限函数 |
| 工具执行 | `create_agent()` 直接分发 | `wrap_tool_call()` 审核后再调用 handler |
| Agent Loop | `stream()` 消费事件 | 完全不变 |

---

## 结合本章代码理解人工审批

[`code.py`](code.py) 把权限拆成三道闸门，而不是把“是否安全”交给模型自己判断：

1. `check_deny_list()` 处理无条件禁止的命令，命中后不能通过用户确认绕过。
2. `check_rules()` 根据工具名和参数判断风险，例如工作区外路径或破坏性 shell 命令。
3. `ask_user()` 只对需要确认的动作询问用户，默认答案是拒绝。

`check_permission(tool_name, args)` 位于 handler 调用之前。拒绝时仍然生成与原调用 ID 对应的 `ToolMessage`。权限拒绝也是一次合法的工具结果，而不是偷偷删除模型的调用，否则消息历史会出现“有 tool call、没有 ToolMessage”的协议断裂。

### LangChain 官方 middleware 模型

LangChain 把 middleware 定义为 Agent 执行图上的扩展点。官方文档把 hook 分成两类：`before_agent`、`before_model` 等 node-style hook 在固定节点前后运行；`wrap_model_call` 与 `wrap_tool_call` 则包住一次具体调用。权限判断需要决定“是否真的执行这次工具”，所以本章选择 `wrap_tool_call`。

`wrap_tool_call(request, handler)` 中两个参数的职责很清楚：

- `request.tool_call` 提供工具名、参数和调用 ID，`request.tool`、`request.state`、`request.runtime` 还可用于更复杂的上下文策略。
- `handler(request)` 继续执行工具并返回 `ToolMessage` 或 `Command`。middleware 可以在调用前审核，也可以在调用后记录结果；不调用 handler 就能短路本次执行。
- 多个 middleware 可通过 `create_agent(..., middleware=[...])` 组合。本章只注册一个权限组件，让 s02 的工具声明和流式循环保持不变。

本章使用阻塞式 `input()`，目的是用最少代码看清 allow / deny / ask 的控制点。`APPROVAL_LOCK` 会串行化同一轮中可能并发出现的多个审批提示，避免它们争用终端输入。

官方资料：[Middleware overview](https://docs.langchain.com/oss/python/langchain/middleware/overview) · [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)

### 确定性规则和模型判断要分开

本章规则是确定性的：相同工具参数应得到相同的初步分类。路径越界、命令词边界和固定 deny list 适合用代码规则处理；只有难以形式化的语义风险才值得增加模型分类器。规则判断便宜、可测试，也不会受到提示注入影响。

### 与 LangChain HITL / LangGraph interrupt 的对应关系

| 本章实现 | LangChain / LangGraph 对应能力 | 差异 |
|---|---|---|
| `PermissionMiddleware.wrap_tool_call()` | 自定义 tool wrap hook | 都在单次工具调用边界控制是否执行 |
| `check_rules()` | `HumanInTheLoopMiddleware` 的 `interrupt_on` 策略 | 本章规则可检查参数内容，直接写在 Python 中 |
| `input("Allow?")` | LangGraph `interrupt()` | 本章阻塞进程，不保存图状态 |
| `allow / deny` | HITL 的 `approve / reject` | 官方中间件还支持 `edit` 工具参数 |
| 当前进程继续执行 | `Command(resume=...)` 恢复图 | 本章不能跨进程恢复 |

LangChain 官方 `HumanInTheLoopMiddleware` 会在模型生成工具调用后、工具执行前触发 interrupt；配合 checkpointer 和稳定的 `thread_id` 保存图状态，之后用 `Command(resume=...)` 提交 `approve`、`edit` 或 `reject` 决策。本章的自定义 middleware 适合教学和本地 CLI，因为批准动作必须在同一进程中完成；需要跨进程恢复、Web 审批或参数编辑时，应切换到官方 HITL 组件。

### 本章应验证的边界

- 本章 deny list 是便于观察控制流的简单字符串匹配；生产实现应解析命令边界并配合沙箱，避免漏判和误判。
- 路径必须先 `resolve()` 再判断是否位于工作区。
- 自动触发任务不能偷偷读取交互式 stdin；后续 cron 和后台章节会继续强化这一点。
- 用户审批的“允许”不能绕过前置的硬拒绝策略。

官方概念：[Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop) · [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) · [Guardrails](https://docs.langchain.com/oss/python/langchain/guardrails)

---

## 试一下

```sh
cd learn-claude-code
python s03_permission/code.py
```

试试这些 prompt：

1. `Create a file called test.txt in the current directory`（应该直接通过）
2. `Delete the file test.txt`（bash + rm 会触发闸门 2）
3. `What files are in the current directory?`（只读，全部通过）
4. `Try to write a file to /etc/something`（写工作区外，触发闸门 2）
5. 在 Windows 上，`del test.txt` 和 `DEL test.txt` 会触发闸门 2，而 `model`、`delimiter` 和 `echo del test.txt` 不会。

观察重点：哪些操作直接通过？哪些需要你确认？哪些被直接拒绝？

---

## 接下来

本章用 `wrap_tool_call()` 隔离了权限审核，但 Agent 还可能需要模型调用前后的日志、提示词调整或会话级状态更新。

s04 Hooks → 继续理解更多执行阶段的钩子，以及不同扩展逻辑应该挂在哪个生命周期位置。
---

<!-- upstream-cc-source:start -->
## 深入 CC 源码


<details>
<summary>深入 CC 源码</summary>

> 以下基于 CC 源码 `types/permissions.ts`、`utils/permissions/permissions.ts`、`toolExecution.ts`、`utils/permissions/yoloClassifier.ts`、`tools/AgentTool/forkSubagent.ts` 的核查。

### 一、PermissionResult：不是 3 种，是 4 种

教学版的三道闸门（deny → ask → allow）和 CC 不完全对应。CC 的 `PermissionResult` 有 4 个 behavior（`types/permissions.ts:241-266`）：

| behavior | 含义 | 教学版对应 |
|----------|------|-----------|
| `allow` | 直接允许 | 闸门 3 通过 |
| `deny` | 直接拒绝 | 闸门 1 命中 |
| `ask` | 弹出对话框问用户 | 闸门 2 命中 |
| `passthrough` | 工具不表态，交给通用管线决定 | 教学版无 |

### 二、生产版的验证阶段

CC 的工具调用不是经过三道闸门，而是经过多个阶段，分布在 `checkPermissionsAndCallTool()`（`toolExecution.ts:599-1745`）、hooks、`hasPermissionsToUseToolInner()`（`utils/permissions/permissions.ts:1158-1310`）和 classifier 逻辑里：

1. **Zod schema 验证**（`toolExecution.ts:614-680`）— 参数类型检查
2. **validateInput()**（`toolExecution.ts:682-733`）— 工具级语义验证
3. **backfillObservableInput()**（`toolExecution.ts:784`）— 补全遗留字段
4. **PreToolUse hooks**（`toolExecution.ts:800-862`）— 钩子可以返回 allow/deny/ask
5. **resolveHookPermissionDecision()**（`toolExecution.ts:921-931`）— 协调钩子+管线决策
6. **hasPermissionsToUseToolInner()**（`permissions.ts:1158-1310`）— 多层规则检查：
   - 整个工具被 deny rule 禁用 → `deny`
   - 整个工具被 ask rule 标记 → `ask`
   - `tool.checkPermissions()` 工具自己的判断
   - 工具自己返回 deny → `deny`
   - `requiresUserInteraction()` → `ask`
   - 内容相关的 ask 规则 → `ask`（不可绕过）
   - 安全检查违规 → `ask`（不可绕过）
   - bypassPermissions 模式 → `allow`
   - 整个工具被 allow rule 放行 → `allow`
   - passthrough → 转为 `ask`

### 三、拒绝列表：不是一个文件，是 8 个来源

CC 没有单一的 deny list。权限规则来自 8 个来源（`types/permissions.ts:54-62`）：

| 来源 | 配置位置 |
|------|---------|
| `userSettings` | `~/.claude/settings.json` |
| `projectSettings` | `.claude/settings.json` |
| `localSettings` | `settings.local.json` |
| `flagSettings` | Feature flags |
| `policySettings` | 企业管理策略 |
| `cliArg` | `--allowedTools` / `--deniedTools` |
| `command` | 内联命令 |
| `session` | 会话内临时授权 |

每条规则格式：`{ toolName: "Bash", ruleBehavior: "deny", ruleContent: "npm publish:*" }`。多个来源的规则合并，高优先级来源覆盖低优先级（从低到高：user < project < local < flag < policy，加上 cliArg、command、session）。

### 四、isDestructive() 是什么

CC 中 `isDestructive`（`Tool.ts:405-406`）**纯粹是 UI 展示用的**——在工具列表里显示 `[destructive]` 标签。它不参与权限决策。默认所有工具都返回 `false`。只有 ExitWorktree（remove 时）和 MCP 工具（依赖 `annotations.destructiveHint`）覆写了它。

### 五、YoloClassifier（自动审批）

CC 的 auto 模式下，不会每次都弹对话框。`classifyYoloAction`（`utils/permissions/yoloClassifier.ts:1012`）把工具调用 + 对话上下文发给一个分类器 LLM 判断是否安全。先尝试 acceptEdits 模式模拟（`permissions.ts:620-656`，如果 acceptEdits 允许 → 直接批准），再查安全工具白名单（`permissions.ts:658-686`），最后才调分类器。分类器连续拒绝太多次 → 回退到人工审批。

### 六、权限冒泡

子 Agent（通过 AgentTool fork 出来的）的 `permissionMode` 设为 `'bubble'`（`forkSubagent.ts:50`）。意思是权限弹窗**冒泡到父 Agent 的终端**，而不是在子 Agent 里静默拒绝。Bash 分类器在这个过程中继续跑——给权限对话框显示的同时在后台判断是否可以自动批准。

### 教学版的简化是刻意的

- 多阶段管线 → 3 道闸门：理解门槛大幅降低
- 8 个规则来源 → 1 个本地 DENY_LIST：概念量可控
- isDestructive → 忽略（教学版没有 UI 层，CC 里它也不参与权限决策）
- YoloClassifier → 省略（依赖于额外的 LLM 调用和遥测系统）
- 权限冒泡 → 省略（s15 才涉及多 Agent）

</details>

<!-- upstream-cc-source:end -->
