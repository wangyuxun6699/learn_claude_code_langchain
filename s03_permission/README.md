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

s02 的循环完全保留。唯一的变动是在工具执行前插入 `check_permission()`。每个工具调用依次经过三道闸门：硬拒绝优先，软询问次之，都没命中就放行。

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
    for pattern in DENY_LIST:
        if pattern in command:
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
        "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
        "message": "Access outside workspace",
    },
    {
        "tools": ["bash"],
        "check": lambda args: contains_destructive_command(args.get("command", "")) or any(
            kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]
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

**三道闸门串在一起**，插在工具执行之前：

```python
def check_permission(block) -> bool:
    # 闸门 1: 硬拒绝
    if block.name == "bash":
        reason = check_deny_list(block.input.get("command", ""))
        if reason:
            print(f"\n⛔ {reason}")
            return False

    # 闸门 2 + 3: 规则匹配 → 用户审批
    reason = check_rules(block.name, block.input)
    if reason:
        decision = ask_user(block.name, block.input, reason)
        if decision == "deny":
            return False

    return True

# 在 agent_loop 中——s02 的循环只加了一行：
for block in tool_calls:
    if not check_permission(block):           # ← 新增
        results.append({... "content": "Permission denied."})
        continue
    output = TOOL_HANDLERS[block.name](**block.input)  # s02 原有
    results.append(...)
```

---

## 相对 s02 的变更

| 组件 | 之前 (s02) | 之后 (s03) |
|------|-----------|-----------|
| 安全模型 | 无（信任模型） | 三道闸门权限管线 |
| 新函数 | — | check_deny_list, check_rules, ask_user, check_permission |
| 循环 | 直接执行所有工具 | 执行前插入 check_permission() |

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

当前权限检查每次都在循环里硬编码 `check_permission()`。如果我想在每次工具执行前后加日志？如果想在某些操作后自动触发 git commit？这些扩展逻辑散落在 loop 里，循环很快就会膨胀。

s04 Hooks → 给循环加钩子，扩展逻辑挂在钩子上，循环保持干净。
---

## 本项目保留的 LangChain / LangGraph 教学补充

> 以下内容来自本仓库对齐前的 README，作为上游课程之外的本地教学补充完整保留。

<!-- local-langchain-additions:start -->
<details>
<summary>展开本仓库原有的 LangChain / LangGraph 教学说明</summary>

# s03: Permission — 安全地让 Agent 行动

> LangChain 教学改编版。章节结构与“深入 CC 源码”部分主要参考 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)。
>
> **Harness 层**：权限系统 — deny、规则判断与用户确认。

[s02](../s02_tool_use/) → **s03** → [s04](../s04_hooks/)

---

## 问题

Agent 拿到写文件和执行命令的能力后，必须区分明确禁止、需要确认和可以自动执行的操作。

---

## 解决方案

![s03: Permission — 安全地让 Agent 行动](images/permission-overview.svg)

本章保留两种自定义权限写法，同时补充 LangChain 真正内置的人审方案：

1. `code.py`：每个工具函数入口主动调用 `check_permission()`；
2. `code_middleware.py`：通过 LangChain 的 `AgentMiddleware.wrap_tool_call()` 在工具执行前统一拦截；
3. `HumanInTheLoopMiddleware`：LangChain 内置的暂停、审批和恢复机制，本章代码尚未接入，用来说明生产级人审与同步 `input()` 的区别。

这里必须区分“框架提供的执行钩子”和“框架自带的权限策略”：

- `wrap_tool_call` 是 LangChain 提供的工具执行拦截点；
- `permission_check`、`check_deny_list()`、`check_rules()` 仍然是本项目编写的策略；
- `HumanInTheLoopMiddleware` 才是 LangChain 开箱即用的人审 middleware。

---

## 三种检查位置

### 一、把检查写在工具函数里

基础版的每个敏感工具都显式调用权限函数：

```python
@tool
def run_bash(command: str) -> str:
    if not check_permission(
        "run_bash",
        {"command": command},
    ):
        return "permission denied"

    return execute_command(command)
```

执行路径：

```text
模型产生 tool call
    ↓
LangGraph ToolNode
    ↓
run_bash()
    ↓
check_permission()
    ├─ deny：返回 "permission denied"
    ├─ ask：同步 input() 询问用户
    └─ allow：执行 subprocess
```

这种写法的最大优点是检查跟着工具本身走。即使以后不通过 `create_agent`，而是直接调用 `run_bash.invoke()`，函数内部的检查仍然存在。因此，路径不能越界、参数必须合法、某类命令永远不能执行等**不可绕过的工具不变量**，适合留在工具函数或它调用的底层执行函数中。

但它也有明显代价：

- `run_bash`、`run_read`、`run_write`、`run_edit` 都要重复写入口检查；
- 新增工具时很容易忘记调用 `check_permission()`；
- 工具必须手工拼出自己的名称和参数字典，参数增加后可能漏传；
- 拒绝时只是返回普通字符串，LangGraph 会生成普通 `ToolMessage`，在监控层不一定能区分“工具正常返回”与“权限拒绝”；
- `ask_user()` 使用阻塞式 `input()`，进程退出后无法恢复审批现场；
- 同一轮出现多个并行工具调用时，多个线程可能同时争用终端输入。

当前基础版已经体现了“容易漏接”的风险：`run_glob` 没有调用 `check_permission()`。虽然现有规则并未限制 glob，所以当前行为没有差异，但将来增加 glob 规则时，基础版还必须记得修改工具函数。

### 二、使用 wrap_tool_call 在工具执行前统一拦截

`code_middleware.py` 把工具函数里的重复检查移到 LangChain 工具执行边界：

```python
class permission_check(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        name = request.tool.name if request.tool else request.tool_call["name"]
        args = request.tool_call.get("args", {})
        if not check_permission(name, args):
            return ToolMessage(
                content="Permission denied",
                tool_call_id=request.tool_call["id"],
                name=name,
                status="error",
            )
        return handler(request)

agent = create_agent(
    model=MODEL,
    tools=TOOLS,
    system_prompt=SYSTEM,
    middleware=[permission_check()],
)
```

`wrap_tool_call` 接收两个关键对象：

- `request`：包含 `tool_call`、解析后的 `tool`、Agent state 和 runtime；
- `handler`：继续执行下一层 middleware 或真实工具的函数。

执行路径：

```text
模型产生 tool call
    ↓
permission_check.wrap_tool_call()
    ├─ deny/用户拒绝
    │    └─ 返回 ToolMessage(status="error")
    └─ allow
         └─ handler(request)
              ↓
           真实工具函数
```

只要工具通过这个 Agent 的 LangGraph runtime 执行，所有工具调用都会经过 middleware。新增第六个工具时，不需要再给函数补一行 `check_permission()`。拒绝结果也可以明确标记为 `ToolMessage(status="error")`。当前代码把类命名为 `permission_check`；生产代码通常会按 Python 类命名习惯写成 `PermissionMiddleware`。

不过 middleware 并没有自动替你判断危险操作。下面这些仍然是项目代码，而不是 LangChain 内置规则：

```python
dangerous = ["rm -rf /", "sudo", "shutdown", ...]

check_deny_list(command)
check_rules(tool_name, args)
ask_user(tool_name, args, reason)
check_permission(tool_name, args)
```

此外，middleware 只保护“经过该 Agent runtime 的调用”。如果其他代码直接调用底层 Python 函数，或者创建另一个没有注册 `PermissionMiddleware` 的 Agent，就可能绕开这一层。因此 middleware 更适合表达**Agent 级策略**，不能代替工具自身必须维护的安全不变量。

LangChain 对 `wrap_tool_call` 的定义是：middleware 可以在工具执行前后拦截、修改或短路调用；允许时调用 `handler(request)`，拒绝时直接返回 `ToolMessage` 或 `Command`。多个 middleware 会自动组合，列表中的第一个是最外层。参见 [LangChain `wrap_tool_call` API](https://reference.langchain.com/python/langchain/agents/middleware/types/AgentMiddleware/wrap_tool_call)。

### 三、LangChain 内置 HumanInTheLoopMiddleware

当前两份教学代码都通过：

```python
input("Allow? [y/N] ")
```

同步等待用户。它只是终端交互，不会真正暂停并持久化 LangGraph。

LangChain 内置的 `HumanInTheLoopMiddleware` 使用 LangGraph interrupt：

1. 模型提出敏感工具调用；
2. middleware 产生 interrupt，工具尚未执行；
3. checkpointer 保存当前图状态；
4. 应用把待审批动作展示给用户；
5. 用户选择 `approve`、`edit` 或 `reject`；
6. 使用相同 `thread_id` 恢复执行。

最小配置如下：

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=MODEL,
    tools=TOOLS,
    system_prompt=SYSTEM,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "run_bash": {
                    "allowed_decisions": [
                        "approve",
                        "reject",
                    ]
                },
                "run_write": True,
                "run_edit": True,
                "run_read": False,
                "run_glob": False,
            }
        )
    ],
    checkpointer=InMemorySaver(),
)

config = {
    "configurable": {
        "thread_id": "permission-demo",
    }
}
```

恢复审批：

```python
from langgraph.types import Command

agent.invoke(
    Command(
        resume={
            "decisions": [
                {"type": "approve"},
            ]
        }
    ),
    config=config,
)
```

内置 HITL 的策略主要按工具名称配置。例如上面的配置会审查所有 `run_bash`，而当前 `check_rules()` 只审查包含 `rm`、`chmod 777` 等关键词的命令。要保留这种**基于参数内容的条件判断**，仍需要自定义 middleware 或直接使用 LangGraph `interrupt()`；不能把 `HumanInTheLoopMiddleware` 理解成自动识别危险命令的分类器。

HITL 也不是 deny list。它解决的是“如何可靠地暂停、展示、批准/编辑/拒绝并恢复”，不是“什么命令危险”。官方文档要求配置 checkpointer，并在恢复时使用相同的 `thread_id`。参见 [LangChain Human-in-the-loop 文档](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)。

---

## 完整对比

| 维度 | 工具函数内检查 | 自定义 `wrap_tool_call` | 内置 `HumanInTheLoopMiddleware` |
|---|---|---|---|
| 检查位置 | 真实工具函数内部 | LangGraph 调用工具之前 | LangGraph 调用工具之前 |
| deny/ask 规则来源 | 项目自定义 | 项目自定义 | 项目配置哪些工具需要人审 |
| 新工具自动覆盖 | 否，必须手动接入 | 是，只要经过该 Agent | 是，需在 `interrupt_on` 配置 |
| 直接调用工具时生效 | 是 | 否 | 否 |
| 基于参数内容判断 | 容易实现 | 容易实现 | 默认按工具名；复杂条件需扩展 |
| 拒绝结果 | 普通字符串 | 可返回 error `ToolMessage` | interrupt 后以 reject 决策恢复 |
| 修改待执行参数 | 需自己实现 | 可改写 request 后调用 handler | 内置 `edit` 决策 |
| 暂停后跨请求恢复 | 否 | 当前实现否 | 是，需要 checkpointer |
| 多个待审批动作 | 多个 `input()` 可能冲突 | 当前实现仍可能冲突 | interrupt 中统一处理 decisions |
| 适合承担的职责 | 不可绕过的工具不变量 | Agent 级 deny/allow/审计策略 | 持久化人工审批工作流 |

---

## 推荐的分层方式

真实项目不必在三种写法中三选一，可以按职责组合：

```text
工具函数 / 底层执行函数
    └─ 参数校验、工作区边界、绝对禁止项

自定义 wrap_tool_call middleware
    └─ 会话策略、角色权限、deny/allow 规则、审计日志

HumanInTheLoopMiddleware / interrupt
    └─ 需要人工决定时暂停、持久化和恢复
```

如果组合自定义 hard-deny 和内置 HITL，应该先把当前 `check_permission()` 拆开，避免 `input()` 与 interrupt 重复询问：

```python
class HardDenyMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        args = request.tool_call.get("args", {})

        reason = None
        if name == "run_bash":
            reason = check_deny_list(
                args.get("command", "")
            )

        if reason:
            return ToolMessage(
                content=reason,
                tool_call_id=request.tool_call["id"],
                name=name,
                status="error",
            )

        return handler(request)


agent = create_agent(
    model=MODEL,
    tools=TOOLS,
    middleware=[
        HardDenyMiddleware(),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "run_bash": True,
                "run_write": True,
                "run_edit": True,
            }
        ),
    ],
    checkpointer=InMemorySaver(),
)
```

这样，明确禁止的操作在最外层直接拒绝；其余被标记的敏感操作进入可恢复的人审流程；即使绕过 Agent，工具底层仍保留路径边界和参数校验。

---

## 本章文件

`code.py` 是带注释的基础版；`code_middleware.py` 是 s03.5 中间件归档版；`code_uncommented.py` 是基础版去掉教学注释的精简版。

---

## 试一下

先在仓库根目录准备环境，然后从根目录按模块运行：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m s03_permission.code
```

> 这些教学 Agent 可以执行命令和修改文件。建议先在测试目录中试用，并认真阅读每次权限提示。

---

## 接下来

权限检查做了——但每次都在循环里硬编码 `check_permission()`。如果我想在每次工具执行前后加日志？如果想在某些操作后自动触发 git commit？这些扩展逻辑散落在 loop 里，循环很快就会膨胀。

s04 Hooks → 给循环加钩子，扩展逻辑挂在钩子上，循环保持干净。

</details>
<!-- local-langchain-additions:end -->

<!-- upstream-cc-source:start -->
## 深入 CC 源码

> 原文：[s03_permission](https://github.com/shareAI-lab/learn-claude-code/blob/67a9126c6435a8654ba7a6f68c0fd2130f00a462/s03_permission/README.md)。以下折叠块保持原文，文中的章号与源码行号沿用该版本。

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
