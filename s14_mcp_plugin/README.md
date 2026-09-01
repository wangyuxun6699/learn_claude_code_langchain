# s14: MCP & Plugin — 把外部工具接进同一个工具池

> LangChain / LangGraph 教学改编版。章节结构参考
> [shareAI-lab/learn-claude-code 的 s14](https://github.com/shareAI-lab/learn-claude-code/blob/main/s14_mcp_plugin/README.md)。
>
> **Harness 层**：MCP Tools — 连接服务、发现工具，并把它们加进 agent 循环。

[s13](../s13_agent_teams/) → **s14** → [s15](../s15_integrated_harness/)

---

## 问题

前几章的工具都直接写在 `code.py` 里。要接入一个文档系统和部署平台，可以再加 `search_docs`、`deploy_status`、`trigger_deploy`，但每接一个服务就要再写一组工具定义、参数 schema 和调用处理函数。

MCP 把这份职责拆开：server 提供「工具列表 + 调用接口」，harness 负责连接它、给模型可见的工具起名字、套上权限检查，再把发现到的工具交给模型。

---

## 解决方案

![s14: MCP 架构](images/mcp-architecture.svg)

本章从 s04 的五个基础工具和 Hook 出发，新增三块：

- `MCPClient`：保存某个 server 返回的工具定义和调用处理器。
- `connect_mcp`：连接一个 server，拿到它的工具列表。
- `assemble_tool_pool`：把基础工具和所有已连接 server 的工具合并成当前工具池。

`docs` 和 `deploy` 是进程内的 mock server，用来模拟 `tools/list`、`tools/call` 和动态工具池。本章不实现真正的 MCP transport（JSON-RPC / OAuth / 资源订阅 / 轮询）。

---

## LangChain 里的关键差异：工具池是“动态”的

s01–s13 里 `create_agent()` 在创建时就把工具编译进了静态 LangGraph。但 MCP 的工具要在**运行时**才会出现：

```
connect_mcp(name="docs")  →  下一轮模型调用就多了 mcp__docs__search
```

所以本章不调用 `create_agent`，而是自己铺开那张 LangGraph：

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def model_node(state):
    tools = assemble_tool_pool()                      # 每轮都取“当前”工具
    return {"messages": [MODEL.bind_tools(tools).invoke(state["messages"])]}

def tools_node(state):
    ...   # PreToolUse / 权限 / 执行 / PostToolUse

graph_builder.add_edge(START, "model")
graph_builder.add_conditional_edges("model", route, {"tools": "tools", "end": END})
graph_builder.add_edge("tools", "model")
```

`route` 判断最后一条消息是否还有 `tool_calls`：有就回到 tools 节点，没有就结束。这正是 `create_agent` 帮我们编译掉的那条循环，现在因为需要运行时改工具集合，就显式写出来。

---

## 工作原理

### 1. MCPClient 保存发现结果和调用处理器

```python
class MCPClient:
    def register(self, tool_defs, handlers):
        self.tools = list(tool_defs)
        self._handlers = dict(handlers)

    def call_tool(self, tool_name, args):
        handler = self._handlers.get(tool_name)
        if not handler:
            return f"MCP error: unknown tool '{tool_name}'"
        try:
            return str(handler(**args))
        except Exception as error:
            return f"MCP error: {type(error).__name__}: {error}"
```

错误被转成字符串返回给模型，而不是把异常抛回 agent 循环。

### 2. connect_mcp 只负责连接和发现

未知 server 名、重复连接都返回普通字符串，不抛异常。连接成功后把 client 写进 `mcp_clients`，下一轮 `assemble_tool_pool()` 就会带上它。

### 3. 前缀把不同 server 的工具区分开

```
mcp__{server}__{tool}
```

`normalize_mcp_name()` 把模型工具名字母表之外的字符替换成下划线；组装时还会检查规范化后的重名和 64 字符上限。

### 4. 动态工具如何保留参数 schema

这是 LangChain 版最讲究的一步：每个 mock 工具是带类型注解的函数，`functools.wraps` 把它的签名复制到转发的 `**kwargs` 包装器上，再交给 `StructuredTool.from_function`。这样 Pydantic 能推断出 `search(query, limit)` 的 schema，同时真正的调用仍然走 `client.call_tool` 的错误边界：

```python
@functools.wraps(handler)
def _runner(**kwargs):
    return client.call_tool(raw_name, kwargs)

StructuredTool.from_function(func=_runner, name=prefixed, description=description)
```

### 5. 权限由 host 决定，不由 server 声称决定

server 可能标注 `readOnlyHint` 或 `destructiveHint`，但那不是授权。本章用 host 侧策略：

```python
MCP_HOST_POLICY = {
    "mcp__docs__search":      "allow",
    "mcp__docs__get_version": "allow",
    "mcp__deploy__status":    "allow",
    "mcp__deploy__trigger":   "confirm",
}
```

未配置的外部工具默认 `confirm`（要用户确认）。描述里含 `readOnly` 不等于可信。

---

## 与 s04 的对比

| 组件 | s04 | s14 |
|---|---|---|
| 基础工具 | 五个固定工具 | 不变 |
| 工具来源 | 定义在 code.py | 基础工具 + 发现的 MCP 工具 |
| 工具池 | 固定 TOOLS | 每轮由 assemble_tool_pool() 组装 |
| 外部工具名 | 无 | mcp__{server}__{tool} |
| 权限 | 命令与路径检查 | 增加 host 侧 MCP 策略 |
| 循环 | create_agent 静态图 | 手写 LangGraph（动态工具） |

本章不携带 Task / Background / Cron / Team / Worktree，它们和 MCP 一起在 s15 汇合。

---

## 运行

```sh
cd learn-claude-code
python -m s14_mcp_plugin.code
```

输入：

```
Connect to the docs server, search for agent hooks, and tell me the current documentation API version.
```

典型工具调用序列：

```
connect_mcp(name="docs")
mcp__docs__search(query="agent hooks")
mcp__docs__get_version()
```

再试：

```
Connect to the deploy server and check the web service status. Do not trigger a deployment.
```

`status` 走 host 策略直接放行；`trigger` 会触发用户确认。

---

## 下一步

MCP 在这里仍是独立分支。s15 Integrated Harness 会把基础工具、Hook、skills、上下文、记忆、任务、后台工作、cron、团队和 MCP 合并进一个 runtime。

<details>
<summary>深入 CC 源码</summary>

> 以下为机制级对照：Claude Code 的 MCP 集成是完整、真实存在的（多种 transport、OAuth、资源 / 提示订阅、按作用域配置 server 与工具），而教学版用进程内 mock server 只演示 `tools/list` 与 `tools/call` 的协议边界。不宣称逐行等价。

### 一、CC 真实 MCP 的能力远多于教学版

CC 内置 MCP client，支持 stdio / SSE / streamable HTTP 等传输、OAuth 授权、resources / prompts 订阅、按 user / project 作用域配置 server，并用允许 / 拒绝列表控制每个 server 暴露哪些工具。教学版把这些压缩成“一个进程内 server + 发现 + 调用”，因为本章只讲工具池如何动态组装，不讲 transport 细节。

### 二、命名空间：CC 同样把外部工具和内置工具放进一个池子

CC 的 MCP 工具与内置工具进入同一个工具池，工具名带 server 命名空间以避开同名冲突。教学版的 `mcp__{server}__{tool}` 前缀、名称规范化与 64 字符上限，复刻的就是这条“多个 server 都可能提供 search / status”的边界处理。

### 三、权限判断不信任 server 自述

CC 由用户在配置里批准 server 与工具；server 声明的 readOnlyHint / destructiveHint 只是提示，不是授权。教学版的宿主侧 `MCP_HOST_POLICY`（未配置默认 confirm）复刻了“权限由 host 决定、不由 server 声称决定”这一原则——description 写着 `readOnly` 不等于可信。

### 四、动态工具池迫使手写循环

CC 连接 server 后，下一轮模型调用就带上新工具。本 LangChain 版为复现这一点，放弃 `create_agent` 的静态工具图，手写 LangGraph 的 model→tools 循环并在每轮 `assemble_tool_pool()`；这正是 s01 里“交给 create_agent 的那条循环”被显式写出来的原因。

### 五、教学版省略了什么

真实 MCP transport（JSON-RPC / OAuth / 握手 / 重连）、资源订阅、轮询、tools/list 变更监听、跨 server 描述冲突的完整处理，都没有在本章实现。mock server 的目标是让“运行时新增工具”这条控制流可观察，而不是立即可用于生产接入。
</details>

