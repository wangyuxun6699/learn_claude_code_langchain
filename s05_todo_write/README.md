# s05: TodoWrite — 让 Agent 知道自己做到哪了

>
> **Harness 层**：任务内计划 — 可见、可更新的 Todo 状态。

[s04](../s04_hooks/) → **s05** → [s06](../s06_subagent/)

---

## 问题

长任务只有消息历史时，模型容易忘记已经完成什么、下一步是什么，也不便于用户观察进度。

---

## 解决方案

![s05: TodoWrite — 让 Agent 知道自己做到哪了](images/todo-overview.svg)


LangChain 的 `TodoListMiddleware` 会注入 `write_todos` 工具并在 Agent state 中维护 todos；本章再把该状态打印到命令行。同时对智能体进行强约束，要求它在动手之前必须先写todo列表,也就是强硬要求智能体进入plan模式，同时如果超过3轮没有查看计划，就给模型提示词做出引导。

---

## 工作原理：LangChain 版本

注入todo
```python
MIDDLEWARE = [
    user_prompt_submit,
    TodoListMiddleware(
        system_prompt="""
        Use write_todos for every non-trivial request.
        Keep one relevant item in_progress and update statuses as work progresses.
        """
    ),
    tool_hook,
    stop_hook,
]

agent = create_agent(model=MODEL, tools=TOOLS,
                     system_prompt=SYSTEM, middleware=MIDDLEWARE)
result = agent.invoke({"messages": messages})
todos = result.get("todos", [])
```
设置强硬提醒
```python
global rounds_since_todo

    if rounds_since_todo >= 3 and messages:
        messages.append({
            "role": "user",
            "content": "<reminder>Update your todos with write_todos before continuing.</reminder>",
        })
        rounds_since_todo = 0
```

`code_streaming.py` 它用 LangGraph streaming，按图步骤增量展示消息和 Todo，而不是在干完活之后再把消息打印出来。

---

## 本章文件

`code.py` 是带注释的 invoke 版；`code_streaming.py` 是 s05.5 流式归档版；`code_uncommented.py` 是 invoke 版去掉教学注释的精简版。

---

## 试一下

先在仓库根目录准备环境，然后从根目录按模块运行：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m s05_todo_write.code
```

> 这些教学 Agent 可以执行命令和修改文件。建议先在测试目录中试用，并认真阅读每次权限提示。

---

## 接下来

s06 把独立子问题交给隔离上下文中的子 Agent。

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

