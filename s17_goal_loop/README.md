# s17: Goal Loop — 目标决定循环何时真正结束

> LangChain / LangGraph 教学改编版。章节结构参考
> [shareAI-lab/learn-claude-code 的 s17](https://github.com/shareAI-lab/learn-claude-code/blob/main/s17_goal_loop/README.md)。
>
> *"模型不再调用工具只说明这一轮想停；一个独立的评估器决定整个目标是否完成。"*
>
> **Harness 层**：持续执行 — 在每轮末尾检查完成条件，还有活就再开一轮。

[s16](../s16_workflow_runtime/) → **s17** → 课程终点

---

## 问题

s01 以来，agent 循环只有一个退出条件：模型不再调用工具就返回。这对于"一直修到测试全过""做完每一条验收标准"不够——模型可能只做了一半。没有新的 tool_use 只证明**这一轮**结束，不证明目标达成。

`/goal` 在真正返回前加一个独立判断。

---

## /goal 是一个会话级的 Stop hook

输入：

```text
/goal pytest tests/auth exits with code 0 and lint reports no errors
```

程序保存完成条件，并立刻把它作为当前任务交给主模型——不需要再发第二条"开始干活"。

主模型停下不调用工具时，循环在返回前跑 Goal Stop hook：

```python
if response.tool_calls:
    messages.extend(execute_tool_calls(response))
    continue

decision = controller.evaluate_after_turn(messages)
if decision["action"] == "block":
    messages.append(HumanMessage(content=decision["reason"]))
    continue
return {"status": decision["action"], ...}
```

没有活动目标时，hook 直接放行，退出条件和 s01 完全一样。

---

## 评估器和执行者是两个模型

主模型负责改代码、跑命令、解任务。Goal 评估器是独立的一次模型调用，只干一件事：判断完成条件。它**没有工具**，不能自己读文件或重跑测试，只能判断对话里已有的内容：

```json
{ "ok": false, "reason": "The conversation does not contain pytest's exit code yet.", "impossible": false }
```

- `ok=true`：条件已被对话里的具体结果满足。
- `ok=false`：需要再来一轮。
- `impossible=true`：任务已不可能完成，交还用户。

评估器输入是最近若干条完整消息；最新一条过大时只留首尾，避免一个工具结果塞满评估请求。评估器 prompt 明确要求"不能假设没报告的命令已成功"。

Goal Loop 不是测试框架——工具仍然做真正的验证，评估器只判断验证结果是否出现在工作记录里。

---

## 好的完成条件是可检查的

"把代码写好"太含糊。有用的条件会说清三件事：

1. **终态**：做完时必须成立什么；
2. **检查**：哪条命令/输出能证明它；
3. **约束**：过程中不能破坏什么。

例如：

```text
/goal finish the authentication migration until pytest tests/auth exits 0,
without modifying test files outside tests/auth
```

需要给无人值守的工作设上限时用主循环的全局轮数限制：

```bash
MAX_TURNS=20 python -m s17_goal_loop.code "python -m pytest tests/auth exits with code 0"
```

---

## 未完成的工作回到同一条循环

评估器说不满足时，返回一条简短理由（例如"还没有完整测试结果，跑 pytest tests/auth 并报告退出码"）。程序把理由追加回 messages 并 `continue`，主模型无需用户输入再跑一轮。没有独立的 continuation 队列：Goal 判断发生在返回边界，未完成的工作也从那个边界回去。

---

## 自动延续仍需出口

Goal 没有隐藏的"再来 20 轮"预算。两道出口在 goal 之外：

- 主循环全局 `MAX_TURNS`；
- 连续 Stop-hook 阻塞上限 `MAX_BLOCKS`。

到上限就交还控制权，**不**标记完成、也不悄悄清掉目标。评估器自己出错同样：停止自动延续、保留目标、把错误交还，而不是假装成功。

---

## 查看 / 替换 / 清除

一次会话最多一个活动目标。

```text
/goal             查看条件、耗时、评估次数、最近理由
/goal <新条件>     替换旧目标并立即按新条件开工
/goal clear       清除目标（stop/off/reset/none/cancel 也叫别名）
```

---

## 本章新增的四个部分

| 部分 | 职责 |
|---|---|
| `GoalState` | 保存条件、评估次数、开始时间、最近理由 |
| `PromptGoalEvaluator` | 用独立模型调用判断对话是否满足条件 |
| `GoalController` | 设置、查看、清除，并运行 Goal Stop hook |
| `run_with_goal` | 把 Stop hook 接到返回边界的主循环 |

---

## 运行

```bash
python -m s17_goal_loop.code                          # 交互式
python -m s17_goal_loop.code "/goal pytest exits 0"   # 命令行直接设定目标
```

可选：用更小的模型做 Goal 评估（`GOAL_EVALUATOR_MODEL_ID`），并设置 `MAX_TURNS` / `MAX_BLOCKS`。

---

## 与 s16 的关系

s16 回答"一批工作怎么跑"：哪些步骤并发、结果怎么核验、中断怎么续跑。

s17 回答"整个任务是否完成"。一个 Workflow 可能成功跑完，而用户的最终需求仍未满足；Workflow 结果进入对话后，Goal 评估器决定会话是停还是继续。两者可单独使用；同一个 host 把它们接起来时，Workflow 完成消息进入对话，Goal Loop 判断整体是否还需要再来一轮。
