# s17: Autonomous Agents — 从响应到主动

> **状态：尚未实现。** 本章目录用于保持与参考仓库一致的 20 章编排。

[s16](../s16_team_protocols/) → **s17** → [s18](../s18_worktree_isolation/)

当前 `code.py` 是 0 字节占位文件。等 LangChain 版本完成后，再补充教学 README；目前不复制参考仓库的 Python 实现，以免把 Anthropic SDK 版本误认为本项目代码。

参考章节：[`shareAI-lab/learn-claude-code/s13_agent_teams`](https://github.com/shareAI-lab/learn-claude-code/tree/main/s13_agent_teams)（自主认领已并入该章）

<!-- upstream-cc-source:start -->
## 深入 CC 源码

> 原文：[s17_autonomous_agents](https://github.com/shareAI-lab/learn-claude-code/blob/67a9126c6435a8654ba7a6f68c0fd2130f00a462/s17_autonomous_agents/README.md)。以下折叠块保持原文，文中的章号与源码行号沿用该版本。

<details>
<summary>深入 CC 源码</summary>

> 教学说明：本章的 idle_poll + auto-claim 机制是教学设计，用统一的轮询函数演示"空闲后找活干"。CC 的实际实现是多个机制的组合，但目标一致——减少 Lead 的手动分配负担。

### 一、CC 的空闲机制：组合路径，不是单一轮询

教学版用一个 `idle_poll()` 统一处理空闲时的 inbox 检查和任务认领。CC 的实际实现是四个机制的组合：

**idle_notification**：队友完成一轮工作后，`sendIdleNotification()`（`inProcessRunner.ts:569-589`）向 Lead 发送空闲通知。Lead 知道队友可用了，可以分配新任务或请求关机。

**mailbox 轮询**：`waitForNextPromptOrShutdown()`（`inProcessRunner.ts:689-868`）是一个 **500ms 轮询循环**，持续检查三类来源：pending user messages、mailbox 文件消息、task list。shutdown_request 被优先处理（`inProcessRunner.ts:768-804`），不会被普通消息饿死。

**task watcher**：`useTaskListWatcher`（`hooks/useTaskListWatcher.ts:34-189`）用 `fs.watch()` 监听 `.claude/tasks/` 目录变化，1 秒 debounce，当新任务创建或依赖解锁时触发检查。依赖判断（`L197-207`）是"blockedBy 中没有未完成的任务"，不是"blockedBy 为空"。

**主动 claim**：轮询循环内部也会调用 `tryClaimNextTask()`（`inProcessRunner.ts:853-860`）——在等待期间主动从 task list 领取任务。所以"队友不主动轮询任务"不准确，CC 同时有被动通知和主动认领。

### 二、任务认领：文件锁 + 原子操作

`claimTask()`（`utils/tasks.ts:541-612`）用 `proper-lockfile` 的任务文件锁，在锁内完成读-检查-改-写。检查项：owner 是否已存在（`L575-576`）、是否已完成（`L580-581`）、blockedBy 中是否有未完成任务（`L585-594`）。`claimTaskWithBusyCheck()`（`utils/tasks.ts:614-692`）用 task-list 级别锁，把 busy check 和 claim 做成原子操作，避免 TOCTOU。

`findAvailableTask()`（`inProcessRunner.ts:595-604`）的依赖判断也是"所有 blockedBy 已完成"，用 `task.blockedBy.every(id => !unresolvedTaskIds.has(id))` 实现。`tryClaimNextTask()`（`inProcessRunner.ts:624-657`）在认领后把状态更新为 `in_progress`，让 UI 立即反映变化。

### 三、教学版 vs CC 对比

| 维度 | 教学版 (s17) | CC |
|------|-------------|-----|
| 空闲机制 | idle_poll 统一轮询（5s） | idle_notification + 500ms mailbox 轮询 + task watcher |
| 任务发现 | scan_unclaimed_tasks（轮询） | useTaskListWatcher（文件监听）+ tryClaimNextTask（主动轮询） |
| 依赖判断 | can_start（所有 blockedBy 已完成） | findAvailableTask（同样语义） |
| 并发安全 | owner 检查（无文件锁） | proper-lockfile 任务锁 + task-list 锁 |
| shutdown 处理 | IDLE 直接分发，WORK 通过 handle_inbox_message | 500ms 轮询中优先处理 shutdown_request |
| 超时退出 | 60s 无新任务 | 无固定超时，Lead 手动 shutdown |
| 身份保持 | messages 长度检测 | context compaction 保留 system prompt |
| claim 失败处理 | 检查返回值，失败不注入 | 文件锁保证原子性 |

教学版的 `idle_poll()` 把 CC 的四个机制合并成一个轮询函数——简化合理，因为核心语义（空闲时找活干、依赖解锁后可认领、shutdown 优先）是一致的。

</details>

<!-- upstream-cc-source:end -->
