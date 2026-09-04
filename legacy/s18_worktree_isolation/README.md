# s18: Worktree Isolation — 并行任务互不干扰

> **状态：尚未实现。** 本章目录用于保持与参考仓库一致的 20 章编排。

[s17](../s17_autonomous_agents/) → **s18** → [s14: MCP & Plugin](../../s14_mcp_plugin/)

当前 `code.py` 是 0 字节占位文件。等 LangChain 版本完成后，再补充教学 README；目前不复制参考仓库的 Python 实现，以免把 Anthropic SDK 版本误认为本项目代码。

参考章节：[`shareAI-lab/learn-claude-code/s13_agent_teams`](https://github.com/shareAI-lab/learn-claude-code/tree/main/s13_agent_teams)（worktree 隔离已并入该章）

<!-- upstream-cc-source:start -->
## 深入 CC 源码

> 原文：[s18_worktree_isolation](https://github.com/shareAI-lab/learn-claude-code/blob/67a9126c6435a8654ba7a6f68c0fd2130f00a462/s18_worktree_isolation/README.md)。以下折叠块保持原文，文中的章号与源码行号沿用该版本。

<details>
<summary>深入 CC 源码</summary>

CC 的 worktree 系统有两条路径：**EnterWorktree**（当前会话切入）和 **AgentTool isolation**（子 agent 隔离）。

### EnterWorktree：当前会话切换

`EnterWorktreeTool.ts:92-97` 创建 worktree 后立即 `process.chdir(worktreePath)`、`setCwd()`、`setOriginalCwd()`、`saveWorktreeState()`。当前会话的工作目录直接切换到 worktree——不是 prompt 提醒，而是进程级目录变更。

`ExitWorktreeTool.ts:261-320` 的 keep/remove 都会 `restoreSessionToOriginalCwd()` 恢复原目录。Remove 时检查未提交改动（`ExitWorktreeTool.ts:190-220`），没有 `discard_changes: true` 就拒绝删除。

### AgentTool isolation：子 agent 隔离

`AgentTool.tsx:590-641` 在 `isolation: "worktree"` 时调用 `createAgentWorktree()` 创建 worktree，用 `cwdOverridePath` 包住子 agent 执行。子 agent 的所有操作自动在 worktree 目录下进行。`AgentTool/prompt.ts:272` 告诉模型：这是临时 worktree，无改动自动清理，有改动返回路径和分支。

`worktree.ts:902-951` 的 `createAgentWorktree()` 不修改全局 session cwd，只给子 agent 用。`worktree.ts:961-1020` 的 `removeAgentWorktree()` 从主 repo root 删除。

### name 校验

`worktree.ts:76-84` 校验 slug：拒绝 `.`/`..`，允许 `[a-zA-Z0-9._-]`。`worktree.ts:48` 定义 `VALID_WORKTREE_SLUG_SEGMENT`。教学版的 `validate_worktree_name` 用同样的规则。

### 路径和分支命名

真实路径是 `.claude/worktrees/`，分支名 `worktree-{slug}`（`worktree.ts:204-227`，斜杠用 `+` 替代）。教学版用 `.worktrees/` 和 `wt/{name}` 简化。

创建时用 `git worktree add -B`（`worktree.ts:326-328`），优先基于 `origin/<defaultBranch>` 而非当前 HEAD。

### 状态管理

CC 没有 task-worktree 绑定。Worktree 状态通过 `PersistedWorktreeSession`（`worktree.ts:756-768`）管理，字段包括 `originalCwd`、`worktreePath`、`worktreeName`、`worktreeBranch`、`originalBranch`、`originalHeadCommit`、`sessionId` 等——没有 taskId。`saveWorktreeState()`（`sessionStorage.ts:2883-2920`）以 `type: 'worktree-state'` 写入 session transcript。

教学版用 task 的 `worktree` 字段做绑定，是教学简化。CC 把 worktree 和 task 作为两个独立系统，通过 Agent 理解上下文来关联。

</details>

<!-- upstream-cc-source:end -->
