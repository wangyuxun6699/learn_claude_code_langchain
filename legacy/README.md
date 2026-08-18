# Legacy（20 章旧编排存档）

本目录保存本仓库从 20 章编排迁移到 17 章编排时移除/合并的章节，内容原样保留，供对照参考。

## 迁移说明

参考仓库 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) 已把主线从 20 章改为 17 章（根目录 `s01_*` ～ `s17_*`）。本仓库同步跟进，把旧 20 章中「没有对应新章号」的章节移入本目录：

| 旧章节 | 去向 |
|---|---|
| s10_system_prompt | 17 章主线不再单独成章，移至本目录 |
| s11_error_recovery | 17 章主线不再单独成章，移至本目录 |
| s16_team_protocols | 团队协议已并入新 s13_agent_teams，移至本目录 |
| s17_autonomous_agents | 自主认领已并入新 s13_agent_teams，移至本目录 |
| s18_worktree_isolation | 任务绑定的 worktree 已并入新 s13_agent_teams，移至本目录 |

## 新旧章节对应关系

| 旧 20 章 | 新 17 章 | 主题 |
|---|---|---|
| s01_agent_loop | s01_agent_loop | Agent Loop |
| s02_tool_use | s02_tool_use | Tool Use |
| s03_permission | s03_permission | Permission |
| s04_hooks | s04_hooks | Hooks |
| s05_todo_write | s05_todo_write | TodoWrite |
| s06_subagent | s06_subagent | Subagent |
| s07_skill_loading | s07_skill_loading | Skill Loading |
| s08_context_compact | s08_context_compact | Context Compact |
| s09_memory | s09_memory | Memory |
| s10_system_prompt | —（移除） | System Prompt |
| s11_error_recovery | —（移除） | Error Recovery |
| s12_task_system | s10_task_system | Task System |
| s13_background_tasks | s11_background_tasks | Background Tasks |
| s14_cron_scheduler | s12_cron_scheduler | Cron Scheduler |
| s15_agent_teams | s13_agent_teams | Agent Teams |
| s16_team_protocols | s13_agent_teams（合并） | Team Protocols |
| s17_autonomous_agents | s13_agent_teams（合并） | 自主认领任务 |
| s18_worktree_isolation | s13_agent_teams（合并） | 任务绑定的 Worktree |
| s19_mcp_plugin | s14_mcp_plugin | MCP Plugin |
| s20_comprehensive | s15_integrated_harness | Integrated Harness |
| —（新增） | s16_workflow_runtime | Workflow Runtime |
| —（新增） | s17_goal_loop | Goal Loop |

## 注意

本目录下的章节仍是旧 20 章编号，其内部 `python -m sXX_*` 命令、章节导航链接和跨章节 import（例如 `from s13_background_tasks import code`）仍指向旧编号，直接运行可能失败。它们只是存档，不作为当前主线的一部分。
