#!/usr/bin/env python3
"""按主题同步指定版本的“深入 CC 源码”原文；没有原文的章节留空。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .merge_chapter_readmes import extract_deep_details
except ImportError:
    from merge_chapter_readmes import extract_deep_details

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- upstream-cc-source:start -->"
END = "<!-- upstream-cc-source:end -->"
REPOSITORY = "https://github.com/shareAI-lab/learn-claude-code"
BRANCH = "fix/s08-s20-sync-frontmatter-parser"
COMMIT = "67a9126c6435a8654ba7a6f68c0fd2130f00a462"

# 旧分支是 20 章，新主线是 17 章；s13 合并了四个团队主题。
CHAPTER_MAP = {
    "s01_agent_loop": ["s01_agent_loop"],
    "s02_tool_use": ["s02_tool_use"],
    "s03_permission": ["s03_permission"],
    "s04_hooks": ["s04_hooks"],
    "s05_todo_write": ["s05_todo_write"],
    "s06_subagent": ["s06_subagent"],
    "s07_skill_loading": ["s07_skill_loading"],
    "s08_context_compact": ["s08_context_compact"],
    "s09_memory": ["s09_memory"],
    "s10_task_system": ["s12_task_system"],
    "s11_background_tasks": ["s13_background_tasks"],
    "s12_cron_scheduler": ["s14_cron_scheduler"],
    "s13_agent_teams": [
        "s15_agent_teams", "s16_team_protocols",
        "s17_autonomous_agents", "s18_worktree_isolation",
    ],
    "s14_mcp_plugin": ["s19_mcp_plugin"],
    # s20 README 存在，但没有“深入 CC 源码”块；新增的两章不存在。
    "s15_integrated_harness": ["s20_comprehensive"],
    "s16_workflow_runtime": [],
    "s17_goal_loop": [],
}


def without_previous_source(text: str) -> str:
    """移除上一次同步区或旧的本地源码补充，不改 README 其他教学内容。"""
    if START in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        return (before.rstrip() + "\n" + after).strip()
    old_heading = "## 本项目保留的 Claude Code 源码补充"
    if old_heading in text:
        before, old = text.split(old_heading, 1)
        deep = extract_deep_details(old)
        if deep:
            after = old[old.index(deep) + len(deep):]
            return (before.rstrip().removesuffix("---").rstrip() + "\n" + after).strip()
    while deep := extract_deep_details(text):
        text = text.replace(deep, "", 1)
    return text.strip()


def sync(upstream_root: Path, root: Path = ROOT) -> dict:
    records = {}
    pending = {}
    for chapter, upstream_chapters in CHAPTER_MAP.items():
        target = root / chapter / "README.md"
        text = without_previous_source(target.read_text(encoding="utf-8"))
        parts = [START, "## 深入 CC 源码", ""]
        sources = []
        for upstream_chapter in upstream_chapters:
            relative = f"{upstream_chapter}/README.md"
            original = (upstream_root / relative).read_text(encoding="utf-8")
            block = extract_deep_details(original)
            sources.append({
                "path": relative,
                "sha256": hashlib.sha256(block.encode("utf-8")).hexdigest() if block else None,
            })
            if not block:
                continue
            parts.extend([
                f"> 原文：[{upstream_chapter}]({REPOSITORY}/blob/{COMMIT}/{relative})。"
                "以下折叠块保持原文，文中的章号与源码行号沿用该版本。",
                "", block, "",
            ])
        parts.append(END)
        # 没有原文时只保留空标题，不编写占位解释或自行补充。
        pending[target] = text + "\n\n" + "\n".join(parts) + "\n"
        records[chapter] = sources
    # 先确认所有来源存在，再统一落盘，避免缺文件时只同步一半。
    for target, text in pending.items():
        target.write_text(text, encoding="utf-8", newline="\n")
    return {"repository": REPOSITORY, "branch": BRANCH, "commit": COMMIT, "chapters": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream_root", type=Path, help="指定分支提交的本地仓库目录")
    args = parser.parse_args()
    record = sync(args.upstream_root.resolve())
    lock_path = ROOT / "upstream.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["cc_source_readmes"] = record
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已同步 {len(record['chapters'])} 份 README；无原文章节留空。")


if __name__ == "__main__":
    main()
