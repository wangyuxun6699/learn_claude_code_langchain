#!/usr/bin/env python3
"""合并上游中文章节 README，并保留本仓库的“深入源码”补充。"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTER_PATTERN = re.compile(r"s\d{2}_.+")
DEEP_SUMMARIES = (
    "<summary>深入 CC 源码</summary>",
    "<summary>深入 Claude Code 源码</summary>",
)
LOCAL_START = "<!-- local-langchain-additions:start -->"
LOCAL_END = "<!-- local-langchain-additions:end -->"


def extract_deep_details(text: str) -> str:
    """提取包含指定 summary 的完整 details 块，支持嵌套 details。"""
    summary_at = min(
        (position for marker in DEEP_SUMMARIES if (position := text.find(marker)) >= 0),
        default=-1,
    )
    if summary_at < 0:
        return ""
    start = text.rfind("<details>", 0, summary_at)
    if start < 0:
        return ""

    token_pattern = re.compile(r"<details>|</details>")
    depth = 0
    for match in token_pattern.finditer(text, start):
        if match.group() == "<details>":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return text[start : match.end()].strip()
    raise ValueError("深入源码 details 块没有闭合")


def extract_local_additions(text: str, deep: str) -> str:
    """读取已标记的本地补充；首次合并时完整保留原 README 的其余内容。"""
    start = text.find(LOCAL_START)
    end = text.find(LOCAL_END)
    if start >= 0 and end > start:
        return text[start + len(LOCAL_START) : end].strip()

    original = text.replace(deep, "", 1).strip() if deep else text.strip()
    if not original:
        return ""
    return "\n".join(
        [
            "<details>",
            "<summary>展开本仓库原有的 LangChain / LangGraph 教学说明</summary>",
            "",
            original,
            "",
            "</details>",
        ]
    )


def adapt_setup(text: str) -> str:
    """把上游 Anthropic 环境说明改为本项目的 OpenAI-compatible 配置。"""
    text = text.replace(
        "填入 ANTHROPIC_API_KEY 和 MODEL_ID",
        "填入 OPENAI_API_KEY、BASE_URL 和 MODEL_ID",
    )
    text = text.replace(
        "ANTHROPIC_API_KEY=...",
        "OPENAI_API_KEY=...\nBASE_URL=https://your-openai-compatible-endpoint/v1",
    )
    return re.sub(r"\n*<!-- translation-sync:[^\n]*-->\s*$", "", text)


def merge(upstream: str, local: str, chapter: str) -> str:
    canonical = adapt_setup(upstream).strip()
    first_line, remainder = canonical.split("\n", 1)
    note = (
        f"> **对齐状态**：本章 `code.py` 对齐上游 `{chapter}`；"
        "模型请求由 `harness/langchain_messages.py` 转换为 LangChain "
        "OpenAI-compatible 调用，循环和 Harness 机制保持上游结构。"
    )
    deep = extract_deep_details(local)
    local_additions = extract_local_additions(local, deep)
    pieces = [first_line, "", note, remainder.strip()]
    if local_additions:
        pieces.extend(
            [
                "---",
                "",
                "## 本项目保留的 LangChain / LangGraph 教学补充",
                "",
                "> 以下内容来自本仓库对齐前的 README，作为上游课程之外的本地教学补充完整保留。",
                "",
                LOCAL_START,
                local_additions,
                LOCAL_END,
            ]
        )
    if deep:
        pieces.extend(
            [
                "---",
                "",
                "## 本项目保留的 Claude Code 源码补充",
                "",
                "> 以下内容来自本仓库原有 README，作为上游课程之外的源码研读补充。",
                "",
                deep,
            ]
        )
    return "\n".join(pieces).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "upstream_root",
        type=Path,
        help="learn-claude-code 上游仓库根目录",
    )
    parser.add_argument(
        "--local-ref",
        help="首次合并时从指定 Git ref 读取原有 README，例如 HEAD",
    )
    args = parser.parse_args()
    upstream_root = args.upstream_root.resolve()

    chapters = sorted(
        path
        for path in ROOT.iterdir()
        if path.is_dir() and CHAPTER_PATTERN.fullmatch(path.name)
    )
    if len(chapters) != 17:
        raise RuntimeError(f"应有 17 个章节，实际发现 {len(chapters)} 个")

    for chapter in chapters:
        local_path = chapter / "README.md"
        upstream_path = upstream_root / chapter.name / "README.zh.md"
        if not upstream_path.is_file():
            raise FileNotFoundError(upstream_path)
        if args.local_ref:
            relative = local_path.relative_to(ROOT).as_posix()
            completed = subprocess.run(
                ["git", "show", f"{args.local_ref}:{relative}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            local_text = completed.stdout
        else:
            local_text = local_path.read_text(encoding="utf-8")
        local_path.write_text(
            merge(
                upstream_path.read_text(encoding="utf-8"),
                local_text,
                chapter.name,
            ),
            encoding="utf-8",
            newline="\n",
        )
        print(local_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
