#!/usr/bin/env python3
"""合并上游中文章节 README，并保留本地代码导读与源码原文区。"""
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
SOURCE_START = "<!-- upstream-cc-source:start -->"
SOURCE_END = "<!-- upstream-cc-source:end -->"


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


def extract_integrated_guide(text: str) -> str:
    """提取以“结合...”开头的本地代码与 LangChain/LangGraph 导读。"""
    match = re.search(r"^## 结合[^\n]*$", text, flags=re.M)
    if not match:
        return ""
    next_heading = re.search(r"^## ", text[match.end() :], flags=re.M)
    end = match.end() + next_heading.start() if next_heading else len(text)
    return text[match.start() : end].strip()


def insert_integrated_guide(text: str, guide: str) -> str:
    """把本地导读放回运行示例之前；没有运行小节时追加到正文末尾。"""
    if not guide:
        return text
    for heading in ("## 试一下", "## 跑起来看看", "## 接下来"):
        marker = f"\n{heading}"
        if marker in text:
            return text.replace(marker, f"\n{guide}\n\n---\n\n{heading}", 1)
    return f"{text.rstrip()}\n\n---\n\n{guide}"


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
    source_section = ""
    if SOURCE_START in local:
        start = local.index(SOURCE_START)
        end = local.index(SOURCE_END, start) + len(SOURCE_END)
        source_section = local[start:end]
        local = local[:start] + local[end:]
    del chapter  # 保留 CLI/API 兼容参数；章节名已包含在上游与本地内容中。
    guide = extract_integrated_guide(local)
    pieces = [insert_integrated_guide(canonical, guide)]
    if source_section:
        pieces.extend(["", source_section])
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
