#!/usr/bin/env python3
"""从各章 ``code.py`` 生成只移除注释的速读版。"""
from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTER_PATTERN = re.compile(r"s\d{2}_.+")


def strip_comments(source: str) -> str:
    """保留代码和 docstring，只删除 COMMENT token 与多余空行。"""
    tokens = [
        token
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type != tokenize.COMMENT
    ]
    text = tokenize.untokenize(tokens)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def main() -> None:
    chapters = sorted(
        path
        for path in ROOT.iterdir()
        if path.is_dir() and CHAPTER_PATTERN.fullmatch(path.name)
    )
    if len(chapters) != 17:
        raise RuntimeError(f"应有 17 个章节，实际发现 {len(chapters)} 个")

    # 同步已有速读版且本次展开了公共依赖的存档章节。
    chapters.extend(
        ROOT / "legacy" / name
        for name in ("s10_system_prompt", "s11_error_recovery")
    )

    for chapter in chapters:
        source_path = chapter / "code.py"
        target_path = chapter / "code_uncommented.py"
        target_path.write_text(
            strip_comments(source_path.read_text(encoding="utf-8")),
            encoding="utf-8",
            newline="\n",
        )
        print(target_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
