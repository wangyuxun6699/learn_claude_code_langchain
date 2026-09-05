"""Offline startup checks for every runnable lesson entry point."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = [
    "s01_agent_loop",
    "s02_tool_use",
    "s03_permission",
    "s04_hooks",
    "s05_todo_write",
    "s06_subagent",
    "s07_skill_loading",
    "s08_context_compact",
    "s09_memory",
    "s10_task_system",
    "s11_background_tasks",
    "s12_cron_scheduler",
    "s13_agent_teams",
    "s14_mcp_plugin",
    "s15_integrated_harness",
    "s16_workflow_runtime",
    "s17_goal_loop",
]


def test_every_chapter_has_runnable_and_readable_variants() -> None:
    for chapter in CHAPTERS:
        chapter_dir = ROOT / chapter
        assert (chapter_dir / "README.md").is_file()
        assert (chapter_dir / "code.py").is_file()
        assert (chapter_dir / "code_uncommented.py").is_file()


@pytest.mark.parametrize("chapter", CHAPTERS)
def test_chapter_starts_and_quits_without_network(
    chapter: str,
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "MODEL_ID": "ci-test-model",
            "OPENAI_API_KEY": "ci-test-key",
            "BASE_URL": "http://127.0.0.1:9/v1",
            "PYTHON_DOTENV_DISABLED": "1",
            "PYTHONUTF8": "1",
        }
    )
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
    )

    result = subprocess.run(
        [sys.executable, "-m", f"{chapter}.code"],
        cwd=tmp_path,
        env=env,
        input="q\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, (
        f"{chapter} failed to start and quit cleanly\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
