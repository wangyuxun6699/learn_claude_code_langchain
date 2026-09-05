from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from test_agent_loop_boundaries import load_lesson

from scripts.merge_chapter_readmes import extract_deep_details, merge
from scripts.sync_cc_source_readmes import END, START

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = sorted(ROOT.glob("s[0-9][0-9]_*"))
UPSTREAM_COMMIT = "08263f49b3d5c895ea61d56a3737d8eebe624f20"


def test_alignment_lock_records_the_synced_upstream_commit() -> None:
    lock = json.loads((ROOT / "upstream.lock.json").read_text(encoding="utf-8"))
    assert lock["repository"] == "https://github.com/shareAI-lab/learn-claude-code"
    assert lock["commit"] == UPSTREAM_COMMIT


def test_chapter_documentation_is_chinese_only() -> None:
    assert len(CHAPTERS) == 17
    for chapter in CHAPTERS:
        assert (chapter / "README.md").is_file()
        assert not (chapter / "README.zh.md").exists()
        assert not (chapter / "README.ja.md").exists()
        assert not (chapter / "README.en.md").exists()
        text = (chapter / "README.md").read_text(encoding="utf-8")
        assert "<!-- local-langchain-additions:start -->" not in text
        assert "## 结合" in text
        assert "https://docs.langchain.com/" in text
        assert "深入 CC 源码" in text or "深入 Claude Code 源码" in text


def test_uncommented_sources_have_the_same_python_ast() -> None:
    for chapter in CHAPTERS:
        commented = ast.parse((chapter / "code.py").read_text(encoding="utf-8"))
        uncommented = ast.parse(
            (chapter / "code_uncommented.py").read_text(encoding="utf-8")
        )
        assert ast.dump(commented, include_attributes=False) == ast.dump(
            uncommented, include_attributes=False
        ), chapter.name


@pytest.mark.parametrize("chapter", CHAPTERS[2:15], ids=lambda path: path.name)
def test_langchain_message_adapter_preserves_tool_protocol(monkeypatch, tmp_path, chapter) -> None:
    langchain_messages = load_lesson(tmp_path, chapter / "code.py")
    calls: dict = {}

    class FakeRunnable:
        def invoke(self, messages):
            calls["messages"] = messages
            return AIMessage(
                content="先读取文件",
                tool_calls=[
                    {
                        "id": "call_2",
                        "name": "read_file",
                        "args": {"path": "README.md"},
                        "type": "tool_call",
                    }
                ],
                usage_metadata={
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                },
                response_metadata={"finish_reason": "tool_calls"},
            )

    class FakeModel:
        def __init__(self, **kwargs):
            calls["model_kwargs"] = kwargs

        def bind_tools(self, tools):
            calls["tools"] = tools
            return FakeRunnable()

    monkeypatch.setattr(langchain_messages, "ChatOpenAI", FakeModel)
    client = langchain_messages.LangChainMessagesClient(
        base_url="https://example.invalid/v1"
    )
    response = client.messages.create(
        model="test-model",
        system=[{"type": "text", "text": "系统提示"}],
        messages=[
            {"role": "user", "content": "开始"},
            {
                "role": "assistant",
                "content": [
                    langchain_messages.ToolUseBlock(
                        id="call_1", name="bash", input={"command": "pwd"}
                    )
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "D:/workspace",
                    }
                ],
            },
        ],
        tools=[
            {
                "name": "read_file",
                "description": "读取文件",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
        max_tokens=321,
    )

    assert response.stop_reason == "tool_use"
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7
    assert response.content[0].type == "text"
    assert response.content[1].type == "tool_use"
    assert response.content[1].input == {"path": "README.md"}
    assert calls["model_kwargs"]["max_completion_tokens"] == 321
    assert calls["tools"][0]["function"]["parameters"]["required"] == ["path"]
    assert isinstance(calls["messages"][0], SystemMessage)
    assert isinstance(calls["messages"][1], HumanMessage)
    assert isinstance(calls["messages"][2], AIMessage)
    assert isinstance(calls["messages"][3], ToolMessage)


def test_cc_source_blocks_match_the_pinned_originals() -> None:
    record = json.loads((ROOT / "upstream.lock.json").read_text(encoding="utf-8"))["cc_source_readmes"]
    assert record["commit"] == "67a9126c6435a8654ba7a6f68c0fd2130f00a462"
    for chapter, sources in record["chapters"].items():
        text = (ROOT / chapter / "README.md").read_text(encoding="utf-8")
        assert text.count(START) == text.count(END) == 1
        section = text.split(START, 1)[1].split(END, 1)[0]
        hashes = []
        while block := extract_deep_details(section):
            hashes.append(hashlib.sha256(block.encode("utf-8")).hexdigest())
            section = section.replace(block, "", 1)
        assert hashes == [source["sha256"] for source in sources if source["sha256"]], chapter
        if not hashes:
            assert section.strip() == "## 深入 CC 源码", chapter
            assert not extract_deep_details(text), chapter


@pytest.mark.parametrize("chapter", ["s13_agent_teams", "s15_integrated_harness", "s16_workflow_runtime"])
def test_readme_merge_keeps_integrated_guide_and_source_section(chapter) -> None:
    local = (ROOT / chapter / "README.md").read_text(encoding="utf-8")
    section = local.split(START, 1)[1].split(END, 1)[0]
    guide = local.split("## 结合", 1)[1].split("\n## ", 1)[0]
    result = merge("# Updated lesson\n\nNew teaching content.\n", local, chapter)
    assert result.split(START, 1)[1].split(END, 1)[0] == section
    assert guide in result
    assert "<!-- local-langchain-additions:start -->" not in result
    assert result.count(START) == result.count(END) == 1


def test_lessons_do_not_import_the_shared_harness() -> None:
    paths = [*ROOT.glob("s*/code*.py"), *ROOT.glob("legacy/s*/code*.py")]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("harness"), path
            elif isinstance(node, ast.Import):
                assert not any(alias.name.startswith("harness") for alias in node.names), path


@pytest.mark.parametrize("chapter", [
    "s01_agent_loop", "s11_background_tasks", "s17_goal_loop",
])
def test_chapter_starts_from_a_single_copied_file(tmp_path, chapter) -> None:
    # Exercise actual CLI imports outside the repository, including Goal's
    # lazy model adapter and Windows process helpers.
    script = tmp_path / "code.py"
    shutil.copy2(ROOT / chapter / "code.py", script)
    env = {
        **os.environ,
        "PYTHONPATH": "",
        "PYTHON_DOTENV_DISABLED": "1",
        "MODEL_ID": "test-model",
        "OPENAI_API_KEY": "test-key",
        "BASE_URL": "https://example.invalid/v1",
    }
    result = subprocess.run(
        [sys.executable, str(script)], input="q\n", cwd=tmp_path, env=env,
        capture_output=True, text=True, errors="replace", timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_team_file_locks_import_without_the_repository(tmp_path) -> None:
    # Load the copied runtime without entering the existing POSIX select-based
    # CLI; file locking must remain usable on Windows as well as Unix.
    script = tmp_path / "code.py"
    shutil.copy2(ROOT / "s13_agent_teams" / "code.py", script)
    env = {**os.environ, "PYTHONPATH": "", "PYTHON_DOTENV_DISABLED": "1", "MODEL_ID": "test-model"}
    result = subprocess.run(
        [sys.executable, "-c", "import runpy; runpy.run_path('code.py')"],
        cwd=tmp_path, env=env, capture_output=True, text=True, errors="replace", timeout=30,
    )
    assert result.returncode == 0, result.stderr
