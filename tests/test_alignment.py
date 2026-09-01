from __future__ import annotations

import ast
import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from harness import langchain_messages

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
        assert "<!-- local-langchain-additions:start -->" in text
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


def test_langchain_message_adapter_preserves_tool_protocol(monkeypatch) -> None:
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
