from langchain.messages import AIMessage, AIMessageChunk, HumanMessage
from test_agent_loop_boundaries import ROOT, load_lesson


def test_s02_builds_create_agent_with_five_tools(monkeypatch, tmp_path):
    lesson = load_lesson(tmp_path, ROOT / "s02_tool_use" / "code.py")
    calls = {}
    fake_model = object()
    fake_agent = object()

    def fake_chat_openai(**kwargs):
        calls["model"] = kwargs
        return fake_model

    def fake_create_agent(**kwargs):
        calls["agent"] = kwargs
        return fake_agent

    monkeypatch.setattr(lesson, "ChatOpenAI", fake_chat_openai)
    monkeypatch.setattr(lesson, "create_agent", fake_create_agent)

    assert lesson.get_agent() is fake_agent
    assert lesson.get_agent() is fake_agent
    assert calls["model"]["model"] == "test-model"
    assert [tool.name for tool in calls["agent"]["tools"]] == [
        "bash",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
    ]
    assert calls["agent"]["system_prompt"] == lesson.SYSTEM


def test_s02_tools_expose_schemas_and_stay_inside_workspace(tmp_path):
    lesson = load_lesson(tmp_path, ROOT / "s02_tool_use" / "code.py")

    assert lesson.read_file.args_schema.model_json_schema()["properties"] == {
        "path": {"title": "Path", "type": "string"},
        "limit": {
            "anyOf": [{"type": "integer"}, {"type": "null"}],
            "default": None,
            "title": "Limit",
        },
    }

    written = lesson.write_file.invoke({"path": "nested/note.txt", "content": "第一行\n第二行\n第三行\n"})
    read = lesson.read_file.invoke({"path": "nested/note.txt", "limit": 2})
    edited = lesson.edit_file.invoke({"path": "nested/note.txt", "old_text": "第二行", "new_text": "已修改"})
    escaped = lesson.write_file.invoke({"path": "../outside.txt", "content": "should not be written"})

    assert written == "Wrote 12 characters to nested/note.txt"
    assert read == "第一行\n第二行\n... (1 more lines)"
    assert edited == "Edited nested/note.txt"
    assert escaped.startswith("Error: Path escapes workspace:")
    assert (tmp_path / "nested" / "note.txt").read_text(encoding="utf-8") == ("第一行\n已修改\n第三行\n")
    assert not (tmp_path.parent / "outside.txt").exists()


def test_s02_streams_tokens_and_keeps_final_history(tmp_path, capsys):
    lesson = load_lesson(tmp_path, ROOT / "s02_tool_use" / "code.py")
    final_messages = [
        HumanMessage(content="读取文件"),
        AIMessage(content="文件内容是……"),
    ]

    class FakeAgent:
        def stream(self, state, *, stream_mode, version):
            assert state["messages"] == [{"role": "user", "content": "读取文件"}]
            assert stream_mode == ["messages", "values"]
            assert version == "v2"
            yield {
                "type": "messages",
                "data": (AIMessageChunk(content="文件内容"), {"langgraph_node": "model"}),
            }
            yield {
                "type": "messages",
                "data": (AIMessageChunk(content="是……"), {"langgraph_node": "model"}),
            }
            yield {"type": "values", "data": {"messages": final_messages}}

    lesson.agent = FakeAgent()
    history = [{"role": "user", "content": "读取文件"}]

    lesson.agent_loop(history)

    assert capsys.readouterr().out == "文件内容是……\n"
    assert history == final_messages
