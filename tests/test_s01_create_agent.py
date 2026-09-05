from langchain.messages import AIMessage, AIMessageChunk, HumanMessage
from lesson_loader import ROOT, load_lesson


def test_s01_builds_and_reuses_create_agent(monkeypatch, tmp_path):
    lesson = load_lesson(tmp_path, ROOT / "s01_agent_loop" / "code.py")
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
    assert calls["agent"] == {
        "model": fake_model,
        "tools": [lesson.bash],
        "system_prompt": lesson.SYSTEM,
    }


def test_s01_streams_tokens_and_keeps_final_history(tmp_path, capsys):
    lesson = load_lesson(tmp_path, ROOT / "s01_agent_loop" / "code.py")
    final_messages = [
        HumanMessage(content="你好"),
        AIMessage(content="你好，我能帮你什么？"),
    ]

    class FakeAgent:
        def stream(self, state, *, stream_mode, version):
            assert state["messages"] == [{"role": "user", "content": "你好"}]
            assert stream_mode == ["messages", "values"]
            assert version == "v2"
            yield {
                "type": "messages",
                "data": (AIMessageChunk(content="你好，"), {"langgraph_node": "model"}),
            }
            yield {
                "type": "messages",
                "data": (AIMessageChunk(content="我能帮你什么？"), {"langgraph_node": "model"}),
            }
            yield {"type": "values", "data": {"messages": final_messages}}

    lesson.agent = FakeAgent()
    history = [{"role": "user", "content": "你好"}]

    lesson.agent_loop(history)

    assert capsys.readouterr().out == "你好，我能帮你什么？\n"
    assert history == final_messages
