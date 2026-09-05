from types import SimpleNamespace

from langchain.messages import AIMessage, ToolMessage
from test_agent_loop_boundaries import load_lesson
from test_alignment import ROOT


def middleware_request(name: str, args: dict, call_id: str = "call_1"):
    return SimpleNamespace(
        tool_call={"name": name, "args": args, "id": call_id, "type": "tool_call"}
    )


def test_s04_registers_all_four_lifecycle_events(tmp_path):
    lesson = load_lesson(tmp_path, ROOT / "s04_hooks" / "code.py")

    assert set(lesson.HOOKS) == {
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "Stop",
    }
    assert lesson.HOOKS["PreToolUse"] == [lesson.permission_hook, lesson.log_hook]
    assert lesson.HOOKS["PostToolUse"] == [lesson.large_output_hook]


def test_trigger_hooks_runs_in_order_and_short_circuits(tmp_path):
    lesson = load_lesson(tmp_path, ROOT / "s04_hooks" / "code.py")
    calls = []
    lesson.HOOKS["PreToolUse"] = [
        lambda _block: calls.append("first"),
        lambda _block: calls.append("block") or "denied",
        lambda _block: calls.append("unreachable"),
    ]

    result = lesson.trigger_hooks(
        "PreToolUse", lesson.ToolUseBlock("call_1", "bash", {"command": "pwd"})
    )

    assert result == "denied"
    assert calls == ["first", "block"]


def test_hook_middleware_wraps_a_successful_tool_call(tmp_path):
    lesson = load_lesson(tmp_path, ROOT / "s04_hooks" / "code.py")
    events = []
    lesson.HOOKS["PreToolUse"] = [
        lambda block: events.append(("pre", block.name, block.input))
    ]
    lesson.HOOKS["PostToolUse"] = [
        lambda block, output: events.append(("post", block.name, output))
    ]
    request = middleware_request("read_file", {"path": "README.md"})
    expected = ToolMessage(content="file contents", tool_call_id="call_1", name="read_file")

    result = lesson.HookMiddleware().wrap_tool_call(request, lambda _request: expected)

    assert result is expected
    assert events == [
        ("pre", "read_file", {"path": "README.md"}),
        ("post", "read_file", "file contents"),
    ]


def test_hook_middleware_returns_a_paired_error_when_pre_hook_blocks(tmp_path):
    lesson = load_lesson(tmp_path, ROOT / "s04_hooks" / "code.py")
    lesson.HOOKS["PreToolUse"] = [lambda _block: "policy denied"]
    lesson.HOOKS["PostToolUse"] = [
        lambda *_args: (_ for _ in ()).throw(AssertionError("post hook must not run"))
    ]

    result = lesson.HookMiddleware().wrap_tool_call(
        middleware_request("bash", {"command": "pwd"}),
        lambda _request: (_ for _ in ()).throw(AssertionError("blocked tool executed")),
    )

    assert result == ToolMessage(
        content="policy denied",
        tool_call_id="call_1",
        name="bash",
        status="error",
    )


def test_s04_create_agent_uses_hook_middleware_and_s03_tools(monkeypatch, tmp_path):
    lesson = load_lesson(tmp_path, ROOT / "s04_hooks" / "code.py")
    calls = {}
    fake_model = object()
    fake_agent = object()

    def fake_chat_openai(**kwargs):
        calls["model"] = kwargs
        return fake_model

    monkeypatch.setattr(lesson, "ChatOpenAI", fake_chat_openai)

    def fake_create_agent(**kwargs):
        calls["agent"] = kwargs
        return fake_agent

    monkeypatch.setattr(lesson, "create_agent", fake_create_agent)

    assert lesson.get_agent() is fake_agent
    assert lesson.get_agent() is fake_agent
    assert [tool.name for tool in calls["agent"]["tools"]] == [
        "bash",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
    ]
    assert len(calls["agent"]["middleware"]) == 1
    assert isinstance(calls["agent"]["middleware"][0], lesson.HookMiddleware)


def test_stop_hook_can_inject_a_message_and_run_agent_again(monkeypatch, tmp_path):
    lesson = load_lesson(tmp_path, ROOT / "s04_hooks" / "code.py")

    class FakeAgent:
        def __init__(self):
            self.calls = []

        def stream(self, payload, **_kwargs):
            current = list(payload["messages"])
            self.calls.append(current)
            yield {
                "type": "values",
                "data": {"messages": [*current, AIMessage(content=f"turn {len(self.calls)}")]},
            }

    fake_agent = FakeAgent()
    monkeypatch.setattr(lesson, "get_agent", lambda: fake_agent)
    stop_calls = []

    def stop_hook(messages):
        stop_calls.append(list(messages))
        return "continue checking" if len(stop_calls) == 1 else None

    lesson.HOOKS["Stop"] = [stop_hook]
    messages = [{"role": "user", "content": "start"}]

    lesson.agent_loop(messages)

    assert len(fake_agent.calls) == 2
    assert fake_agent.calls[1][-1] == {"role": "user", "content": "continue checking"}
    assert len(stop_calls) == 2
    assert messages[-1].content == "turn 2"


def test_stop_summary_counts_langchain_and_course_tool_results(tmp_path):
    lesson = load_lesson(tmp_path, ROOT / "s04_hooks" / "code.py")
    messages = [
        ToolMessage(content="one", tool_call_id="call_1"),
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_2", "content": "two"}
            ],
        },
    ]

    assert lesson._tool_result_count(messages) == 2
