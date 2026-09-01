"""s17: Goal Loop -- the model stops, an independent evaluator decides (uncommented)."""
import os
import sys
import re
import json
import time
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage

load_dotenv(override=True)

MODEL_ID = os.getenv("MODEL_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("BASE_URL")
GOAL_EVALUATOR_MODEL_ID = os.getenv("GOAL_EVALUATOR_MODEL_ID") or MODEL_ID
MAX_TURNS = int(os.getenv("MAX_TURNS", "20"))
MAX_BLOCKS = int(os.getenv("MAX_BLOCKS", "5"))
WORKDIR = Path.cwd()


def build_model(model_id=None) -> ChatOpenAI:
    return ChatOpenAI(model=model_id or MODEL_ID, max_completion_tokens=8000, temperature=0,
                      api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


WORKER_MODEL = build_model()
EVALUATOR_MODEL = build_model(GOAL_EVALUATOR_MODEL_ID)

SYSTEM = (
    f"you are a coding agent at {WORKDIR}. Use tools to solve tasks. "
    "After running a verification command, report the command and its result clearly "
    "enough for an independent evaluator to inspect. Act, don't explain."
)

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}


def register_hook(event, callback):
    HOOKS[event].append(callback)


def trigger_hooks(event, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


from harness.security import check_deny_list


def resolve_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (WORKDIR / candidate).resolve()


def check_rules(tool_name: str, args: dict) -> str | None:
    if tool_name == "run_bash":
        command = args.get("command", "")
        if command.strip().lower().startswith("del ") or any(kw in command for kw in ["rm ", "> /etc/", "chmod 777"]):
            return "potentially destructive command"
    if tool_name in ("run_write", "run_edit", "run_read"):
        if not resolve_path(args.get("path", "")).is_relative_to(WORKDIR):
            return "Working outside workspace"
    return None


def ask_user(tool_name: str, args: dict, reason: str) -> bool:
    print(f"\nWarning: {reason}")
    print(f"Tool: {tool_name}({args})")
    return input("Allow? [y/N] ").strip().lower() in ("y", "yes")


def check_permission(tool_name: str, args: dict) -> bool:
    if tool_name == "run_bash":
        reason = check_deny_list(args.get("command", ""))
        if reason:
            print(f"\nBlocked:{reason}")
            return False
    reason = check_rules(tool_name, args)
    if reason:
        return ask_user(tool_name, args, reason)
    return True


def on_pre_tool_use(tool_name, tool_args):
    print("[PreToolUse]", tool_name, tool_args)
    if not check_permission(tool_name, tool_args):
        return "Permission denied"


register_hook("UserPromptSubmit", lambda c: print("[UserPromptSubmit]", c))
register_hook("PreToolUse", on_pre_tool_use)
register_hook("PostToolUse", lambda n, a, r: print("[PostToolUse]", n))
register_hook("Stop", lambda msgs: print("[Stop]", len(msgs)))


@tool
def run_bash(command: str) -> str:
    """Execute a shell command in the current workspace."""
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except Exception as e:
        return f"Error: {e}"


@tool
def run_read(path: str, limit: int | None = None) -> str:
    """Read a UTF-8 text file, optionally limiting the returned line count."""
    try:
        lines = resolve_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"...({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@tool
def run_write(path: str, content: str) -> str:
    """Write UTF-8 content to a file, replacing its existing content."""
    try:
        file_path = resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"write {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace the first exact occurrence of old_text in a UTF-8 file."""
    try:
        file_path = resolve_path(path)
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"edit {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def run_glob(pattern: str) -> str:
    """Find workspace files matching a glob pattern."""
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


TOOLS = [run_bash, run_read, run_write, run_edit, run_glob]
TOOL_MAP = {t.name: t for t in TOOLS}


class GoalState:
    def __init__(self):
        self.condition = None
        self.started_at = None
        self.evaluations = 0
        self.last_reason = None
        self.blocks = 0
        self.phase = None


class PromptGoalEvaluator:
    SYSTEM = (
        "You are a goal evaluator. Judge whether the goal condition is satisfied by the "
        "conversation below. Return ONLY a JSON object with keys ok (bool), reason (string), "
        "impossible (bool). ok=true means concrete results already present in the conversation "
        "prove the condition. Do NOT assume an unreported command succeeded. Set impossible=true "
        "only if the goal can no longer be achieved."
    )

    def evaluate(self, condition, messages, max_retries=2):
        conversation = self._render(self._trim(messages))
        last_error = None
        for _ in range(max_retries + 1):
            try:
                response = EVALUATOR_MODEL.invoke([
                    SystemMessage(content=self.SYSTEM),
                    HumanMessage(content=f"Goal condition:\n{condition}\n\nConversation:\n{conversation}"),
                ])
                data = self._parse_json(response.content)
                return {"ok": bool(data.get("ok")), "reason": str(data.get("reason", "")),
                        "impossible": bool(data.get("impossible"))}
            except Exception as e:
                last_error = e
        return {"ok": False, "reason": f"evaluator error after retries: {last_error}", "impossible": False,
                "error": True}

    @staticmethod
    def _render(messages):
        lines = []
        for m in messages:
            role = m.__class__.__name__.replace("Message", "").lower()
            content = getattr(m, "content", "")
            text = content if isinstance(content, str) else json.dumps(content)
            lines.append(f"[{role}] {text}")
        return "\n".join(lines)

    @staticmethod
    def _trim(messages, max_chars=12000):
        if not messages:
            return []
        total = sum(len(str(getattr(m, "content", ""))) for m in messages)
        if total <= max_chars:
            return messages
        kept = []
        budget = max_chars
        last = messages[-1]
        last_text = str(getattr(last, "content", ""))
        if len(last_text) > budget:
            half = max(1, budget // 2)
            clipped = last_text[:half] + "\n...(truncated middle)...\n" + last_text[-half:]
            last = HumanMessage(content=clipped)
            budget = 0
        else:
            budget -= len(last_text)
        kept = [last]
        for m in reversed(messages[:-1]):
            text = str(getattr(m, "content", ""))
            if len(text) <= budget:
                kept.insert(0, m)
                budget -= len(text)
            else:
                break
        return kept

    @staticmethod
    def _parse_json(raw):
        text = str(raw).strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        start = text.find("{")
        if start == -1:
            raise ValueError("evaluator output has no JSON object")
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
        raise ValueError("unbalanced JSON in evaluator output")


class GoalController:
    def __init__(self):
        self.goal = GoalState()
        self.evaluator = PromptGoalEvaluator()

    def set(self, condition):
        g = self.goal
        g.condition = condition
        g.started_at = time.time()
        g.evaluations = 0
        g.last_reason = None
        g.blocks = 0
        g.phase = None
        return f"goal set: {condition}"

    def clear(self):
        self.goal = GoalState()
        return "goal cleared"

    def status(self):
        g = self.goal
        if not g.condition:
            return "no active goal"
        elapsed = int(time.time() - (g.started_at or time.time()))
        return (f"goal: {g.condition}\nelapsed: {elapsed}s, evaluations: {g.evaluations}, "
                f"blocks: {g.blocks}\nlast reason: {g.last_reason or '(none)'}\nphase: {g.phase or 'active'}")

    def evaluate_after_turn(self, messages):
        g = self.goal
        if not g.condition or g.phase:
            return {"action": "allow", "reason": ""}
        g.evaluations += 1
        result = self.evaluator.evaluate(g.condition, messages)
        if result.get("error"):
            return {"action": "error", "reason": result["reason"]}
        if result["impossible"]:
            g.phase = "impossible"
            g.last_reason = result["reason"]
            return {"action": "impossible", "reason": result["reason"]}
        if result["ok"]:
            g.phase = "complete"
            g.last_reason = result["reason"] or "goal satisfied"
            return {"action": "allow", "reason": g.last_reason}
        g.blocks += 1
        g.last_reason = result["reason"] or "condition not yet satisfied"
        return {"action": "block", "reason": g.last_reason}


def execute_tool_calls(response):
    outputs = []
    for tool_call in response.tool_calls:
        name = tool_call["name"]
        args = tool_call.get("args", {})
        blocked = trigger_hooks("PreToolUse", name, args)
        if blocked:
            outputs.append(ToolMessage(content=str(blocked), tool_call_id=tool_call["id"], name=name, status="error"))
            continue
        tool = TOOL_MAP.get(name)
        if tool is None:
            output = f"Error: unknown tool {name}"
        else:
            try:
                output = tool.invoke(args)
            except Exception as e:
                output = f"Error: {e}"
        result = ToolMessage(content=str(output), tool_call_id=tool_call["id"], name=name)
        trigger_hooks("PostToolUse", name, args, result)
        outputs.append(result)
    return outputs


def final_text(response):
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return str(content)


def run_with_goal(messages, controller):
    if messages:
        trigger_hooks("UserPromptSubmit", getattr(messages[-1], "content", messages[-1]))
    llm = WORKER_MODEL.bind_tools(TOOLS)
    turns = 0
    while turns < MAX_TURNS:
        turns += 1
        response = llm.invoke([SystemMessage(content=SYSTEM)] + messages)
        messages.append(response)
        if response.tool_calls:
            messages.extend(execute_tool_calls(response))
            continue
        decision = controller.evaluate_after_turn(messages)
        if decision["action"] == "block":
            if controller.goal.blocks > MAX_BLOCKS:
                return {"status": "max_blocks", "text": final_text(response),
                        "reason": f"exceeded {MAX_BLOCKS} consecutive stop blocks"}
            messages.append(HumanMessage(content=decision["reason"]))
            continue
        if decision["action"] == "allow":
            trigger_hooks("Stop", messages)
            return {"status": controller.goal.phase or "completed", "text": final_text(response),
                    "reason": decision["reason"]}
        if decision["action"] == "impossible":
            return {"status": "impossible", "text": final_text(response), "reason": decision["reason"]}
        if decision["action"] == "error":
            return {"status": "error", "text": final_text(response), "reason": decision["reason"]}
    return {"status": "max_turns", "text": "(turn limit reached)", "reason": f"exceeded {MAX_TURNS} turns"}


CLEAR_ALIASES = {"clear", "stop", "off", "reset", "none", "cancel"}


def handle_goal_command(query, controller):
    rest = query.strip()[len("/goal"):].strip()
    if not rest:
        print(controller.status())
        return "status"
    if rest.lower() in CLEAR_ALIASES:
        print(controller.clear())
        return "clear"
    print(controller.set(rest))
    return "set"


def print_status(result):
    print(f"\n[status={result['status']}] {result['text']}")
    if result.get("reason"):
        print(f"[reason] {result['reason']}")
    print()


def run_once(initial=None):
    controller = GoalController()
    history = []
    if initial and initial.strip().startswith("/goal") and initial.strip()[len("/goal"):].strip():
        condition = initial.strip()[len("/goal"):].strip()
        if condition.lower() in CLEAR_ALIASES:
            print(controller.clear())
            return
        controller.set(condition)
        history.append(HumanMessage(content=condition))
        print_status(run_with_goal(history, controller))
        return

    print("s17: Goal Loop")
    print("输入 /goal <条件> 设定目标并开始；/goal 查看状态；/goal clear 清除；q 退出。\n")
    while True:
        try:
            query = input("\033[36ms17 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip().startswith("/goal"):
            action = handle_goal_command(query, controller)
            if action == "set":
                history.append(HumanMessage(content=controller.goal.condition))
                print_status(run_with_goal(history, controller))
            continue
        history.append(HumanMessage(content=query))
        print_status(run_with_goal(history, controller))


if __name__ == "__main__":
    run_once(sys.argv[1] if len(sys.argv) > 1 else None)
