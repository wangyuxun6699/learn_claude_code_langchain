"""s17: Goal Loop -- 模型提出停止，独立评估器决定是否继续。

s01 以来，agent 循环只有一个退出条件：模型不再调用工具就返回。

这对普通对话够用，但对"一直修到测试全过""做完每一条验收标准"这类任务不够：
模型可能只做了一半就以为完工。没有新的 tool_use 只说明**这一轮**想停，并不证明**整个目标**达成。

/goal 在真正返回前加一个独立判断：
1. /goal 保存一个"完成条件"，并立刻把它作为当前任务交给主模型（无需再发一条"开始干活"）。
2. 主模型停止调用工具后，循环在返回边界跑 Goal Stop hook：让一个**没有工具、只读对话**的评估器
   判断条件是否已被对话里的具体结果满足。
3. 不满足 -> 把评估器给的简短理由追加回 messages，continue 再跑一轮；满足 -> 返回；不可能 -> 交还用户。

这是建立在 s04 内核（五个基础工具 + Hook + 权限）之上的独立机制示例。
"""

import os
import sys
import re
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage

load_dotenv(override=True)

MODEL_ID = os.getenv("MODEL_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("BASE_URL")
# 评估器可以用更小的模型；未设置时复用 MODEL_ID。
GOAL_EVALUATOR_MODEL_ID = os.getenv("GOAL_EVALUATOR_MODEL_ID") or MODEL_ID
# 自动延续的两道出口：全局轮数上限 + 连续 Stop-hook 阻塞上限。
MAX_TURNS = int(os.getenv("MAX_TURNS", "20"))
MAX_BLOCKS = int(os.getenv("MAX_BLOCKS", "5"))

WORKDIR = Path.cwd()


def build_model(model_id: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_id or MODEL_ID,
        max_completion_tokens=8000,
        temperature=0,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )


WORKER_MODEL = build_model()
EVALUATOR_MODEL = build_model(GOAL_EVALUATOR_MODEL_ID)

# 主模型的 system prompt：验证命令的结果要报得足够清楚，供独立评估器检查。
SYSTEM = (
    f"you are a coding agent at {WORKDIR}. Use tools to solve tasks. "
    "After running a verification command, report the command and its result clearly "
    "enough for an independent evaluator to inspect. Act, don't explain."
)


# ---------------------------------------------------------------------------
# 权限与 Hook（s04 内核）
# ---------------------------------------------------------------------------

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}


def register_hook(event, callback):
    HOOKS[event].append(callback)


def trigger_hooks(event, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]


def check_deny_list(command: str) -> str | None:
    for pattern in dangerous:
        if pattern in command:
            return f"blocked:{pattern} is on the deny list"
    return None


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


# ---------------------------------------------------------------------------
# 基础工具（s04 内核）
# ---------------------------------------------------------------------------

@tool
def run_bash(command: str) -> str:
    """Execute a shell command in the current workspace."""
    try:
        r = subprocess_run(command)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except Exception as e:
        return f"Error: {e}"


def subprocess_run(command: str):
    import subprocess
    return subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)


@tool
def run_read(path: str, limit: int | None = None) -> str:
    """Read a UTF-8 text file, optionally limiting the returned line count."""
    try:
        lines = resolve_path(path).read_text().splitlines()
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
        file_path.write_text(content)
        return f"write {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace the first exact occurrence of old_text in a UTF-8 file."""
    try:
        file_path = resolve_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
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


# ---------------------------------------------------------------------------
# GoalState / PromptGoalEvaluator / GoalController
# ---------------------------------------------------------------------------

class GoalState:
    def __init__(self):
        self.condition: str | None = None
        self.started_at: float | None = None
        self.evaluations: int = 0
        self.last_reason: str | None = None
        self.blocks: int = 0
        self.phase: str | None = None  # None | "complete" | "impossible"


class PromptGoalEvaluator:
    """用一次独立模型调用判断完成条件，不持有任何工具。"""

    SYSTEM = (
        "You are a goal evaluator. Judge whether the goal condition is satisfied by the "
        "conversation below. Return ONLY a JSON object with keys ok (bool), reason (string), "
        "impossible (bool). ok=true means concrete results already present in the conversation "
        "prove the condition (exit codes, command output, etc). Do NOT assume an unreported "
        "command succeeded. Set impossible=true only if the goal can no longer be achieved."
    )

    def evaluate(self, condition: str, messages: list) -> dict:
        trimmed = self._trim(messages)
        conversation = self._render(trimmed)
        try:
            response = EVALUATOR_MODEL.invoke([
                SystemMessage(content=self.SYSTEM),
                HumanMessage(content=f"Goal condition:\n{condition}\n\nConversation:\n{conversation}"),
            ])
            data = self._parse_json(response.content)
            return {
                "ok": bool(data.get("ok")),
                "reason": str(data.get("reason", "")),
                "impossible": bool(data.get("impossible")),
            }
        except Exception as e:
            # 评估器自己出错：停止自动延续，保留 goal，把错误交给用户，而不是假装成功。
            return {"ok": False, "reason": f"evaluator error: {e}", "impossible": False, "error": True}

    @staticmethod
    def _render(messages: list) -> str:
        lines = []
        for m in messages:
            role = m.__class__.__name__.replace("Message", "").lower()
            content = getattr(m, "content", "")
            text = content if isinstance(content, str) else json.dumps(content)
            lines.append(f"[{role}] {text}")
        return "\n".join(lines)

    @staticmethod
    def _trim(messages: list, max_chars: int = 12000) -> list:
        # 保留最近的完整消息；最新一条若过大，只留首尾，避免一个工具结果塞满评估请求。
        if not messages:
            return []
        total = sum(len(str(getattr(m, "content", ""))) for m in messages)
        if total <= max_chars:
            return messages
        # 从后往前保留，直到预算用尽；最新一条超大时截首尾。
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
    def _parse_json(raw) -> dict:
        text = str(raw).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        return json.loads(text)


class GoalController:
    def __init__(self):
        self.goal = GoalState()
        self.evaluator = PromptGoalEvaluator()

    def set(self, condition: str) -> str:
        self.goal.condition = condition
        self.goal.started_at = time.time()
        self.goal.evaluations = 0
        self.goal.last_reason = None
        self.goal.blocks = 0
        self.goal.phase = None
        return f"goal set: {condition}"

    def clear(self) -> str:
        self.goal = GoalState()
        return "goal cleared"

    def status(self) -> str:
        g = self.goal
        if not g.condition:
            return "no active goal"
        elapsed = int(time.time() - (g.started_at or time.time()))
        return (
            f"goal: {g.condition}\n"
            f"elapsed: {elapsed}s, evaluations: {g.evaluations}, blocks: {g.blocks}\n"
            f"last reason: {g.last_reason or '(none)'}\n"
            f"phase: {g.phase or 'active'}"
        )

    def evaluate_after_turn(self, messages: list) -> dict:
        """Goal Stop hook：在循环返回边界判断。返回 {action, reason}。"""
        if not self.goal.condition or self.goal.phase:
            # 无 goal（或已 complete/impossible）-> 放行，停止。
            return {"action": "allow", "reason": ""}

        self.goal.evaluations += 1
        result = self.evaluator.evaluate(self.goal.condition, messages)

        if result.get("error"):
            return {"action": "error", "reason": result["reason"]}

        if result["impossible"]:
            self.goal.phase = "impossible"
            self.goal.last_reason = result["reason"]
            return {"action": "impossible", "reason": result["reason"]}

        if result["ok"]:
            self.goal.phase = "complete"
            self.goal.last_reason = result["reason"] or "goal satisfied"
            return {"action": "allow", "reason": self.goal.last_reason}

        # 未满足 -> 阻塞返回，再跑一轮。
        self.goal.blocks += 1
        self.goal.last_reason = result["reason"] or "condition not yet satisfied"
        return {"action": "block", "reason": self.goal.last_reason}


# ---------------------------------------------------------------------------
# 主循环：s04 内核 + Goal Stop hook
# ---------------------------------------------------------------------------

def execute_tool_calls(response: AIMessage) -> list:
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


def run_with_goal(messages: list, controller: GoalController) -> dict:
    """手动 while 循环：模型停止工具调用后，由 Goal Stop hook 决定是否再来一轮。"""
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

        # 模型本轮不再调用工具 -> 返回边界 -> Goal Stop hook。
        decision = controller.evaluate_after_turn(messages)

        if decision["action"] == "block":
            if controller.goal.blocks > MAX_BLOCKS:
                return {"status": "max_blocks", "text": final_text(response),
                        "reason": f"exceeded {MAX_BLOCKS} consecutive stop blocks"}
            messages.append(HumanMessage(content=decision["reason"]))
            continue

        if decision["action"] == "allow":
            trigger_hooks("Stop", messages)
            return {"status": controller.goal.phase or "completed",
                    "text": final_text(response), "reason": decision["reason"]}

        if decision["action"] == "impossible":
            return {"status": "impossible", "text": final_text(response), "reason": decision["reason"]}

        if decision["action"] == "error":
            return {"status": "error", "text": final_text(response), "reason": decision["reason"]}

    return {"status": "max_turns", "text": "(turn limit reached)", "reason": f"exceeded {MAX_TURNS} turns"}


def final_text(response: AIMessage) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return str(content)


# ---------------------------------------------------------------------------
# /goal 命令解析与 CLI
# ---------------------------------------------------------------------------

CLEAR_ALIASES = {"clear", "stop", "off", "reset", "none", "cancel"}


def handle_goal_command(query: str, controller: GoalController) -> str:
    """解析 /goal 命令；返回 controller 状态文字或 "set" 表示已设置需要启动工作。"""
    rest = query.strip()[len("/goal"):].strip()
    if not rest:
        print(controller.status())
        return "status"
    if rest.lower() in CLEAR_ALIASES:
        print(controller.clear())
        return "clear"
    print(controller.set(rest))
    return "set"


def run_once(initial: str | None = None):
    controller = GoalController()
    history = []

    if initial and initial.strip().startswith("/goal") and initial.strip()[len("/goal"):].strip():
        condition = initial.strip()[len("/goal"):].strip()
        if condition.lower() in CLEAR_ALIASES:
            print(controller.clear())
            return
        controller.set(condition)
        history.append(HumanMessage(content=condition))
        result = run_with_goal(history, controller)
        print_status(result)
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
                condition = controller.goal.condition
                history.append(HumanMessage(content=condition))
                result = run_with_goal(history, controller)
                print_status(result)
            continue

        history.append(HumanMessage(content=query))
        result = run_with_goal(history, controller)
        print_status(result)


def print_status(result: dict):
    print(f"\n[status={result['status']}] {result['text']}")
    if result.get("reason"):
        print(f"[reason] {result['reason']}")
    print()


if __name__ == "__main__":
    run_once(sys.argv[1] if len(sys.argv) > 1 else None)
