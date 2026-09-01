"""s16: Workflow Runtime -- saved orchestration + journal resume (uncommented)."""
import os
import sys
import json
import re
import time
import uuid
import hashlib
import asyncio
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from harness.paths import safe_path
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import create_agent

load_dotenv(override=True)

MODEL_ID = os.getenv("MODEL_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("BASE_URL")
RUNTIME_DIR = Path(__file__).resolve().parent / ".runtime"

# run_read 只允许读仓库根目录（工作区）内的文件，防止读任意绝对路径。
WORKDIR = Path(__file__).resolve().parent.parent
RUNNER = None


class WorkflowInputError(Exception):
    pass


class SimpleJsonSchema:
    def __init__(self, schema):
        self.schema = schema

    def validate(self, value):
        return self._validate(value, self.schema, "#")

    def _validate(self, value, schema, path):
        if not isinstance(schema, dict):
            return True, ""
        stype = schema.get("type")
        if stype == "object":
            if not isinstance(value, dict):
                return False, f"{path} must be an object"
            props = schema.get("properties", {}) or {}
            required = schema.get("required", []) or []
            for k in required:
                if k not in value:
                    return False, f"{path} missing required key '{k}'"
            if schema.get("additionalProperties") is False:
                for k in value:
                    if k not in props:
                        return False, f"{path} unexpected key '{k}'"
            for k, v in value.items():
                if k in props:
                    ok, err = self._validate(v, props[k], f"{path}.{k}")
                    if not ok:
                        return ok, err
            return True, ""
        if stype == "array":
            if not isinstance(value, list):
                return False, f"{path} must be an array"
            for i, item in enumerate(value):
                ok, err = self._validate(item, schema.get("items", {}), f"{path}[{i}]")
                if not ok:
                    return ok, err
            return True, ""
        if stype == "string" and not isinstance(value, str):
            return False, f"{path} must be a string"
        if stype == "boolean" and not isinstance(value, bool):
            return False, f"{path} must be a boolean"
        if stype in ("number", "integer") and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            return False, f"{path} must be a number"
        enum = schema.get("enum")
        if enum is not None and value not in enum:
            return False, f"{path} must be one of {enum}"
        return True, ""


def parse_json(raw):
    if isinstance(raw, (dict, list)):
        return raw
    text = str(raw).strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = re.sub(r"^" + fence + r"[a-zA-Z]*\s*", "", text)
        text = re.sub(fence + r"\s*$", "", text).strip()
    try:
        return json.loads(text)
    except Exception as e:
        raise WorkflowInputError(f"invalid JSON output: {e}")


def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (10 ** 10)


def make_key(kind, label, prompt, schema):
    basis = f"{kind}|{label}|{prompt}|{json.dumps(schema, sort_keys=True)}"
    return f"{kind}-{_stable_hash(basis):010d}"


MISS = object()


class WorkflowJournal:
    def __init__(self, path: Path):
        self.path = path
        self.cache = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.cache[entry["key"]] = entry["value"]
        self._f = path.open("a", encoding="utf-8")

    def record(self, key, value):
        self._f.write(json.dumps({"key": key, "value": value}) + "\n")
        self._f.flush()
        self.cache[key] = value

    def cached(self, key):
        return self.cache.get(key, MISS)

    def close(self):
        self._f.close()


class LocalWorkflowTask:
    def __init__(self, name, run_id):
        self.name = name
        self.run_id = run_id
        self.status = "created"
        self.progress = []
        self.agents = 0
        self.tokens = 0

    def progress_event(self, ptype, **data):
        self.progress.append({"type": ptype, **data})
        payload = " ".join(f"{k}={v}" for k, v in data.items())
        print(f"  progress   {ptype} {payload}".rstrip())

    def serialize(self):
        return {
            "name": self.name, "runId": self.run_id, "status": self.status,
            "agents": self.agents, "tokens": self.tokens,
            "events": sum(1 for p in self.progress if p["type"] in ("workflow_agent",)),
        }


class LiveAgentRunner:
    def run(self, prompt, schema, label) -> str:
        system = "You are a workflow agent. Return ONLY valid JSON (no prose, no code fences)."
        if schema:
            system += f"\nSchema: {json.dumps(schema)}"
        model = ChatOpenAI(model=MODEL_ID, max_completion_tokens=4000, temperature=0,
                           api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        return str(model.invoke([SystemMessage(content=system), HumanMessage(content=prompt)]).content)


class MockAgentRunner:
    def __init__(self):
        self.calls = 0

    def run(self, prompt, schema, label) -> str:
        self.calls += 1
        if str(label).startswith("audit:"):
            dim = str(label).split(":", 1)[1]
            return json.dumps({"findings": [
                {"title": f"{dim} issue A", "description": f"a {dim} concern"},
                {"title": f"{dim} issue B", "description": f"another {dim} concern"},
            ]})
        if str(label).startswith("verify:"):
            return json.dumps({"isReal": (_stable_hash(str(label)) % 2) == 0})
        return json.dumps({"findings": []})


class ExecutionState:
    def __init__(self, journal, runner, task, resume_run_id=None):
        self.journal = journal
        self.runner = runner
        self.task = task
        self.resume_run_id = resume_run_id

    def phase(self, title):
        self.task.progress_event("phase", phase=title)

    def log(self, message):
        self.task.progress_event("log", message=message)

    async def agent(self, prompt, schema=None, label=None, phase=None):
        if phase:
            self.phase(phase)
        key = make_key("agent", label, prompt, schema)
        cached = self.journal.cached(key)
        if cached is not MISS:
            self.task.progress_event("workflow_agent", label=label, status="cached")
            return cached
        self.task.progress_event("workflow_agent", label=label, status="started")
        result = await self._run_and_validate(prompt, schema, label)
        self.journal.record(key, result)
        self.task.agents += 1
        self.task.tokens += len(str(result))
        self.task.progress_event("workflow_agent", label=label, status="done")
        return result

    async def _run_and_validate(self, prompt, schema, label):
        run = await asyncio.to_thread(self.runner.run, prompt, schema, label)
        value = parse_json(run)
        if schema is not None:
            ok, err = SimpleJsonSchema(schema).validate(value)
            if not ok:
                run2 = await asyncio.to_thread(self.runner.run, prompt + "\n\nReturn valid JSON matching the schema.", schema, label)
                value = parse_json(run2)
                ok, err = SimpleJsonSchema(schema).validate(value)
                if not ok:
                    raise WorkflowInputError(f"agent(label={label}) returned invalid output: {err}")
        return value

    async def parallel(self, thunks):
        return await asyncio.gather(*[t() for t in thunks])

    async def pipeline(self, items, *stages):
        async def run_item(item, idx):
            value = item
            for stage in stages:
                value = await stage(value, item, idx)
            return value

        return await asyncio.gather(*[run_item(it, i) for i, it in enumerate(items)])

    async def workflow(self, name, args=None):
        entry = WORKFLOWS.get(name)
        if not entry:
            raise WorkflowInputError(f"unknown workflow '{name}'")
        _, script_fn = entry
        sub_task = LocalWorkflowTask(name=name, run_id=self.task.run_id + "-sub")
        sub_ctx = ExecutionState(self.journal, self.runner, sub_task, self.resume_run_id)
        return await script_fn(sub_ctx, args or {})


WORKFLOW_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def validate_meta(meta):
    if not isinstance(meta, dict):
        raise WorkflowInputError("meta must be an object literal")
    if not meta.get("name") or not meta.get("description"):
        raise WorkflowInputError("meta requires name and description")
    if not isinstance(meta["name"], str) or not WORKFLOW_NAME_RE.fullmatch(meta["name"]):
        raise WorkflowInputError("meta.name must be a safe 1-64 character slug")
    if "phases" in meta and (not isinstance(meta["phases"], list)
                             or not all(isinstance(p, str) and p for p in meta["phases"])):
        raise WorkflowInputError("meta.phases must contain non-empty strings")
    return meta


FINDINGS_SCHEMA = {
    "type": "object",
    "properties": {"findings": {"type": "array", "items": {
        "type": "object",
        "properties": {"title": {"type": "string"}, "description": {"type": "string"}},
        "required": ["title", "description"],
    }}},
    "required": ["findings"],
}
VERDICT_SCHEMA = {"type": "object", "properties": {"isReal": {"type": "boolean"}}, "required": ["isReal"]}
DIMENSIONS = ["security", "performance", "maintainability"]


async def sample_workflow(ctx, args):
    ctx.phase("Review")
    changes = args.get("changes", "")

    async def audit(_v, dimension, _i):
        out = await ctx.agent(f"Inspect this change for {dimension} issues:\n{changes}",
                              schema=FINDINGS_SCHEMA, label=f"audit:{dimension}", phase="Review")
        return {"dimension": dimension, "findings": out["findings"]}

    async def verify(audited, dimension, _i):
        ctx.phase("Verify")
        verdicts = await ctx.parallel([
            (lambda f=f: ctx.agent(f"Verify this finding:\n{changes}\n\n{f}",
                                   schema=VERDICT_SCHEMA, label=f"verify:{dimension}:{f['title']}"))
            for f in audited["findings"]
        ])
        return {"dimension": dimension,
                "confirmed": [f for f, v in zip(audited["findings"], verdicts) if v and v.get("isReal")]}

    results = await ctx.pipeline(DIMENSIONS, audit, verify)
    confirmed = [f for r in results if r for f in r["confirmed"]]
    ctx.log(f"Confirmed {len(confirmed)} real issue(s)")
    return {"confirmed": confirmed}


SAMPLE_META = {"name": "review-changes", "description": "Review code changes",
               "phases": ["Review", "Verify"]}
WORKFLOWS = {"review-changes": (SAMPLE_META, sample_workflow)}


class WorkflowTool:
    def run(self, meta, script_fn, args=None, resume_from_run_id=None):
        validate_meta(meta)
        args = args or {}
        if resume_from_run_id:
            run_id = resume_from_run_id
            if not self._snapshot_path(run_id).exists():
                raise WorkflowInputError(f"unknown resume_from_run_id '{run_id}'")
        else:
            run_id = self._reserve_run_id(meta)
        journal = WorkflowJournal(self._journal_path(run_id))
        task = LocalWorkflowTask(name=meta["name"], run_id=run_id)
        task.status = "running"
        task.progress_event("task_started", name=meta["name"], runId=run_id)
        ctx = ExecutionState(journal, RUNNER, task, resume_from_run_id)
        try:
            result = asyncio.run(script_fn(ctx, args))
            task.status = "completed"
        except Exception as e:
            task.status = "failed"
            result = {"error": f"{type(e).__name__}: {e}"}
        journal.close()
        self._write_output(run_id, result)
        self._write_snapshot(run_id, meta, args, task)
        task.progress_event("task_notification", status=task.status, files=self._run_files(run_id))
        return {"launched": {"runId": run_id, "name": meta["name"]}, "result": result, "task": task.serialize()}

    def _lock_path(self, run_id): return RUNTIME_DIR / f"{run_id}.lock"

    def _journal_path(self, run_id): return RUNTIME_DIR / f"{run_id}.journal.jsonl"

    def _snapshot_path(self, run_id): return RUNTIME_DIR / f"{run_id}.json"

    def _output_path(self, run_id): return RUNTIME_DIR / f"{run_id}.output.json"

    def _reserve_run_id(self, meta):
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        for _ in range(100):
            run_id = f"{meta['name']}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            try:
                self._lock_path(run_id).open("x").close()
                return run_id
            except FileExistsError:
                continue
        raise WorkflowInputError("could not reserve a fresh run id")

    def _run_files(self, run_id):
        return [p.name for p in RUNTIME_DIR.glob(f"{run_id}*")]

    def _write_output(self, run_id, result):
        self._output_path(run_id).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_snapshot(self, run_id, meta, args, task):
        self._snapshot_path(run_id).write_text(json.dumps({
            "runId": run_id, "name": meta["name"], "args": args, "task": task.serialize(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")


def _last_run_id():
    if not RUNTIME_DIR.exists():
        return None
    snapshots = sorted(RUNTIME_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    return snapshots[-1].stem if snapshots else None


@tool
def run_workflow(name: str, args: dict | None = None, resume_from_run_id: str | None = None) -> str:
    """Run a saved workflow by name (e.g. 'review-changes')."""
    entry = WORKFLOWS.get(name)
    if not entry:
        return f"Error: unknown workflow '{name}'. Available: {', '.join(WORKFLOWS)}"
    try:
        out = WorkflowTool().run(entry[0], entry[1], args, resume_from_run_id)
        return json.dumps(out, ensure_ascii=False)
    except WorkflowInputError as e:
        return f"Error: {e}"


def run_demo():
    global RUNNER
    RUNNER = MockAgentRunner()
    print("=== demo: run review-changes (mock) ===")
    out = WorkflowTool().run(validate_meta(SAMPLE_META), sample_workflow, {"changes": "demo diff"})
    run_id = out["launched"]["runId"]
    print("\nresult:", out["result"], "agents:", out["task"]["agents"])
    print("=== demo: resume same runId ===")
    out2 = WorkflowTool().run(validate_meta(SAMPLE_META), sample_workflow,
                              {"changes": "demo diff"}, resume_from_run_id=run_id)
    print("\nresult:", out2["result"], "agents:", out2["task"]["agents"], "tokens:", out2["task"]["tokens"])
    print("\nStored under:", RUNTIME_DIR)


def run_resume():
    global RUNNER
    run_id = _last_run_id()
    if not run_id:
        print("no previous run to resume; run demo first")
        return
    RUNNER = MockAgentRunner()
    snapshot = json.loads((RUNTIME_DIR / f"{run_id}.json").read_text(encoding="utf-8"))
    meta = {"name": snapshot["name"], "description": "resumed", "phases": SAMPLE_META.get("phases")}
    out = WorkflowTool().run(validate_meta(meta), WORKFLOWS[snapshot["name"]][1],
                             snapshot["args"], resume_from_run_id=run_id)
    print("\nresult:", out["result"], "agents:", out["task"]["agents"], "tokens:", out["task"]["tokens"])


def run_interactive():
    global RUNNER
    RUNNER = LiveAgentRunner()

    @tool
    def run_read(path: str, limit: int | None = None) -> str:
        """Read a UTF-8 text file, optionally limiting line count."""
        try:
            lines = safe_path(WORKDIR, path).read_text(encoding="utf-8", errors="replace").splitlines()
            if limit and limit < len(lines):
                lines = lines[:limit] + [f"...({len(lines) - limit} more lines)"]
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    model = ChatOpenAI(model=MODEL_ID, max_completion_tokens=8000, temperature=0,
                       api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    agent = create_agent(
        model=model, tools=[run_read, run_workflow],
        system_prompt=("You are a coding agent. Run the saved 'review-changes' workflow with "
                       "run_workflow(name='review-changes', args={'changes': '...'}). Summarize the result."),
    )
    print("s16: Workflow Runtime (interactive)")
    history = []
    while True:
        try:
            query = input("\033[36ms16 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append(HumanMessage(content=query))
        result = agent.invoke({"messages": history})
        history[:] = result["messages"]
        last = history[-1]
        if isinstance(last, AIMessage):
            content = last.content
            if isinstance(content, list):
                content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
            print(content)
        print()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "interactive"
    if mode == "demo":
        run_demo()
    elif mode == "resume":
        run_resume()
    else:
        run_interactive()
