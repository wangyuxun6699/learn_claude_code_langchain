"""
s14：Cron Scheduler。

本章在 s13 的后台任务 Agent 上增加定时触发能力。核心数据流分成四层：

    Scheduler 负责判断时间
        -> cron_queue 保存已经到期的任务
        -> Queue Processor 等待 Agent 空闲
        -> Agent consumer 把任务作为 HumanMessage 注入会话

调度线程从不直接调用模型。这样即使 Agent 正在处理用户请求，时间检查仍会继续；
反过来，模型执行较慢也不会阻塞下一次时间检查。持久化只保存任务定义，Python
进程退出后 daemon 线程不会继续运行。

运行方式：python -m s14_cron_scheduler.code
"""

from __future__ import annotations

import html
import json
import secrets
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from threading import Event, Lock, RLock, Thread
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from s13_background_tasks import code as base


# ============================================================
# 1. Cron 任务模型与进程内共享状态
# ============================================================

# 沿用参考仓库的文件名。文件位于启动程序时的工作目录，而不是包目录。
DURABLE_PATH = base.WORKDIR / ".scheduled_tasks.json"
# 与真实 Claude Code 的任务数上限保持一致，避免失控的模型无限注册任务。
MAX_CRON_JOBS = 50


@dataclass
class CronJob:
    id: str
    cron: str
    prompt: str
    recurring: bool
    durable: bool


scheduled_jobs: dict[str, CronJob] = {}
# deque 只保存“已经触发、尚未交付”的任务；它不是任务定义的真实来源。
cron_queue: deque[CronJob] = deque()
# 值包含日期，确保每天同一分钟都能触发，同时同一分钟不会重复触发。
last_fired: dict[str, str] = {}

# scheduled_jobs、cron_queue 和 last_fired 必须在同一把锁下保持一致。
# save_durable_jobs 会在已持锁的函数中再次取得锁，所以这里必须使用 RLock。
cron_lock = RLock()
# 用户输入与自动定时回合共享同一个 Agent 状态，不允许并发修改。
agent_lock = RLock()
service_lock = Lock()
stop_event = Event()
services_started = False


# ============================================================
# 2. 五段式 Cron 的校验与匹配
# ============================================================

def _cron_field_matches(field: str, value: int) -> bool:
    """Return whether one validated cron field matches a value."""
    # 逗号列表递归复用同一套匹配规则，例如 1,5,10。
    if "," in field:
        return any(
            _cron_field_matches(part.strip(), value)
            for part in field.split(",")
        )

    if field == "*":
        return True

    # */N 从字段最小自然起点按 N 取模；分钟的 */5 会匹配 0、5、10……
    if field.startswith("*/"):
        return value % int(field[2:]) == 0

    if "-" in field:
        low_text, high_text = field.split("-", 1)
        return int(low_text) <= value <= int(high_text)

    return value == int(field)


def _validate_cron_field(
    field: str,
    low: int,
    high: int,
) -> str | None:
    """Validate one cron field against its numeric bounds."""
    if not field:
        return "字段不能为空"

    # 校验必须发生在匹配前，否则 int(field) 可能让调度线程抛异常。
    if "," in field:
        parts = field.split(",")

        if any(not part.strip() for part in parts):
            return f"列表格式无效：{field}"

        for part in parts:
            error = _validate_cron_field(
                part.strip(),
                low,
                high,
            )

            if error:
                return error

        return None

    if field == "*":
        return None

    if field.startswith("*/"):
        step_text = field[2:]

        if not step_text.isdigit():
            return f"步长无效：{field}"

        if int(step_text) <= 0:
            return f"步长必须大于 0：{field}"

        return None

    if "-" in field:
        start_text, end_text = field.split("-", 1)

        if not start_text.isdigit() or not end_text.isdigit():
            return f"范围格式无效：{field}"

        start = int(start_text)
        end = int(end_text)

        if not low <= start <= high or not low <= end <= high:
            return f"范围越界：{field}，允许 {low}-{high}"

        if start > end:
            return f"范围起点大于终点：{field}"

        return None

    if not field.isdigit():
        return f"字段格式无效：{field}"

    value = int(field)

    if not low <= value <= high:
        return f"值越界：{value}，允许 {low}-{high}"

    return None


def validate_cron(cron_expr: str) -> str | None:
    """Validate a five-field cron expression."""
    if not isinstance(cron_expr, str):
        return "cron 表达式必须是字符串"

    fields = cron_expr.strip().split()

    if len(fields) != 5:
        return f"cron 表达式必须有 5 段，实际有 {len(fields)} 段"

    # 星期采用 cron 约定：0=周日，1=周一，……，6=周六。
    specs = [
        ("分钟", 0, 59),
        ("小时", 0, 23),
        ("日", 1, 31),
        ("月", 1, 12),
        ("星期", 0, 6),
    ]

    for field, (name, low, high) in zip(fields, specs):
        error = _validate_cron_field(field, low, high)

        if error:
            return f"{name}字段错误：{error}"

    return None


def cron_matches(cron_expr: str, current: datetime) -> bool:
    """Return whether a cron expression matches a local datetime."""
    if validate_cron(cron_expr) is not None:
        return False

    minute, hour, day, month, weekday = cron_expr.strip().split()
    # datetime.weekday() 是周一=0，需要转换为 cron 的周日=0。
    cron_weekday = (current.weekday() + 1) % 7

    if not _cron_field_matches(minute, current.minute):
        return False

    if not _cron_field_matches(hour, current.hour):
        return False

    if not _cron_field_matches(month, current.month):
        return False

    day_matches = _cron_field_matches(day, current.day)
    weekday_matches = _cron_field_matches(
        weekday,
        cron_weekday,
    )

    # 标准 cron 语义：日和星期都受约束时采用 OR，而不是 AND。
    day_is_wildcard = day == "*"
    weekday_is_wildcard = weekday == "*"

    if day_is_wildcard and weekday_is_wildcard:
        return True

    if day_is_wildcard:
        return weekday_matches

    if weekday_is_wildcard:
        return day_matches

    return day_matches or weekday_matches


# ============================================================
# 3. 持久化与任务注册表
# ============================================================

def save_durable_jobs() -> None:
    """Atomically persist durable cron job definitions."""
    with cron_lock:
        tasks = [
            asdict(job)
            for job in scheduled_jobs.values()
            if job.durable
        ]
        payload = json.dumps(
            {"tasks": tasks},
            ensure_ascii=False,
            indent=2,
        )

        # 先完整写临时文件，再原子替换，避免进程中断留下半截 JSON。
        temporary_path = DURABLE_PATH.with_suffix(".json.tmp")
        temporary_path.write_text(payload, encoding="utf-8")
        temporary_path.replace(DURABLE_PATH)


def _load_job(raw_job: Any) -> CronJob:
    # JSON 能解码并不代表字段类型可信；磁盘输入仍要逐项校验。
    if not isinstance(raw_job, dict):
        raise ValueError("任务记录必须是对象")

    job = CronJob(**raw_job)

    if not isinstance(job.id, str) or not job.id.strip():
        raise ValueError("任务 ID 不能为空")

    error = validate_cron(job.cron)

    if error:
        raise ValueError(error)

    if not isinstance(job.prompt, str) or not job.prompt.strip():
        raise ValueError("prompt 不能为空")

    if not isinstance(job.recurring, bool):
        raise ValueError("recurring 必须是布尔值")

    if job.durable is not True:
        raise ValueError("持久化文件只能包含 durable 任务")

    return job


def load_durable_jobs() -> None:
    """Restore valid durable jobs without letting one bad record abort startup."""
    if not DURABLE_PATH.is_file():
        return

    try:
        payload = json.loads(
            DURABLE_PATH.read_text(encoding="utf-8")
        )
    except Exception as exc:
        print(
            f"[cron] 无法读取 {DURABLE_PATH.name}："
            f"{type(exc).__name__}：{exc}"
        )
        return

    # 当前格式是 {"tasks": [...]}；同时接受参考实现早期使用的裸列表。
    if isinstance(payload, dict):
        raw_jobs = payload.get("tasks")
    else:
        raw_jobs = payload

    if not isinstance(raw_jobs, list):
        print(
            f"[cron] 无法读取 {DURABLE_PATH.name}："
            "tasks 必须是列表"
        )
        return

    loaded = 0

    with cron_lock:
        # 单条损坏记录只会被跳过，不会让其他有效任务无法恢复。
        for index, raw_job in enumerate(raw_jobs):
            if len(scheduled_jobs) >= MAX_CRON_JOBS:
                print(
                    f"[cron] 最多恢复 {MAX_CRON_JOBS} 个任务，"
                    "其余记录已跳过"
                )
                break

            try:
                job = _load_job(raw_job)

                if job.id in scheduled_jobs:
                    raise ValueError(f"任务 ID 重复：{job.id}")

                scheduled_jobs[job.id] = job
                loaded += 1

            except Exception as exc:
                print(
                    f"[cron] 跳过第 {index + 1} 条无效任务："
                    f"{type(exc).__name__}：{exc}"
                )

    if loaded:
        print(f"[cron] 已恢复 {loaded} 个持久任务")


def schedule_job(
    cron: str,
    prompt: str,
    recurring: bool = True,
    durable: bool = True,
) -> CronJob:
    """Validate and register a new cron job."""
    if not isinstance(cron, str):
        raise ValueError("cron 表达式必须是字符串")

    if not isinstance(prompt, str):
        raise ValueError("prompt 必须是字符串")

    if not isinstance(recurring, bool):
        raise ValueError("recurring 必须是布尔值")

    if not isinstance(durable, bool):
        raise ValueError("durable 必须是布尔值")

    cron = cron.strip()
    prompt = prompt.strip()
    error = validate_cron(cron)

    if error:
        raise ValueError(error)

    if not prompt:
        raise ValueError("prompt 不能为空")

    # 注册和持久化构成一个小事务：写盘失败时回滚内存中的新任务。
    with cron_lock:
        if len(scheduled_jobs) >= MAX_CRON_JOBS:
            raise ValueError(
                f"定时任务最多 {MAX_CRON_JOBS} 个，请先取消一个"
            )

        while True:
            job_id = f"cron_{secrets.token_hex(4)}"

            if job_id not in scheduled_jobs:
                break

        job = CronJob(
            id=job_id,
            cron=cron,
            prompt=prompt,
            recurring=recurring,
            durable=durable,
        )
        scheduled_jobs[job.id] = job

        try:
            if durable:
                save_durable_jobs()
        except Exception:
            scheduled_jobs.pop(job.id, None)
            raise

    print(
        f"[cron register] {job.id} "
        f"'{job.cron}' -> {job.prompt[:60]}"
    )
    return job


def cancel_job(job_id: str) -> CronJob | None:
    """Cancel one job and return it, or return None if it does not exist."""
    with cron_lock:
        job = scheduled_jobs.pop(job_id, None)

        if job is None:
            return None

        # 删除 durable 任务时也要立即重写文件；失败则恢复内存状态。
        try:
            if job.durable:
                save_durable_jobs()
        except Exception:
            scheduled_jobs[job.id] = job
            raise

        last_fired.pop(job_id, None)

    print(f"[cron cancel] {job_id}")
    return job


def list_jobs() -> list[CronJob]:
    """Return a stable snapshot of all registered jobs."""
    with cron_lock:
        return sorted(
            scheduled_jobs.values(),
            key=lambda job: job.id,
        )


def consume_cron_queue() -> list[CronJob]:
    """Remove and return every fired job waiting for delivery."""
    with cron_lock:
        # 批量快照并清空，使同一批任务只交付一次。
        jobs = list(cron_queue)
        cron_queue.clear()
        return jobs


def has_cron_queue() -> bool:
    """Return whether fired jobs are waiting for delivery."""
    with cron_lock:
        return bool(cron_queue)


# ============================================================
# 4. Producer：独立调度线程
# ============================================================

def cron_scheduler_loop() -> None:
    """Poll once a second and enqueue each matching job once per minute."""
    while not stop_event.wait(1.0):
        current = datetime.now()
        # 只记录 HH:MM 会导致第二天相同时间被误认为已经触发过。
        minute_marker = current.strftime("%Y-%m-%d %H:%M")

        with cron_lock:
            for job in list(scheduled_jobs.values()):
                try:
                    if not cron_matches(job.cron, current):
                        continue

                    if last_fired.get(job.id) == minute_marker:
                        continue

                    # 一次性 durable 任务先从注册表和磁盘删除，成功后才入队。
                    # 这能避免写盘失败时执行了任务，却在重启后再次执行。
                    if not job.recurring:
                        scheduled_jobs.pop(job.id, None)

                        try:
                            if job.durable:
                                save_durable_jobs()
                        except Exception:
                            scheduled_jobs[job.id] = job
                            raise

                    # 调度线程只生产工作，不直接执行模型回合。
                    cron_queue.append(job)
                    last_fired[job.id] = minute_marker
                    print(
                        f"\n[cron fire] {job.id} "
                        f"-> {job.prompt[:60]}"
                    )

                except Exception as exc:
                    print(
                        f"[cron error] {job.id}："
                        f"{type(exc).__name__}：{exc}"
                    )


# ============================================================
# 5. 暴露给模型的三个 Cron 工具
# ============================================================

@tool("schedule_cron")
def run_schedule_cron(
    cron: str,
    prompt: str,
    recurring: bool = True,
    durable: bool = True,
) -> str:
    """创建五段式 cron 任务；时间按本机时区解释。"""
    try:
        job = schedule_job(
            cron,
            prompt,
            recurring,
            durable,
        )
    except Exception as exc:
        return (
            "创建定时任务失败："
            f"{type(exc).__name__}：{exc}"
        )

    storage = "持久" if job.durable else "会话"
    mode = "周期" if job.recurring else "一次性"
    return (
        f"已创建 {job.id}：'{job.cron}' -> {job.prompt} "
        f"[{mode}, {storage}]"
    )


@tool("list_crons")
def run_list_crons() -> str:
    """列出当前注册的全部 cron 定时任务。"""
    jobs = list_jobs()

    if not jobs:
        return "当前没有 cron 定时任务。"

    lines: list[str] = []

    for job in jobs:
        mode = "recurring" if job.recurring else "one-shot"
        storage = "durable" if job.durable else "session"
        lines.append(
            f"{job.id}: '{job.cron}' -> {job.prompt} "
            f"[{mode}, {storage}]"
        )

    return "\n".join(lines)


@tool("cancel_cron")
def run_cancel_cron(job_id: str) -> str:
    """根据任务 ID 取消 cron 定时任务。"""
    try:
        job = cancel_job(job_id)
    except Exception as exc:
        return (
            "取消定时任务失败："
            f"{type(exc).__name__}：{exc}"
        )

    if job is None:
        return f"找不到定时任务 {job_id}"

    return f"已取消 {job_id}"


# s13 的 8 个文件、命令和任务工具，加上本章的 3 个 Cron 工具。
TOOLS = [
    *base.TOOLS,
    run_schedule_cron,
    run_list_crons,
    run_cancel_cron,
]


@dynamic_prompt
def runtime_system_prompt(
    request: ModelRequest[Any],
) -> str:
    """Extend the s13 prompt with cron scheduling instructions."""
    # 复用 s13 的动态工作区、记忆和后台任务提示，再追加本章边界。
    prompt = base.get_system_prompt(
        base.build_prompt_context(request)
    )
    return (
        f"{prompt}\n\n"
        "You can manage scheduled work with schedule_cron, "
        "list_crons, and cancel_cron. Cron uses five fields: "
        "minute hour day month weekday. Times use the machine's "
        "local timezone. A durable job definition survives process "
        "restarts, but jobs run only while this agent process is alive."
    )


agent = create_agent(
    model=base.model,
    tools=TOOLS,
    middleware=[
        base.BackgroundNotificationMiddleware(),
        runtime_system_prompt,
    ],
    name="cron_scheduler",
)

session_state: dict[str, Any] = {"messages": []}


# ============================================================
# 6. Consumer：LangGraph 会话与到期消息注入
# ============================================================

def agent_loop() -> None:
    """Execute one LangChain agent turn and retain its final state."""
    # stream_mode="values" 每次返回完整状态，用消息键过滤已经打印的历史。
    seen = {
        base.message_key(message)
        for message in session_state.get("messages", [])
    }
    final_state: dict[str, Any] | None = None

    try:
        for state in agent.stream(
            session_state,
            stream_mode="values",
            config={"recursion_limit": 128},
        ):
            final_state = state

            for message in state.get("messages", []):
                key = base.message_key(message)

                if key in seen:
                    continue

                seen.add(key)
                base.print_message(message)

    # 即使模型或工具异常，也保留本轮最后一个可用状态。
    finally:
        if final_state is not None:
            session_state.clear()
            session_state.update(final_state)


def _scheduled_message(job: CronJob) -> HumanMessage:
    # prompt 来源可能包含 XML 特殊字符，转义后再放进结构化边界。
    return HumanMessage(
        content=(
            "<scheduled_task>\n"
            f"  <id>{html.escape(job.id)}</id>\n"
            f"  <cron>{html.escape(job.cron)}</cron>\n"
            f"  <prompt>{html.escape(job.prompt)}</prompt>\n"
            "</scheduled_task>"
        )
    )


def run_agent_turn_locked(
    user_query: str | None = None,
) -> None:
    """Run a user or scheduled turn while the caller holds agent_lock."""
    if user_query is not None:
        session_state["messages"].append(
            HumanMessage(content=user_query)
        )

    # 用户请求和到期任务可以在同一个回合中一起交给模型。
    fired_jobs = consume_cron_queue()

    for job in fired_jobs:
        session_state["messages"].append(
            _scheduled_message(job)
        )
        print(
            f"[inject cron] {job.id} "
            f"-> {job.prompt[:60]}"
        )

    if user_query is None and not fired_jobs:
        return

    agent_loop()


# ============================================================
# 7. Queue Processor：Agent 空闲时自动交付
# ============================================================

def queue_processor_loop() -> None:
    """Deliver queued jobs automatically whenever the agent is idle."""
    while not stop_event.wait(0.2):
        if not has_cron_queue():
            continue

        # 非阻塞取锁：用户回合正在运行时不等待，也不抢占它。
        if not agent_lock.acquire(blocking=False):
            continue

        try:
            if has_cron_queue():
                print(
                    "\n[queue processor] "
                    "正在交付定时任务"
                )
                run_agent_turn_locked()
                print()

        except Exception as exc:
            print(
                "[queue processor error] "
                f"{type(exc).__name__}：{exc}"
            )

        finally:
            agent_lock.release()


def start_services() -> None:
    """Start the scheduler and queue processor exactly once."""
    global services_started

    with service_lock:
        if services_started:
            return

        stop_event.clear()
        load_durable_jobs()

        # 两个线程都是 daemon；主进程退出时不会阻止 Python 结束。
        Thread(
            target=cron_scheduler_loop,
            name="cron-scheduler",
            daemon=True,
        ).start()
        Thread(
            target=queue_processor_loop,
            name="cron-queue-processor",
            daemon=True,
        ).start()
        services_started = True


# ============================================================
# 8. 命令行入口
# ============================================================

def main() -> None:
    start_services()
    print("s14: LangChain cron scheduler")
    print("输入问题后回车发送；输入 q 退出。\n")

    while True:
        try:
            query = input("s14 >> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.strip().lower() in {"", "q", "exit"}:
            break

        with agent_lock:
            try:
                run_agent_turn_locked(query)
            except Exception as exc:
                print(
                    "错误："
                    f"{type(exc).__name__}：{exc}"
                )

            print()

    # 通知 Scheduler 和 Queue Processor 退出；后台 shell 线程同样是 daemon。
    stop_event.set()
    running = base.count_running_background_tasks()

    if running:
        print(
            f"仍有 {running} 个后台任务；"
            "进程退出后 daemon 线程会停止。"
        )


if __name__ == "__main__":
    main()
