"""s07：Skills 按需加载机制。

这一章在 s06 的父/子 Agent、工具、权限 Hook 和 Todo 中间件之上，增加了 Skills：
1. 启动时只扫描每个 SKILL.md 的名称和简介，组成一个很小的技能目录；
2. 目录进入 system prompt，让模型知道“有哪些能力”，但不立刻占用完整上下文；
3. 真正命中某个技能时，模型调用 load_skill，再读取完整说明；
4. 技能正文中出现的相对路径，以对应技能目录而不是进程目录为基准解释。

这种“先发现、后加载”的设计叫渐进式披露。它既让 Agent 能发现扩展能力，又避免把所有
SKILL.md 一次性塞进提示词。文件后半部分仍保留 s06 的隔离子 Agent、权限检查和流式 CLI，
方便观察 Skills 如何嵌入一套完整的 Agent 运行时。"""

from __future__ import annotations

import glob
import os
import subprocess
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any


from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import(
    AgentState,
    TodoListMiddleware,
    after_agent,
    before_agent,
    wrap_tool_call
)

from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.runtime import Runtime
from langgraph.types import Command

import yaml

# ------------------------------------------------------------------
# 章节说明：1. 环境、工作区与技能注册表
# 程序启动时先读取 .env；所有文件和技能路径都锚定到当前工作区。
# ------------------------------------------------------------------


load_dotenv(override=True)

WORKDIR = Path.cwd().resolve()
SKILL_DIR = WORKDIR/"skills"

SKILL_REGISTRY: dict[str, dict[str, str]] = {}



# ------------------------------------------------------------------
# 章节说明：2. 技能发现：解析元数据并建立轻量目录
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 函数 _parse_frontmatter
# 解析 SKILL.md 顶部由 --- 包围的 YAML 元数据，并同时返回正文。
# 没有完整 frontmatter 或 YAML 格式错误时采用空字典，单个坏技能不会让扫描器崩溃。
# 正文与原始文本分开返回，目录阶段只取元数据，真正 load_skill 时仍能交付完整文件。
# ------------------------------------------------------------------
def _parse_frontmatter(raw: str) -> tuple[dict[str,Any], str]:
    """解析skill.md开头的yaml frontmatter"""
    # splitlines 不保留换行符，便于精确比较分隔行是否为 ---。
    lines = raw.splitlines()
    # 没有 frontmatter 时，整个文件都是正文；扫描器仍可用目录名作为技能名。
    if not lines or lines[0].strip() != "---":
        return {}, raw
    
    try:
        # 从第二行开始找闭合分隔符；start=1 使索引仍对应原始 lines。
        end_index = next(
            index
            for index, lines in enumerate(lines[1:], start=1)
            if lines.strip() == "---"
        )

    # 先找到第二个 ---；next 没找到会抛 StopIteration，随后安全退化为无元数据。
    except StopIteration:
        return {}, raw
    

    # frontmatter 不含两条 ---；body 则从闭合分隔符下一行开始。
    yaml_text = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index+1:])

    try: 
        # safe_load 不构造任意 Python 对象，适合读取仓库内声明式元数据。
        metadata = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError:
        metadata = {}

    # 合法 YAML 也可能是列表或字符串，但技能元数据必须是键值映射。
    if not isinstance(metadata, dict):
           metadata = {} 

    return metadata, body
    

# ------------------------------------------------------------------
# 函数 _scan_skills
# 扫描 skills/*/SKILL.md，把技能名称、简介、完整内容和根目录登记到内存注册表。
# 这里故意只做本地文件发现，不调用模型；重复名称直接报错，避免模型加载到含糊的技能。
# description 被压成单行以节省 system prompt，content 则原样保留，供按需加载使用。
# ------------------------------------------------------------------
def _scan_skills() -> None:
    """启动的时候自动扫描skills"""
    # 没有 skills 目录是允许状态，目录提示词稍后会显示 no skills found。
    if not SKILL_DIR.exists():
        return
    
    # sorted 让扫描和 catalog 顺序跨平台稳定，便于缓存与测试。
    for manifest in sorted(SKILL_DIR.glob("*/SKILL.md")):
        raw = manifest.read_text(
            encoding="utf-8",
            errors="replace",
        )


        metadata, body = _parse_frontmatter(raw)


        # 显式 name 优先；省略时用技能目录名作为稳定兜底。
        name = str(
            metadata.get("name")
            or manifest.parent.name
        )

# 简介优先取 YAML description；没有时用正文中第一个 Markdown 标题兜底。

        description = metadata.get("description")

        if not description:
            # 只取第一条 Markdown 标题，不把整段正文提前泄露进 catalog。
            description = next(
                (
                    line.lstrip("#").strip()
                    for line in body.splitlines()
                    if line.lstrip().startswith("#")
                ),
                name,
            )
            # YAML 的 description 可能使用 | 写成多行，
            # catalog 中应压缩为一行，避免浪费上下文。
        description = " ".join(
            str(description).split()
        )

        # relative_to(WORKDIR) 让返回给模型的 root 简短、可移植，不暴露多余绝对路径。
        # 重名若静默覆盖，会导致 catalog 描述与真正加载内容不确定。
        if name in SKILL_REGISTRY:
            raise ValueError(
                f"Duplicate skill name: {name}"
            )
        
        # 记录相对根目录，让技能正文引用 scripts/foo.py 时有明确解析基准。
        skill_root = manifest.parent.relative_to(
            WORKDIR
        ).as_posix()


        SKILL_REGISTRY[name] = {
            "name": name,
            "description": description,
            "content": raw,
            "root": skill_root,
        }

# ------------------------------------------------------------------
# 函数 list_skills
# 把注册表压缩成“名称 + 一行简介”的目录文本，供 system prompt 做能力发现。

# ------------------------------------------------------------------
def list_skills() ->str:
    """生成注入system_prompt的轻量技能目录"""
    # 空目录也返回明确文本，避免 system prompt 出现一个无内容标题。
    if not SKILL_REGISTRY:
        return "- (no skills found)"
    
    return "\n".join(
        f"- {skill['name']}: {skill['description']}"
        for skill in SKILL_REGISTRY.values()
    )



# ------------------------------------------------------------------
# 章节说明：扫描必须发生在构造 system prompt 之前，否则注入的目录会一直为空。
# ------------------------------------------------------------------
# 扫描和 catalog 构建只在模块加载时执行一次；热新增技能需要重启或显式重扫。
_scan_skills()

# catalog 只含 name/description，完整 content 仍留在注册表中。
SKILL_CATALOG = list_skills()

# ------------------------------------------------------------------
# 章节说明：3. 模型配置与技能目录提示词
# 这里只把目录注入提示词，完整技能正文仍留在 SKILL_REGISTRY 中等待 load_skill。
# ------------------------------------------------------------------
MODEL_ID = os.getenv("MODEL_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("BASE_URL")


# f-string 在模块导入时固化目录；运行期间新增技能需要重新扫描并重建提示词。
SKILL_SYSTEM_PROMPT = f"""
Available skills:

{SKILL_CATALOG}

Skills contain specialized instructions that should be loaded only when
relevant.

When a request clearly matches a skill description:

1. Call load_skill using the exact skill name.
2. Read and follow the returned instructions before doing the task.
3. Do not guess a skill's full instructions from its description.
4. Load only skills relevant to the current request.
"""

# ------------------------------------------------------------------
# 章节说明：4. 生命周期 Hook：把日志和权限策略接到 LangChain
# ------------------------------------------------------------------
#用来区分hook日志来自子agent还是父agent
AGENT_SCOPE: ContextVar[str] = ContextVar(
    "agent_scope",
    default="parent"
)


HOOKS: dict[str, list[Callable[..., Any]]] = {
    "UserPromptSubmit": [],
    "PreToolUse" : [],
    "PostToolUse": [],
    "Stop": [],
}

# ------------------------------------------------------------------
# 函数 register_hook
# 向指定生命周期事件注册回调。先校验事件名，可尽早发现拼写错误。
# 同一事件允许多个回调，保存顺序就是稍后触发顺序。
# ------------------------------------------------------------------
def register_hook(event: str,callback: Callable[..., Any]) -> None:
    """注册一个hook"""

    # 拼错事件名应立即失败，不能创建一个永远不会触发的隐形事件。
    if event not in HOOKS:
        raise ValueError(f"unknown hook event:{event}")
    
    # 列表追加保留声明顺序，多个权限或审计 Hook 的顺序因此可预测。
    HOOKS[event].append(callback)


# ------------------------------------------------------------------
# 函数 trigger_hook
# 依注册顺序执行回调，并返回第一个非 None 结果。
# PreToolUse 把非 None 当作拦截原因；纯日志 Hook 通常返回 None，让后续回调继续运行。
# ------------------------------------------------------------------
def trigger_hook(event:str, *args: Any)-> Any|None:
    """按照顺序执行hook"""
    # 找不到回调时自然返回 None，等价于“不拦截”。
    for callable in HOOKS.get(event,[]):
        result = callable(*args)

        if result is not None:
            return result
    return None

# ------------------------------------------------------------------
# 函数 user_prompt_submit
# 把 LangChain 的 before_agent 生命周期适配成自定义 UserPromptSubmit 事件。
# 消息可能是普通 dict，也可能是 LangChain 消息对象，因此分别读取 content。
# ------------------------------------------------------------------
@before_agent
def user_prompt_submit(state:AgentState, runtime:Runtime) -> dict[str, Any] |None:
    """父agent运行的时候触发"""

    # before_agent 得到本轮输入前已经合并好的完整消息状态。
    messages = state.get("messages", [])

    if not messages:
        return None
    
    # CLI 刚追加的用户输入位于末尾；历史消息不重复触发提交事件。
    last_messages = messages[-1]

    if isinstance(last_messages, dict):
        content = last_messages.get("content")

    else:
        content = getattr(last_messages,"content",None)
    
    trigger_hook("UserPromptSubmit", content)

    return None


# ------------------------------------------------------------------
# 函数 tool_hook
# 包裹每次工具调用：先运行 PreToolUse，获准后调用真实 handler，最后运行 PostToolUse。
# 被拒绝时返回带原 tool_call_id 的错误 ToolMessage，保证模型侧工具调用协议仍然闭合。
# ------------------------------------------------------------------
@wrap_tool_call
def tool_hook(
    request: ToolCallRequest,
    handler: Callable[
        [ToolCallRequest],
        ToolMessage | Command,
    ],
) -> ToolMessage | Command:
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})
    tool_call_id = request.tool_call["id"]

    # 真实工具尚未执行，此处是权限系统唯一可靠的前置拦截点。
    blocked_reason = trigger_hook(
        "PreToolUse",
        tool_name,
        tool_args,
    )

    # 即使拒绝，也必须使用相同 id 回一条 ToolMessage，否则消息序列会不合法。
    if blocked_reason:
        return ToolMessage(
            content=str(blocked_reason),
            tool_call_id = tool_call_id,
            name = tool_name,
            status = "error",
        )
    

    # 只有通过 PreToolUse 才进入真实 handler。PostToolUse 只观察已发生的结果。
    result = handler(request)

    # after_agent 只做观察和日志，因此最终返回 None，不更新 state。
    trigger_hook(
        "PostToolUse",
        tool_name,
        tool_args,
        result,
    )

    return result


# ------------------------------------------------------------------
# 函数 stop_hook
# 在一次 Agent invoke/stream 完成后触发 Stop 回调，用于统计或清理。
# 返回 None 表示不修改 LangGraph state。
# ------------------------------------------------------------------
@after_agent
def stop_hook(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    """父agent结束时触发"""
    trigger_hook(
        "Stop",
        state.get("messages", [])
    )

    return None

# ------------------------------------------------------------------
# 章节说明：5. 两级权限模型
# DANGEROUS_COMMANDS 直接拒绝；POTENTIALLY_DESTRUCTIVE_COMMANDS 需要人工确认。
# ------------------------------------------------------------------
DANGEROUS_COMMANDS = [
    "rm -rf /",
    "sudo",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=",
    "> /dev/sda",
]

POTENTIALLY_DESTRUCTIVE_COMMANDS = [
    "rm ",
    "> /etc/",
    "chmod 777",
]

# ------------------------------------------------------------------
# 函数 resolve_path
# 把工具参数中的相对路径锚定到 WORKDIR，并解析为规范化绝对路径。
# 它只负责标准化；是否越出工作区由权限规则在执行工具前判断。
# ------------------------------------------------------------------
def resolve_path(raw_path:str)-> Path:
    """相对路径转换为绝对"""

    candidate = Path(raw_path)

    # 保留绝对路径的真实含义，让权限层能准确提示“工作区外访问”。
    if candidate.is_absolute():
        return candidate.resolve()
    
    return (WORKDIR/candidate).resolve()


# ------------------------------------------------------------------
# 函数 check_deny_list
# 对 shell 命令做不可绕过的高危模式检查。命中后直接返回原因，不询问用户。
# 这是教学版字符串匹配，不等价于完整 shell 解析器，生产环境需要更严格的隔离。
# ------------------------------------------------------------------
def check_deny_list(command:str) ->str | None:
    # 统一大小写后再匹配，防止简单大小写变化绕过教学规则。
    normalized = command.lower()
    for pattern in DANGEROUS_COMMANDS:
        if pattern.lower() in normalized:
            return f"Blocked:{pattern} is in the deny list"
        
    
    return None

# ------------------------------------------------------------------
# 函数 ask_user
# 在终端显示调用来源、工具名、参数和风险原因，并要求用户明确输入 y/yes。
# 默认答案为拒绝；ContextVar 让提示能区分父 Agent 与子 Agent。
# ------------------------------------------------------------------
def ask_user(tool_name:str, args:dict[str, Any],reason: str) ->bool:

    # scope 只改变提示标签，批准动作仍由当前终端用户完成。
    scope = AGENT_SCOPE.get()

    print(f"\nWarning: [{scope}] Permission required")
    print(f"Reason: {reason}")
    print(f"Tool: {tool_name}")
    print(f"Arguments: {args}")

    # 提示中的大写 N 表示默认拒绝；空输入不会被视为同意。
    choice = input("Allow? [y/N] ").strip().lower()

    return choice in {"y", "yes"}



# ------------------------------------------------------------------
# 函数 check_rules
# 检查需要人工确认的软规则：潜在破坏性命令，以及读写编辑越出工作区。
# 返回字符串代表需要确认，返回 None 代表没有命中软规则；它本身不执行任何操作。
# ------------------------------------------------------------------
def  check_rules(
        tool_name: str,
        args: dict[str, Any]
) -> str| None:
    """检查用户的操作"""
    

    # shell 命令先做软规则匹配；真正的硬拒绝列表在 check_permission 中更早判断。
    if tool_name == "run_bash":
        command = str(args.get("command", ""))
        normalized = command.lower()

        if normalized.strip().startswith("del "):
            return "Potentially destructive shell command: del "
        for pattern in POTENTIALLY_DESTRUCTIVE_COMMANDS:
            if pattern.lower() in normalized:
                return(
                    "Potentially destructive shell command: "
                    f"{pattern}"
                )
            

    # 读、写、编辑共享 path 参数，统一拒绝 resolve 后越出 WORKDIR 的目标。
    if tool_name in {"run_read","run_write","run_edit"}:
        raw_path = str(args.get("path", ""))

        try:
            # 必须先 resolve 再比较，../ 和符号链接才不会欺骗边界判断。
            target = resolve_path(raw_path)
        except(OSError, RuntimeError, ValueError) as exc:
            return f"Invalid path:{exc}"
        
        if not target.is_relative_to(WORKDIR):
            return f"Operation accesses outside workspace: {target}"
        
    return None


# ------------------------------------------------------------------
# 函数 check_permission
# 统一编排权限决策：先执行绝对拒绝列表，再执行需要确认的规则。
# 最终只返回布尔值，供 PreToolUse Hook 转换为允许或 Permission denied。
# ------------------------------------------------------------------
def check_permission(
        tool_name: str,
        args: dict[str, Any]
)-> bool:
    "执行deny list 和权限规则检查"

    if tool_name == "run_bash":
        command = str(args.get("command", ""))

        # 硬拒绝不弹确认框，避免用户误按 y 放行极高风险命令。
        denied_reason = check_deny_list(command)

        if denied_reason:
            print(f"\nBlocked: {denied_reason}")
            return False
        
    # 硬拒绝未命中后再运行软规则，软规则允许用户基于完整参数做决定。
    confirmation_reason = check_rules(
        tool_name,
        args,
    )

    if confirmation_reason:
        return ask_user(
            tool_name,
            args,
            confirmation_reason,
        )
    
    return True

# ------------------------------------------------------------------
# 函数 on_user_prompt_submit
# UserPromptSubmit 的示例回调，仅记录本轮输入，不改变 Agent 状态。
# ------------------------------------------------------------------
def on_user_prompt_submit(content: Any) -> None:
    print(f"[UserPromptSubmit] {content}")

# ------------------------------------------------------------------
# 函数 on_pre_tool_use
# 记录工具调用并执行权限检查；返回拒绝文本即可阻止 handler 真正运行。
# ------------------------------------------------------------------
def on_pre_tool_use(
        tool_name: str,
        tool_args: dict[str, Any],
) -> str | None:
    scope = AGENT_SCOPE.get()

    print(f"[{scope} PreToolUse] {tool_name}")
    print(f"Arguments: {tool_args}")

    if not check_permission(tool_name, tool_args):
        return "Permission denied"
    
    return None

# ------------------------------------------------------------------
# 函数 on_post_tool_use
# 记录工具结果的短预览。限制为 500 字符，避免日志被大文件内容淹没。
# ------------------------------------------------------------------
def on_post_tool_use(
        tool_name: str,
        tool_args: dict[str, Any],
        result: ToolMessage | Command,
) ->None:
    scope = AGENT_SCOPE.get()

    print(f"[{scope} PostToolUse] {tool_name}")

    # ToolMessage 有 content；Command 等特殊结果使用对象自身作为可观测文本。
    content = getattr(result, "content", result)
    preview = str(content)

    if len(preview) >500:
        preview = preview[:500] + "...(truncated)"

    print(f"Result: {preview}")


# ------------------------------------------------------------------
# 函数 on_stop
# 遍历最终消息，统计 AIMessage 中的工具调用数量，输出一次回合级摘要。
# ------------------------------------------------------------------
def on_stop(messages: list[Any]) ->None:
    tool_call_count = 0
    # 一个 AIMessage 可包含多个 tool_calls，所以累计调用数组长度而非消息数。
    for message in messages:
        if isinstance(message, AIMessage):
            tool_call_count += len(message.tool_calls or [])

    print(
        f"[Stop] messages={len(messages)}, "
        f"tool_calls={tool_call_count}"
    )
        

# ------------------------------------------------------------------
# 章节说明：把具体回调装入通用 Hook 分发器。注册顺序会影响执行顺序。
# ------------------------------------------------------------------
register_hook(
    "UserPromptSubmit",
    on_user_prompt_submit,
)

register_hook(
    "PreToolUse",
    on_pre_tool_use,
)

register_hook(
    "PostToolUse",
    on_post_tool_use,
)

register_hook(
    "Stop",
    on_stop,
)



# ------------------------------------------------------------------
# 章节说明：6. 给父/子 Agent 共用的基础工具
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 函数 load_skill
# 按精确名称取出完整技能说明，这是渐进式披露的第二阶段。
# 返回值同时声明 skill root，提醒模型据此解析 SKILL.md 内的相对引用。
# 该函数由 @tool 暴露；它的 docstring 会成为模型可见的工具描述，因此不能随意删除。
# ------------------------------------------------------------------
@tool
def load_skill(name: str) -> str:
    """Load the full instructions for a skill.

    Args:
        name: Exact skill name shown in the available-skills catalog.
    """

    # 要求精确名称，避免相似匹配把错误技能说明注入高优先级上下文。
    skill = SKILL_REGISTRY.get(name)

    if skill is None:
        # 失败时返回可恢复信息，模型可以从 available 列表修正名称后重试。
        available = ", ".join(SKILL_REGISTRY)
        return (
            f"Skill not found: {name}. "
            f"Available skills: {available or '(none)'}"
        )

    # root 放在正文之前，使模型读取后续相对路径时已经知道解析规则。
    return (
        f"Loaded skill: {skill['name']}\n"
        f"Skill root: {skill['root']}\n"
        "Resolve relative paths mentioned by this skill "
        "against the skill root above.\n\n"
        f"{skill['content']}"
    )

# ------------------------------------------------------------------
# 函数 run_bash
# 在 WORKDIR 中运行 shell 命令，合并 stdout/stderr，并把超时和系统错误转成文本。
# shell=True 能演示完整命令，但风险较高，所以调用前必须经过 tool_hook 权限层。
# 输出截到 50000 字符，避免一次命令无限膨胀上下文。
# ------------------------------------------------------------------
@tool
# 安全边界：shell=True 仅为教学演示，黑名单/路径检查不等于安全边界；生产请使用权限中间件 + 沙箱。
def run_bash(command: str) -> str:
    """Execute a shell command in the current workspace."""

    # subprocess.run 是同步调用；120 秒超时防止 CLI 永久卡在一个命令上。
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
        )

        # 合并两个流，让非零命令的诊断不会丢失在 stderr。
        output = (
            result.stdout
            + result.stderr
        ).strip()

        if not output:
            output = "(no output)"

        # 返回码单独置顶，模型无需从错误文本猜测命令是否成功。
        if result.returncode != 0:
            output = (
                f"Exit code: {result.returncode}\n"
                f"{output}"
            )

        # 即使命令输出更长也只交回前 50000 字符；s08 会进一步处理大工具结果。
        return output[:50000]

    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120 seconds"

    except OSError as exc:
        return f"Error: {exc}"


# ------------------------------------------------------------------
# 函数 run_read
# 以 UTF-8 读取文本，可用 limit 限制返回行数；剩余行数会通过占位文本告知模型。
# 路径标准化与工作区权限检查分层完成，文件错误统一作为字符串返回给 Agent。
# ------------------------------------------------------------------
@tool
def run_read(
    path: str,
    limit: int | None = None,
) -> str:
    """Read a UTF-8 text file.

    Args:
        path: File path, normally relative to the workspace.
        limit: Optional maximum number of lines to return.
    """

    try:
        # resolve_path 只规范路径；工作区外访问已由调用前的权限 Hook 决定。
        file_path = resolve_path(path)

        # errors="replace" 让少量非法字节变成替换字符，不让整个回合因解码中断。
        lines = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()

        # limit 只控制交给模型的行数，磁盘文件保持不变。
        if limit is not None and limit >= 0 and limit < len(lines):
            remaining = len(lines) - limit

            lines = [
                *lines[:limit],
                f"...({remaining} more lines)",
            ]

        return "\n".join(lines)

    except (OSError, ValueError) as exc:
        return f"Error: {exc}"


# ------------------------------------------------------------------
# 函数 run_write
# 创建缺失的父目录并以 UTF-8 覆盖写入完整内容。
# 这是有副作用的工具；是否允许写入由 PreToolUse 在函数执行前决定。
# ------------------------------------------------------------------
@tool
def run_write(
    path: str,
    content: str,
) -> str:
    """Write UTF-8 content to a file, replacing existing content.

    Args:
        path: Target file path.
        content: Complete new file content.
    """

    try:
        file_path = resolve_path(path)

        # 一次性创建所有父目录，允许模型写入尚不存在的嵌套路径。
        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_path.write_text(
            content,
            encoding="utf-8",
        )

        return f"Wrote {len(content)} characters to {path}"

    except (OSError, ValueError) as exc:
        return f"Error: {exc}"


# ------------------------------------------------------------------
# 函数 run_edit
# 执行一次精确文本替换，只替换第一个匹配，便于模型做可预测的小范围编辑。
# 找不到 old_text 时不写文件，而是返回明确错误，避免静默产生错误修改。
# ------------------------------------------------------------------
@tool
def run_edit(
    path: str,
    old_text: str,
    new_text: str,
) -> str:
    """Replace the first exact occurrence of text in a UTF-8 file.

    Args:
        path: Target file path.
        old_text: Exact text that should be replaced.
        new_text: Replacement text.
    """

    try:
        file_path = resolve_path(path)

        # 编辑采用“读取 -> 精确查找 -> 只替换第一处 -> 整体写回”的教学流程。
        current_content = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        # 精确匹配失败时停止，不尝试模糊替换，因为猜错位置比显式失败更危险。
        if old_text not in current_content:
            return f"Error: old_text was not found in {path}"

        # count=1 防止一个常见片段在文件中所有位置同时被替换。
        updated_content = current_content.replace(
            old_text,
            new_text,
            1,
        )

        file_path.write_text(
            updated_content,
            encoding="utf-8",
        )

        return f"Edited {path}"

    except (OSError, ValueError) as exc:
        return f"Error: {exc}"


# ------------------------------------------------------------------
# 函数 run_glob
# 在工作区内递归匹配 glob，并只保留 resolve 后仍位于 WORKDIR 的路径。
# 结果排序保证相同文件集得到稳定输出，有利于模型推理和测试复现。
# ------------------------------------------------------------------
@tool
def run_glob(pattern: str) -> str:
    """Find workspace files matching a glob pattern.

    Args:
        pattern: Pattern such as "*.py" or "src/**/*.py".
    """

    try:
        results: list[str] = []

        # root_dir 让返回值天然是工作区相对路径，便于直接交给 read/edit。
        for match in glob.glob(
            pattern,
            root_dir=WORKDIR,
            recursive=True,
        ):
            full_path = (WORKDIR / match).resolve()

            # resolve 后再次检查，过滤可能通过符号链接逃逸工作区的匹配项。
            if full_path.is_relative_to(WORKDIR):
                results.append(match)

        if not results:
            return "(no matches)"

        return "\n".join(sorted(results))

    except (OSError, ValueError) as exc:
        return f"Error: {exc}"

# ------------------------------------------------------------------
# 章节说明：BASE_TOOLS 刻意不含 task，子 Agent 因此不能再递归创建孙 Agent。
# ------------------------------------------------------------------
#这个列表不包含task防止循环调用
BASE_TOOLS = [
    run_bash,
    run_read,
    run_write,
    run_edit,
    run_glob,
    load_skill,
]


# ------------------------------------------------------------------
# 章节说明：7. 模型与隔离子 Agent
# temperature=0 让教学演示更稳定；max_completion_tokens 限制单次回复上限。
# ------------------------------------------------------------------
MODEL = ChatOpenAI(
    model=MODEL_ID,
    max_completion_tokens=8000,
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

SUB_SYSTEM = f"""
You are an isolated coding subagent working in:

{WORKDIR}

{SKILL_SYSTEM_PROMPT}

Complete the exact task given by the parent agent.

Rules:

1. Work independently and use the available tools when necessary.
2. You do not have access to the parent's conversation history.
3. Do not assume information that was not included in the task description.
4. Do not delegate or attempt to create another agent.
5. When finished, return a concise but complete summary.
6. Include relevant file paths, findings, changes, verification results,
   and unresolved problems in the final summary.
"""


# 子 Agent 继续使用权限 Hook。
# 不传入 TodoListMiddleware，因为 s06 的子 Agent 只有基础工具。
# SUB_MIDDLEWARE = [
#     tool_hook,
# ]

# 子 Agent 只有 BASE_TOOLS；middleware=[tool_hook] 让它同样受权限系统约束。
SUB_AGENT = create_agent(
    model=MODEL,
    tools=BASE_TOOLS,
    system_prompt=SUB_SYSTEM,
    middleware=[tool_hook],
    name="worker",
)

# ------------------------------------------------------------------
# 函数 extract_final_text
# 从后向前寻找最后一条有文本的 AIMessage，兼容纯字符串和多内容块两种响应格式。
# 父 Agent 只需要子 Agent 的最终结论，因此不会把子 Agent 的完整消息历史合并回来。
# ------------------------------------------------------------------
def extract_final_text(messages: list[Any]) -> str:
    """读取最后一条aimessage的中文文本内容"""

    # 倒序是因为历史里可能有多条规划/工具调用 AIMessage，最后一条才是结论。
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue


        content = message.content

        if isinstance(content, str):
            if content.strip():
                return content.strip()
            

            continue

        # 非字符串、非列表内容通常不是可展示文本，继续寻找更早的 AIMessage。
        if not isinstance(content, list):
            continue

        texts: list[str] = []

        # 逐块兼容 str、dict 和 provider 对象三种表示。
        for block in content:
            text:str | None =None

            if isinstance(block, str):
                text = block

            elif isinstance(block,dict):
                possible_text = block.get("text")

                if isinstance(possible_text, str):
                    text = possible_text

            else:
                possible_text = getattr(
                    block,
                    "text",
                    None
                )

                if isinstance(possible_text, str):
                    text = possible_text

            if text and text.strip():
                texts.append(text.strip())

        if texts:
            return "\n".join(texts)
        
    return ""

# ------------------------------------------------------------------
# 函数 task
# 创建一次隔离的子 Agent 调用：只传 description，不传父对话，从而控制上下文污染。
# 调用期间用 ContextVar 标记 sub；finally 无论成功或异常都恢复旧值，避免作用域泄漏。
# 递归上限、一般异常和无文本结论都被转换为父 Agent 能继续处理的工具返回值。
# ------------------------------------------------------------------
@tool("task")
def task(description: str) -> str:
    """Launch an isolated subagent for a complex subtask.

    The subagent receives only this description, not the parent conversation.
    Include the complete objective, paths, constraints and expected output.
    Only the final textual conclusion is returned.
    """

    print("\n\033[35m[Subagent spawned]\033[0m")
    print(f"Task: {description}")

    # set 返回 token，reset(token) 能恢复进入 task 前的精确 ContextVar 状态。
    scope_token = AGENT_SCOPE.set("sub")

    try:

        # 子 Agent 的 messages 只有一条由 description 构造的用户消息，这就是上下文隔离边界。
        result = SUB_AGENT.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": description,
                    }
                ]
            },
            config={"recursion_limit": 128},
        )

        # 只提取最终结论返回父 Agent；探索过程不会挤占父 Agent 的上下文窗口。
        summary = extract_final_text(
            result.get("messages", [])
        )

        return (
            summary
            or "Subagent finished without a textual conclusion."
        )

    except GraphRecursionError:
        return (
            "Subagent stopped because it reached "
            "the execution limit."
        )

    except Exception as exc:
        return (
            "Subagent failed: "
            f"{type(exc).__name__}: {exc}"
        )

    finally:
        # finally 在 return 和异常两条路径都会执行，防止后续父工具日志误标为 sub。
        AGENT_SCOPE.reset(scope_token)
        print("\033[35m[Subagent done]\033[0m")

# ------------------------------------------------------------------
# 章节说明：8. 父 Agent：负责规划，把复杂文件任务委托给 task
# ------------------------------------------------------------------
PARENT_SYSTEM = f"""
You are a coding agent working in:

{WORKDIR}
{SKILL_SYSTEM_PROMPT}

You must use write_todos for every non-trivial request.

Before using run_bash, run_read, run_write, run_edit, run_glob, or task,
create or update the todo list.

Use task when a subproblem is:

- complex and self-contained
- likely to require reading many files
- likely to require several tool calls
- useful to solve in an isolated context

The task subagent cannot see this conversation. Every task description must
therefore contain:

- the precise objective
- relevant files and directories
- necessary background information
- constraints
- expected output
- whether files may be modified

The task tool returns only the subagent's final conclusion. Its intermediate
messages are deliberately discarded.

After receiving a task result, evaluate it, verify it when necessary, and
continue working on the parent request.

For this demonstration, you MUST use task for every request that requires
reading, writing, editing, or executing files. The parent agent must not
perform those operations directly.
"""


# 父 Agent 额外拥有 task；子 Agent 使用的 BASE_TOOLS 不含它。
PARENT_TOOLS = [
    *BASE_TOOLS,
    task,
]

# ------------------------------------------------------------------
# 章节说明：Middleware 顺序是用户输入 Hook → Todo → 工具权限 Hook → Stop；顺序本身就是行为。
# ------------------------------------------------------------------
PARENT_MIDDLEWARE = [
    user_prompt_submit,

    # TodoListMiddleware 会额外提供 write_todos 工具，并把 todos 字段写进 Agent state。
    TodoListMiddleware(
        system_prompt="""
        You must call write_todos before calling run_bash, run_read, run_write,
        run_edit, run_glob, or task.

        For every non-empty, non-trivial request:

        1. Create at least one todo item before using another tool.
        2. Keep exactly one relevant item in_progress while working.
        3. Update todo statuses as work progresses.
        4. Mark items completed only after the work is actually complete.
        """,
        tool_description="""
        Create or update the current task list. This is a mandatory planning tool.
        Call it before using run_bash, run_read, run_write, run_edit, run_glob,
        or task.
        """,
    ),

    tool_hook,
    stop_hook,
]

# ------------------------------------------------------------------
# 章节说明：create_agent 把模型、工具、系统提示和中间件组装成可 stream/invoke 的 LangGraph。
# ------------------------------------------------------------------
agent = create_agent(
    model= MODEL,
    tools=PARENT_TOOLS,
    system_prompt=PARENT_SYSTEM,
    middleware=PARENT_MIDDLEWARE,
    name="parent",
)


# ------------------------------------------------------------------
# 章节说明：9. 流式终端展示
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 函数 content_to_text
# 把模型内容统一转换成终端文本，兼容字符串、字典内容块和带 text 属性的对象。
# 未知非列表类型使用 str 兜底，使日志展示不会因供应商响应格式差异而中断。
# ------------------------------------------------------------------
def content_to_text(content: Any) -> str:
    """把 LangChain 消息内容转换成适合终端输出的文本。"""

    # OpenAI-compatible chat completion 的常见返回是简单字符串。
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return str(content)

    texts: list[str] = []

    # content block 列表常见于多模态或不同 provider 的统一消息格式。
    for block in content:
        if isinstance(block, str):
            texts.append(block)
            continue

        if isinstance(block, dict):
            text = block.get("text")

            if isinstance(text, str):
                texts.append(text)

            continue

        text = getattr(block, "text", None)

        if isinstance(text, str):
            texts.append(text)

    return "\n".join(texts)


# ------------------------------------------------------------------
# 函数 print_message
# 按消息类型打印新增事件：AIMessage 展示工具调用和回复，ToolMessage 展示状态与结果。
# 只负责界面输出，不修改 state；因此可与 stream_mode='values' 的增量游标配合。
# ------------------------------------------------------------------
def print_message(message: Any) -> None:
    """打印一条新产生的 Agent 消息。"""

    if isinstance(message, AIMessage):
        # AIMessage 可同时带文本和工具调用，不能在打印 tool_calls 后直接 return。
        if message.tool_calls:
            print("\n模型调用工具：")

            for tool_call in message.tool_calls:
                print(f"工具名：{tool_call['name']}")
                print(
                    "参数："
                    f"{tool_call.get('args', {})}"
                )

        text = content_to_text(message.content)

        if text.strip():
            print("\n模型回复：")
            print(text)

        return

    if isinstance(message, ToolMessage):
        print("\n工具返回结果：")
        print(f"工具名：{message.name}")
        print(f"状态：{getattr(message, 'status', 'success')}")
        print(f"内容：{message.content}")
        return

    content = getattr(message, "content", message)

    print("\n消息：")
    print(content_to_text(content))


# ------------------------------------------------------------------
# 函数 print_todos
# 把 TodoListMiddleware 维护的结构化列表格式化为带状态的编号列表。
# ------------------------------------------------------------------
def print_todos(todos: list[dict[str, Any]]) -> None:
    """打印 Todo 状态。"""

    print("\n当前 Todo：")

    for index, todo_item in enumerate(
        todos,
        start=1,
    ):
        status = todo_item.get(
            "status",
            "pending",
        )

        content = todo_item.get(
            "content",
            "",
        )

        print(
            f"{index}. [{status}] {content}"
        )


# ============================================================
# 父 Agent 主循环
# ============================================================

# ------------------------------------------------------------------
# 函数 agent_loop
# 消费 Agent 的状态流，只打印上次游标之后的新消息以及发生变化的 Todo。
# 流结束后用 final_state 覆盖 session_state，使 messages、todos 和扩展状态跨用户轮次保留。
# ------------------------------------------------------------------
def agent_loop(
    session_state: dict[str, Any],
) -> None:
    """运行父 Agent，并把最终状态保存到 session_state。"""

    existing_messages = session_state.get(
        "messages",
        [],
    )

    # 用消息数量作为显示游标，避免 stream 每次给出完整 state 时重复打印旧消息。
    seen_message_count = len(existing_messages)
    last_todos = session_state.get("todos")
    final_state: dict[str, Any] | None = None

    # values 模式输出每一步的完整 state；updates 模式则只输出节点增量。
    for state in agent.stream(
        session_state,
        stream_mode="values",
    ):
        final_state = state

        todos = state.get("todos")

        # 结构化比较避免相同 Todo 在模型/工具每个节点后重复打印。
        if todos is not None and todos != last_todos:
            print_todos(todos)
            last_todos = todos

        current_messages = state.get(
            "messages",
            [],
        )

        new_messages = current_messages[
            seen_message_count:
        ]

        for message in new_messages:
            print_message(message)

        seen_message_count = len(
            current_messages
        )

    # 必须保存整个 final_state，而非仅 messages；否则 Todo 和后续扩展字段会丢失。
    if final_state is not None:
        # 不只保存 messages，也保存 TodoListMiddleware 的 todos 状态。
        session_state.clear()
        session_state.update(final_state)


# ============================================================
# CLI
# ============================================================

# ------------------------------------------------------------------
# 章节说明：10. 命令行入口
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 函数 main
# 提供交互式 CLI：收集用户输入、追加 Human 消息、运行 Agent，并处理退出与异常。
# session_state 在 while 循环外创建，所以同一进程中的多轮对话共享完整状态。
# ------------------------------------------------------------------
def main() -> None:
    print("s07: Skill Loading — catalog in SYSTEM, content on demand")
    print("输入问题，回车发送。输入 q 退出。\n")

    # 保存完整状态，而不只是 messages，
    # 这样 todos 也能跨用户轮次保留。
    session_state: dict[str, Any] = {
        "messages": [],
    }

    while True:
        try:
            query = input(
                "\033[36ms07 >> \033[0m"
            )

        except (
            EOFError,
            KeyboardInterrupt,
        ):
            print()
            break

        if query.strip().lower() in {
            "",
            "q",
            "exit",
        }:
            break

        # 普通 role/content 字典会在图入口统一转换为 HumanMessage。
        session_state.setdefault(
            "messages",
            [],
        ).append(
            {
                "role": "user",
                "content": query,
            }
        )

        try:
            agent_loop(session_state)

        # 该异常说明模型与工具循环没有在上限内收敛，单独给出明确诊断。
        except GraphRecursionError:
            print(
                "\nAgent stopped because it reached "
                "the execution limit."
            )

        except Exception as exc:
            print(
                "\nAgent error: "
                f"{type(exc).__name__}: {exc}"
            )

        print()


if __name__ == "__main__":
    main()
