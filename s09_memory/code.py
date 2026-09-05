"""s09：基于 Markdown 文件的跨会话长期记忆。

s08 的 compact 解决的是“当前会话太长”，而本章解决“程序退出后如何记住稳定信息”。
长期记忆被保存为带 YAML frontmatter 的 Markdown 文件，并通过 MEMORY.md 提供轻量索引。
每个用户回合依次经历四个阶段：
1. 检索：根据最近用户消息，从索引中挑出最多几条相关记忆；
2. 注入：只在发给模型的临时请求中加入索引和相关正文，不污染真实消息历史；
3. 提取：回合结束后，从对话中提取值得跨会话保存的偏好、反馈和项目事实；
4. 合并：记忆过多时去重、消除过时内容，再重建索引。

本文件复用 s08 的模型、工具、压缩和终端输出，只实现记忆存储与中间件。理解重点是：
短期消息状态、压缩摘要和磁盘长期记忆是三个生命周期不同的数据层。"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from  typing import Any , NotRequired

import yaml

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
)
from langchain_core.messages import(
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langgraph.errors import GraphRecursionError

# ------------------------------------------------------------------
# 章节说明：1. 复用 s08 的 Agent 基础设施
# 导入 s08 会得到同一个 WORKDIR、MODEL、工具、父提示词和压缩中间件。
# ------------------------------------------------------------------
from langgraph.runtime import Runtime

from s08_context_compact import code as s08

# s09 不另建模型，保证记忆分类器与主 Agent 使用相同 API 配置。
WORKDIR = s08.WORKDIR
MODEL = s08.MODEL

# ------------------------------------------------------------------
# 章节说明：2. 长期记忆目录与策略常量
# 四种类型便于提示模型区分稳定偏好、工作反馈、项目事实和外部参考。
# ------------------------------------------------------------------
MEMORY_DIR = WORKDIR / ".memory"
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"
MEMORY_TYPES ={
    "user",
    "feedback",
    "project",
    "reference",
}

# 最多注入五条相关正文，防止长期记忆本身反过来挤爆上下文。
MAX_RELEVANT_MEMORIES = 5
CONSOIDATE_THRESHOLD = 10


# ------------------------------------------------------------------
# 章节说明：3. 消息与模型 JSON 输出的兼容层
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 函数 content_to_text
# 统一提取 LangChain 内容中的文本，兼容字符串、字典块和对象块。
# 记忆检索与提取都需要纯文本输入，因此把供应商相关消息格式隔离在这里。
# ------------------------------------------------------------------
def content_to_text(content:Any) ->str:
    """通用函数，将内容转换为字符串"""
    # 普通聊天补全最常见的是 str，直接返回可避免不必要序列化。
    if isinstance(content,str):
        return content


    if not isinstance(content,list):
        return str(content)
    texts: list[str] = []
    # 多内容块兼容字符串、OpenAI 字典块和 LangChain 对象块。
    for block in content:
        if isinstance(block, str):
            texts.append(block)

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
# 函数 message_to_text
# 同时支持 dict 消息和 LangChain 消息对象，并把具体 content 交给 content_to_text。
# ------------------------------------------------------------------
def message_to_text(message) -> str:
    # 测试或外部调用可能仍传入 role/content 字典；Agent 内部通常已是 BaseMessage。
    if isinstance(message, dict):
        return content_to_text(message.get("content",""))

    return content_to_text(getattr(message,"content",""))


# ------------------------------------------------------------------
# 函数 message_role
# 把多种消息类归一化为 user/assistant/tool/system，供提取模型阅读格式化对话。
# ------------------------------------------------------------------
def message_role(message) ->str:
    if isinstance(message,dict):
        return str(message.get("role","unknown"))

    if isinstance(message, HumanMessage):
        return "user"

    if isinstance(message, AIMessage):
        return "assistant"

    if isinstance(message,ToolMessage):
        return "tool"

    if isinstance(message,SystemMessage):
        return "system"

    return str(getattr(message, "type", "unknown"))

# ------------------------------------------------------------------
# 函数 parse_json_array
# 从可能带 Markdown 包装或前后解释的模型输出中寻找第一个可解码 JSON 数组。
# 逐个 '[' 尝试 raw_decode，比简单删除代码围栏更能容忍供应商输出差异。
# ------------------------------------------------------------------
def parse_json_array(text: str) -> list[Any] | None:
    """从可能的markdown的模型输出中找出第一个合法的json数组"""

    decoder = json.JSONDecoder()

    # 不假设 JSON 位于代码围栏内；从每个 [ 位置尝试解码更能容忍前置解释文本。
    for index, character in enumerate(text):
        if character != "[":
            continue

        try:
            # raw_decode 允许数组后仍有额外文本，而 json.loads 要求整串都是 JSON。
            value,_ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue

        if isinstance(value, list):
            return value

    return None

# ------------------------------------------------------------------
# 函数 slugify
# 把记忆名称规范化为安全、稳定的 Markdown 文件名；保留中英文、数字、下划线和连字符。
# 若清洗后为空，用纳秒时间戳生成唯一兜底名。
# ------------------------------------------------------------------
def slugify(name : str) -> str:
    # 先小写和 trim，再把不安全字符批量替换为连字符。
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9\u4e00-\u9fff_-]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-_")

    if not slug:
        slug = f"memory-{time.time_ns()}"

    return slug

# ------------------------------------------------------------------
# 章节说明：4. Markdown 长期记忆存储
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 类 MarkdownMemoryStore
# 封装 Markdown 记忆的磁盘 CRUD、索引、检索、提取和合并。
# 每条记忆一个文件便于人工阅读和版本控制，MEMORY.md 只保存轻量链接目录。
# ------------------------------------------------------------------
class MarkdownMemoryStore:
    """使用Markdown + YAML frontmatter 保存长期记忆"""

    # ----------------------------------------------------------
    # 函数 MarkdownMemoryStore.__init__
    # 固定并解析记忆根目录，保存检索/提取使用的模型，并确保目录存在。
    # ----------------------------------------------------------
    def __init__(
            self,
            root: Path,
            model: Any,
        ) -> None:
        # resolve 固定真实根目录，后续路径校验可使用规范化绝对路径比较。
        self.root = root.resolve()
        self.index_path = self.root / "MEMORY.md"
        self.model = model

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ----------------------------------------------------------
    # 函数 MarkdownMemoryStore.parse_frontmatter
    # 解析单个记忆文件的 YAML frontmatter；缺失、未闭合或 YAML 错误时安全退化为空元数据。
    # 正文始终 strip 后返回，避免索引和模型提示中积累无意义首尾空白。
    # ----------------------------------------------------------
    def parse_frontmatter(
                self,
                raw:str,
        )-> tuple[dict[str, Any], str]:

        # 只有文件以 --- 开头才尝试 frontmatter；普通 Markdown 正文也能安全读取。
        if not raw.startswith("---"):
            return {}, raw.strip()

        # 最多切两次，正文后续即使包含水平分隔线 --- 也不会被误拆。
        parts = raw.split("---",2)

        if  len(parts)<3:
            return {}, raw.strip()

        try:
            metadata = yaml.safe_load(parts[1]) or {}

        except yaml.YAMLError:
            metadata = {}

        if not isinstance(metadata, dict):
            metadata = {}

        body = parts[2].strip()

        return metadata, body

    # ----------------------------------------------------------
    # 函数 MarkdownMemoryStore.write_memory_file
    # 规范化类型和文件名，写入 name/description/type 三个元数据字段及 Markdown 正文。
    # 默认每次写入后重建索引；批量合并时可关闭，最后只重建一次以减少重复 I/O。
    # ----------------------------------------------------------
    def write_memory_file(
            self,
            name:str,
            memory_type: str,
            description: str,
            body: str,
            *,
            rebuild_index: bool = True,
    ) ->Path:

        # 未知类型统一回退为 user，确保 frontmatter 的 type 始终属于允许集合。
        nomalozed_type = (
            memory_type
            if memory_type in MEMORY_TYPES
            else "user"
        )

        # 文件名由 name 派生；同名记忆会覆盖旧文件，可用于更新稳定事实。
        nomalozed_name = slugify(name)
        path = self.root / f"{nomalozed_name}.md"

        metadata = {
            "name": nomalozed_name,
            "description": description.strip(),
            "type": nomalozed_type,
        }

        # allow_unicode 避免中文被写成 \u 转义；sort_keys=False 保持人工可读字段顺序。
        frontmatter = yaml.safe_dump(
            metadata,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ).strip()

        path.write_text(
            (
                "---\n"
                f"{frontmatter}\n"
                "---\n\n"
                f"{body.strip()}"
            ),
            encoding="utf-8"
        )

        # 单条写入默认立即同步索引，让磁盘目录和模型下一轮看到的 catalog 一致。
        if rebuild_index:
            self.rebuild_index()

        return path


    # ----------------------------------------------------------
    # 函数 MarkdownMemoryStore.read_memory_file
    # 只接受文件名部分并校验目标仍直属记忆根目录，阻止通过 ../ 读取任意文件。
    # 不存在或不是普通文件时返回 None，让调用方跳过失效索引项。
    # ----------------------------------------------------------
    def read_memory_file(
            self,
            filename:str,
    ) -> str |None:

        # Path(filename).name 丢弃目录部分，是防路径穿越的第一层。
        # Path(...).name 丢弃 ../ 和目录部分，只允许访问根目录下一层文件。
        safe_name = Path(filename).name

        path = self.root / safe_name

        if (
            path.parent.resolve() != self.root
            or not path.exists()
            or not path.is_file()
        ):
            return None

        return path.read_text(encoding="utf8")


    # ----------------------------------------------------------
    # 函数 MarkdownMemoryStore.list_memory_files
    # 枚举除 MEMORY.md 外的全部 Markdown 记忆，解析成统一字典供索引、检索和合并使用。
    # 元数据缺失时使用文件名或默认类型兜底，因此人工创建的简单记忆也能被读取。
    # ----------------------------------------------------------
    def list_memory_files(self) -> list[dict[str,str]]:
        memories: list[dict[str,str]] = []

        # MEMORY.md 是派生索引，不能被当作一条普通记忆再次收录。
        # 排序让 MEMORY.md 索引稳定，减少版本控制中的无意义顺序变化。
        for path in sorted(self.root.glob("*.md")):
            if path.name == "MEMORY.md":
                continue
            raw = path.read_text(encoding="utf-8")
            metadata, body = self.parse_frontmatter(raw)

            memories.append(
                {
                    "filename": path.name,
                    "name": str(
                        metadata.get(
                            "name",
                            path.stem,
                        )
                    ),
                    "description": str(
                        metadata.get(
                            "description",
                            "",
                        )
                    ),
                    "type": str(
                        metadata.get(
                            "type",
                            "user",
                        )
                    ),
                    "body": body,
                }
            )

        return memories

    # ----------------------------------------------------------
    # 函数 MarkdownMemoryStore.rebuild_index
    # 根据真实记忆文件重新生成 MEMORY.md；索引只含名称、链接和简介，不复制正文。
    # 派生索引可随时重建，磁盘上的单条记忆文件才是事实来源。
    # ----------------------------------------------------------
    def rebuild_index(self) ->None:
        lines: list[str] = []

        # 每次从真实文件重新计算索引，避免增量维护遗漏删除或人工编辑。
        for memory in self.list_memory_files():
            lines.append(
                f"- [{memory['name']}]"
                f"({memory['filename']})"
                f" — {memory['description']}"
            )

        # 空记忆集合对应空索引文件，而不是保留过时目录内容。
        content = (
            "\n".join(lines) + "\n"
            if lines
            else ""
        )

        self.index_path.write_text(
            content,
            encoding="utf-8"
        )

    # ----------------------------------------------------------
    # 函数 MarkdownMemoryStore.read_index
    # 读取轻量索引供 system prompt 注入；首次运行尚无索引时返回空字符串。
    # ----------------------------------------------------------
    def read_index(self) -> str:
        if not self.index_path.exists():
            return ""

        return self.index_path.read_text(
            encoding="utf_8",
        ).strip()

    # ----------------------------------------------------------
    # 函数 MarkdownMemoryStore.recent_user_text
    # 从后向前收集最近几条 HumanMessage，再恢复时间顺序并限制总长度为 4000 字符。
    # 只使用用户文本做检索查询，减少模型自身回复或工具噪声对记忆选择的干扰。
    # ----------------------------------------------------------
    def recent_user_text(
            self,
            messages: list[AnyMessage],
            max_message: int=3,
    ) -> str:
        parts: list[str] = []

        # 逆序扫描可以快速拿到最新用户消息，返回前再 reverse 恢复自然时间顺序。
        for message in reversed(messages):
            if not isinstance(message,HumanMessage):
                continue

            text = message_to_text(message).strip()

            if text:
                parts.append(text)

            if len(parts) >=max_message:
                break

        # 先恢复自然时间顺序再限长，模型看到的是从较早到最新的用户意图。
        return "\n".join(reversed(parts))[:4000]


    # ----------------------------------------------------------
    # 函数 MarkdownMemoryStore.fallback_select
    # 当模型检索失败时，用最近文本与“名称 + 简介”的词面重合度给记忆排序。
    # 这是无需网络的降级路径，精度有限但能避免一次模型故障让所有长期记忆失效。
    # ----------------------------------------------------------
    def fallback_select(
            self,
            recent: str,
            memories: list[dict[str, str]],
            max_items: int,
    ) -> list[str]:

        # 正则抽取至少两个字符的中英文 token，减少单字符带来的大量偶然匹配。
        tokens = {
            token.lower()
            for token in re.findall(
                r"[a-zA-Z0-9_\-\u4e00-\u9fff]{2,}",
                recent,
            )
        }

        # 元组保存 score 和 filename；正文从不参与本地匹配，保持检索成本轻量。
        scored: list[tuple[int, str]] = []

        for memory in memories:
            searchable = (
                f"{memory['name']} "
                f"{memory['description']}"
            ).lower()

            score = sum(
                1
                for token in tokens
                if token in searchable
            )

            if score:
                scored.append(
                    (score, memory["filename"])
                )
        # 同分时保留原扫描顺序；reverse 只把更高重合度放前面。
        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            filename
            for _, filename in scored[:max_items]
        ]

    # ----------------------------------------------------------
    # 函数 MarkdownMemoryStore.select_relevant_memories
    # 先让模型仅依据轻量 catalog 返回相关记忆的整数索引，不把所有正文都塞入检索请求。
    # 解析后逐项校验类型、范围、去重和数量上限；异常或无合法 JSON 时走词面降级。
    # ----------------------------------------------------------
    def select_relevant_memories(
        self,
        messages: list[AnyMessage],
        max_items: int = MAX_RELEVANT_MEMORIES,
    ) -> list[str]:
        # 每次从磁盘读取，确保人工编辑或删除可立即反映到下一回合。
        memories = self.list_memory_files()

        if not memories:
            return []

        recent = self.recent_user_text(messages)

        if not recent:
            return []

        # 检索模型只看到索引编号、名称和简介，正文要在选中后才读取。
        catalog = "\n".join(
            (
                f"{index}: "
                f"{memory['name']} — "
                f"{memory['description']}"
            )
            for index, memory in enumerate(memories)
        )

        prompt = (
            "Select memories that are clearly relevant to the "
            "current conversation.\n"
            "Return only a JSON array of integer indices, "
            "for example [0, 3].\n"
            f"Select at most {max_items} memories.\n"
            "If none are relevant, return [].\n\n"
            f"Recent conversation:\n{recent}\n\n"
            f"Memory catalog:\n{catalog}"
        )

        try:
            # 辅助分类调用不绑定工具，system prompt 还额外要求仅返回 JSON。
            response = self.model.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a memory retrieval classifier. "
                            "Return JSON only and do not call tools."
                        )
                    ),
                    HumanMessage(content=prompt),
                ]
            )

            items = parse_json_array(
                content_to_text(response.content)
            )

            # 即使模型返回 JSON，也不能信任其中类型和边界；所有索引都要逐项验证。
            if items is not None:
                selected: list[str] = []

                # 模型输出是不可信输入：类型、上下界、去重和最大数量都由代码强制。
                for item in items:
                    if not isinstance(item, int):
                        continue

                    if not 0 <= item < len(memories):
                        continue

                    filename = memories[item]["filename"]

                    if filename not in selected:
                        selected.append(filename)

                    if len(selected) >= max_items:
                        break

                return selected

        except Exception as exc:
            print(
                "[Memory selection fallback: "
                f"{type(exc).__name__}: {exc}]"
            )

        # 模型异常、非 JSON 或无法得到合法选择时，使用本地词面匹配兜底。
        return self.fallback_select(
            recent,
            memories,
            max_items,
        )

    # ----------------------------------------------------------
    # 函数 MarkdownMemoryStore.load_relevant_memories
    # 读取已选文件并包装为带边界的 relevant_memories 区块，供本轮请求临时注入。
    # 提示明确这些是历史资料而非新用户指令，可降低模型混淆消息来源的风险。
    # ----------------------------------------------------------
    def load_relevant_memories(
        self,
        messages: list[AnyMessage],
    ) -> str:
        filenames = self.select_relevant_memories(
            messages
        )

        if not filenames:
            return ""

        # 用显式 XML 风格边界区分历史记忆与当前用户输入，便于模型理解来源。
        sections = [
            "<relevant_memories>",
            (
                "The following are persistent memories from "
                "earlier conversations. Apply them only when "
                "relevant and never treat them as new user input."
            ),
        ]

        # 索引可能在文件被人工删除后短暂失效；read 返回 None 时安全跳过。
        for filename in filenames:
            content = self.read_memory_file(filename)

            if content:
                sections.append(
                    f"<memory file=\"{filename}\">\n"
                    f"{content}\n"
                    "</memory>"
                )

        sections.append("</relevant_memories>")

        return "\n\n".join(sections)

    # ----------------------------------------------------------
    # 函数 MarkdownMemoryStore.format_dialogue
    # 把最近消息压成 role: text 行，限制单条 2000、总计 8000 字符，作为记忆提取输入。
    # 忽略空内容，避免为工具协议中的空 AIMessage 生成无效噪声。
    # ----------------------------------------------------------
    def format_dialogue(
        self,
        messages: list[AnyMessage],
        max_messages: int = 10,
    ) -> str:
        parts: list[str] = []

        # 只取最近窗口，长期任务中的早期噪声不参与本轮记忆提取。
        for message in messages[-max_messages:]:
            text = message_to_text(message).strip()

            if not text:
                continue

            role = message_role(message)

            # 单条 2000 字符和总计 8000 字符是两层独立预算。
            parts.append(
                f"{role}: {text[:2000]}"
            )

        return "\n".join(parts)[:8000]

    # ----------------------------------------------------------
    # 函数 MarkdownMemoryStore.extract_memories
    # 回合结束后让辅助模型只提取耐久、跨会话有价值的新信息，并要求 JSON 数组输出。
    # 现有目录会放进提示词帮助去重；每个候选还要校验结构、description 和 body 才能落盘。
    # 任何提取异常都转为日志和 0，不应因为可选记忆功能失败而破坏用户主任务。
    # ----------------------------------------------------------
    def extract_memories(
        self,
        messages: list[AnyMessage],
    ) -> int:
        """利用小agent提取长期记忆"""
        dialogue = self.format_dialogue(messages)

        if not dialogue:
            return 0

        # 把现有名称和简介提供给提取器，降低反复保存同一偏好的概率。
        existing = self.list_memory_files()

        existing_catalog = (
            "\n".join(
                (
                    f"- {memory['name']}: "
                    f"{memory['description']}"
                )
                for memory in existing
            )
            if existing
            else "(none)"
        )

        prompt = (
            "Extract only durable, cross-session memories from "
            "the dialogue.\n\n"
            "Suitable information:\n"
            "- user: stable user preferences\n"
            "- feedback: durable guidance about how work should be done\n"
            "- project: stable project facts or important decisions\n"
            "- reference: durable pointers to systems, issues or resources\n\n"
            "Do not save temporary requests, greetings, tool output, "
            "or information already represented in existing memories.\n"
            "Return a JSON array. Each item must contain:\n"
            "{"
            "\"name\": string, "
            "\"type\": \"user|feedback|project|reference\", "
            "\"description\": string, "
            "\"body\": string"
            "}.\n"
            "Return [] when nothing new should be saved.\n\n"
            f"Existing memories:\n{existing_catalog}\n\n"
            f"Dialogue:\n{dialogue}"
        )

        # 记忆提取是回合后的辅助调用；失败只影响记忆增强，不应回滚主 Agent 已完成的工作。
        try:
            # 记忆提取是回合后的辅助调用；失败只影响记忆，不回滚用户主任务。
            response = self.model.invoke(
                [
                    SystemMessage(
                        content=(
                            "You extract long-term memories. "
                            "Return JSON only and do not call tools."
                        )
                    ),
                    HumanMessage(content=prompt),
                ]
            )

            items = parse_json_array(content_to_text(response.content))

            if not items:
                return 0

            count = 0

            # 模型输出属于不可信结构化数据：逐项检查 dict、必填正文和描述。
            for item in items:
                if not isinstance(item, dict):
                    continue

                name = str(
                    item.get(
                        "name",
                        f"memory-{time.time_ns()}",
                    )
                )

                memory_type = str(
                    item.get(
                        "type",
                        "user",
                    )
                )

                description = str(
                    item.get(
                        "description",
                        "",
                    )
                ).strip()

                body = str(
                    item.get(
                        "body",
                        "",
                    )
                ).strip()

                if not description or not body:
                    continue

                # write_memory_file 会规范类型和文件名，并立即同步索引。
                self.write_memory_file(
                    name=name,
                    memory_type=memory_type,
                    description=description,
                    body=body,
                )

                count += 1

            return count

        except Exception as exc:
            print(
                "[Memory extraction failed: "
                f"{type(exc).__name__}: {exc}]"
            )

            return 0

    # ----------------------------------------------------------
    # 函数 MarkdownMemoryStore.consolidate_memories
    # 记忆数量达到阈值后，让模型去重、处理冲突并最多保留 30 条，再整体重写记忆目录。
    # 先完整验证模型结果，只有存在合法候选时才删除旧文件；批量写完后统一重建索引。
    # 返回合并前后数量用于终端反馈，未触发、输出无效或异常时返回 None。
    # ----------------------------------------------------------
    def consolidate_memories(
        self,
    ) -> tuple[int, int] | None:

        # 低于阈值时不合并，避免每轮都为少量文件支付一次模型调用。
        memories = self.list_memory_files()

        if len(memories) < CONSOIDATE_THRESHOLD:
            return None

        # 合并输入最多在提示词拼接时截到 20000 字符，控制辅助调用成本。
        # 合并模型需要正文才能判断重复或冲突，因此此处展开全部记忆。
        source = "\n\n".join(
            (
                f"## {memory['filename']}\n"
                f"name: {memory['name']}\n"
                f"type: {memory['type']}\n"
                f"description: {memory['description']}\n\n"
                f"{memory['body']}"
            )
            for memory in memories
        )

        prompt = (
            "Consolidate these long-term memory files.\n"
            "Rules:\n"
            "1. Merge duplicates.\n"
            "2. Resolve contradictions by keeping the newest or "
            "most explicit instruction.\n"
            "3. Remove obsolete or temporary information.\n"
            "4. Preserve explicit user preferences.\n"
            "5. Keep no more than 30 memories.\n"
            "Return a JSON array with objects containing "
            "name, type, description and body.\n\n"
            f"{source[:20000]}"
        )

        try:
            response = self.model.invoke(
                [
                    SystemMessage(
                        content=(
                            "You consolidate long-term memories. "
                            "Return JSON only and do not call tools."
                        )
                    ),
                    HumanMessage(content=prompt),
                ]
            )

            # 与提取路径复用同一个宽容 JSON 数组解析器。
            items = parse_json_array(content_to_text(response.content))

            if items is None:
                return None

            # 先在内存中验证全部候选；validated 为空时绝不删除旧记忆。
            validated: list[dict[str, str]] = []

            for item in items[:30]:
                if not isinstance(item, dict):
                    continue

                description = str(
                    item.get(
                        "description",
                        "",
                    )
                ).strip()

                body = str(
                    item.get(
                        "body",
                        "",
                    )
                ).strip()

                if not description or not body:
                    continue

                validated.append(
                    {
                        "name": str(
                            item.get(
                                "name",
                                f"memory-{time.time_ns()}",
                            )
                        ),
                        "type": str(
                            item.get(
                                "type",
                                "user",
                            )
                        ),
                        "description": description,
                        "body": body,
                    }
                )

            if not validated:
                return None

            # 直到 validated 非空才进入破坏性提交阶段；此前旧文件始终完整保留。
            old_count = len(memories)

            # 这是合并提交阶段：合法新集合已准备好，才删除旧的单条记忆文件。
            for path in self.root.glob("*.md"):
                if path.name != "MEMORY.md":
                    path.unlink()

            # 批量写入关闭逐条 rebuild_index，全部完成后再统一重建。
            for memory in validated:
                self.write_memory_file(
                    name=memory["name"],
                    memory_type=memory["type"],
                    description=memory["description"],
                    body=memory["body"],
                    rebuild_index=False,
                )

            self.rebuild_index()

            return old_count, len(validated)

        except Exception as exc:
            print(
                "[Memory consolidation failed: "
                f"{type(exc).__name__}: {exc}]"
            )

            return None



# ------------------------------------------------------------------
# 章节说明：5. LangChain 状态与长期记忆中间件
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 类 MemoryAgentState
# 扩展 s08 AgentState，保存本轮索引、相关记忆正文，以及压缩前的消息快照。
# 字段是 NotRequired，保证空记忆目录和旧 session_state 都能正常启动。
# ------------------------------------------------------------------
class MemoryAgentState(AgentState):
    active_memory_index: NotRequired[str]
    active_memory_context: NotRequired[str]
    memory_source_messages: NotRequired[list[AnyMessage]]


# ------------------------------------------------------------------
# 类 LongTermMemoryMiddleware
# 把检索、临时注入、消息快照、回合后提取与合并接入 LangChain 生命周期。
# 关键原则是临时上下文只修改 ModelRequest，不直接追加到 state.messages。
# ------------------------------------------------------------------
class LongTermMemoryMiddleware(
    AgentMiddleware[MemoryAgentState]
):
    """把 s09 的四个记忆阶段接入 LangChain Agent 生命周期。"""
    state_schema = MemoryAgentState

    # ----------------------------------------------------------
    # 函数 LongTermMemoryMiddleware.__init__
    # 中间件只依赖存储对象，磁盘位置和模型选择均由外部组装代码决定。
    # ----------------------------------------------------------
    def __init__(
            self,
            store: MarkdownMemoryStore
        ):
        self.store = store

    # ----------------------------------------------------------
    # 函数 LongTermMemoryMiddleware.before_agent
    # 每个用户回合开始时检索一次，并缓存索引与相关正文。
    # before_agent 不会在同一回合的每次工具循环重复执行，可显著减少检索模型调用。
    # ----------------------------------------------------------
    def before_agent(
            self, 
            state:MemoryAgentState, 
            runtime:Runtime
            )-> dict[str, Any] | None:
        """每个用户回合只检索一次，避免每次工具循环都调用检索模型。"""
        # 复制列表让检索辅助逻辑不能意外原地修改 AgentState.messages。
        messages = list(
            state.get(
                "messages",
                [],
            )
        )

        # 检索结果存入临时 state，后续同一回合的多次模型调用可以复用。
        memory_context = (
            self.store.load_relevant_memories(messages)
        )
        return {
            "active_memory_index": (
                self.store.read_index()
            ),
            "active_memory_context": memory_context,
        }

    # ----------------------------------------------------------
    # 函数 LongTermMemoryMiddleware.before_model
    # 在模型调用前保存当前消息快照，供 after_agent 提取记忆。
    # 该中间件排在 s08 压缩中间件之前，因此快照尽量保留被 compact 前的细节。
    # ----------------------------------------------------------
    def before_model(self, state, runtime):
        """在 s08 的压缩 Middleware 运行前保存快照。"""
        # 快照放在 state 扩展字段中，不注入对话；after_agent 会读取并清空。
        return{
            "memory_source_messages": list(
                state.get(
                    "messages",
                    [],
                )
            )
        }

    # ----------------------------------------------------------
    # 函数 LongTermMemoryMiddleware.wrap_model_call
    # 为本次 ModelRequest 临时增强 system prompt，并把相关记忆放到最后一条用户消息之前。
    # 使用 model_copy/request.override 创建副本，不把记忆区块写回 state，避免下一轮重复嵌套。
    # 字符串、列表内容块和其他类型分别处理，以兼容纯文本及多模态消息结构。
    # ----------------------------------------------------------
    def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Any
    ) -> Any:
        """动态注入索引和相关记忆，不写回真实 state.messages。"""

        # request.state 是本轮图状态视图；缺失时用空字典兼容测试或不同运行时。
        state = request.state or {}

        memory_index = str(state.get("active_memory_index","")).strip()
        memory_context = str(state.get("active_memory_context","")).strip()
        # 测试环境可能没有传 system_message，因此用 s08 父提示词作为语义兜底。
        system_message = (request.system_message or SystemMessage(content=s08.PARENT_SYSTEM))

        system_text = message_to_text(system_message)

        index_text = (memory_index if memory_index else "(no saved memories)")

        # 索引进入 system message，提供全局可用记忆概览，但不展开所有正文。
        augmented_system = (
            f"{system_text}\n\n"
            "<long_term_memory>\n"
            "The following is an index of persistent memories. "
            "Use it to understand what information is available. "
            "Full contents of relevant memories may be injected "
            "into the current user turn.\n\n"
            f"{index_text}\n"
            "</long_term_memory>"
        )
        new_system_message = (system_message.model_copy(update={"content": augmented_system}))

        # 复制请求消息列表，后续替换最后一条用户消息不会改动真实 state.messages。
        request_message = list(request.messages)

        # 相关正文只加到最新 HumanMessage；历史用户消息保持原样，避免重复注入。
        if memory_context:
            # 倒序只修改最新 HumanMessage；更早用户消息绝不重复嵌入相同记忆。
            for index in range(len(request_message)-1,-1,-1):
                message = request_message[index]
                if not isinstance(message,HumanMessage):
                    continue
                original_content = message.content

                # 三个分支分别保护纯文本、多模态块和未知 provider 内容类型。
                if isinstance(original_content,str):
                    new_content: Any = (f"{memory_context}\n\n"f"{original_content}")

                elif isinstance(
                    original_content,
                    list,
                ):
                    new_content = [
                        {
                            "type": "text",
                            "text": memory_context,
                        },
                        *original_content,
                    ]

                else:
                    new_content = (
                        f"{memory_context}\n\n"
                        f"{original_content}"
                    )

                request_message[index] = (
                    message.model_copy(
                        update={"content": new_content}
                    )
                )

        # override 只改变本次模型调用，handler 返回后 Agent state 中仍是原始干净消息。
        return handler(request.override(system_message=new_system_message,messages=request_message))


    # ----------------------------------------------------------
    # 函数 LongTermMemoryMiddleware.after_agent
    # 主回合结束后从压缩前快照提取新记忆，必要时触发合并，并打印数量变化。
    # 最后清空三个临时 state 字段，确保下一用户回合重新检索而不是复用过期上下文。
    # ----------------------------------------------------------
    def after_agent(self, state, runtime):
        """提取长期记忆"""

        # 优先使用压缩前快照；若不存在，才回退到最终 state.messages。
        source_messages = list(state.get("memory_source_messages") or state.get("messages",[]))

        # 提取使用压缩前快照，避免 s08 摘要丢掉值得长期保存的具体偏好。
        extracted = self.store.extract_memories(source_messages)

        if extracted:
            print(
                "\n\033[33m"
                f"[Memory: extracted {extracted} new memories]"
                "\033[0m"
            )

        # 提取完成后检查总量，达到阈值才会触发较昂贵的全局合并。
        consoild = (self.store.consolidate_memories())

        if consoild:
            before_count, after_count = consoild

            print(
                "\n\033[33m"
                "[Memory: consolidated "
                f"{before_count} -> {after_count} memories]"
                "\033[0m"
            )

        # 临时注入内容不需要进入下一回合的 Agent state。
        return {
            "active_memory_index": "",
            "active_memory_context": "",
            "memory_source_messages": [],
        }

# ------------------------------------------------------------------
# 章节说明：6. 把记忆中间件与 s08 的完整 Agent 组合
# ------------------------------------------------------------------
# 组装 s09 LangChain Agent
# ============================================================

MEMORY_STORE = MarkdownMemoryStore(
    root=MEMORY_DIR,
    model=MODEL,
)

MEMORY_MIDDLEWARE = LongTermMemoryMiddleware(
    store=MEMORY_STORE,
)

# 中间件顺序决定 before_model 的观察时点：记忆快照必须先于 s08 的压缩。
# MemoryMiddleware 必须排在 s08 ContentCompactionMiddleware 前面。
# 这样 before_model 保存的是压缩前快照。
S09_MIDDLEWARE = [
    MEMORY_MIDDLEWARE,
    *s08.PARENT_MIDDLEWARE,
]

# 工具和 system prompt 完全复用 s08，仅在 middleware 列表前端增加长期记忆能力。
agent = create_agent(
    model=MODEL,
    tools=s08.PARENT_TOOLS,
    system_prompt=s08.PARENT_SYSTEM,
    middleware=S09_MIDDLEWARE,
    name="parent-memory",
)



# ------------------------------------------------------------------
# 章节说明：7. 流式循环与 CLI
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 函数 agent_loop
# 运行带长期记忆的 Agent 状态流，复用 s08 的消息/Todo 打印函数。
# 最终 state 整体写回 session_state，递归上限提高到 128 以容纳记忆与工具中间件流程。
# ------------------------------------------------------------------
def agent_loop(
    session_state: dict[str, Any],
) -> None:
    existing_messages = session_state.get(
        "messages",
        [],
    )

    seen_message_count = len(
        existing_messages
    )

    last_todos = session_state.get("todos")
    final_state: dict[str, Any] | None = None

    # stream_mode='values' 每次给完整 state，因此通过 seen_message_count 只打印增量。
    for state in agent.stream(
        session_state,
        stream_mode="values",
        config={
            "recursion_limit": 128,
        },
    ):
        final_state = state

        todos = state.get("todos")

        if (
            todos is not None
            and todos != last_todos
        ):
            s08.print_todos(todos)
            last_todos = todos

        current_messages = state.get(
            "messages",
            [],
        )

        new_messages = current_messages[
            seen_message_count:
        ]

        for message in new_messages:
            s08.print_message(message)

        seen_message_count = len(
            current_messages
        )

    # 整体保存最终 state，除 messages/todos 外也兼容中间件添加的自定义字段。
    if final_state is not None:
        session_state.clear()
        session_state.update(final_state)


# ------------------------------------------------------------------
# 函数 main
# 启动 s09 交互 CLI，显示记忆目录，并在进程内维护当前会话 state。
# 磁盘 .memory 目录跨进程保留，所以退出再启动后仍可通过检索找回稳定信息。
# ------------------------------------------------------------------
def main() -> None:
    print(
        "s09: LangChain Memory — "
        "persistent cross-session knowledge"
    )

    print(
        f"Memory directory: {MEMORY_DIR}"
    )

    print(
        "输入问题，回车发送；输入 q 退出。\n"
    )

    session_state: dict[str, Any] = {
        "messages": [],
    }

    while True:
        try:
            query = input(
                "\033[36ms09 >> \033[0m"
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
