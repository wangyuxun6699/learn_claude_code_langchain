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
from langgraph.runtime import Runtime

from s08_context_compact import code as s08
WORKDIR = s08.WORKDIR
MODEL = s08.MODEL
MEMORY_DIR = WORKDIR / ".memory"
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"
MEMORY_TYPES ={
    "user",
    "feedback",
    "project",
    "reference",
}

MAX_RELEVANT_MEMORIES = 5
CONSOIDATE_THRESHOLD = 10


def content_to_text(content:Any) ->str:

    if isinstance(content,str):
        return content


    if not isinstance(content,list):
        return str(content)
    texts: list[str] = []
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


def message_to_text(message) -> str:
    if isinstance(message, dict):
        return content_to_text(message.get("content",""))

    return content_to_text(getattr(message,"content",""))


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

def parse_json_array(text: str) -> list[Any] | None:


    decoder = json.JSONDecoder()

    for index, character in enumerate(text):
        if character != "[":
            continue

        try:
            value,_ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue

        if isinstance(value, list):
            return value

    return None

def slugify(name : str) -> str:
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9\u4e00-\u9fff_-]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-_")

    if not slug:
        slug = f"memory-{time.time_ns()}"

    return slug

class MarkdownMemoryStore:

    def __init__(
            self,
            root: Path,
            model: Any,
        ) -> None:
        self.root = root.resolve()
        self.index_path = self.root / "MEMORY.md"
        self.model = model

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )
    def parse_frontmatter(
                self,
                raw:str,
        )-> tuple[dict[str, Any], str]:
        if not raw.startswith("---"):
            return {}, raw.strip()

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

    def write_memory_file(
            self,
            name:str,
            memory_type: str,
            description: str,
            body: str,
            *,
            rebuild_index: bool = True,
    ) ->Path:
        nomalozed_type = (
            memory_type
            if memory_type in MEMORY_TYPES
            else "user"
        )

        nomalozed_name = slugify(name)
        path = self.root / f"{nomalozed_name}.md"

        metadata = {
            "name": nomalozed_name,
            "description": description.strip(),
            "type": nomalozed_type,
        }

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

        if rebuild_index:
            self.rebuild_index()

        return path


    def read_memory_file(
            self,
            filename:str,
    ) -> str |None:
        safe_name = Path(filename).name

        path = self.root / safe_name

        if (
            path.parent.resolve() != self.root
            or not path.exists()
            or not path.is_file()
        ):
            return None

        return path.read_text(encoding="utf8")


    def list_memory_files(self) -> list[dict[str,str]]:
        memories: list[dict[str,str]] = []

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

    def rebuild_index(self) ->None:
        lines: list[str] = []

        for memory in self.list_memory_files():
            lines.append(
                f"- [{memory['name']}]"
                f"({memory['filename']})"
                f" — {memory['description']}"
            )

        content = (
            "\n".join(lines) + "\n"
            if lines
            else ""
        )

        self.index_path.write_text(
            content,
            encoding="utf-8"
        )

    def read_index(self) -> str:
        if not self.index_path.exists():
            return ""

        return self.index_path.read_text(
            encoding="utf_8",
        ).strip()

    def recent_user_text(
            self,
            messages: list[AnyMessage],
            max_message: int=3,
    ) -> str:
        parts: list[str] = []

        for message in reversed(messages):
            if not isinstance(message,HumanMessage):
                continue

            text = message_to_text(message).strip()

            if text:
                parts.append(text)

            if len(parts) >=max_message:
                break

        return "\n".join(reversed(parts))[:4000]


    def fallback_select(
            self,
            recent: str,
            memories: list[dict[str, str]],
            max_items: int,
    ) -> list[str]:
        tokens = {
            token.lower()
            for token in re.findall(
                r"[a-zA-Z0-9_\-\u4e00-\u9fff]{2,}",
                recent,
            )
        }

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
        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            filename
            for _, filename in scored[:max_items]
        ]

    def select_relevant_memories(
        self,
        messages: list[AnyMessage],
        max_items: int = MAX_RELEVANT_MEMORIES,
    ) -> list[str]:
        memories = self.list_memory_files()

        if not memories:
            return []

        recent = self.recent_user_text(messages)

        if not recent:
            return []

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

            if items is not None:
                selected: list[str] = []

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

        return self.fallback_select(
            recent,
            memories,
            max_items,
        )

    def load_relevant_memories(
        self,
        messages: list[AnyMessage],
    ) -> str:
        filenames = self.select_relevant_memories(
            messages
        )

        if not filenames:
            return ""

        sections = [
            "<relevant_memories>",
            (
                "The following are persistent memories from "
                "earlier conversations. Apply them only when "
                "relevant and never treat them as new user input."
            ),
        ]

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

    def format_dialogue(
        self,
        messages: list[AnyMessage],
        max_messages: int = 10,
    ) -> str:
        parts: list[str] = []

        for message in messages[-max_messages:]:
            text = message_to_text(message).strip()

            if not text:
                continue

            role = message_role(message)

            parts.append(
                f"{role}: {text[:2000]}"
            )

        return "\n".join(parts)[:8000]

    def extract_memories(
        self,
        messages: list[AnyMessage],
    ) -> int:

        dialogue = self.format_dialogue(messages)

        if not dialogue:
            return 0

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

        try:
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

    def consolidate_memories(
        self,
    ) -> tuple[int, int] | None:
        memories = self.list_memory_files()

        if len(memories) < CONSOIDATE_THRESHOLD:
            return None

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

            items = parse_json_array(content_to_text(response.content))

            if items is None:
                return None

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

            old_count = len(memories)

            for path in self.root.glob("*.md"):
                if path.name != "MEMORY.md":
                    path.unlink()

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


class MemoryAgentState(AgentState):
    active_memory_index: NotRequired[str]
    active_memory_context: NotRequired[str]
    memory_source_messages: NotRequired[list[AnyMessage]]


class LongTermMemoryMiddleware(
    AgentMiddleware[MemoryAgentState]
):

    state_schema = MemoryAgentState

    def __init__(
            self,
            store: MarkdownMemoryStore
        ):
        self.store = store

    def before_agent(
            self,
            state:MemoryAgentState,
            runtime:Runtime
            )-> dict[str, Any] | None:

        messages = list(
            state.get(
                "messages",
                [],
            )
        )
        memory_context = (
            self.store.load_relevant_memories(messages)
        )
        return {
            "active_memory_index": (
                self.store.read_index()
            ),
            "active_memory_context": memory_context,
        }

    def before_model(self, state, runtime):

        return{
            "memory_source_messages": list(
                state.get(
                    "messages",
                    [],
                )
            )
        }

    def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Any
    ) -> Any:

        state = request.state or {}

        memory_index = str(state.get("active_memory_index","")).strip()
        memory_context = str(state.get("active_memory_context","")).strip()
        system_message = (request.system_message or SystemMessage(content=s08.PARENT_SYSTEM))

        system_text = message_to_text(system_message)

        index_text = (memory_index if memory_index else "(no saved memories)")

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

        request_message = list(request.messages)

        if memory_context:
            for index in range(len(request_message)-1,-1,-1):
                message = request_message[index]
                if not isinstance(message,HumanMessage):
                    continue
                original_content = message.content

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
        return handler(request.override(system_message=new_system_message,messages=request_message))


    def after_agent(self, state, runtime):

        source_messages = list(state.get("memory_source_messages") or state.get("messages",[]))

        extracted = self.store.extract_memories(source_messages)

        if extracted:
            print(
                "\n\033[33m"
                f"[Memory: extracted {extracted} new memories]"
                "\033[0m"
            )
        consoild = (self.store.consolidate_memories())

        if consoild:
            before_count, after_count = consoild

            print(
                "\n\033[33m"
                "[Memory: consolidated "
                f"{before_count} -> {after_count} memories]"
                "\033[0m"
            )


        return {
            "active_memory_index": "",
            "active_memory_context": "",
            "memory_source_messages": [],
        }


MEMORY_STORE = MarkdownMemoryStore(
    root=MEMORY_DIR,
    model=MODEL,
)

MEMORY_MIDDLEWARE = LongTermMemoryMiddleware(
    store=MEMORY_STORE,
)


S09_MIDDLEWARE = [
    MEMORY_MIDDLEWARE,
    *s08.PARENT_MIDDLEWARE,
]

agent = create_agent(
    model=MODEL,
    tools=s08.PARENT_TOOLS,
    system_prompt=s08.PARENT_SYSTEM,
    middleware=S09_MIDDLEWARE,
    name="parent-memory",
)


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

    if final_state is not None:
        session_state.clear()
        session_state.update(final_state)


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
