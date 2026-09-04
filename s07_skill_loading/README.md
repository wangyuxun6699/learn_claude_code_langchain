# s07: Skill Loading — 用到时再加载

s01 → s02 → s03 → s04 → s05 → s06 → `s07` → [s08](../s08_context_compact/) → s09 → ... → s16 → s17

> system prompt 保存技能目录；`load_skill` 返回完整的 `SKILL.md`。
>
> **Harness 层**：知识加载 — 让模型先知道有哪些技能，再按名称读取内容。

---

## 问题

假设某个项目有一套 React 组件规范、一份 SQL 风格指南和一份 API 设计文档。我们希望 Agent 在开发过程中遵守这些规范，最直接的做法就是把它们全部放进 system prompt：

```python
SYSTEM = (
    f"You are a coding agent. "
    + open("docs/react-style.md").read()
    + open("docs/sql-style.md").read()
    + open("docs/api-design.md").read()
)
```

这种做法能让 Agent 读到所有规范，但问题在于，三份文档被固定放进了 system prompt，无法根据当前任务只选择需要的那一份。每次调用 LLM 时，三份文档的全文都会一起发送给模型。当前任务只修改 React 组件时，实际需要的只有 React 组件规范；SQL 风格指南和 API 设计文档与任务无关，却仍然占用输入 token 和上下文窗口，留给代码、对话和工具结果的空间也会变少。

---

## 解决方案

![Skill Overview](images/skill-overview.svg)

启动时，`SkillLoader` 扫描 `skills/*/SKILL.md`，读取 YAML frontmatter 中的 `name` 和 `description`，并把这份目录加入 system prompt。模型需要完整说明时，调用 `load_skill(name)`；返回的 `SKILL.md` 作为 `tool_result` 追加到消息列表。

| 内容 | 进入模型的位置 | 何时加入 |
|------|----------------|----------|
| 技能名称和描述 | system prompt | 启动时 |
| 完整 `SKILL.md` | `tool_result` | 调用 `load_skill` 时 |

---

## 工作原理

每个技能是一个包含 `SKILL.md` 的目录：

```text
skills/
  agent-builder/SKILL.md
  code-review/SKILL.md
  mcp-builder/SKILL.md
  pdf/SKILL.md
```

### 扫描技能

```python
class SkillLoader:
    def scan(self):
        self.skills.clear()
        skills_root = self.skills_dir.resolve()
        for manifest in sorted(self.skills_dir.glob("*/SKILL.md")):
            if (not manifest.is_file()
                    or not manifest.resolve().is_relative_to(skills_root)):
                continue
            content = manifest.read_text(encoding="utf-8")
            metadata, body = self.parse_frontmatter(content)
            raw_name = metadata.get("name")
            name = raw_name.strip() if isinstance(raw_name, str) else ""
            name = name or manifest.parent.name
            raw_description = metadata.get("description")
            description = (raw_description.strip()
                           if isinstance(raw_description, str) else "")
            description = description or body.split("\n", 1)[0]
            description = " ".join(str(description).lstrip("# ").split())
            self.skills[name] = {
                "name": name,
                "description": description,
                "content": content,
            }
```

`catalog()` 只输出名称和描述：

```text
- code-review: Perform thorough code reviews...
- pdf: Process PDF files...
```

### 组装 system prompt

```python
def build_system_prompt() -> str:
    return (
        f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. "
        "Act, don't explain.\n\n"
        f"Skills available:\n{SKILL_LOADER.catalog()}\n\n"
        "Use load_skill to read the full instructions when a skill applies."
    )
```

固定的 Agent 指令和扫描得到的技能目录在这里组成实际传给模型的 system prompt。

### 加载完整内容

```python
def load(self, name: str) -> str:
    skill = self.skills.get(name)
    if skill:
        return skill["content"]
    available = ", ".join(self.skills) or "none"
    return f"Error: Unknown skill '{name}'. Available: {available}"
```

`name` 用于查询启动时建立的注册表，不会被当作文件路径。工具返回后，原有 Agent Loop 会把内容作为新的 `tool_result` 消息追加。

---

## 结合本章代码理解 Skills 与渐进披露

[`code.py`](code.py) 的 `SkillLoader` 把技能分成“目录信息”和“完整内容”两级。启动时扫描 `skills/*/SKILL.md`，只把名称和描述放进 system prompt；模型确认需要某个技能后，再调用 `load_skill(name)` 读取完整文件。

### 扫描阶段

`parse_frontmatter()` 只在首行和后续独立行都是 `---` 时解析 YAML，使用 `yaml.safe_load()`，并对无效或非字典 metadata 降级。`scan()` 还会：

- 将缺失的 `name` 回退为目录名。
- 将缺失的 `description` 回退为正文首行。
- 只接受 `SKILLS_DIR` 内真实的 `SKILL.md`，避免路径逃逸。
- 保留完整原文，确保加载时 frontmatter 和正文都可见。

### 为什么不在启动时加载全部技能

如果有几十个技能，把所有说明塞进 system prompt 会增加 token 成本，也会让模型在无关规则之间摇摆。本章 system prompt 只包含 catalog：

```text
- skill-name: 一句话描述
```

完整内容通过 `load_skill` 工具按需进入 ToolMessage。这就是渐进披露：先暴露“有什么”，再加载“怎么做”。

### 与 LangChain Skills 模式的关系

LangChain Skills 指南也把 skill 视为轻量、以 prompt 为主的专业能力，适合不需要独立状态和强隔离的知识包。它和 s06 子 Agent 的区别是：

| Skills | Subagents |
|---|---|
| 扩展同一个 Agent 的指令和知识 | 启动独立上下文中的 Agent |
| 成本低、组合轻 | 隔离强、可使用专属工具和模型 |
| 加载后仍由主 Agent 执行 | 子 Agent 自己推理和执行 |

在 `create_agent()` 中，可以把 `load_skill` 写成普通 `@tool`；也可以通过 middleware 动态修改 system prompt 或可见工具集合。若技能正文需要跨多轮长期保留，应明确它进入的是 thread state、runtime context 还是外部 Store，避免每次模型调用都重复加载。

### 运行时注意点

- 当前 `SYSTEM = build_system_prompt()` 在模块加载时生成；运行中新增技能需要重新扫描并重建 prompt。
- skill 内容应视为指令资料，但仍不能绕过宿主权限和工具边界。
- description 是模型选择技能的主要路由信号，应具体说明触发场景，而不是写成泛泛介绍。

官方概念：[Skills](https://docs.langchain.com/oss/python/langchain/multi-agent/skills) · [Context engineering](https://docs.langchain.com/oss/python/langchain/context-engineering)

---

## 试一下

```sh
cd learn-claude-code
python s07_skill_loading/code.py
```

试试这些 prompt：

1. `What skills are available?`
2. `Load the code-review skill and follow its instructions`
3. `Review README.md and load the relevant skill first`

观察 system prompt 中是否只有技能目录，以及调用 `load_skill` 后是否出现完整的 `SKILL.md` 内容。

---

## 接下来

随着工具调用增加，`messages[]` 会积累较早的文件内容和工具结果。

s08 Context Compact → 缩短较早的消息，为后续调用保留上下文空间。
---

<!-- upstream-cc-source:start -->
## 深入 CC 源码

> 原文：[s07_skill_loading](https://github.com/shareAI-lab/learn-claude-code/blob/67a9126c6435a8654ba7a6f68c0fd2130f00a462/s07_skill_loading/README.md)。以下折叠块保持原文，文中的章号与源码行号沿用该版本。

<details>
<summary>深入 CC 源码</summary>

> 以下基于 CC 源码 `loadSkillsDir.ts`、`SkillTool.ts`、`bundledSkills.ts`、`commands.ts` 的分析。

### 一、技能来源：不是只有一个 skills/ 目录

教学版假设所有技能在 `skills/` 目录下。CC 实际从多个来源加载，分布在多个文件中：`loadSkillsDir.ts` 负责从 user/project/`--add-dir` 目录和 legacy commands（`.claude/commands/`）加载；`bundledSkills.ts` 负责内置技能；`SkillTool.ts` 处理 MCP 远程技能；`commands.ts` 负责命令聚合。类型包括 managed/policy skills、user skills（`~/.claude/skills/`）、project skills（`.claude/skills/`）、`--add-dir` skills、legacy commands、dynamic skills、conditional skills（带 `paths` frontmatter，按文件路径激活）、bundled skills、plugin skills、MCP skills。

### 二、SKILL.md Frontmatter 常见字段

CC 的 SKILL.md YAML frontmatter 由 `parseSkillFrontmatterFields()` 解析（`loadSkillsDir.ts`），常见字段包括：

| 字段 | 用途 |
|------|------|
| `name` / `description` | 显示名称和描述 |
| `when_to_use` | 指导模型何时调用 |
| `allowed-tools` | 技能可用工具的自动允许列表 |
| `context` | `inline`（默认）或 `fork`（作为子 Agent 运行） |
| `model` | 模型覆盖（haiku/sonnet/opus/inherit） |
| `hooks` | 技能级别的 hook 配置 |
| `paths` | 条件激活的 glob 模式 |
| `user-invocable` | 用户可以通过 `/name` 调用 |

完整字段列表随版本迭代会变化，以上仅列出教学版涉及的核心字段。

### 三、两级加载的精确实现

1. **Catalog（启动时）**：`getSkillDirCommands()` 扫描目录 → 注册为 `Command` 对象，只包含元数据。`getSkillListingAttachments()` 把技能列表格式化为附件，预算为上下文窗口的 ~1%（上限 8000 字符）。
2. **Load（调用时）**：模型调 `Skill` 工具（输入字段是 `skill` + 可选 `args`，教学版用 `name`）→ `getPromptForCommand()` 展开完整 SKILL.md 内容 → `SkillTool` 返回的 tool_result 展示文本只是 `"Launching skill: {name}"`，真正的技能内容通过 `newMessages` 注入对话。教学版把两者合并为"通过 tool_result 注入"是一种简化；加载后的 SKILL.md 仍可作为指引，帮助模型后续通过现有 file/bash 工具访问相关资源。

### 教学版的简化是刻意的

- 多文件多来源 → 1 个 `skills/` 目录：足以展示两级加载的核心概念
- 多个 frontmatter 字段 → 只解析 name/description：减少解析复杂度
- forked skills（`context: 'fork'`）→ 省略：教学版只展开 inline 技能加载
- `Skill` 工具输入 `skill`+`args` → 教学版用 `name`：避免参数解析的额外复杂度

</details>

<!-- upstream-cc-source:end -->
