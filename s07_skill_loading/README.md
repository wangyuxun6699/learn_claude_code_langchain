# s07: Skill Loading — 用到时再加载

> **对齐状态**：本章 `code.py` 对齐上游 `s07_skill_loading` 的结构；模型适配与本章机制在 `code.py` 中直接实现，使用 LangChain OpenAI-compatible 调用。
[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md)

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

## 本项目保留的 LangChain / LangGraph 教学补充

> 以下内容来自本仓库对齐前的 README，作为上游课程之外的本地教学补充完整保留。

<!-- local-langchain-additions:start -->
<details>
<summary>展开本仓库原有的 LangChain / LangGraph 教学说明</summary>

# s07: Skill Loading — 按需加载专业知识

> LangChain 教学改编版。章节结构与“深入 CC 源码”部分主要参考 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)。
>
> **Harness 层**：渐进式披露 — 目录常驻、正文按需载入。

[s06](../s06_subagent/) → **s07** → [s08](../s08_context_compact/)

---

## 问题

把所有专业说明一次性塞进 system prompt 会浪费上下文，也会让无关规则互相干扰。

---

## 解决方案

![s07: Skill Loading — 按需加载专业知识](images/skill-overview.svg)

启动时只扫描 `skills/*/SKILL.md` 的名称和描述；模型判断相关后，调用 `load_skill` 读取完整正文，随后注入system prompt。

---

## 工作原理：LangChain 版本

```python
SKILL_REGISTRY: dict[str, dict[str, str]] = {}

def scan_skills() -> None:
    for manifest in sorted(SKILL_DIR.glob("*/SKILL.md")):
        raw = manifest.read_text(encoding="utf-8")
        metadata, body = parse_frontmatter(raw)
        name = str(metadata.get("name") or manifest.parent.name)
        SKILL_REGISTRY[name] = {
            "description": str(metadata.get("description") or name),
            "content": raw,
        }

@tool
def load_skill(name: str) -> str:
    """Load one skill's complete instructions by exact name."""
    return SKILL_REGISTRY[name]["content"]
```

system prompt 中只注入 `name: description` 目录；完整 `SKILL.md` 只有在工具调用后才进入消息历史。

---

## 本章文件

- `code.py`：带注释教学版（可直接运行）；仓库根目录的 `skills/` 是可直接扫描的示例。
- `code_uncommented.py`：无教学注释的精简版。

---

## 试一下

先在仓库根目录准备环境，然后从根目录按模块运行：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m s07_skill_loading.code
```

> 这些教学 Agent 可以执行命令和修改文件。建议先在测试目录中试用，并认真阅读每次权限提示。

---

## 接下来

s08 处理不断增长的消息、工具结果与上下文窗口。

</details>
<!-- local-langchain-additions:end -->

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
