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

