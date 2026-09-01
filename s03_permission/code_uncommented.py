from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_core.tools import tool
import os, subprocess
from pathlib import Path
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
WORKDIR = Path.cwd()
MODEL_ID = os.getenv('MODEL_ID')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SYSTEM = f'you are a coding agent at {WORKDIR}. Use tools to solve tasks. Act dont explain'
OPENAI_BASE_URL = os.getenv('BASE_URL')
from harness.security import check_deny_list

def resolve_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (WORKDIR / candidate).resolve()


def check_rules(tool_name: str, args: dict) -> str | None:
    if tool_name == 'run_bash':
        command = args.get('command', '')
        if command.strip().lower().startswith("del ") or any((kw in command for kw in ['rm ', '> /etc/', 'chmod 777'])):
            return 'potentially destructive command'
    if tool_name in ('run_write', 'run_edit', 'run_read'):
        path = args.get('path', '')
        if not resolve_path(path).is_relative_to(WORKDIR):
            return 'Working outside workspace'
    return None

def ask_user(tool_name: str, args: dict, reason: str) -> bool:
    print(f'\nWarning: {reason}')
    print(f'Tool: {tool_name}({args})')
    choice = input('Allow? [y/N] ').strip().lower()
    return choice in ('y', 'yes')

def check_permission(tool_name: str, args: dict) -> bool:
    if tool_name == 'run_bash':
        reason = check_deny_list(args.get('command', ''))
        if reason:
            print(f'\nBlocked: {reason}')
            return False
    reason = check_rules(tool_name, args)
    if reason:
        return ask_user(tool_name, args, reason)
    return True

@tool
# 安全边界：shell=True 仅为教学演示，黑名单/路径检查不等于安全边界；生产请使用权限中间件 + 沙箱。
def run_bash(command: str) -> str:
    """Execute a shell command in the current workspace."""
    if not check_permission('run_bash', {'command': command}):
        return 'permission denied'
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else '(no output)'
    except subprocess.TimeoutExpired:
        return 'Error: Timeout(120s)'
    except OSError as e:
        return f'Error: {e}'

@tool
def run_read(path: str, limit: int | None=None) -> str:
    """Read a UTF-8 text file, optionally limiting the returned line count."""
    if not check_permission('run_read', {'path': path}):
        return 'Permission denied.'
    try:
        lines = resolve_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f'...({len(lines) - limit} more lines)']
        return '\n'.join(lines)
    except Exception as e:
        return f'Error: {e}'

@tool
def run_write(path: str, content: str) -> str:
    """Write UTF-8 content to a file, replacing its existing content."""
    if not check_permission('run_write', {'path': path, 'content': content}):
        return 'permission is denied'
    try:
        file_path = resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f'write {len(content)} bytes to {path}'
    except Exception as e:
        return f'Error: {e}'

@tool
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace the first exact occurrence of old_text in a UTF-8 file."""
    if not check_permission('run_edit', {'path': path, 'old_text': old_text, 'new_text': new_text}):
        return 'permission is denied'
    try:
        file_path = resolve_path(path)
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f'Error: text not found in {path}'
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f'edit {path}'
    except Exception as e:
        return f'Error: {e}'

@tool
def run_glob(pattern: str) -> str:
    """Find workspace files matching a glob pattern."""
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return '\n'.join(results) if results else '(no matches)'
    except Exception as e:
        return f'Error: {e}'

def print_assistant_message(message: AIMessage) -> None:
    content = message.content
    if isinstance(content, str):
        print(content)
        return
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get('type') == 'text':
                    print(block.get('text', ''))
            elif hasattr(block, 'text'):
                print(block.text)
TOOLS = [run_bash, run_edit, run_glob, run_write, run_read]
MODEL = ChatOpenAI(model=MODEL_ID, max_completion_tokens=8000, temperature=0, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
agent = create_agent(model=MODEL, tools=TOOLS, system_prompt=SYSTEM)

def agent_loop(messages: list) -> None:
    result = agent.invoke({'messages': messages})
    new_messages = result['messages'][len(messages):]
    for message in new_messages:
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print('模型调用工具:')
            for tool_call in message.tool_calls:
                print('工具名：', tool_call['name'])
                print('参数：', tool_call.get('args', {}))
        elif message.__class__.__name__ == 'ToolMessage':
            print('工具返回结果：')
            print('工具名', getattr(message, 'name', None))
            print('内容:', message.content)
        else:
            print('模型回复:')
            print(getattr(message, 'content', message))
        print()
    messages[:] = result['messages']
if __name__ == '__main__':
    print('s03: Permission')
    print('输入问题，回车发送。输入 q 退出。\n')
    history = []
    while True:
        try:
            query = input('\x1b[36ms03 >> \x1b[0m')
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ('q', 'exit', ''):
            break
        history.append({'role': 'user', 'content': query})
        agent_loop(history)
