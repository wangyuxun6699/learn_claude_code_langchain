from collections.abc import Callable
from typing import Any
from dotenv import load_dotenv
load_dotenv(override=True)
from langchain.agents.middleware import AgentState, before_agent, after_agent, wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from pathlib import Path
from langchain_core.tools import tool
import os, subprocess
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
HOOKS = {'UserPromptSubmit': [], 'PreToolUse': [], 'PostToolUse': [], 'Stop': []}

def register_hook(event: str, callback):
    HOOKS[event].append(callback)

def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None

@before_agent
def user_prompt_submit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    message = state.get('messages', [])
    if message:
        last = message[-1]
        content = last.get('content') if isinstance(last, dict) else getattr(last, 'content', None)
        trigger_hooks('UserPromptSubmit', content)
    return None

@wrap_tool_call
def tool_hook(request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command:
    tool_name = request.tool_call['name']
    tool_args = request.tool_call.get('args', {})
    blockd = trigger_hooks('PreToolUse', tool_name, tool_args)
    if blockd:
        return ToolMessage(content=str(blockd), tool_call_id=request.tool_call['id'], name=tool_name, status='error')
    result = handler(request)
    trigger_hooks('PostToolUse', tool_name, tool_args, result)
    return result

@after_agent
def stop_hook(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    trigger_hooks('Stop', state.get('messages', []))
    return None
WORKDIR = Path.cwd()
MODEL_ID = os.getenv('MODEL_ID')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SYSTEM = f'you are a coding agent at {WORKDIR}. Use tools to solve tasks. Act dont explain'
OPENAI_BASE_URL = os.getenv('BASE_URL')
dangerous = ['rm -rf /', 'sudo', 'shutdown', 'reboot', 'mkfs', 'dd if=', '> /dev/sda']

def check_deny_list(command: str) -> str | None:
    for pattern in dangerous:
        if pattern in command:
            return f'blocked:{pattern} is on the deny list'
    return None

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
            print(f'\nBlocked:{reason}')
            return False
    reason = check_rules(tool_name, args)
    if reason:
        return ask_user(tool_name, args, reason)
    return True

def on_user_prompt_submit(content):
    print('[UserPromptSubmit]', content)

def on_pre_tool_use(tool_name, tool_args):
    print('[PreToolUse]', tool_name, tool_args)
    if not check_permission(tool_name, tool_args):
        return 'Permission denied'

def on_post_tool_use(tool_name, tool_args, result):
    print('[PostToolUse]', tool_name, tool_args)
    print('result:', getattr(result, 'content', result))

def on_stop(messages):
    print('[Stop]', len(messages))
register_hook('UserPromptSubmit', on_user_prompt_submit)
register_hook('PreToolUse', on_pre_tool_use)
register_hook('PostToolUse', on_post_tool_use)
register_hook('Stop', on_stop)

@tool
# 安全边界：shell=True 仅为教学演示，黑名单/路径检查不等于安全边界；生产请使用权限中间件 + 沙箱。
def run_bash(command: str) -> str:
    """Execute a shell command in the current workspace."""
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
    try:
        lines = resolve_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f'...({len(lines) - limit} more lines)']
        return '\n'.join(lines)
    except Exception as e:
        return f'Error: {e}'

@tool
def run_write(path: str, content: str) -> str:
    """Write UTF-8 content to a file, replacing its existing content."""
    try:
        file_path = resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f'write {len(content)} bytes to {path}'
    except Exception as e:
        return f'Error: {e}'

@tool
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace the first exact occurrence of old_text in a UTF-8 file."""
    try:
        file_path = resolve_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f'Error: text not found in {path}'
        file_path.write_text(text.replace(old_text, new_text, 1))
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
MIDDLEWARE = [user_prompt_submit, tool_hook, stop_hook]
MODEL = ChatOpenAI(model=MODEL_ID, max_completion_tokens=8000, temperature=0, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
agent = create_agent(model=MODEL, tools=TOOLS, system_prompt=SYSTEM, middleware=MIDDLEWARE)

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
    print('s04: Hook')
    print('输入问题，回车发送。输入 q 退出。\n')
    history = []
    while True:
        try:
            query = input('\x1b[36ms04 >> \x1b[0m')
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ('q', 'exit', ''):
            break
        history.append({'role': 'user', 'content': query})
        agent_loop(history)
