import os
try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
    readline.parse_and_bind('set enable-meta-keybindings on')
except ImportError:
    pass
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import StructuredTool
load_dotenv(override=True)
MODEL = os.environ['MODEL_ID']
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('BASE_URL')
WORKDIR = Path.cwd()
SYSTEM = f'you are a coding agent at {WORKDIR}. Use bash to solve tasks. Act dont explain'

# 安全边界：shell=True 仅为教学演示，黑名单/路径检查不等于安全边界；生产请使用权限中间件 + 沙箱。
def run_bash(command: str) -> str:
    dangerous = ['rm -rf /', 'sudo', 'shutdown', 'reboot', '>/dev/']
    if any((d in command for d in dangerous)):
        return 'Error : Dangerous command blocked'
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else '(no output)'
    except subprocess.TimeoutExpired:
        return 'Error: Timeout(120s)'
    except OSError as e:
        return f'Error: {e}'
bash_tool = StructuredTool.from_function(func=run_bash, name='bash', description='Run a shell command')
TOOLS = [bash_tool]

def build_chat_model():
    kwargs = {'model': MODEL, 'max_completion_tokens': 8000, 'temperature': 0}
    if OPENAI_API_KEY:
        kwargs['api_key'] = OPENAI_API_KEY
    if OPENAI_BASE_URL:
        kwargs['base_url'] = OPENAI_BASE_URL
    return ChatOpenAI(**kwargs)
chat_model = build_chat_model()
agent = create_agent(model=chat_model, tools=TOOLS, system_prompt=SYSTEM)

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

def print_tool_activity(message: AIMessage | ToolMessage) -> None:
    if isinstance(message, AIMessage):
        for tool_call in message.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call.get('args', {})
            command = tool_args.get('command', '')
            if tool_name == 'bash':
                print(f'\x1b[33m$ {command}\x1b[0m')
        return
    if isinstance(message, ToolMessage):
        print(str(message.content)[:200])

def agent_loop(messages: list) -> None:
    result = agent.invoke({'messages': messages})
    new_messages = result['messages'][len(messages):]
    for message in new_messages:
        print_tool_activity(message)
    messages[:] = result['messages']
if __name__ == '__main__':
    history = []
    while True:
        try:
            query = input('\x1b[36ms01 >> \x1b[0m')
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ('q', 'exit', ''):
            break
        history.append(HumanMessage(content=query))
        agent_loop(history)
        last_message = history[-1]
        if isinstance(last_message, AIMessage):
            print_assistant_message(last_message)
        print()
