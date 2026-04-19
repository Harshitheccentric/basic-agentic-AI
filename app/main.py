import argparse
import os
import sys
import subprocess

from openai import OpenAI
import json
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def read(file_path: str) -> str:
    with open(file=file_path, mode='r') as file:
        # print(f"READING {file_path}...")
        file_content = file.read()
        return file_content
    

def write(file_path: str, content: str) -> str:
    with open(file=file_path, mode='w') as file:
        file.write(content)
        return content


def bash(command: str) -> str:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout= 60)
        return result.stdout if result.returncode == 0 else result.stderr
    except subprocess.TimeoutExpired:
        return "Command timed out (60s)"
    

FUNCTIONS = {
    'Read': read,
    'Write': write,
    'Bash' : bash
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    global_messages = [{"role": "user", "content": args.p}]

    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            # model="z-ai/glm-4.5-air:free",
            messages=global_messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "description": "Read and return the contents of a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "The path to the file to read"
                                }
                            },
                            "required": ["file_path"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "Write",
                        "description": "Write content to a file",
                        "parameters": {
                            "type": "object",
                            "required": ["file_path", "content"],
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "The path of the file to write to"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The content to write to the file"
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "description": "Execute a shell command",
                        "parameters": {
                            "type": "object",
                            "required": ["command"],
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "The command to execute"
                                }
                            }
                        }
                    }
                }
            ]
        )

        # print(chat.choices[0])

        global_messages.append({
            'role': 'assistant', 
            'content': chat.choices[0].message.content, 
            'tool_calls': chat.choices[0].message.tool_calls
        })
        
        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        # You can use print statements as follows for debugging, they'll be visible when running tests.
        print("Logs from your program will appear here!", file=sys.stderr)

        # TODO: Uncomment the following line to pass the first stage
        
        if chat.choices[0].message.tool_calls:
            for tool in chat.choices[0].message.tool_calls:
                tool_name = tool.function.name
                tool_args = json.loads(tool.function.arguments)
                # print('tool-name:',tool_name)
                # print('tool-args:',tool_args)
                content = FUNCTIONS[tool_name](**tool_args)
                global_messages.append({
                    "role": "tool",
                    "tool_call_id": tool.id,
                    "content": content 
                })
        
        else:
            print(chat.choices[0].message.content)
            break

if __name__ == "__main__":
    main()
