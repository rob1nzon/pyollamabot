import logging
import os
import json
import ollama
import asyncio
from ollama import Client
import inspect
from typing import Callable, List, Dict, Any
import io
from contextlib import redirect_stdout
from pyollamabot.search import search_internet
from scapy.all import (
    IP, TCP, UDP, ICMP, 
    Ether, ARP, DNS, Raw,
    sr, sr1, srp, srp1,
    sniff, send, sendp,
    wrpcap, rdpcap,
    hexdump, ls
)

from pyollamabot.easydiffusion import create_image, poll_image_status

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# Global history storage
chat_history: List[Dict[str, Any]] = []

OLLAMA_HOST = os.getenv(key="OLLAMA_HOST")
OLLAMA_MODEL = os.getenv(key="OLLAMA_MODEL")

from docstring_parser import parse

def functions_to_metadata(functions: List[Callable]) -> List[Dict]:
    """
    Extracts function metadata from a list of functions and returns it in the desired format.

    Args:
        functions (List[Callable]): List of functions to extract metadata from.

    Returns:
        List[Dict]: List of function metadata in the desired format.
    """
    metadata = []
    for func in functions:
        # Get the function docstring
        doc = inspect.getdoc(func)
        if not doc:
            # If the function has no docstring, skip it
            continue

        # Parse the docstring using the docstring-parser library
        parsed_doc = parse(doc)

        # Extract the relevant metadata
        metadata.append({
            'type': 'function',
            'function': {
                'name': func.__name__,
                'description': parsed_doc.short_description,
                'parameters': {
                    'type': 'object',
                    'properties': {
                        param.arg_name: {
                            'type': str(param.type_name),
                            'description': param.description
                        }
                        for param in parsed_doc.params
                    },
                    'required': [param.arg_name for param in parsed_doc.params]
                }
            }
        })

    return metadata

def analyze_chat_history(query: str = None) -> str:
    """Analyze chat history with optional filtering by query

    Args:
        query (str, optional): Search term to filter messages
    """


    global chat_history

    logger.debug("analyze")
    
    filtered_history = chat_history
    if query:
        filtered_history = [
            msg for msg in filtered_history 
            if query.lower() in str(msg.get('content', '')).lower()
        ]
    logger.debug(filtered_history)
    msg_parts = []
    for message in filtered_history:
        # Для сообщений-словарей (обычные сообщения)
        if isinstance(message, dict):
            # Проверяем наличие всех необходимых ключей
            if 'user' in message and 'content' in message:
                user_name = message['user'].get('name', 'Unknown') if isinstance(message['user'], dict) else 'Unknown'
                content = message['content']
                msg_parts.append(f"{user_name}: {content}")
        
        # Для объектов Message (ответы бота)
        else:
            # Используем getattr с дефолтными значениями на случай отсутствия атрибутов
            user_name = getattr(message.user, 'name', 'Unknown') if hasattr(message, 'user') else 'Unknown'
            content = getattr(message, 'content', '')
            if content:  # Пропускаем пустые сообщения
                msg_parts.append(f"{user_name}: {content}")

    log_message = ' '.join(msg_parts)
    return log_message, None

def get_chat_history(limit: int = None) -> str:
    """Get recent chat history with optional limit

    Args:
        limit (int, optional): Number of most recent messages to return
    """
    global chat_history

    logger.debug("get chat history")
   
    
    if limit:
        try:
            limit = int(limit)
            history = chat_history[-limit:]
        except ValueError:
            return "Invalid limit value. Must be an integer", None
    else:
        history = chat_history
    logger.debug(history)
    msg_parts = []
    for message in history:
        # Для сообщений-словарей (обычные сообщения)
        if isinstance(message, dict):
            # Проверяем наличие всех необходимых ключей
            if 'user' in message and 'content' in message:
                user_name = message['user'].get('name', 'Unknown') if isinstance(message['user'], dict) else 'Unknown'
                content = message['content']
                msg_parts.append(f"{user_name}: {content}")
        
        # Для объектов Message (ответы бота)
        else:
            # Используем getattr с дефолтными значениями на случай отсутствия атрибутов
            user_name = getattr(message.user, 'name', 'Unknown') if hasattr(message, 'user') else 'Unknown'
            content = getattr(message, 'content', '')
            if content:  # Пропускаем пустые сообщения
                msg_parts.append(f"{user_name}: {content}")

    log_message = ' '.join(msg_parts)
    return log_message, None

def execute_python(code: str) -> str:
    """Execute Python code in a safe environment and return the output. Use scapy for network

    Args:
        code (str): Python code to execute
    """
    # Create string buffer to capture output
    f = io.StringIO()
    result = ""
    
    try:
        # Execute code with restricted globals and capture stdout
        with redirect_stdout(f):
            # Create safe globals dictionary
            safe_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'round': round,
                    'abs': abs,
                    'sum': sum,
                    'min': min,
                    'max': max,
                    'sorted': sorted,
                    'reversed': reversed,
                },
                # Add scapy classes and functions
                'scapy': {
                    'IP': IP,
                    'TCP': TCP,
                    'UDP': UDP,
                    'ICMP': ICMP,
                    'Ether': Ether,
                    'ARP': ARP,
                    'DNS': DNS,
                    'Raw': Raw,
                    'sr': sr,
                    'sr1': sr1,
                    'srp': srp,
                    'srp1': srp1,
                    'sniff': sniff,
                    'send': send,
                    'sendp': sendp,
                    'wrpcap': wrpcap,
                    'rdpcap': rdpcap,
                    'hexdump': hexdump,
                    'ls': ls
                }
            }
            
            # Execute the code
            exec(code, safe_globals, {})
        
        # Get output
        output = f.getvalue()
        if output:
            result = f"Output:\n{output}"
        else:
            result = "Code executed successfully (no output)"
            
    except Exception as e:
        result = f"Error: {str(e)}"
    
    return result, None

def create_picture(description) -> str:
    """Create picture, add object on picture, change picture, if change remember history. Use english

    Args:
        description (str): Translate to english and description of what you want to draw, more details
    """  
    image64 = ""
    stream=create_image(description)
    image64 = poll_image_status(stream)
    return f'create_picture({description})', image64

async def ask_model(messages: List[Dict[str, Any]]):
    model = OLLAMA_MODEL
    image = None
    client = ollama.AsyncClient(host=f'http://{OLLAMA_HOST}:11434')

    # First API call: Send the query and function description to the model
    response = await client.chat(
        model=model,
        messages=messages,
        tools=functions_to_metadata([
            create_picture, 
            search_internet, 
            execute_python,
            analyze_chat_history,
            get_chat_history
        ])
    )

    # Store in global history
    chat_history.append(response['message'])
    messages.append(response['message'])

    full_response = ''
    # Check if the model decided to use the provided function
    if not response['message'].get('tool_calls'):
        full_response += response['message']['content']
        return full_response, image

    # Process function calls made by the model
    if response['message'].get('tool_calls'):
        print('Call function')
        available_functions = {
            'create_picture': create_picture,
            'search_internet': search_internet,
            'execute_python': execute_python,
            'analyze_chat_history': analyze_chat_history,
            'get_chat_history': get_chat_history
        }
        for tool in response['message']['tool_calls']:
            function_to_call = available_functions[tool['function']['name']]
            print(tool)
            function_response, image = function_to_call(**tool['function']['arguments'])
            # Add function response to the conversation
            tool_message = {
                'role': 'tool',
                'content': function_response,
            }
            messages.append(tool_message)
            chat_history.append(tool_message)

    # Second API call: Get final response from the model
    final_response = await client.chat(model=model, messages=messages)
    chat_history.append(final_response['message'])
    full_response += final_response['message']['content']

    return full_response, image
