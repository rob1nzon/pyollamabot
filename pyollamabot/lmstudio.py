import logging
import os
import aiohttp
import base64
import inspect
from typing import Callable, List, Dict, Any
import io
from contextlib import redirect_stdout
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

LMSTUDIO_HOST = os.getenv(key="LMSTUDIO_HOST", default="localhost")
LMSTUDIO_PORT = os.getenv(key="LMSTUDIO_PORT", default="1234")
LMSTUDIO_MODEL = os.getenv(key="LMSTUDIO_MODEL", default="local-model")
LMSTUDIO_MODEL_VISION = os.getenv(key="LMSTUDIO_MODEL_VISION", default="local-model")


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
            msg for msg in chat_history 
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

async def analyze_image(image_data: bytes, question: str = None) -> str:
    """Analyze image using model and answer a question about it

    Args:
        image_data (bytes): Raw image data to analyze
        question (str, optional): Question to ask about the image. If None, will describe the image.
    """
    
    # Convert image to base64
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    # Use the provided question or default to description if none provided
    prompt = question if question else 'Describe this photo in detail'
    
    # Prepare the request payload
    payload = {
        "model": LMSTUDIO_MODEL_VISION,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }
    
    async with aiohttp.ClientSession() as session:
        url = f"http://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}/v1/chat/completions"
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result['choices'][0]['message']['content']
            else:
                raise Exception(f"LM Studio API error: {response.status}")

def create_picture(description) -> str:
    """Create picture, add object on picture, change picture, if change remember history. Use english

    Args:
        description (str): Translate to english and description of what you want to draw, more details
    """  
    image64 = ""
    stream = create_image(description)
    image64 = poll_image_status(stream)
    return f'create_picture({description})', image64

async def ask_model(messages: List[Dict[str, Any]]):
    """Send messages to LM Studio and get response"""
    
    # Prepare the request payload
    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.7,
        "stream": False
    }
    
    async with aiohttp.ClientSession() as session:
        url = f"http://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}/v1/chat/completions"
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                
                # Store in global history
                chat_history.append({
                    'role': 'assistant',
                    'content': result['choices'][0]['message']['content']
                })
                
                return result['choices'][0]['message']['content'], None
            else:
                error_text = await response.text()
                raise Exception(f"LM Studio API error: {response.status} - {error_text}")