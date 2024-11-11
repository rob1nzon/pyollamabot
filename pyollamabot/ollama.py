import os
import json
import ollama
import asyncio
from ollama import Client
import inspect
from typing import Callable, List, Dict

from pyollamabot.easydiffusion import create_image, poll_image_status


OLLAMA_HOST = os.getenv(key="OLLAMA_HOST")
OLLAMA_MODEL = os.getenv(key="OLLAMA_MODEL")

import inspect
from typing import Callable, List, Dict

import inspect
from typing import Callable, List, Dict
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

# Simulates an API call to get flight times
# In a real application, this would fetch data from a live database or API
def create_picture(description) -> str:
    """Create picture, add object on picture, change picture, with english description

    Args:
        description (str): Translate to ENGLISH description of what you want to draw, on english, more details
    """  
    image64 = ""
    stream=create_image(description)
    image64 = poll_image_status(stream)
    return f'create_picture({description})', image64


async def ask_model(messages):
    model = OLLAMA_MODEL
    image = None
    client = ollama.AsyncClient(host=f'http://{OLLAMA_HOST}:11434')

    # First API call: Send the query and function description to the model
    response = await client.chat(
        model=model,
        messages=messages,
        tools=functions_to_metadata([create_picture])
    )

    # Add the model's response to the conversation history
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
        }
        for tool in response['message']['tool_calls']:
            function_to_call = available_functions[tool['function']['name']]
            function_response, image = function_to_call(tool['function']['arguments']['description'])
            # Add function response to the conversation
            messages.append({
                'role': 'tool',
                'content': function_response,
            })
            full_response += f"```{function_response}```"

    # Second API call: Get final response from the model
    final_response = await client.chat(model=model, messages=messages)
    full_response += final_response['message']['content']

    return full_response, image
