#!/usr/bin/python

# This is a simple echo bot using the decorator mechanism.
# It echoes any incoming text messages.
import asyncio
import os
import random
import time
import logging
import io
import sys
import json
import re
from contextlib import redirect_stdout
import textwrap
from datetime import datetime, timedelta

# Utility functions for message handling
def sanitize_message(text):
    """
    Sanitize message text by replacing angle brackets that aren't part of HTML tags.
    This prevents issues with Telegram's HTML parser.
    """
    if not text:
        return text
    
    # Don't modify HTML tags like <pre>, <code>, <b>, etc.
    # This regex looks for < or > that aren't part of an HTML tag
    # It preserves HTML tags while replacing standalone angle brackets
    html_tags = r'</?(?:pre|code|b|i|u|s|a|em|strong|ins|del|strike|blockquote|sup|sub|span)[^>]*>'
    
    # First, temporarily replace valid HTML tags
    placeholder = "HTMLTAG_PLACEHOLDER_"
    count = 0
    placeholders = {}
    
    def replace_tag(match):
        nonlocal count
        placeholder_key = f"{placeholder}{count}"
        placeholders[placeholder_key] = match.group(0)
        count += 1
        return placeholder_key
    
    # Replace HTML tags with placeholders
    text_with_placeholders = re.sub(html_tags, replace_tag, text, flags=re.IGNORECASE)
    
    # Now replace remaining angle brackets
    text_with_placeholders = text_with_placeholders.replace('<', '&lt;').replace('>', '&gt;')
    
    # Restore HTML tags
    for key, value in placeholders.items():
        text_with_placeholders = text_with_placeholders.replace(key, value)
    
    return text_with_placeholders

def split_long_message(text, max_length=4000):
    """
    Split a long message into smaller chunks to avoid Telegram's message size limits.
    Returns a list of message chunks.
    """
    if not text:
        return [text]
    
    if len(text) <= max_length:
        return [text]
    
    # Split message into chunks
    chunks = []
    
    # Try to split at paragraph boundaries first
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed max_length, start a new chunk
        if len(current_chunk) + len(paragraph) + 2 > max_length:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                # If a single paragraph is too long, split it by sentences or just by length
                if len(paragraph) > max_length:
                    # Split by sentences if possible
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    for sentence in sentences:
                        if len(sentence) > max_length:
                            # If even a sentence is too long, split by max_length
                            for i in range(0, len(sentence), max_length):
                                chunks.append(sentence[i:i+max_length])
                        else:
                            if current_chunk and len(current_chunk) + len(sentence) + 1 <= max_length:
                                current_chunk += " " + sentence
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk)
                                current_chunk = sentence
                else:
                    chunks.append(paragraph)
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# Function to handle special tags like <think> and <reasoning>
def process_special_tags(text):
    """
    Process special tags like <think> and <reasoning> and convert them to Telegram-compatible format.
    
    Args:
        text: The text to process
        
    Returns:
        Processed text with special tags converted
    """
    if not text:
        return text
    
    # Replace special tags with Telegram-compatible tags
    result = text
    result = result.replace('<reasoning>', '<blockquote expandable>').replace('</reasoning>', '</blockquote>')
    result = result.replace('<think>', '<blockquote expandable>').replace('</think>', '</blockquote>')
    result = result.replace('<answer>', '').replace('</answer>', '')
    
    return result

# Helper function for sending messages safely
async def send_message_safely(bot, chat_id, text, message=None, parse_mode=None, **kwargs):
    """
    Send a message safely by sanitizing text and splitting long messages.
    
    Args:
        bot: The bot instance
        chat_id: Chat ID to send the message to
        text: Message text
        message: Optional message object to reply to (not just the ID)
        parse_mode: Optional parse mode (HTML, Markdown, etc.)
        **kwargs: Additional arguments to pass to send_message
    
    Returns:
        The last sent message
    """
    if not text:
        return None
    
    # Process special tags first
    processed_text = process_special_tags(text)
    
    # Sanitize the message to handle angle brackets
    sanitized_text = sanitize_message(processed_text)
    
    # Split long messages
    message_chunks = split_long_message(sanitized_text)
    
    last_message = None
    for i, chunk in enumerate(message_chunks):
        try:
            # For the first chunk, reply to the original message
            # For subsequent chunks, send as regular messages
            if i == 0 and message:
                last_message = await bot.reply_to(
                    message=message,
                    text=chunk,
                    parse_mode=parse_mode,
                    **kwargs
                )
            else:
                last_message = await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode=parse_mode,
                    **kwargs
                )
            
            # Small delay between chunks to ensure order
            if i < len(message_chunks) - 1:
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error sending message chunk {i+1}/{len(message_chunks)}: {e}")
            # If sending fails, try without parse_mode as a fallback
            if parse_mode:
                try:
                    last_message = await bot.send_message(
                        chat_id=chat_id,
                        text=f"Error sending formatted message. Sending without formatting: {chunk}",
                        parse_mode=None
                    )
                except Exception as inner_e:
                    logger.error(f"Failed to send even unformatted message: {inner_e}")
    
    return last_message

from pyollamabot.ollama import ask_model, create_picture
from pyollamabot.transcribe import WyomingTranscriber
from gradio_client import Client

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from telebot.async_telebot import AsyncTeleBot
from telebot.types import Message, ReactionTypeEmoji
import base64
from io import BytesIO
from PIL import Image

# Function to convert markdown code blocks to HTML
def convert_code_blocks_to_html(text):
    if not text:
        return text
    
    # Pattern to match code blocks with triple backticks
    # This will match both with and without language specification
    pattern = r'```(?:(\w+)\n)?(.*?)```'
    
    def replace_code_block(match):
        language = match.group(1) or ''
        code = match.group(2)
        # If language is specified, add it as a class
        lang_attr = f' class="language-{language}"' if language else ''
        return f'<pre><code{lang_attr}>{code}</code></pre>'
    
    # Replace all code blocks in the text
    result = re.sub(pattern, replace_code_block, text, flags=re.DOTALL)
    logger.debug(f"Converted code blocks in text: {result}")
    return result

token = os.getenv(key="TOKEN")
bot = AsyncTeleBot(token)

# Handle '/start' and '/help'
@bot.message_handler(commands=['help', 'start'])
async def send_welcome(message):
    text = '''Hi, I am EchoBot.
Just write me something and I will repeat it!

Команды для каканья 💩:
/show - показать статистику каканья
/streaks - показать активные стрики
/test_reminder - тестовое напоминание о стриках

Просто напишите "я покакал" чтобы отметиться!'''
    await send_message_safely(bot, message.chat.id, text, message)

@bot.message_handler(commands=['show'])
async def show_poop_stats(message):
    data = load_poop_counter()
    if not data["counters"]:
        await send_message_safely(bot, message.chat.id, "Пока никто не какал 💩", message)
        return
    
    stats = []
    for user_id, info in data["counters"].items():
        streak = info.get('current_streak', 0)
        streak_text = f" (стрик: {streak})" if streak > 0 else ""
        stats.append(f"{info['username']}: {info['count']} раз{streak_text}")
    
    response = "Статистика каканья 💩:\n" + "\n".join(stats)
    await send_message_safely(bot, message.chat.id, response, message)

@bot.message_handler(commands=['streaks'])
async def show_streak_stats(message):
    data = load_poop_counter()
    if not data["counters"]:
        await send_message_safely(bot, message.chat.id, "Пока никто не какал 💩", message)
        return
    
    # Sort by current streak (descending)
    sorted_users = sorted(
        data["counters"].items(),
        key=lambda x: x[1].get('current_streak', 0),
        reverse=True
    )
    
    stats = []
    today = get_today_date()
    
    for user_id, info in sorted_users:
        streak = info.get('current_streak', 0)
        if streak > 0:
            # Check if pooped today
            poop_dates = info.get('poop_dates', [])
            pooped_today = today in poop_dates
            status = "✅" if pooped_today else "⚠️"
            stats.append(f"{status} {info['username']}: {streak} дней подряд")
    
    if not stats:
        response = "Ни у кого нет активного стрика 💩"
    else:
        response = "Стрики каканья 🔥💩:\n" + "\n".join(stats)
        response += "\n\n✅ - покакал сегодня\n⚠️ - еще не какал сегодня"
    
    await send_message_safely(bot, message.chat.id, response, message)

@bot.message_handler(commands=['test_reminder'])
async def test_reminder(message):
    """Test the streak reminder functionality"""
    await send_streak_reminders()
    await send_message_safely(bot, message.chat.id, "Тестовое напоминание отправлено (если есть пользователи со стриками, которые не какали сегодня)", message)

@bot.message_handler(commands=['generate_image'])
async def generate_image(message):
	msg = message.text.replace('/generate_image', '').strip()
	_, image = create_picture(msg)
	await bot.send_chat_action(chat_id=message.chat.id, action='upload_photo')
	image_data = base64.b64decode(image.split(',')[1])
	image_bin = Image.open(BytesIO(image_data))
	await bot.send_photo(chat_id=message.chat.id,reply_to_message_id=message.id, photo=image_bin, has_spoiler=True)

@bot.message_handler(commands=['web_use'])
async def generate_image(message):
    msg = message.text.replace('/web_use', '').strip()
    client = Client("http://host.docker.internal:7788/")
    result = client.predict(
        agent_type="custom",
        llm_provider="openai",
        llm_model_name="nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
        llm_num_ctx=16000,
        llm_temperature=0.6,
        llm_base_url="",
        llm_api_key="",
        use_own_browser=False,
        keep_browser_open=False,
        headless=False,
        disable_security=True,
        window_w=1280,
        window_h=1100,
        save_recording_path="./tmp/record_videos",
        save_agent_history_path="./tmp/agent_history",
        save_trace_path="./tmp/traces",
        enable_recording=True,
        task=msg,
        add_infos="",
        max_steps=100,
        use_vision=False,
        max_actions_per_step=10,
        tool_calling_method="auto",
        chrome_cdp="",
        max_input_tokens=128000,
        api_name="/run_with_stream"
    )
	# await bot.send_chat_action(chat_id=message.chat.id, action='upload_photo')
    logger.debug(result)
    answer =result[1]
    agent_gif = result[5].split("=")[-1]
    image_url = f"http://host.docker.internal:7788/gradio_api/file={agent_gif}"
    
    # Download the GIF from the URL using aiohttp
    import aiohttp
    from io import BytesIO
    
    try:
        # Download the GIF
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download GIF: HTTP {response.status}")
                
                # Read the content
                content = await response.read()
                
                # Create a BytesIO object from the downloaded content
                gif_data = BytesIO(content)
        
        # Send the video note
        await bot.send_video_note(message.chat.id, gif_data)
        
        # Also send the answer text if available
        if answer:
            await send_message_safely(bot, message.chat.id, answer, message, parse_mode='HTML')
    except Exception as e:
        logger.error(f"Error downloading or sending GIF: {e}", exc_info=True)
        await send_message_safely(bot, message.chat.id, f"Error processing GIF: {str(e)}", message)
    
	# image_data = base64.b64decode(image.split(',')[1])
	# image_bin = Image.open(BytesIO(image_data))
	# await bot.send_photo(chat_id=message.chat.id,reply_to_message_id=message.id, photo=image_bin, has_spoiler=True)


def role(msg):
	if msg.is_bot:
		return "assistant"
	return "user"

@bot.message_handler(content_types=['photo'])
async def photo(message):
    try:
        bot_info = await bot.get_me()
        mention = (message.reply_to_message and message.reply_to_message.from_user.username == bot_info.username)
        direct = message.caption and message.caption.startswith(f'@{bot_info.username}')

        rand_value = random.randint(0, 200)
        if not mention and not direct and rand_value != 42:
            return

        await bot.send_chat_action(chat_id=message.chat.id, action='typing')
        
        # Get caption if it exists
        caption = message.caption.replace(f'@{bot_info.username}', '').strip() if message.caption else ""
        caption_reply = ""
        # If replying to a message with photo, use that photo instead
        if message.reply_to_message and message.reply_to_message.photo:
            fileID = message.reply_to_message.photo[-1].file_id
            caption_reply = message.reply_to_message.caption
        else:
            fileID = message.photo[-1].file_id
        file_info = await bot.get_file(fileID)
        downloaded_file = await bot.download_file(file_info.file_path)

        # Analyze image using moondream
        from pyollamabot.ollama import analyze_image
        # Get caption if it exists to use as a question
        caption = message.caption.replace(f'@{bot_info.username}', '').strip() if message.caption else None
        image_description = await analyze_image(downloaded_file, caption)

        # Store the image analysis in history
        if not history.get(message.chat.id):
            history[message.chat.id] = []
        history[message.chat.id].append([message.from_user.first_name, f"[Sent a photo: {image_description}]"])

        # Store in ollama's global history
        from pyollamabot.ollama import chat_history
        chat_history.append({
            'role': role(message.from_user),
            'content': f"[Photo: {image_description}]",
            'user': {
                'id': message.from_user.id,
                'name': message.from_user.first_name
            },
            'chat_id': message.chat.id,
            'message_id': message.message_id,
            'type': 'photo'
        })

        # Process the image description with the main model
        bot_info = await bot.get_me()
        # Prepare message content based on whether it's a reply to a photo or a new photo
        if message.reply_to_message and message.reply_to_message.photo:
            messages = [{'role': 'user', 'content': f"Пользователь спрашивает{' с подписью: ' + caption if caption else ''} {' с подписью: ' + caption_reply if caption_reply else ''} про фотографию. Твой ответ: {image_description}"}]
        else:
            messages = [{'role': 'user', 'content': f"Пользователь отправил фотографию{' с подписью: ' + caption if caption else ''} {' с подписью: ' + caption_reply if caption_reply else ''}. Твой ответ: {image_description}"}]
        
        await bot.send_chat_action(chat_id=message.chat.id, action='typing')
        answer, _ = await ask_model(messages=messages)
        
        if answer and not 'im_start' in answer:
            # Convert code blocks to HTML
            answer = convert_code_blocks_to_html(answer)
            await send_message_safely(bot, message.chat.id, answer, message, parse_mode='HTML')
            # Store bot's response in histories
            history[message.chat.id].append([bot_info.first_name, answer])
            chat_history.append({
                'role': 'assistant',
                'content': answer,
                'chat_id': message.chat.id,
                'message_id': message.message_id,
                'type': 'photo_response'
            })
            
    except Exception as e:
        logger.error(f"Error processing photo: {e}", exc_info=True)
        await send_message_safely(bot, message.chat.id, 'Произошла ошибка при обработке фотографии.', message)

history = {}

def load_poop_counter():
    file_path = '/app/poop_counter.json'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Ensure we have the chat_ids field
            if 'chat_ids' not in data:
                data['chat_ids'] = []
            return data
    except FileNotFoundError:
        return {"counters": {}, "chat_ids": []}

def save_poop_counter(data):
    file_path = '/app/poop_counter.json'
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.debug(f"Successfully saved poop counter data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving poop counter data: {e}")

def get_today_date():
    """Get today's date in YYYY-MM-DD format (Moscow timezone)"""
    moscow_tz = datetime.now() + timedelta(hours=3)  # UTC+3 Moscow time
    return moscow_tz.strftime('%Y-%m-%d')

def calculate_streak(user_data):
    """Calculate current streak for a user"""
    if 'poop_dates' not in user_data:
        return 0
    
    poop_dates = sorted(user_data['poop_dates'])
    if not poop_dates:
        return 0
    
    today = get_today_date()
    current_streak = 0
    
    # Convert string dates to datetime objects for easier calculation
    date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in poop_dates]
    today_obj = datetime.strptime(today, '%Y-%m-%d')
    
    # Start from today and go backwards
    check_date = today_obj
    
    while True:
        check_date_str = check_date.strftime('%Y-%m-%d')
        if check_date_str in poop_dates:
            current_streak += 1
            check_date -= timedelta(days=1)
        else:
            break
    
    return current_streak

def add_chat_id(chat_id):
    """Add chat ID to the list of active chats"""
    data = load_poop_counter()
    if chat_id not in data["chat_ids"]:
        data["chat_ids"].append(chat_id)
        save_poop_counter(data)

def increment_poop_counter(user_id, username):
    data = load_poop_counter()
    today = get_today_date()
    
    if str(user_id) not in data["counters"]:
        data["counters"][str(user_id)] = {
            "count": 0,
            "username": username,
            "poop_dates": [],
            "current_streak": 0
        }
    
    user_data = data["counters"][str(user_id)]
    
    # Initialize missing fields for existing users
    if 'poop_dates' not in user_data:
        user_data['poop_dates'] = []
    if 'current_streak' not in user_data:
        user_data['current_streak'] = 0
    
    # Only increment if not already pooped today
    if today not in user_data['poop_dates']:
        user_data["count"] += 1
        user_data['poop_dates'].append(today)
        
        # Calculate new streak
        user_data['current_streak'] = calculate_streak(user_data)
    
    user_data["username"] = username  # Update username in case it changed
    save_poop_counter(data)
    return user_data["count"], user_data['current_streak']

@bot.message_handler(content_types=['voice', 'video_note'])
async def voice(message):
    try:
        logger.debug(f"Received voice message from user {message.from_user.id}")
        await bot.send_chat_action(chat_id=message.chat.id, action='typing')
        
        # Download voice/video file
        file_id = message.voice.file_id if message.content_type == 'voice' else message.video_note.file_id
        logger.debug(f"Getting file info for {message.content_type} message {file_id}")
        file_info = await bot.get_file(file_id)
        logger.debug(f"Downloading file from path: {file_info.file_path}")
        file_data = await bot.download_file(file_info.file_path)
        logger.debug(f"Downloaded file, size: {len(file_data)} bytes")
        
        # Save temporary file
        file_ext = "ogg" if message.content_type == 'voice' else "mp4"
        temp_input = f"temp_input.{file_ext}"
        temp_output = "temp_output.wav"
        with open(temp_input, "wb") as f:
            f.write(file_data)
            
        # Convert to WAV format using ffmpeg
        logger.debug("Converting to WAV format")
        ffmpeg_cmd = [
            'ffmpeg', '-i', temp_input,
            '-ar', '16000',  # Sample rate
            '-ac', '1',      # Mono
            '-acodec', 'pcm_s16le'  # 16-bit PCM
        ]
        # For video, extract only audio
        if message.content_type == 'video_note':
            ffmpeg_cmd.extend(['-vn'])  # Disable video processing
        ffmpeg_cmd.append(temp_output)
        
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"FFmpeg conversion failed: {stderr.decode()}")
            await send_message_safely(bot, message.chat.id, 'Ошибка при конвертации аудио.', message)
            return
            
        # Read converted audio
        with open(temp_output, "rb") as f:
            converted_audio = f.read()
            
        # Clean up temporary files
        try:
            os.remove(temp_input)
            os.remove(temp_output)
        except Exception as e:
            logger.error(f"Error removing temporary files: {e}")
        
        # Transcribe voice
        logger.debug("Initializing transcriber")
        transcriber = WyomingTranscriber()
        logger.debug("Starting transcription")
        text = await transcriber.transcribe(converted_audio)
        logger.debug(f"Transcription result: {text}")
        
        if not text:
            logger.warning("Transcription failed or returned empty text")
            await send_message_safely(bot, message.chat.id, 'Не удалось распознать голосовое сообщение. Попробуйте еще раз или отправьте текст.', message)
            return
		
        # Process special tags and convert code blocks to HTML
        logger.debug(f"Original transcription: {text}")
        text = process_special_tags(text)
        text = convert_code_blocks_to_html(text)
        logger.debug(f"Converted transcription: {text}")
        await send_message_safely(bot, message.chat.id, text, message, parse_mode='HTML')
        
        # Store transcribed message in histories
        if not history.get(message.chat.id):
            history[message.chat.id] = []
        history[message.chat.id].append([message.from_user.first_name, text])
        
        # Store in ollama's global history
        from pyollamabot.ollama import chat_history
        chat_history.append({
            'role': role(message.from_user),
            'content': text,
            'user': {
                'id': message.from_user.id,
                'name': message.from_user.first_name
            },
            'chat_id': message.chat.id,
            'message_id': message.message_id,
            'type': 'voice'
        })

        rand_value = random.randint(0, 100)
        if  rand_value != 42:
            return
	
        # Process transcribed text like regular messages
        logger.debug("Processing transcribed text")
        bot_info = await bot.get_me()
        messages = [{'role': f'{role(message.from_user)}', 'content': text}]
        
        await bot.send_chat_action(chat_id=message.chat.id, action='typing')
        logger.debug("Sending transcribed text to model")
        answer, _ = await ask_model(messages=messages)
        logger.debug(f"Raw model response: {answer}")
        logger.debug(f"Model response preview: {answer[:100]}...")
        
        if answer and not 'im_start' in answer:
                logger.debug("Sending text response")
                # Process special tags and convert code blocks to HTML
                logger.debug(f"Original message: {answer}")
                answer = process_special_tags(answer)
                answer = convert_code_blocks_to_html(answer)
                logger.debug(f"Converted message: {answer}")
                await send_message_safely(bot, message.chat.id, answer, message, parse_mode='HTML')
                # Store bot's response in histories
                history[message.chat.id].append([bot_info.first_name, answer])
                chat_history.append({
                    'role': 'assistant',
                    'content': answer,
                    'chat_id': message.chat.id,
                    'message_id': message.message_id,
                    'type': 'voice_response'
                })
                
    except Exception as e:
        logger.error(f"Error processing voice message: {e}", exc_info=True)
        await send_message_safely(bot, message.chat.id, 'Произошла ошибка при обработке голосового сообщения.', message)

# Handle all other messages with content_type 'text' (content_types defaults to ['text'])
@bot.message_handler(func=lambda message: True)
async def echo_message(message: Message):
	bot_info = await bot.get_me()
	if not history.get(message.chat.id):
		history[message.chat.id] = []
	history[message.chat.id].append([message.from_user.first_name, message.text])
	
	# Store in ollama's global history
	from pyollamabot.ollama import chat_history
	chat_history.append({
		'role': role(message.from_user),
		'content': message.text,
		'user': {
			'id': message.from_user.id,
			'name': message.from_user.first_name
		},
		'chat_id': message.chat.id,
		'message_id': message.message_id
	})

	if len(history[message.chat.id]) > 10000:
		msg_history = ' '.join([f'{user}:{msg}' for user, msg in history[message.chat.id]])
		history[message.chat.id] = []
		messages = []

	msg = message.text.replace(f'@{bot_info.username}', '').strip()
	
	# Track this chat ID for reminders
	add_chat_id(message.chat.id)
	
	# Check for "я покакал" message first, before any other processing
	logger.debug(f"Checking poop message. Original text: '{message.text}', Cleaned msg: '{msg}'")
	if "покакал" in msg.lower():
		logger.debug(f"Poop detected for user {message.from_user.first_name} (ID: {message.from_user.id})")
		count, streak = increment_poop_counter(message.from_user.id, message.from_user.first_name)
		logger.debug(f"Updated poop count for {message.from_user.first_name}: {count}, streak: {streak}")
		
		if streak > 1:
			response = f"{message.from_user.first_name} покакала уже {count} раз и продолжает! 🔥 Стрик: {streak} дней подряд!"
		else:
			response = f"{message.from_user.first_name} покакала уже {count} раз и продолжает"
		
		await send_message_safely(bot, message.chat.id, response)
		return

	mention = (message.reply_to_message and message.reply_to_message.from_user.username == bot_info.username)
	direct = message.text.startswith(f'@{bot_info.username}')

	rand_value = random.randint(0, 200)
	if not mention and not direct and rand_value != 42:
		return
	await bot.send_chat_action(chat_id=message.chat.id, action='typing')
	

	# Check if replying to a message with photo
	if message.reply_to_message and message.reply_to_message.photo:
		# Download and analyze the photo
		fileID = message.reply_to_message.photo[-1].file_id
		file_info = await bot.get_file(fileID)
		downloaded_file = await bot.download_file(file_info.file_path)

		# Analyze image using moondream
		from pyollamabot.ollama import analyze_image
		# Use the user's message as the question about the image
		image_description = await analyze_image(downloaded_file, msg)
		caption_reply = message.reply_to_message.caption
		# Store the image analysis in history with the user's question
		if not history.get(message.chat.id):
			history[message.chat.id] = []
		history[message.chat.id].append([message.from_user.first_name, f"[Asked: '{msg}' about photo: {image_description}]"])

		# Store in ollama's global history with the user's question
		chat_history.append({
			'role': role(message.from_user),
			'content': f"[Question: '{msg}' about photo: {image_description}]",
			'user': {
				'id': message.from_user.id,
				'name': message.from_user.first_name
			},
			'chat_id': message.chat.id,
			'message_id': message.message_id,
			'type': 'photo_question'
		})

		messages = [{'role': 'user', 'content': f"Пользователь спрашивает: {msg} про фотографию. Твой ответ: {image_description}"}]
		answer, _ = await ask_model(messages=messages)
		logger.debug(f"Raw model response: {answer}")
		if answer and not 'im_start' in answer:
			# Process special tags and convert code blocks to HTML
			logger.debug(f"Original message: {answer}")
			answer = process_special_tags(answer)
			answer = convert_code_blocks_to_html(answer)
			logger.debug(f"Converted message: {answer}")
			await send_message_safely(bot, message.chat.id, answer, message, parse_mode='HTML')
			# Store bot's response in histories
			history[message.chat.id].append([bot_info.first_name, answer])
			chat_history.append({
				'role': 'assistant',
				'content': answer,
				'chat_id': message.chat.id,
				'message_id': message.message_id,
				'type': 'photo_question_response'
			})
	else:
		if rand_value == 42:
			msg = f"Ты приятель пользователя, он говорит тебе: {msg}. Что бы ты ему ответил в неформальном стиле. Подшути над ним. Коротко. В одно предложение. {msg}"

		for _ in range(10):
			messages = []
			if message.reply_to_message:
				messages.append({'role': f'{role(message.reply_to_message.from_user)}', 'content': f'{message.reply_to_message.text}'})
			messages.append({'role': f'{role(message.from_user)}', 'content': f'{msg}'})
			answer, image = await ask_model(messages=messages)
			logger.debug(f"Raw model response: {answer}")
			if answer and not 'im_start' in answer:
				if image:
					await bot.send_chat_action(chat_id=message.chat.id, action='upload_photo')
					image_data = base64.b64decode(image.split(',')[1])
					image_bin = Image.open(BytesIO(image_data))
					# Process special tags and convert code blocks to HTML
					logger.debug(f"Original photo caption: {answer}")
					answer = process_special_tags(answer)
					answer = convert_code_blocks_to_html(answer)
					logger.debug(f"Converted photo caption: {answer}")
					# Sanitize caption but don't split it (Telegram has a smaller limit for captions)
					sanitized_caption = sanitize_message(answer)
					await bot.send_photo(chat_id=message.chat.id, reply_to_message_id=message.id, photo=image_bin, caption=sanitized_caption, parse_mode='HTML')
				else:
					# Process special tags and convert code blocks to HTML
					logger.debug(f"Original message: {answer}")
					answer = process_special_tags(answer)
					answer = convert_code_blocks_to_html(answer)
					logger.debug(f"Converted message: {answer}")
					await send_message_safely(bot, message.chat.id, answer, message, parse_mode='HTML')
					# Store bot's response in histories
					history[message.chat.id].append([bot_info.first_name, answer])
					chat_history.append({
						'role': 'assistant',
						'content': answer,
						'chat_id': message.chat.id,
						'message_id': message.message_id
					})
				break
			time.sleep(2)


# Background task for streak reminders
async def streak_reminder_task():
    """Background task that runs every hour to check for streak reminders"""
    while True:
        try:
            await asyncio.sleep(3600)  # Check every hour
            
            # Get current Moscow time
            moscow_time = datetime.now() + timedelta(hours=3)
            current_hour = moscow_time.hour
            
            # Send reminders at 22:00 (10 PM) Moscow time - 2 hours before midnight
            if current_hour == 22:
                await send_streak_reminders()
                
        except Exception as e:
            logger.error(f"Error in streak reminder task: {e}", exc_info=True)

async def send_streak_reminders():
    """Send reminders to users with active streaks who haven't pooped today"""
    try:
        data = load_poop_counter()
        if not data["counters"]:
            return
        
        today = get_today_date()
        reminders_needed = []
        
        for user_id, info in data["counters"].items():
            streak = info.get('current_streak', 0)
            if streak > 0:
                # Check if pooped today
                poop_dates = info.get('poop_dates', [])
                pooped_today = today in poop_dates
                
                if not pooped_today:
                    reminders_needed.append({
                        'user_id': user_id,
                        'username': info['username'],
                        'streak': streak
                    })
        
        if reminders_needed:
            # Sort by streak (highest first)
            reminders_needed.sort(key=lambda x: x['streak'], reverse=True)
            
            # Create reminder message
            reminder_text = "⚠️ НАПОМИНАНИЕ О СТРИКЕ! ⚠️\n\n"
            reminder_text += "Следующие пользователи имеют активный стрик, но еще не какали сегодня:\n\n"
            
            for reminder in reminders_needed:
                reminder_text += f"🔥 {reminder['username']}: {reminder['streak']} дней подряд\n"
            
            reminder_text += f"\n⏰ До конца дня осталось 2 часа!\nНе потеряйте свой стрик! 💩"
            
            # Send to all tracked chats
            chat_ids = data.get("chat_ids", [])
            for chat_id in chat_ids:
                try:
                    await bot.send_message(chat_id, reminder_text)
                    logger.info(f"Sent streak reminder to chat {chat_id}")
                except Exception as e:
                    logger.error(f"Failed to send reminder to chat {chat_id}: {e}")
            
            logger.info(f"Streak reminders sent for {len(reminders_needed)} users to {len(chat_ids)} chats")
            
    except Exception as e:
        logger.error(f"Error sending streak reminders: {e}", exc_info=True)

# Start the bot and background tasks
async def main():
    # Start the streak reminder task
    reminder_task = asyncio.create_task(streak_reminder_task())
    
    # Start bot polling
    try:
        await bot.polling()
    finally:
        reminder_task.cancel()

asyncio.run(main())
