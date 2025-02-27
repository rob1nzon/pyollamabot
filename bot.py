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
from contextlib import redirect_stdout
from pyollamabot.ollama import ask_model, create_picture
from pyollamabot.transcribe import WyomingTranscriber

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from telebot.async_telebot import AsyncTeleBot
from telebot.types import Message, ReactionTypeEmoji
import base64
from io import BytesIO
from PIL import Image

token = os.getenv(key="TOKEN")
bot = AsyncTeleBot(token)

# Handle '/start' and '/help'
@bot.message_handler(commands=['help', 'start'])
async def send_welcome(message):
    text = 'Hi, I am EchoBot.\nJust write me something and I will repeat it!\nUse /выполни to execute Python code.'
    await bot.reply_to(message, text)

@bot.message_handler(commands=['show'])
async def show_poop_stats(message):
    data = load_poop_counter()
    if not data["counters"]:
        await bot.reply_to(message, "Пока никто не какал 💩")
        return
    
    stats = []
    for user_id, info in data["counters"].items():
        stats.append(f"{info['username']}: {info['count']} раз")
    
    response = "Статистика каканья 💩:\n" + "\n".join(stats)
    await bot.reply_to(message, response)

@bot.message_handler(commands=['generate_image'])
async def generate_image(message):
	msg = message.text.replace('/generate_image', '').strip()
	_, image = create_picture(msg)
	await bot.send_chat_action(chat_id=message.chat.id, action='upload_photo')
	image_data = base64.b64decode(image.split(',')[1])
	image_bin = Image.open(BytesIO(image_data))
	await bot.send_photo(chat_id=message.chat.id,reply_to_message_id=message.id, photo=image_bin, has_spoiler=True)


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
        
        # If replying to a message with photo, use that photo instead
        if message.reply_to_message and message.reply_to_message.photo:
            fileID = message.reply_to_message.photo[-1].file_id
        else:
            fileID = message.photo[-1].file_id
        file_info = await bot.get_file(fileID)
        downloaded_file = await bot.download_file(file_info.file_path)

        # Analyze image using moondream
        from pyollamabot.ollama import analyze_image
        image_description = await analyze_image(downloaded_file)

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
            messages = [{'role': 'user', 'content': f"Пользователь спрашивает{' с подписью: ' + caption if caption else ''} про фотографию на которой: {image_description}"}]
        else:
            messages = [{'role': 'user', 'content': f"Пользователь отправил фотографию{' с подписью: ' + caption if caption else ''}. Вот что на ней: {image_description}"}]
        
        await bot.send_chat_action(chat_id=message.chat.id, action='typing')
        answer, _ = await ask_model(messages=messages)
        
        if answer and not 'im_start' in answer:
            await bot.reply_to(message, answer, parse_mode='HTML')
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
        await bot.reply_to(message, 'Произошла ошибка при обработке фотографии.')

history = {}

def load_poop_counter():
    file_path = '/app/poop_counter.json'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"counters": {}}

def save_poop_counter(data):
    file_path = '/app/poop_counter.json'
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.debug(f"Successfully saved poop counter data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving poop counter data: {e}")

def increment_poop_counter(user_id, username):
    data = load_poop_counter()
    if str(user_id) not in data["counters"]:
        data["counters"][str(user_id)] = {"count": 0, "username": username}
    data["counters"][str(user_id)]["count"] += 1
    data["counters"][str(user_id)]["username"] = username  # Update username in case it changed
    save_poop_counter(data)
    return data["counters"][str(user_id)]["count"]

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
            await bot.reply_to(message, 'Ошибка при конвертации аудио.')
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
            await bot.reply_to(message, 'Не удалось распознать голосовое сообщение. Попробуйте еще раз или отправьте текст.')
            return
		
        # Convert any <reasoning> tags to Telegram spoiler tags
        logger.debug(f"Original transcription: {text}")
        text = text.replace('<reasoning>', '<blockquote expandable>').replace('</reasoning>', '</blockquote>')
        logger.debug(f"Converted transcription: {text}")
        await bot.reply_to(message=message, text=text, parse_mode='HTML')
        
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
                # Convert any <reasoning> tags to Telegram spoiler tags and remove <answer> tags
                logger.debug(f"Original message: {answer}")
                answer = answer.replace('<reasoning>', ' expandable="1"').replace('</reasoning>', '</blockquote>')
                answer = answer.replace('<answer>', '').replace('</answer>', '')
                logger.debug(f"Converted message: {answer}")
                await bot.reply_to(message, answer, parse_mode='HTML')
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
        await bot.reply_to(message, 'Произошла ошибка при обработке голосового сообщения.')

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
	
	# Check for "я покакал" message first, before any other processing
	logger.debug(f"Checking poop message. Original text: '{message.text}', Cleaned msg: '{msg}'")
	if "покакал" in msg.lower():
		logger.debug(f"Poop detected for user {message.from_user.first_name} (ID: {message.from_user.id})")
		count = increment_poop_counter(message.from_user.id, message.from_user.first_name)
		logger.debug(f"Updated poop count for {message.from_user.first_name}: {count}")
		response = f"{message.from_user.first_name} покакал уже {count} раз и продолжает"
		await bot.send_message(message.chat.id, response)
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
		image_description = await analyze_image(downloaded_file)

		# Store the image analysis in history
		if not history.get(message.chat.id):
			history[message.chat.id] = []
		history[message.chat.id].append([message.from_user.first_name, f"[Asked about photo: {image_description}]"])

		# Store in ollama's global history
		chat_history.append({
			'role': role(message.from_user),
			'content': f"[Question about photo: {image_description}]",
			'user': {
				'id': message.from_user.id,
				'name': message.from_user.first_name
			},
			'chat_id': message.chat.id,
			'message_id': message.message_id,
			'type': 'photo_question'
		})

		messages = [{'role': 'user', 'content': f"Пользователь спрашивает: {msg} про фотографию на которой: {image_description}"}]
		answer, _ = await ask_model(messages=messages)
		logger.debug(f"Raw model response: {answer}")
		if answer and not 'im_start' in answer:
			# Convert any <reasoning> tags to Telegram spoiler tags and remove <answer> tags
			logger.debug(f"Original message: {answer}")
			answer = answer.replace('<reasoning>', '<blockquote expandable>').replace('</reasoning>', '</blockquote>')
			answer = answer.replace('<answer>', '').replace('</answer>', '')
			logger.debug(f"Converted message: {answer}")
			await bot.reply_to(message, answer, parse_mode='HTML')
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
			msg = f"Ты приятель пользователя, он говорит тебе: {msg}. Что бы ты ему ответил в неформальном стиле. Подшути над ним. Коротко. {msg}"

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
					# Convert any <reasoning> tags to Telegram spoiler tags
					logger.debug(f"Original photo caption: {answer}")
					answer = answer.replace('<reasoning>', '<blockquote expandable>').replace('</reasoning>', '</blockquote>')
					logger.debug(f"Converted photo caption: {answer}")
					await bot.send_photo(chat_id=message.chat.id,reply_to_message_id=message.id, photo=image_bin, caption=answer, parse_mode='HTML')
				else:
					# Convert any <reasoning> tags to Telegram spoiler tags and remove <answer> tags
					logger.debug(f"Original message: {answer}")
					answer = answer.replace('<reasoning>', '<blockquote expandable>').replace('</reasoning>', '</blockquote>')
					answer = answer.replace('<answer>', '').replace('</answer>', '')
					logger.debug(f"Converted message: {answer}")
					await bot.reply_to(message, answer, parse_mode='HTML')
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


asyncio.run(bot.polling())
