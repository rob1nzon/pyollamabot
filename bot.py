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
	fileID = message.photo[-1].file_id
	file_info = await bot.get_file(fileID)
	downloaded_file = await bot.download_file(file_info.file_path)
	# if random.randint(20,50) == 42:
		# await bot.reply_to(message, 'баян', parse_mode='HTML')

history = {}

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
		
        await bot.reply_to(message=message, text=text, parse_mode='HTML')
        
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
        logger.debug(f"Model response: {answer[:100]}...")
        
        if answer and not 'im_start' in answer:
                logger.debug("Sending text response")
                await bot.reply_to(message, answer, parse_mode='HTML')
                
    except Exception as e:
        logger.error(f"Error processing voice message: {e}", exc_info=True)
        await bot.reply_to(message, 'Произошла ошибка при обработке голосового сообщения.')

# Handle all other messages with content_type 'text' (content_types defaults to ['text'])
@bot.message_handler(func=lambda message: True)
async def echo_message(message: Message):
	bot_info = await bot.get_me()
	if not history.get(message.chat.id):
		# await bot.send_dice(message.chat.id)
		history[message.chat.id] = []
	history[message.chat.id].append([message.from_user.first_name, message.text])

	if len(history[message.chat.id]) > 70:
		msg_history = ' '.join([f'{user}:{msg}' for user, msg in history[message.chat.id]])
		history[message.chat.id] = []
		messages = []
		messages.append({'role': f'{role(message.from_user)}', 'content': f'Необходим краткий пересказ по общению в этом чате. Укажи кратко у кого что случилось в несколько предложений. История чата: {msg_history}'})
		answer, image = await ask_model(messages=messages)
		# await bot.send_message(message.chat.id, f'#summary || {answer}||')
	mention = (message.reply_to_message and message.reply_to_message.from_user.username == bot_info.username)
	direct = message.text.startswith(f'@{bot_info.username}')

	rand_value = random.randint(0, 200)
	if not mention and not direct and rand_value != 42:
		return
	await bot.send_chat_action(chat_id=message.chat.id, action='typing')


	msg = message.text.replace(f'@{bot_info.username}', '').strip()
	if rand_value == 42:
		msg = f"Ты приятель пользователя, он говорит тебе: {msg}. Что бы ты ему ответил в неформальном стиле. Подшути над ним. Коротко. {msg}"

	for _ in range(10):
		messages = []
		if message.reply_to_message:
			messages.append({'role': f'{role(message.reply_to_message.from_user)}', 'content': f'{message.reply_to_message.text}'})
		messages.append({'role': f'{role(message.from_user)}', 'content': f'{msg}'})
		answer, image = await ask_model(messages=messages)
		if answer and not 'im_start' in answer:
			if image:
				await bot.send_chat_action(chat_id=message.chat.id, action='upload_photo')
				image_data = base64.b64decode(image.split(',')[1])
				image_bin = Image.open(BytesIO(image_data))
				await bot.send_photo(chat_id=message.chat.id,reply_to_message_id=message.id, photo=image_bin, caption=answer, parse_mode='HTML')
			else:
				await bot.reply_to(message, answer, parse_mode='HTML')
			break
		time.sleep(2)


asyncio.run(bot.polling())
