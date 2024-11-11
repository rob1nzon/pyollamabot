#!/usr/bin/python

# This is a simple echo bot using the decorator mechanism.
# It echoes any incoming text messages.
import asyncio
import os
import time
from pyollamabot.ollama import ask_model

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
    text = 'Hi, I am EchoBot.\nJust write me something and I will repeat it!'
    await bot.reply_to(message, text)


def role(msg):
	if msg.is_bot:
		return "assistant"
	return "user"

# Handle all other messages with content_type 'text' (content_types defaults to ['text'])
@bot.message_handler(func=lambda message: True)
async def echo_message(message: Message):
	bot_info = await bot.get_me()

	mention = (message.reply_to_message and message.reply_to_message.from_user.username == bot_info.username)
	direct = message.text.startswith(f'@{bot_info.username}')

	if not mention and not direct:
		return
	
	msg = message.text.replace(f'@{bot_info.username}', '').strip()

	for _ in range(10):
		messages = []
		if message.reply_to_message:
			messages.append({'role': f'{role(message.reply_to_message.from_user)}', 'content': f'{message.reply_to_message.text}'})
		messages.append({'role': f'{role(message.from_user)}', 'content': f'{msg}'})
		answer, image = await ask_model(messages=messages)
		if answer and len(answer)>23:
			if image:
				image_data = base64.b64decode(image.split(',')[1])
				image_bin = Image.open(BytesIO(image_data))
				await bot.send_photo(chat_id=message.chat.id,reply_to_message_id=message.id, photo=image_bin, caption=answer)
			else:
				await bot.reply_to(message, answer)
			break
		time.sleep(2)


asyncio.run(bot.polling())
