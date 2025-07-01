#!/usr/bin/env python3
"""
Тестовый скрипт для проверки генерации поздравлений
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем функции из bot.py
from bot import generate_achievement_congratulation, check_achievements

async def test_congratulations():
    """Тестируем генерацию поздравлений"""
    
    print("🎉 Тестирование генерации поздравлений\n")
    
    # Тестовые достижения
    test_achievements = [
        {
            'type': 'round_number',
            'value': 100,
            'title': 'Круглое число: 100!'
        },
        {
            'type': 'palindrome',
            'value': 121,
            'title': 'Палиндром: 121!'
        },
        {
            'type': 'special',
            'value': 42,
            'title': '42 - Ответ на главный вопрос жизни, вселенной и всего такого!'
        },
        {
            'type': 'streak',
            'value': 7,
            'title': 'Стрик 7 дней!'
        }
    ]
    
    username = "TestUser"
    
    for achievement in test_achievements:
        print(f"🏆 Тестируем достижение: {achievement['title']}")
        try:
            congratulation = await generate_achievement_congratulation(achievement, username)
            print(f"✅ Поздравление: {congratulation}")
        except Exception as e:
            print(f"❌ Ошибка: {str(e)}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_congratulations())