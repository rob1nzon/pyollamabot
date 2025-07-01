#!/usr/bin/env python3
"""
Тестовый скрипт для проверки множественных сессий в день
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем функции из bot.py
from bot import increment_poop_counter, load_poop_counter, save_poop_counter

def test_multiple_sessions():
    """Тестируем множественные сессии в день"""
    
    print("🧪 Тестирование множественных сессий в день\n")
    
    # Тестовый пользователь
    test_user_id = 999999999
    test_username = "TestUser"
    
    print(f"Тестируем пользователя: {test_username} (ID: {test_user_id})")
    
    # Симулируем несколько сессий в день
    for session in range(1, 6):
        try:
            count, streak, sessions_today = increment_poop_counter(test_user_id, test_username)
            print(f"Сессия {session}: Общий счет: {count}, Стрик: {streak}, Сессий сегодня: {sessions_today}")
        except Exception as e:
            print(f"Ошибка в сессии {session}: {e}")
    
    # Проверяем данные
    data = load_poop_counter()
    if str(test_user_id) in data["counters"]:
        user_data = data["counters"][str(test_user_id)]
        print(f"\nИтоговые данные пользователя:")
        print(f"- Общий счет: {user_data['count']}")
        print(f"- Стрик: {user_data['current_streak']}")
        print(f"- Сессии: {user_data['poop_sessions']}")
        
        # Очищаем тестовые данные
        del data["counters"][str(test_user_id)]
        save_poop_counter(data)
        print(f"\nТестовые данные очищены")
    
    print("\n✅ Тест завершен!")

if __name__ == "__main__":
    test_multiple_sessions()