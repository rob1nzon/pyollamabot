#!/usr/bin/env python3
"""
Тестовый скрипт для проверки системы достижений
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем функции из bot.py
from bot import check_achievements

def test_achievements():
    """Тестируем различные типы достижений"""
    
    print("🧪 Тестирование системы достижений\n")
    
    # Тестовые случаи
    test_cases = [
        (10, 1, "Круглое число 10"),
        (25, 1, "Круглое число 25"),
        (42, 1, "Особое число 42"),
        (69, 1, "Особое число 69"),
        (100, 1, "Круглое число 100"),
        (111, 1, "Повторяющиеся цифры 111"),
        (121, 1, "Палиндром 121"),
        (123, 1, "Последовательные цифры 123"),
        (234, 1, "Последовательные цифры 234"),
        (420, 1, "Особое число 420"),
        (666, 1, "Особое число 666"),
        (777, 1, "Особое число 777"),
        (1221, 1, "Палиндром 1221"),
        (1337, 1, "Особое число 1337"),
        (50, 7, "Круглое число + стрик 7 дней"),
        (75, 14, "Круглое число + стрик 14 дней"),
        (200, 30, "Круглое число + стрик 30 дней"),
    ]
    
    for count, streak, description in test_cases:
        achievements = check_achievements(count, streak)
        if achievements:
            print(f"✅ {description}:")
            for achievement in achievements:
                print(f"   🏆 {achievement['title']} (тип: {achievement['type']})")
        else:
            print(f"❌ {description}: Нет достижений")
        print()

if __name__ == "__main__":
    test_achievements()