#!/usr/bin/env python3
"""
Скрипт для миграции данных из старой структуры poop_dates в новую poop_sessions
"""

import json
import os

def migrate_poop_counter():
    """Мигрирует данные из старой структуры в новую"""
    
    file_path = 'poop_counter.json'
    
    # Читаем текущие данные
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Файл poop_counter.json не найден")
        return
    
    migrated_count = 0
    
    # Мигрируем каждого пользователя
    for user_id, user_data in data["counters"].items():
        if 'poop_dates' in user_data and 'poop_sessions' not in user_data:
            # Конвертируем poop_dates в poop_sessions
            poop_dates = user_data['poop_dates']
            poop_sessions = {}
            
            for date in poop_dates:
                poop_sessions[date] = 1  # По умолчанию 1 сессия за день
            
            user_data['poop_sessions'] = poop_sessions
            del user_data['poop_dates']  # Удаляем старое поле
            
            migrated_count += 1
            print(f"Мигрирован пользователь {user_data['username']} ({user_id})")
    
    # Сохраняем обновленные данные
    if migrated_count > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"\nМиграция завершена! Обновлено пользователей: {migrated_count}")
    else:
        print("Миграция не требуется - все пользователи уже используют новую структуру")

if __name__ == "__main__":
    migrate_poop_counter()