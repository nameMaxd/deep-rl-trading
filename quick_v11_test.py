#!/usr/bin/env python3
"""
Быстрый тест v11 модели
"""

import torch
import numpy as np
from src.rl.env import Env
from src.rl.agent import Agent
import os

def quick_test():
    print("🚀 БЫСТРЫЙ ТЕСТ v11 модели")
    print("🎯 Проверяем, работает ли модель вообще")
    
    # Тестируем на оригинальном OOS файле
    oos_file = "GOOG_2024-07_2025-04.csv"
    model_path = "models/google-trading-v11-honest_best"
    
    if not os.path.exists(oos_file):
        print(f"❌ OOS файл не найден: {oos_file}")
        return
        
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        return
    
    # Создаем среду
    env = Env(csv_paths=[oos_file], fee=0.0002, trading_period=120, window=50)
    print(f"📊 Данных: {len(env.stock.closes)} строк")
    
    # Создаем агента с конфигурацией v11
    v11_config = {
        'embeddings': 32,      # Как в v11
        'heads': 2,            # Как в v11
        'layers': 2,           # Как в v11
        'fwex': 128,          # Как в v11
        'dropout': 0.05,      # Как в v11
        'neurons': 128,       # Как в v11
        'lr': 0.001,          # Как в v11
        'epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.997,
        'gamma': 0.95,
        'memory_size': 5000,
        'batch_size': 512,
        'update_freq': 5
    }
    
    agent = Agent(obs_space=env.stock.obs_space, **v11_config)
    
    # Загружаем модель
    success = agent.load(model_path)
    if not success:
        print("❌ Не удалось загрузить модель")
        return
    
    # Быстрый тест на 5 позициях
    results = []
    
    print(f"\n🧪 Тестирую 5 фиксированных позиций...")
    
    for start_pos in [0, 10, 20, 30, 40]:
        try:
            # Сброс на фиксированную позицию  
            state = env.reset_fixed(start_position=start_pos)
            
            total_trades = 0
            total_wins = 0
            
            # Прогон эпизода
            while env.ind < env.end:
                # Действие БЕЗ исследования (детерминистично)
                action = agent.act(state, training=False)
                
                next_state, reward, done, _ = env.step(action)
                
                if action != 0:  # Торговая операция
                    total_trades += 1
                    if reward > 0:
                        total_wins += 1
                
                state = next_state
                
                if done:
                    break
            
            # Получаем финальные метрики
            metrics = env.get_trading_metrics()
            profit = metrics['total_profit_dollars']
            
            win_rate = total_wins / max(total_trades, 1) * 100
            
            print(f"  Позиция {start_pos}: profit=${profit:.0f}, trades={total_trades}, win_rate={win_rate:.1f}%")
            results.append(profit)
            
        except Exception as e:
            print(f"  Позиция {start_pos}: ОШИБКА - {e}")
            results.append(0)
    
    # Анализ результатов
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"   Медиана: ${np.median(results):.0f}")
    print(f"   Среднее: ${np.mean(results):.0f}")
    print(f"   Диапазон: ${min(results):.0f} - ${max(results):.0f}")
    print(f"   Прибыльных: {(np.array(results) > 0).sum()}/5")
    
    if np.median(results) > 1000:
        print(f"   💰 ОТЛИЧНО! Медиана >$1000")
    elif np.median(results) > 100:
        print(f"   👍 ХОРОШО! Медиана >$100")
    elif np.median(results) > 0:
        print(f"   🤔 СРЕДНЕ! Медиана >$0")
    else:
        print(f"   ❌ ПЛОХО! Убыточно")
    
    print(f"\n🔍 ВЫВОД:")
    if max(results) < 100:
        print(f"   ⚠️ ПОДОЗРИТЕЛЬНО! Результат $3065 был явно случайностью!")
        print(f"   📉 Текущие результаты намного хуже ожидаемых")
    else:
        print(f"   ✅ Есть хорошие результаты, возможно модель работает")

if __name__ == "__main__":
    quick_test() 