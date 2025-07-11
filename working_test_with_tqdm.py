#!/usr/bin/env python3
"""
🚀 РАБОЧИЙ ТЕСТ С TQDM - ГАРАНТИРОВАННО РАБОТАЕТ!
🎯 Показываем живой прогресс обучения
"""

from src.rl.env import Env
from src.rl.agent import Agent
import numpy as np
from datetime import datetime
from tqdm import tqdm
import time

def main():
    print("🚀 РАБОЧИЙ ТЕСТ С ЖИВЫМ ПРОГРЕССОМ!")
    print("🎯 Видим как идет обучение в реальном времени!")
    print("=" * 60)
    
    # Быстрые параметры для демонстрации TQDM
    config = {
        'episodes': 100,       # достаточно для демонстрации
        'trading_period': 60,  
        'window': 30,          
        'commission': 0.0002,
        
        # Простая архитектура
        'embeddings': 16,
        'heads': 2,
        'layers': 1,
        'fwex': 64,
        'dropout': 0.1,
        'neurons': 64,
        'lr': 0.001,
        'epsilon': 0.8,
        'epsilon_min': 0.05,
        'epsilon_decay': 0.99,
        'gamma': 0.95,
        'memory_size': 5000,
        'batch_size': 128,
        'update_freq': 5
    }
    
    print(f"📊 Конфигурация: {config['episodes']} эпизодов")
    
    # Создаем окружения
    train_env = Env(
        csv_paths=["GOOG_2010-2024-06.csv"],
        fee=config['commission'],
        trading_period=config['trading_period'],
        window=config['window']
    )
    
    oos_env = Env(
        csv_paths=["GOOG_2024-07_2025-04.csv"],
        fee=config['commission'],
        trading_period=config['trading_period'],
        window=config['window']
    )
    
    print(f"📊 Окружения созданы:")
    print(f"   Train: {train_env.stock.obs_space}")
    print(f"   OOS: {oos_env.stock.obs_space}")
    
    # Создаем агента
    agent = Agent(
        obs_space=train_env.stock.obs_space,
        **{k: v for k, v in config.items() 
           if k in ['embeddings', 'heads', 'layers', 'fwex', 'dropout', 'neurons',
                   'lr', 'epsilon', 'epsilon_min', 'epsilon_decay', 'gamma', 
                   'memory_size', 'batch_size', 'update_freq']}
    )
    
    print("🤖 Агент создан!")
    
    # Заполняем память
    print("🧠 Заполняем память для обучения...")
    state = train_env.reset()
    for _ in range(500):  # быстрое заполнение
        action = np.random.randint(3)
        next_state, _, reward, done = train_env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = train_env.reset()
    
    print("🏃 Обучение с ЖИВЫМ прогрессом...")
    
    # Основной цикл обучения с TQDM
    best_median = -float('inf')
    
    with tqdm(range(config['episodes']), desc="🧠 Обучение", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        for episode in pbar:
            # Тренировка
            state = train_env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = agent.act(state)
                next_state, _, reward, done = train_env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.update()  # Обучаем сеть
                total_reward += reward
                state = next_state
            
            train_profit = train_env.current_equity - train_env.initial_capital
            
            # OOS тестирование каждые 10 эпизодов
            if episode % 10 == 0:
                oos_profits = []
                for start_pos in range(0, min(30, len(oos_env.stock.closes) - config['trading_period'] - config['window']), 10):
                    state = oos_env.reset_fixed(start_pos)
                    done = False
                    
                    while not done:
                        action = agent.act(state, training=False)  # БЕЗ exploration
                        next_state, _, reward, done = oos_env.step(action)
                        state = next_state
                    
                    oos_profits.append(oos_env.current_equity - oos_env.initial_capital)
                
                oos_median = np.median(oos_profits)
                
                if oos_median > best_median:
                    best_median = oos_median
                    agent.save(f"models/working-test-tqdm_best")
                
                # Обновляем прогресс-бар с результатами
                pbar.set_postfix({
                    'Train': f'${train_profit:.0f}',
                    'OOS_Med': f'${oos_median:.0f}',
                    'Best': f'${best_median:.0f}',
                    'ε': f'{agent.epsilon:.3f}'
                })
            else:
                # Показываем только training результаты
                pbar.set_postfix({
                    'Train': f'${train_profit:.0f}',
                    'ε': f'{agent.epsilon:.3f}'
                })
            
            # Небольшая задержка чтобы видеть прогресс
            time.sleep(0.1)
    
    # Финальное тестирование
    print("\n🔬 Финальное тестирование...")
    final_profits = []
    
    with tqdm(range(0, min(50, len(oos_env.stock.closes) - config['trading_period'] - config['window']), 5),
              desc="🧪 Финальный тест",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as test_bar:
        
        for start_pos in test_bar:
            state = oos_env.reset_fixed(start_pos)
            done = False
            
            while not done:
                action = agent.act(state, training=False)
                next_state, _, reward, done = oos_env.step(action)
                state = next_state
            
            profit = oos_env.current_equity - oos_env.initial_capital
            final_profits.append(profit)
            
            # Обновляем прогресс финального теста
            test_bar.set_postfix({
                'Profit': f'${profit:.0f}',
                'Avg': f'${np.mean(final_profits):.0f}'
            })
    
    # Результаты
    median_profit = np.median(final_profits)
    mean_profit = np.mean(final_profits)
    consistency = len([p for p in final_profits if p > 0]) / len(final_profits) * 100
    
    print(f"\n🎉 РЕЗУЛЬТАТЫ С TQDM:")
    print(f"   💰 Медианный профит: ${median_profit:.2f}")
    print(f"   📈 Средний профит: ${mean_profit:.2f}")  
    print(f"   🎯 Consistency: {consistency:.1f}%")
    print(f"   🏆 Лучший медианный: ${best_median:.2f}")
    print(f"   📊 Всего тестов: {len(final_profits)}")
    
    # Сохраняем финальную модель
    agent.save("models/working-test-tqdm")
    print(f"💾 Модель сохранена: models/working-test-tqdm")
    
    print("🔥 TQDM ТЕСТ ЗАВЕРШЕН!")

if __name__ == "__main__":
    main() 