#!/usr/bin/env python3
"""
🤖 TRANSFORMER v14 С TQDM ПРОГРЕСС-БАРАМИ!
🎯 Мощная трансформер архитектура с живым прогрессом
"""

from src.rl.env import Env
from src.rl.agent import Agent
import numpy as np
from tqdm import tqdm
import time

def main():
    print("🤖 TRANSFORMER v14 С ЖИВЫМ TQDM ПРОГРЕССОМ!")
    print("🎯 Мощная трансформер архитектура!")
    print("=" * 60)
    
    # Мощная трансформер конфигурация
    config = {
        'episodes': 250,
        'trading_period': 120,
        'window': 50,
        'commission': 0.0002,
        
        # Трансформер параметры
        'embeddings': 64,
        'heads': 4,
        'layers': 3,
        'fwex': 256,
        'dropout': 0.1,
        'neurons': 256,
        'lr': 0.0005,
        'epsilon': 0.8,
        'epsilon_min': 0.005,
        'epsilon_decay': 0.9965,
        'gamma': 0.95,
        'memory_size': 15000,
        'batch_size': 512,
        'update_freq': 3
    }
    
    print(f"📊 TRANSFORMER конфигурация: {config['episodes']} эпизодов")
    
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
    
    print(f"📊 TRANSFORMER окружения:")
    print(f"   Train: {train_env.stock.obs_space.shape}")
    print(f"   OOS: {oos_env.stock.obs_space.shape}")
    
    # Создаем мощного трансформер агента
    agent = Agent(
        obs_space=train_env.stock.obs_space,
        **{k: v for k, v in config.items() 
           if k not in ['episodes', 'trading_period', 'window', 'commission']}
    )
    
    print(f"🤖 МОЩНЫЙ трансформер агент создан:")
    print(f"   Архитектура: {config['embeddings']} emb, {config['heads']} heads, {config['layers']} layers")
    print(f"   FC: {config['neurons']} -> {config['neurons']} -> 3")
    
    # Заполняем память
    print("🧠 Заполняем TRANSFORMER память...")
    state = train_env.reset()
    for _ in range(1500):  # Больше для мощной модели
        action = np.random.randint(3)
        next_state, _, reward, done = train_env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = train_env.reset()
    
    print("🏃 TRANSFORMER обучение с ЖИВЫМ прогрессом...")
    
    # Основной цикл обучения с TQDM
    best_median = -float('inf')
    
    with tqdm(range(config['episodes']), desc="🤖 TRANSFORMER", 
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
                agent.update()
                total_reward += reward
                state = next_state
            
            train_profit = train_env.current_equity - train_env.initial_capital
            
            # OOS тестирование каждые 25 эпизодов
            if episode % 25 == 0:
                oos_profits = []
                for start_pos in range(0, min(50, len(oos_env.stock.closes) - config['trading_period'] - config['window']), 10):
                    state = oos_env.reset_fixed(start_pos)
                    done = False
                    
                    while not done:
                        action = agent.act(state, training=False)
                        next_state, _, reward, done = oos_env.step(action)
                        state = next_state
                    
                    oos_profits.append(oos_env.current_equity - oos_env.initial_capital)
                
                oos_median = np.median(oos_profits)
                
                if oos_median > best_median:
                    best_median = oos_median
                    agent.save(f"models/tqdm-transformer-v14_best")
                
                # Обновляем прогресс-бар
                pbar.set_postfix({
                    'Train': f'${train_profit:.0f}',
                    'OOS_Med': f'${oos_median:.0f}',
                    'Best': f'${best_median:.0f}',
                    'ε': f'{agent.epsilon:.3f}',
                    'Trades': f'{train_env.trade_count}'
                })
            else:
                # Показываем только training
                pbar.set_postfix({
                    'Train': f'${train_profit:.0f}',
                    'ε': f'{agent.epsilon:.3f}',
                    'Trades': f'{train_env.trade_count}'
                })
            
            time.sleep(0.03)  # Быстрее для мощной модели
    
    # Финальное тестирование
    print("\n🔬 TRANSFORMER финальное тестирование...")
    final_profits = []
    
    with tqdm(range(0, min(140, len(oos_env.stock.closes) - config['trading_period'] - config['window']), 7),
              desc="🧪 TRANSFORMER тест",
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
            
            test_bar.set_postfix({
                'Profit': f'${profit:.0f}',
                'Avg': f'${np.mean(final_profits):.0f}'
            })
    
    # TRANSFORMER результаты
    median_profit = np.median(final_profits)
    mean_profit = np.mean(final_profits)
    consistency = len([p for p in final_profits if p > 0]) / len(final_profits) * 100
    
    print(f"\n🤖 TRANSFORMER v14 РЕЗУЛЬТАТЫ:")
    print(f"   💰 Медианный профит: ${median_profit:.2f}")
    print(f"   📈 Средний профит: ${mean_profit:.2f}")  
    print(f"   🎯 Consistency: {consistency:.1f}%")
    print(f"   🏆 Лучший медианный: ${best_median:.2f}")
    print(f"   📊 Всего тестов: {len(final_profits)}")
    
    # Сохраняем финальную модель
    agent.save("models/tqdm-transformer-v14")
    print(f"💾 TRANSFORMER модель сохранена: models/tqdm-transformer-v14")
    
    print("🤖 TRANSFORMER v14 ЗАВЕРШЕН!")
    
    return {
        'name': 'TRANSFORMER v14',
        'median_profit': median_profit,
        'mean_profit': mean_profit,
        'consistency': consistency,
        'best_median': best_median,
        'total_tests': len(final_profits)
    }

if __name__ == "__main__":
    main() 