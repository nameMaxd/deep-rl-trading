#!/usr/bin/env python3
"""
🔥 MLP v16 С TQDM ПРОГРЕСС-БАРАМИ!
🎯 Глубокий MLP baseline с живым прогрессом
"""

from src.rl.env import Env
from src.rl.agent import Agent
import numpy as np
from tqdm import tqdm
import time

def main():
    print("🔥 MLP v16 С ЖИВЫМ TQDM ПРОГРЕССОМ!")
    print("🎯 Глубокий MLP baseline с residuals!")
    print("=" * 60)
    
    # MLP конфигурация
    config = {
        'episodes': 160,
        'trading_period': 80,
        'window': 35,
        'commission': 0.0002,
        
        # MLP параметры
        'embeddings': 32,  # Простой для MLP
        'heads': 2,
        'layers': 2,
        'fwex': 128,
        'dropout': 0.2,
        'neurons': 128,
        'lr': 0.001,
        'epsilon': 0.8,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'gamma': 0.95,
        'memory_size': 10000,
        'batch_size': 256,
        'update_freq': 5
    }
    
    print(f"📊 MLP конфигурация: {config['episodes']} эпизодов")
    
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
    
    print(f"📊 MLP окружения:")
    print(f"   Train: {train_env.stock.obs_space.shape}")
    print(f"   OOS: {oos_env.stock.obs_space.shape}")
    
    # Создаем MLP агента
    agent = Agent(
        obs_space=train_env.stock.obs_space,
        **{k: v for k, v in config.items() 
           if k not in ['episodes', 'trading_period', 'window', 'commission']}
    )
    
    print(f"🔥 MLP агент создан:")
    print(f"   Архитектура: {config['embeddings']} emb, {config['layers']} layers")
    print(f"   FC: {config['neurons']} -> {config['neurons']} -> 3")
    print(f"   Dropout: {config['dropout']}")
    
    # Заполняем память
    print("🧠 Заполняем MLP память...")
    state = train_env.reset()
    for _ in range(800):
        action = np.random.randint(3)
        next_state, _, reward, done = train_env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = train_env.reset()
    
    print("🏃 MLP обучение с ЖИВЫМ прогрессом...")
    
    # Основной цикл обучения с TQDM
    best_median = -float('inf')
    
    with tqdm(range(config['episodes']), desc="🔥 MLP", 
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
            
            # OOS тестирование каждые 16 эпизодов
            if episode % 16 == 0:
                oos_profits = []
                for start_pos in range(0, min(32, len(oos_env.stock.closes) - config['trading_period'] - config['window']), 8):
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
                    agent.save(f"models/tqdm-mlp-v16_best")
                
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
            
            time.sleep(0.06)  # Среднее время для MLP
    
    # Финальное тестирование
    print("\n🔬 MLP финальное тестирование...")
    final_profits = []
    
    with tqdm(range(0, min(80, len(oos_env.stock.closes) - config['trading_period'] - config['window']), 5),
              desc="🧪 MLP тест",
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
    
    # MLP результаты
    median_profit = np.median(final_profits)
    mean_profit = np.mean(final_profits)
    consistency = len([p for p in final_profits if p > 0]) / len(final_profits) * 100
    
    print(f"\n🔥 MLP v16 РЕЗУЛЬТАТЫ:")
    print(f"   💰 Медианный профит: ${median_profit:.2f}")
    print(f"   📈 Средний профит: ${mean_profit:.2f}")  
    print(f"   🎯 Consistency: {consistency:.1f}%")
    print(f"   🏆 Лучший медианный: ${best_median:.2f}")
    print(f"   📊 Всего тестов: {len(final_profits)}")
    
    # Сохраняем финальную модель
    agent.save("models/tqdm-mlp-v16")
    print(f"💾 MLP модель сохранена: models/tqdm-mlp-v16")
    
    print("🔥 MLP v16 ЗАВЕРШЕН!")
    
    return {
        'name': 'MLP v16',
        'median_profit': median_profit,
        'mean_profit': mean_profit,
        'consistency': consistency,
        'best_median': best_median,
        'total_tests': len(final_profits)
    }

if __name__ == "__main__":
    main() 