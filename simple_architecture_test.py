#!/usr/bin/env python3
"""
🚀 ПРОСТОЙ ТЕСТ АРХИТЕКТУРЫ
🎯 Базовый трансформер для демонстрации
"""

from src.rl.env import Env
from src.rl.agent import Agent
import numpy as np
from datetime import datetime

def main():
    print("🚀 ПРОСТОЙ ТЕСТ АРХИТЕКТУРЫ")
    print("🎯 Показываем что система работает!")
    print("=" * 50)
    
    # Простые параметры
    config = {
        'episodes': 50,        # короткое обучение
        'trading_period': 60,  # короткий период
        'window': 30,          # маленькое окно
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
    
    print(f"📊 Конфигурация:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print("=" * 50)
    
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
    print(f"   Train: {train_env.stock.obs_space.shape}")
    print(f"   OOS: {oos_env.stock.obs_space.shape}")
    
    # Создаем агента
    agent_params = {k: v for k, v in config.items() 
                   if k in ['embeddings', 'heads', 'layers', 'fwex', 'dropout', 'neurons',
                           'lr', 'epsilon', 'epsilon_min', 'epsilon_decay', 'gamma',
                           'memory_size', 'batch_size', 'update_freq']}
    
    agent = Agent(obs_space=train_env.stock.obs_space, **agent_params)
    
    print(f"🤖 Агент создан!")
    
    # Логирование
    log_file = "models/simple-architecture-test.log"
    model_name = "simple-architecture-test"
    
    best_median = -float('inf')
    
    print(f"\n🏃 Быстрое обучение на {config['episodes']} эпизодов...")
    
    with open(log_file, "w", encoding='utf-8') as f:
        f.write(f"Simple Architecture Test - {datetime.now()}\n")
        f.write(f"Config: {config}\n")
        f.write("=" * 80 + "\n")
    
    for episode in range(config['episodes']):
        # Тренировка
        state = train_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, trade_action, reward, done = train_env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            total_reward += reward
            state = next_state
        
        # Получаем метрики тренировки
        train_metrics = train_env.get_trading_metrics()
        
        # OOS тестирование каждые 10 эпизодов
        if episode % 10 == 0:
            # Тестируем на 3 позициях
            oos_profits = []
            
            for start_pos in range(0, min(30, len(oos_env.stock.closes) - config['trading_period'] - config['window']), 10):
                # Используем reset_fixed для честного тестирования
                state = oos_env.reset_fixed(start_pos)
                done = False
                
                while not done:
                    action = agent.act(state, training=False)  # БЕЗ exploration
                    next_state, trade_action, reward, done = oos_env.step(action)
                    state = next_state
                
                oos_metrics = oos_env.get_trading_metrics()
                oos_profits.append(oos_metrics['total_profit_dollars'])
            
            median_oos = np.median(oos_profits) if oos_profits else 0
            mean_oos = np.mean(oos_profits) if oos_profits else 0
            consistency = (np.array(oos_profits) > 0).mean() * 100 if oos_profits else 0
            
            # Сохраняем лучшую модель
            if median_oos > best_median:
                best_median = median_oos
                agent.save(f"models/{model_name}_best")
                print(f"💾 Новая лучшая медиана: ${median_oos:.0f} (эпизод {episode})")
            
            # Логирование
            log_entry = (
                f"Ep: {episode:3d} | Train: ${train_metrics['total_profit_dollars']:4.0f} | OOS Med: ${median_oos:4.0f}\n"
                f"    Trades: {train_metrics['num_trades']} | Win%: {train_metrics['win_rate']*100:.1f}\n"
                f"    OOS: Med ${median_oos:.0f}, Mean ${mean_oos:.0f}, Consistency {consistency:.1f}%\n"
                f"    Epsilon: {agent.epsilon:.3f} | Best: ${best_median:.0f}\n"
                f"    All OOS: {[int(p) for p in oos_profits]}\n"
                + "-" * 60
            )
            
            print(log_entry)
            
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(log_entry + "\n")
    
    # Финальное сохранение
    agent.save(f"models/{model_name}")
    
    print(f"\n🏁 Тест завершен!")
    print(f"💾 Модель: models/{model_name}")
    print(f"🏆 Лучшая: models/{model_name}_best")
    print(f"📊 Лог: {log_file}")
    
    # Финальное тестирование
    print(f"\n🔬 Финальный тест...")
    final_profits = []
    
    for start_pos in range(0, min(50, len(oos_env.stock.closes) - config['trading_period'] - config['window']), 5):
        state = oos_env.reset_fixed(start_pos)
        done = False
        
        while not done:
            action = agent.act(state, training=False)
            next_state, trade_action, reward, done = oos_env.step(action)
            state = next_state
        
        oos_metrics = oos_env.get_trading_metrics()
        final_profits.append(oos_metrics['total_profit_dollars'])
    
    final_median = np.median(final_profits)
    final_consistency = (np.array(final_profits) > 0).mean() * 100
    final_std = np.std(final_profits)
    
    print(f"📈 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   Медианный профит: ${final_median:.2f}")
    print(f"   Consistency: {final_consistency:.1f}%")
    print(f"   Стандартное отклонение: ${final_std:.2f}")
    print(f"   Диапазон: ${min(final_profits):.0f} - ${max(final_profits):.0f}")
    
    if final_consistency > 50 and final_median > 50:
        print("✅ Архитектура работает!")
    else:
        print("❌ Требуется доработка...")

if __name__ == "__main__":
    main() 