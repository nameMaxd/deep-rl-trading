#!/usr/bin/env python3
"""
🚀 БЫСТРЫЙ РАБОЧИЙ ТЕСТ С ПРОГРЕСС-БАРОМ
🎯 Показываем прогресс обучения!
"""

from src.rl.env import Env
from src.rl.agent import Agent
import numpy as np
from datetime import datetime
from tqdm import tqdm
import time

def main():
    print("🚀 БЫСТРЫЙ ТЕСТ С ПРОГРЕСС-БАРОМ!")
    print("🎯 Видим как работает обучение!")
    print("=" * 50)
    
    # Быстрые параметры для демонстрации
    config = {
        'episodes': 30,        # быстро для демо
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
    
    print(f"📊 Окружения: Train {train_env.stock.obs_space.shape}, OOS {oos_env.stock.obs_space.shape}")
    
    # Создаем агента
    agent_params = {k: v for k, v in config.items() 
                   if k in ['embeddings', 'heads', 'layers', 'fwex', 'dropout', 'neurons',
                           'lr', 'epsilon', 'epsilon_min', 'epsilon_decay', 'gamma',
                           'memory_size', 'batch_size', 'update_freq']}
    
    agent = Agent(obs_space=train_env.stock.obs_space, **agent_params)
    print(f"🤖 Агент создан! Epsilon: {agent.epsilon:.3f}")
    
    # Логирование
    log_file = "models/quick-working-test.log"
    best_median = -float('inf')
    
    # ОБУЧЕНИЕ С ПРОГРЕСС-БАРОМ!
    print(f"\n🏃 Обучение с прогресс-баром...")
    
    with open(log_file, "w", encoding='utf-8') as f:
        f.write(f"Quick Working Test - {datetime.now()}\n")
        f.write(f"Config: {config}\n")
        f.write("=" * 80 + "\n")
    
    # ГЛАВНЫЙ ЦИКЛ С TQDM!
    pbar = tqdm(range(config['episodes']), desc="🧠 Обучение", 
                ncols=100, colour='green')
    
    for episode in pbar:
        # Тренировка
        state = train_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action = agent.act(state)
            next_state, trade_action, reward, done = train_env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            total_reward += reward
            state = next_state
            steps += 1
        
        # Получаем метрики тренировки
        train_metrics = train_env.get_trading_metrics()
        
        # OOS тестирование каждые 5 эпизодов
        if episode % 5 == 0:
            oos_profits = []
            
            # Быстрое тестирование на 3 позициях
            for start_pos in range(0, min(30, len(oos_env.stock.closes) - config['trading_period'] - config['window']), 10):
                state = oos_env.reset_fixed(start_pos)
                done = False
                
                while not done:
                    action = agent.act(state, training=False)  # БЕЗ exploration
                    next_state, trade_action, reward, done = oos_env.step(action)
                    state = next_state
                
                oos_metrics = oos_env.get_trading_metrics()
                oos_profits.append(oos_metrics['total_profit_dollars'])
            
            median_oos = np.median(oos_profits) if oos_profits else 0
            consistency = (np.array(oos_profits) > 0).mean() * 100 if oos_profits else 0
            
            # Сохраняем лучшую модель
            if median_oos > best_median:
                best_median = median_oos
                agent.save(f"models/quick-working-test_best")
            
            # Обновляем прогресс-бар с метриками
            pbar.set_postfix({
                'Train': f"${train_metrics['total_profit_dollars']:.0f}",
                'OOS': f"${median_oos:.0f}",
                'Trades': train_metrics['num_trades'],
                'Win%': f"{train_metrics['win_rate']*100:.1f}",
                'ε': f"{agent.epsilon:.3f}",
                'Best': f"${best_median:.0f}"
            })
            
            # Логирование
            log_entry = (
                f"Ep: {episode:3d} | Train: ${train_metrics['total_profit_dollars']:4.0f} | OOS Med: ${median_oos:4.0f}\n"
                f"    Trades: {train_metrics['num_trades']} | Win%: {train_metrics['win_rate']*100:.1f} | Steps: {steps}\n"
                f"    Consistency: {consistency:.1f}% | Epsilon: {agent.epsilon:.3f} | Best: ${best_median:.0f}\n"
                f"    OOS profits: {[int(p) for p in oos_profits]}\n"
                + "-" * 60
            )
            
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(log_entry + "\n")
        else:
            # Обновляем только основные метрики
            pbar.set_postfix({
                'Train': f"${train_metrics['total_profit_dollars']:.0f}",
                'Trades': train_metrics['num_trades'],
                'Win%': f"{train_metrics['win_rate']*100:.1f}",
                'ε': f"{agent.epsilon:.3f}",
                'Steps': steps
            })
        
        # Небольшая задержка чтобы видеть прогресс
        time.sleep(0.1)
    
    pbar.close()
    
    # Финальное сохранение
    agent.save(f"models/quick-working-test")
    
    print(f"\n🏁 Быстрый тест завершен!")
    print(f"💾 Модель: models/quick-working-test")
    print(f"🏆 Лучшая: models/quick-working-test_best")
    print(f"📊 Лог: {log_file}")
    
    # Финальное тестирование
    print(f"\n🔬 Финальный тест с прогрессом...")
    final_profits = []
    
    test_positions = list(range(0, min(20, len(oos_env.stock.closes) - config['trading_period'] - config['window']), 2))
    
    with tqdm(test_positions, desc="🧪 Финальный тест", ncols=80, colour='blue') as test_pbar:
        for start_pos in test_pbar:
            state = oos_env.reset_fixed(start_pos)
            done = False
            
            while not done:
                action = agent.act(state, training=False)
                next_state, trade_action, reward, done = oos_env.step(action)
                state = next_state
            
            oos_metrics = oos_env.get_trading_metrics()
            profit = oos_metrics['total_profit_dollars']
            final_profits.append(profit)
            
            # Обновляем прогресс
            test_pbar.set_postfix({
                'Profit': f"${profit:.0f}",
                'Avg': f"${np.mean(final_profits):.0f}"
            })
    
    final_median = np.median(final_profits)
    final_consistency = (np.array(final_profits) > 0).mean() * 100
    final_std = np.std(final_profits)
    
    print(f"\n📈 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   💰 Медианный профит: ${final_median:.2f}")
    print(f"   🎯 Consistency: {final_consistency:.1f}%")
    print(f"   📊 Стандартное отклонение: ${final_std:.2f}")
    print(f"   📈 Диапазон: ${min(final_profits):.0f} - ${max(final_profits):.0f}")
    print(f"   🔥 Лучшая модель: ${best_median:.0f}")
    
    if final_consistency > 50 and final_median > 50:
        print("✅ СИСТЕМА РАБОТАЕТ! 🎉")
    else:
        print("⚠️ Требуется доработка...")

if __name__ == "__main__":
    main() 