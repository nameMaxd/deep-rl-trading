#!/usr/bin/env python3
"""
🧠 LSTM v13 С TQDM ПРОГРЕСС-БАРАМИ!
🎯 Простая рекуррентная архитектура с живым прогрессом
"""

from src.rl.env import Env
from src.rl.agent import Agent
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import time

class LSTMAgent(Agent):
    def __init__(self, obs_space, **kwargs):
        # Сохраняем LSTM-специфичные параметры
        self.lstm_hidden = kwargs.pop('lstm_hidden', 128)
        self.lstm_layers = kwargs.pop('lstm_layers', 2)
        self.fc_hidden = kwargs.pop('fc_hidden', 64)
        
        # Передаем только базовые параметры в родительский класс
        super().__init__(obs_space, **kwargs)
        
        # Переопределяем архитектуру на LSTM
        num_features = obs_space.shape[1] if len(obs_space.shape) == 3 else obs_space.shape[0]
        
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=0.1 if self.lstm_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(self.lstm_hidden, self.fc_hidden)
        self.fc2 = nn.Linear(self.fc_hidden, 32)
        self.fc3 = nn.Linear(32, 3)
        self.dropout = nn.Dropout(0.1)
        
        print(f"🧠 LSTM агент создан:")
        print(f"   Архитектура: LSTM({self.lstm_hidden}) x {self.lstm_layers}")
        print(f"   FC: {self.fc_hidden} -> 32 -> 3")

def main():
    print("🧠 LSTM v13 С ЖИВЫМ TQDM ПРОГРЕССОМ!")
    print("🎯 Простая рекуррентная архитектура!")
    print("=" * 60)
    
    # LSTM конфигурация
    config = {
        'episodes': 200,
        'trading_period': 90,
        'window': 40,
        'commission': 0.0002,
        
        # LSTM параметры
        'lstm_hidden': 128,
        'lstm_layers': 2,
        'fc_hidden': 64,
        'dropout': 0.1,
        'lr': 0.001,
        'epsilon': 0.05,        # НИЗКИЙ стартовый epsilon!
        'epsilon_min': 0.001,   # Очень низкий минимум
        'epsilon_decay': 0.999, # Медленный спад
        'gamma': 0.95,
        'memory_size': 10000,
        'batch_size': 256,
        'update_freq': 5
    }
    
    print(f"📊 LSTM конфигурация: {config['episodes']} эпизодов")
    
    # Создаем окружения
    print("📈 Загружаем данные...")
    train_env = Env(
        csv_paths=["GOOG_2010-2024-06.csv"],
        fee=config['commission'],
        trading_period=config['trading_period'],
        window=config['window']
    )
    
    oos_env = Env(
        csv_paths=["GOOG_2024-07_2025-04.csv"],  # Правильные OOS данные!
        fee=config['commission'],
        trading_period=config['trading_period'],
        window=config['window']
    )
    
    print(f"📊 ДАННЫЕ ЗАГРУЖЕНЫ:")
    print(f"   🎯 Train: 2010-2024 данные, shape: {train_env.stock.obs_space.shape}")
    print(f"   🧪 OOS: 2024-07_2025-04 данные, shape: {oos_env.stock.obs_space.shape}")
    print(f"   📏 Trading period: {config['trading_period']} дней")
    print(f"   🪟 Window: {config['window']} дней")
    print("=" * 60)
    
    # Создаем LSTM агента
    agent = LSTMAgent(
        obs_space=train_env.stock.obs_space,
        **{k: v for k, v in config.items() 
           if k not in ['episodes', 'trading_period', 'window', 'commission']}
    )
    
    # Заполняем память
    print("🧠 Заполняем LSTM память...")
    state = train_env.reset()
    for _ in range(1000):
        action = np.random.randint(3)
        next_state, _, reward, done = train_env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = train_env.reset()
    
    print("🏃 LSTM обучение с ЖИВЫМ прогрессом...")
    print("📈 Каждую эпоху: OOS тест")
    print("📊 Каждые 10 эпох: подробный отчет")
    print("=" * 60)
    
    # Основной цикл обучения с TQDM
    best_median = -float('inf')
    
    with tqdm(range(config['episodes']), desc="🧠 LSTM", 
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
            
            # OOS тестирование КАЖДУЮ эпоху
            oos_profits = []
            for start_pos in range(0, min(20, len(oos_env.stock.closes) - config['trading_period'] - config['window']), 5):
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
                agent.save(f"models/tqdm-lstm-v13_best")
            
            # Обновляем прогресс-бар
            pbar.set_postfix({
                'Train': f'${train_profit:.0f}',
                'OOS_Med': f'${oos_median:.0f}',
                'Best': f'${best_median:.0f}',
                'ε': f'{agent.epsilon:.3f}',
                'Trades': f'{train_env.trade_count}'
            })
            
            # ОТЧЕТ КАЖДЫЕ 10 ЭПОХ
            if (episode + 1) % 10 == 0:
                print(f"\n📊 ЭПОХА {episode + 1}/{config['episodes']} ОТЧЕТ:")
                print(f"   💰 Train profit: ${train_profit:.2f}")
                print(f"   🧪 OOS median: ${oos_median:.2f}")
                print(f"   🏆 Best OOS: ${best_median:.2f}")
                print(f"   📈 Epsilon: {agent.epsilon:.4f}")
                print(f"   🔄 Trades: {train_env.trade_count}")
                print(f"   📊 OOS range: ${min(oos_profits):.0f} to ${max(oos_profits):.0f}")
            
            time.sleep(0.05)  # Чтобы видеть прогресс
    
    # Финальное тестирование
    print("\n🔬 LSTM финальное тестирование...")
    final_profits = []
    
    with tqdm(range(0, min(100, len(oos_env.stock.closes) - config['trading_period'] - config['window']), 5),
              desc="🧪 LSTM тест",
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
    
    # LSTM результаты
    median_profit = np.median(final_profits)
    mean_profit = np.mean(final_profits)
    consistency = len([p for p in final_profits if p > 0]) / len(final_profits) * 100
    
    print(f"\n🧠 LSTM v13 РЕЗУЛЬТАТЫ:")
    print(f"   💰 Медианный профит: ${median_profit:.2f}")
    print(f"   📈 Средний профит: ${mean_profit:.2f}")  
    print(f"   🎯 Consistency: {consistency:.1f}%")
    print(f"   🏆 Лучший медианный: ${best_median:.2f}")
    print(f"   📊 Всего тестов: {len(final_profits)}")
    
    # Сохраняем финальную модель
    agent.save("models/tqdm-lstm-v13")
    print(f"💾 LSTM модель сохранена: models/tqdm-lstm-v13")
    
    print("🧠 LSTM v13 ЗАВЕРШЕН!")
    
    return {
        'name': 'LSTM v13',
        'median_profit': median_profit,
        'mean_profit': mean_profit,
        'consistency': consistency,
        'best_median': best_median,
        'total_tests': len(final_profits)
    }

if __name__ == "__main__":
    main() 