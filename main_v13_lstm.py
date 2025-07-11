#!/usr/bin/env python3
"""
🧠 LSTM АТАКА: v13 с исправленным epsilon!
🎯 ЦЕЛЬ: Простая LSTM архитектура БЕЗ adaptive epsilon
"""

import torch
import torch.nn as nn
import numpy as np
from src.rl.env import Env
from src.rl.agent import Agent
import os
from datetime import datetime

class LSTMAgent(Agent):
    """LSTM агент без всяких трансформеров"""
    
    def __init__(self, obs_space, **kwargs):
        # Переопределяем архитектуру
        self.lstm_hidden = kwargs.get('lstm_hidden', 128)
        self.lstm_layers = kwargs.get('lstm_layers', 2)
        self.fc_hidden = kwargs.get('fc_hidden', 64)
        
        super().__init__(obs_space, **kwargs)
        
    def _create_network(self):
        """Создаем LSTM сеть вместо трансформера"""
        
        feature_dim = self.obs_space[1]  # количество фичей
        sequence_len = self.obs_space[2]  # длина последовательности
        
        return LSTMQNetwork(
            feature_dim=feature_dim,
            sequence_len=sequence_len,
            lstm_hidden=self.lstm_hidden,
            lstm_layers=self.lstm_layers,
            fc_hidden=self.fc_hidden,
            action_size=3,
            dropout=self.dropout
        )
    
    def act(self, state, training=True):
        """ФИКСИРОВАННЫЙ epsilon decay БЕЗ adaptive boost!"""
        if training and np.random.random() <= self.epsilon:
            action = np.random.choice(3)
            if self.steps % 1000 == 0:
                print(f"🎲 Random action: {action}, epsilon: {self.epsilon:.3f}")
            return action
        
        # Нейросеть
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        action = q_values.cpu().data.numpy().argmax()
        
        if self.steps % 1000 == 0:
            print(f"🧠 Neural action: {action}, Q: {q_values.cpu().data.numpy()}")
        
        return action
    
    def step(self, state, action, reward, next_state, done):
        """Шаг LSTM агента с ПРАВИЛЬНЫМ epsilon decay"""
        self.memory.add(state, action, reward, next_state, done)
        self.steps += 1
        
        # ПРОСТОЙ epsilon decay БЕЗ adaptive boost
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.steps % 1000 == 0:
                print(f"📉 Epsilon decay: {self.epsilon:.3f}")
        
        if len(self.memory) > self.batch_size and self.steps % self.update_freq == 0:
            experiences = self.memory.sample()
            self.learn(experiences)


class LSTMQNetwork(nn.Module):
    """Простая LSTM Q-сеть"""
    
    def __init__(self, feature_dim, sequence_len, lstm_hidden, lstm_layers, fc_hidden, action_size, dropout=0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.sequence_len = sequence_len
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        
        # Input embedding (optional)
        self.input_norm = nn.LayerNorm(feature_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False
        )
        
        # FC layers
        self.fc1 = nn.Linear(lstm_hidden, fc_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden // 2)
        self.fc_out = nn.Linear(fc_hidden // 2, action_size)
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # x: (batch, timesteps, features)
        batch_size = x.size(0)
        
        # Normalize input
        x = self.input_norm(x)
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take last output
        last_output = lstm_out[:, -1, :]  # (batch, lstm_hidden)
        
        # FC layers
        x = self.activation(self.fc1(last_output))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        q_values = self.fc_out(x)
        
        return q_values


def enhanced_features(data, window=30):
    """
    🚀 ENHANCED FEATURE SET для LSTM
    Все фичи из exp05.md + временные лаги
    """
    
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame(data.copy())
    features = []
    
    print("🚀 Создаю ENHANCED LSTM фичи...")
    
    # 1. БАЗОВЫЕ ЦЕНЫ (нормализованные)
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            normalized = (df[col] - df[col].rolling(window).mean()) / df[col].rolling(window).std()
            features.append(normalized.fillna(0).values)
    
    # 2. ОБЪЕМЫ
    if 'volume' in df.columns:
        vol_norm = (df['volume'] - df['volume'].rolling(window).mean()) / df['volume'].rolling(window).std()
        features.append(vol_norm.fillna(0).values)
        
        # Volume rate of change
        vol_roc = df['volume'].pct_change(5).fillna(0)
        features.append(vol_roc.values)
    
    # 3. ВРЕМЕННЫЕ ЛАГИ (1, 2, 3, 5, 10 периодов назад)
    close_prices = df['close'] if 'close' in df.columns else df.iloc[:, 3]
    for lag in [1, 2, 3, 5, 10]:
        lagged = close_prices.shift(lag)
        lag_returns = ((close_prices - lagged) / lagged).fillna(0)
        features.append(lag_returns.values)
        print(f"   ✅ Лаг {lag} периодов")
    
    # 4. СКОЛЬЗЯЩИЕ СРЕДНИЕ разных периодов
    for period in [5, 10, 20, 50]:
        if len(close_prices) > period:
            ma = close_prices.rolling(period).mean()
            ma_ratio = (close_prices / ma - 1).fillna(0)
            features.append(ma_ratio.values)
            print(f"   ✅ MA-{period}")
    
    # 5. ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
    # RSI
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).fillna(50) / 100  # нормализуем 0-1
    features.append(rsi.values)
    
    # MACD
    ema12 = close_prices.ewm(span=12).mean()
    ema26 = close_prices.ewm(span=26).mean()
    macd = (ema12 - ema26).fillna(0)
    macd_norm = (macd - macd.rolling(window).mean()) / macd.rolling(window).std()
    features.append(macd_norm.fillna(0).values)
    
    # 6. ВОЛАТИЛЬНОСТЬ (скользящая)
    returns = close_prices.pct_change().fillna(0)
    volatility = returns.rolling(window=20).std().fillna(0)
    features.append(volatility.values)
    
    # 7. ВРЕМЕННЫЕ ФИЧИ
    # Позиция в сессии (0-1)
    session_pos = np.arange(len(df)) / len(df)
    features.append(session_pos)
    
    # День недели (если есть индекс с датами)
    try:
        if hasattr(df.index, 'weekday'):
            weekday = df.index.weekday / 6.0  # нормализуем 0-1
            features.append(weekday.values)
        else:
            # Создаем искусственный день недели
            weekday = (np.arange(len(df)) % 7) / 6.0
            features.append(weekday)
    except:
        weekday = (np.arange(len(df)) % 7) / 6.0
        features.append(weekday)
    
    # 8. MOMENTUM фичи
    # Rate of Change разных периодов
    for period in [5, 10, 20]:
        roc = close_prices.pct_change(period).fillna(0)
        features.append(roc.values)
    
    # Объединяем все фичи
    feature_matrix = np.column_stack(features)
    feature_count = feature_matrix.shape[1]
    
    print(f"🎯 Создано {feature_count} ENHANCED фичей для LSTM:")
    feature_names = [
        'open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm', 'volume_roc',
        'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_10',
        'ma_5', 'ma_10', 'ma_20', 'ma_50',
        'rsi', 'macd', 'volatility', 'session_pos', 'weekday',
        'roc_5', 'roc_10', 'roc_20'
    ]
    
    for i, name in enumerate(feature_names[:feature_count]):
        print(f"   {i+1:2d}. {name}")
    
    # Проверяем на NaN и Inf
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return feature_matrix


def main():
    """LSTM эксперимент v13"""
    
    print("🧠 LSTM АТАКА v13: Простая архитектура!")
    print("🎯 ЦЕЛЬ: LSTM + фиксированный epsilon + enhanced фичи")
    print("🔧 БЕЗ adaptive epsilon boost!")
    print("=" * 60)
    
    # Параметры LSTM эксперимента
    config = {
        'episodes': 200,
        'trading_period': 90,  # средний период
        'window': 40,          # чуть больше окно
        'target_profit': 500,  # умеренная цель
        'commission': 0.0002,
        
        # LSTM архитектура
        'lstm_hidden': 128,
        'lstm_layers': 2,
        'fc_hidden': 64,
        'dropout': 0.1,
        
        # Обучение
        'lr': 0.001,
        'epsilon': 0.9,        # стартуем высоко
        'epsilon_min': 0.01,   # но опускаем низко
        'epsilon_decay': 0.995, # медленный decay
        'gamma': 0.95,
        'memory_size': 10000,
        'batch_size': 256,
        'update_freq': 5
    }
    
    print(f"📊 LSTM КОНФИГУРАЦИЯ:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print("=" * 60)
    
    # Создаем окружение с enhanced фичами
    train_env = Env(
        csv_paths=["GOOG_2010_2024-06.csv"],
        fee=config['commission'],
        trading_period=config['trading_period'],
        window=config['window'],
        feature_extractor=enhanced_features
    )
    
    oos_env = Env(
        csv_paths=["GOOG_2024-07_2025-04.csv"],
        fee=config['commission'],
        trading_period=config['trading_period'],
        window=config['window'],
        feature_extractor=enhanced_features
    )
    
    print(f"📊 LSTM окружение:")
    print(f"   Train observation space: {train_env.stock.obs_space}")
    print(f"   OOS observation space: {oos_env.stock.obs_space}")
    print(f"   Target profit: ${config['target_profit']}")
    
    # Создаем LSTM агента
    agent = LSTMAgent(
        obs_space=train_env.stock.obs_space,
        **{k: v for k, v in config.items() if k not in ['episodes', 'trading_period', 'window', 'target_profit', 'commission']}
    )
    
    print(f"🧠 LSTM агент создан:")
    print(f"   Архитектура: LSTM({config['lstm_hidden']}) x {config['lstm_layers']}")
    print(f"   FC: {config['fc_hidden']} -> {config['fc_hidden']//2} -> 3")
    print(f"   Epsilon: {config['epsilon']} -> {config['epsilon_min']} (decay: {config['epsilon_decay']})")
    
    # Логирование
    log_file = f"models/google-trading-v13-lstm.log"
    model_name = "google-trading-v13-lstm"
    
    best_median = -float('inf')
    stability_count = 0
    
    print(f"\n🧠 Начинаю LSTM обучение...")
    
    with open(log_file, "w", encoding='utf-8') as f:
        f.write(f"🧠 LSTM Training Log v13 - {datetime.now()}\n")
        f.write(f"Config: {config}\n")
        f.write("=" * 80 + "\n")
    
    for episode in range(config['episodes']):
        # Тренировка
        train_env.stock.reset()
        total_reward = 0
        trades = 0
        wins = 0
        
        while not train_env.stock.done:
            state = train_env.stock.get_state()
            action = agent.act(state)
            reward, profit, trade_made = train_env.stock.step(action)
            next_state = train_env.stock.get_state()
            
            agent.step(state, action, reward, next_state, train_env.stock.done)
            
            total_reward += reward
            if trade_made:
                trades += 1
                if profit > 0:
                    wins += 1
        
        train_profit = train_env.stock.total_profit
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        # OOS тестирование каждые 10 эпизодов
        if episode % 10 == 0:
            # Тестируем на 5 разных позициях
            oos_profits = []
            for start_pos in range(0, min(50, len(oos_env.stock.closes) - config['trading_period'] - config['window']), 10):
                oos_env.stock.reset_fixed(start_pos)
                oos_profit = 0
                
                while not oos_env.stock.done:
                    state = oos_env.stock.get_state()
                    action = agent.act(state, training=False)  # БЕЗ exploration
                    reward, profit, trade_made = oos_env.stock.step(action)
                    oos_profit += profit
                
                oos_profits.append(oos_env.stock.total_profit)
            
            median_oos = np.median(oos_profits)
            mean_oos = np.mean(oos_profits)
            consistency = (np.array(oos_profits) > 0).mean() * 100
            
            # Проверяем стабильность
            if median_oos > 30:  # минимальная прибыльность
                stability_count += 1
            
            # Сохраняем лучшую модель
            if median_oos > best_median:
                best_median = median_oos
                agent.save(f"models/{model_name}_best")
                print(f"💾 Новая лучшая медиана: ${median_oos:.0f} (эпизод {episode})")
            
            # Логирование
            log_entry = (
                f"Ep: {episode:3d} | Train: ${train_profit:4.0f} | OOS Med: ${median_oos:4.0f}\\n"
                f"    Train: {trades} trades, {win_rate:.1f}% win\\n"
                f"    OOS: Med ${median_oos:.0f}, Mean ${mean_oos:.0f}, Consistency {consistency:.1f}%\\n"
                f"    Epsilon: {agent.epsilon:.3f} | Best median: ${best_median:.0f}\\n"
                f"    All OOS: {[int(p) for p in oos_profits]}\\n"
                + "-" * 60
            )
            
            print(log_entry.replace("\\n", "\n"))
            
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(log_entry + "\n")
    
    # Финальное сохранение
    agent.save(f"models/{model_name}")
    
    print(f"\n🧠 LSTM обучение v13 завершено!")
    print(f"💾 Модель сохранена: models/{model_name}")
    print(f"🏆 Лучшая медиана: models/{model_name}_best")
    print(f"📊 Лог сохранен: {log_file}")
    print(f"🎯 Стабильных результатов: {stability_count}/{config['episodes']//10}")
    
    # Финальное тестирование
    print(f"\n🔬 Финальное LSTM тестирование...")
    final_profits = []
    for start_pos in range(0, min(100, len(oos_env.stock.closes) - config['trading_period'] - config['window']), 5):
        oos_env.stock.reset_fixed(start_pos)
        
        while not oos_env.stock.done:
            state = oos_env.stock.get_state()
            action = agent.act(state, training=False)
            reward, profit, trade_made = oos_env.stock.step(action)
        
        final_profits.append(oos_env.stock.total_profit)
    
    final_median = np.median(final_profits)
    final_consistency = (np.array(final_profits) > 0).mean() * 100
    final_std = np.std(final_profits)
    
    print(f"📈 ФИНАЛЬНЫЕ LSTM РЕЗУЛЬТАТЫ:")
    print(f"   Медианный профит: ${final_median:.2f}")
    print(f"   Consistency: {final_consistency:.1f}%")
    print(f"   Стандартное отклонение: ${final_std:.2f}")
    print(f"   Диапазон: ${min(final_profits):.0f} - ${max(final_profits):.0f}")
    
    if final_consistency > 60 and final_median > 100:
        print("✅ LSTM показал хорошие результаты!")
    else:
        print("❌ LSTM требует доработки...")


if __name__ == "__main__":
    main() 