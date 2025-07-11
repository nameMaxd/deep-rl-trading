#!/usr/bin/env python3
"""
🔥 MLP АТАКА: v16 с простой многослойной сетью!
🎯 ЦЕЛЬ: Baseline - проверяем могут ли простые MLP конкурировать
"""

import torch
import torch.nn as nn
import numpy as np
from src.rl.env import Env
from src.rl.agent import Agent
import os
from datetime import datetime

class MLPAgent(Agent):
    """MLP агент с простой feedforward архитектурой"""
    
    def __init__(self, obs_space, **kwargs):
        # MLP специфичные параметры
        self.hidden_sizes = kwargs.get('hidden_sizes', [512, 256, 128])
        self.use_batch_norm = kwargs.get('use_batch_norm', True)
        self.use_residual = kwargs.get('use_residual', True)
        
        super().__init__(obs_space, **kwargs)
        
    def _create_network(self):
        """Создаем MLP сеть"""
        
        feature_dim = self.obs_space[1]  # количество фичей
        sequence_len = self.obs_space[2]  # длина последовательности
        
        # Для MLP просто разворачиваем в плоский вектор
        input_size = feature_dim * sequence_len
        
        return MLPQNetwork(
            input_size=input_size,
            hidden_sizes=self.hidden_sizes,
            action_size=3,
            dropout=self.dropout,
            use_batch_norm=self.use_batch_norm,
            use_residual=self.use_residual
        )
    
    def act(self, state, training=True):
        """ФИКСИРОВАННЫЙ epsilon decay БЕЗ adaptive boost!"""
        if training and np.random.random() <= self.epsilon:
            action = np.random.choice(3)
            if self.steps % 1000 == 0:
                print(f"🎲 MLP Random action: {action}, epsilon: {self.epsilon:.3f}")
            return action
        
        # Разворачиваем state для MLP
        state_flat = state.reshape(1, -1)  # (1, features * sequence)
        state_tensor = torch.FloatTensor(state_flat).to(self.device)
        q_values = self.q_network(state_tensor)
        action = q_values.cpu().data.numpy().argmax()
        
        if self.steps % 1000 == 0:
            print(f"🔥 MLP Neural action: {action}, Q: {q_values.cpu().data.numpy()}")
        
        return action
    
    def step(self, state, action, reward, next_state, done):
        """MLP агент шаг с правильным epsilon decay"""
        # Разворачиваем состояния для MLP
        state_flat = state.reshape(-1)
        next_state_flat = next_state.reshape(-1)
        
        self.memory.add(state_flat, action, reward, next_state_flat, done)
        self.steps += 1
        
        # ПРОСТОЙ epsilon decay БЕЗ adaptive boost
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.steps % 1000 == 0:
                print(f"📉 MLP Epsilon decay: {self.epsilon:.3f}")
        
        if len(self.memory) > self.batch_size and self.steps % self.update_freq == 0:
            experiences = self.memory.sample()
            self.learn(experiences)


class ResidualBlock(nn.Module):
    """Residual блок для MLP"""
    
    def __init__(self, size, dropout=0.1, use_batch_norm=True):
        super().__init__()
        
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.dropout = nn.Dropout(dropout)
        self.use_batch_norm = use_batch_norm
        
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(size)
            self.bn2 = nn.BatchNorm1d(size)
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        residual = x
        
        out = self.fc1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        
        # Residual connection
        out = out + residual
        out = self.activation(out)
        
        return out


class MLPQNetwork(nn.Module):
    """Простая но мощная MLP Q-сеть"""
    
    def __init__(self, input_size, hidden_sizes, action_size, dropout=0.1, use_batch_norm=True, use_residual=True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.action_size = action_size
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            in_size = hidden_sizes[i]
            out_size = hidden_sizes[i + 1]
            
            layers.append(nn.Linear(in_size, out_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            # Добавляем residual блоки для глубоких сетей
            if use_residual and out_size >= 128:
                layers.append(ResidualBlock(out_size, dropout, use_batch_norm))
        
        self.features = nn.Sequential(*layers)
        
        # Output layer
        self.output = nn.Linear(hidden_sizes[-1], action_size)
        
        # Специальная инициализация
        self._init_weights()
        
    def _init_weights(self):
        """Продвинутая инициализация весов"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He initialization для ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: (batch, input_size) - уже плоский
        
        # Feature extraction
        features = self.features(x)
        
        # Output
        q_values = self.output(features)
        
        return q_values


def mlp_features(data, window=50):
    """
    🔥 MLP FEATURE SET - максимально информативные статистические фичи
    """
    
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame(data.copy())
    features = []
    
    print("🔥 Создаю MLP-оптимизированные фичи...")
    
    close_prices = df['close'] if 'close' in df.columns else df.iloc[:, 3]
    
    # 1. БАЗОВЫЕ СТАТИСТИЧЕСКИЕ ФИЧИ
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            prices = df[col]
            
            # Сырые нормализованные цены
            normalized = (prices - prices.rolling(window).mean()) / prices.rolling(window).std()
            features.append(normalized.fillna(0).values)
            
            # Returns разных периодов
            for period in [1, 2, 3, 5, 10]:
                returns = prices.pct_change(period).fillna(0)
                features.append(returns.values)
                
            # Z-score (насколько цена отклонена от нормы)
            z_score = (prices - prices.rolling(window).mean()) / prices.rolling(window).std()
            features.append(z_score.fillna(0).values)
    
    # 2. МНОЖЕСТВЕННЫЕ СТАТИСТИКИ СКОЛЬЗЯЩИХ ОКОН
    for period in [5, 10, 15, 20, 30, 50]:
        # Среднее
        ma = close_prices.rolling(period).mean()
        ma_ratio = (close_prices / ma - 1).fillna(0)
        features.append(ma_ratio.values)
        
        # Стандартное отклонение
        std = close_prices.rolling(period).std()
        std_norm = (std - std.rolling(window).mean()) / std.rolling(window).std()
        features.append(std_norm.fillna(0).values)
        
        # Минимум и максимум
        rolling_min = close_prices.rolling(period).min()
        rolling_max = close_prices.rolling(period).max()
        
        min_ratio = (close_prices - rolling_min) / (rolling_max - rolling_min)
        features.append(min_ratio.fillna(0.5).values)
        
        # Медиана
        median = close_prices.rolling(period).median()
        median_ratio = (close_prices / median - 1).fillna(0)
        features.append(median_ratio.values)
        
        # Квантили
        q25 = close_prices.rolling(period).quantile(0.25)
        q75 = close_prices.rolling(period).quantile(0.75)
        
        iqr_position = (close_prices - q25) / (q75 - q25)
        features.append(iqr_position.fillna(0.5).values)
    
    # 3. ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ МНОЖЕСТВЕННЫХ ПЕРИОДОВ
    
    # RSI разных периодов
    for period in [7, 14, 21, 28]:
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).fillna(50) / 100
        features.append(rsi.values)
        
        # RSI divergence
        rsi_ma = rsi.rolling(10).mean()
        rsi_div = (rsi - rsi_ma).fillna(0)
        features.append(rsi_div.values)
    
    # MACD системы
    for fast, slow, signal in [(12, 26, 9), (5, 13, 5), (19, 39, 9)]:
        ema_fast = close_prices.ewm(span=fast).mean()
        ema_slow = close_prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        
        # MACD line
        macd_norm = (macd - macd.rolling(window).mean()) / macd.rolling(window).std()
        features.append(macd_norm.fillna(0).values)
        
        # MACD signal
        signal_norm = (macd_signal - macd_signal.rolling(window).mean()) / macd_signal.rolling(window).std()
        features.append(signal_norm.fillna(0).values)
        
        # MACD histogram
        histogram = macd - macd_signal
        hist_norm = (histogram - histogram.rolling(window).mean()) / histogram.rolling(window).std()
        features.append(hist_norm.fillna(0).values)
    
    # 4. ОБЪЕМНЫЕ ИНДИКАТОРЫ (если есть)
    if 'volume' in df.columns:
        vol = df['volume']
        
        # Volume statistics
        vol_norm = (vol - vol.rolling(window).mean()) / vol.rolling(window).std()
        features.append(vol_norm.fillna(0).values)
        
        # Volume trends
        for period in [5, 10, 20]:
            vol_ma = vol.rolling(period).mean()
            vol_trend = (vol / vol_ma - 1).fillna(0)
            features.append(vol_trend.values)
            
            # Volume momentum
            vol_momentum = vol.pct_change(period).fillna(0)
            features.append(vol_momentum.values)
        
        # Price-Volume relationships
        for period in [10, 20]:
            pv_corr = close_prices.rolling(period).corr(vol).fillna(0)
            features.append(pv_corr.values)
            
        # On-Balance Volume
        obv = (vol * np.sign(close_prices.diff())).cumsum()
        obv_norm = (obv - obv.rolling(window).mean()) / obv.rolling(window).std()
        features.append(obv_norm.fillna(0).values)
    
    # 5. ВОЛАТИЛЬНОСТЬ И РИСК МЕТРИКИ
    returns = close_prices.pct_change().fillna(0)
    
    for period in [5, 10, 20, 30]:
        # Volatility
        vol = returns.rolling(period).std().fillna(0)
        features.append(vol.values)
        
        # Downside volatility
        downside_vol = returns[returns < 0].rolling(period).std().fillna(0)
        features.append(downside_vol.values)
        
        # Skewness
        skew = returns.rolling(period).skew().fillna(0)
        features.append(skew.values)
        
        # Kurtosis
        kurt = returns.rolling(period).kurt().fillna(0)
        features.append(kurt.values)
        
        # Sharpe ratio (упрощенный)
        sharpe = returns.rolling(period).mean() / vol
        features.append(sharpe.fillna(0).values)
    
    # 6. ПРАЙС ЭКШН ПАТТЕРНЫ
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        # Свечные паттерны
        body = abs(df['close'] - df['open'])
        full_range = df['high'] - df['low']
        
        # Body ratio
        body_ratio = body / full_range
        features.append(body_ratio.fillna(0).values)
        
        # Upper shadow
        upper_shadow = df[['open', 'close']].max(axis=1)
        upper_wick = (df['high'] - upper_shadow) / full_range
        features.append(upper_wick.fillna(0).values)
        
        # Lower shadow
        lower_shadow = df[['open', 'close']].min(axis=1)
        lower_wick = (lower_shadow - df['low']) / full_range
        features.append(lower_wick.fillna(0).values)
        
        # Gap analysis
        gaps = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        features.append(gaps.fillna(0).values)
    
    # 7. ВРЕМЕННЫЕ И ЦИКЛИЧЕСКИЕ ФИЧИ
    
    # Position in session
    session_pos = np.arange(len(df)) / len(df)
    features.append(session_pos)
    
    # Time-based features
    try:
        if hasattr(df.index, 'weekday'):
            # Day of week (encoded)
            for i in range(7):
                day_feature = (df.index.weekday == i).astype(float)
                features.append(day_feature.values)
            
            # Month (encoded)
            for i in range(1, 13):
                month_feature = (df.index.month == i).astype(float)
                features.append(month_feature.values)
    except:
        # Fallback artificial time features
        weekday = np.arange(len(df)) % 7
        month = (np.arange(len(df)) // 22) % 12
        
        for i in range(7):
            features.append((weekday == i).astype(float))
        for i in range(12):
            features.append((month == i).astype(float))
    
    # Объединяем все фичи
    feature_matrix = np.column_stack(features)
    feature_count = feature_matrix.shape[1]
    
    print(f"🔥 Создано {feature_count} MLP статистических фичей!")
    print(f"   Базовые цены: ~28 фичей")
    print(f"   Скользящие статистики: ~30 фичей")
    print(f"   Технические индикаторы: ~21 фичей")
    print(f"   Объемы: ~15 фичей")
    print(f"   Волатильность/риск: ~20 фичей")
    print(f"   Прайс экшн: ~5 фичей")
    print(f"   Временные: ~20 фичей")
    
    # Проверяем на NaN и Inf
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return feature_matrix


def main():
    """MLP эксперимент v16"""
    
    print("🔥 MLP АТАКА v16: Простая но мощная архитектура!")
    print("🎯 ЦЕЛЬ: Baseline - могут ли простые MLP конкурировать?")
    print("💪 Глубокие residual MLP с batch normalization")
    print("=" * 60)
    
    # Параметры MLP эксперимента
    config = {
        'episodes': 160,
        'trading_period': 80,   # короче для быстроты
        'window': 35,           # компактное окно
        'target_profit': 600,   # реалистичная цель
        'commission': 0.0002,
        
        # MLP архитектура
        'hidden_sizes': [512, 256, 128, 64], # Глубокая сеть
        'use_batch_norm': True,
        'use_residual': True,
        'dropout': 0.2,         # больше dropout для регуляризации
        
        # Обучение
        'lr': 0.001,            # стандартный LR
        'epsilon': 0.8,         # высокий старт
        'epsilon_min': 0.01,    # стандартный финиш
        'epsilon_decay': 0.995, # медленный decay
        'gamma': 0.95,
        'memory_size': 10000,
        'batch_size': 256,      # стандартный batch
        'update_freq': 5
    }
    
    print(f"📊 MLP КОНФИГУРАЦИЯ:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print("=" * 60)
    
    # Создаем окружение с MLP фичами
    train_env = Env(
        csv_paths=["GOOG_2010_2024-06.csv"],
        fee=config['commission'],
        trading_period=config['trading_period'],
        window=config['window'],
        feature_extractor=mlp_features
    )
    
    oos_env = Env(
        csv_paths=["GOOG_2024-07_2025-04.csv"],
        fee=config['commission'],
        trading_period=config['trading_period'],
        window=config['window'],
        feature_extractor=mlp_features
    )
    
    print(f"📊 MLP окружение:")
    print(f"   Train observation space: {train_env.stock.obs_space}")
    print(f"   OOS observation space: {oos_env.stock.obs_space}")
    print(f"   Flattened input size: {train_env.stock.obs_space[1] * train_env.stock.obs_space[2]}")
    print(f"   Target profit: ${config['target_profit']}")
    
    # Создаем MLP агента
    agent = MLPAgent(
        obs_space=train_env.stock.obs_space,
        **{k: v for k, v in config.items() if k not in ['episodes', 'trading_period', 'window', 'target_profit', 'commission']}
    )
    
    print(f"🔥 MLP агент создан:")
    print(f"   Архитектура: {config['hidden_sizes']}")
    print(f"   Batch norm: {config['use_batch_norm']}, Residual: {config['use_residual']}")
    print(f"   Epsilon: {config['epsilon']} -> {config['epsilon_min']} (decay: {config['epsilon_decay']})")
    print(f"   БЕЗ adaptive epsilon boost!")
    
    # Логирование
    log_file = f"models/google-trading-v16-mlp.log"
    model_name = "google-trading-v16-mlp"
    
    best_median = -float('inf')
    stability_count = 0
    
    print(f"\n🔥 Начинаю MLP обучение...")
    
    with open(log_file, "w", encoding='utf-8') as f:
        f.write(f"🔥 MLP Training Log v16 - {datetime.now()}\n")
        f.write(f"Config: {config}\n")
        f.write("💪 Deep Residual MLP with BatchNorm\n")
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
                
                while not oos_env.stock.done:
                    state = oos_env.stock.get_state()
                    action = agent.act(state, training=False)  # БЕЗ exploration
                    reward, profit, trade_made = oos_env.stock.step(action)
                
                oos_profits.append(oos_env.stock.total_profit)
            
            median_oos = np.median(oos_profits)
            mean_oos = np.mean(oos_profits)
            consistency = (np.array(oos_profits) > 0).mean() * 100
            
            # Проверяем стабильность
            if median_oos > 100:  # хорошая прибыльность для MLP
                stability_count += 1
            
            # Сохраняем лучшую модель
            if median_oos > best_median:
                best_median = median_oos
                agent.save(f"models/{model_name}_best")
                print(f"💾 Новая лучшая MLP медиана: ${median_oos:.0f} (эпизод {episode})")
            
            # Логирование
            log_entry = (
                f"Ep: {episode:3d} | Train: ${train_profit:4.0f} | OOS Med: ${median_oos:4.0f}\\n"
                f"    Train: {trades} trades, {win_rate:.1f}% win\\n"
                f"    OOS: Med ${median_oos:.0f}, Mean ${mean_oos:.0f}, Consistency {consistency:.1f}%\\n"
                f"    Epsilon: {agent.epsilon:.4f} | Best median: ${best_median:.0f}\\n"
                f"    All OOS: {[int(p) for p in oos_profits]}\\n"
                + "-" * 60
            )
            
            print(log_entry.replace("\\n", "\n"))
            
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(log_entry + "\n")
    
    # Финальное сохранение
    agent.save(f"models/{model_name}")
    
    print(f"\n🔥 MLP обучение v16 завершено!")
    print(f"💾 Модель сохранена: models/{model_name}")
    print(f"🏆 Лучшая медиана: models/{model_name}_best")
    print(f"📊 Лог сохранен: {log_file}")
    print(f"🎯 Стабильных результатов: {stability_count}/{config['episodes']//10}")
    
    # Финальное тестирование
    print(f"\n🔬 Финальное MLP тестирование...")
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
    
    print(f"📈 ФИНАЛЬНЫЕ MLP РЕЗУЛЬТАТЫ:")
    print(f"   Медианный профит: ${final_median:.2f}")
    print(f"   Consistency: {final_consistency:.1f}%")
    print(f"   Стандартное отклонение: ${final_std:.2f}")
    print(f"   Диапазон: ${min(final_profits):.0f} - ${max(final_profits):.0f}")
    
    if final_consistency > 65 and final_median > 300:
        print("✅ MLP показал отличные результаты!")
    elif final_consistency > 50 and final_median > 150:
        print("🟡 MLP показал средние результаты")
    else:
        print("❌ MLP требует доработки...")


if __name__ == "__main__":
    main() 