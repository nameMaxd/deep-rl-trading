#!/usr/bin/env python3
"""
🤖 ТРАНСФОРМЕР ФИКС v14: Убираем adaptive epsilon!
🎯 ЦЕЛЬ: Мощная архитектура БЕЗ epsilon артефактов
"""

import torch
import numpy as np
from src.rl.env import Env
from src.rl.agent import Agent
import os
from datetime import datetime

class FixedTransformerAgent(Agent):
    """Трансформер агент с ФИКСИРОВАННЫМ epsilon decay"""
    
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
            print(f"🤖 Transformer action: {action}, Q: {q_values.cpu().data.numpy()}")
        
        return action
    
    def step(self, state, action, reward, next_state, done):
        """ПРАВИЛЬНЫЙ epsilon decay БЕЗ adaptive boost"""
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


def mega_features(data, window=50):
    """
    🚀 МЕГА FEATURE SET для трансформера
    Все возможные фичи для максимальной мощности
    """
    
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame(data.copy())
    features = []
    
    print("🚀 Создаю МЕГА фичи для трансформера...")
    
    # 1. БАЗОВЫЕ ЦЕНЫ (нормализованные + сырые)
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            # Нормализованные
            normalized = (df[col] - df[col].rolling(window).mean()) / df[col].rolling(window).std()
            features.append(normalized.fillna(0).values)
            
            # Returns
            returns = df[col].pct_change().fillna(0)
            features.append(returns.values)
    
    # 2. ОБЪЕМЫ (множественные трансформации)
    if 'volume' in df.columns:
        vol = df['volume']
        
        # Нормализованный объем
        vol_norm = (vol - vol.rolling(window).mean()) / vol.rolling(window).std()
        features.append(vol_norm.fillna(0).values)
        
        # Volume ROC разных периодов
        for period in [1, 3, 5, 10]:
            vol_roc = vol.pct_change(period).fillna(0)
            features.append(vol_roc.values)
        
        # Volume moving averages ratios
        for period in [5, 10, 20]:
            vol_ma = vol.rolling(period).mean()
            vol_ratio = (vol / vol_ma - 1).fillna(0)
            features.append(vol_ratio.values)
    
    # 3. МНОЖЕСТВЕННЫЕ ВРЕМЕННЫЕ ЛАГИ
    close_prices = df['close'] if 'close' in df.columns else df.iloc[:, 3]
    
    # Лаги разных периодов
    for lag in [1, 2, 3, 5, 7, 10, 15, 20]:
        lagged = close_prices.shift(lag)
        lag_returns = ((close_prices - lagged) / lagged).fillna(0)
        features.append(lag_returns.values)
    
    # 4. МНОЖЕСТВЕННЫЕ СКОЛЬЗЯЩИЕ СРЕДНИЕ
    for period in [3, 5, 8, 10, 15, 20, 30, 50, 100]:
        if len(close_prices) > period:
            ma = close_prices.rolling(period).mean()
            ma_ratio = (close_prices / ma - 1).fillna(0)
            features.append(ma_ratio.values)
            
            # MA slope (наклон)
            ma_slope = ma.diff(5).fillna(0) / ma.shift(5)
            features.append(ma_slope.fillna(0).values)
    
    # 5. ПРОДВИНУТЫЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
    
    # RSI множественных периодов
    for period in [7, 14, 21]:
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).fillna(50) / 100
        features.append(rsi.values)
    
    # MACD семейство
    for fast, slow in [(8, 21), (12, 26), (19, 39)]:
        ema_fast = close_prices.ewm(span=fast).mean()
        ema_slow = close_prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        
        # MACD line
        macd_norm = (macd - macd.rolling(window).mean()) / macd.rolling(window).std()
        features.append(macd_norm.fillna(0).values)
        
        # MACD signal
        signal = macd.ewm(span=9).mean()
        signal_norm = (signal - signal.rolling(window).mean()) / signal.rolling(window).std()
        features.append(signal_norm.fillna(0).values)
        
        # MACD histogram
        histogram = macd - signal
        hist_norm = (histogram - histogram.rolling(window).mean()) / histogram.rolling(window).std()
        features.append(hist_norm.fillna(0).values)
    
    # 6. ВОЛАТИЛЬНОСТЬ (множественная)
    returns = close_prices.pct_change().fillna(0)
    
    for period in [5, 10, 15, 20, 30]:
        volatility = returns.rolling(window=period).std().fillna(0)
        features.append(volatility.values)
        
        # Volatility of volatility
        vol_vol = volatility.rolling(window=10).std().fillna(0)
        features.append(vol_vol.values)
    
    # 7. BOLLINGER BANDS разных периодов
    for period in [10, 20, 30]:
        bb_middle = close_prices.rolling(period).mean()
        bb_std = close_prices.rolling(period).std()
        
        # Position in bands
        bb_position = ((close_prices - bb_middle) / (2 * bb_std)).fillna(0)
        features.append(bb_position.values)
        
        # Band width
        bb_width = (bb_std / bb_middle).fillna(0)
        features.append(bb_width.values)
    
    # 8. MOMENTUM фичи (расширенные)
    
    # ROC разных периодов
    for period in [1, 3, 5, 10, 15, 20, 30]:
        roc = close_prices.pct_change(period).fillna(0)
        features.append(roc.values)
    
    # Stochastic разных периодов
    for period in [5, 14, 21]:
        if len(df) > period:
            low_min = df['low'].rolling(period).min() if 'low' in df.columns else close_prices.rolling(period).min()
            high_max = df['high'].rolling(period).max() if 'high' in df.columns else close_prices.rolling(period).max()
            stoch_k = ((close_prices - low_min) / (high_max - low_min)).fillna(0.5)
            features.append(stoch_k.values)
            
            # Stoch D (smoothed)
            stoch_d = stoch_k.rolling(3).mean().fillna(0.5)
            features.append(stoch_d.values)
    
    # 9. ЦЕНОВЫЕ ПАТТЕРНЫ
    
    if 'high' in df.columns and 'low' in df.columns and 'open' in df.columns:
        # Typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        typical_norm = (typical_price - typical_price.rolling(window).mean()) / typical_price.rolling(window).std()
        features.append(typical_norm.fillna(0).values)
        
        # Price gaps
        price_gaps = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        features.append(price_gaps.fillna(0).values)
        
        # High-Low spread
        hl_spread = (df['high'] - df['low']) / df['close']
        features.append(hl_spread.fillna(0).values)
        
        # Open-Close ratio
        oc_ratio = (df['open'] - df['close']) / df['close']
        features.append(oc_ratio.fillna(0).values)
        
        # Doji pattern
        doji = np.abs(df['open'] - df['close']) / (df['high'] - df['low'])
        features.append(doji.fillna(0).values)
    
    # 10. ВРЕМЕННЫЕ ФИЧИ (расширенные)
    
    # Позиция в сессии
    session_pos = np.arange(len(df)) / len(df)
    features.append(session_pos)
    
    # Синусоидальные временные паттерны
    daily_cycle = np.sin(2 * np.pi * np.arange(len(df)) / 252)  # годовой цикл
    features.append(daily_cycle)
    
    weekly_cycle = np.sin(2 * np.pi * np.arange(len(df)) / 5)   # недельный цикл
    features.append(weekly_cycle)
    
    # Отсчет времени от начала
    time_from_start = np.arange(len(df)) / 252  # в годах
    features.append(time_from_start)
    
    # День недели
    try:
        if hasattr(df.index, 'weekday'):
            weekday_sin = np.sin(2 * np.pi * df.index.weekday / 7)
            weekday_cos = np.cos(2 * np.pi * df.index.weekday / 7)
        else:
            weekday_sin = np.sin(2 * np.pi * (np.arange(len(df)) % 7) / 7)
            weekday_cos = np.cos(2 * np.pi * (np.arange(len(df)) % 7) / 7)
        
        features.append(weekday_sin)
        features.append(weekday_cos)
    except:
        weekday_sin = np.sin(2 * np.pi * (np.arange(len(df)) % 7) / 7)
        weekday_cos = np.cos(2 * np.pi * (np.arange(len(df)) % 7) / 7)
        features.append(weekday_sin)
        features.append(weekday_cos)
    
    # Объединяем все фичи
    feature_matrix = np.column_stack(features)
    feature_count = feature_matrix.shape[1]
    
    print(f"🤖 Создано {feature_count} МЕГА фичей для трансформера!")
    print(f"   Базовые цены: 8 фичей")
    print(f"   Объемы: ~15 фичей")
    print(f"   Лаги: 8 фичей")
    print(f"   MA: ~27 фичей")
    print(f"   Технические: ~30 фичей")
    print(f"   Волатильность: ~15 фичей")
    print(f"   Bollinger: ~12 фичей")
    print(f"   Momentum: ~20 фичей")
    print(f"   Паттерны: ~5 фичей")
    print(f"   Временные: ~8 фичей")
    
    # Проверяем на NaN и Inf
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return feature_matrix


def main():
    """Фиксированный трансформер эксперимент v14"""
    
    print("🤖 ТРАНСФОРМЕР ФИКС v14: БЕЗ adaptive epsilon!")
    print("🎯 ЦЕЛЬ: Мощная архитектура + правильный epsilon")
    print("🔧 УБИРАЕМ adaptive epsilon boost навсегда!")
    print("=" * 60)
    
    # Параметры мощного трансформера
    config = {
        'episodes': 250,
        'trading_period': 120,  # полный период
        'window': 50,           # полное окно
        'target_profit': 1000,  # амбициозная цель
        'commission': 0.0002,
        
        # Мощная трансформер архитектура
        'embeddings': 64,       # много эмбеддингов
        'heads': 4,             # много голов
        'layers': 3,            # много слоев
        'fwex': 256,           # большая FC сеть
        'dropout': 0.1,
        'neurons': 256,
        
        # Обучение
        'lr': 0.0005,          # низкий LR для стабильности
        'epsilon': 0.8,        # высокий старт
        'epsilon_min': 0.005,  # очень низкий финиш
        'epsilon_decay': 0.9965, # очень медленный decay
        'gamma': 0.95,
        'memory_size': 15000,
        'batch_size': 512,
        'update_freq': 3
    }
    
    print(f"📊 МОЩНАЯ ТРАНСФОРМЕР КОНФИГУРАЦИЯ:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print("=" * 60)
    
    # Создаем окружение стандартно (мега фичи уже включены в stock.py)
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
    
    print(f"📊 МОЩНОЕ окружение:")
    print(f"   Train observation space: {train_env.stock.obs_space}")
    print(f"   OOS observation space: {oos_env.stock.obs_space}")
    print(f"   Target profit: ${config['target_profit']}")
    
    # Создаем фиксированного трансформер агента
    agent = FixedTransformerAgent(
        obs_space=train_env.stock.obs_space,
        **{k: v for k, v in config.items() if k not in ['episodes', 'trading_period', 'window', 'target_profit', 'commission']}
    )
    
    print(f"🤖 МОЩНЫЙ трансформер агент создан:")
    print(f"   Архитектура: {config['embeddings']} emb, {config['heads']} heads, {config['layers']} layers")
    print(f"   FC: {config['fwex']} -> {config['neurons']} -> 3")
    print(f"   Epsilon: {config['epsilon']} -> {config['epsilon_min']} (decay: {config['epsilon_decay']})")
    print(f"   БЕЗ adaptive epsilon boost!")
    
    # Логирование
    log_file = f"models/google-trading-v14-transformer-fixed.log"
    model_name = "google-trading-v14-transformer-fixed"
    
    best_median = -float('inf')
    stability_count = 0
    
    print(f"\n🤖 Начинаю МОЩНОЕ трансформер обучение...")
    
    with open(log_file, "w", encoding='utf-8') as f:
        f.write(f"🤖 Fixed Transformer Training Log v14 - {datetime.now()}\n")
        f.write(f"Config: {config}\n")
        f.write("🔧 БЕЗ adaptive epsilon boost!\n")
        f.write("=" * 80 + "\n")
    
    for episode in range(config['episodes']):
        # Тренировка
        state = train_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, trade_action, reward, done = train_env.step(action)
            
            agent.step(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
        
        # Получаем метрики из окружения
        metrics = train_env.get_trading_metrics()
        train_profit = metrics['total_profit_dollars']
        win_rate = metrics['win_rate'] * 100
        
        # OOS тестирование каждые 15 эпизодов
        if episode % 15 == 0:
            # Тестируем на 7 разных позициях
            oos_profits = []
            for start_pos in range(0, min(70, len(oos_env.stock.closes) - config['trading_period'] - config['window']), 10):
                state = oos_env.reset_fixed(start_pos)
                done = False
                
                while not done:
                    action = agent.act(state, training=False)  # БЕЗ exploration
                    next_state, trade_action, reward, done = oos_env.step(action)
                    state = next_state
                
                oos_metrics = oos_env.get_trading_metrics()
                oos_profits.append(oos_metrics['total_profit_dollars'])
            
            median_oos = np.median(oos_profits)
            mean_oos = np.mean(oos_profits)
            consistency = (np.array(oos_profits) > 0).mean() * 100
            
            # Проверяем стабильность
            if median_oos > 100:  # хорошая прибыльность
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
                f"    Epsilon: {agent.epsilon:.4f} | Best median: ${best_median:.0f}\\n"
                f"    All OOS: {[int(p) for p in oos_profits]}\\n"
                + "-" * 60
            )
            
            print(log_entry.replace("\\n", "\n"))
            
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(log_entry + "\n")
    
    # Финальное сохранение
    agent.save(f"models/{model_name}")
    
    print(f"\n🤖 МОЩНОЕ трансформер обучение v14 завершено!")
    print(f"💾 Модель сохранена: models/{model_name}")
    print(f"🏆 Лучшая медиана: models/{model_name}_best")
    print(f"📊 Лог сохранен: {log_file}")
    print(f"🎯 Стабильных результатов: {stability_count}/{config['episodes']//15}")
    
    # Финальное тестирование
    print(f"\n🔬 Финальное МОЩНОЕ тестирование...")
    final_profits = []
    for start_pos in range(0, min(140, len(oos_env.stock.closes) - config['trading_period'] - config['window']), 7):
        state = oos_env.reset_fixed(start_pos)
        done = False
        
        while not done:
            action = agent.act(state, training=False)
            next_state, trade_action, reward, done = oos_env.step(action)
            state = next_state
        
        final_metrics = oos_env.get_trading_metrics()
        final_profits.append(final_metrics['total_profit_dollars'])
    
    final_median = np.median(final_profits)
    final_consistency = (np.array(final_profits) > 0).mean() * 100
    final_std = np.std(final_profits)
    
    print(f"📈 ФИНАЛЬНЫЕ МОЩНЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   Медианный профит: ${final_median:.2f}")
    print(f"   Consistency: {final_consistency:.1f}%")
    print(f"   Стандартное отклонение: ${final_std:.2f}")
    print(f"   Диапазон: ${min(final_profits):.0f} - ${max(final_profits):.0f}")
    
    if final_consistency > 65 and final_median > 300:
        print("✅ МОЩНЫЙ трансформер показал отличные результаты!")
    elif final_consistency > 50 and final_median > 150:
        print("🟡 МОЩНЫЙ трансформер показал средние результаты")
    else:
        print("❌ МОЩНЫЙ трансформер требует доработки...")


if __name__ == "__main__":
    main() 