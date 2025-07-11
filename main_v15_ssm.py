#!/usr/bin/env python3
"""
🐍 SSM (MAMBA) АТАКА: v15 с State Space Models!
🎯 ЦЕЛЬ: Тестируем современную SSM архитектуру vs трансформеры/LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.rl.env import Env
from src.rl.agent import Agent
import os
from datetime import datetime
import math

class SSMAgent(Agent):
    """SSM агент с State Space Model архитектурой"""
    
    def __init__(self, obs_space, **kwargs):
        # SSM специфичные параметры
        self.d_state = kwargs.get('d_state', 16)        # Размер скрытого состояния
        self.d_conv = kwargs.get('d_conv', 4)           # Размер конволюции
        self.expand = kwargs.get('expand', 2)           # Фактор расширения
        self.dt_rank = kwargs.get('dt_rank', None)      # Ранг для dt
        self.ssm_layers = kwargs.get('ssm_layers', 4)   # Количество SSM слоев
        
        super().__init__(obs_space, **kwargs)
        
    def _create_network(self):
        """Создаем SSM сеть"""
        
        feature_dim = self.obs_space[1]  # количество фичей
        sequence_len = self.obs_space[2]  # длина последовательности
        
        return SSMQNetwork(
            d_model=feature_dim,
            sequence_len=sequence_len,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            dt_rank=self.dt_rank,
            ssm_layers=self.ssm_layers,
            action_size=3,
            dropout=self.dropout
        )
    
    def act(self, state, training=True):
        """ФИКСИРОВАННЫЙ epsilon decay БЕЗ adaptive boost!"""
        if training and np.random.random() <= self.epsilon:
            action = np.random.choice(3)
            if self.steps % 1000 == 0:
                print(f"🎲 SSM Random action: {action}, epsilon: {self.epsilon:.3f}")
            return action
        
        # Нейросеть
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        action = q_values.cpu().data.numpy().argmax()
        
        if self.steps % 1000 == 0:
            print(f"🐍 SSM Neural action: {action}, Q: {q_values.cpu().data.numpy()}")
        
        return action
    
    def step(self, state, action, reward, next_state, done):
        """SSM агент шаг с правильным epsilon decay"""
        self.memory.add(state, action, reward, next_state, done)
        self.steps += 1
        
        # ПРОСТОЙ epsilon decay БЕЗ adaptive boost
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.steps % 1000 == 0:
                print(f"📉 SSM Epsilon decay: {self.epsilon:.3f}")
        
        if len(self.memory) > self.batch_size and self.steps % self.update_freq == 0:
            experiences = self.memory.sample()
            self.learn(experiences)


class SSMBlock(nn.Module):
    """Один блок SSM (упрощенная версия Mamba)"""
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=None):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * d_model)
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        
        # Проекции
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # SSM параметры
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Инициализация A матрицы (стабильная)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D параметр (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Выходная проекция
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Нормализация
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        x: (B, L, D) где B=batch, L=length, D=d_model
        """
        B, L, D = x.shape
        
        # Residual connection
        residual = x
        
        # Нормализация
        x = self.norm(x)
        
        # Входная проекция
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # каждый (B, L, d_inner)
        
        # Конволюция (применяется по временной размерности)
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]  # обрезаем до исходной длины
        x = x.transpose(1, 2)  # (B, L, d_inner)
        
        # Активация
        x = F.silu(x)
        
        # SSM операция
        x = self.ssm(x)
        
        # Gating
        x = x * F.silu(z)
        
        # Выходная проекция
        out = self.out_proj(x)
        
        # Residual connection
        return out + residual
    
    def ssm(self, x):
        """Упрощенная SSM операция"""
        B, L, D = x.shape
        
        # Получаем SSM параметры из x
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, B_proj, C_proj = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # dt проекция
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)
        
        # A матрица
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Дискретизация (упрощенная)
        # Обычно используется ZOH, но для простоты используем Euler
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_proj.unsqueeze(-2)  # (B, L, d_inner, d_state)
        
        # Сканирование (упрощенная версия)
        # В реальной Mamba используется эффективная CUDA реализация
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for i in range(L):
            # Обновление состояния
            h = dA[:, i] * h + dB[:, i] * x[:, i:i+1, :].transpose(-1, -2)
            
            # Выход
            y = torch.einsum('bdn,bn->bd', h, C_proj[:, i])  # (B, d_inner)
            
            # Skip connection
            y = y + self.D * x[:, i]
            
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)  # (B, L, d_inner)


class SSMQNetwork(nn.Module):
    """SSM Q-Network для RL"""
    
    def __init__(self, d_model, sequence_len, d_state, d_conv, expand, dt_rank, ssm_layers, action_size, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.sequence_len = sequence_len
        self.ssm_layers = ssm_layers
        
        # Input embedding/normalization
        self.input_norm = nn.LayerNorm(d_model)
        
        # SSM блоки
        self.ssm_blocks = nn.ModuleList([
            SSMBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank
            ) for _ in range(ssm_layers)
        ])
        
        # Final layers
        self.final_norm = nn.LayerNorm(d_model)
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        
        # Output head
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, d_model // 4)
        self.fc_out = nn.Linear(d_model // 4, action_size)
        
        self.activation = nn.GELU()  # GELU activation как в современных моделях
        
    def forward(self, x):
        # x: (batch, timesteps, features)
        
        # Input normalization
        x = self.input_norm(x)
        
        # Применяем SSM блоки
        for ssm_block in self.ssm_blocks:
            x = ssm_block(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Global pooling (берем среднее по временной размерности)
        x = x.transpose(1, 2)  # (batch, features, timesteps)
        x = self.pooling(x).squeeze(-1)  # (batch, features)
        
        # Output head
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        q_values = self.fc_out(x)
        
        return q_values


def ssm_features(data, window=50):
    """
    🐍 SSM FEATURE SET - оптимизированные для последовательных зависимостей
    """
    
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame(data.copy())
    features = []
    
    print("🐍 Создаю SSM-оптимизированные фичи...")
    
    # 1. БАЗОВЫЕ ПОСЛЕДОВАТЕЛЬНЫЕ ФИЧИ
    close_prices = df['close'] if 'close' in df.columns else df.iloc[:, 3]
    
    # Цены (нормализованные)
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            normalized = (df[col] - df[col].rolling(window).mean()) / df[col].rolling(window).std()
            features.append(normalized.fillna(0).values)
    
    # 2. МНОГОМАСШТАБНЫЕ ВРЕМЕННЫЕ ЛАГИ (SSM хорошо обрабатывает)
    for lag in [1, 2, 3, 5, 8, 13, 21, 34]:  # Фибоначчи последовательность
        lagged = close_prices.shift(lag)
        lag_returns = ((close_prices - lagged) / lagged).fillna(0)
        features.append(lag_returns.values)
    
    # 3. ЭКСПОНЕНЦИАЛЬНЫЕ СКОЛЬЗЯЩИЕ СРЕДНИЕ (больше весов недавним данным)
    for span in [3, 5, 8, 13, 21, 34, 55, 89]:
        ema = close_prices.ewm(span=span).mean()
        ema_ratio = (close_prices / ema - 1).fillna(0)
        features.append(ema_ratio.values)
        
        # EMA momentum
        ema_momentum = ema.pct_change(5).fillna(0)
        features.append(ema_momentum.values)
    
    # 4. ВОЛАТИЛЬНОСТЬ РАЗНЫХ ВРЕМЕННЫХ ГОРИЗОНТОВ
    returns = close_prices.pct_change().fillna(0)
    
    for period in [3, 5, 8, 13, 21, 34]:
        vol = returns.rolling(window=period).std().fillna(0)
        features.append(vol.values)
        
        # Normalized volatility
        vol_norm = (vol - vol.rolling(window).mean()) / vol.rolling(window).std()
        features.append(vol_norm.fillna(0).values)
    
    # 5. ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ с разными периодами
    
    # RSI разных периодов
    for period in [5, 8, 13, 21]:
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).fillna(50) / 100
        features.append(rsi.values)
    
    # MACD разных конфигураций
    for fast, slow in [(5, 13), (8, 21), (13, 34), (21, 55)]:
        ema_fast = close_prices.ewm(span=fast).mean()
        ema_slow = close_prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        
        macd_norm = (macd - macd.rolling(window).mean()) / macd.rolling(window).std()
        features.append(macd_norm.fillna(0).values)
    
    # 6. ОБЪЕМЫ (если есть)
    if 'volume' in df.columns:
        vol = df['volume']
        
        # Volume normalization
        vol_norm = (vol - vol.rolling(window).mean()) / vol.rolling(window).std()
        features.append(vol_norm.fillna(0).values)
        
        # Volume momentum разных периодов
        for period in [3, 5, 8, 13]:
            vol_momentum = vol.pct_change(period).fillna(0)
            features.append(vol_momentum.values)
        
        # Price-Volume relationship
        pv_corr = close_prices.rolling(21).corr(vol).fillna(0)
        features.append(pv_corr.values)
    
    # 7. ЦИКЛИЧЕСКИЕ ВРЕМЕННЫЕ ФИЧИ (SSM может выучить циклы)
    
    # Позиция в разных циклах
    session_pos = np.arange(len(df)) / len(df)
    features.append(session_pos)
    
    # Синусоидальные циклы разных периодов
    for period in [5, 13, 21, 55, 252]:  # от недели до года
        sin_cycle = np.sin(2 * np.pi * np.arange(len(df)) / period)
        cos_cycle = np.cos(2 * np.pi * np.arange(len(df)) / period)
        features.append(sin_cycle)
        features.append(cos_cycle)
    
    # 8. МНОГОМАСШТАБНЫЕ MOMENTUM ФИЧИ
    for period in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
        momentum = close_prices.pct_change(period).fillna(0)
        features.append(momentum.values)
        
        # Normalized momentum
        momentum_norm = (momentum - momentum.rolling(window).mean()) / momentum.rolling(window).std()
        features.append(momentum_norm.fillna(0).values)
    
    # 9. АВТОКОРРЕЛЯЦИОННЫЕ ФИЧИ (SSM может выучить зависимости)
    for lag in [1, 3, 5, 8, 13]:
        autocorr = returns.rolling(21).apply(lambda x: x.autocorr(lag=lag) if len(x) > lag else 0).fillna(0)
        features.append(autocorr.values)
    
    # Объединяем все фичи
    feature_matrix = np.column_stack(features)
    feature_count = feature_matrix.shape[1]
    
    print(f"🐍 Создано {feature_count} SSM-оптимизированных фичей:")
    print(f"   Базовые цены: 4 фичи")
    print(f"   Временные лаги: 8 фичей")
    print(f"   EMA системы: 16 фичей")
    print(f"   Волатильность: 12 фичей")
    print(f"   Технические: 8 фичей")
    print(f"   Объемы: ~8 фичей")
    print(f"   Циклические: 11 фичей")
    print(f"   Momentum: 18 фичей")
    print(f"   Автокорреляции: 5 фичей")
    
    # Проверяем на NaN и Inf
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return feature_matrix


def main():
    """SSM эксперимент v15"""
    
    print("🐍 SSM (MAMBA) АТАКА v15: State Space Models!")
    print("🎯 ЦЕЛЬ: Тестируем передовую SSM архитектуру")
    print("🚀 Mamba-like селективные state space models")
    print("=" * 60)
    
    # Параметры SSM эксперимента
    config = {
        'episodes': 180,
        'trading_period': 100,  # средний период
        'window': 45,           # средний размер окна
        'target_profit': 800,   # амбициозная цель
        'commission': 0.0002,
        
        # SSM архитектура
        'd_state': 16,          # Размер скрытого состояния
        'd_conv': 4,            # Конволюционный размер
        'expand': 2,            # Фактор расширения
        'dt_rank': None,        # Автоматический ранг
        'ssm_layers': 4,        # 4 SSM блока
        'dropout': 0.1,
        
        # Обучение
        'lr': 0.0008,           # немного выше чем у трансформера
        'epsilon': 0.85,        # высокий старт
        'epsilon_min': 0.008,   # низкий финиш
        'epsilon_decay': 0.996, # медленный decay
        'gamma': 0.95,
        'memory_size': 12000,
        'batch_size': 384,
        'update_freq': 4
    }
    
    print(f"📊 SSM КОНФИГУРАЦИЯ:")
    for key, value in config.items():
        if key == 'dt_rank' and value is None:
            print(f"   {key}: auto")
        else:
            print(f"   {key}: {value}")
    print("=" * 60)
    
    # Создаем окружение с SSM фичами
    train_env = Env(
        csv_paths=["GOOG_2010_2024-06.csv"],
        fee=config['commission'],
        trading_period=config['trading_period'],
        window=config['window'],
        feature_extractor=ssm_features
    )
    
    oos_env = Env(
        csv_paths=["GOOG_2024-07_2025-04.csv"],
        fee=config['commission'],
        trading_period=config['trading_period'],
        window=config['window'],
        feature_extractor=ssm_features
    )
    
    print(f"📊 SSM окружение:")
    print(f"   Train observation space: {train_env.stock.obs_space}")
    print(f"   OOS observation space: {oos_env.stock.obs_space}")
    print(f"   Target profit: ${config['target_profit']}")
    
    # Создаем SSM агента
    agent = SSMAgent(
        obs_space=train_env.stock.obs_space,
        **{k: v for k, v in config.items() if k not in ['episodes', 'trading_period', 'window', 'target_profit', 'commission']}
    )
    
    print(f"🐍 SSM агент создан:")
    print(f"   Архитектура: {config['ssm_layers']} SSM блоков")
    print(f"   State size: {config['d_state']}, Conv: {config['d_conv']}, Expand: {config['expand']}")
    print(f"   Epsilon: {config['epsilon']} -> {config['epsilon_min']} (decay: {config['epsilon_decay']})")
    print(f"   БЕЗ adaptive epsilon boost!")
    
    # Логирование
    log_file = f"models/google-trading-v15-ssm.log"
    model_name = "google-trading-v15-ssm"
    
    best_median = -float('inf')
    stability_count = 0
    
    print(f"\n🐍 Начинаю SSM обучение...")
    
    with open(log_file, "w", encoding='utf-8') as f:
        f.write(f"🐍 SSM Training Log v15 - {datetime.now()}\n")
        f.write(f"Config: {config}\n")
        f.write("🚀 State Space Models (Mamba-like)\n")
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
        
        # OOS тестирование каждые 12 эпизодов
        if episode % 12 == 0:
            # Тестируем на 6 разных позициях
            oos_profits = []
            for start_pos in range(0, min(60, len(oos_env.stock.closes) - config['trading_period'] - config['window']), 10):
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
            if median_oos > 150:  # хорошая прибыльность для SSM
                stability_count += 1
            
            # Сохраняем лучшую модель
            if median_oos > best_median:
                best_median = median_oos
                agent.save(f"models/{model_name}_best")
                print(f"💾 Новая лучшая SSM медиана: ${median_oos:.0f} (эпизод {episode})")
            
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
    
    print(f"\n🐍 SSM обучение v15 завершено!")
    print(f"💾 Модель сохранена: models/{model_name}")
    print(f"🏆 Лучшая медиана: models/{model_name}_best")
    print(f"📊 Лог сохранен: {log_file}")
    print(f"🎯 Стабильных результатов: {stability_count}/{config['episodes']//12}")
    
    # Финальное тестирование
    print(f"\n🔬 Финальное SSM тестирование...")
    final_profits = []
    for start_pos in range(0, min(120, len(oos_env.stock.closes) - config['trading_period'] - config['window']), 6):
        oos_env.stock.reset_fixed(start_pos)
        
        while not oos_env.stock.done:
            state = oos_env.stock.get_state()
            action = agent.act(state, training=False)
            reward, profit, trade_made = oos_env.stock.step(action)
        
        final_profits.append(oos_env.stock.total_profit)
    
    final_median = np.median(final_profits)
    final_consistency = (np.array(final_profits) > 0).mean() * 100
    final_std = np.std(final_profits)
    
    print(f"📈 ФИНАЛЬНЫЕ SSM РЕЗУЛЬТАТЫ:")
    print(f"   Медианный профит: ${final_median:.2f}")
    print(f"   Consistency: {final_consistency:.1f}%")
    print(f"   Стандартное отклонение: ${final_std:.2f}")
    print(f"   Диапазон: ${min(final_profits):.0f} - ${max(final_profits):.0f}")
    
    if final_consistency > 70 and final_median > 400:
        print("✅ SSM показал отличные результаты!")
    elif final_consistency > 55 and final_median > 200:
        print("🟡 SSM показал средние результаты")
    else:
        print("❌ SSM требует доработки...")


if __name__ == "__main__":
    main() 