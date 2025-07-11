#!/usr/bin/env python3
"""
🤖 TRANSFORMER v14 С ЖИВЫМ TQDM ПРОГРЕССОМ!
🎯 Мощная трансформер архитектура!
"""

from src.rl.env import Env
from src.rl.agent import Agent
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import sys

# ПРИНУДИТЕЛЬНЫЙ ВЫВОД
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def log_and_print(message, log_file="transformer_v14.log"):
    """Принудительный вывод в консоль И файл"""
    print(message, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")
        f.flush()

class TransformerAgent(Agent):
    def __init__(self, obs_space, **kwargs):
        super().__init__(obs_space, **kwargs)
        self.obs_space = obs_space
        self.embeddings = kwargs.get('embeddings', 64)
        self.heads = kwargs.get('heads', 4)
        self.layers = kwargs.get('layers', 3)
        self.fwex = kwargs.get('fwex', 256)
        self.dropout = kwargs.get('dropout', 0.1)
        self.neurons = kwargs.get('neurons', 256)
        self.lr = kwargs.get('lr', 0.0005)
        self.epsilon = kwargs.get('epsilon', 0.8)
        self.epsilon_min = kwargs.get('epsilon_min', 0.005)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.9965)
        self.gamma = kwargs.get('gamma', 0.95)
        self.memory_size = kwargs.get('memory_size', 15000)
        self.batch_size = kwargs.get('batch_size', 512)
        self.update_freq = kwargs.get('update_freq', 3)

        self.model = TransformerModel(obs_space, self.embeddings, self.heads, self.layers, self.fwex, self.dropout, self.neurons)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.memory = []
        self.memory_size = self.memory_size
        self.batch_size = self.batch_size
        self.update_freq = self.update_freq

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def act(self, state, training=True):
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(3) # Assuming 3 actions for simplicity
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # Add batch dimension
                action_probs = self.model(state_tensor)
                action = torch.argmax(action_probs).item()
                return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        # Compute Q values
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
        q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Get current Q values from model
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # Compute loss
        loss = self.criterion(current_q_values, q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        log_and_print(f"Model saved to {filename}")

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        log_and_print(f"Model loaded from {filename}")

class TransformerModel(nn.Module):
    def __init__(self, obs_space, embeddings, heads, layers, fwex, dropout, neurons):
        super().__init__()
        self.embeddings = embeddings
        self.heads = heads
        self.layers = layers
        self.fwex = fwex
        self.dropout = dropout
        self.neurons = neurons

        self.embedding_layer = nn.Linear(obs_space, embeddings)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embeddings, heads, fwex, dropout)
            for _ in range(layers)
        ])
        self.fc = nn.Sequential(
            nn.Linear(embeddings, neurons),
            nn.ReLU(),
            nn.Linear(neurons, 3) # 3 actions
        )

    def forward(self, x):
        x = self.embedding_layer(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.fc(x)
        return x

def main():
    log_and_print("🤖 TRANSFORMER v14 С ЖИВЫМ TQDM ПРОГРЕССОМ!")
    log_and_print("🎯 Мощная трансформер архитектура!")
    log_and_print("=" * 60)
    
    # TRANSFORMER конфигурация
    config = {
        'episodes': 250,
        'trading_period': 120,
        'window': 50,
        'commission': 0.0002,
        
        # TRANSFORMER параметры
        'embeddings': 64,
        'heads': 4,
        'layers': 3,
        'fwex': 256,
        'dropout': 0.1,
        'neurons': 256,
        'lr': 0.0005,
        'epsilon': 0.05,        # НИЗКИЙ стартовый epsilon!
        'epsilon_min': 0.001,   # Очень низкий минимум
        'epsilon_decay': 0.999, # Медленный спад
        'gamma': 0.95,
        'memory_size': 15000,
        'batch_size': 512,
        'update_freq': 3
    }
    
    log_and_print(f"📊 TRANSFORMER конфигурация: {config['episodes']} эпизодов")
    
    # Создаем окружения
    log_and_print("📈 Загружаем данные...")
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
    
    log_and_print(f"📊 ДАННЫЕ ЗАГРУЖЕНЫ:")
    log_and_print(f"   🎯 Train: 2010-2024 данные, shape: {train_env.stock.obs_space.shape}")
    log_and_print(f"   🧪 OOS: 2024-07_2025-04 данные, shape: {oos_env.stock.obs_space.shape}")
    log_and_print(f"   📏 Trading period: {config['trading_period']} дней")
    log_and_print(f"   🪟 Window: {config['window']} дней")
    log_and_print("=" * 60)
    
    # Создаем мощного трансформер агента
    agent = Agent(
        obs_space=train_env.stock.obs_space,
        **{k: v for k, v in config.items() 
           if k not in ['episodes', 'trading_period', 'window', 'commission']}
    )
    
    # Заполняем память
    log_and_print("🧠 Заполняем TRANSFORMER память...")
    state = train_env.reset()
    for _ in range(2000):
        action = np.random.randint(3)
        next_state, _, reward, done = train_env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = train_env.reset()
    
    log_and_print("🏃 TRANSFORMER обучение с ЖИВЫМ прогрессом...")
    log_and_print("📈 Каждую эпоху: OOS тест")
    log_and_print("📊 Каждые 10 эпох: подробный отчет")
    log_and_print("=" * 60)
    
    # Очистка старого лог файла
    with open("transformer_v14.log", "w") as f:
        f.write("TRANSFORMER v14 TRAINING LOG\n")
        f.write("=" * 50 + "\n")
    
    # Основной цикл обучения с TQDM
    best_median = -float('inf')
    
    for episode in range(config['episodes']):
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
        max_start_pos = len(oos_env.stock.closes) - config['trading_period'] - config['window'] - 1
        if max_start_pos > 0:
            for start_pos in range(0, min(15, max_start_pos), 3):
                state = oos_env.reset_fixed(start_pos)
                done = False
                
                while not done:
                    action = agent.act(state, training=False)
                    next_state, _, reward, done = oos_env.step(action)
                    state = next_state
                
                oos_profits.append(oos_env.current_equity - oos_env.initial_capital)
        
        # Защита от пустого списка
        if len(oos_profits) == 0:
            oos_profits = [0.0]
        
        oos_median = np.median(oos_profits)
        
        if oos_median > best_median:
            best_median = oos_median
            agent.save(f"models/tqdm-transformer-v14_best")
        
        # ОТЧЕТ КАЖДЫЕ 10 ЭПОХ + ПРИНУДИТЕЛЬНЫЙ ВЫВОД
        if (episode + 1) % 10 == 0:
            report = f"""
📊 ЭПОХА {episode + 1}/{config['episodes']} ОТЧЕТ:
   💰 Train profit: ${train_profit:.2f}
   🧪 OOS median: ${oos_median:.2f}
   🏆 Best OOS: ${best_median:.2f}
   📈 Epsilon: {agent.epsilon:.4f}
   🔄 Trades: {train_env.trade_count}
   📊 OOS range: ${min(oos_profits):.0f} to ${max(oos_profits):.0f}
   📈 OOS tests: {len(oos_profits)}
"""
            log_and_print(report)
        
        # Краткий прогресс каждые 5 эпох
        elif (episode + 1) % 5 == 0:
            short_report = f"ЭПОХА {episode + 1}: Train=${train_profit:.0f}, OOS=${oos_median:.0f}, Best=${best_median:.0f}, Tests={len(oos_profits)}"
            log_and_print(short_report)
        
        time.sleep(0.01)  # Небольшая пауза
    
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