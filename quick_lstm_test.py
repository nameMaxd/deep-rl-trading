#!/usr/bin/env python3
"""
🧠 БЫСТРЫЙ LSTM ТЕСТ - убеждаемся что всё работает!
"""

from src.rl.env import Env
from src.rl.agent import Agent
import numpy as np

def main():
    print("🧠 БЫСТРЫЙ LSTM ТЕСТ")
    
    # Создаем простое окружение
    train_env = Env(
        csv_paths=["GOOG_2010-2024-06.csv"],
        fee=0.0002,
        trading_period=60,  # короче
        window=30           # меньше окно
    )
    
    print(f"📊 Окружение создано:")
    print(f"   Observation space: {train_env.stock.obs_space.shape}")
    
    # Создаем обычного агента
    agent = Agent(
        obs_space=train_env.stock.obs_space,
        lr=0.001,
        epsilon=0.8,
        epsilon_min=0.05,
        epsilon_decay=0.99,
        memory_size=5000,
        batch_size=128
    )
    
    print(f"🤖 Агент создан: {agent.epsilon}")
    
    # Быстрая тренировка на 20 эпизодов
    print("🏃 Быстрая тренировка...")
    
    for episode in range(20):
        train_env.reset()
        total_reward = 0
        
        while not train_env.stock.done:
            state = train_env.stock.get_state()
            action = agent.act(state)
            
            next_state, trade_action, reward, done = train_env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            total_reward += reward
        
        metrics = train_env.get_trading_metrics()
        
        if episode % 5 == 0:
            print(f"Ep {episode}: Profit ${metrics['total_profit_dollars']:.0f}, "
                  f"Trades {metrics['num_trades']}, "
                  f"Win% {metrics['win_rate']*100:.1f}, "
                  f"ε: {agent.epsilon:.3f}")
    
    # Сохраняем модель
    agent.save("models/quick-lstm-test")
    print("💾 Модель сохранена!")
    
    print("✅ LSTM тест завершен успешно!")

if __name__ == "__main__":
    main() 