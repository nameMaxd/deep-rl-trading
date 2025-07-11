from src.ml.dqtn import DQTN
import numpy as np
import torch
import torch.nn.functional as F
import random
from collections import deque


class Agent:
    """Улучшенный DQN Agent с адаптивным exploration и regularization"""
    
    def __init__(self, obs_space, embeddings=32, heads=1, layers=1, fwex=32,
                 dropout=0.15, neurons=64, lr=0.001, epsilon=1.0, epsilon_min=0.1, 
                 epsilon_decay=0.995, gamma=0.95, memory_size=2000, batch_size=32,
                 update_freq=10):
        
        # Параметры окружения
        self.obs_space = obs_space
        self.action_size = 3  # hold, buy, sell
        
        # Улучшенные параметры обучения
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.steps = 0
        
        # Adaptive exploration
        self.performance_history = deque(maxlen=50)
        self.epsilon_boost_counter = 0
        self.last_exploration_boost = 0
        
        # Curiosity-driven exploration
        self.action_frequency = {0: 0, 1: 0, 2: 0}
        self.state_visit_count = {}
        
        # Память для опыта
        self.memory = deque(maxlen=memory_size)
        
        # Получаем размеры observation space
        if len(obs_space.shape) == 3:
            num_features, seq_len = obs_space.shape[1], obs_space.shape[2]
        else:
            num_features, seq_len = obs_space.shape[0], obs_space.shape[1]
        
        print(f"🤖 Инициализация улучшенного агента v5:")
        print(f"   Observation space: {obs_space.shape}")
        print(f"   Features: {num_features}, Sequence: {seq_len}")
        
        # Создаем сеть с улучшенными параметрами
        self.q_network = DQTN(
            embeddings=embeddings,
            heads=heads,
            layers=layers,
            fwex=fwex,
            dropout=dropout,  # Немного увеличили dropout
            neurons=neurons,
            lr=lr,
            view_size=seq_len
        )
        
        self.device = self.q_network.device
        print(f"   Устройство: {self.device}")
        print(f"   Адаптивный epsilon: {epsilon} -> {epsilon_min}")
        print(f"   Curiosity exploration: Включено")

    def remember(self, state, action, reward, next_state, done):
        """Сохранение опыта в память с проверкой качества"""
        # Простая фильтрация: избегаем дублирования слишком похожих состояний
        if len(self.memory) > 0:
            last_state, _, _, _, _ = self.memory[-1]
            state_similarity = np.corrcoef(state.flatten(), last_state.flatten())[0, 1]
            if state_similarity > 0.99:  # Слишком похожие состояния
                return
                
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """Улучшенный выбор действия с адаптивным exploration"""
        if training:
            # Адаптивный epsilon на основе производительности
            current_epsilon = self._get_adaptive_epsilon()
            
            if np.random.random() <= current_epsilon:
                # Curiosity-driven exploration
                action = self._curiosity_action(state)
            else:
                # Exploitation через нейросеть
                action = self._greedy_action(state)
        else:
            # В тестовом режиме только exploitation
            action = self._greedy_action(state)
        
        # Обновляем статистики
        self.action_frequency[action] += 1
        
        return action

    def _get_adaptive_epsilon(self):
        """Адаптивный epsilon на основе производительности"""
        base_epsilon = self.epsilon
        
        # Если производительность плохая, увеличиваем exploration
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(list(self.performance_history)[-10:])
            if recent_performance < -0.01:  # Плохая производительность
                boost_factor = min(2.0, 1.0 + abs(recent_performance) * 10)
                return min(0.8, base_epsilon * boost_factor)
        
        return base_epsilon

    def _curiosity_action(self, state):
        """Curiosity-driven exploration - выбираем менее частые действия"""
        # Находим наименее используемое действие
        min_freq = min(self.action_frequency.values())
        least_used_actions = [action for action, freq in self.action_frequency.items() 
                             if freq == min_freq]
        
        # Добавляем небольшую случайность
        if np.random.random() < 0.3:
            return random.choice(least_used_actions)
        else:
            return random.randrange(self.action_size)

    def _greedy_action(self, state):
        """Жадное действие через нейросеть"""
        q_values = self.q_network.predict(state)
        return np.argmax(q_values)

    def replay(self):
        """Улучшенное обучение с regularization"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Выбираем случайный батч
        batch = random.sample(self.memory, self.batch_size)
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Обучаем сеть
        loss = self.q_network.train_on_batch(
            states, actions, rewards, next_states, dones, self.gamma
        )
        
        # Адаптивное обновление epsilon
        self._update_epsilon_adaptive(np.mean(rewards))
        
        return loss

    def _update_epsilon_adaptive(self, batch_reward):
        """Адаптивное обновление epsilon"""
        # Сохраняем производительность
        self.performance_history.append(batch_reward)
        
        # Стандартное снижение epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # ОТКЛЮЧАЕМ adaptive epsilon boost - он портит результаты!
        # if len(self.performance_history) >= 20:
        #     recent_avg = np.mean(list(self.performance_history)[-10:])
        #     older_avg = np.mean(list(self.performance_history)[-20:-10])
        #     
        #     if recent_avg <= older_avg and self.steps - self.last_exploration_boost > 100:
        #         self.epsilon = min(0.5, self.epsilon * 1.2)  # Boost exploration
        #         self.last_exploration_boost = self.steps
        #         print(f"🔍 Exploration boost! Epsilon: {self.epsilon:.3f}")

    def update(self):
        """Обновление агента после каждого шага"""
        self.steps += 1
        
        # Более частое обучение для лучшей стабильности
        if self.steps % self.update_freq == 0:
            return self.replay()
        
        return 0.0

    def get_exploration_stats(self):
        """Статистика exploration"""
        total_actions = sum(self.action_frequency.values())
        if total_actions == 0:
            return {"hold": 0, "buy": 0, "sell": 0}
        
        return {
            "hold": self.action_frequency[0] / total_actions,
            "buy": self.action_frequency[1] / total_actions, 
            "sell": self.action_frequency[2] / total_actions
        }

    def save(self, filepath):
        """Расширенное сохранение модели"""
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.q_network.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'performance_history': list(self.performance_history),
            'action_frequency': self.action_frequency
        }, filepath)
        print(f"💾 Модель v5 сохранена: {filepath}")

    def load(self, filepath):
        """Расширенная загрузка модели"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            self.q_network.load_state_dict(checkpoint['model_state_dict'])
            self.q_network.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.steps = checkpoint.get('steps', 0)
            
            if 'performance_history' in checkpoint:
                self.performance_history = deque(checkpoint['performance_history'], maxlen=50)
            if 'action_frequency' in checkpoint:
                self.action_frequency = checkpoint['action_frequency']
                
            print(f"📥 Модель v5 загружена: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False

    def get_action_name(self, action):
        """Получение названия действия"""
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        return action_names.get(action, "UNKNOWN")

    def get_stats(self):
        """Расширенная статистика агента"""
        exploration_stats = self.get_exploration_stats()
        
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'steps': self.steps,
            'device': str(self.device),
            'performance_trend': np.mean(list(self.performance_history)[-10:]) if len(self.performance_history) >= 10 else 0,
            'action_distribution': exploration_stats,
            'exploration_balance': min(exploration_stats.values()) / max(exploration_stats.values()) if max(exploration_stats.values()) > 0 else 0
        }
