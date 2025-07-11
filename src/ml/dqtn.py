import torch
import torch.nn as nn
import numpy as np
import math


class DQTN(nn.Module):
    def __init__(self, embeddings, heads, layers, fwex, dropout, neurons, lr=0.001, view_size=30):
        super(DQTN, self).__init__()
        
        self.view_size = view_size
        
        # Получаем размер входных данных динамически
        self.input_size = None  # Будет установлен при первом forward
        self.embeddings = embeddings
        
        # Входной слой эмбеддингов (будет создан динамически)
        self.input_embedder = None
        
        # Позиционное кодирование
        self.pos_encoder = PositionalEncoding(embeddings, dropout, max_len=view_size * 20)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embeddings,
            nhead=heads,
            dim_feedforward=fwex,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=layers)
        
        # Финальные слои
        self.layer_norm = nn.LayerNorm(embeddings)
        self.fc_layers = nn.Sequential(
            nn.Linear(embeddings, neurons),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(neurons, neurons // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(neurons // 2, 3)  # 3 действия: hold, buy, sell
        )
        
        # Инициализация весов
        self._init_weights()
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _init_weights(self):
        """Инициализация весов Xavier"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _create_input_embedder(self, num_features):
        """Создаем входной эмбеддер для конкретного количества фичей"""
        self.input_embedder = nn.Linear(num_features, self.embeddings)
        nn.init.xavier_uniform_(self.input_embedder.weight)
        if self.input_embedder.bias is not None:
            nn.init.zeros_(self.input_embedder.bias)
        self.input_embedder.to(self.device)

    def forward(self, x):
        """
        Forward pass для многомерных технических индикаторов
        x shape: (batch_size, num_features, sequence_length)
        """
        batch_size, num_features, seq_len = x.shape
        
        # Создаем входной эмбеддер при первом вызове
        if self.input_embedder is None:
            self._create_input_embedder(num_features)
            print(f"🧠 Создан эмбеддер: {num_features} фичей -> {self.embeddings} эмбеддингов")
        
        # Транспонируем для (batch_size, seq_len, num_features)
        x = x.transpose(1, 2)
        
        # Создаем эмбеддинги
        embeddings = self.input_embedder(x)  # (batch_size, seq_len, embeddings)
        
        # Добавляем позиционное кодирование
        embeddings = self.pos_encoder(embeddings)
        
        # Пропускаем через трансформер
        transformer_out = self.transformer(embeddings)
        
        # Берем последний временной шаг и нормализуем
        last_state = self.layer_norm(transformer_out[:, -1, :])
        
        # Финальные слои
        output = self.fc_layers(last_state)
        
        return output

    def predict(self, state):
        """Предсказание для одного состояния"""
        self.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                state = state.unsqueeze(0).to(self.device)
            
            q_values = self(state)
            return q_values.cpu().numpy()[0]

    def train_on_batch(self, states, actions, rewards, next_states, dones, gamma=0.9):
        """Обучение на батче данных"""
        self.train()
        
        # Конвертируем в тензоры
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Текущие Q-значения
        current_q_values = self(states).gather(1, actions.unsqueeze(1))
        
        # Следующие Q-значения (Double DQN)
        with torch.no_grad():
            next_q_values = self(next_states).max(1)[0]
            target_q_values = rewards + (gamma * next_q_values * ~dones)
        
        # Huber loss
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        # Обратное распространение
        self.optimizer.zero_grad()
        loss.backward()
        
        # Градиентное обрезание
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()


class PositionalEncoding(nn.Module):
    """Позиционное кодирование для трансформера"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return self.dropout(x)
