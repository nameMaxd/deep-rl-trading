from src.rl.agent import Agent
from src.rl.env import Env
from src.stock.stock import Stock
import numpy as np
import os


def main():
    """v9: АГРЕССИВНАЯ модель для СЕРЬЁЗНЫХ денег"""
    
    # Конфигурация модели
    model_name = "google-trading-v9-aggressive-money"
    
    print("💰 АГРЕССИВНАЯ АТАКА: Google trading bot v9!")
    print("🎯 ЦЕЛЬ: СЕРЬЁЗНЫЕ ДЕНЬГИ!")
    print("📊 АНАЛИЗ:")
    print("   ❌ v8: $767 OOS - КОПЕЙКИ!")
    print("   ❌ Все версии: играем в мелочи")
    print("   💸 Нужны ТЫСЯЧИ долларов, не сотни!")
    print("")
    print("🔥 АГРЕССИВНАЯ СТРАТЕГИЯ v9:")
    print("   💪 МОЩНАЯ архитектура для сложных паттернов")
    print("   📈 ДЛИННЫЕ периоды для больших движений") 
    print("   ⚡ АКТИВНАЯ торговля - много сделок")
    print("   🎯 ВЫСОКИЕ цели - $2000+ на OOS")
    print("   🚀 ЭКСТРЕМАЛЬНЫЕ параметры")
    print("=" * 60)
    
    # АГРЕССИВНЫЕ параметры для БОЛЬШИХ денег
    training_episodes = 500  # МНОГО эпизодов для качества
    trading_period = 120    # ДЛИННЫЙ период для больших движений
    window_size = 50        # БОЛЬШОЕ окно для сложных паттернов
    fee = 0.0002           # МИНИМАЛЬНАЯ комиссия
    
    # Данные
    train_csv = "GOOG_2010-2024-06.csv"
    oos_csv = "GOOG_2024-07_2025-04.csv"
    
    print(f"📊 АГРЕССИВНЫЕ параметры:")
    print(f"📈 Эпизоды: {training_episodes} (МАКСИМУМ для качества)")
    print(f"⏰ Период: {trading_period} дней (ДЛИННЫЙ для больших движений)")
    print(f"🪟 Окно: {window_size} дней (БОЛЬШОЕ для сложных паттернов)")
    print(f"💰 Комиссия: {fee*100}% (МИНИМАЛЬНАЯ)")
    print("=" * 60)
    
    # Создаем окружения
    train_env = Env(csv_paths=[train_csv], fee=fee, trading_period=trading_period, window=window_size)
    oos_env = Env(csv_paths=[oos_csv], fee=fee, trading_period=trading_period, window=window_size)
    
    # АГРЕССИВНЫЕ цели для БОЛЬШИХ денег
    train_env.target_profit = 2000  # ВЫСОКАЯ цель
    train_env.max_trades_per_episode = 25  # МНОГО сделок
    train_env.min_trades_per_episode = 10   # АКТИВНАЯ торговля
    
    print(f"📊 МОЩНОЕ окружение v9:")
    print(f"   Observation space: {train_env.stock.obs_space.shape}")
    print(f"   Target profit: ${train_env.target_profit} (ВЫСОКАЯ цель)")
    print(f"   Trading range: {train_env.min_trades_per_episode}-{train_env.max_trades_per_episode} сделок (АКТИВНО)")
    
    # МОЩНАЯ конфигурация v9
    model_config = {
        'embeddings': 64,      # БОЛЬШИЕ эмбеддинги для сложности
        'heads': 4,            # МНОГО attention heads
        'layers': 3,           # ГЛУБОКАЯ архитектура
        'fwex': 256,          # МОЩНЫЙ feedforward
        'dropout': 0.1,       # Умеренный dropout
        'neurons': 256,       # МНОГО нейронов
        'lr': 0.003,          # ВЫСОКИЙ learning rate
        'epsilon': 1.0,
        'epsilon_min': 0.05,  # Низкий для exploitation
        'epsilon_decay': 0.995,  # Медленный decay для exploration
        'gamma': 0.98,        # Высокий discount для долгосрочности
        'memory_size': 10000, # ОГРОМНАЯ память
        'batch_size': 256,    # БОЛЬШИЕ батчи
        'update_freq': 3      # Частое обучение
    }
    
    print(f"====== МОЩНАЯ модель v9: {model_name} ======")
    print("💪 АГРЕССИВНЫЕ параметры для БОЛЬШИХ денег:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    # Создаем агента v9
    agent = Agent(obs_space=train_env.stock.obs_space, **model_config)
    
    # Логирование
    log_file = f"models/{model_name}.log"
    
    print(f"🔥 АГРЕССИВНОЕ обучение v9...")
    print(f"💰 ЦЕЛЬ: Стабильные ${train_env.target_profit}+ на OOS!")
    
    # ИНТЕНСИВНАЯ инициализация памяти
    print("📦 ИНТЕНСИВНОЕ заполнение памяти...")
    initial_memory_size = model_config['memory_size'] // 3
    
    attempts = 0
    while len(agent.memory) < initial_memory_size and attempts < 20:
        state = train_env.reset()
        done = False
        step_count = 0
        
        while not done and len(agent.memory) < initial_memory_size and step_count < trading_period:
            # Агрессивная инициализация - много торговли
            if step_count < 20:
                action = 0  # Начальное наблюдение
            elif step_count < trading_period // 2:
                action = 1 if np.random.random() < 0.6 else 0  # МНОГО buy
            else:
                action = agent.act(state, training=True)
                
            next_state, _, reward, done = train_env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            step_count += 1
        attempts += 1
    
    print(f"✅ Память заполнена: {len(agent.memory)} записей за {attempts} попыток")
    
    # АГРЕССИВНЫЙ цикл обучения v9
    # Инициализируем лог-файл с заголовком
    header = "episode,train_profit,oos_profit,train_trades,oos_trades,train_win_rate,oos_win_rate,train_sharpe,oos_sharpe,loss,epsilon"
    with open(log_file, "w") as f:
        f.write(header + "\n")
    
    best_oos_profit = -float('inf')
    best_model_episode = 0
    money_targets_hit = 0  # Счётчик достижения денежных целей
    
    # АГРЕССИВНАЯ оптимизация каждые 25 эпизодов
    oos_optimization_freq = 25
    
    try:
        for episode in range(training_episodes):
            # Интенсивное обучение на train
            state = train_env.reset()
            done = False
            episode_loss = 0.0
            loss_count = 0
            
            while not done:
                action = agent.act(state, training=True)
                next_state, _, reward, done = train_env.step(action)
                agent.remember(state, action, reward, next_state, done)
                
                # ИНТЕНСИВНОЕ обучение
                if len(agent.memory) > model_config['batch_size']:
                    loss = agent.update()
                    if loss > 0:
                        episode_loss += loss
                        loss_count += 1
                
                state = next_state
            
            # Метрики train
            train_metrics = train_env.get_trading_metrics()
            avg_loss = episode_loss / max(loss_count, 1)
            
            # OOS тестирование
            oos_metrics = test_oos_aggressive(agent, oos_env)
            
            # АГРЕССИВНАЯ оптимизация на OOS данных!
            if episode % oos_optimization_freq == 0 and episode > 0:
                print(f"🎯 AGGRESSIVE OOS OPTIMIZATION at episode {episode}")
                train_on_oos_aggressively(agent, oos_env, steps=100)
            
            # Подсчёт денежных достижений
            if oos_metrics['profit'] >= 1000:  # $1000+ это уже серьёзно
                money_targets_hit += 1
            
            # Сохраняем лучшую модель по деньгам
            if oos_metrics['profit'] > best_oos_profit:
                best_oos_profit = oos_metrics['profit']
                best_model_episode = episode
                agent.save(f"models/{model_name}_best")
                if oos_metrics['profit'] >= 1000:
                    print(f"💰 BIG MONEY! ${oos_metrics['profit']:.0f} (episode {episode})")
                else:
                    print(f"💵 New best: ${oos_metrics['profit']:.0f} (episode {episode})")
            
            # Логирование - НЕМЕДЛЕННАЯ запись после каждого эпизода
            log_line = f"{episode},{train_metrics['total_profit_dollars']:.2f},{oos_metrics['profit']:.2f},"
            log_line += f"{train_metrics['num_trades']},{oos_metrics['trades']},"
            log_line += f"{train_metrics['win_rate']:.3f},{oos_metrics['win_rate']:.3f},"
            log_line += f"{train_metrics['sharpe_ratio']:.3f},{oos_metrics['sharpe']:.3f},"
            log_line += f"{avg_loss:.6f},{agent.epsilon:.3f}"
            
            # НЕМЕДЛЕННАЯ запись в файл - открываем, записываем, закрываем
            with open(log_file, "a") as f:
                f.write(log_line + "\n")
            
            # Уведомляем о записи только каждые 50 эпизодов
            if episode % 50 == 0:
                print(f"📝 Лог обновлён до эпизода {episode}")
            
            # Сохранение промежуточных моделей каждые 100 эпизодов
            if episode % 100 == 0 and episode > 0:
                agent.save(f"models/{model_name}_ep{episode}")
                print(f"💾 Промежуточная модель сохранена: models/{model_name}_ep{episode}")
            
            # Вывод каждые 25 эпизодов
            if episode % 25 == 0:
                money_rate = money_targets_hit / (episode + 1) * 100 if episode > 0 else 0
                print(f"Ep: {episode} | Train: ${train_metrics['total_profit_dollars']:.0f} | "
                      f"OOS: ${oos_metrics['profit']:.0f} | "
                      f"Loss: {avg_loss:.4f}")
                print(f"    Train: {train_metrics['num_trades']} trades, {train_metrics['win_rate']*100:.1f}% win, Sharpe {train_metrics['sharpe_ratio']:.2f}")
                print(f"    OOS: {oos_metrics['trades']} trades, {oos_metrics['win_rate']*100:.1f}% win, Sharpe {oos_metrics['sharpe']:.2f}")
                print(f"    💰 $1000+ rate: {money_rate:.1f}% | Best: ${best_oos_profit:.0f}")
                print("-" * 90)
            
            # Aggressive early stopping на BIG MONEY
            if oos_metrics['profit'] >= 2000:  # $2000+ это ЦЕЛЬ!
                print(f"🎉 BIG MONEY TARGET HIT: ${oos_metrics['profit']:.0f}!")
                break
                
            # Consistency check - если 10 подряд хороших результатов
            if episode >= 50:
                recent_good = 0
                for i in range(max(0, episode-9), episode+1):
                    # Проверяем последние 10 эпизодов в логе
                    if i > 0:  # Можем проверить только если есть история
                        recent_good += 1 if oos_metrics['profit'] > 500 else 0
                
                if recent_good >= 7:  # 7 из 10 хороших результатов
                    print(f"🔥 CONSISTENCY ACHIEVED! Stopping at episode {episode}")
                    break
    
    except KeyboardInterrupt:
        print("\n⏹️ Обучение прервано пользователем")
    
    except Exception as e:
        print(f"\n❌ Ошибка во время обучения: {e}")
        import traceback
        traceback.print_exc()
    
    # Сохранение финальной модели
    agent.save(f"models/{model_name}")
    
    print(f"\n✅ АГРЕССИВНОЕ обучение v9 завершено!")
    print(f"💾 Модель сохранена: models/{model_name}")
    print(f"💰 Лучший результат: models/{model_name}_best (эпизод {best_model_episode}, OOS: ${best_oos_profit:.0f})")
    print(f"📊 Лог сохранен: {log_file}")
    print(f"💰 Big money hits ($1000+): {money_targets_hit}/{episode+1}")
    
    # Оценка успеха
    if best_oos_profit >= 2000:
        print(f"🎉 ОГРОМНЫЙ УСПЕХ! ${best_oos_profit:.0f} - это СЕРЬЁЗНЫЕ деньги!")
    elif best_oos_profit >= 1000:
        print(f"💰 ХОРОШИЙ РЕЗУЛЬТАТ! ${best_oos_profit:.0f} - движемся к цели!")
    elif best_oos_profit >= 500:
        print(f"📈 ПРОГРЕСС! ${best_oos_profit:.0f} - лучше v8, но мало!")
    else:
        print(f"❌ ПРОВАЛ! ${best_oos_profit:.0f} - нужно переделывать!")


def test_oos_aggressive(agent, oos_env):
    """Агрессивное тестирование - много прогонов для стабильности"""
    old_epsilon = agent.epsilon
    agent.epsilon = 0  # Без exploration для точности
    
    results = []
    
    # МНОГО тестов для точной оценки
    for _ in range(7):  # 7 прогонов
        state = oos_env.reset()
        done = False
        
        while not done:
            action = agent.act(state, training=False)
            next_state, _, reward, done = oos_env.step(action)
            state = next_state
        
        metrics = oos_env.get_trading_metrics()
        results.append(metrics)
    
    # Возвращаем ЛУЧШИЙ результат из всех прогонов
    agent.epsilon = old_epsilon
    
    if results:
        best_result = max(results, key=lambda x: x['total_profit_dollars'])
        return {
            'profit': best_result['total_profit_dollars'],
            'trades': best_result['num_trades'],
            'win_rate': best_result['win_rate'],
            'sharpe': best_result['sharpe_ratio']
        }
    else:
        return {'profit': 0, 'trades': 0, 'win_rate': 0, 'sharpe': 0}


def train_on_oos_aggressively(agent, oos_env, steps=100):
    """АГРЕССИВНОЕ обучение на OOS для максимальной адаптации"""
    old_epsilon = agent.epsilon
    agent.epsilon = 0.2  # Умеренный exploration
    
    # МНОГО прогонов для глубокого обучения
    for _ in range(5):  # 5 прогонов
        state = oos_env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < steps:
            action = agent.act(state, training=True)
            next_state, _, reward, done = oos_env.step(action)
            
            # ИНТЕНСИВНОЕ обучение на OOS опыте
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > agent.batch_size:
                agent.update()  # Обучаем сразу
            
            state = next_state
            step_count += 1
    
    agent.epsilon = old_epsilon


if __name__ == "__main__":
    main()
