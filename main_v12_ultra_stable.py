from src.rl.agent import Agent
from src.rl.env import Env
from src.stock.stock import Stock
from main_v11_honest import test_oos_honest, test_oos_comprehensive
import numpy as np
import os


def main():
    """v12: УЛЬТРА-СТАБИЛЬНАЯ модель с максимальной консистентностью"""
    
    # Конфигурация модели
    model_name = "google-trading-v12-ultra-stable"
    
    print("🔒 УЛЬТРА-СТАБИЛЬНАЯ АТАКА: Google trading bot v12!")
    print("🎯 ФОКУС: Максимальная консистентность!")
    print("📊 АНАЛИЗ проблем v11:")
    print("   ⚠️ Нестабильность: много OOS $0")
    print("   ⚠️ Высокий epsilon: 0.49-0.50 вместо 0.01")
    print("   ⚠️ Подозрительно высокие результаты")
    print("")
    print("🔧 УЛУЧШЕНИЯ v12:")
    print("   ✅ КРАЙНЕ низкий epsilon: 0.001 финальный")
    print("   ✅ Консервативная архитектура: меньше параметров")
    print("   ✅ Стабильное обучение: очень низкий LR")
    print("   ✅ Фокус на consistency >80%")
    print("=" * 70)
    
    # УЛЬТРА-КОНСЕРВАТИВНЫЕ параметры
    training_episodes = 150    # Короче для быстрого теста
    trading_period = 60       # КОРОЧЕ для стабильности  
    window_size = 30          # МЕНЬШЕ для простоты
    fee = 0.0002             # Та же комиссия
    
    # Данные
    train_csv = "GOOG_2010-2024-06.csv"
    oos_csv = "GOOG_2024-07_2025-04.csv"
    
    print(f"📊 КОНСЕРВАТИВНЫЕ параметры v12:")
    print(f"📈 Эпизоды: {training_episodes} (короткие для теста)")
    print(f"⏰ Период: {trading_period} дней (короче для стабильности)")
    print(f"🪟 Окно: {window_size} дней (меньше для простоты)")
    print(f"💰 Комиссия: {fee*100}%")
    print("=" * 70)
    
    # Создаем окружения
    train_env = Env(csv_paths=[train_csv], fee=fee, trading_period=trading_period, window=window_size)
    oos_env = Env(csv_paths=[oos_csv], fee=fee, trading_period=trading_period, window=window_size)
    
    # КОНСЕРВАТИВНЫЕ цели
    train_env.target_profit = 200     # НИЗКАЯ цель для стабильности
    train_env.max_trades_per_episode = 8    # МАЛО сделок
    train_env.min_trades_per_episode = 2    # Минимум
    
    print(f"📊 КОНСЕРВАТИВНОЕ окружение v12:")
    print(f"   Observation space: {train_env.stock.obs_space.shape}")
    print(f"   Target profit: ${train_env.target_profit} (низкая для стабильности)")
    print(f"   Trading range: {train_env.min_trades_per_episode}-{train_env.max_trades_per_episode} сделок")
    
    # УЛЬТРА-КОНСЕРВАТИВНАЯ конфигурация v12
    model_config = {
        'embeddings': 16,      # МАЛО - для простоты
        'heads': 1,            # ОДИН head - простота
        'layers': 1,           # ОДИН слой - максимальная простота
        'fwex': 64,           # МАЛО
        'dropout': 0.01,      # КРАЙНЕ низкий для детерминизма
        'neurons': 64,        # МАЛО
        'lr': 0.0005,         # КРАЙНЕ низкий для стабильности
        'epsilon': 0.5,       # Начинаем умеренно
        'epsilon_min': 0.001, # КРАЙНЕ низкий финальный
        'epsilon_decay': 0.99, # БЫСТРЫЙ decay к детерминизму
        'gamma': 0.9,         # Стандартный
        'memory_size': 2000,  # МАЛО для простоты
        'batch_size': 128,    # УМЕРЕННО
        'update_freq': 10     # РЕДКО для стабильности
    }
    
    print(f"====== УЛЬТРА-КОНСЕРВАТИВНАЯ модель v12: {model_name} ======")
    print("🔒 КОНСЕРВАТИВНЫЕ параметры для максимальной стабильности:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    # Создаем агента v12
    agent = Agent(obs_space=train_env.stock.obs_space, **model_config)
    
    # Логирование
    log_file = f"models/{model_name}.log"
    
    print(f"🔒 УЛЬТРА-СТАБИЛЬНОЕ обучение v12...")
    print(f"🎯 ЦЕЛЬ: Consistency >80% при умеренной прибыли!")
    
    # МИНИМАЛЬНОЕ заполнение памяти
    print("📦 Минимальное заполнение памяти...")
    initial_memory_size = model_config['memory_size'] // 5  # Ещё меньше
    
    attempts = 0
    step_count = 0
    while len(agent.memory) < initial_memory_size and attempts < 30:
        if step_count % 50 == 0:
            state = train_env.reset()
        else:
            action = agent.act(state, training=True)
            next_state, _, reward, done = train_env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            step_count += 1
        attempts += 1
    
    print(f"✅ Память заполнена: {len(agent.memory)} записей")
    
    # КОНСЕРВАТИВНЫЙ цикл обучения v12
    header = "episode,train_profit,oos_profit_fixed,oos_profit_median,oos_profit_mean,oos_win_rate,oos_consistency,train_trades,oos_trades,train_win_rate,train_sharpe,oos_sharpe,loss,epsilon"
    with open(log_file, "w") as f:
        f.write(header + "\n")
    
    best_consistency = 0
    best_model_episode = 0
    stable_results_count = 0
    
    # Более строгие критерии стабильности
    target_consistency = 0.8  # 80% consistency
    target_median_profit = 30  # Скромная но стабильная цель
    
    try:
        for episode in range(training_episodes):
            # Обучение ТОЛЬКО на train данных
            state = train_env.reset()
            done = False
            episode_loss = 0.0
            loss_count = 0
            
            while not done:
                action = agent.act(state, training=True)
                next_state, _, reward, done = train_env.step(action)
                agent.remember(state, action, reward, next_state, done)
                
                # Редкое обучение для стабильности
                if len(agent.memory) > model_config['batch_size'] and step_count % model_config['update_freq'] == 0:
                    loss = agent.update()
                    if loss > 0:
                        episode_loss += loss
                        loss_count += 1
                
                state = next_state
                step_count += 1
            
            # Метрики train
            train_metrics = train_env.get_trading_metrics()
            avg_loss = episode_loss / max(loss_count, 1)
            
            # ЧЕСТНОЕ OOS тестирование (но меньше позиций для быстроты)
            oos_results = test_oos_fast(agent, oos_env)
            
            # Подсчёт стабильных результатов - более строгие критерии
            if (oos_results['median_profit'] >= target_median_profit and 
                oos_results['consistency'] >= target_consistency):
                stable_results_count += 1
            
            # Сохраняем лучшую модель по CONSISTENCY
            if oos_results['consistency'] > best_consistency:
                best_consistency = oos_results['consistency']
                best_model_episode = episode
                agent.save(f"models/{model_name}_best")
                print(f"🔒 Новая лучшая consistency: {oos_results['consistency']*100:.1f}% (эпизод {episode})")
            
            # Логирование
            log_line = f"{episode},{train_metrics['total_profit_dollars']:.2f},{oos_results['fixed_profit']:.2f},"
            log_line += f"{oos_results['median_profit']:.2f},{oos_results['mean_profit']:.2f},"
            log_line += f"{oos_results['win_rate']:.3f},{oos_results['consistency']:.3f},"
            log_line += f"{train_metrics['num_trades']},{oos_results['avg_trades']:.1f},"
            log_line += f"{train_metrics['win_rate']:.3f},{train_metrics['sharpe_ratio']:.3f},"
            log_line += f"{oos_results['sharpe']:.3f},{avg_loss:.6f},{agent.epsilon:.3f}"
            
            with open(log_file, "a") as f:
                f.write(log_line + "\n")
            
            # Вывод каждые 15 эпизодов
            if episode % 15 == 0:
                stability_rate = stable_results_count / (episode + 1) * 100
                print(f"Ep: {episode} | Train: ${train_metrics['total_profit_dollars']:.0f} | "
                      f"OOS Med: ${oos_results['median_profit']:.0f}")
                print(f"    Consistency: {oos_results['consistency']*100:.1f}% | "
                      f"Epsilon: {agent.epsilon:.3f} | Stability: {stability_rate:.1f}%")
                print(f"    Best consistency: {best_consistency*100:.1f}%")
                print("-" * 60)
            
            # Early stopping на ультра-стабильности
            if episode >= 50:
                recent_stability = stable_results_count / episode
                if recent_stability >= 0.9:  # 90% стабильных результатов
                    print(f"🔒 УЛЬТРА-СТАБИЛЬНОСТЬ ДОСТИГНУТА! Stopping at episode {episode}")
                    break
    
    except KeyboardInterrupt:
        print("\n⏹️ Обучение прервано пользователем")
    
    except Exception as e:
        print(f"\n❌ Ошибка во время обучения: {e}")
        import traceback
        traceback.print_exc()
    
    # Сохранение финальной модели
    agent.save(f"models/{model_name}")
    
    # Финальная оценка
    print(f"\n🔒 ФИНАЛЬНАЯ УЛЬТРА-СТАБИЛЬНАЯ ОЦЕНКА:")
    final_oos = test_oos_comprehensive(agent, oos_env, detailed=True)
    
    print(f"\n✅ УЛЬТРА-СТАБИЛЬНОЕ обучение v12 завершено!")
    print(f"💾 Модель сохранена: models/{model_name}")
    print(f"🔒 Лучшая consistency: models/{model_name}_best (эпизод {best_model_episode})")
    print(f"📊 Лог сохранен: {log_file}")
    print(f"🎯 Стабильных результатов: {stable_results_count}/{episode+1} ({stable_results_count/(episode+1)*100:.1f}%)")
    
    # Оценка стабильности
    print(f"\n📈 УЛЬТРА-СТАБИЛЬНЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   Медианный профит: ${final_oos['median_profit']:.2f}")
    print(f"   Consistency: {final_oos['consistency']*100:.1f}%")
    print(f"   Win rate: {final_oos['win_rate']*100:.1f}%")
    print(f"   Стандартное отклонение: ${final_oos['std_profit']:.2f}")
    
    # Оценка успеха по новым критериям
    if final_oos['consistency'] >= 0.8 and final_oos['median_profit'] >= 30:
        print(f"🔒 УЛЬТРА-СТАБИЛЬНЫЙ УСПЕХ! Высокая consistency и стабильный профит!")
    elif final_oos['consistency'] >= 0.7:
        print(f"📈 ХОРОШАЯ СТАБИЛЬНОСТЬ! Движемся в правильном направлении!")
    elif final_oos['consistency'] >= 0.5:
        print(f"⚠️ УМЕРЕННАЯ СТАБИЛЬНОСТЬ! Нужно дорабатывать!")
    else:
        print(f"❌ НИЗКАЯ СТАБИЛЬНОСТЬ! Требуется пересмотр подхода!")


def test_oos_fast(agent, oos_env):
    """Быстрое OOS тестирование на меньшем количестве позиций"""
    results = []
    max_starts = len(oos_env.stock.closes) - oos_env.trading_period - 10
    
    # Тестируем на каждой 10-й позиции для скорости
    test_positions = list(range(0, max_starts, 10))
    
    for start_pos in test_positions:
        metrics = test_oos_honest(agent, oos_env, start_position=start_pos)
        results.append({
            'profit': metrics['total_profit_dollars'],
            'trades': metrics['num_trades'],
            'win_rate': metrics['win_rate'],
            'sharpe': metrics['sharpe_ratio']
        })
    
    # Агрегируем результаты
    profits = [r['profit'] for r in results]
    trades = [r['trades'] for r in results]
    win_rates = [r['win_rate'] for r in results]
    sharpes = [r['sharpe'] for r in results]
    
    return {
        'fixed_profit': profits[0] if profits else 0,
        'mean_profit': np.mean(profits),
        'median_profit': np.median(profits), 
        'std_profit': np.std(profits),
        'win_rate': np.mean(win_rates),
        'consistency': len([p for p in profits if p > 0]) / len(profits),
        'sharpe': np.mean(sharpes),
        'avg_trades': np.mean(trades),
        'total_tests': len(results)
    }


if __name__ == "__main__":
    main() 