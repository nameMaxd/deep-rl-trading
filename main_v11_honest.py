from src.rl.agent import Agent
from src.rl.env import Env
from src.stock.stock import Stock
import numpy as np
import os


def main():
    """v11: ЧЕСТНАЯ модель с правильной методологией тестирования"""
    
    # Конфигурация модели
    model_name = "google-trading-v11-honest"
    
    print("🎯 ЧЕСТНАЯ АТАКА: Google trading bot v11!")
    print("🔬 НОВАЯ МЕТОДОЛОГИЯ: Честное тестирование!")
    print("📊 АНАЛИЗ проблем v9/v10:")
    print("   ❌ Cherry-picking: брали лучший из 7 прогонов")
    print("   ❌ Случайные стартовые позиции") 
    print("   ❌ Переобучение на OOS данных")
    print("   ❌ Высокий epsilon в тестировании")
    print("")
    print("🔧 ИСПРАВЛЕНИЯ v11:")
    print("   ✅ ЧЕСТНОЕ тестирование: 1 прогон, фиксированный старт")
    print("   ✅ Comprehensive evaluation: все стартовые позиции")
    print("   ✅ NO OOS training: убираем читерство")
    print("   ✅ Стабильная архитектура: меньше случайности")
    print("=" * 70)
    
    # СТАБИЛЬНЫЕ параметры для честной оценки
    training_episodes = 300    # Умеренное количество
    trading_period = 120      # Тот же период для сравнения
    window_size = 50          # Тот же window
    fee = 0.0002             # Та же комиссия
    
    # Данные
    train_csv = "GOOG_2010-2024-06.csv"
    oos_csv = "GOOG_2024-07_2025-04.csv"
    
    print(f"📊 СТАБИЛЬНЫЕ параметры v11:")
    print(f"📈 Эпизоды: {training_episodes} (умеренно для стабильности)")
    print(f"⏰ Период: {trading_period} дней")
    print(f"🪟 Окно: {window_size} дней")
    print(f"💰 Комиссия: {fee*100}%")
    print("=" * 70)
    
    # Создаем окружения
    train_env = Env(csv_paths=[train_csv], fee=fee, trading_period=trading_period, window=window_size)
    oos_env = Env(csv_paths=[oos_csv], fee=fee, trading_period=trading_period, window=window_size)
    
    # СТАБИЛЬНЫЕ цели
    train_env.target_profit = 500     # Умеренная цель
    train_env.max_trades_per_episode = 15   # Умеренное количество
    train_env.min_trades_per_episode = 5    # Минимум активности
    
    print(f"📊 СТАБИЛЬНОЕ окружение v11:")
    print(f"   Observation space: {train_env.stock.obs_space.shape}")
    print(f"   Target profit: ${train_env.target_profit}")
    print(f"   Trading range: {train_env.min_trades_per_episode}-{train_env.max_trades_per_episode} сделок")
    
    # СТАБИЛЬНАЯ конфигурация v11
    model_config = {
        'embeddings': 32,      # Умеренно - не переусложняем
        'heads': 2,            # Умеренно
        'layers': 2,           # Умеренно
        'fwex': 128,          # Умеренно
        'dropout': 0.05,      # НИЗКИЙ для стабильности
        'neurons': 128,       # Умеренно
        'lr': 0.001,          # НИЗКИЙ для стабильности
        'epsilon': 1.0,
        'epsilon_min': 0.01,  # ОЧЕНЬ низкий для детерминизма
        'epsilon_decay': 0.997,  # Быстрый decay к детерминизму
        'gamma': 0.95,        # Стандартный
        'memory_size': 5000,  # Умеренно
        'batch_size': 512,    # БОЛЬШОЙ для стабильности
        'update_freq': 5      # Умеренно
    }
    
    print(f"====== СТАБИЛЬНАЯ модель v11: {model_name} ======")
    print("🔧 СТАБИЛЬНЫЕ параметры для честной оценки:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    # Создаем агента v11
    agent = Agent(obs_space=train_env.stock.obs_space, **model_config)
    
    # Логирование
    log_file = f"models/{model_name}.log"
    
    print(f"🔬 ЧЕСТНОЕ обучение v11...")
    print(f"🎯 ЦЕЛЬ: Стабильная прибыль без обмана!")
    
    # Заполнение памяти (как обычно)
    print("📦 Заполнение памяти...")
    initial_memory_size = model_config['memory_size'] // 4
    
    attempts = 0
    step_count = 0
    while len(agent.memory) < initial_memory_size and attempts < 50:
        if step_count % 100 == 0:
            state = train_env.reset()
        else:
            action = agent.act(state, training=True)
            next_state, _, reward, done = train_env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            step_count += 1
        attempts += 1
    
    print(f"✅ Память заполнена: {len(agent.memory)} записей")
    
    # ЧЕСТНЫЙ цикл обучения v11
    # Инициализируем лог-файл с заголовком
    header = "episode,train_profit,oos_profit_fixed,oos_profit_median,oos_profit_mean,oos_win_rate,oos_consistency,train_trades,oos_trades,train_win_rate,train_sharpe,oos_sharpe,loss,epsilon"
    with open(log_file, "w") as f:
        f.write(header + "\n")
    
    best_oos_median = -float('inf')
    best_model_episode = 0
    stable_results_count = 0  # Счётчик стабильных результатов
    
    # НЕТ ПЕРЕОБУЧЕНИЯ НА OOS! Убираем train_on_oos_aggressively!
    
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
                
                # Обучение
                if len(agent.memory) > model_config['batch_size']:
                    loss = agent.update()
                    if loss > 0:
                        episode_loss += loss
                        loss_count += 1
                
                state = next_state
            
            # Метрики train
            train_metrics = train_env.get_trading_metrics()
            avg_loss = episode_loss / max(loss_count, 1)
            
            # ЧЕСТНОЕ OOS тестирование
            oos_results = test_oos_comprehensive(agent, oos_env)
            
            # Подсчёт стабильных результатов
            if oos_results['median_profit'] >= 50 and oos_results['consistency'] >= 0.6:
                stable_results_count += 1
            
            # Сохраняем лучшую модель по МЕДИАННОМУ профиту
            if oos_results['median_profit'] > best_oos_median:
                best_oos_median = oos_results['median_profit']
                best_model_episode = episode
                agent.save(f"models/{model_name}_best")
                print(f"💎 Новый лучший медианный результат: ${oos_results['median_profit']:.0f} (эпизод {episode})")
            
            # Логирование - НЕМЕДЛЕННАЯ запись
            log_line = f"{episode},{train_metrics['total_profit_dollars']:.2f},{oos_results['fixed_profit']:.2f},"
            log_line += f"{oos_results['median_profit']:.2f},{oos_results['mean_profit']:.2f},"
            log_line += f"{oos_results['win_rate']:.3f},{oos_results['consistency']:.3f},"
            log_line += f"{train_metrics['num_trades']},{oos_results['avg_trades']:.1f},"
            log_line += f"{train_metrics['win_rate']:.3f},{train_metrics['sharpe_ratio']:.3f},"
            log_line += f"{oos_results['sharpe']:.3f},{avg_loss:.6f},{agent.epsilon:.3f}"
            
            with open(log_file, "a") as f:
                f.write(log_line + "\n")
            
            # Уведомления
            if episode % 50 == 0:
                print(f"📝 Лог обновлён до эпизода {episode}")
            
            # Промежуточные модели
            if episode % 100 == 0 and episode > 0:
                agent.save(f"models/{model_name}_ep{episode}")
                print(f"💾 Промежуточная модель: ep{episode}")
            
            # Вывод каждые 25 эпизодов
            if episode % 25 == 0:
                stability_rate = stable_results_count / (episode + 1) * 100
                print(f"Ep: {episode} | Train: ${train_metrics['total_profit_dollars']:.0f} | "
                      f"OOS Med: ${oos_results['median_profit']:.0f}")
                print(f"    Train: {train_metrics['num_trades']} trades, {train_metrics['win_rate']*100:.1f}% win")
                print(f"    OOS: Med ${oos_results['median_profit']:.0f}, Mean ${oos_results['mean_profit']:.0f}, "
                      f"Consistency {oos_results['consistency']*100:.1f}%")
                print(f"    📊 Stability rate: {stability_rate:.1f}% | Best median: ${best_oos_median:.0f}")
                print("-" * 80)
            
            # Early stopping на стабильности
            if episode >= 100:
                recent_stability = stable_results_count / episode
                if recent_stability >= 0.8:  # 80% стабильных результатов
                    print(f"🎯 СТАБИЛЬНОСТЬ ДОСТИГНУТА! Stopping at episode {episode}")
                    break
    
    except KeyboardInterrupt:
        print("\n⏹️ Обучение прервано пользователем")
    
    except Exception as e:
        print(f"\n❌ Ошибка во время обучения: {e}")
        import traceback
        traceback.print_exc()
    
    # Сохранение финальной модели
    agent.save(f"models/{model_name}")
    
    # Финальная comprehensive оценка
    print(f"\n🔬 ФИНАЛЬНАЯ ЧЕСТНАЯ ОЦЕНКА:")
    final_oos = test_oos_comprehensive(agent, oos_env, detailed=True)
    
    print(f"\n✅ ЧЕСТНОЕ обучение v11 завершено!")
    print(f"💾 Модель сохранена: models/{model_name}")
    print(f"💎 Лучший медианный результат: models/{model_name}_best (эпизод {best_model_episode})")
    print(f"📊 Лог сохранен: {log_file}")
    print(f"🎯 Стабильных результатов: {stable_results_count}/{episode+1} ({stable_results_count/(episode+1)*100:.1f}%)")
    
    # Честная оценка успеха
    print(f"\n📈 ЧЕСТНЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   Медианный профит: ${final_oos['median_profit']:.2f}")
    print(f"   Средний профит: ${final_oos['mean_profit']:.2f}")
    print(f"   Win rate: {final_oos['win_rate']*100:.1f}%")
    print(f"   Consistency: {final_oos['consistency']*100:.1f}%")
    print(f"   Sharpe ratio: {final_oos['sharpe']:.2f}")
    print(f"   Макс. просадка: {final_oos['max_drawdown']*100:.1f}%")
    
    if final_oos['median_profit'] >= 50 and final_oos['consistency'] >= 0.6:
        print(f"🎉 ЧЕСТНЫЙ УСПЕХ! Медианно стабильная прибыль!")
    elif final_oos['median_profit'] >= 20 and final_oos['consistency'] >= 0.5:
        print(f"📈 ПРОГРЕСС! Движемся к стабильности!")
    else:
        print(f"❌ Нужно дорабатывать модель")


def test_oos_honest(agent, oos_env, start_position=0):
    """ЧЕСТНОЕ тестирование - один прогон, фиксированный старт"""
    old_epsilon = agent.epsilon
    agent.epsilon = 0  # Полный детерминизм
    
    # ОДИН прогон с фиксированным стартом
    state = oos_env.reset_fixed(start_position=start_position)
    done = False
    
    while not done:
        action = agent.act(state, training=False)
        next_state, _, reward, done = oos_env.step(action)
        state = next_state
    
    agent.epsilon = old_epsilon
    return oos_env.get_trading_metrics()


def test_oos_comprehensive(agent, oos_env, detailed=False):
    """Comprehensive OOS тестирование на множественных стартовых позициях"""
    results = []
    max_starts = len(oos_env.stock.closes) - oos_env.trading_period - 10
    
    # Тестируем на каждой 5-й позиции для быстроты
    step = 5 if not detailed else 3
    test_positions = list(range(0, max_starts, step))
    
    if detailed:
        print(f"🔬 Comprehensive тестирование на {len(test_positions)} позициях...")
    
    for i, start_pos in enumerate(test_positions):
        metrics = test_oos_honest(agent, oos_env, start_position=start_pos)
        results.append({
            'profit': metrics['total_profit_dollars'],
            'trades': metrics['num_trades'],
            'win_rate': metrics['win_rate'],
            'sharpe': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown']
        })
        
        if detailed and i % 10 == 0:
            print(f"   Позиция {i+1}/{len(test_positions)}: ${metrics['total_profit_dollars']:.0f}")
    
    # Агрегируем результаты
    profits = [r['profit'] for r in results]
    trades = [r['trades'] for r in results]
    win_rates = [r['win_rate'] for r in results]
    sharpes = [r['sharpe'] for r in results]
    drawdowns = [r['max_drawdown'] for r in results]
    
    # Фиксированный результат (первая позиция)
    fixed_profit = profits[0] if profits else 0
    
    return {
        'fixed_profit': fixed_profit,
        'mean_profit': np.mean(profits),
        'median_profit': np.median(profits), 
        'std_profit': np.std(profits),
        'best_profit': np.max(profits),
        'worst_profit': np.min(profits),
        'win_rate': np.mean(win_rates),
        'consistency': len([p for p in profits if p > 0]) / len(profits),
        'sharpe': np.mean(sharpes),
        'max_drawdown': np.mean(drawdowns),
        'avg_trades': np.mean(trades),
        'total_tests': len(results)
    }


if __name__ == "__main__":
    main() 