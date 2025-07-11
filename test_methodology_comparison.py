"""
Сравнение старой (cherry-picking) и новой (честной) методологии тестирования
"""
from src.rl.agent import Agent
from src.rl.env import Env
from main_v11_honest import test_oos_honest, test_oos_comprehensive
import numpy as np
import os


def test_old_methodology(agent, oos_env):
    """Старая методология: cherry-picking лучшего из 7 прогонов"""
    old_epsilon = agent.epsilon
    agent.epsilon = 0.3  # Как в старых логах
    
    results = []
    
    # 7 прогонов с случайными стартами (как в старом коде)
    for _ in range(7):
        state = oos_env.reset()  # Случайный старт!
        done = False
        
        while not done:
            action = agent.act(state, training=False)
            next_state, _, reward, done = oos_env.step(action)
            state = next_state
        
        metrics = oos_env.get_trading_metrics()
        results.append(metrics)
    
    # Возвращаем ЛУЧШИЙ результат (cherry-picking!)
    agent.epsilon = old_epsilon
    
    if results:
        best_result = max(results, key=lambda x: x['total_profit_dollars'])
        all_profits = [r['total_profit_dollars'] for r in results]
        return {
            'cherry_picked_profit': best_result['total_profit_dollars'],
            'all_profits': all_profits,
            'mean_profit': np.mean(all_profits),
            'median_profit': np.median(all_profits),
            'trades': best_result['num_trades'],
            'win_rate': best_result['win_rate'],
            'sharpe': best_result['sharpe_ratio']
        }
    else:
        return {'cherry_picked_profit': 0, 'all_profits': [], 'mean_profit': 0, 'median_profit': 0, 'trades': 0, 'win_rate': 0, 'sharpe': 0}


def main():
    """Сравнение методологий на существующей модели"""
    
    print("🔬 СРАВНЕНИЕ МЕТОДОЛОГИЙ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    print("📊 Тестируем на модели elysium-v1 (если есть)")
    
    # Настройки (как в v11)
    trading_period = 120
    window_size = 50
    fee = 0.0002
    oos_csv = "GOOG_2024-07_2025-04.csv"
    
    # Создаем окружение
    oos_env = Env(csv_paths=[oos_csv], fee=fee, trading_period=trading_period, window=window_size)
    
    # Создаем простого агента для демонстрации
    agent = Agent(obs_space=oos_env.stock.obs_space, epsilon=0.3)
    
    # Пытаемся загрузить существующую модель
    model_path = "models/elysium-v1"
    if os.path.exists(model_path):
        print(f"📥 Загружаем модель: {model_path}")
        agent.load(model_path)
    else:
        print("⚠️ Модель не найдена, используем случайного агента для демонстрации")
    
    print("\n🔍 ТЕСТ 1: Старая методология (cherry-picking)")
    print("-" * 40)
    
    old_results = test_old_methodology(agent, oos_env)
    print(f"Cherry-picked profit: ${old_results['cherry_picked_profit']:.0f}")
    print(f"Все 7 результатов: {[f'${p:.0f}' for p in old_results['all_profits']]}")
    print(f"Реальное среднее: ${old_results['mean_profit']:.0f}")
    print(f"Медианное: ${old_results['median_profit']:.0f}")
    print(f"Trades: {old_results['trades']}, Win rate: {old_results['win_rate']*100:.1f}%")
    
    print("\n🎯 ТЕСТ 2: Новая методология (честная)")
    print("-" * 40)
    
    new_results = test_oos_comprehensive(agent, oos_env, detailed=True)
    print(f"Фиксированный результат: ${new_results['fixed_profit']:.0f}")
    print(f"Медианный результат: ${new_results['median_profit']:.0f}")
    print(f"Средний результат: ${new_results['mean_profit']:.0f}")
    print(f"Лучший результат: ${new_results['best_profit']:.0f}")
    print(f"Худший результат: ${new_results['worst_profit']:.0f}")
    print(f"Consistency: {new_results['consistency']*100:.1f}%")
    print(f"Протестировано позиций: {new_results['total_tests']}")
    
    print("\n📊 СРАВНЕНИЕ")
    print("=" * 60)
    
    cherry_vs_median = old_results['cherry_picked_profit'] - new_results['median_profit']
    cherry_vs_mean = old_results['cherry_picked_profit'] - new_results['mean_profit']
    
    print(f"Cherry-picked: ${old_results['cherry_picked_profit']:.0f}")
    print(f"Честное медианное: ${new_results['median_profit']:.0f}")
    print(f"Разница (переоценка): ${cherry_vs_median:.0f}")
    print(f"")
    print(f"Cherry-picked: ${old_results['cherry_picked_profit']:.0f}")
    print(f"Честное среднее: ${new_results['mean_profit']:.0f}")
    print(f"Разница (переоценка): ${cherry_vs_mean:.0f}")
    
    print(f"\n💡 ВЫВОДЫ:")
    if cherry_vs_median > 100:
        print(f"❌ Старая методология СИЛЬНО переоценивает результат на ${cherry_vs_median:.0f}")
    elif cherry_vs_median > 50:
        print(f"⚠️ Старая методология переоценивает результат на ${cherry_vs_median:.0f}")
    elif cherry_vs_median > 0:
        print(f"📊 Старая методология слегка переоценивает на ${cherry_vs_median:.0f}")
    else:
        print(f"✅ Методологии дают схожие результаты")
    
    print(f"\n🎯 ЧЕСТНАЯ ОЦЕНКА:")
    if new_results['consistency'] >= 0.6:
        print(f"✅ Хорошая consistency: {new_results['consistency']*100:.1f}%")
    else:
        print(f"❌ Низкая consistency: {new_results['consistency']*100:.1f}%")
    
    if new_results['median_profit'] >= 50:
        print(f"✅ Приемлемый медианный профит: ${new_results['median_profit']:.0f}")
    else:
        print(f"❌ Низкий медианный профит: ${new_results['median_profit']:.0f}")


if __name__ == "__main__":
    main() 