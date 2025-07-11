#!/usr/bin/env python3
"""
Тестирование v11 модели на РАЗНЫХ OOS периодах
Цель: проверить, не случайные ли хорошие результаты $3065
"""

import torch
import numpy as np
import pandas as pd
from src.rl.env import Env
from src.rl.agent import Agent
import os

def test_model_on_different_oos():
    """Тестируем модель на разных OOS периодах"""
    
    print("🔬 ТЕСТИРОВАНИЕ v11 на РАЗНЫХ OOS периодах!")
    print("🎯 ЦЕЛЬ: Проверить, не случайность ли $3065 результат")
    print("="*70)
    
    # Используем разные CSV файлы для разных периодов
    test_files = [
        # Основной тест на оригинальном файле
        {"name": "Оригинал OOS 2024-07 до 2025-04", "file": "GOOG_2024-07_2025-04.csv"},
        
        # Тест на тренировочных данных (как OOS)
        {"name": "Train данные как OOS 2010-2024-06", "file": "GOOG_2010-2024-06.csv"},
    ]
    
    # Загружаем лучшую модель v11
    model_path = "models/google-trading-v11-honest_best"
    
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        return
    
    results = []
    
    for test_case in test_files:
        print(f"\n🧪 ТЕСТ: {test_case['name']}")
        print(f"📁 Файл: {test_case['file']}")
        
        try:
            if not os.path.exists(test_case['file']):
                print(f"⚠️ Файл не найден: {test_case['file']}")
                continue
            
            # Создаем окружение для OOS теста
            env = Env(
                csv_paths=[test_case['file']], 
                fee=0.0002, 
                trading_period=120, 
                window=50
            )
            
            print(f"📊 Загружено данных: {len(env.stock.closes)} строк")
            
            # Создаем агента с той же архитектурой что в v11
            agent = Agent(obs_space=env.stock.obs_space)
            
            # Загружаем веса
            agent.load(model_path)
            
            # Тестируем на 15 разных стартовых позициях
            profits = []
            trades_list = []
            win_rates = []
            
            max_start_pos = len(env.stock.closes) - 120 - 50  # окно + период
            step_size = max(1, max_start_pos // 15)  # 15 тестов максимум
            
            test_positions = []
            for i in range(0, min(max_start_pos, 150), max(step_size, 10)):
                test_positions.append(i)
            
            print(f"📊 Тестирую позиции: {test_positions[:5]}...{test_positions[-5:] if len(test_positions) > 5 else ''}")
            
            for i, start_pos in enumerate(test_positions[:15]):
                # Сбрасываем среду на фиксированную позицию
                state = env.reset_fixed(start_position=start_pos)
                
                episode_trades = 0
                episode_wins = 0
                
                while env.ind < env.end:
                    # Действие без исследования (epsilon=0)
                    action = agent.act(state, training=False)
                    
                    # Выполняем действие
                    next_state, reward, done, _ = env.step(action)
                    
                    if action != 0:  # Если была торговля
                        episode_trades += 1
                        if reward > 0:
                            episode_wins += 1
                    
                    state = next_state
                    
                    if done or env.ind >= env.end:
                        break
                
                # Получаем метрики эпизода
                metrics = env.get_trading_metrics()
                profit = metrics['total_profit_dollars']
                
                profits.append(profit)
                trades_list.append(episode_trades)
                
                win_rate = episode_wins / max(episode_trades, 1)
                win_rates.append(win_rate)
                
                if i % 3 == 0:  # Прогресс каждые 3 теста
                    print(f"  Тест {i+1}/15: start={start_pos}, profit=${profit:.0f}, trades={episode_trades}")
                
                if len(profits) >= 15:  # Ограничиваем 15 тестами
                    break
            
            if profits:
                median_profit = np.median(profits)
                mean_profit = np.mean(profits)
                std_profit = np.std(profits)
                positive_rate = (np.array(profits) > 0).mean() * 100
                avg_trades = np.mean(trades_list)
                avg_win_rate = np.mean(win_rates)
                
                print(f"💰 Результаты ({len(profits)} тестов):")
                print(f"   Медиана: ${median_profit:.0f}")
                print(f"   Среднее: ${mean_profit:.0f}")
                print(f"   Стандарт. откл.: ${std_profit:.0f}")
                print(f"   Прибыльных: {positive_rate:.1f}%")
                print(f"   Диапазон: ${min(profits):.0f} - ${max(profits):.0f}")
                print(f"   Сделок в среднем: {avg_trades:.1f}")
                print(f"   Win rate: {avg_win_rate*100:.1f}%")
                
                results.append({
                    'test_name': test_case['name'],
                    'file': test_case['file'],
                    'data_points': len(env.stock.closes),
                    'tests_count': len(profits),
                    'median_profit': median_profit,
                    'mean_profit': mean_profit,
                    'std_profit': std_profit,
                    'positive_rate': positive_rate,
                    'min_profit': min(profits),
                    'max_profit': max(profits),
                    'avg_trades': avg_trades,
                    'avg_win_rate': avg_win_rate,
                    'all_profits': profits
                })
                
        except Exception as e:
            print(f"❌ Ошибка при тестировании {test_case['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Итоговый анализ
    print("\n" + "="*70)
    print("📈 ИТОГОВЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ:")
    print("="*70)
    
    if results:
        df = pd.DataFrame(results)
        
        print(f"\n🎯 Успешно протестировано: {len(results)} периодов")
        print(f"📊 Общая статистика:")
        print(f"   Средняя медиана: ${df['median_profit'].mean():.0f}")
        print(f"   Медиана медиан: ${df['median_profit'].median():.0f}")
        print(f"   Лучший результат: ${df['median_profit'].max():.0f}")
        print(f"   Худший результат: ${df['median_profit'].min():.0f}")
        print(f"   Стабильность: {(df['median_profit'] > 0).mean() * 100:.1f}% периодов прибыльны")
        
        # Топ результаты
        print(f"\n🏆 РЕЗУЛЬТАТЫ:")
        for i, row in df.iterrows():
            print(f"   {row['test_name']}: ${row['median_profit']:.0f}")
        
        # Анализ оригинального результата
        original = df[df['test_name'].str.contains('Оригинал')]
        if len(original) > 0:
            orig_median = original.iloc[0]['median_profit']
            print(f"\n🎯 АНАЛИЗ оригинального результата ($3065 ожидался):")
            print(f"   Результат на оригинальном OOS: ${orig_median:.0f}")
            
            # Сравнение с тренировочными данными
            train_test = df[df['test_name'].str.contains('Train')]
            if len(train_test) > 0:
                train_median = train_test.iloc[0]['median_profit']
                print(f"   Результат на тренировочных данных: ${train_median:.0f}")
                
                if orig_median > train_median:
                    print(f"   ✅ OOS лучше train данных - хороший знак!")
                else:
                    print(f"   ⚠️ OOS хуже train - возможно переобучение")
                    
                if orig_median >= 1000:
                    print(f"   💰 ОТЛИЧНЫЙ результат - более $1000!")
                elif orig_median >= 500:
                    print(f"   👍 ХОРОШИЙ результат - более $500")
                elif orig_median >= 100:
                    print(f"   🤔 УМЕРЕННЫЙ результат - более $100")
                else:
                    print(f"   ❌ ПЛОХОЙ результат - менее $100")
        
        # Сохраняем детальный отчет
        with open("oos_variation_test_results.txt", "w", encoding='utf-8') as f:
            f.write("🔬 ТЕСТИРОВАНИЕ v11 НА РАЗНЫХ OOS ПЕРИОДАХ\n")
            f.write("="*70 + "\n\n")
            
            for result in results:
                f.write(f"Тест: {result['test_name']}\n")
                f.write(f"Файл: {result['file']}\n")
                f.write(f"Данных: {result['data_points']}, Тестов: {result['tests_count']}\n")
                f.write(f"Медиана: ${result['median_profit']:.0f}\n")
                f.write(f"Среднее: ${result['mean_profit']:.0f}\n")
                f.write(f"Диапазон: ${result['min_profit']:.0f} - ${result['max_profit']:.0f}\n")
                f.write(f"Прибыльных: {result['positive_rate']:.1f}%\n")
                f.write(f"Сделок: {result['avg_trades']:.1f}, Win rate: {result['avg_win_rate']*100:.1f}%\n")
                f.write(f"Все результаты: {result['all_profits']}\n")
                f.write("-"*50 + "\n")
        
        print(f"\n💾 Детальный отчет сохранен: oos_variation_test_results.txt")
        
    else:
        print("❌ Не удалось получить результаты ни для одного периода!")

if __name__ == "__main__":
    test_model_on_different_oos() 