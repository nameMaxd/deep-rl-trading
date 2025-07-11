"""
Батч-скрипт для сравнения нескольких версий торгового бота подряд
"""
import subprocess
import os
import time
import pandas as pd
from datetime import datetime


def run_version(script_name, version_name):
    """Запускает версию и возвращает результаты"""
    print(f"\n{'='*60}")
    print(f"🚀 ЗАПУСК: {version_name}")
    print(f"📜 Скрипт: {script_name}")
    print(f"⏰ Время: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Запускаем скрипт
        result = subprocess.run(['python', script_name], 
                              capture_output=True, 
                              text=True, 
                              timeout=3600)  # 1 час таймаут
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ {version_name} завершён за {duration/60:.1f} мин")
        
        # Проверяем успех
        if result.returncode == 0:
            print(f"✅ Успешно: {version_name}")
            return {'success': True, 'duration': duration, 'output': result.stdout}
        else:
            print(f"❌ Ошибка в {version_name}: {result.stderr}")
            return {'success': False, 'duration': duration, 'error': result.stderr}
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Таймаут {version_name} (>1 час)")
        return {'success': False, 'duration': 3600, 'error': 'Timeout'}
    except Exception as e:
        print(f"❌ Исключение в {version_name}: {e}")
        return {'success': False, 'duration': 0, 'error': str(e)}


def analyze_results():
    """Анализирует результаты всех версий"""
    print(f"\n{'='*80}")
    print("📊 СРАВНИТЕЛЬНЫЙ АНАЛИЗ ВСЕХ ВЕРСИЙ")
    print(f"{'='*80}")
    
    versions = [
        ('v11', 'models/google-trading-v11-honest.log'),
        ('v12', 'models/google-trading-v12-ultra-stable.log')
    ]
    
    results_summary = []
    
    for version_name, log_file in versions:
        if os.path.exists(log_file):
            try:
                df = pd.read_csv(log_file)
                
                # Анализируем результаты
                final_episode = df.iloc[-1]
                
                # Статистики
                summary = {
                    'version': version_name,
                    'episodes': len(df),
                    'final_median_profit': final_episode['oos_profit_median'],
                    'final_consistency': final_episode['oos_consistency'],
                    'final_epsilon': final_episode['epsilon'],
                    'best_median_profit': df['oos_profit_median'].max(),
                    'best_consistency': df['oos_consistency'].max(),
                    'avg_consistency': df['oos_consistency'].mean(),
                    'consistency_std': df['oos_consistency'].std()
                }
                
                results_summary.append(summary)
                
                print(f"\n🔍 {version_name.upper()} АНАЛИЗ:")
                print(f"   Эпизодов: {summary['episodes']}")
                print(f"   Финальный медианный OOS: ${summary['final_median_profit']:.0f}")
                print(f"   Финальная consistency: {summary['final_consistency']*100:.1f}%")
                print(f"   Финальный epsilon: {summary['final_epsilon']:.3f}")
                print(f"   Лучший медианный OOS: ${summary['best_median_profit']:.0f}")
                print(f"   Лучшая consistency: {summary['best_consistency']*100:.1f}%")
                print(f"   Средняя consistency: {summary['avg_consistency']*100:.1f}%")
                print(f"   Стабильность (std): {summary['consistency_std']*100:.1f}%")
                
            except Exception as e:
                print(f"❌ Ошибка анализа {version_name}: {e}")
        else:
            print(f"⚠️ Лог {version_name} не найден: {log_file}")
    
    # Сравнение
    if len(results_summary) >= 2:
        print(f"\n🏆 СРАВНЕНИЕ ВЕРСИЙ:")
        print("-" * 50)
        
        # Сортируем по consistency
        sorted_by_consistency = sorted(results_summary, 
                                     key=lambda x: x['avg_consistency'], 
                                     reverse=True)
        
        print("📊 По средней consistency:")
        for i, result in enumerate(sorted_by_consistency):
            icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"   {icon} {result['version']}: {result['avg_consistency']*100:.1f}%")
        
        # Сортируем по прибыли
        sorted_by_profit = sorted(results_summary, 
                                key=lambda x: x['best_median_profit'], 
                                reverse=True)
        
        print("\n💰 По лучшему медианному профиту:")
        for i, result in enumerate(sorted_by_profit):
            icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"   {icon} {result['version']}: ${result['best_median_profit']:.0f}")
        
        # Рекомендации
        best_consistency = sorted_by_consistency[0]
        best_profit = sorted_by_profit[0]
        
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        if best_consistency['version'] == best_profit['version']:
            print(f"   🎯 {best_consistency['version']} - лучший во всём!")
        else:
            print(f"   🔒 {best_consistency['version']} - для стабильности")
            print(f"   💰 {best_profit['version']} - для прибыли")


def main():
    """Основная функция батч-тестирования"""
    print("🔬 БАТЧ-СРАВНЕНИЕ ВЕРСИЙ ТОРГОВОГО БОТА")
    print("=" * 60)
    print(f"⏰ Старт: {datetime.now().strftime('%H:%M:%S')}")
    
    # Список версий для тестирования
    versions_to_test = [
        ('main_v12_ultra_stable.py', 'v12 Ultra-Stable'),
        # Добавим другие версии по мере создания
    ]
    
    results = {}
    total_start = time.time()
    
    # Запускаем версии подряд
    for script, name in versions_to_test:
        if os.path.exists(script):
            results[name] = run_version(script, name)
        else:
            print(f"⚠️ Скрипт не найден: {script}")
            results[name] = {'success': False, 'error': 'Script not found'}
    
    total_time = time.time() - total_start
    
    # Сводка запусков
    print(f"\n{'='*80}")
    print("📋 СВОДКА ЗАПУСКОВ")
    print(f"{'='*80}")
    print(f"⏰ Общее время: {total_time/60:.1f} мин")
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    print(f"✅ Успешных: {successful}/{total}")
    
    for name, result in results.items():
        status = "✅" if result['success'] else "❌"
        duration = result.get('duration', 0) / 60
        print(f"   {status} {name}: {duration:.1f} мин")
    
    # Анализируем результаты
    analyze_results()
    
    print(f"\n🎉 БАТЧ-ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print(f"⏰ Время: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main() 