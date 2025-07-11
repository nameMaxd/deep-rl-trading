#!/usr/bin/env python3
"""
🏆 БОЛЬШОЕ СРАВНЕНИЕ АРХИТЕКТУР!
🎯 Запускаем все версии подряд и сравниваем результаты
"""

import subprocess
import time
import os
import json
from datetime import datetime
import re

def run_experiment(script_name, description):
    """Запускает один эксперимент и возвращает результаты"""
    
    print(f"\n🚀 ЗАПУСК: {description}")
    print(f"📁 Скрипт: {script_name}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Запускаем скрипт
        result = subprocess.run(
            ["python", script_name], 
            capture_output=True, 
            text=True, 
            timeout=3600  # 1 час максимум
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} завершен успешно!")
            print(f"⏱️ Время выполнения: {duration:.0f} секунд")
            
            # Извлекаем финальные результаты из вывода
            output = result.stdout
            final_results = extract_final_results(output, script_name)
            final_results['duration'] = duration
            final_results['success'] = True
            final_results['script'] = script_name
            final_results['description'] = description
            
            return final_results
            
        else:
            print(f"❌ {description} завершен с ошибкой!")
            print(f"Ошибка: {result.stderr}")
            
            return {
                'script': script_name,
                'description': description,
                'success': False,
                'error': result.stderr,
                'duration': duration
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} превысил лимит времени!")
        return {
            'script': script_name,
            'description': description,
            'success': False,
            'error': 'Timeout',
            'duration': 3600
        }
    except Exception as e:
        print(f"💥 Ошибка при запуске {description}: {e}")
        return {
            'script': script_name,
            'description': description,
            'success': False,
            'error': str(e),
            'duration': 0
        }

def extract_final_results(output, script_name):
    """Извлекает финальные результаты из вывода скрипта"""
    
    results = {
        'median_profit': 0,
        'consistency': 0,
        'std_dev': 0,
        'best_median': 0,
        'stability_count': 0,
        'range_min': 0,
        'range_max': 0
    }
    
    try:
        # Ищем финальные результаты
        lines = output.split('\n')
        
        for i, line in enumerate(lines):
            # Медианный профит
            if 'Медианный профит:' in line:
                match = re.search(r'\\$([\\d\\.\\-]+)', line)
                if match:
                    results['median_profit'] = float(match.group(1))
            
            # Consistency
            if 'Consistency:' in line:
                match = re.search(r'([\\d\\.]+)%', line)
                if match:
                    results['consistency'] = float(match.group(1))
            
            # Стандартное отклонение
            if 'Стандартное отклонение:' in line:
                match = re.search(r'\\$([\\d\\.]+)', line)
                if match:
                    results['std_dev'] = float(match.group(1))
            
            # Лучшая медиана во время обучения
            if 'Best median:' in line:
                match = re.search(r'\\$([\\d\\.\\-]+)', line)
                if match:
                    results['best_median'] = max(results['best_median'], float(match.group(1)))
            
            # Диапазон
            if 'Диапазон:' in line:
                match = re.search(r'\\$([\\d\\.\\-]+) - \\$([\\d\\.\\-]+)', line)
                if match:
                    results['range_min'] = float(match.group(1))
                    results['range_max'] = float(match.group(2))
            
            # Стабильные результаты
            if 'Стабильных результатов:' in line:
                match = re.search(r'(\\d+)/(\\d+)', line)
                if match:
                    results['stability_count'] = int(match.group(1))
                    results['total_tests'] = int(match.group(2))
    
    except Exception as e:
        print(f"⚠️ Ошибка при извлечении результатов: {e}")
    
    return results

def create_comparison_report(all_results):
    """Создает итоговый отчет сравнения"""
    
    report_file = f"architecture_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 🏆 БОЛЬШОЕ СРАВНЕНИЕ АРХИТЕКТУР\\n\\n")
        f.write(f"**Дата проведения:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\\n\\n")
        f.write("## 📊 Сравнительная таблица\\n\\n")
        f.write("| Архитектура | Медиана $ | Consistency % | Лучшая $ | Стабильность | Время мин | Статус |\\n")
        f.write("|-------------|-----------|---------------|----------|--------------|-----------|--------|\\n")
        
        # Сортируем по медианной прибыли
        successful_results = [r for r in all_results if r['success']]
        successful_results.sort(key=lambda x: x.get('median_profit', -1000), reverse=True)
        
        for result in all_results:
            if result['success']:
                name = result['description'].replace('v13_LSTM', 'LSTM').replace('v14_Transformer_Fixed', 'Transformer').replace('v15_SSM', 'SSM').replace('v16_MLP', 'MLP')
                median = result.get('median_profit', 0)
                consistency = result.get('consistency', 0)
                best = result.get('best_median', 0)
                stability = f"{result.get('stability_count', 0)}/{result.get('total_tests', 0)}" if result.get('total_tests') else "N/A"
                duration = result['duration'] / 60
                status = "✅"
                
                f.write(f"| {name} | ${median:.0f} | {consistency:.1f}% | ${best:.0f} | {stability} | {duration:.0f} | {status} |\\n")
            else:
                name = result['description']
                f.write(f"| {name} | N/A | N/A | N/A | N/A | {result['duration']/60:.0f} | ❌ |\\n")
        
        f.write("\\n")
        
        # Детальный анализ
        f.write("## 🔍 Детальный анализ\\n\\n")
        
        if successful_results:
            winner = successful_results[0]
            f.write(f"### 🏆 ПОБЕДИТЕЛЬ: {winner['description']}\\n\\n")
            f.write(f"**Медианная прибыль:** ${winner.get('median_profit', 0):.2f}\\n")
            f.write(f"**Consistency:** {winner.get('consistency', 0):.1f}%\\n")
            f.write(f"**Лучший результат:** ${winner.get('best_median', 0):.0f}\\n")
            f.write(f"**Диапазон:** ${winner.get('range_min', 0):.0f} - ${winner.get('range_max', 0):.0f}\\n")
            f.write(f"**Время обучения:** {winner['duration']/60:.0f} минут\\n\\n")
        
        # Ranking по разным метрикам
        f.write("### 📈 Рейтинги по метрикам\\n\\n")
        
        if len(successful_results) > 1:
            # По медианной прибыли
            f.write("**По медианной прибыли:**\\n")
            for i, result in enumerate(successful_results, 1):
                f.write(f"{i}. {result['description']}: ${result.get('median_profit', 0):.0f}\\n")
            f.write("\\n")
            
            # По consistency
            consistency_sorted = sorted(successful_results, key=lambda x: x.get('consistency', 0), reverse=True)
            f.write("**По стабильности (consistency):**\\n")
            for i, result in enumerate(consistency_sorted, 1):
                f.write(f"{i}. {result['description']}: {result.get('consistency', 0):.1f}%\\n")
            f.write("\\n")
            
            # По лучшему результату
            best_sorted = sorted(successful_results, key=lambda x: x.get('best_median', 0), reverse=True)
            f.write("**По лучшему результату во время обучения:**\\n")
            for i, result in enumerate(best_sorted, 1):
                f.write(f"{i}. {result['description']}: ${result.get('best_median', 0):.0f}\\n")
            f.write("\\n")
        
        # Выводы и рекомендации
        f.write("## 💡 Выводы и рекомендации\\n\\n")
        
        if successful_results:
            best_profit = successful_results[0]
            best_consistency = max(successful_results, key=lambda x: x.get('consistency', 0))
            
            f.write(f"1. **Лучшая прибыльность:** {best_profit['description']} (${best_profit.get('median_profit', 0):.0f})\\n")
            f.write(f"2. **Лучшая стабильность:** {best_consistency['description']} ({best_consistency.get('consistency', 0):.1f}%)\\n")
            
            # Анализ архитектур
            f.write("\\n### 🏗️ Анализ архитектур:\\n\\n")
            
            architecture_analysis = {
                'LSTM': 'Простая, быстрая, хорошо обрабатывает последовательности',
                'Transformer': 'Мощная, attention механизм, но может переобучаться',
                'SSM': 'Современная, эффективная для длинных последовательностей',
                'MLP': 'Baseline, простая реализация, быстрое обучение'
            }
            
            for result in successful_results:
                arch_type = None
                for arch in architecture_analysis.keys():
                    if arch in result['description'].upper():
                        arch_type = arch
                        break
                
                if arch_type:
                    f.write(f"**{arch_type}:** {architecture_analysis[arch_type]}\\n")
                    f.write(f"- Результат: ${result.get('median_profit', 0):.0f} медиана, {result.get('consistency', 0):.1f}% consistency\\n")
                    f.write(f"- Время: {result['duration']/60:.0f} минут\\n\\n")
        
        else:
            f.write("❌ Все эксперименты завершились неудачно. Требуется отладка.\\n\\n")
        
        # Детали ошибок
        failed_results = [r for r in all_results if not r['success']]
        if failed_results:
            f.write("## ❌ Неудачные эксперименты\\n\\n")
            for result in failed_results:
                f.write(f"**{result['description']}:**\\n")
                f.write(f"- Ошибка: {result.get('error', 'Unknown error')}\\n")
                f.write(f"- Время до ошибки: {result['duration']/60:.0f} минут\\n\\n")
    
    return report_file

def main():
    """Основная функция сравнения"""
    
    print("🏆 ЗАПУСК БОЛЬШОГО СРАВНЕНИЯ АРХИТЕКТУР!")
    print("🎯 Тестируем: LSTM, Transformer, SSM, MLP")
    print("=" * 70)
    
    # Список экспериментов для запуска
    experiments = [
        ("main_v13_lstm.py", "v13_LSTM - Простая LSTM архитектура"),
        ("main_v14_transformer_fixed.py", "v14_Transformer_Fixed - Фиксированный трансформер"),
        ("main_v15_ssm.py", "v15_SSM - State Space Models (Mamba)"),
        ("main_v16_mlp.py", "v16_MLP - Глубокие MLP с residuals")
    ]
    
    # Проверяем наличие всех скриптов
    missing_scripts = []
    for script, desc in experiments:
        if not os.path.exists(script):
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"❌ Отсутствуют скрипты: {missing_scripts}")
        print("Создайте все скрипты перед запуском сравнения!")
        return
    
    # Запускаем все эксперименты
    all_results = []
    total_start_time = time.time()
    
    for i, (script, description) in enumerate(experiments, 1):
        print(f"\n🔄 Эксперимент {i}/{len(experiments)}")
        result = run_experiment(script, description)
        all_results.append(result)
        
        # Небольшая пауза между экспериментами
        if i < len(experiments):
            print("⏸️ Пауза 30 секунд перед следующим экспериментом...")
            time.sleep(30)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print("\n" + "=" * 70)
    print("🏁 ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print(f"⏱️ Общее время: {total_duration/60:.0f} минут")
    print("=" * 70)
    
    # Создаем отчет
    report_file = create_comparison_report(all_results)
    
    print(f"\n📊 Создан отчет сравнения: {report_file}")
    
    # Краткая сводка в консоль
    print("\n🏆 КРАТКИЕ РЕЗУЛЬТАТЫ:")
    successful = [r for r in all_results if r['success']]
    
    if successful:
        successful.sort(key=lambda x: x.get('median_profit', -1000), reverse=True)
        
        print("\nРейтинг по медианной прибыли:")
        for i, result in enumerate(successful, 1):
            median = result.get('median_profit', 0)
            consistency = result.get('consistency', 0)
            name = result['description'].split(' - ')[0]
            print(f"{i}. {name}: ${median:.0f} (consistency {consistency:.1f}%)")
        
        winner = successful[0]
        print(f"\n🥇 ПОБЕДИТЕЛЬ: {winner['description']}")
        print(f"   💰 Медиана: ${winner.get('median_profit', 0):.2f}")
        print(f"   📊 Consistency: {winner.get('consistency', 0):.1f}%")
        print(f"   ⚡ Время: {winner['duration']/60:.0f} минут")
    else:
        print("❌ Все эксперименты завершились неудачно!")
    
    failed = [r for r in all_results if not r['success']]
    if failed:
        print(f"\n❌ Неудачных экспериментов: {len(failed)}")
        for result in failed:
            print(f"   - {result['description']}: {result.get('error', 'Unknown error')}")
    
    # Сохраняем сырые данные
    with open(f"raw_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json", 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Сравнение завершено! Проверьте отчет: {report_file}")

if __name__ == "__main__":
    main() 