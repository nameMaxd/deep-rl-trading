#!/usr/bin/env python3
"""
🚀🚀🚀 БОЛЬШОЕ СРАВНЕНИЕ АРХИТЕКТУР С TQDM! 🚀🚀🚀
🎯 Запускаем все 4 архитектуры и сравниваем результаты!
"""

import subprocess
import sys
import time
from datetime import datetime
import json

def run_architecture(script_name, arch_name):
    """Запускает архитектуру и возвращает результат"""
    print(f"\n🚀 ЗАПУСКАЮ {arch_name}...")
    print(f"📝 Скрипт: {script_name}")
    print("=" * 60)
    
    try:
        # Запускаем скрипт
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True,
                              timeout=3600)  # 1 час максимум
        
        if result.returncode == 0:
            print(f"✅ {arch_name} ЗАВЕРШЕН УСПЕШНО!")
            return {'success': True, 'arch': arch_name}
        else:
            print(f"❌ {arch_name} ЗАВЕРШЕН С ОШИБКОЙ!")
            return {'success': False, 'arch': arch_name, 'error': 'Non-zero exit code'}
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {arch_name} ПРЕВЫШЕН ТАЙМАУТ!")
        return {'success': False, 'arch': arch_name, 'error': 'Timeout'}
    except Exception as e:
        print(f"💥 {arch_name} ОШИБКА: {e}")
        return {'success': False, 'arch': arch_name, 'error': str(e)}

def main():
    print("🚀🚀🚀 БОЛЬШОЕ СРАВНЕНИЕ АРХИТЕКТУР! 🚀🚀🚀")
    print("🎯 ТЕСТИРУЕМ ВСЕ 4 АРХИТЕКТУРЫ С TQDM!")
    print("=" * 80)
    
    start_time = datetime.now()
    print(f"⏰ Начало: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Архитектуры для тестирования
    architectures = [
        ("tqdm_lstm_v13.py", "🧠 LSTM v13"),
        ("tqdm_transformer_v14.py", "🤖 TRANSFORMER v14"), 
        ("tqdm_mlp_v16.py", "🔥 MLP v16"),
        ("working_test_with_tqdm.py", "⚡ BASELINE")  # Простой baseline
    ]
    
    results = []
    
    print(f"\n📋 ПЛАН ЗАПУСКА:")
    for i, (script, name) in enumerate(architectures, 1):
        print(f"   {i}. {name} - {script}")
    
    print(f"\n🎯 ЦЕЛЬ: Сравнить все архитектуры честно!")
    print(f"⏱️ Ожидаемое время: ~40-60 минут")
    print("🔥 ПОЕХАЛИ!")
    
    # Запускаем архитектуры последовательно
    for i, (script, name) in enumerate(architectures, 1):
        print(f"\n{'='*80}")
        print(f"🎯 ЭТАП {i}/{len(architectures)}: {name}")
        print(f"{'='*80}")
        
        arch_start = time.time()
        result = run_architecture(script, name)
        arch_time = time.time() - arch_start
        
        result['duration'] = arch_time
        result['script'] = script
        results.append(result)
        
        print(f"⏱️ Время выполнения: {arch_time:.1f} секунд")
        
        if result['success']:
            print(f"✅ {name} ГОТОВ!")
        else:
            print(f"❌ {name} НЕ УДАЛСЯ: {result.get('error', 'Unknown error')}")
        
        # Небольшая пауза между архитектурами
        if i < len(architectures):
            print(f"\n⏸️ Пауза 5 секунд перед следующей архитектурой...")
            time.sleep(5)
    
    # Итоговый отчет
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"🏁 БОЛЬШОЕ СРАВНЕНИЕ ЗАВЕРШЕНО!")
    print(f"{'='*80}")
    print(f"⏰ Общее время: {total_time}")
    print(f"📊 Результаты:")
    
    successful = 0
    failed = 0
    
    for result in results:
        status = "✅ УСПЕШНО" if result['success'] else "❌ ОШИБКА"
        duration = f"{result['duration']:.1f}s"
        print(f"   {result['arch']}: {status} ({duration})")
        
        if result['success']:
            successful += 1
        else:
            failed += 1
    
    print(f"\n📈 ИТОГИ:")
    print(f"   ✅ Успешных: {successful}")
    print(f"   ❌ Неудачных: {failed}")
    print(f"   📊 Успешность: {successful/len(architectures)*100:.1f}%")
    
    # Сохраняем отчет
    report = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'total_duration': str(total_time),
        'results': results,
        'summary': {
            'successful': successful,
            'failed': failed,
            'success_rate': successful/len(architectures)*100
        }
    }
    
    with open('БОЛЬШОЕ_СРАВНЕНИЕ_ОТЧЕТ.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Отчет сохранен: БОЛЬШОЕ_СРАВНЕНИЕ_ОТЧЕТ.json")
    
    if successful > 0:
        print(f"\n🎉 ХОТЯ БЫ {successful} АРХИТЕКТУР РАБОТАЮТ!")
        print(f"📊 Теперь можно сравнить результаты в models/ папке")
        print(f"🔍 Ищите файлы: tqdm-*_best для лучших моделей")
    
    if failed > 0:
        print(f"\n⚠️ {failed} архитектур не удались - проверьте ошибки выше")
    
    print(f"\n🏆 БОЛЬШОЕ СРАВНЕНИЕ АРХИТЕКТУР ЗАВЕРШЕНО!")
    print(f"🔥 EPIC NEURAL NETWORK BATTLE COMPLETE! 🔥")

if __name__ == "__main__":
    main() 