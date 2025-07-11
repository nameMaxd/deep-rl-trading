# 🏆 БОЛЬШОЕ СРАВНЕНИЕ АРХИТЕКТУР\n\n**Дата проведения:** 2025-07-11 14:37\n\n## 📊 Сравнительная таблица\n\n| Архитектура | Медиана $ | Consistency % | Лучшая $ | Стабильность | Время мин | Статус |\n|-------------|-----------|---------------|----------|--------------|-----------|--------|\n| v13_LSTM - Простая LSTM архитектура | N/A | N/A | N/A | N/A | 0 | ❌ |\n| v14_Transformer_Fixed - Фиксированный трансформер | N/A | N/A | N/A | N/A | 0 | ❌ |\n| v15_SSM - State Space Models (Mamba) | N/A | N/A | N/A | N/A | 0 | ❌ |\n| v16_MLP - Глубокие MLP с residuals | N/A | N/A | N/A | N/A | 0 | ❌ |\n\n## 🔍 Детальный анализ\n\n### 📈 Рейтинги по метрикам\n\n## 💡 Выводы и рекомендации\n\n❌ Все эксперименты завершились неудачно. Требуется отладка.\n\n## ❌ Неудачные эксперименты\n\n**v13_LSTM - Простая LSTM архитектура:**\n- Ошибка: Traceback (most recent call last):
  File "C:\Users\Yurchenko\cascadeweb\deep-rl-trading\main_v13_lstm.py", line 435, in <module>
    main() 
    ^^^^^^
  File "C:\Users\Yurchenko\cascadeweb\deep-rl-trading\main_v13_lstm.py", line 281, in main
    train_env = Env(
                ^^^^
TypeError: Env.__init__() got an unexpected keyword argument 'feature_extractor'
\n- Время до ошибки: 0 минут\n\n**v14_Transformer_Fixed - Фиксированный трансформер:**\n- Ошибка: Traceback (most recent call last):
  File "C:\Users\Yurchenko\cascadeweb\deep-rl-trading\main_v14_transformer_fixed.py", line 466, in <module>
    main() 
    ^^^^^^
  File "C:\Users\Yurchenko\cascadeweb\deep-rl-trading\main_v14_transformer_fixed.py", line 310, in main
    train_env = Env(
                ^^^^
TypeError: Env.__init__() got an unexpected keyword argument 'feature_extractor'
\n- Время до ошибки: 0 минут\n\n**v15_SSM - State Space Models (Mamba):**\n- Ошибка: Traceback (most recent call last):
  File "C:\Users\Yurchenko\cascadeweb\deep-rl-trading\main_v15_ssm.py", line 595, in <module>
    main() 
    ^^^^^^
  File "C:\Users\Yurchenko\cascadeweb\deep-rl-trading\main_v15_ssm.py", line 439, in main
    train_env = Env(
                ^^^^
TypeError: Env.__init__() got an unexpected keyword argument 'feature_extractor'
\n- Время до ошибки: 0 минут\n\n**v16_MLP - Глубокие MLP с residuals:**\n- Ошибка: Traceback (most recent call last):
  File "C:\Users\Yurchenko\cascadeweb\deep-rl-trading\main_v16_mlp.py", line 607, in <module>
    main() 
    ^^^^^^
  File "C:\Users\Yurchenko\cascadeweb\deep-rl-trading\main_v16_mlp.py", line 450, in main
    train_env = Env(
                ^^^^
TypeError: Env.__init__() got an unexpected keyword argument 'feature_extractor'
\n- Время до ошибки: 0 минут\n\n