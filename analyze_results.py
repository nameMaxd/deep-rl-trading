import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def analyze_training_log(log_path):
    """Анализ логов обучения с расширенными метриками для v5"""
    try:
        df = pd.read_csv(f"models/{log_path}.log")
        
        # Создание графиков
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'Training Analysis v5: {log_path}', fontsize=16)
        
        # 1. Profit в долларах
        axes[0, 0].plot(df['episode'], df['profit_dollars'], label='Training', alpha=0.7, color='blue')
        if 'oos_profit_dollars' in df.columns:
            axes[0, 0].plot(df['episode'], df['oos_profit_dollars'], label='OOS', alpha=0.7, color='red')
        axes[0, 0].set_title('Profit ($)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Profit ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Sharpe Ratio
        axes[0, 1].plot(df['episode'], df['sharpe_ratio'], label='Training', alpha=0.7, color='blue')
        if 'oos_sharpe_ratio' in df.columns:
            axes[0, 1].plot(df['episode'], df['oos_sharpe_ratio'], label='OOS', alpha=0.7, color='red')
        axes[0, 1].set_title('Sharpe Ratio')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Количество сделок
        axes[0, 2].plot(df['episode'], df['num_trades'], label='Training', alpha=0.7, color='blue')
        if 'oos_num_trades' in df.columns:
            axes[0, 2].plot(df['episode'], df['oos_num_trades'], label='OOS', alpha=0.7, color='red')
        axes[0, 2].set_title('Number of Trades')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Trades')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. Win Rate
        axes[1, 0].plot(df['episode'], df['win_rate'] * 100, label='Training', alpha=0.7, color='blue')
        if 'oos_win_rate' in df.columns:
            axes[1, 0].plot(df['episode'], df['oos_win_rate'] * 100, label='OOS', alpha=0.7, color='red')
        axes[1, 0].set_title('Win Rate (%)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Win Rate (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 5. Loss и Epsilon
        axes[1, 1].plot(df['episode'], df['loss'], alpha=0.7, label='Loss', color='purple')
        ax_twin = axes[1, 1].twinx()
        ax_twin.plot(df['episode'], df['epsilon'], alpha=0.7, label='Epsilon', color='orange')
        if 'adaptive_epsilon' in df.columns:
            ax_twin.plot(df['episode'], df['adaptive_epsilon'], alpha=0.7, label='Adaptive ε', color='green')
        axes[1, 1].set_title('Loss & Epsilon')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        ax_twin.set_ylabel('Epsilon')
        axes[1, 1].legend(loc='upper left')
        ax_twin.legend(loc='upper right')
        axes[1, 1].grid(True)
        
        # 6. Exploration Balance (если есть)
        if 'exploration_balance' in df.columns:
            axes[1, 2].plot(df['episode'], df['exploration_balance'], alpha=0.7, color='green')
            axes[1, 2].set_title('Exploration Balance')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Balance (min/max actions)')
            axes[1, 2].grid(True)
        else:
            axes[1, 2].text(0.5, 0.5, 'No exploration data', ha='center', va='center', transform=axes[1, 2].transAxes)
        
        # 7. Action Distribution (если есть)
        if all(col in df.columns for col in ['hold_pct', 'buy_pct', 'sell_pct']):
            axes[2, 0].plot(df['episode'], df['hold_pct'] * 100, label='Hold', alpha=0.7)
            axes[2, 0].plot(df['episode'], df['buy_pct'] * 100, label='Buy', alpha=0.7)
            axes[2, 0].plot(df['episode'], df['sell_pct'] * 100, label='Sell', alpha=0.7)
            axes[2, 0].set_title('Action Distribution (%)')
            axes[2, 0].set_xlabel('Episode')
            axes[2, 0].set_ylabel('Percentage')
            axes[2, 0].legend()
            axes[2, 0].grid(True)
        else:
            axes[2, 0].text(0.5, 0.5, 'No action data', ha='center', va='center', transform=axes[2, 0].transAxes)
        
        # 8. Cumulative Performance
        cumulative_train = np.cumsum(df['profit_dollars'])
        axes[2, 1].plot(df['episode'], cumulative_train, label='Cumulative Training', alpha=0.7, color='blue')
        if 'oos_profit_dollars' in df.columns:
            # Скользящее среднее для OOS
            oos_smooth = df['oos_profit_dollars'].rolling(window=10, min_periods=1).mean()
            axes[2, 1].plot(df['episode'], oos_smooth, label='OOS (smoothed)', alpha=0.7, color='red')
        axes[2, 1].set_title('Cumulative Performance')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Cumulative Profit ($)')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        # 9. Performance Correlation
        if 'oos_profit_dollars' in df.columns:
            # Скользящая корреляция
            window = 20
            rolling_corr = []
            for i in range(window, len(df)):
                train_window = df['profit_dollars'].iloc[i-window:i]
                oos_window = df['oos_profit_dollars'].iloc[i-window:i]
                corr = np.corrcoef(train_window, oos_window)[0, 1]
                rolling_corr.append(corr if not np.isnan(corr) else 0)
            
            axes[2, 2].plot(df['episode'].iloc[window:], rolling_corr, alpha=0.7, color='purple')
            axes[2, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[2, 2].set_title('Rolling Correlation (Train vs OOS)')
            axes[2, 2].set_xlabel('Episode')
            axes[2, 2].set_ylabel('Correlation')
            axes[2, 2].grid(True)
        else:
            axes[2, 2].text(0.5, 0.5, 'No OOS data', ha='center', va='center', transform=axes[2, 2].transAxes)
        
        plt.tight_layout()
        plt.savefig(f'models/{log_path}_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Закрываем график вместо показа
        print(f"📊 Улучшенный график v5 сохранен: models/{log_path}_analysis.png")
        
        # Печать статистики v5
        print(f"\n📊 АНАЛИЗ РЕЗУЛЬТАТОВ v5: {log_path}")
        print("=" * 60)
        
        print("\n🎯 ОБУЧЕНИЕ:")
        print(f"  Финальный Profit: ${df['profit_dollars'].iloc[-1]:.0f}")
        print(f"  Максимальный Profit: ${df['profit_dollars'].max():.0f}")
        print(f"  Минимальный Profit: ${df['profit_dollars'].min():.0f}")
        print(f"  Средний Sharpe: {df['sharpe_ratio'].mean():.3f}")
        print(f"  Финальный Sharpe: {df['sharpe_ratio'].iloc[-1]:.3f}")
        print(f"  Среднее кол-во сделок: {df['num_trades'].mean():.1f}")
        print(f"  Средний Win Rate: {df['win_rate'].mean()*100:.1f}%")
        print(f"  Средний Loss: {df['loss'].mean():.4f}")
        print(f"  Финальный Loss: {df['loss'].iloc[-1]:.4f}")
        print(f"  Финальный Epsilon: {df['epsilon'].iloc[-1]:.3f}")
        
        if 'oos_profit_dollars' in df.columns:
            print("\n🧪 OUT-OF-SAMPLE:")
            print(f"  Финальный Profit: ${df['oos_profit_dollars'].iloc[-1]:.0f}")
            print(f"  Максимальный Profit: ${df['oos_profit_dollars'].max():.0f}")
            print(f"  Минимальный Profit: ${df['oos_profit_dollars'].min():.0f}")
            print(f"  Средний Sharpe: {df['oos_sharpe_ratio'].mean():.3f}")
            print(f"  Финальный Sharpe: {df['oos_sharpe_ratio'].iloc[-1]:.3f}")
            print(f"  Среднее кол-во сделок: {df['oos_num_trades'].mean():.1f}")
            print(f"  Средний Win Rate: {df['oos_win_rate'].mean()*100:.1f}%")
        
        # Корреляция между train и OOS
        if 'oos_profit_dollars' in df.columns:
            corr_profit = np.corrcoef(df['profit_dollars'], df['oos_profit_dollars'])[0, 1]
            corr_sharpe = np.corrcoef(df['sharpe_ratio'], df['oos_sharpe_ratio'])[0, 1]
            print(f"\n📈 КОРРЕЛЯЦИЯ TRAIN vs OOS:")
            print(f"  Profit correlation: {corr_profit:.3f}")
            print(f"  Sharpe correlation: {corr_sharpe:.3f}")
        
        # Анализ exploration
        if all(col in df.columns for col in ['hold_pct', 'buy_pct', 'sell_pct']):
            print(f"\n🎲 EXPLORATION АНАЛИЗ:")
            print(f"  Финальное распределение действий:")
            print(f"    Hold: {df['hold_pct'].iloc[-1]*100:.1f}%")
            print(f"    Buy: {df['buy_pct'].iloc[-1]*100:.1f}%")
            print(f"    Sell: {df['sell_pct'].iloc[-1]*100:.1f}%")
            if 'exploration_balance' in df.columns:
                print(f"  Средний exploration balance: {df['exploration_balance'].mean():.3f}")
        
        # Анализ версий v5
        print(f"\n🚀 АНАЛИЗ УЛУЧШЕНИЙ v5:")
        print(f"  🎯 Упрощенные индикаторы: 11 вместо 23")
        print(f"  🔄 Reward engineering: Включен")
        print(f"  🎲 Адаптивный epsilon: Включен")
        print(f"  🤔 Curiosity exploration: Включен")
        print(f"  ⚡ Частое обучение: Каждые 10 шагов")
        print(f"  📈 Больше торговых периодов: 50 дней")
        
        # Сравнение с предыдущими версиями
        print(f"\n🔄 СРАВНЕНИЕ ВСЕХ ВЕРСИЙ:")
        print(f"  v1 (базовая): Training $1144, OOS $-342, Corr: 0.15")
        print(f"  v2 (улучшенная): Training $198, OOS $-195, Corr: -0.01")
        print(f"  v3 (финальная): Training $-195, OOS $-195, Corr: -0.01")
        print(f"  v4 (23 индикатора): Training $-2212, OOS $0, Corr: 0.05")
        print(f"  v5 (умная): Training ${df['profit_dollars'].iloc[-1]:.0f}, OOS ${df['oos_profit_dollars'].iloc[-1]:.0f}, Corr: {corr_profit:.2f}")
        
        # Выводы и оценка
        print(f"\n💡 КЛЮЧЕВЫЕ ВЫВОДЫ v5:")
        
        oos_improvement = df['oos_profit_dollars'].iloc[-1] > 0
        low_loss = df['loss'].iloc[-1] < 0.02
        good_trading_activity = df['num_trades'].mean() > 5
        
        if oos_improvement:
            print("  ✅ УСПЕХ: OOS положительный - модель обобщает!")
        
        if low_loss:
            print("  ✅ УСПЕХ: Низкий финальный loss - стабильное обучение!")
        
        if good_trading_activity:
            print("  ✅ УСПЕХ: Хорошая торговая активность!")
        
        # Проблемы
        if df['profit_dollars'].iloc[-1] < -1000:
            print("  ⚠️ ПРОБЛЕМА: Большие потери на обучении")
        
        if abs(corr_profit) < 0.1:
            print("  ⚠️ ПРОБЛЕМА: Слабая корреляция train-OOS")
        
        # Динамика обучения
        profit_trend = np.polyfit(range(len(df)), df['profit_dollars'], 1)[0]
        if profit_trend > 0:
            print("  📈 ПОЗИТИВНО: Восходящий тренд прибыли при обучении")
        else:
            print("  📉 ВНИМАНИЕ: Нисходящий тренд прибыли при обучении")
        
        # Финальная оценка
        score = 0
        if oos_improvement: score += 3
        if low_loss: score += 2
        if good_trading_activity: score += 2
        if abs(corr_profit) > 0.1: score += 1
        if profit_trend > 0: score += 1
        
        print(f"\n🏆 ИТОГОВАЯ ОЦЕНКА v5: {score}/9")
        if score >= 7:
            print("  🌟 ОТЛИЧНО: Модель показывает отличные результаты!")
        elif score >= 5:
            print("  ✅ ХОРОШО: Модель показывает хорошие результаты!")
        elif score >= 3:
            print("  ⚡ УДОВЛЕТВОРИТЕЛЬНО: Есть улучшения, но нужна доработка")
        else:
            print("  ❌ ПЛОХО: Модель требует серьезных изменений")
        
        return df
        
    except Exception as e:
        print(f"Ошибка анализа: {str(e)}")
        return None


if __name__ == "__main__":
    # Анализ результатов v5
    analyze_training_log("google-trading-v5-smart") 