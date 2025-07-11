import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


class TechnicalIndicators:
    """
    Расширенная библиотека технических индикаторов для трейдинга
    """
    
    @staticmethod
    def volume_profile(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                      volume: np.ndarray, num_levels: int = 20) -> Dict[str, np.ndarray]:
        """
        ГОРИЗОНТАЛЬНЫЕ ОБЪЕМЫ (Volume Profile) - ключевой индикатор
        Показывает на каких ценовых уровнях торговалось больше всего объема
        """
        # Диапазон цен
        price_min, price_max = np.min(low), np.max(high)
        price_step = (price_max - price_min) / num_levels
        
        # Создаем уровни цен
        price_levels = np.linspace(price_min, price_max, num_levels + 1)
        
        # Распределяем объем по уровням
        volume_at_price = np.zeros(num_levels)
        poc_values = []  # Point of Control для каждого периода
        
        for i in range(len(close)):
            # Определяем в какой уровень попадает цена закрытия
            level_idx = min(int((close[i] - price_min) / price_step), num_levels - 1)
            volume_at_price[level_idx] += volume[i]
            
            # POC (Point of Control) - уровень с максимальным объемом
            if i > 0:
                current_profile = volume_at_price.copy()
                poc_level = np.argmax(current_profile)
                poc_price = price_levels[poc_level] + price_step / 2
                poc_values.append(poc_price)
            else:
                poc_values.append(close[i])
        
        return {
            'volume_profile': volume_at_price,
            'price_levels': price_levels,
            'poc': np.array(poc_values),  # Point of Control
            'value_area_high': np.percentile(price_levels, 85),  # Верх зоны стоимости
            'value_area_low': np.percentile(price_levels, 15)    # Низ зоны стоимости
        }
    
    @staticmethod
    def order_flow_imbalance(high: np.ndarray, low: np.ndarray, 
                           close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        Дисбаланс ордеров - показывает преобладание покупок или продаж
        """
        # Упрощенный расчет: если цена растет - покупки, падает - продажи
        price_change = np.diff(close, prepend=close[0])
        buying_pressure = np.where(price_change > 0, volume, 0)
        selling_pressure = np.where(price_change < 0, volume, 0)
        
        # Кумулятивный дисбаланс
        net_flow = buying_pressure - selling_pressure
        cumulative_flow = np.cumsum(net_flow)
        
        return cumulative_flow
    
    @staticmethod
    def support_resistance_levels(high: np.ndarray, low: np.ndarray, 
                                close: np.ndarray, window: int = 20) -> Dict[str, np.ndarray]:
        """
        Динамические уровни поддержки и сопротивления
        """
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(close)):
            # Локальные максимумы и минимумы
            period_high = high[i-window:i+1]
            period_low = low[i-window:i+1]
            
            # Сопротивление - среднее из локальных максимумов
            local_peaks = []
            for j in range(1, len(period_high)-1):
                if period_high[j] > period_high[j-1] and period_high[j] > period_high[j+1]:
                    local_peaks.append(period_high[j])
            
            resistance = np.mean(local_peaks) if local_peaks else np.max(period_high)
            resistance_levels.append(resistance)
            
            # Поддержка - среднее из локальных минимумов
            local_valleys = []
            for j in range(1, len(period_low)-1):
                if period_low[j] < period_low[j-1] and period_low[j] < period_low[j+1]:
                    local_valleys.append(period_low[j])
            
            support = np.mean(local_valleys) if local_valleys else np.min(period_low)
            support_levels.append(support)
        
        # Заполняем начальные значения
        resistance_levels = [resistance_levels[0]] * window + resistance_levels
        support_levels = [support_levels[0]] * window + support_levels
        
        return {
            'resistance': np.array(resistance_levels),
            'support': np.array(support_levels)
        }
    
    @staticmethod
    def market_structure(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Анализ рыночной структуры: тренды, пробои, консолидация
        """
        # Высокие максимумы и высокие минимумы (восходящий тренд)
        hh = np.zeros(len(close))  # Higher Highs
        hl = np.zeros(len(close))  # Higher Lows
        
        # Низкие максимумы и низкие минимумы (нисходящий тренд)
        lh = np.zeros(len(close))  # Lower Highs  
        ll = np.zeros(len(close))  # Lower Lows
        
        for i in range(2, len(close)):
            # Проверяем тренды
            if high[i] > high[i-1] and high[i-1] > high[i-2]:
                hh[i] = 1
            if low[i] > low[i-1] and low[i-1] > low[i-2]:
                hl[i] = 1
            if high[i] < high[i-1] and high[i-1] < high[i-2]:
                lh[i] = 1
            if low[i] < low[i-1] and low[i-1] < low[i-2]:
                ll[i] = 1
        
        # Общий индикатор тренда
        trend_strength = (hh + hl) - (lh + ll)
        
        return {
            'higher_highs': hh,
            'higher_lows': hl,
            'lower_highs': lh,
            'lower_lows': ll,
            'trend_strength': trend_strength
        }
    
    @staticmethod
    def liquidity_levels(high: np.ndarray, low: np.ndarray, volume: np.ndarray, 
                        percentile: float = 95) -> Dict[str, np.ndarray]:
        """
        Уровни ликвидности - где концентрируются крупные объемы
        """
        # Высокообъемные уровни
        volume_threshold = np.percentile(volume, percentile)
        high_volume_mask = volume >= volume_threshold
        
        liquidity_zones = np.zeros(len(high))
        liquidity_zones[high_volume_mask] = 1
        
        # Кластеры ликвидности
        liquidity_clusters = []
        cluster_strength = []
        
        for i in range(len(high)):
            if liquidity_zones[i] == 1:
                # Находим ближайшие уровни ликвидности
                cluster_price = (high[i] + low[i]) / 2
                cluster_vol = volume[i]
                
                liquidity_clusters.append(cluster_price)
                cluster_strength.append(cluster_vol)
            else:
                liquidity_clusters.append(0)
                cluster_strength.append(0)
        
        return {
            'liquidity_zones': liquidity_zones,
            'liquidity_clusters': np.array(liquidity_clusters),
            'cluster_strength': np.array(cluster_strength)
        }
    
    @staticmethod
    def advanced_rsi(close: np.ndarray, period: int = 14) -> Dict[str, np.ndarray]:
        """
        Расширенный RSI с дополнительными сигналами
        """
        # Стандартный RSI
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Экспоненциальные скользящие средние
        alpha = 1.0 / period
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        
        avg_gain[0] = np.mean(gain[:period])
        avg_loss[0] = np.mean(loss[:period])
        
        for i in range(1, len(gain)):
            avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
            avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Дополнительные сигналы
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
        rsi_divergence = np.zeros_like(rsi)
        
        # Простой расчет дивергенции
        for i in range(period, len(rsi)):
            price_trend = close[i] - close[i-period]
            rsi_trend = rsi[i] - rsi[i-period]
            
            # Бычья дивергенция: цена падает, RSI растет
            if price_trend < 0 and rsi_trend > 0:
                rsi_divergence[i] = 1
            # Медвежья дивергенция: цена растет, RSI падает
            elif price_trend > 0 and rsi_trend < 0:
                rsi_divergence[i] = -1
        
        return {
            'rsi': rsi,
            'oversold': rsi_oversold.astype(int),
            'overbought': rsi_overbought.astype(int),
            'divergence': rsi_divergence
        }
    
    @staticmethod
    def volume_weighted_average_price(high: np.ndarray, low: np.ndarray, 
                                    close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        VWAP - Объемно-взвешенная средняя цена
        """
        typical_price = (high + low + close) / 3
        cumulative_pv = np.cumsum(typical_price * volume)
        cumulative_volume = np.cumsum(volume)
        
        vwap = cumulative_pv / cumulative_volume
        return vwap
    
    @staticmethod
    def bollinger_bands_advanced(close: np.ndarray, period: int = 20, 
                                std_dev: float = 2) -> Dict[str, np.ndarray]:
        """
        Расширенные полосы Боллинджера с дополнительными сигналами
        """
        sma = np.convolve(close, np.ones(period)/period, mode='valid')
        sma = np.concatenate([np.full(period-1, sma[0]), sma])
        
        # Скользящее стандартное отклонение
        rolling_std = np.zeros_like(close)
        for i in range(period-1, len(close)):
            rolling_std[i] = np.std(close[i-period+1:i+1])
        
        # Заполняем начальные значения
        rolling_std[:period-1] = rolling_std[period-1]
        
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        
        # Дополнительные сигналы
        bb_width = (upper_band - lower_band) / sma  # Ширина полос
        bb_position = (close - lower_band) / (upper_band - lower_band)  # Позиция цены
        
        # Сжатие полос (низкая волатильность)
        bb_squeeze = bb_width < np.percentile(bb_width, 20)
        
        return {
            'upper_band': upper_band,
            'middle_band': sma,
            'lower_band': lower_band,
            'bb_width': bb_width,
            'bb_position': bb_position,
            'bb_squeeze': bb_squeeze.astype(int)
        }
    
    @staticmethod
    def ichimoku_cloud(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      tenkan_period: int = 9, kijun_period: int = 26, 
                      senkou_span_b_period: int = 52) -> Dict[str, np.ndarray]:
        """
        Облако Ишимоку - комплексный индикатор тренда
        """
        # Tenkan-sen (Conversion Line)
        tenkan_sen = np.zeros_like(close)
        for i in range(tenkan_period-1, len(close)):
            period_high = np.max(high[i-tenkan_period+1:i+1])
            period_low = np.min(low[i-tenkan_period+1:i+1])
            tenkan_sen[i] = (period_high + period_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = np.zeros_like(close)
        for i in range(kijun_period-1, len(close)):
            period_high = np.max(high[i-kijun_period+1:i+1])
            period_low = np.min(low[i-kijun_period+1:i+1])
            kijun_sen[i] = (period_high + period_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = np.zeros_like(close)
        for i in range(senkou_span_b_period-1, len(close)):
            period_high = np.max(high[i-senkou_span_b_period+1:i+1])
            period_low = np.min(low[i-senkou_span_b_period+1:i+1])
            senkou_span_b[i] = (period_high + period_low) / 2
        
        # Chikou Span (Lagging Span) - смещенная на 26 периодов назад цена закрытия
        chikou_span = np.roll(close, -kijun_period)
        
        # Заполняем начальные значения
        tenkan_sen[:tenkan_period-1] = tenkan_sen[tenkan_period-1]
        kijun_sen[:kijun_period-1] = kijun_sen[kijun_period-1]
        senkou_span_b[:senkou_span_b_period-1] = senkou_span_b[senkou_span_b_period-1]
        
        # Облако (Kumo)
        cloud_top = np.maximum(senkou_span_a, senkou_span_b)
        cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
        
        # Сигналы
        above_cloud = close > cloud_top
        below_cloud = close < cloud_bottom
        in_cloud = ~above_cloud & ~below_cloud
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span,
            'cloud_top': cloud_top,
            'cloud_bottom': cloud_bottom,
            'above_cloud': above_cloud.astype(int),
            'below_cloud': below_cloud.astype(int),
            'in_cloud': in_cloud.astype(int)
        } 