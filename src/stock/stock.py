from src.stock.data_center import load_stock, load_stock_from_csv
from src.stock.indicators import TechnicalIndicators
import numpy as np
import pandas as pd


class Stock:
    """
    Stock object that carries price and associated info, observation space for NN,
    along with methods for calculating various indicators
    """
    def __init__(self, ticker=None, period="6000d", timeframe="1d", window=364, csv_path=None):
        self.ticker = ticker
        self.timeframe = timeframe
        self.period = period
        self.window = window
        self.csv_path = csv_path

        # Loading in price data
        if csv_path:
            df = load_stock_from_csv(csv_path)
            self.ticker = csv_path.split('/')[-1].split('.')[0]  # Extract ticker from filename
        else:
            df = self.import_data()
            
        if df is None:
            raise ValueError(f"Failed to load data for {ticker or csv_path}")
            
        self.opens = df["Open"].to_numpy()
        self.highs = df["High"].to_numpy()
        self.lows = df["Low"].to_numpy()
        self.closes = df["Close"].to_numpy()
        self.volume = df["Volume"].to_numpy()
        self.dates = df.index

        # Calculate technical indicators
        self.indicators = self._calculate_indicators()

        # Observation space for neural network
        self.obs_space = None
        self.create_observation_space()

    def _calculate_indicators(self) -> dict:
        """🔥 МЕГА-ВЕРСИЯ: Максимум некоррелированных фичей для СЕРЬЁЗНЫХ денег!"""
        indicators = {}
        
        # Базовые статистики для нормализации
        price_mean = np.mean(self.closes)
        price_std = np.std(self.closes)
        volume_mean = np.mean(self.volume)
        volume_std = np.std(self.volume)
        
        print("💰 Создаю МЕГА-НАБОР фичей для больших денег...")
        
        # ========== ГРУППА 1: БАЗОВЫЕ ЦЕНЫ (5 фичей) ==========
        indicators['open_norm'] = (self.opens - price_mean) / price_std
        indicators['high_norm'] = (self.highs - price_mean) / price_std
        indicators['low_norm'] = (self.lows - price_mean) / price_std
        indicators['close_norm'] = (self.closes - price_mean) / price_std
        indicators['volume_norm'] = (self.volume - volume_mean) / volume_std
        
        # ========== ГРУППА 2: ОБЪЕМНЫЙ АНАЛИЗ (4 фичи) ==========
        # Volume Profile - КЛЮЧЕВОЙ индикатор!
        volume_profile = TechnicalIndicators.volume_profile(
            self.highs, self.lows, self.closes, self.volume, num_levels=20
        )
        indicators['volume_poc'] = (volume_profile['poc'] - price_mean) / price_std
        
        # VWAP
        vwap = TechnicalIndicators.volume_weighted_average_price(
            self.highs, self.lows, self.closes, self.volume
        )
        indicators['vwap'] = (vwap - price_mean) / price_std
        
        # Volume Rate of Change
        volume_roc = np.diff(self.volume, prepend=self.volume[0]) / (self.volume + 1e-8)
        indicators['volume_roc'] = np.clip(volume_roc, -3, 3)
        
        # Accumulation/Distribution Line
        ad_line = self._calculate_ad_line()
        indicators['ad_line'] = (ad_line - np.mean(ad_line)) / (np.std(ad_line) + 1e-8)
        
        # ========== ГРУППА 3: ТРЕНДОВЫЕ ИНДИКАТОРЫ (6 фичей) ==========
        # Multiple SMAs
        sma_5 = self.get_sma(self.closes, 5)
        sma_10 = self.get_sma(self.closes, 10)
        sma_20 = self.get_sma(self.closes, 20)
        sma_50 = self.get_sma(self.closes, 50)
        indicators['sma_5'] = (np.array(sma_5) - price_mean) / price_std
        indicators['sma_20'] = (np.array(sma_20) - price_mean) / price_std
        
        # EMAs (некоррелированные с SMA)
        ema_12 = self.get_ema(self.closes, 12)
        ema_26 = self.get_ema(self.closes, 26)
        indicators['ema_12'] = (np.array(ema_12) - price_mean) / price_std
        indicators['ema_26'] = (np.array(ema_26) - price_mean) / price_std
        
        # MACD
        macd_line = np.array(ema_12) - np.array(ema_26)
        macd_signal = self.get_ema(macd_line, 9)
        indicators['macd'] = macd_line / price_std
        indicators['macd_signal'] = np.array(macd_signal) / price_std
        
        # ========== ГРУППА 4: MOMENTUM И OSCILLATORS (5 фичей) ==========
        # RSI Advanced
        rsi_data = TechnicalIndicators.advanced_rsi(self.closes, period=14)
        indicators['rsi'] = (rsi_data['rsi'] - 50) / 50
        
        # Stochastic %K and %D
        stoch_k, stoch_d = self._calculate_stochastic()
        indicators['stoch_k'] = (stoch_k - 50) / 50
        indicators['stoch_d'] = (stoch_d - 50) / 50
        
        # Williams %R
        williams_r = self._calculate_williams_r()
        indicators['williams_r'] = williams_r / 100
        
        # Rate of Change
        roc = self._calculate_roc(period=10)
        indicators['roc'] = roc / 100
        
        # ========== ГРУППА 5: ВОЛАТИЛЬНОСТЬ (4 фичи) ==========
        # True Range
        true_range = self._calculate_true_range()
        atr = self._calculate_atr(true_range, period=14)
        indicators['atr'] = atr / price_std
        
        # Bollinger Bands
        bb_data = self.get_bollinger_bands(data=self.closes, term="short")
        bb_upper = np.array([x[0] for x in bb_data])
        bb_lower = np.array([x[1] for x in bb_data])
        bb_width = (bb_upper - bb_lower) / price_std
        bb_position = (self.closes[:len(bb_width)] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        indicators['bb_width'] = bb_width
        indicators['bb_position'] = np.clip(bb_position, 0, 1)
        
        # Historical Volatility (разные периоды)
        vol_short = self._calculate_rolling_volatility(period=10)
        vol_long = self._calculate_rolling_volatility(period=30)
        indicators['volatility_short'] = vol_short / price_std
        
        # ========== ГРУППА 6: ЦЕНОВЫЕ ПАТТЕРНЫ (4 фичи) ==========
        # Typical Price
        typical_price = (self.highs + self.lows + self.closes) / 3
        indicators['typical_price'] = (typical_price - price_mean) / price_std
        
        # Price gaps
        price_gaps = self._calculate_price_gaps()
        indicators['price_gaps'] = price_gaps / price_std
        
        # High-Low spread
        hl_spread = (self.highs - self.lows) / self.closes
        indicators['hl_spread'] = np.clip(hl_spread, 0, 0.2) * 10  # Normalize to 0-2
        
        # Open-Close difference
        oc_diff = (self.closes - self.opens) / self.opens
        indicators['oc_ratio'] = np.clip(oc_diff, -0.1, 0.1) * 10  # Normalize
        
        # ========== ГРУППА 7: ADVANCED ИНДИКАТОРЫ (5 фичей) ==========
        # CCI (Commodity Channel Index)
        cci = self.get_cci(period=20)
        indicators['cci'] = np.clip(np.array(cci), -300, 300) / 300
        
        # Money Flow Index
        mfi = self._calculate_mfi()
        indicators['mfi'] = (mfi - 50) / 50
        
        # Chaikin Oscillator
        chaikin_osc = self._calculate_chaikin_oscillator()
        indicators['chaikin_osc'] = chaikin_osc / (np.std(chaikin_osc) + 1e-8)
        
        # Directional Movement Index
        dmi_plus, dmi_minus, adx = self._calculate_dmi()
        indicators['dmi_plus'] = dmi_plus / 100
        indicators['adx'] = adx / 100
        
        # ========== ГРУППА 8: TIME-BASED FEATURES (3 фичи) ==========
        # Day of week effect
        if hasattr(self, 'dates') and len(self.dates) > 0:
            weekdays = [d.weekday() for d in self.dates]
            # Extend to match price data length
            while len(weekdays) < len(self.closes):
                weekdays.append(weekdays[-1] if weekdays else 0)
            weekdays = np.array(weekdays[:len(self.closes)])
            indicators['weekday'] = (weekdays - 2) / 2  # Center around Wednesday
        else:
            indicators['weekday'] = np.zeros(len(self.closes))
        
        # Month effect
        if hasattr(self, 'dates') and len(self.dates) > 0:
            months = [d.month for d in self.dates]
            while len(months) < len(self.closes):
                months.append(months[-1] if months else 6)
            months = np.array(months[:len(self.closes)])
            indicators['month'] = (months - 6.5) / 6.5  # Center around July
        else:
            indicators['month'] = np.zeros(len(self.closes))
        
        # Position in trading session (assuming daily data)
        session_position = np.arange(len(self.closes)) % 252 / 252  # Yearly cycle
        indicators['session_pos'] = (session_position - 0.5) * 2  # -1 to 1
        
        print(f"💰 Создано {len(indicators)} МЕГА-фичей для анализа!")
        print("📊 Группы фичей:")
        print("   1. Базовые цены: 5 фичей")
        print("   2. Объемный анализ: 4 фичи") 
        print("   3. Трендовые: 6 фичей")
        print("   4. Momentum: 5 фичей")
        print("   5. Волатильность: 4 фичи")
        print("   6. Ценовые паттерны: 4 фичи")
        print("   7. Advanced: 5 фичей")
        print("   8. Time-based: 3 фичи")
        
        return indicators

    def _calculate_ad_line(self):
        """Accumulation/Distribution Line"""
        ad_line = []
        ad_value = 0
        
        for i in range(len(self.closes)):
            if self.highs[i] != self.lows[i]:
                clv = ((self.closes[i] - self.lows[i]) - (self.highs[i] - self.closes[i])) / (self.highs[i] - self.lows[i])
                ad_value += clv * self.volume[i]
            ad_line.append(ad_value)
            
        return np.array(ad_line)

    def _calculate_stochastic(self, k_period=14, d_period=3):
        """Stochastic Oscillator"""
        stoch_k = []
        stoch_d = []
        
        for i in range(len(self.closes)):
            start_idx = max(0, i - k_period + 1)
            lowest_low = np.min(self.lows[start_idx:i+1])
            highest_high = np.max(self.highs[start_idx:i+1])
            
            if highest_high != lowest_low:
                k_value = ((self.closes[i] - lowest_low) / (highest_high - lowest_low)) * 100
            else:
                k_value = 50
                
            stoch_k.append(k_value)
            
            # %D is SMA of %K
            if len(stoch_k) >= d_period:
                d_value = np.mean(stoch_k[-d_period:])
            else:
                d_value = np.mean(stoch_k)
            stoch_d.append(d_value)
            
        return np.array(stoch_k), np.array(stoch_d)

    def _calculate_williams_r(self, period=14):
        """Williams %R"""
        williams_r = []
        
        for i in range(len(self.closes)):
            start_idx = max(0, i - period + 1)
            highest_high = np.max(self.highs[start_idx:i+1])
            lowest_low = np.min(self.lows[start_idx:i+1])
            
            if highest_high != lowest_low:
                wr = ((highest_high - self.closes[i]) / (highest_high - lowest_low)) * -100
            else:
                wr = -50
                
            williams_r.append(wr)
            
        return np.array(williams_r)

    def _calculate_roc(self, period=10):
        """Rate of Change"""
        roc = []
        
        for i in range(len(self.closes)):
            if i >= period and self.closes[i-period] != 0:
                roc_value = ((self.closes[i] - self.closes[i-period]) / self.closes[i-period]) * 100
            else:
                roc_value = 0
            roc.append(roc_value)
            
        return np.array(roc)

    def _calculate_true_range(self):
        """True Range calculation"""
        true_range = []
        
        for i in range(len(self.closes)):
            if i == 0:
                tr = self.highs[i] - self.lows[i]
            else:
                tr = max(
                    self.highs[i] - self.lows[i],
                    abs(self.highs[i] - self.closes[i-1]),
                    abs(self.lows[i] - self.closes[i-1])
                )
            true_range.append(tr)
            
        return np.array(true_range)

    def _calculate_atr(self, true_range, period=14):
        """Average True Range"""
        atr = []
        
        for i in range(len(true_range)):
            start_idx = max(0, i - period + 1)
            atr_value = np.mean(true_range[start_idx:i+1])
            atr.append(atr_value)
            
        return np.array(atr)

    def _calculate_rolling_volatility(self, period=20):
        """Rolling volatility calculation"""
        volatility = []
        
        for i in range(len(self.closes)):
            start_idx = max(0, i - period + 1)
            returns = np.diff(np.log(self.closes[start_idx:i+1])) if i > start_idx else [0]
            vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
            volatility.append(vol)
            
        return np.array(volatility)

    def _calculate_price_gaps(self):
        """Price gaps between sessions"""
        gaps = []
        
        for i in range(len(self.opens)):
            if i == 0:
                gap = 0
            else:
                gap = self.opens[i] - self.closes[i-1]
            gaps.append(gap)
            
        return np.array(gaps)

    def _calculate_mfi(self, period=14):
        """Money Flow Index"""
        typical_prices = (self.highs + self.lows + self.closes) / 3
        money_flow = typical_prices * self.volume
        
        mfi = []
        
        for i in range(len(self.closes)):
            if i == 0:
                mfi.append(50)
                continue
                
            start_idx = max(0, i - period + 1)
            
            positive_flow = 0
            negative_flow = 0
            
            for j in range(start_idx, i):
                if typical_prices[j+1] > typical_prices[j]:
                    positive_flow += money_flow[j+1]
                elif typical_prices[j+1] < typical_prices[j]:
                    negative_flow += money_flow[j+1]
            
            if negative_flow == 0:
                mfi_value = 100
            elif positive_flow == 0:
                mfi_value = 0
            else:
                mfi_ratio = positive_flow / negative_flow
                mfi_value = 100 - (100 / (1 + mfi_ratio))
                
            mfi.append(mfi_value)
            
        return np.array(mfi)

    def _calculate_chaikin_oscillator(self):
        """Chaikin Oscillator"""
        ad_line = self._calculate_ad_line()
        ema_3 = self.get_ema(ad_line, 3)
        ema_10 = self.get_ema(ad_line, 10)
        
        chaikin_osc = np.array(ema_3) - np.array(ema_10)
        return chaikin_osc

    def _calculate_dmi(self, period=14):
        """Directional Movement Index"""
        plus_dm = []
        minus_dm = []
        
        for i in range(len(self.closes)):
            if i == 0:
                plus_dm.append(0)
                minus_dm.append(0)
                continue
                
            high_diff = self.highs[i] - self.highs[i-1]
            low_diff = self.lows[i-1] - self.lows[i]
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm.append(high_diff)
            else:
                plus_dm.append(0)
                
            if low_diff > high_diff and low_diff > 0:
                minus_dm.append(low_diff)
            else:
                minus_dm.append(0)
        
        plus_dm = np.array(plus_dm)
        minus_dm = np.array(minus_dm)
        true_range = self._calculate_true_range()
        
        # Calculate smoothed values
        plus_di = []
        minus_di = []
        adx = []
        
        for i in range(len(self.closes)):
            start_idx = max(0, i - period + 1)
            
            smoothed_plus_dm = np.mean(plus_dm[start_idx:i+1])
            smoothed_minus_dm = np.mean(minus_dm[start_idx:i+1])
            smoothed_tr = np.mean(true_range[start_idx:i+1])
            
            if smoothed_tr > 0:
                plus_di_val = (smoothed_plus_dm / smoothed_tr) * 100
                minus_di_val = (smoothed_minus_dm / smoothed_tr) * 100
            else:
                plus_di_val = 0
                minus_di_val = 0
                
            plus_di.append(plus_di_val)
            minus_di.append(minus_di_val)
            
            # ADX calculation
            if plus_di_val + minus_di_val > 0:
                dx = abs(plus_di_val - minus_di_val) / (plus_di_val + minus_di_val) * 100
            else:
                dx = 0
                
            if len(adx) == 0:
                adx.append(dx)
            else:
                # Smoothed ADX
                adx_val = (adx[-1] * (period - 1) + dx) / period
                adx.append(adx_val)
        
        return np.array(plus_di), np.array(minus_di), np.array(adx)

    def import_data(self, ticker=None, period=None, timeframe=None):
        """Import data from yfinance with Stock params if none are specified"""
        if self.csv_path:
            return load_stock_from_csv(self.csv_path)
        
        ticker = self.ticker if not ticker else ticker
        period = self.period if not period else period
        timeframe = self.timeframe if not timeframe else timeframe

        return load_stock(ticker, period, timeframe)

    def create_observation_space(self):
        """🔥 МЕГА observation space с МАКСИМУМОМ фичей"""
        if self.csv_path:
            daily = self.import_data()
        else:
            daily = self.import_data(timeframe="1d")
        
        obs_space = []
        
        # ВСЕ ДОСТУПНЫЕ ФИЧИ! (должно быть ~36 фичей)
        feature_keys = list(self.indicators.keys())
        
        # Убираем высококоррелированные фичи
        excluded_features = ['sma_10', 'sma_50', 'volatility_long']  # Коррелируют с другими
        feature_keys = [k for k in feature_keys if k not in excluded_features]
        
        print(f"💰 МЕГА-набор: {len(feature_keys)} НЕКОРРЕЛИРОВАННЫХ фичей:")
        for i, key in enumerate(feature_keys, 1):
            print(f"   {i:2d}. {key}")

        print(f"📊 Observation space: ({len(self.closes) - self.window}, {len(feature_keys)}, {self.window})")

        for i in range(self.window, len(self.closes)):
            features = []
            
            for key in feature_keys:
                if key in self.indicators:
                    window_data = self.indicators[key][max(0, i-self.window+1):i+1]
                    
                    if len(window_data) < self.window:
                        pad_size = self.window - len(window_data)
                        window_data = np.concatenate([
                            np.full(pad_size, window_data[0] if len(window_data) > 0 else 0),
                            window_data
                        ])
                    
                    # Clip extreme values для стабильности
                    window_data = np.clip(window_data, -10, 10)
                    features.append(window_data)
                else:
                    features.append(np.zeros(self.window))
            
            if features:
                combined_features = np.array(features)
                obs_space.append(combined_features)

        if not obs_space:
            raise ValueError("Failed to create MEGA observation space")
            
        # Обрезаем исходные данные
        self.opens = self.opens[self.window:]
        self.highs = self.highs[self.window:]
        self.lows = self.lows[self.window:]
        self.closes = self.closes[self.window:]
        self.volume = self.volume[self.window:]
        
        self.obs_space = np.array(obs_space)
        
        print(f"🚀 МЕГА observation space создан: {self.obs_space.shape}")
        print(f"   Features: {len(feature_keys)} МАКСИМАЛЬНЫХ фичей!")
        print(f"   Window: {self.window} временных точек")
        print("💰 Готов для СЕРЬЁЗНОГО анализа и БОЛЬШИХ денег!")

    def reverse(self):
        """Method to reverse the entire stock data, potentially helps balance training"""
        self.opens = self.opens[::-1]
        self.highs = self.highs[::-1]
        self.lows = self.lows[::-1]
        self.closes = self.closes[::-1]
        self.volume = self.volume[::-1]
        self.obs_space = self.obs_space[::-1, :]

    def get_sma(self, data, mav=30):
        dat = []
        for i in range(len(data)):
            if i < mav:
                dat.append(np.mean(data[:i+1]))
            else:
                dat.append(np.mean(data[i-mav:i+1]))
        return dat

    def get_ema(self, data, period):
        ema = []
        for i in range(len(data)):
            if i == 0:
                ema.append(data[i])
            else:
                k = 2 / (period + 1)
                ema.append(data[i] * k + ema[-1] * (1 - k))
        return ema

    def get_sd(self, data, period):
        return np.array([np.std(data[i-period:i+1]) if i >= period else np.std(data[:i+1]) for i in range(len(data))])

    def get_cci(self, period):
        mean_prices = np.array([np.mean([self.highs[i], self.lows[i], self.closes[i]]) for i in range(len(self.lows))])
        mav_prices = self.get_sma(mean_prices, period)
        mean_dev = self.get_sma(np.abs(mean_prices - mav_prices), period)

        cci = []
        for i in range(len(mav_prices)):
            if i == 0:
                cci.append(0)
            elif mean_dev[i] == 0:
                cci.append(0)
            else:
                val = (mean_prices[i] - mav_prices[i]) / (0.015 * mean_dev[i])
                if val > 300:
                    val = 300
                elif val < -300:
                    val = -300
                cci.append(val)

        # Return normalized CCI
        return np.array((np.array(cci) - min(cci)) / (max(cci) - min(cci)))

    def get_rsi(self, period):
        # Calculate RSI
        rsi = [0 for _ in range(period)]
        for i in range(period, len(self.closes)):
            ups = []
            downs = []
            for n in range(i, i-period, -1):
                up = self.closes[n] - self.closes[n-1] if self.closes[n] > self.closes[n-1] else 0
                down = abs(self.closes[n] - self.closes[n-1]) if self.closes[n] < self.closes[n-1] else 0
                ups.append(up)
                downs.append(down)
            avg_up = np.mean(ups)
            avg_down = np.mean(downs)

            if avg_down == 0:
                rsi.append(100)
            else:
                rsi.append(100 - (100 / (1 + avg_up / avg_down)))

        # Return normalized RSI
        return np.array((np.array(rsi) - min(rsi[period+1:])) /
                        (max(rsi[period+1:]) - min(rsi[period+1:])))

    def get_bollinger_bands(self, data=None, term="long"):
        if data is None:
            data = self.closes

        if term == "xtralong":
            mav = self.get_sma(data, 200)
            sd = 3 * self.get_sd(data, 200)
        elif term == "long":
            mav = self.get_sma(data, 50)
            sd = 2.5 * self.get_sd(data, 50)
        elif term == "med":
            mav = self.get_sma(data, 20)
            sd = 2 * self.get_sd(data, 20)
        else:
            mav = self.get_sma(data, 10)
            sd = 1.5 * self.get_sd(data, 10)
        return mav + sd, mav - sd

    def dy(self, data, mav=None):
        if mav:
            data = self.moving_average(data, mav)
        dy = [0]
        for i in range(1, len(data)):
            dy.append((data[i] - data[i-1]) / data[i-1])
        return dy
