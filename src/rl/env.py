import matplotlib.pyplot as plt
from src.stock.stock import Stock
import numpy as np
import random
import pickle
import os


class Env:
    def __init__(self, tickers=None, fee=0.001, trading_period=100, csv_paths=None, window=364):
        # Initializes the trading environment with optional tickers, a trading fee, and a specified trading period.
        self.tickers = tickers
        self.fee = fee
        self.trading_period = trading_period
        self.stocks = []
        self.csv_paths = csv_paths
        self.window = window

        # Epoch specific trading params
        self.actions = []
        self.state_actions = [0 for _ in range(7)]
        self.rewards = []
        self.trades = []
        self.in_trade = False
        self.entry_price = 0
        self.initial_capital = 10000  # Starting capital in $
        
        # ЦЕЛЬ-ОРИЕНТИРОВАННЫЕ reward components
        self.recent_returns = []
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown_current = 0
        self.peak_equity = self.initial_capital
        self.current_equity = self.initial_capital
        
        # Стратегические параметры
        self.target_profit = 500  # Целевая прибыль
        self.max_trades_per_episode = 8  # Максимум сделок
        self.min_trades_per_episode = 3  # Минимум сделок
        self.hold_time = 0  # Время удержания позиции

        self.stock = None
        self.ind = 0
        self.start = 0
        self.end = 0

        self.load_data(tickers=tickers, csv_paths=csv_paths)
        self.reset()

    def load_stock(self, stock):
        # Sets the current stock to the provided stock and resets the environment state.
        self.stock = stock
        state = self.reset(flip=False, stock=False)
        return state

    def reset(self, flip=True, stock=True):
        # Resets the environment to a new trading session. Optionally flips the stock data and selects a random stock.
        if stock:
            self.stock = random.choice(self.stocks)

        if flip and random.random() > 0.5:
            self.stock.reverse()

        self.actions = []
        self.rewards = []
        self.trades = []
        self.in_trade = False
        self.entry_price = 0
        self.hold_time = 0
        
        # Reset цель-ориентированные компоненты
        self.recent_returns = []
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown_current = 0
        self.peak_equity = self.initial_capital
        self.current_equity = self.initial_capital

        self.start = random.sample(range(1, len(self.stock.closes) - self.trading_period - 1), 1)[0]
        self.end = self.start + self.trading_period
        self.ind = self.start

        return self.stock.obs_space[self.ind, :, :]

    def reset_fixed(self, start_position=0, flip=False):
        """ЧЕСТНЫЙ reset для фиксированного тестирования без случайности"""
        # НЕ выбираем случайную акцию - используем текущую
        
        # НЕ переворачиваем данные случайно
        if flip:
            self.stock.reverse()

        self.actions = []
        self.rewards = []
        self.trades = []
        self.in_trade = False
        self.entry_price = 0
        self.hold_time = 0
        
        # Reset цель-ориентированные компоненты
        self.recent_returns = []
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown_current = 0
        self.peak_equity = self.initial_capital
        self.current_equity = self.initial_capital

        # ФИКСИРОВАННАЯ стартовая позиция - НЕ случайная!
        max_start = len(self.stock.closes) - self.trading_period - 1
        self.start = min(start_position, max_start)  # Защита от выхода за границы
        self.end = self.start + self.trading_period
        self.ind = self.start

        return self.stock.obs_space[self.ind, :, :]

    def step(self, action):
        """
        ЦЕЛЬ-ОРИЕНТИРОВАННЫЙ step: стабильный профит $500+ на OOS
        """
        prev_close = self.stock.closes[self.ind - 1]
        current_close = self.stock.closes[self.ind]
        p_change = np.log(current_close / prev_close)
        
        # Компоненты reward
        profit_reward = 0
        quality_reward = 0  
        risk_penalty = 0
        strategy_reward = 0
        
        if not self.in_trade:
            if action == 1:  # BUY
                # Вход в позицию
                self.in_trade = True
                self.entry_price = self.stock.opens[self.ind]
                self.trade_count += 1
                self.hold_time = 0
                
                # Reward за качественный entry
                entry_return = np.log(current_close / self.entry_price)
                profit_reward = entry_return
                
                # Bonus за entry в правильный момент (если цена пошла вверх)
                if entry_return > 0:
                    quality_reward += 0.01
                
                # Fee
                profit_reward -= self.fee
                
                self.trades.append((self.ind, 1, self.entry_price))
                
            else:  # HOLD when not in trade
                # Небольшой penalty за бездействие, но не критичный
                if len(self.actions) > 10:  # После разминки
                    strategy_reward = -0.0001
                    
        else:  # В позиции
            self.hold_time += 1
            
            if action == 0:  # SELL
                # Выход из позиции
                self.in_trade = False
                exit_price = current_close
                
                # Рассчитываем прибыль от сделки
                trade_return = np.log(exit_price / self.entry_price)
                shares = self.current_equity // self.entry_price
                trade_profit = shares * (exit_price - self.entry_price) - 2 * shares * self.entry_price * self.fee
                
                # Обновляем equity
                self.current_equity += trade_profit
                if self.current_equity > self.peak_equity:
                    self.peak_equity = self.current_equity
                
                # Основной profit reward
                profit_reward = trade_return - self.fee
                
                # Quality reward - за хорошие сделки
                if trade_return > 0:
                    self.winning_trades += 1
                    # Bonus за прибыльную сделку
                    quality_reward += 0.02
                    # Extra bonus за большую прибыль
                    if trade_return > 0.02:  # >2% profit
                        quality_reward += 0.01
                else:
                    self.losing_trades += 1
                    # Penalty за убыточную сделку
                    quality_reward -= 0.01
                    
                # Win rate bonus
                if self.trade_count > 0:
                    win_rate = self.winning_trades / self.trade_count
                    if win_rate > 0.6:
                        quality_reward += 0.005
                        
                # Penalty за слишком долгое удержание
                if self.hold_time > 15:
                    quality_reward -= 0.002
                    
                self.trades.append((self.ind, 0, exit_price))
                self.entry_price = 0
                self.hold_time = 0
                
            else:  # HOLD when in trade
                # Находимся в позиции - получаем изменение цены
                profit_reward = p_change
                
                # Penalty за слишком долгое удержание
                if self.hold_time > 20:
                    risk_penalty -= 0.001

        # Risk Management Penalties
        current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        if current_drawdown > self.max_drawdown_current:
            self.max_drawdown_current = current_drawdown
            
        # Penalty за большую просадку
        if self.max_drawdown_current > 0.15:  # >15% drawdown
            risk_penalty -= 0.01
        elif self.max_drawdown_current > 0.10:  # >10% drawdown  
            risk_penalty -= 0.005

        # Strategy rewards
        progress = (self.ind - self.start) / self.trading_period
        
        # Penalty за overtrade
        if self.trade_count > self.max_trades_per_episode:
            strategy_reward -= 0.01
            
        # Penalty за недостаточную активность (в конце периода)
        if progress > 0.8 and self.trade_count < self.min_trades_per_episode:
            strategy_reward -= 0.005

        # Сохраняем returns для Sharpe
        self.recent_returns.append(profit_reward)
        if len(self.recent_returns) > 20:
            self.recent_returns.pop(0)
            
        # Sharpe-based bonus
        sharpe_bonus = 0
        if len(self.recent_returns) >= 10:
            returns_array = np.array(self.recent_returns)
            if np.std(returns_array) > 0:
                sharpe = np.mean(returns_array) / np.std(returns_array)
                if sharpe > 1.0:
                    sharpe_bonus = 0.005
                elif sharpe < -0.5:
                    sharpe_bonus = -0.005

        # Итоговый reward
        total_reward = profit_reward + quality_reward + risk_penalty + strategy_reward + sharpe_bonus

        # Прогрессивный bonus в конце эпизода
        if self.ind >= self.end - 1:
            episode_bonus = self._calculate_final_episode_bonus()
            total_reward += episode_bonus

        self.ind += 1
        self.rewards.append(total_reward)
        self.actions.append(action)

        done = False
        if self.ind >= self.end or self.ind == len(self.stock.closes) - 1:
            done = True

        next_state = self.stock.obs_space[self.ind, :, :]

        return next_state, action, total_reward, done

    def _calculate_final_episode_bonus(self):
        """Финальный bonus based on ЦЕЛЬ: $500+ profit"""
        episode_bonus = 0
        
        # Главный bonus - за достижение target profit
        total_profit = self.current_equity - self.initial_capital
        if total_profit >= self.target_profit:
            episode_bonus += 0.1  # Большой bonus за цель
        elif total_profit >= self.target_profit * 0.5:
            episode_bonus += 0.05  # Меньший за половину цели
        elif total_profit < -200:
            episode_bonus -= 0.05  # Penalty за большие потери
            
        # Win rate bonus
        if self.trade_count > 0:
            win_rate = self.winning_trades / self.trade_count
            if win_rate >= 0.6:
                episode_bonus += 0.02
            elif win_rate < 0.4:
                episode_bonus -= 0.01
                
        # Trading activity bonus
        if self.min_trades_per_episode <= self.trade_count <= self.max_trades_per_episode:
            episode_bonus += 0.01
            
        # Risk management bonus
        if self.max_drawdown_current < 0.1:
            episode_bonus += 0.01
            
        return episode_bonus

    def get_cumulative_rewards(self):
        # Computes the cumulative rewards over the trading period. (current not used for training)
        cumulative_reward = 1
        cumulative_rewards = []
        for i in range(len(self.rewards)):
            if self.actions[i] == 1 or (i > 0 and self.actions[i] != self.actions[i-1]):
                cumulative_reward *= (1 + self.rewards[i])
            cumulative_rewards.append(cumulative_reward)

        return cumulative_rewards, cumulative_reward

    def get_trading_metrics(self):
        """Calculate advanced trading metrics"""
        if len(self.trades) < 2:
            return {
                'total_profit_dollars': self.current_equity - self.initial_capital,
                'num_trades': self.trade_count,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'avg_trade_return': 0,
                'max_drawdown': self.max_drawdown_current
            }
        
        # Calculate trade profits in dollars
        trade_profits = []
        capital = self.initial_capital
        equity_curve = [capital]
        
        i = 0
        while i < len(self.trades) - 1:
            if self.trades[i][1] == 1 and i + 1 < len(self.trades) and self.trades[i+1][1] == 0:
                # Buy and sell pair
                entry_price = self.trades[i][2]
                exit_price = self.trades[i+1][2]
                
                # Calculate profit per share
                profit_per_share = exit_price - entry_price
                
                # Calculate number of shares we could buy
                shares = capital // entry_price
                
                # Calculate total profit/loss
                total_profit = shares * profit_per_share - 2 * shares * entry_price * self.fee
                trade_profits.append(total_profit)
                
                capital += total_profit
                equity_curve.append(capital)
                i += 2
            else:
                i += 1
        
        if not trade_profits:
            return {
                'total_profit_dollars': self.current_equity - self.initial_capital,
                'num_trades': self.trade_count,
                'sharpe_ratio': 0,
                'win_rate': self.winning_trades / max(self.trade_count, 1),
                'avg_trade_return': 0,
                'max_drawdown': self.max_drawdown_current
            }
        
        # Calculate metrics
        total_profit = sum(trade_profits)
        num_trades = len(trade_profits)
        win_rate = len([p for p in trade_profits if p > 0]) / num_trades if num_trades > 0 else 0
        avg_trade_return = np.mean(trade_profits)
        
        # Sharpe ratio (annualized)
        if len(trade_profits) > 1:
            returns = np.array(trade_profits) / self.initial_capital
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
            
        # Maximum drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'total_profit_dollars': total_profit,
            'num_trades': num_trades,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'max_drawdown': max_drawdown
        }

    def render(self, action_types=[], wait=False):
        """
        Visualizes the trading actions and stock price over the trading period.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Stock price and actions
        dates = range(self.start, self.start + len(self.actions))
        prices = self.stock.closes[self.start:self.start + len(self.actions)]

        ax1.plot(dates, prices, label='Stock Price', color='blue')

        # Mark trades
        for trade in self.trades:
            idx, action_type, price = trade
            color = 'green' if action_type == 1 else 'red'
            label = 'Buy' if action_type == 1 else 'Sell'
            ax1.scatter(idx, price, color=color, s=100, alpha=0.7, label=label)

        ax1.set_title(f'Trading Performance - {self.stock.ticker}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)

        # Rewards
        ax2.plot(dates, np.cumsum(self.rewards), label='Cumulative Reward', color='purple')
        ax2.set_title('Cumulative Rewards')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Cumulative Reward')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        if wait:
            plt.show()
        else:
            plt.savefig(f'models/trading_visualization_{self.stock.ticker}.png', dpi=300, bbox_inches='tight')
            plt.close()

    def load_data(self, p="data/train", tickers=None, csv_paths=None):
        """
        Loads stock data from either CSV files or fetches data using tickers.
        """
        if csv_paths:
            # Load from CSV files
            for csv_path in csv_paths:
                try:
                    stock = Stock(csv_path=csv_path, window=self.window)
                    self.stocks.append(stock)
                    print(f"Stocks loaded: {len(self.stocks)}.")
                except Exception as e:
                    print(f"Error loading {csv_path}: {e}")
        else:
            # Original loading logic for tickers
            if tickers:
                for ticker in tickers:
                    try:
                        stock = Stock(ticker=ticker, window=self.window)
                        self.stocks.append(stock)
                        print(f"Stock {ticker} loaded.")
                    except Exception as e:
                        print(f"Error loading {ticker}: {e}")
            else:
                # Load from directory (original logic)
                for file in os.listdir(p):
                    if ".DS" not in file:
                        file_path = os.path.join(p, file)
                        with open(file_path, "rb") as f:
                            s = pickle.load(f)
                        self.stocks.append(s)
                        print(f"Stock loaded from {file_path}.")

        if not self.stocks:
            raise ValueError("No stocks were loaded!")

        print(f"Total stocks loaded: {len(self.stocks)}")
