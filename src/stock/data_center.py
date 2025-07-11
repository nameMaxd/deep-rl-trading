import yfinance as yf
import pandas as pd


def load_stock(ticker, period, timeframe):
    # Load in data from yfinance API
    try:
        df = yf.download(tickers=ticker, interval=timeframe, period=period)
        return df

    except Exception as e:
        print(str(e))


def load_stock_from_csv(csv_path):
    """Load stock data from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Keep only OHLC and Volume columns
        if 'Adj Close' in df.columns:
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return None
