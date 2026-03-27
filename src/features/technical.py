"""Technical indicator features."""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import yfinance as yf
from ..utils.cache import FeatureCache
from ..utils.config import get_config
from .base import BaseFeatureExtractor


class TechnicalFeatures(BaseFeatureExtractor):
    """Extract technical indicator features."""

    def __init__(self, cache: Optional[FeatureCache] = None):
        super().__init__(cache)
        self.config = get_config().get_section('features').get('technical', {})
        self.ma_periods = self.config.get('ma_periods', [5, 10, 20, 60])
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        self.bollinger_period = self.config.get('bollinger_period', 20)
        self.bollinger_std = self.config.get('bollinger_std', 2)

    @property
    def feature_type(self) -> str:
        return 'technical'

    def extract(self, stock_code: str, **kwargs) -> pd.DataFrame:
        """Extract technical features for a stock."""
        days = kwargs.get('days', 120)
        try:
            data = yf.download(stock_code, period=f"{days}d", auto_adjust=False, progress=False)
            if data.empty:
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        df = pd.DataFrame(index=data.index)
        df['stock_code'] = stock_code

        # Price data
        df['close'] = data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
        df['open'] = data['Open'].iloc[:, 0] if isinstance(data['Open'], pd.DataFrame) else data['Open']
        df['high'] = data['High'].iloc[:, 0] if isinstance(data['High'], pd.DataFrame) else data['High']
        df['low'] = data['Low'].iloc[:, 0] if isinstance(data['Low'], pd.DataFrame) else data['Low']
        df['volume'] = data['Volume'].iloc[:, 0] if isinstance(data['Volume'], pd.DataFrame) else data['Volume']

        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving averages
        for period in self.ma_periods:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ma_{period}_ratio'] = df['close'] / df[f'ma_{period}']

        # Exponential moving averages
        for period in [12, 26]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self.bollinger_period).mean()
        bb_std = df['close'].rolling(window=self.bollinger_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.bollinger_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.bollinger_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()

        # Volume indicators
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        df['volume_change'] = df['volume'].pct_change()

        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

        # Volatility
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()

        # Historical high/low
        df['high_20d'] = df['close'].rolling(window=20).max()
        df['low_20d'] = df['close'].rolling(window=20).min()
        df['high_low_ratio'] = df['close'] / df['high_20d']
        df['close_low_ratio'] = df['close'] / df['low_20d']

        # Price relative to high/low
        df['price_position'] = (df['close'] - df['low_20d']) / (df['high_20d'] - df['low_20d'] + 1e-10)

        # Fill NaN values with forward fill then backward fill for remaining
        df = df.ffill().bfill()
        df = df.reset_index()
        # Ensure date column name is lowercase
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'date'}, inplace=True)
        elif 'index' in df.columns:
            df.rename(columns={'index': 'date'}, inplace=True)

        return df
