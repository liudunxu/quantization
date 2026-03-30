# CLAUDE.md - Stock Trading ML Decision System

## Project Overview
This project implements a machine learning-based stock trading decision system that:
- Uses CatBoost as the primary prediction model
- Incorporates multiple feature types (fundamentals, industry, market, time-series)
- Implements file-system based caching for all features
- Provides backtesting against multiple strategies
- Supports A-shares, HK, and US stocks

## Architecture

### Directory Structure
```
/root/work/quarnt/
├── configs/           # Configuration files (config.yaml)
├── cache/             # Feature cache (parquet, never expires)
├── data/              # Raw data storage
├── models/            # Trained model files
├── src/
│   ├── data_providers/ # Multi-provider data fetching (yfinance, akshare, tushare)
│   ├── features/      # Feature engineering
│   ├── models/        # CatBoost model training/prediction
│   ├── backtest/      # Backtesting engine
│   └── utils/         # Utilities (cache, config, stock_info)
├── scripts/           # Entry scripts (decide.py, backtest.py)
└── tests/             # Unit tests
```

### Core Modules

#### 1. Data Providers (`src/data_providers/`)
Multi-provider data fetching with automatic fallback:
- **yfinance**: Global stocks (US, HK) - default first choice
- **akshare**: Chinese A-shares/HK fallback
- **tushare**: Chinese A-shares/HK (requires TUSHARE_TOKEN env var)
- Retry logic (3 attempts, configurable delay)
- Auto-fallback chain per market type

#### 2. Feature Engineering (`src/features/`)
- **Technical Features**: MA, RSI, MACD, Bollinger, ATR, ADX, Stochastic, MFI, CCI, momentum
- **Fundamental Features**: PE, PB, ROE, revenue growth, debt ratio
- **Industry Features**: Sector performance, sector ETF proxies
- **Market Features**: Index performance, market sentiment
- **Money Flow Features**: A-share specific (automatically skipped for US/HK stocks)
- **Cache System**: File-based parquet, never expires, manual refresh/delete

#### 3. Model (`src/models/`)
- CatBoost classifier for buy/sell/hold decisions
- Auto class balancing, early stopping
- Feature importance analysis
- Composite label generation (combining returns, trend, momentum, market)

#### 4. Backtest (`src/backtest/`)
- Multiple strategies: ML Strategy, Buy & Hold, High Sell Low Buy, and more
- Dynamic position sizing based on ATR risk model
- Support for adding to existing positions (pyramiding)
- Metrics: Returns, Sharpe ratio, max drawdown, win rate, trade count

#### 5. Scripts (`scripts/`)
- `decide.py`: Trading decision with confidence/probabilities and suggested position
- `backtest.py`: Strategy comparison backtest

## Position Sizing Model

The system uses ATR-based position sizing:
- **Risk per trade**: 1% of capital
- **ATR multiplier**: 1.2 (stop loss distance)
- **Minimum lot**: 50 shares for expensive stocks (>$100), 100 shares otherwise
- **Maximum lots per trade**: 3 lots (300 shares)

Position formula: `shares = risk_amount / (ATR * atr_multiplier / price)`

## Market Filtering

- Uses 3-day average market return to determine regime (smoother than single day)
- Bear market threshold varies by market (A-shares: -1%, US: -0.5%)
- High confidence buy signals (≥0.7) allowed even in bear market
- All strategies use consistent initial position sizing

## Feature Caching

### Cache Location
`/root/work/quarnt/cache/`

### Cache Structure
- `{stock_code}_fundamental.parquet` - Fundamental data
- `{stock_code}_technical.parquet` - Technical indicators
- `{stock_code}_market.parquet` - Market data
- `{stock_code}_industry.parquet` - Industry data
- `{stock_code}_money_flow.parquet` - Money flow data (A-shares only)

### Cache Commands
```python
# Manual refresh
from src.utils.cache import FeatureCache
cache = FeatureCache()
cache.refresh('000001.SZ')

# Delete cache
cache.delete('000001.SZ')

# Clear all
cache.clear_all()
```

## Stock Code Format
- A-shares: `{code}.SZ` or `{code}.SH` (e.g., 000001.SZ, 600000.SH)
- HK stocks: `{code}.HK` (e.g., 0700.HK)
- US stocks: `{code}` (e.g., AAPL, GOOGL)

## Usage

### Run Decision
```bash
python scripts/decide.py --stock 000001.SZ --train-days 365 --backtest-days 30
python scripts/decide.py --stock 0700.HK --refresh  # Force refresh cache
python scripts/decide.py --stock AAPL
```

**Decision Flow:** Train model → Backtest strategies → Output each strategy's decision → Use best-performer as final recommendation.

**Key Parameters:**
- `--train-days`: Training data size (365 default). Larger = more stable but may include outdated patterns.
- `--backtest-days`: Backtest window (30 default). Used to select best strategy and evaluate performance.

**Output Sections:**
- **BACKTEST RESULTS**: Sorted by total return (highest first)
- **ALL STRATEGY DECISIONS**: Each strategy's decision, return, and score
- **FINAL RECOMMENDATION**: 
  - Majority vote decision
  - Best strategy by score
  - Best strategy by return
  - Suggested position (lots, shares, position %, stop loss, take profit)

### Run Backtest
```bash
python scripts/backtest.py --stock 000001.SZ --days 30 --train-days 365
python scripts/backtest.py --stock 603986.SH --days 30 --output results.csv
```

## Configuration
See `configs/config.yaml` for model parameters and feature settings.

## Dependencies
See `requirements.txt`
