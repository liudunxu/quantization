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
├── configs/           # Configuration files
├── cache/             # Feature cache (never expires)
├── data/              # Raw data storage
├── models/            # Trained model files
├── src/
│   ├── features/      # Feature engineering
│   ├── models/        # Model training and prediction
│   ├── backtest/      # Backtesting engine
│   └── utils/         # Utilities (cache, helpers)
├── scripts/           # Entry point scripts
└── tests/             # Unit tests
```

### Core Modules

#### 1. Feature Engineering (`src/features/`)
- **Fundamental Features**: PE, PB, ROE, revenue growth, debt ratio
- **Industry Features**: Sector performance, industry correlation
- **Market Features**: Index performance, market sentiment, money flow
- **Time Series Features**: Moving averages, RSI, MACD, Bollinger Bands, momentum
- **Cache System**: File-based, never expires, supports manual refresh/delete

#### 2. Model (`src/models/`)
- CatBoost classifier for buy/sell/hold decisions
- Feature importance analysis
- Model persistence and loading

#### 3. Backtest (`src/backtest/`)
- Implements multiple strategies:
  - **ML Strategy**: Based on model predictions
  - **Buy & Hold**: Baseline strategy
  - **High Sell Low Buy**: Contrarian strategy
- Metrics: Returns, Sharpe ratio, max drawdown, win rate

#### 4. Decision Script (`scripts/decide.py`)
- Input: Stock code (e.g., 000001.SZ, 0700.HK, AAPL)
- Output: Trading decision (buy/sell/hold) with confidence

## Feature Caching

### Cache Location
`/root/work/quarnt/cache/`

### Cache Structure
- `{stock_code}_fundamental.parquet` - Fundamental data
- `{stock_code}_technical.parquet` - Technical indicators
- `{stock_code}_market.parquet` - Market data
- `{stock_code}_industry.parquet` - Industry data

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
python scripts/decide.py --stock 000001.SZ
python scripts/decide.py --stock 0700.HK
python scripts/decide.py --stock AAPL
```

### Run Backtest
```bash
python scripts/backtest.py --stock 000001.SZ --days 30
```

## Configuration
See `configs/config.yaml` for model parameters and feature settings.

## Dependencies
See `requirements.txt`
