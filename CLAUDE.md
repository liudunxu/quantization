# CLAUDE.md - Stock Trading ML Decision System

## Project Overview

Machine learning-based stock trading decision system supporting A-shares (A股), HK (港股), and US (美股) markets. Uses CatBoost/LightGBM/XGBoost ensemble with 150+ features, multi-strategy backtesting, and rule-based signals.

## Architecture

```
/root/work/quarnt/
├── configs/
│   └── config.yaml           # All configuration (no hardcoded values)
├── cache/
│   ├── feature_cache.db      # SQLite feature cache
│   └── strategy_params.db   # Strategy parameters storage
├── data/                    # Raw data storage
├── logs/                    # Log files
├── models/                  # Trained model files
├── src/
│   ├── backtest/            # Backtesting engine + 9 rule strategies
│   ├── data_providers/       # Multi-provider data fetching (baostock, yfinance, akshare, openbb, tushare)
│   ├── display/             # Output formatting
│   ├── features/           # Feature engineering (technical, fundamental, sentiment, alpha)
│   ├── models/             # CatBoost/LightGBM/XGBoost + ensemble
│   ├── optimization/       # Scenario optimization
│   ├── pipelines/          # Data and model pipelines
│   ├── predictors/          # Ensemble predictor + technical signals
│   └── utils/              # Cache, config, stock_info, perf_monitor
├── scripts/
│   ├── decide.py            # Trading decision entry point
│   ├── backtest.py          # Strategy backtest entry point
│   ├── predict.py          # Fast API prediction endpoint
│   └── explore_params.py   # Parameter optimization
└── tests/                   # Unit tests
```

## Core Modules

### 1. Data Providers (`src/data_providers/`)

Multi-provider data fetching with automatic fallback. Priority by market:

| Market | Priority Order |
|--------|---------------|
| A-shares | baostock → akshare → tushare → openbb → yfinance |
| HK | openbb → yfinance → akshare → tushare |
| US | openbb → yfinance |

**Providers:**
- `baostock_provider.py`: Chinese A-shares (free, forward-adjusted prices)
- `yfinance_provider.py`: Global stocks (US, HK)
- `akshare_provider.py`: Chinese A-shares/HK fallback
- `openbb_provider.py`: Global stocks via OpenBB ODP
- `tushare_provider.py`: Chinese A-shares/HK (requires `TUSHARE_TOKEN`)
- `fetch_stock_data.py`: Unified data fetching entry
- `fetch_realtime_price()`: Intraday real-time price

### 2. Features (`src/features/`)

150+ features across categories:

| Category | Count | Examples |
|----------|-------|----------|
| Technical | ~80 | MA, RSI, MACD, Bollinger, ATR, ADX, Stochastic |
| Fundamental | ~15 | PE, PB, ROE, revenue growth, debt ratio |
| Market | ~20 | Index performance, correlation, Beta |
| Sentiment | ~15 | News, social media, search trends |
| Alpha | 70+ | Qlib-style Z-score normalized factors |
| Industry | varies | Sector performance, sector ETF proxies |
| Money Flow | varies | A-share specific (skipped for US/HK) |

**Key Files:**
- `technical.py`: 36 focused extraction methods
- `alpha_features.py`: Qlib-style alpha factors
- `enhanced_sentiment.py`: Combined sentiment scoring
- `combinator.py`: Feature combination with batch merge optimization

**Cache:** SQLite-based (`cache/feature_cache.db`), never expires, manual refresh via `FeatureCache`.

### 3. Models (`src/models/`)

Multi-model ensemble support:
- `trainer.py`: CatBoost training
- `lgbm_model.py`: LightGBM model
- `xgboost_model.py`: XGBoost model
- `multi_model.py`: Ensemble with bagging

**Features:**
- Buy/Hold/Sell classification
- Auto class balancing (SMOTE + weight adjustment)
- Early stopping, feature importance selection
- Bootstrap class balancing (ensures 3 classes in all samples)
- Ensemble bagging for diversity

### 4. Backtest (`src/backtest/`)

**Rule Strategies:**
- BuyAndHoldStrategy (benchmark)
- HighSellLowBuyStrategy
- MAGoldenCrossStrategy
- BullTrendStrategy
- ShrinkPullbackStrategy
- BottomVolumeStrategy
- BoxOscillationStrategy
- EmotionCycleStrategy
- VolumeBreakoutStrategy
- OneYangThreeYinStrategy
- MACDDivergenceStrategy

**ML Strategies:**
- MLStrategy
- HybridStrategy (ML + rule-based)
- RollingMLStrategy
- RollingHybridStrategy

**Risk Control:**
- Max drawdown threshold (default 20%) - force sell and stop trading
- Dynamic position sizing via ATR risk model
- Pyramiding support (adding to positions)

### 5. Predictors (`src/predictors/`)

`EnsemblePredictor` combines:
- ML model prediction (weight: 0.35)
- Technical signals (weight: 0.25)
- Momentum signals (weight: 0.15)
- Trend signals (weight: 0.10)
- Alpha signals (weight: 0.05)
- Rule strategy signals (weight: 0.10)

## Stock Code Format

- A-shares: `{code}.SZ` or `{code}.SH` (e.g., 000001.SZ, 600000.SH)
- HK stocks: `{code}.HK` (e.g., 0700.HK)
- US stocks: `{code}` (e.g., AAPL, GOOGL)
- Indices: `{code}.SH` (e.g., 000001.SH for 上证指数)

## Usage

### Decision Script
```bash
python scripts/decide.py --stock 000001.SZ --train-days 365 --backtest-days 30
python scripts/decide.py --stock 0700.HK --refresh
python scripts/decide.py --stock AAPL
python scripts/decide.py --stock 000001.SH --index  # Index prediction
```

**Output:** Backtest results sorted by return → All strategy decisions → Final recommendation (majority vote, best by score/return, suggested position).

### Backtest Script
```bash
python scripts/backtest.py --stock 000001.SZ --days 30 --train-days 365
python scripts/backtest.py --stock 603986.SH --days 30 --output results.csv
```

### Run Tests
```bash
python -m pytest tests/ -v
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train-days` | 365 | Training data size |
| `--backtest-days` | 30 | Backtest window |
| `--refresh` | false | Force refresh cache |
| `--exclude-dates` | false | Exclude extreme volatility dates |
| `--exclude-threshold` | 2.0 | Extreme volatility std multiplier |

## Market-Specific Defaults

| Market | Bear Threshold | Rolling Window | Retrain Interval |
|--------|--------------|----------------|-----------------|
| A-share | -1.2% | 60 days | 8 days |
| HK | -0.8% | 80 days | 10 days |
| US | -0.5% | 100 days | 12 days |

## Feature Caching

```python
from src.utils.cache import FeatureCache
cache = FeatureCache()
cache.refresh('000001.SZ')  # Force refresh
cache.delete('000001.SZ')   # Delete single stock
cache.clear_all()           # Clear all
```

## Configuration

All settings in `configs/config.yaml`:
- `data.backtest_days`, `data.train_days`
- `model.catboost.*`, `model.{a_share,hk,us}.*`
- `backtest.initial_cash`, `backtest.commission`, `backtest.slippage`
- `cache.backend`: "sqlite" | "hybrid" (Redis + SQLite fallback)

## Dependencies

See `requirements.txt`. Key dependencies: catboost, lightgbm, xgboost, pandas, numpy, akshare, baostock, yfinance, openbb.

## Completed Features (Recent)

- [x] A-share index prediction (`--index` flag)
- [x] HTTP API with FastAPI (`predict.py`)
- [x] Fail-safe Redis cache (Upstash) + SQLite fallback
- [x] POST `/stocks` endpoint with zone filter
- [x] 12 major A-share indices support
- [x] Multi-model ensemble (CatBoost + LightGBM + XGBoost)
- [x] 15 rule-based strategies voting
- [x] Qlib-style Alpha features (70+)
- [x] Enhanced sentiment analysis (news, social media, search trends)
- [x] Economic calendar web scraping (FOMC, China policy, NFP)
- [x] Code quality: ruff clean (183 lint fixes)
- [x] Refactored technical.py (673→36 methods)
- [x] Refactored decide.py (439→6 focused functions)
