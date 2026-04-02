# AGENTS.md - AI Assistant Guidelines for Stock Trading ML System

## Overview
This document provides guidelines for AI assistants working on this stock trading ML decision system.

## Project Context
- **Language**: Python 3.x
- **ML Framework**: CatBoost
- **Primary Markets**: A-shares (China), HK stocks, US stocks
- **Architecture**: Modular with data providers, features, models, and backtest engine

## Key Conventions

### Code Style
- Follow existing code patterns in the project
- Use type hints for function parameters
- Add docstrings for public functions
- Keep functions focused and modular

### File Organization
```
src/
├── data_providers/  # Data fetching (one file per provider)
├── features/        # Feature engineering (one file per feature type)
├── models/          # ML model training/prediction
├── backtest/        # Backtesting engine and strategies
└── utils/           # Utilities (cache, config, stock_info, important_dates)
```

### Naming Conventions
- Stock codes: `{code}.{exchange}` (e.g., `000001.SZ`, `AAPL`, `0700.HK`)
- Feature columns: lowercase with underscores (e.g., `ma_5`, `rsi_14`)
- Strategy classes: PascalCase (e.g., `MLStrategy`, `BuyAndHoldStrategy`)

## Common Tasks

### Adding a New Data Provider
1. Create new file in `src/data_providers/`
2. Implement `BaseDataProvider` interface
3. Add to `__init__.py` exports
4. Update `fetch_stock_data.py` fallback logic
5. Update `configs/config.yaml` if needed

### Adding a New Strategy
1. Create strategy class in `src/backtest/`
2. Implement `generate_signals()` method
3. Add to `get_market_strategies()` in `strategies.py`
4. Test with `scripts/backtest.py`

### Modifying Model Parameters
1. Update `configs/config.yaml` under `model` section
2. Market-specific params under `model.a_share`, `model.hk`, `model.us`
3. Test with `scripts/decide.py`

## Important Files

| File | Purpose |
|------|---------|
| `scripts/decide.py` | Main entry point for trading decisions |
| `scripts/backtest.py` | Strategy comparison and backtesting |
| `scripts/predict.py` | Stock price prediction with ML |
| `scripts/explore_params.py` | Parameter optimization |
| `src/backtest/engine.py` | Core backtesting engine with risk control |
| `src/models/trainer.py` | CatBoost model training with ensemble |
| `src/utils/important_dates.py` | Important dates management for extreme volatility filtering |
| `configs/config.yaml` | All configuration parameters |

## Testing

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Test Specific Stock
```bash
python scripts/decide.py --stock 000001.SZ --train-days 100 --backtest-days 20
```

### Test with Extreme Date Filtering
```bash
# Predict with extreme date filtering
python scripts/predict.py --stock 000001.SZ --exclude-dates

# Optimize parameters with extreme date filtering
python scripts/explore_params.py --stock 000001.SZ --exclude-dates
```

### Test Different Markets
```bash
# A-share
python scripts/decide.py --stock 000001.SZ

# HK
python scripts/decide.py --stock 0700.HK

# US
python scripts/decide.py --stock AAPL
```

## Configuration Reference

### Risk Management
```yaml
backtest:
  max_drawdown_threshold: 0.20  # 20% max drawdown
```

### Data Providers
```yaml
data_providers:
  priority:
    a_share: ['baostock', 'akshare', 'tushare', 'openbb', 'yfinance']
    hk: ['openbb', 'yfinance', 'akshare', 'tushare']
    us: ['openbb', 'yfinance']
```

### Model Parameters
```yaml
model:
  catboost:
    iterations: 300
    depth: 4
    learning_rate: 0.03
    n_estimators: 3  # Ensemble size
```

## Troubleshooting

### Common Issues

1. **"Found only 2 unique classes" error**
   - The system automatically handles this by adding synthetic samples
   - No action needed

2. **Data provider fails**
   - System automatically falls back to next provider
   - Check `logs/trading.log` for details

3. **Low trading frequency**
   - Adjust `confidence_threshold` in config
   - Check market-specific parameters

4. **Extreme date filtering not working**
   - Check if `cache/important_dates.db` exists
   - Verify data has required columns (date, open, high, low, close)
   - Lower `--exclude-threshold` to detect more dates

## Git Workflow

### Commit Message Format
```
feat: add new feature
fix: bug fix
docs: documentation update
refactor: code refactoring
test: add tests
chore: maintenance tasks
```

### Before Committing
1. Run tests: `python -m pytest tests/ -v`
2. Test with sample stocks
3. Update documentation if needed

## Dependencies

### Required
```bash
pip install catboost scikit-learn pandas numpy yfinance akshare baostock
```

### Optional
```bash
pip install openbb  # For OpenBB data provider
```

## Support
- Documentation: See `CLAUDE.md` and `README.md`
- Issues: Check logs in `logs/` directory
