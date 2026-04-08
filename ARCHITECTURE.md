# Stock Trading ML Decision System - 技术架构文档

## 1. 系统概述

本系统是一个基于机器学习的股票交易决策系统，支持A股、港股、美股三大市场。系统采用模块化设计，通过特征工程、机器学习模型训练、多策略回测对比，为用户提供交易决策建议。

### 1.1 核心功能

- **数据获取**：多数据源支持（baostock、akshare、yfinance、openbb、tushare）
- **特征工程**：技术指标、基本面、市场情绪、Alpha因子等多维度特征（150+）
- **模型训练**：多模型支持（CatBoost、LightGBM、XGBoost）+ 集成学习
- **策略回测**：9种规则策略 + 4种ML策略
- **参数优化**：支持为特定股票寻找最优策略参数
- **交易决策**：综合多策略信号生成最终交易建议
- **HTTP API**：提供 FastAPI 服务，支持快速预测调用

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Scripts Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ decide.py   │  │ backtest.py │  │ explore_params.py          │  │
│  │ 决策入口    │  │ 回测入口    │  │ 参数优化入口                │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ predict.py  │  │predict_worker│ │                            │  │
│  │ 预测入口    │  │ JS worker   │  │                            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                          Core Modules                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │   features   │  │   models    │  │      backtest           │  │
│  │   特征工程   │  │   ML模型    │  │      回测引擎           │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │data_providers│  │   display   │  │       utils             │  │
│  │   数据获取   │  │   结果展示   │  │       工具类            │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  pipelines   │  │optimization │  │      predictors          │  │
│  │   流水线    │  │   优化场景   │  │      信号生成           │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                         Storage Layer                                │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    SQLite Databases                             ││
│  │  cache/feature_cache.db          cache/strategy_params.db       ││
│  │  特征数据缓存                    策略参数存储                    ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

## 3. 模块详解

### 3.1 data_providers - 数据获取层

负责从不同数据源获取股票数据，支持自动降级。

```
data_providers/
├── base.py              # 基础接口定义
├── baostock_provider.py # A股数据（baostock）
├── akshare_provider.py  # A股/港股数据（akshare）
├── yfinance_provider.py # 港股/美股数据
├── openbb_provider.py   # OpenBB数据源
├── tushare_provider.py  # Tushare数据源
├── sentiment_provider.py # 情绪数据
└── fetch_stock_data.py  # 数据获取统一入口
```

**数据源优先级：**
- A股：baostock > akshare > tushare > openbb > yfinance
- 港股：openbb > yfinance > akshare > tushare
- 美股：openbb > yfinance

### 3.2 features - 特征工程层

计算和管理各类特征指标。

```
features/
├── base.py              # 特征提取基类
├── technical.py         # 技术指标（MA、RSI、MACD、布林带等）
├── fundamental.py       # 基本面指标（PE、PB、ROE等）
├── market.py            # 市场指标（指数表现、相关性等）
├── industry.py          # 行业指标
├── sentiment.py         # 情绪指标
├── enhanced_sentiment.py # 增强情绪分析
├── money_flow.py        # 资金流向（A股）
├── southbound_flow.py   # 南向资金流（港股）
├── us_market_sentiment.py # 美股市场情绪
├── alpha_features.py    # Alpha因子（70+）
├── index_features.py    # 指数特征
├── company_events.py    # 公司事件特征
└── combinator.py        # 特征组合器
```

**特征类别：**
- 技术特征：~80个（均线、动量、波动率等）
- 基本面特征：~15个（估值、盈利、成长性等）
- 市场特征：~20个（指数、相关性、Beta等）
- 情绪特征：~15个（新闻情绪、社交媒体等）
- Alpha因子：70+个（Qlib风格Z-score标准化）

### 3.3 models - 机器学习层

多模型支持及集成学习。

```
models/
├── base.py           # 模型基类
├── trainer.py        # CatBoost模型训练和预测
├── lgbm_model.py     # LightGBM模型
├── xgboost_model.py  # XGBoost模型
└── multi_model.py    # 多模型集成
```

**模型特性：**
- 支持算法：CatBoost、LightGBM、XGBoost
- 输出：Buy / Hold / Sell 三分类
- 特征自动选择：基于重要性排序
- 类别平衡：SMOTE + 类权重调整
- 集成学习：bagging 多样性

### 3.4 backtest - 回测引擎层

提供策略回测和对比功能。

```
backtest/
├── engine.py          # 回测引擎核心
├── strategies.py      # 策略配置
└── rule_strategies.py # 基于规则的策略实现
```

**支持的策略：**

| 类型 | 策略 | 说明 |
|------|------|------|
| 规则策略 | Buy & Hold | 买入持有基准 |
| | High Sell Low Buy | 高抛低吸 |
| | MA Golden Cross | 均线金叉 |
| | Bull Trend | 趋势跟随 |
| | Shrink Pullback | 缩量回调 |
| | Bottom Volume | 底部放量 |
| | Box Oscillation | 箱体震荡 |
| | Volume Breakout | 放量突破 |
| | MACD Divergence | MACD背驰 |
| | Emotion Cycle | 情绪周期 |
| ML策略 | ML Strategy | 纯ML预测 |
| | Hybrid Strategy | ML + 规则混合 |
| | Rolling ML | 滚动训练ML |
| | Rolling Hybrid | 滚动训练混合 |

### 3.5 pipelines - 流水线层

核心数据处理和模型训练流水线。

```
pipelines/
├── data_pipeline.py   # 数据获取流水线
└── model_pipeline.py  # 模型训练/预测流水线
```

### 3.6 optimization - 优化层

参数优化场景定义。

```
optimization/
└── scenarios.py       # 预测场景参数配置
```

### 3.7 predictors - 预测信号层

综合多信号源生成交易信号。

```
predictors/
├── ensemble_predictor.py  # 集成预测器
└── technical_signals.py   # 技术信号生成
```

### 3.8 display - 展示层

负责结果格式化和输出。

```
display/
├── formatters.py           # 输出格式化
└── prediction_formatter.py  # 预测输出格式化
```

### 3.9 utils - 工具层

提供缓存、配置、参数管理等基础设施。

```
utils/
├── cache.py            # 特征缓存（SQLite）
├── config.py           # 配置管理（YAML）
├── stock_info.py       # 股票信息解析
├── strategy_params.py  # 策略参数管理
├── important_dates.py  # 重要日期管理
└── perf_monitor.py    # 性能监控
```

## 4. 数据流

### 4.1 预测流程 (predict.py)

```
┌─────────────┐
│  输入股票代码 │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 1. 获取数据                          │
│    - 调用 data_providers 获取历史数据 │
│    - 从缓存加载或重新获取             │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 2. 特征工程                          │
│    - combinator 组合所有特征          │
│    - 计算技术指标、基本面、市场等     │
│    - 计算 Alpha 因子                  │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 3. 生成预测信号                      │
│    - ML 模型预测概率                  │
│    - 技术分析 18 种指标综合判断       │
│    - 多时间框架趋势确认              │
│    - 动量分析                        │
│    - 趋势强度 ADX                    │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 4. 集成预测                          │
│    - ensemble_predictor 综合多信号   │
│    - 加权投票生成最终信号            │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 5. 输出预测结果                      │
│    - 方向、信号、置信度              │
│    - 各信号源分析                    │
│    - 模型性能指标                    │
└─────────────────────────────────────┘
```

### 4.2 参数优先级

```
股票特定参数 (stock_code=9626.HK, market=hk)
        │
        ▼ 如果没有找到
市场级别参数 (market=hk)
        │
        ▼ 如果没有找到
默认参数 (market=default)
        │
        ▼ 如果没有找到
硬编码默认值 (DEFAULT_RULE_PARAMS)
```

## 5. 数据库设计

### 5.1 feature_cache.db

```sql
-- 特征缓存元数据
CREATE TABLE cache_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code TEXT NOT NULL,
    feature_type TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    params TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    row_count INTEGER,
    column_count INTEGER,
    size_bytes INTEGER,
    UNIQUE(stock_code, feature_type, cache_key)
);

-- 特征数据存储
CREATE TABLE cache_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code TEXT NOT NULL,
    feature_type TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    data BLOB NOT NULL,
    FOREIGN KEY (stock_code, feature_type, cache_key)
        REFERENCES cache_metadata(stock_code, feature_type, cache_key)
        ON DELETE CASCADE
);
```

### 5.2 strategy_params.db

```sql
-- 策略参数表
CREATE TABLE strategy_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code TEXT,           -- 股票代码（可为空，表示市场级别）
    market TEXT NOT NULL,      -- 市场类型
    strategy_name TEXT NOT NULL,
    params TEXT NOT NULL,      -- JSON格式参数
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(stock_code, market, strategy_name)
);

-- 市场参数表
CREATE TABLE market_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code TEXT,           -- 股票代码（可为空）
    market TEXT NOT NULL,
    params TEXT NOT NULL,      -- JSON格式参数
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(stock_code, market)
);
```

## 6. HTTP API 服务

### 6.1 启动服务

```bash
python scripts/predict.py --serve --host 0.0.0.0 --port $PORT
```

### 6.2 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/predict` | GET/POST | 股票预测 |
| `/stocks/{code}/info` | GET | 股票信息 |
| `/stocks` | GET | 股票列表 |

### 6.3 性能优化参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `fast_mode` | bool | 快速模式，跳过训练/评估/实时价格 | false |
| `skip_training` | bool | 跳过模型训练 | false |
| `skip_eval` | bool | 跳过模型评估 | false |
| `skip_realtime` | bool | 跳过实时价格查询 | false |
| `skip_params` | bool | 跳过优化参数查询 | false |
| `train_days` | int | 训练天数 | 365 |
| `threshold` | float | 涨跌阈值 | 0.008 |

## 7. 关键设计决策

### 7.1 为什么选择 CatBoost？

1. **处理类别特征**：自动处理分类变量
2. **内置过拟合保护**：Ordered Boosting
3. **无需大量调参**：默认参数表现良好
4. **特征重要性**：提供可靠的特征排序

### 7.2 为什么使用 SQLite 缓存？

1. **单文件存储**：便于管理和备份
2. **事务支持**：数据一致性保证
3. **无需服务器**：零配置部署
4. **查询灵活**：支持复杂查询

### 7.3 多模型集成策略

1. **bagging多样性**：不同随机种子训练多个模型
2. **投票机制**：多数投票决定最终信号
3. **置信度加权**：根据各模型表现分配权重

## 8. 性能优化

### 8.1 数据获取优化

- **缓存机制**：避免重复获取历史数据
- **增量更新**：只获取新增数据
- **多数据源降级**：主源失败自动切换

### 8.2 计算优化

- **向量化计算**：使用 pandas/numpy 向量化操作
- **懒加载**：只在需要时计算特征
- **并行处理**：多股票并行处理支持

### 8.3 内存优化

- **数据类型优化**：使用适当的 dtype
- **及时释放**：处理完立即释放内存
- **分块处理**：大数据集分块处理

## 9. 扩展指南

### 9.1 添加新的数据源

1. 创建 `src/data_providers/xxx_provider.py`
2. 实现 `BaseDataProvider` 接口
3. 在 `fetch_stock_data.py` 中注册
4. 更新 `config.yaml` 中的优先级

### 9.2 添加新的特征

1. 在 `src/features/` 中创建或修改提取器
2. 在 `combinator.py` 中注册
3. 更新 `display/formatters.py` 中的描述

### 9.3 添加新的策略

1. 在 `src/backtest/rule_strategies.py` 中实现策略类
2. 在 `strategies.py` 中添加配置
3. 在 `explore_params.py` 中添加参数空间

### 9.4 添加新的预测模型

1. 在 `src/models/` 中创建模型类
2. 实现 `BaseModel` 接口
3. 在 `multi_model.py` 中注册

## 10. 目录结构

```
quantization/
├── pyproject.toml              # 项目配置和依赖管理 (uv)
├── requirements.txt            # pip 依赖文件
├── railway.json                # Railway 部署配置
├── runtime.txt                 # Python 版本
├── configs/
│   └── config.yaml             # 全局配置
├── cache/
│   ├── feature_cache.db        # 特征缓存数据库
│   └── strategy_params.db      # 策略参数数据库
├── data/                       # 原始数据存储
├── logs/                       # 日志文件
├── models/                     # 模型文件
├── scripts/
│   ├── decide.py               # 交易决策脚本
│   ├── backtest.py             # 回测脚本
│   ├── predict.py              # 涨跌预测脚本
│   ├── predict_worker.js       # JS Worker
│   └── explore_params.py       # 参数优化脚本
├── src/
│   ├── pipelines/              # 核心流水线
│   │   ├── data_pipeline.py    # 数据获取流水线
│   │   └── model_pipeline.py   # 模型训练/预测流水线
│   ├── optimization/           # 优化模块
│   │   └── scenarios.py        # 场景定义
│   ├── display/                # 显示模块
│   │   ├── formatters.py       # 输出格式化
│   │   └── prediction_formatter.py # 预测输出格式化
│   ├── features/               # 特征工程
│   │   ├── technical.py        # 技术指标
│   │   ├── fundamental.py       # 基本面数据
│   │   ├── market.py           # 市场/大盘数据
│   │   ├── industry.py         # 行业数据
│   │   ├── sentiment.py        # 情绪分析
│   │   ├── enhanced_sentiment.py # 增强情绪分析
│   │   ├── money_flow.py       # 资金流向
│   │   ├── southbound_flow.py  # 南向资金流
│   │   ├── us_market_sentiment.py # 美股情绪
│   │   ├── alpha_features.py   # Alpha因子
│   │   ├── index_features.py   # 指数特征
│   │   ├── company_events.py  # 公司事件
│   │   ├── base.py            # 基类
│   │   └── combinator.py       # 特征合并
│   ├── models/                 # 模型训练与预测
│   │   ├── base.py            # 模型基类
│   │   ├── trainer.py         # CatBoost 模型
│   │   ├── lgbm_model.py      # LightGBM 模型
│   │   ├── xgboost_model.py   # XGBoost 模型
│   │   └── multi_model.py     # 多模型集成
│   ├── backtest/               # 回测引擎
│   │   ├── engine.py           # 回测引擎核心
│   │   ├── strategies.py      # 策略配置
│   │   └── rule_strategies.py  # 基于规则的策略
│   ├── predictors/             # 预测信号生成
│   │   ├── ensemble_predictor.py  # 集成预测器
│   │   └── technical_signals.py   # 技术信号
│   ├── data_providers/         # 数据源
│   │   ├── base.py            # 基类
│   │   ├── baostock_provider.py
│   │   ├── akshare_provider.py
│   │   ├── yfinance_provider.py
│   │   ├── openbb_provider.py
│   │   ├── tushare_provider.py
│   │   ├── sentiment_provider.py
│   │   └── fetch_stock_data.py
│   └── utils/                  # 工具
│       ├── cache.py            # 特征缓存
│       ├── config.py           # 配置管理
│       ├── stock_info.py       # 股票信息
│       ├── strategy_params.py # 策略参数管理
│       ├── important_dates.py  # 重要日期管理
│       └── perf_monitor.py    # 性能监控
├── stocks.txt                  # 股票列表
├── strategy/                   # 策略参考文档
│   └── references/
└── tests/                      # 测试
```

## 11. 依赖关系

```
核心依赖：
├── catboost        # 机器学习模型
├── lightgbm        # LightGBM 模型
├── xgboost         # XGBoost 模型
├── scikit-learn    # 数据预处理
├── pandas          # 数据处理
├── numpy           # 数值计算
└── scipy           # 科学计算

数据源：
├── baostock        # A股数据
├── akshare         # A/H股数据
├── yfinance        # 港美股数据
├── tushare         # Tushare 数据
└── openbb          # 综合数据平台

工具：
├── pyyaml          # 配置解析
├── python-dotenv   # 环境变量
└── joblib          # 缓存序列化

API 服务：
├── fastapi         # HTTP 服务
└── uvicorn         # ASGI 服务器
```

## 12. 风险控制

### 12.1 最大回撤控制

- **阈值**：20% 最大回撤（可在 `config.yaml` 配置）
- **行为**：当回撤超过阈值，强制平仓并停止交易

### 12.2 仓位管理

基于 ATR 的仓位管理：
- **每笔风险**：1% 资金
- **ATR 乘数**：1.2（止损距离）
- **最小交易量**：高价股（>$100）50股，其他100股
- **最大交易量**：每笔3手（300股）
- **初始仓位**：使用50%资金，保留加仓空间

**计算公式**：`shares = risk_amount / (ATR * atr_multiplier / price)`
