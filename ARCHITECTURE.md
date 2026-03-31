# Stock Trading ML Decision System - 技术架构文档

## 1. 系统概述

本系统是一个基于机器学习的股票交易决策系统，支持A股、港股、美股三大市场。系统采用模块化设计，通过特征工程、机器学习模型训练、多策略回测对比，为用户提供交易决策建议。

### 1.1 核心功能

- **数据获取**：多数据源支持（baostock、akshare、yfinance、openbb）
- **特征工程**：技术指标、基本面、市场情绪等多维度特征
- **模型训练**：基于 CatBoost 的梯度提升模型
- **策略回测**：9种规则策略 + 4种ML策略
- **参数优化**：支持为特定股票寻找最优策略参数
- **交易决策**：综合多策略信号生成最终交易建议

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Scripts Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ decide.py   │  │ backtest.py │  │ explore_params.py           │  │
│  │ 决策入口    │  │ 回测入口    │  │ 参数优化入口                │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                          Core Modules                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │   features   │  │   models     │  │      backtest            │  │
│  │   特征工程   │  │   ML模型     │  │      回测引擎            │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │data_providers│  │   display    │  │       utils              │  │
│  │   数据获取   │  │   结果展示   │  │       工具类             │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                         Storage Layer                                │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    SQLite Databases                             ││
│  │  feature_cache.db          strategy_params.db                   ││
│  │  特征数据缓存              策略参数存储                        ││
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
├── akshare_provider.py  # A股数据（akshare）
├── yfinance_provider.py # 港股/美股数据
├── openbb_provider.py   # OpenBB数据源
├── sentiment_provider.py# 情绪数据
└── fetch_stock_data.py  # 数据获取统一入口
```

**数据源优先级：**
- A股：baostock > akshare > tushare > openbb > yfinance
- 港股：openbb > yfinance > akshare
- 美股：openbb > yfinance

### 3.2 features - 特征工程层

计算和管理各类特征指标。

```
features/
├── base.py         # 特征提取基类
├── technical.py    # 技术指标（MA、RSI、MACD、布林带等）
├── fundamental.py  # 基本面指标（PE、PB、ROE等）
├── market.py       # 市场指标（指数表现、相关性等）
├── industry.py     # 行业指标
├── sentiment.py    # 情绪指标
├── money_flow.py   # 资金流向
└── combinator.py   # 特征组合器
```

**特征类别：**
- 技术特征：~80个（均线、动量、波动率等）
- 基本面特征：~15个（估值、盈利、成长性等）
- 市场特征：~20个（指数、相关性、Beta等）
- 情绪特征：~15个（新闻情绪、社交媒体等）

### 3.3 models - 机器学习层

基于 CatBoost 的分类模型。

```
models/
├── __init__.py
└── trainer.py  # 模型训练和预测
```

**模型特性：**
- 算法：CatBoost（梯度提升决策树）
- 输出：Buy / Hold / Sell 三分类
- 特征自动选择：基于重要性排序
- 类别平衡：SMOTE + 类权重调整

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
| ML策略 | ML Strategy | 纯ML预测 |
| | Hybrid Strategy | ML + 规则混合 |
| | Rolling ML | 滚动训练ML |
| | Rolling Hybrid | 滚动训练混合 |

### 3.5 utils - 工具层

提供缓存、配置、参数管理等基础设施。

```
utils/
├── cache.py            # 特征缓存（SQLite）
├── config.py           # 配置管理（YAML）
├── stock_info.py       # 股票信息解析
└── strategy_params.py  # 策略参数管理
```

### 3.6 display - 展示层

负责结果格式化和输出。

```
display/
├── __init__.py
└── formatters.py  # 输出格式化
```

## 4. 数据流

### 4.1 决策流程 (decide.py)

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
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 3. 训练模型                          │
│    - 划分训练集和评估集               │
│    - 训练 CatBoost 模型              │
│    - 输出特征重要性                   │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 4. 回测对比                          │
│    - 加载优化参数（如有）             │
│    - 运行多种策略回测                 │
│    - 计算各策略表现指标               │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 5. 生成决策                          │
│    - 各策略生成信号                   │
│    - 选择最优策略                     │
│    - 输出最终推荐                     │
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

## 6. 配置管理

### 6.1 config.yaml 结构

```yaml
# 模型配置
model:
  catboost:
    iterations: 500
    depth: 6
    learning_rate: 0.03
  training:
    forward_days: 1      # 预测天数
    threshold: 0.003     # 分类阈值
  strategy:
    min_samples: 20
    confidence_threshold: 0.20

# 回测配置
backtest:
  initial_cash: 100000
  commission: 0.001      # 手续费率
  slippage: 0.001        # 滑点
  max_drawdown_threshold: 0.20

# 数据源配置
data_providers:
  priority:
    a_share: ['baostock', 'akshare', 'tushare']
    hk: ['openbb', 'yfinance', 'akshare']
    us: ['openbb', 'yfinance']
```

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

### 7.3 参数优先级设计

支持三个级别的参数配置：
- **股票级别**：针对特定股票优化的参数
- **市场级别**：针对特定市场的通用参数
- **默认级别**：兜底默认参数

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

## 10. 依赖关系

```
核心依赖：
├── catboost        # 机器学习模型
├── scikit-learn    # 数据预处理
├── pandas          # 数据处理
├── numpy           # 数值计算
└── scipy           # 科学计算

数据源：
├── baostock        # A股数据
├── akshare         # A/H股数据
├── yfinance        # 港美股数据
└── openbb          # 综合数据平台

工具：
├── pyyaml          # 配置解析
├── python-dotenv   # 环境变量
└── joblib          # 缓存序列化
```

## 11. 测试策略

- **单元测试**：`tests/` 目录下的测试用例
- **集成测试**：使用 `scripts/decide.py` 端到端测试
- **回归测试**：历史数据回测验证

运行测试：
```bash
python -m pytest tests/ -v
```
