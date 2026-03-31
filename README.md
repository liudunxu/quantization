# Stock Trading ML Decision System

基于机器学习的股票交易决策系统，支持A股、港股、美股。

## 功能特性

- **多维度特征工程**：基本面、行业数据、大盘信息、技术指标、时序特征、市场情绪
- **CatBoost模型**：高性能梯度提升决策，自动平衡类别权重
- **智能缓存**：基于 SQLite 的特征缓存和参数存储
- **回测系统**：支持多种策略对比（9种规则策略 + 4种ML策略）
- **参数优化**：支持为特定股票寻找最优策略参数
- **跨市场支持**：A股（000001.SZ）、港股（0700.HK）、美股（AAPL）

## 快速开始

### 方式一：使用 uv（推荐）

[uv](https://github.com/astral-sh/uv) 是一个极快的 Python 包管理器，推荐使用。

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv sync

# 运行决策脚本
uv run python scripts/decide.py --stock 000001.SZ

# 运行回测脚本
uv run python scripts/backtest.py --stock 000001.SZ --days 30
```

### 方式二：使用 pip

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 运行脚本
python scripts/decide.py --stock 000001.SZ
```

### 安装可选依赖

```bash
# 使用 uv
uv sync --extra openbb      # 安装 OpenBB 数据源
uv sync --extra sentiment   # 安装情绪分析依赖 (SnowNLP)
uv sync --extra dev         # 安装开发依赖 (pytest, ruff)

# 使用 pip
pip install openbb           # 安装 OpenBB 数据源
pip install snownlp          # 安装情绪分析依赖
```

## 使用指南

### 1. 运行决策

输出交易决策（买入/持有/卖出）、置信度、概率分布、回测对比和特征重要性：

```bash
# A股
python scripts/decide.py --stock 000001.SZ

# 港股
python scripts/decide.py --stock 0700.HK

# 美股
python scripts/decide.py --stock AAPL

# 自定义参数
python scripts/decide.py --stock 0700.HK --train-days 200 --backtest-days 30 --refresh
```

### 2. 回测对比

对比多种策略在指定回测期的表现：

```bash
# 基础用法
python scripts/backtest.py --stock 000001.SZ --days 30

# 自定义参数
python scripts/backtest.py --stock 0700.HK --days 30 --train-days 200 --initial-cash 100000

# 导出结果
python scripts/backtest.py --stock 000001.SZ --days 30 --output results.csv
```

### 3. 参数优化

为特定股票寻找最优的基于规则的策略参数，优化后的参数会自动应用到后续的决策和回测中。

```bash
# 优化所有规则策略参数
python scripts/explore_params.py --stock 000001.SZ

# 优化特定策略
python scripts/explore_params.py --stock 000001.SZ --strategies ma_golden_cross box_oscillation

# 使用随机搜索（更快）
python scripts/explore_params.py --stock 000001.SZ --search-method random --random-samples 100

# 优化更长周期
python scripts/explore_params.py --stock 000001.SZ --train-days 365 --backtest-days 90

# 指定优化指标
python scripts/explore_params.py --stock 000001.SZ --metric sharpe_ratio

# 试运行（不保存结果）
python scripts/explore_params.py --stock 000001.SZ --dry-run
```

**可优化的策略：**
- `ma_golden_cross` - 均线金叉策略
- `bull_trend` - 趋势跟随策略
- `shrink_pullback` - 缩量回调策略
- `bottom_volume` - 底部放量策略
- `box_oscillation` - 箱体震荡策略
- `volume_breakout` - 放量突破策略
- `macd_divergence` - MACD背驰策略
- `emotion_cycle` - 情绪周期策略

## 参数优先级系统

系统支持三级参数优先级：**股票代码 > 市场 > 默认**

```
┌─────────────────────────────────────────────────────┐
│  股票特定参数 (9626.HK)                             │
│  ma_golden_cross: {fast_ma: 3, slow_ma: 30}         │
└─────────────────────┬───────────────────────────────┘
                      │ 如果没有
                      ▼
┌─────────────────────────────────────────────────────┐
│  市场级别参数 (hk)                                  │
│  ma_golden_cross: {fast_ma: 5, slow_ma: 10}         │
└─────────────────────┬───────────────────────────────┘
                      │ 如果没有
                      ▼
┌─────────────────────────────────────────────────────┐
│  默认参数                                            │
│  ma_golden_cross: {fast_ma: 5, slow_ma: 10}         │
└─────────────────────┬───────────────────────────────┘
                      │ 如果还没有
                      ▼
┌─────────────────────────────────────────────────────┐
│  硬编码默认值 (代码中的 DEFAULT_RULE_PARAMS)          │
└─────────────────────────────────────────────────────┘
```

### 参数管理 API

```python
from src.utils import get_param_manager

pm = get_param_manager()

# 获取参数（自动按优先级降级）
params = pm.get_strategy_params('ma_golden_cross', 'hk', '9626.HK')

# 设置股票特定参数
pm.set_strategy_params(
    strategy_name='ma_golden_cross',
    params={'fast_ma': 3, 'slow_ma': 30, 'volume_ratio': 2.0},
    market='hk',
    stock_code='9626.HK',
    description='Optimized for 9626.HK'
)

# 设置市场级别参数
pm.set_strategy_params(
    strategy_name='ma_golden_cross',
    params={'fast_ma': 5, 'slow_ma': 10},
    market='hk'
)

# 列出所有参数
pm.list_params(stock_code='9626.HK')

# 删除参数
pm.delete_strategy_params('ma_golden_cross', 'hk', '9626.HK')
```

## 策略说明

### 规则策略

| 策略 | 说明 | 核心参数 |
|------|------|----------|
| Buy & Hold | 买入后一直持有，作为基准 | - |
| High Sell Low Buy | 逆势策略，价格高位卖出、低位买入 | `lookback`, `threshold` |
| MA Golden Cross | 均线金叉策略，金叉买入死叉卖出 | `fast_ma`, `slow_ma`, `volume_ratio` |
| Bull Trend | 趋势跟随策略，多头排列时买入 | `ma5_period`, `ma10_period`, `ma20_period` |
| Shrink Pullback | 缩量回调策略，回调缩量时买入 | `lookback`, `volume_shrink` |
| Bottom Volume | 底部放量策略，下跌后放量买入 | `drop_threshold`, `volume_multiplier` |
| Box Oscillation | 箱体震荡策略，箱底买入箱顶卖出 | `lookback`, `support_margin` |
| Volume Breakout | 放量突破策略，突破高点买入 | `lookback`, `volume_multiplier` |
| MACD Divergence | MACD背驰策略，底背驰买入 | `lookback` |
| Emotion Cycle | 情绪周期策略，超卖买入超买卖出 | `rsi_oversold`, `rsi_overbought` |

### ML策略

| 策略 | 说明 |
|------|------|
| ML Strategy | 基于CatBoost模型预测，支持置信度阈值过滤 |
| Hybrid Strategy | ML策略 + High Sell Low Buy 混合，确认信号 |
| Rolling ML | 滚动训练的ML策略，适应市场变化 |
| Rolling Hybrid | 滚动训练的混合策略 |

### 优化指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| `total_return` | 总收益率 | 最终权益 / 初始资金 - 1 |
| `sharpe_ratio` | 夏普比率 | (平均收益 - 无风险利率) / 波动率 |
| `win_rate` | 胜率 | 盈利次数 / 总交易次数 |
| `composite` | 复合评分 | 0.35×收益 + 0.30×夏普 + 0.20×胜率 + 0.15×回撤 |

## 项目结构

```
quarnt/
├── pyproject.toml              # 项目配置和依赖管理 (uv)
├── requirements.txt            # pip 依赖文件
├── configs/
│   └── config.yaml             # 全局配置
├── cache/
│   ├── feature_cache.db        # 特征缓存数据库
│   └── strategy_params.db      # 策略参数数据库
├── data/                       # 原始数据存储
├── models/                     # 模型文件
├── src/
│   ├── features/               # 特征工程
│   │   ├── technical.py        # 技术指标
│   │   ├── fundamental.py      # 基本面数据
│   │   ├── market.py           # 市场/大盘数据
│   │   ├── industry.py         # 行业数据
│   │   ├── sentiment.py        # 情绪分析
│   │   └── combinator.py       # 特征合并
│   ├── models/                 # 模型训练与预测
│   │   └── trainer.py          # CatBoost 模型
│   ├── backtest/               # 回测引擎
│   │   ├── engine.py           # 回测引擎核心
│   │   ├── strategies.py       # 策略配置
│   │   └── rule_strategies.py  # 基于规则的策略
│   └── utils/                  # 工具
│       ├── cache.py            # 特征缓存
│       ├── config.py           # 配置管理
│       ├── stock_info.py       # 股票信息
│       └── strategy_params.py  # 策略参数管理
├── scripts/                    # 入口脚本
│   ├── decide.py               # 交易决策脚本
│   ├── backtest.py             # 回测脚本
│   └── explore_params.py       # 参数优化脚本
└── tests/                      # 测试
```

## 决策流程

`decide.py` 运行流程：

```
┌─────────────────────────────────────────────────────────┐
│  1. 获取数据 & 特征工程                                   │
│     - 获取股票历史数据                                    │
│     - 计算技术指标、基本面、市场、情绪等特征               │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  2. 训练 ML 模型                                         │
│     - 使用 train_days 数据训练 CatBoost 模型              │
│     - 输出特征重要性 TOP 10                               │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  3. 运行多策略回测                                        │
│     - 加载优化参数（如有）                                │
│     - 对比 13 种策略在 backtest_days 窗口内的表现          │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  4. 输出各策略决策                                        │
│     - 每种策略的买入/持有/卖出信号                        │
│     - 置信度、回测收益、夏普比率                          │
└─────────────────────┬───────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  5. 最终推荐                                              │
│     - 选择回测表现最优的策略                              │
│     - 输出最终决策: BUY / HOLD / SELL                    │
└─────────────────────────────────────────────────────────┘
```

## 常见问题 (FAQ)

### Q: `--train-days` 参数是越大越好还是越小越好？

**没有绝对最优值，取决于市场特征：**

| train_days | 优点 | 缺点 | 推荐场景 |
|------------|------|------|----------|
| 越大 (365-730) | 样本多，模型稳定，减少过拟合 | 可能包含过时模式 | 趋势稳定的市场 |
| 越小 (90-180) | 更贴近近期市场状态 | 样本少，容易过拟合 | 风格切换快的市场 |

**建议：** 对同一只股票用不同值跑回测对比，选择回测表现最好的。

### Q: `--backtest-days` 参数如何影响决策？

**作用：**

1. **决定训练集大小** - `train_days` 固定时，`backtest_days` 越大则训练集越小
2. **决定回测评估期** - 用这个窗口判断哪种策略最优

**权衡：**
- **大 (60-90天)**：结果更统计显著，但可能错过近期变化
- **小 (15-20天)**：更贴近当前，但结果不稳定

**推荐：** 默认 30 天是合理折中（一个月交易日约 20-22 天）

### Q: 模型会自动保存吗？

**不会。** 每次运行 `decide.py` 都从头训练新模型，不保存/加载模型文件。

### Q: 参数优化结果如何生效？

参数优化后会保存到 `cache/strategy_params.db`，后续运行 `decide.py` 或 `backtest.py` 时会自动读取：

```
优先级: 股票特定参数 > 市场级别参数 > 默认参数 > 硬编码默认值
```

### Q: 如何验证参数优化效果？

```bash
# 1. 先运行优化（保存参数）
python scripts/explore_params.py --stock 000001.SZ --strategies ma_golden_cross

# 2. 运行回测（会自动使用优化后的参数）
python scripts/backtest.py --stock 000001.SZ --days 30

# 3. 对比优化前后的表现
```

## 缓存管理

```python
from src.utils.cache import FeatureCache

cache = FeatureCache()

# 查看缓存状态
cache.get_cache_info('000001.SZ')

# 刷新单个股票缓存
cache.refresh('000001.SZ')

# 删除单个股票缓存
cache.delete('000001.SZ')

# 清除所有缓存
cache.clear_all()
```

## 特征列表

### 技术特征
- **移动平均线**：MA5、MA10、MA20、MA60 及价格比率
- **RSI**：相对强弱指数（14日）
- **MACD**：快线、慢线、信号线、柱状图
- **布林带**：上轨、中轨、下轨、带宽、位置
- **ATR**：平均真实波幅（14日）
- **随机振荡器**：Stoch_K、Stoch_D（14日）
- **ADX**：平均方向指数（14日）
- **MFI**：资金流量指数（14日）
- **CCI**：顺势指标（14日）
- **成交量**：MA5、MA20、量比、成交量变化率

### 动量特征
- 5/10/20日动量
- 2/3日短期收益
- 动量加速度（5日动量 - 10日动量）

### 风险调整特征
- 波动率（20日年化）
- 类夏普比率（5日收益/波动率）
- 变异系数
- 最大回撤（20日窗口）

### 基本面特征
- PE、PB、ROE、资产负债率
- 营收增长率、净利润增长率
- 毛利率、净利率

### 市场特征
- 大盘指数表现（上证指数/恒生指数/标普500）
- 资金流向
- 市场情绪指标

## 回测指标

| 指标 | 说明 |
|------|------|
| Total Return | 总收益率 |
| Buy & Hold | 买入持有收益率 |
| vs Benchmark | 相对基准超额收益 |
| Sharpe Ratio | 夏普比率（年化） |
| Max Drawdown | 最大回撤 |
| Win Rate | 胜率 |
| Total Trades | 总交易次数 |

## 配置

编辑 `configs/config.yaml` 修改：

```yaml
model:
  catboost:
    iterations: 500
    depth: 6
    learning_rate: 0.03
  training:
    forward_days: 1      # 预测天数（越短越激进）
    threshold: 0.003     # 阈值（越低信号越多）
  strategy:
    min_samples: 20      # 最少样本数
    confidence_threshold: 0.20  # 置信度阈值

backtest:
  initial_cash: 100000
  commission: 0.001      # 手续费
  slippage: 0.001        # 滑点
```

## License

MIT
