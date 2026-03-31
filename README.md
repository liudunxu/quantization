# Stock Trading ML Decision System

基于机器学习的股票交易决策系统，支持A股、港股、美股。

## 功能特性

- **多维度特征工程**：基本面、行业数据、大盘信息、技术指标、时序特征
- **CatBoost模型**：高性能梯度提升决策，自动平衡类别权重
- **智能缓存**：基于文件系统的特征缓存，永不过期
- **回测系统**：支持多种策略对比（买入持有、高卖低买、ML策略）
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

### 运行决策

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

### 回测

对比多种策略在指定回测期的表现：

```bash
# 基础用法
python scripts/backtest.py --stock 000001.SZ --days 30

# 自定义参数
python scripts/backtest.py --stock 0700.HK --days 30 --train-days 200 --initial-cash 100000

# 导出结果
python scripts/backtest.py --stock 000001.SZ --days 30 --output results.csv
```

### 参数优化

为特定股票寻找最优的基于规则的策略参数：

```bash
# 优化所有规则策略参数
python scripts/explore_params.py --stock 000001.SZ

# 优化特定策略
python scripts/explore_params.py --stock 000001.SZ --strategies ma_golden_cross box_oscillation

# 使用随机搜索（更快）
python scripts/explore_params.py --stock 000001.SZ --search-method random --random-samples 100

# 优化更长周期
python scripts/explore_params.py --stock 000001.SZ --train-days 365 --backtest-days 90

# 试运行（不保存结果）
python scripts/explore_params.py --stock 000001.SZ --dry-run
```

**参数优先级：** `股票代码 > 市场 > 默认`

优化后的参数会保存到 SQLite 数据库（`cache/strategy_params.db`），后续 `decide.py` 和 `backtest.py` 会自动使用这些优化参数。

## 项目结构

```
quarnt/
├── pyproject.toml     # 项目配置和依赖管理 (uv)
├── requirements.txt   # pip 依赖文件
├── configs/           # 配置文件（config.yaml）
├── cache/             # 特征缓存（parquet格式）
│   ├── feature_cache.db      # 特征缓存数据库
│   └── strategy_params.db    # 策略参数数据库
├── data/              # 原始数据存储
├── models/            # 模型文件
├── src/
│   ├── features/      # 特征工程
│   │   ├── technical.py    # 技术指标
│   │   ├── fundamental.py  # 基本面数据
│   │   ├── market.py       # 市场/大盘数据
│   │   ├── industry.py     # 行业数据
│   │   ├── sentiment.py    # 情绪分析
│   │   └── combinator.py   # 特征合并
│   ├── models/        # 模型训练与预测
│   ├── backtest/      # 回测引擎
│   │   ├── engine.py       # 回测引擎核心
│   │   ├── strategies.py   # 策略配置
│   │   └── rule_strategies.py  # 基于规则的策略
│   └── utils/         # 工具（缓存、配置）
│       ├── cache.py            # 特征缓存
│       ├── config.py           # 配置管理
│       ├── stock_info.py       # 股票信息
│       └── strategy_params.py  # 策略参数管理
├── scripts/           # 入口脚本
│   ├── decide.py          # 交易决策脚本
│   ├── backtest.py        # 回测脚本
│   └── explore_params.py  # 参数优化脚本
└── tests/             # 测试
```

## 策略说明

| 策略 | 说明 |
|------|------|
| Buy & Hold | 买入后一直持有，作为基准 |
| High Sell Low Buy | 逆势策略，价格高位卖出、低位买入 |
| ML Strategy | 基于CatBoost模型预测，支持置信度阈值过滤 |
| Hybrid Strategy | ML策略 + High Sell Low Buy 混合，确认信号 |

### 决策流程

`decide.py` 运行时会：

1. **训练模型** - 使用 train_days 数据训练 CatBoost 模型
2. **回测对比** - 在 backtest_days 窗口内对比 4 种策略表现
3. **输出各策略 Decision** - 每种策略的买入/持有/卖出信号
4. **自动选择最优** - 选择回测收益率最高的策略，以其决策为最终推荐

### 策略选择机制

```python
# 回测窗口内表现最优的策略决定最终 action
if best_strategy_return > others:
    final_decision = best_strategy.action
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
   ```
   train_df = features_df[:-backtest_days]  # 训练用
   eval_df = features_df[-backtest_days:]   # 验证用
   ```

2. **决定回测评估期** - 用这个窗口判断哪种策略最优

3. **权衡：**
   - **大 (60-90天)**：结果更统计显著，但可能错过近期变化
   - **小 (15-20天)**：更贴近当前，但结果不稳定

**推荐：** 默认 30 天是合理折中（一个月交易日约 20-22 天）

### Q: 模型会自动保存吗？

**不会。** 每次运行 `decide.py` 都从头训练新模型，不保存/加载模型文件。

### Q: 如何选择最优参数？

```bash
# 对比不同参数组合
python scripts/decide.py --stock 000001.SZ --train-days 180 --backtest-days 30
python scripts/decide.py --stock 000001.SZ --train-days 365 --backtest-days 30
python scripts/decide.py --stock 000001.SZ --train-days 365 --backtest-days 15
```

回测表现最好的参数组合，就是该股票/市场的最优选择。

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
