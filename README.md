# Stock Trading ML Decision System

基于机器学习的股票交易决策系统，支持A股、港股、美股。

## 功能特性

- **多维度特征工程**：基本面、行业数据、大盘信息、技术指标、时序特征
- **CatBoost模型**：高性能梯度提升决策，自动平衡类别权重
- **智能缓存**：基于文件系统的特征缓存，永不过期
- **回测系统**：支持多种策略对比（买入持有、高卖低买、ML策略）
- **跨市场支持**：A股（000001.SZ）、港股（0700.HK）、美股（AAPL）

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
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

## 项目结构

```
quarnt/
├── configs/           # 配置文件（config.yaml）
├── cache/             # 特征缓存（parquet格式）
├── data/              # 原始数据存储
├── models/            # 模型文件
├── src/
│   ├── features/      # 特征工程
│   │   ├── technical.py    # 技术指标
│   │   ├── fundamental.py  # 基本面数据
│   │   ├── market.py       # 市场/大盘数据
│   │   ├── industry.py     # 行业数据
│   │   └── combinator.py   # 特征合并
│   ├── models/        # 模型训练与预测
│   ├── backtest/      # 回测引擎
│   └── utils/         # 工具（缓存、配置）
├── scripts/           # 入口脚本
│   ├── decide.py      # 交易决策脚本
│   └── backtest.py    # 回测脚本
└── tests/             # 测试
```

## 策略说明

| 策略 | 说明 |
|------|------|
| ML策略 | 基于CatBoost模型预测，支持置信度阈值过滤 |
| 买入持有 | 买入后一直持有，作为基准 |
| 高卖低买 | 逆势策略，价格高位卖出、低位买入 |

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
