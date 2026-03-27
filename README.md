# Stock Trading ML Decision System

基于机器学习的股票交易决策系统，支持A股、港股、美股。

## 功能特性

- **多维度特征工程**：基本面、行业数据、大盘信息、时序特征
- **CatBoost模型**：高性能梯度提升决策
- **智能缓存**：基于文件系统的特征缓存，永不过期
- **回测系统**：支持多种策略对比（买入持有、高卖低买、ML策略）
- **跨市场支持**：A股、港股、美股

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行决策

```bash
# A股
python scripts/decide.py --stock 000001.SZ

# 港股
python scripts/decide.py --stock 0700.HK

# 美股
python scripts/decide.py --stock AAPL
```

### 回测

```bash
python scripts/backtest.py --stock 000001.SZ --days 30
```

## 项目结构

```
quarnt/
├── configs/           # 配置文件
├── cache/             # 特征缓存
├── data/              # 数据存储
├── models/            # 模型文件
├── src/
│   ├── features/      # 特征工程
│   ├── models/        # 模型
│   ├── backtest/      # 回测
│   └── utils/         # 工具
├── scripts/           # 脚本
└── tests/             # 测试
```

## 策略说明

| 策略 | 说明 |
|------|------|
| ML策略 | 基于CatBoost模型预测 |
| 买入持有 | 买入后一直持有 |
| 高卖低买 | 逆势交易策略 |

## 缓存管理

```python
from src.utils.cache import FeatureCache

cache = FeatureCache()
cache.refresh('000001.SZ')  # 刷新缓存
cache.delete('000001.SZ')   # 删除缓存
cache.clear_all()           # 清除所有缓存
```

## 特征列表

### 基本面特征
- PE、PB、ROE、资产负债率
- 营收增长率、净利润增长率
- 毛利率、净利率

### 技术特征
- 移动平均线（MA5、MA10、MA20、MA60）
- RSI、MACD、布林带
- 成交量变化率

### 市场特征
- 大盘指数表现
- 资金流向
- 市场情绪指标

## 回测指标

- 总收益率
- 夏普比率
- 最大回撤
- 胜率
- 交易次数

## License

MIT
