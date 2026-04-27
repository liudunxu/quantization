# 简易股票涨跌预测指南

## 快速开始

### 1. 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 2. 基本预测

```bash
# A股预测
python scripts/predict.py --stock 000001.SZ

# 港股预测
python scripts/predict.py --stock 0700.HK

# 美股预测
python scripts/predict.py --stock AAPL
```

## 预测原理

系统通过以下步骤预测股票涨跌：

```
┌─────────────────────────────────────────┐
│  1. 获取历史数据                          │
│     - 价格、成交量、技术指标               │
│     - 最近365个交易日                      │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│  2. 特征工程                              │
│     - 技术指标：MA、RSI、MACD、布林带      │
│     - 动量指标：5/10/20日动量             │
│     - 市场指标：大盘相关性、Beta           │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│  3. 模型训练                              │
│     - CatBoost + LightGBM + XGBoost     │
│     - 多模型集成投票                      │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│  4. 综合预测                              │
│     - ML模型预测 (35%)                   │
│     - 技术分析 (25%)                     │
│     - 动量分析 (15%)                     │
│     - 趋势强度 (10%)                     │
│     - 其他信号 (15%)                     │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│  5. 输出结果                              │
│     - 方向：看涨/看跌                     │
│     - 置信度：0-100%                     │
│     - 概率分布                           │
└─────────────────────────────────────────┘
```

## 输出结果解读

### 预测结果示例

```
方向 (Direction)  : UP ↑          ← 预测上涨
信号 (Signal)     : 看涨 (BULLISH) ← 看涨信号
置信度 (Confidence): 75.0% [高]    ← 置信度较高

ML模型概率分布:
  看涨 (UP)   : 72.3%  ← 模型认为72%概率上涨
  持有 (HOLD) : 18.5%
  看跌 (DOWN) :  9.2%
```

### 关键指标说明

| 指标 | 含义 | 参考价值 |
|------|------|----------|
| **方向** | 预测涨跌方向 | 核心结论 |
| **置信度** | 预测的确定程度 | >60%较可靠 |
| **UP概率** | 上涨概率 | >65%倾向买入 |
| **DOWN概率** | 下跌概率 | >65%倾向卖出 |

## 常用参数

### 基础参数

```bash
# 设置训练天数（更多数据=更稳定）
python scripts/predict.py --stock 000001.SZ --train-days 500

# 设置涨跌阈值（越低信号越多）
python scripts/predict.py --stock 000001.SZ --threshold 0.005

# 排除极端波动日期
python scripts/predict.py --stock 000001.SZ --exclude-dates
```

### 输出格式

```bash
# JSON格式（便于程序处理）
python scripts/predict.py --stock 000001.SZ --output json

# CSV格式（便于批量分析）
python scripts/predict.py --stock 000001.SZ --output csv
```

### 模型选择

```bash
# 使用多模型集成（默认，更准确）
python scripts/predict.py --stock 000001.SZ

# 使用单模型（更快）
python scripts/predict.py --stock 000001.SZ --single-model
```

## 使用建议

### 1. 信号可靠性判断

| 置信度 | 可靠性 | 建议 |
|--------|--------|------|
| >70% | 高 | 可作为主要参考 |
| 60-70% | 中 | 结合其他分析 |
| <60% | 低 | 仅作参考 |

### 2. 最佳实践

- **多股票对比**：同时预测多只股票，选择信号最强的
- **结合基本面**：ML预测+财务分析+行业分析
- **设置止损**：即使预测看涨，也要设置止损位
- **分散投资**：不要把所有资金押在单只股票

### 3. 避免的误区

- ❌ 盲目相信单一预测结果
- ❌ 忽视置信度，只看方向
- ❌ 不设止损，全仓操作
- ❌ 频繁交易，追涨杀跌

## 快速命令参考

```bash
# A股 - 平安银行
python scripts/predict.py --stock 000001.SZ

# A股 - 贵州茅台
python scripts/predict.py --stock 600519.SH

# 港股 - 腾讯
python scripts/predict.py --stock 0700.HK

# 美股 - 苹果
python scripts/predict.py --stock AAPL

# 美股 - 英伟达
python scripts/predict.py --stock NVDA

# 指数 - 上证指数
python scripts/predict.py --index 000001

# 指数 - 沪深300
python scripts/predict.py --index 000300
```

## API调用

### 启动服务

```bash
python scripts/predict.py --serve --port 8000
```

### 调用预测

```bash
# GET请求
curl "http://localhost:8000/predict?stock=000001.SZ"

# POST请求
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"stock": "000001.SZ"}'
```

### 返回结果

```json
{
  "stock_code": "000001.SZ",
  "direction": "UP",
  "signal": "BULLISH",
  "confidence": 0.75,
  "probabilities": {
    "up": 0.723,
    "hold": 0.185,
    "down": 0.092
  },
  "model_accuracy": 0.685
}
```

## 注意事项

1. **预测不等于保证**：股市有风险，预测仅供参考
2. **历史表现不代表未来**：过去准确率不保证未来
3. **市场突发事件**：政策、新闻等突发事件无法预测
4. **流动性风险**：小盘股可能有流动性问题
5. **交易成本**：考虑手续费、滑点等成本

## 常见问题

### Q: 预测准确率有多少？

A: 模型评估准确率通常在55-70%之间，具体取决于市场和股票。

### Q: 应该完全按照预测操作吗？

A: 不建议。预测只是参考，需要结合自己的判断和风险承受能力。

### Q: 多久更新一次预测？

A: 建议每天收盘后更新，获取最新数据。

### Q: 哪些股票预测效果好？

A: 大盘蓝筹股、流动性好的股票效果较好。小盘股、妖股效果较差。
