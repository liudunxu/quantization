#!/usr/bin/env python3
"""
差异化预测策略配置

针对不同市场、行业、股票类型采用不同的预测参数和策略。

Usage:
    # 查看所有策略配置
    python scripts/prediction_strategies.py --list

    # 获取特定股票的策略
    python scripts/prediction_strategies.py --stock 000001.SZ

    # 应用策略进行预测
    python scripts/prediction_strategies.py --stock 000001.SZ --apply
"""

import argparse
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PredictionStrategy:
    """预测策略配置."""
    name: str
    description: str
    # 模型参数
    iterations: int = 300
    depth: int = 4
    learning_rate: float = 0.03
    n_estimators: int = 3
    # 预测参数
    forward_days: int = 1
    threshold: float = 0.008
    # 信号权重
    ml_weight: float = 0.35
    technical_weight: float = 0.25
    momentum_weight: float = 0.15
    trend_weight: float = 0.10
    # 标签权重
    trend_label_weight: float = 0.30
    momentum_label_weight: float = 0.30
    market_label_weight: float = 0.20
    # 置信度阈值
    confidence_threshold: float = 0.60
    # 特殊设置
    use_composite_labels: bool = True
    exclude_dates: bool = False


# ============================================================
# 市场级别策略
# ============================================================

MARKET_STRATEGIES = {
    "cn": PredictionStrategy(
        name="A股默认策略",
        description="适合A股市场的保守策略，阈值较低以捕捉更多信号",
        iterations=300,
        depth=4,
        learning_rate=0.03,
        n_estimators=3,
        forward_days=1,
        threshold=0.005,  # 基于回测优化: 低阈值效果更好
        ml_weight=0.30,
        technical_weight=0.30,  # A股技术分析更有效
        momentum_weight=0.20,
        trend_weight=0.10,
        trend_label_weight=0.25,
        momentum_label_weight=0.35,
        market_label_weight=0.25,
        confidence_threshold=0.55,
        exclude_dates=True,  # A股有涨跌停，需要排除极端日期
    ),
    "hk": PredictionStrategy(
        name="港股默认策略",
        description="适合港股市场的平衡策略",
        iterations=300,
        depth=4,
        learning_rate=0.03,
        n_estimators=3,
        forward_days=1,
        threshold=0.005,  # 基于回测优化: 低阈值效果更好
        ml_weight=0.35,
        technical_weight=0.25,
        momentum_weight=0.15,
        trend_weight=0.15,
        trend_label_weight=0.30,
        momentum_label_weight=0.30,
        market_label_weight=0.20,
        confidence_threshold=0.55,
    ),
    "us": PredictionStrategy(
        name="美股默认策略",
        description="适合美股市场的积极策略，趋势跟随",
        iterations=400,
        depth=5,
        learning_rate=0.025,
        n_estimators=4,
        forward_days=1,
        threshold=0.005,  # 基于回测优化: 低阈值效果更好
        ml_weight=0.35,
        technical_weight=0.20,
        momentum_weight=0.15,
        trend_weight=0.20,  # 美股趋势性强
        trend_label_weight=0.35,
        momentum_label_weight=0.25,
        market_label_weight=0.20,
        confidence_threshold=0.55,
    ),
}


# ============================================================
# 行业级别策略
# ============================================================

INDUSTRY_STRATEGIES = {
    # 科技/半导体 - 高波动，动量重要
    "semiconductor": PredictionStrategy(
        name="半导体行业策略",
        description="高波动行业，强调动量和技术分析",
        iterations=400,
        depth=5,
        learning_rate=0.025,
        threshold=0.010,
        ml_weight=0.30,
        technical_weight=0.25,
        momentum_weight=0.25,  # 半导体动量重要
        trend_weight=0.10,
        trend_label_weight=0.25,
        momentum_label_weight=0.40,  # 动量权重高
        market_label_weight=0.15,
        confidence_threshold=0.58,
    ),
    # 互联网 - 中等波动，趋势跟随
    "internet": PredictionStrategy(
        name="互联网行业策略",
        description="中等波动，注重趋势确认",
        iterations=350,
        depth=5,
        learning_rate=0.025,
        threshold=0.009,
        ml_weight=0.35,
        technical_weight=0.25,
        momentum_weight=0.15,
        trend_weight=0.15,
        trend_label_weight=0.35,
        momentum_label_weight=0.25,
        market_label_weight=0.20,
        confidence_threshold=0.60,
    ),
    # 银行 - 低波动，保守策略
    "bank": PredictionStrategy(
        name="银行行业策略",
        description="低波动行业，保守策略",
        iterations=300,
        depth=4,
        learning_rate=0.03,
        threshold=0.005,  # 银行波动小，低阈值
        ml_weight=0.35,
        technical_weight=0.30,
        momentum_weight=0.10,  # 银行动量弱
        trend_weight=0.15,
        trend_label_weight=0.30,
        momentum_label_weight=0.20,
        market_label_weight=0.30,  # 银行与大盘相关性强
        confidence_threshold=0.55,
    ),
    # 汽车 - 高波动，事件驱动
    "auto": PredictionStrategy(
        name="汽车行业策略",
        description="高波动行业，事件驱动明显",
        iterations=400,
        depth=5,
        learning_rate=0.025,
        threshold=0.011,
        ml_weight=0.30,
        technical_weight=0.25,
        momentum_weight=0.20,
        trend_weight=0.15,
        trend_label_weight=0.30,
        momentum_label_weight=0.30,
        market_label_weight=0.20,
        confidence_threshold=0.58,
        exclude_dates=True,
    ),
    # 消费 - 稳定，趋势跟随
    "consumer": PredictionStrategy(
        name="消费行业策略",
        description="相对稳定，注重趋势",
        iterations=350,
        depth=4,
        learning_rate=0.03,
        threshold=0.007,
        ml_weight=0.35,
        technical_weight=0.25,
        momentum_weight=0.15,
        trend_weight=0.15,
        trend_label_weight=0.35,
        momentum_label_weight=0.25,
        market_label_weight=0.20,
        confidence_threshold=0.60,
    ),
    # 有色金属 - 高波动，周期性强
    "metals": PredictionStrategy(
        name="有色金属行业策略",
        description="周期性强，波动大",
        iterations=400,
        depth=5,
        learning_rate=0.025,
        threshold=0.012,  # 高波动
        ml_weight=0.30,
        technical_weight=0.25,
        momentum_weight=0.20,
        trend_weight=0.15,
        trend_label_weight=0.30,
        momentum_label_weight=0.30,
        market_label_weight=0.25,
        confidence_threshold=0.58,
    ),
}


# ============================================================
# 特殊股票策略
# ============================================================

SPECIAL_STRATEGIES = {
    # 大盘指数
    "index": PredictionStrategy(
        name="指数预测策略",
        description="大盘指数预测，更注重市场整体",
        iterations=500,
        depth=5,
        learning_rate=0.02,
        threshold=0.005,  # 指数波动小
        ml_weight=0.35,
        technical_weight=0.25,
        momentum_weight=0.15,
        trend_weight=0.15,
        trend_label_weight=0.30,
        momentum_label_weight=0.25,
        market_label_weight=0.30,  # 指数本身就是市场
        confidence_threshold=0.55,
    ),
    # 高价股
    "expensive": PredictionStrategy(
        name="高价股策略",
        description="价格>100的股票，更保守",
        iterations=400,
        depth=5,
        learning_rate=0.02,
        threshold=0.008,
        ml_weight=0.35,
        technical_weight=0.25,
        momentum_weight=0.15,
        trend_weight=0.15,
        confidence_threshold=0.65,  # 高价股需要更高置信度
    ),
    # 中概股
    "china_adr": PredictionStrategy(
        name="中概股策略",
        description="在美上市的中国公司，受双重影响",
        iterations=400,
        depth=5,
        learning_rate=0.025,
        threshold=0.012,
        ml_weight=0.30,
        technical_weight=0.25,
        momentum_weight=0.20,
        trend_weight=0.15,
        trend_label_weight=0.30,
        momentum_label_weight=0.30,
        market_label_weight=0.25,
        confidence_threshold=0.60,
    ),
}


# ============================================================
# 股票到行业的映射
# ============================================================

STOCK_INDUSTRY_MAP = {
    # A股 - 半导体
    "600111.SH": "semiconductor", "603986.SH": "semiconductor",
    "002156.SZ": "semiconductor",
    # A股 - 电子
    "601138.SH": "electronics", "002475.SZ": "electronics",
    # A股 - 银行
    "600036.SH": "bank", "601288.SH": "bank",
    # A股 - 消费
    "000333.SZ": "consumer", "600887.SH": "consumer",
    # A股 - 电力/能源
    "600900.SH": "utility",
    # A股 - 有色金属
    "600362.SH": "metals", "000807.SZ": "metals", "601020.SH": "metals",
    # A股 - 电气设备
    "603191.SH": "electronics",
    # 港股 - 互联网
    "9988.HK": "internet", "0700.HK": "internet", "1810.HK": "internet",
    "3690.HK": "internet", "1024.HK": "internet", "9626.HK": "internet",
    "1357.HK": "internet",
    # 港股 - 半导体
    "0981.HK": "semiconductor", "1347.HK": "semiconductor",
    # 港股 - 汽车
    "9880.HK": "auto", "9868.HK": "auto",
    # 港股 - 消费
    "2020.HK": "consumer", "2097.HK": "consumer",
    # 港股 - 旅游
    "9961.HK": "consumer",
    # 美股 - 科技
    "AAPL": "electronics", "MSFT": "software", "GOOGL": "internet",
    "AMZN": "ecommerce", "META": "internet", "TSLA": "auto",
    # 美股 - 半导体
    "NVDA": "semiconductor", "AMD": "semiconductor", "INTC": "semiconductor",
    "QCOM": "semiconductor", "AVGO": "semiconductor", "MU": "semiconductor",
    "ASML": "semiconductor_equipment", "TSM": "semiconductor",
    # 美股 - 中概股
    "PDD": "china_adr", "BABA": "china_adr", "JD": "china_adr", "BIDU": "china_adr",
}

# 指数列表
INDEX_CODES = {
    "000001.SH", "399001.SZ", "399006.SZ", "HSTECH.HK",
}


def get_strategy_for_stock(stock_code: str) -> PredictionStrategy:
    """获取股票的最佳预测策略.
    
    优先级: 特殊股票 > 行业 > 市场 > 默认
    """
    # 1. 检查是否是指数
    if stock_code in INDEX_CODES:
        return SPECIAL_STRATEGIES["index"]
    
    # 2. 检查是否是中概股
    if stock_code in ["PDD", "BABA", "JD", "BIDU"]:
        return SPECIAL_STRATEGIES["china_adr"]
    
    # 3. 获取行业策略
    industry = STOCK_INDUSTRY_MAP.get(stock_code)
    if industry and industry in INDUSTRY_STRATEGIES:
        return INDUSTRY_STRATEGIES[industry]
    
    # 4. 获取市场策略
    if stock_code.endswith(".SH") or stock_code.endswith(".SZ"):
        return MARKET_STRATEGIES["cn"]
    elif stock_code.endswith(".HK"):
        return MARKET_STRATEGIES["hk"]
    else:
        return MARKET_STRATEGIES["us"]


def get_all_strategies() -> Dict[str, List[Dict]]:
    """获取所有策略配置."""
    result = {
        "market_strategies": {},
        "industry_strategies": {},
        "special_strategies": {},
    }
    
    for name, strategy in MARKET_STRATEGIES.items():
        result["market_strategies"][name] = {
            "name": strategy.name,
            "description": strategy.description,
            "threshold": strategy.threshold,
            "confidence_threshold": strategy.confidence_threshold,
        }
    
    for name, strategy in INDUSTRY_STRATEGIES.items():
        result["industry_strategies"][name] = {
            "name": strategy.name,
            "description": strategy.description,
            "threshold": strategy.threshold,
            "momentum_weight": strategy.momentum_weight,
        }
    
    for name, strategy in SPECIAL_STRATEGIES.items():
        result["special_strategies"][name] = {
            "name": strategy.name,
            "description": strategy.description,
        }
    
    return result


def strategy_to_dict(strategy: PredictionStrategy) -> Dict:
    """将策略转换为字典."""
    return {
        "name": strategy.name,
        "description": strategy.description,
        "iterations": strategy.iterations,
        "depth": strategy.depth,
        "learning_rate": strategy.learning_rate,
        "n_estimators": strategy.n_estimators,
        "forward_days": strategy.forward_days,
        "threshold": strategy.threshold,
        "ml_weight": strategy.ml_weight,
        "technical_weight": strategy.technical_weight,
        "momentum_weight": strategy.momentum_weight,
        "trend_weight": strategy.trend_weight,
        "trend_label_weight": strategy.trend_label_weight,
        "momentum_label_weight": strategy.momentum_label_weight,
        "market_label_weight": strategy.market_label_weight,
        "confidence_threshold": strategy.confidence_threshold,
        "use_composite_labels": strategy.use_composite_labels,
        "exclude_dates": strategy.exclude_dates,
    }


def main():
    parser = argparse.ArgumentParser(description="差异化预测策略配置")
    parser.add_argument("--list", action="store_true", help="查看所有策略")
    parser.add_argument("--stock", type=str, help="获取特定股票的策略")
    parser.add_argument("--apply", action="store_true", help="应用策略进行预测")
    parser.add_argument("--market", choices=["cn", "hk", "us"], help="获取市场策略")
    parser.add_argument("--industry", type=str, help="获取行业策略")
    args = parser.parse_args()
    
    if args.list:
        print("\n" + "=" * 60)
        print("  差异化预测策略配置")
        print("=" * 60)
        
        strategies = get_all_strategies()
        
        print("\n  市场策略:")
        for name, info in strategies["market_strategies"].items():
            print(f"    {name}: {info['name']}")
            print(f"      阈值: {info['threshold']}, 置信度: {info['confidence_threshold']}")
        
        print("\n  行业策略:")
        for name, info in strategies["industry_strategies"].items():
            print(f"    {name}: {info['name']}")
            print(f"      阈值: {info['threshold']}, 动量权重: {info['momentum_weight']}")
        
        print("\n  特殊策略:")
        for name, info in strategies["special_strategies"].items():
            print(f"    {name}: {info['name']}")
        
        return
    
    if args.stock:
        strategy = get_strategy_for_stock(args.stock)
        print(f"\n  股票: {args.stock}")
        print(f"  策略: {strategy.name}")
        print(f"  描述: {strategy.description}")
        print(f"\n  参数:")
        print(f"    阈值: {strategy.threshold}")
        print(f"    置信度阈值: {strategy.confidence_threshold}")
        print(f"    ML权重: {strategy.ml_weight}")
        print(f"    技术权重: {strategy.technical_weight}")
        print(f"    动量权重: {strategy.momentum_weight}")
        print(f"    趋势权重: {strategy.trend_weight}")
        
        if args.apply:
            print(f"\n  应用策略进行预测...")
            # 这里可以调用predict.py
            import subprocess
            cmd = [
                "python", "scripts/predict.py",
                "--stock", args.stock,
                "--threshold", str(strategy.threshold),
                "--ml-weight", str(strategy.ml_weight),
                "--technical-weight", str(strategy.technical_weight),
                "--momentum-weight", str(strategy.momentum_weight),
            ]
            if strategy.exclude_dates:
                cmd.append("--exclude-dates")
            subprocess.run(cmd)
        return
    
    if args.market:
        strategy = MARKET_STRATEGIES.get(args.market)
        if strategy:
            print(f"\n  市场: {args.market}")
            print(f"  策略: {strategy.name}")
            print(f"  描述: {strategy.description}")
            print(f"\n  配置:")
            print(json.dumps(strategy_to_dict(strategy), indent=2, ensure_ascii=False))
        return
    
    if args.industry:
        strategy = INDUSTRY_STRATEGIES.get(args.industry)
        if strategy:
            print(f"\n  行业: {args.industry}")
            print(f"  策略: {strategy.name}")
            print(f"  描述: {strategy.description}")
            print(f"\n  配置:")
            print(json.dumps(strategy_to_dict(strategy), indent=2, ensure_ascii=False))
        else:
            print(f"  未找到行业 '{args.industry}' 的策略")
            print(f"  可用行业: {', '.join(INDUSTRY_STRATEGIES.keys())}")
        return
    
    # 默认显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()
