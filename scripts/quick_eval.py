#!/usr/bin/env python3
"""
快速预测效果评估

使用简化方法快速评估预测准确性，适用于大批量测试。

Usage:
    # 快速评估所有市场
    python scripts/quick_eval.py

    # 评估特定市场
    python scripts/quick_eval.py --market cn --max-stocks 5

    # 评估单只股票
    python scripts/quick_eval.py --stock 000001.SZ
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from src.features import get_feature_combinator
from src.models import StockTradingModel
from src.utils import get_cache, get_config


# 精简的股票列表
EVAL_STOCKS = {
    "cn": {
        "600036.SH": "招商银行",
        "000333.SZ": "美的集团",
        "600111.SH": "北方华创",
        "000001.SH": "上证指数",
    },
    "hk": {
        "0700.HK": "腾讯控股",
        "9988.HK": "阿里巴巴",
        "1810.HK": "小米集团",
    },
    "us": {
        "AAPL": "苹果",
        "NVDA": "英伟达",
        "MSFT": "微软",
    },
}


def quick_evaluate(
    stock_code: str,
    train_days: int = 150,
    eval_days: int = 20,
    threshold: float = 0.008,
) -> Optional[Dict]:
    """快速评估单只股票的预测效果."""
    try:
        config = get_config()
        cache = get_cache(config.get("data.cache_dir", "cache"))
        combinator = get_feature_combinator(cache)
        
        # 获取数据
        total_days = train_days + eval_days + 10
        df = combinator.get_combined_features(stock_code, total_days)
        
        if df.empty or len(df) < train_days + 10:
            return None
        
        # 划分训练/测试
        train_df = df.iloc[:train_days]
        test_df = df.iloc[train_days:]
        
        # 训练模型 (使用轻量级配置)
        model = StockTradingModel({
            "iterations": 100,
            "depth": 3,
            "learning_rate": 0.05,
            "n_estimators": 2,
        })
        model.train(train_df, forward_days=1, threshold=threshold)
        
        # 逐日预测并统计
        correct = 0
        total = 0
        up_correct = 0
        up_total = 0
        down_correct = 0
        down_total = 0
        
        for i in range(len(test_df) - 1):
            try:
                current_close = test_df.iloc[i]["close"]
                next_close = test_df.iloc[i + 1]["close"]
                actual_return = (next_close - current_close) / current_close
                actual_direction = 1 if actual_return > threshold else (-1 if actual_return < -threshold else 0)
                
                # 使用当前行预测
                pred_df = test_df.iloc[[i]]
                pred_action, pred_conf = model.predict(pred_df)
                
                # 确定预测方向
                pred_proba = model.predict_proba(pred_df)
                buy_prob = pred_proba.get("buy_probability", 0.33)
                sell_prob = pred_proba.get("sell_probability", 0.33)
                
                if buy_prob > sell_prob:
                    pred_direction = 1
                elif sell_prob > buy_prob:
                    pred_direction = -1
                else:
                    pred_direction = 0
                
                total += 1
                if pred_direction == actual_direction:
                    correct += 1
                
                if pred_direction == 1:
                    up_total += 1
                    if actual_direction == 1:
                        up_correct += 1
                elif pred_direction == -1:
                    down_total += 1
                    if actual_direction == -1:
                        down_correct += 1
                        
            except Exception:
                continue
        
        if total < 5:
            return None
        
        return {
            "stock_code": stock_code,
            "accuracy": correct / total,
            "up_precision": up_correct / up_total if up_total > 0 else 0,
            "down_precision": down_correct / down_total if down_total > 0 else 0,
            "samples": total,
            "up_signals": up_total,
            "down_signals": down_total,
        }
        
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description="快速预测效果评估")
    parser.add_argument("--stock", type=str, help="单只股票")
    parser.add_argument("--market", choices=["cn", "hk", "us", "all"], default="all")
    parser.add_argument("--max-stocks", type=int, default=4)
    parser.add_argument("--train-days", type=int, default=150)
    parser.add_argument("--eval-days", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.008)
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  快速预测效果评估")
    print("=" * 60)
    print(f"  训练天数: {args.train_days}")
    print(f"  评估天数: {args.eval_days}")
    print(f"  阈值: {args.threshold}")
    
    # 获取股票列表
    if args.stock:
        stocks = {args.stock: args.stock}
    elif args.market == "all":
        stocks = {}
        for m in EVAL_STOCKS.values():
            stocks.update(m)
    else:
        stocks = EVAL_STOCKS.get(args.market, {})
    
    if args.max_stocks:
        stocks = dict(list(stocks.items())[:args.max_stocks])
    
    print(f"  评估股票数: {len(stocks)}")
    print()
    
    # 评估
    results = []
    for code, name in stocks.items():
        print(f"  评估 {code} ({name})...", end=" ", flush=True)
        
        result = quick_evaluate(
            code,
            train_days=args.train_days,
            eval_days=args.eval_days,
            threshold=args.threshold,
        )
        
        if result:
            results.append(result)
            print(f"准确率: {result['accuracy']:.1%}")
        else:
            print("跳过")
    
    if not results:
        print("\n  没有有效结果")
        return
    
    # 统计
    df = pd.DataFrame(results)
    
    print(f"\n{'─' * 60}")
    print("  汇总统计")
    print(f"{'─' * 60}")
    print(f"  平均准确率     : {df['accuracy'].mean():.1%}")
    print(f"  平均看涨精准率 : {df['up_precision'].mean():.1%}")
    print(f"  平均看跌精准率 : {df['down_precision'].mean():.1%}")
    print(f"  最高准确率     : {df['accuracy'].max():.1%}")
    print(f"  最低准确率     : {df['accuracy'].min():.1%}")
    
    print(f"\n{'─' * 60}")
    print("  详细结果")
    print(f"{'─' * 60}")
    for _, row in df.sort_values("accuracy", ascending=False).iterrows():
        print(f"  {row['stock_code']:<12} 准确率: {row['accuracy']:.1%} "
              f"看涨: {row['up_precision']:.1%} ({row['up_signals']}) "
              f"看跌: {row['down_precision']:.1%} ({row['down_signals']})")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
