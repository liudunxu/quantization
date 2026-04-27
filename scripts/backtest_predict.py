#!/usr/bin/env python3
"""
预测效果回测脚本

基于历史数据评估预测脚本的准确率，分析不同市场/行业的表现差异。

Usage:
    # 回测所有预定义股票
    python scripts/backtest_predict.py

    # 回测特定市场
    python scripts/backtest_predict.py --market cn
    python scripts/backtest_predict.py --market hk
    python scripts/backtest_predict.py --market us

    # 回测特定股票
    python scripts/backtest_predict.py --stock 000001.SZ
    python scripts/backtest_predict.py --stock 0700.HK

    # 自定义参数
    python scripts/backtest_predict.py --train-days 365 --eval-days 60 --forward-days 1
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

from src.features import get_feature_combinator
from src.models import MultiModelEnsemble, StockTradingModel
from src.pipelines import DataPipeline
from src.predictors import EnsemblePredictor
from src.utils import StockInfoResolver, get_cache, get_config


# ============================================================
# 股票列表定义 (与 predict.py 保持一致)
# ============================================================

STOCK_LISTS = {
    "cn": {
        "name": "A股",
        "stocks": {
            # 指数
            "000001.SH": "上证指数",
            "399001.SZ": "深证成指",
            "399006.SZ": "创业板指",
            # 科技
            "600111.SH": "北方华创",
            "603986.SH": "兆易创新",
            "601138.SH": "工业富联",
            "002475.SZ": "立讯精密",
            "002156.SZ": "通富微电",
            # 金融
            "600036.SH": "招商银行",
            "601288.SH": "农业银行",
            # 消费
            "000333.SZ": "美的集团",
            "600887.SH": "伊利股份",
            # 能源
            "600900.SH": "长江电力",
            "600362.SH": "江西铜业",
            "000807.SZ": "云铝股份",
            # 医药/其他
            "601020.SH": "华钰矿业",
            "603191.SH": "望变电气",
        },
    },
    "hk": {
        "name": "港股",
        "stocks": {
            # 指数
            "HSTECH.HK": "恒生科技指数",
            # 科技
            "9988.HK": "阿里巴巴",
            "0700.HK": "腾讯控股",
            "1810.HK": "小米集团",
            "3690.HK": "美团",
            "1024.HK": "快手",
            "9626.HK": "贝壳-W",
            # 半导体
            "0981.HK": "中芯国际",
            "1347.HK": "华虹半导体",
            # 汽车
            "9880.HK": "小鹏汽车",
            "9868.HK": "零跑汽车",
            # 消费
            "2020.HK": "安踏体育",
            "2097.HK": "蜜雪集团",
            # 其他
            "9961.HK": "携程集团",
            "1357.HK": "美图",
        },
    },
    "us": {
        "name": "美股",
        "stocks": {
            # 科技巨头
            "AAPL": "苹果",
            "MSFT": "微软",
            "GOOGL": "谷歌",
            "AMZN": "亚马逊",
            "NVDA": "英伟达",
            "META": "Meta",
            "TSLA": "特斯拉",
            # 半导体
            "AMD": "超威半导体",
            "INTC": "英特尔",
            "QCOM": "高通",
            "AVGO": "博通",
            "MU": "美光科技",
            "ASML": "阿斯麦",
            "TSM": "台积电",
            # 中概股
            "PDD": "拼多多",
            "BABA": "阿里巴巴",
            "JD": "京东",
            "BIDU": "百度",
        },
    },
}

# 行业分类映射
INDUSTRY_MAP = {
    # A股
    "600111.SH": "半导体", "603986.SH": "半导体", "601138.SH": "电子",
    "002475.SZ": "电子", "002156.SZ": "半导体",
    "600036.SH": "银行", "601288.SH": "银行",
    "000333.SZ": "家电", "600887.SH": "食品饮料",
    "600900.SH": "电力", "600362.SH": "有色金属", "000807.SZ": "有色金属",
    "601020.SH": "有色金属", "603191.SH": "电气设备",
    # 港股
    "9988.HK": "互联网", "0700.HK": "互联网", "1810.HK": "消费电子",
    "3690.HK": "互联网", "1024.HK": "互联网", "9626.HK": "互联网",
    "0981.HK": "半导体", "1347.HK": "半导体",
    "9880.HK": "汽车", "9868.HK": "汽车",
    "2020.HK": "服装", "2097.HK": "餐饮",
    "9961.HK": "旅游", "1357.HK": "互联网",
    # 美股
    "AAPL": "消费电子", "MSFT": "软件", "GOOGL": "互联网",
    "AMZN": "电商", "NVDA": "半导体", "META": "互联网",
    "TSLA": "汽车", "AMD": "半导体", "INTC": "半导体",
    "QCOM": "半导体", "AVGO": "半导体", "MU": "半导体",
    "ASML": "半导体设备", "TSM": "半导体代工",
    "PDD": "电商", "BABA": "电商", "JD": "电商", "BIDU": "互联网",
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="预测效果回测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stock", type=str, help="单只股票代码 (如 000001.SZ)"
    )
    parser.add_argument(
        "--market", choices=["cn", "hk", "us", "all"], default="all",
        help="市场选择 (默认: all)"
    )
    parser.add_argument(
        "--train-days", type=int, default=250, help="训练天数 (默认: 250)"
    )
    parser.add_argument(
        "--eval-days", type=int, default=60, help="评估天数 (默认: 60)"
    )
    parser.add_argument(
        "--forward-days", type=int, default=1, help="预测天数 (默认: 1)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.008, help="涨跌阈值 (默认: 0.008)"
    )
    parser.add_argument(
        "--step", type=int, default=5, help="滑动窗口步长 (默认: 5天)"
    )
    parser.add_argument(
        "--max-stocks", type=int, default=None, help="每市场最多回测股票数"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="输出结果到JSON文件"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="详细输出"
    )
    return parser.parse_args()


def get_stock_list(market: str, stock: str = None) -> Dict[str, str]:
    """获取要回测的股票列表."""
    if stock:
        try:
            info = StockInfoResolver.resolve(stock)
            return {stock: info.name if info.name else stock}
        except ValueError:
            return {stock: stock}

    if market == "all":
        all_stocks = {}
        for m in STOCK_LISTS.values():
            all_stocks.update(m["stocks"])
        return all_stocks
    
    return STOCK_LISTS.get(market, {}).get("stocks", {})


def backtest_single_stock(
    stock_code: str,
    stock_name: str,
    train_days: int,
    eval_days: int,
    forward_days: int,
    threshold: float,
    step: int,
    verbose: bool = False,
) -> Optional[Dict]:
    """对单只股票进行历史预测回测.
    
    使用滑动窗口方式模拟真实预测场景。
    """
    try:
        config = get_config()
        cache = get_cache(config.get("data.cache_dir", "cache"))
        pipeline = DataPipeline(cache, config)
        
        # 获取数据 (训练天数 + 评估天数 + 一些缓冲)
        total_days = train_days + eval_days + 30
        df = pipeline.fetch_features(stock_code, total_days)
        
        if df.empty or len(df) < train_days + 20:
            if verbose:
                print(f"  [SKIP] {stock_code} ({stock_name}): 数据不足")
            return None
        
        predictions = []
        actuals = []
        dates = []
        
        # 滑动窗口回测
        total_len = len(df)
        start_idx = train_days
        end_idx = total_len - forward_days
        
        for i in range(start_idx, end_idx, step):
            # 训练数据
            train_df = df.iloc[max(0, i - train_days):i]
            
            if len(train_df) < 50:
                continue
            
            # 当前日期和实际未来收益
            current_date = df.iloc[i].get("date", i)
            current_close = df.iloc[i]["close"]
            future_close = df.iloc[min(i + forward_days, total_len - 1)]["close"]
            actual_return = (future_close - current_close) / current_close
            actual_direction = 1 if actual_return > threshold else (-1 if actual_return < -threshold else 0)
            
            # 训练模型并预测
            try:
                model = StockTradingModel({
                    "iterations": 100,
                    "depth": 3,
                    "learning_rate": 0.05,
                    "n_estimators": 2,
                })
                model.train(
                    train_df,
                    forward_days=forward_days,
                    threshold=threshold,
                )
                
                pred_action, pred_confidence = model.predict(df.iloc[[i]])
                pred_proba = model.predict_proba(df.iloc[[i]])
                
                # 确定预测方向
                buy_prob = pred_proba.get("buy_probability", 0.33)
                sell_prob = pred_proba.get("sell_probability", 0.33)
                hold_prob = pred_proba.get("hold_probability", 0.34)
                
                if buy_prob > sell_prob and buy_prob > hold_prob:
                    pred_direction = 1
                elif sell_prob > buy_prob and sell_prob > hold_prob:
                    pred_direction = -1
                else:
                    pred_direction = 0
                
                predictions.append(pred_direction)
                actuals.append(actual_direction)
                dates.append(current_date)
                
            except Exception as e:
                if verbose:
                    print(f"  [WARN] {stock_code} at {current_date}: {e}")
                continue
        
        if len(predictions) < 5:
            return None
        
        # 计算指标
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # 整体准确率
        correct = np.sum(predictions == actuals)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0
        
        # 方向性准确率 (忽略HOLD，只看涨跌方向是否正确)
        non_hold_mask = (predictions != 0) & (actuals != 0)
        if np.sum(non_hold_mask) > 0:
            direction_correct = np.sum(predictions[non_hold_mask] == actuals[non_hold_mask])
            direction_accuracy = direction_correct / np.sum(non_hold_mask)
        else:
            direction_accuracy = 0
        
        # 看涨准确率
        up_mask = predictions == 1
        if np.sum(up_mask) > 0:
            up_correct = np.sum(actuals[up_mask] == 1)
            up_precision = up_correct / np.sum(up_mask)
        else:
            up_precision = 0
        
        # 看跌准确率
        down_mask = predictions == -1
        if np.sum(down_mask) > 0:
            down_correct = np.sum(actuals[down_mask] == -1)
            down_precision = down_correct / np.sum(down_mask)
        else:
            down_precision = 0
        
        # 实际涨跌分布
        up_actual = np.sum(actuals == 1)
        down_actual = np.sum(actuals == -1)
        hold_actual = np.sum(actuals == 0)
        
        # 预测分布
        up_pred = np.sum(predictions == 1)
        down_pred = np.sum(predictions == -1)
        hold_pred = np.sum(predictions == 0)
        
        # 信息系数 (IC) - 预测概率与实际收益的相关性
        # 使用简单的方向预测与实际方向的相关性
        ic = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
        
        return {
            "stock_code": stock_code,
            "stock_name": stock_name,
            "industry": INDUSTRY_MAP.get(stock_code, "未知"),
            "sample_count": total,
            "accuracy": accuracy,
            "direction_accuracy": direction_accuracy,
            "up_precision": up_precision,
            "down_precision": down_precision,
            "ic": ic,
            "pred_distribution": {
                "up": int(up_pred),
                "down": int(down_pred),
                "hold": int(hold_pred),
            },
            "actual_distribution": {
                "up": int(up_actual),
                "down": int(down_actual),
                "hold": int(hold_actual),
            },
        }
        
    except Exception as e:
        if verbose:
            print(f"  [ERROR] {stock_code} ({stock_name}): {e}")
        return None


def analyze_results(results: List[Dict]) -> Dict:
    """分析回测结果，生成统计报告."""
    if not results:
        return {"error": "No valid results"}
    
    df = pd.DataFrame(results)
    
    # 整体统计
    overall = {
        "total_stocks": len(df),
        "avg_accuracy": df["accuracy"].mean(),
        "avg_direction_accuracy": df["direction_accuracy"].mean(),
        "avg_up_precision": df["up_precision"].mean(),
        "avg_down_precision": df["down_precision"].mean(),
        "avg_ic": df["ic"].mean(),
        "best_accuracy": df["accuracy"].max(),
        "worst_accuracy": df["accuracy"].min(),
    }
    
    # 按市场统计
    market_stats = {}
    for _, row in df.iterrows():
        code = row["stock_code"]
        if code.endswith(".SH") or code.endswith(".SZ"):
            market = "cn"
        elif code.endswith(".HK"):
            market = "hk"
        else:
            market = "us"
        
        if market not in market_stats:
            market_stats[market] = []
        market_stats[market].append(row)
    
    market_summary = {}
    for market, rows in market_stats.items():
        mdf = pd.DataFrame(rows)
        market_summary[market] = {
            "count": len(mdf),
            "avg_accuracy": mdf["accuracy"].mean(),
            "avg_direction_accuracy": mdf["direction_accuracy"].mean(),
            "avg_up_precision": mdf["up_precision"].mean(),
            "avg_down_precision": mdf["down_precision"].mean(),
        }
    
    # 按行业统计
    industry_stats = {}
    for _, row in df.iterrows():
        industry = row["industry"]
        if industry not in industry_stats:
            industry_stats[industry] = []
        industry_stats[industry].append(row)
    
    industry_summary = {}
    for industry, rows in industry_stats.items():
        idf = pd.DataFrame(rows)
        if len(idf) >= 2:  # 至少2只股票才有统计意义
            industry_summary[industry] = {
                "count": len(idf),
                "avg_accuracy": idf["accuracy"].mean(),
                "avg_direction_accuracy": idf["direction_accuracy"].mean(),
            }
    
    # 按准确率排名
    top_stocks = df.nlargest(10, "accuracy")[["stock_code", "stock_name", "accuracy", "direction_accuracy"]].to_dict("records")
    bottom_stocks = df.nsmallest(10, "accuracy")[["stock_code", "stock_name", "accuracy", "direction_accuracy"]].to_dict("records")
    
    return {
        "overall": overall,
        "by_market": market_summary,
        "by_industry": industry_summary,
        "top_stocks": top_stocks,
        "bottom_stocks": bottom_stocks,
        "details": results,
    }


def print_report(analysis: Dict, verbose: bool = False):
    """打印回测报告."""
    print("\n" + "=" * 70)
    print("  预测效果回测报告")
    print("=" * 70)
    
    # 整体统计
    overall = analysis["overall"]
    print(f"\n{'─' * 70}")
    print("  整体统计")
    print(f"{'─' * 70}")
    print(f"  回测股票数     : {overall['total_stocks']}")
    print(f"  平均准确率     : {overall['avg_accuracy']:.2%}")
    print(f"  平均方向准确率 : {overall['avg_direction_accuracy']:.2%}")
    print(f"  平均看涨精准率 : {overall['avg_up_precision']:.2%}")
    print(f"  平均看跌精准率 : {overall['avg_down_precision']:.2%}")
    print(f"  平均IC         : {overall['avg_ic']:.4f}")
    print(f"  最高准确率     : {overall['best_accuracy']:.2%}")
    print(f"  最低准确率     : {overall['worst_accuracy']:.2%}")
    
    # 按市场统计
    print(f"\n{'─' * 70}")
    print("  分市场统计")
    print(f"{'─' * 70}")
    market_names = {"cn": "A股", "hk": "港股", "us": "美股"}
    for market, stats in analysis["by_market"].items():
        name = market_names.get(market, market)
        print(f"\n  {name} ({stats['count']}只):")
        print(f"    平均准确率     : {stats['avg_accuracy']:.2%}")
        print(f"    平均方向准确率 : {stats['avg_direction_accuracy']:.2%}")
        print(f"    看涨精准率     : {stats['avg_up_precision']:.2%}")
        print(f"    看跌精准率     : {stats['avg_down_precision']:.2%}")
    
    # 按行业统计
    print(f"\n{'─' * 70}")
    print("  分行业统计 (>=2只股票)")
    print(f"{'─' * 70}")
    sorted_industries = sorted(
        analysis["by_industry"].items(),
        key=lambda x: x[1]["avg_accuracy"],
        reverse=True,
    )
    for industry, stats in sorted_industries:
        print(f"  {industry:<12} : 准确率 {stats['avg_accuracy']:.2%}, "
              f"方向准确率 {stats['avg_direction_accuracy']:.2%} ({stats['count']}只)")
    
    # Top/Bottom 股票
    print(f"\n{'─' * 70}")
    print("  准确率最高 Top 5")
    print(f"{'─' * 70}")
    for i, stock in enumerate(analysis["top_stocks"][:5], 1):
        print(f"  {i}. {stock['stock_code']:<12} {stock['stock_name']:<12} "
              f"准确率: {stock['accuracy']:.2%}, 方向: {stock['direction_accuracy']:.2%}")
    
    print(f"\n{'─' * 70}")
    print("  准确率最低 Bottom 5")
    print(f"{'─' * 70}")
    for i, stock in enumerate(analysis["bottom_stocks"][:5], 1):
        print(f"  {i}. {stock['stock_code']:<12} {stock['stock_name']:<12} "
              f"准确率: {stock['accuracy']:.2%}, 方向: {stock['direction_accuracy']:.2%}")
    
    # 详细结果
    if verbose:
        print(f"\n{'─' * 70}")
        print("  详细结果")
        print(f"{'─' * 70}")
        for r in analysis["details"]:
            print(f"  {r['stock_code']:<12} {r['stock_name']:<12} "
                  f"准确率: {r['accuracy']:.2%}, 样本: {r['sample_count']}")
    
    print("\n" + "=" * 70)


def generate_optimization_suggestions(analysis: Dict) -> Dict[str, Dict]:
    """基于回测结果生成优化建议."""
    suggestions = {}
    
    # 分析各市场的最佳参数
    for market, stats in analysis["by_market"].items():
        avg_acc = stats["avg_accuracy"]
        up_prec = stats["avg_up_precision"]
        down_prec = stats["avg_down_precision"]
        
        market_suggestion = {
            "current_accuracy": avg_acc,
            "threshold_suggestion": 0.008,
            "model_params": {},
            "strategy_notes": [],
        }
        
        # 阈值建议
        if up_prec > down_prec + 0.1:
            market_suggestion["threshold_suggestion"] = 0.006
            market_suggestion["strategy_notes"].append("看涨精准率明显高于看跌，可降低阈值捕捉更多上涨机会")
        elif down_prec > up_prec + 0.1:
            market_suggestion["threshold_suggestion"] = 0.012
            market_suggestion["strategy_notes"].append("看跌精准率较高，建议提高阈值减少误判")
        
        # 模型参数建议
        if avg_acc < 0.55:
            market_suggestion["model_params"] = {
                "iterations": 500,
                "depth": 5,
                "learning_rate": 0.02,
            }
            market_suggestion["strategy_notes"].append("准确率偏低，建议增加模型复杂度")
        elif avg_acc > 0.65:
            market_suggestion["model_params"] = {
                "iterations": 300,
                "depth": 4,
                "learning_rate": 0.03,
            }
            market_suggestion["strategy_notes"].append("准确率较好，保持当前参数")
        
        suggestions[market] = market_suggestion
    
    # 分析行业特征
    for industry, stats in analysis["by_industry"].items():
        if stats["avg_accuracy"] > 0.65:
            if "top_industries" not in suggestions:
                suggestions["top_industries"] = []
            suggestions["top_industries"].append({
                "industry": industry,
                "accuracy": stats["avg_accuracy"],
                "note": f"{industry}行业预测效果较好，可适当增加仓位",
            })
        elif stats["avg_accuracy"] < 0.50:
            if "bottom_industries" not in suggestions:
                suggestions["bottom_industries"] = []
            suggestions["bottom_industries"].append({
                "industry": industry,
                "accuracy": stats["avg_accuracy"],
                "note": f"{industry}行业预测效果较差，建议降低权重或采用更保守策略",
            })
    
    return suggestions


def main():
    """Main function."""
    args = parse_args()
    
    print("=" * 70)
    print("  预测效果回测系统")
    print("=" * 70)
    print(f"  训练天数   : {args.train_days}")
    print(f"  评估天数   : {args.eval_days}")
    print(f"  预测天数   : {args.forward_days}")
    print(f"  涨跌阈值   : {args.threshold}")
    print(f"  滑动步长   : {args.step}天")
    
    # 获取股票列表
    stock_list = get_stock_list(args.market, args.stock)
    if args.max_stocks:
        stock_list = dict(list(stock_list.items())[:args.max_stocks])
    
    print(f"  回测股票数 : {len(stock_list)}")
    print(f"  市场       : {args.market}")
    print()
    
    # 逐个回测
    results = []
    total = len(stock_list)
    
    for i, (code, name) in enumerate(stock_list.items(), 1):
        print(f"  [{i}/{total}] 回测 {code} ({name})...", end=" ", flush=True)
        
        result = backtest_single_stock(
            stock_code=code,
            stock_name=name,
            train_days=args.train_days,
            eval_days=args.eval_days,
            forward_days=args.forward_days,
            threshold=args.threshold,
            step=args.step,
            verbose=args.verbose,
        )
        
        if result:
            results.append(result)
            print(f"准确率: {result['accuracy']:.2%}")
        else:
            print("跳过")
    
    if not results:
        print("\n  没有有效的回测结果")
        return
    
    # 分析结果
    analysis = analyze_results(results)
    
    # 打印报告
    print_report(analysis, args.verbose)
    
    # 生成优化建议
    suggestions = generate_optimization_suggestions(analysis)
    
    print(f"\n{'─' * 70}")
    print("  优化建议")
    print(f"{'─' * 70}")
    for market, suggestion in suggestions.items():
        if market in ["cn", "hk", "us"]:
            market_names = {"cn": "A股", "hk": "港股", "us": "美股"}
            print(f"\n  {market_names.get(market, market)}:")
            print(f"    建议阈值 : {suggestion['threshold_suggestion']}")
            if suggestion["strategy_notes"]:
                for note in suggestion["strategy_notes"]:
                    print(f"    - {note}")
    
    if "top_industries" in suggestions:
        print(f"\n  预测效果好的行业:")
        for item in suggestions["top_industries"]:
            print(f"    ✓ {item['industry']} (准确率: {item['accuracy']:.2%})")
    
    if "bottom_industries" in suggestions:
        print(f"\n  预测效果差的行业:")
        for item in suggestions["bottom_industries"]:
            print(f"    ✗ {item['industry']} (准确率: {item['accuracy']:.2%})")
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "analysis": {k: v for k, v in analysis.items() if k != "details"},
                "suggestions": suggestions,
                "details": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n  结果已保存到: {args.output}")
    
    print("\n" + "=" * 70)
    print("  回测完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
