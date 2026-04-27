#!/usr/bin/env python3
"""
预测效果优化测试

系统化测试各种优化方法，找到最佳组合。

Usage:
    python scripts/optimize_predict.py --stock 0700.HK
    python scripts/optimize_predict.py --market hk --max-stocks 3
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from src.features import get_feature_combinator
from src.models import MultiModelEnsemble, StockTradingModel
from src.utils import get_cache, get_config


# 测试股票列表
TEST_STOCKS = {
    "hk": ["0700.HK", "1024.HK", "1347.HK", "1810.HK", "3690.HK"],
    "cn": ["600036.SH", "000333.SZ", "600111.SH"],
    "us": ["AAPL", "NVDA", "MSFT"],
}


def load_cached_data(code: str) -> Optional[pd.DataFrame]:
    """加载缓存数据."""
    cache_dir = Path("cache")
    f = cache_dir / f"{code}_technical.parquet"
    if f.exists():
        df = pd.read_parquet(f)
        if len(df) >= 80:
            return df
    return None


def evaluate_predictions(
    model, test_df: pd.DataFrame, threshold: float,
    confidence_gap: float = 0.0, min_confidence: float = 0.0,
    use_trend_filter: bool = False, trend_window: int = 5,
) -> Dict:
    """评估模型预测效果."""
    correct = 0
    total = 0
    skipped = 0
    
    # 盈亏统计
    wins = []
    losses = []
    
    for i in range(max(trend_window, 1), len(test_df) - 1):
        try:
            current_close = test_df.iloc[i]["close"]
            next_close = test_df.iloc[i + 1]["close"]
            actual_return = (next_close - current_close) / current_close
            actual_dir = 1 if actual_return > threshold else (-1 if actual_return < -threshold else 0)
            
            # 获取预测概率
            pred_proba = model.predict_proba(test_df.iloc[[i]])
            buy_prob = pred_proba.get("buy_probability", 0.33)
            sell_prob = pred_proba.get("sell_probability", 0.33)
            hold_prob = pred_proba.get("hold_probability", 0.34)
            
            # 高置信度过滤
            max_prob = max(buy_prob, sell_prob, hold_prob)
            if min_confidence > 0 and max_prob < min_confidence:
                skipped += 1
                continue
            
            # 信号差距过滤
            prob_diff = abs(buy_prob - sell_prob)
            if confidence_gap > 0 and prob_diff < confidence_gap:
                skipped += 1
                continue
            
            # 趋势过滤
            if use_trend_filter:
                recent_returns = []
                for j in range(1, trend_window + 1):
                    r = (test_df.iloc[i]["close"] - test_df.iloc[i - j]["close"]) / test_df.iloc[i - j]["close"]
                    recent_returns.append(r)
                avg_return = np.mean(recent_returns)
                
                # 如果趋势不明确，跳过
                if abs(avg_return) < 0.005:
                    skipped += 1
                    continue
                
                # 趋势方向与预测方向一致才预测
                pred_dir_trend = 1 if avg_return > 0 else -1
                pred_dir_signal = 1 if buy_prob > sell_prob else (-1 if sell_prob > buy_prob else 0)
                
                if pred_dir_trend != pred_dir_signal and pred_dir_signal != 0:
                    skipped += 1
                    continue
            
            # 确定预测方向
            pred_dir = 0
            if buy_prob > sell_prob + confidence_gap:
                pred_dir = 1
            elif sell_prob > buy_prob + confidence_gap:
                pred_dir = -1
            
            total += 1
            if pred_dir == actual_dir:
                correct += 1
                if pred_dir != 0:
                    wins.append(abs(actual_return))
            else:
                if pred_dir != 0:
                    losses.append(abs(actual_return))
                    
        except Exception:
            continue
    
    # 计算指标
    accuracy = correct / total if total > 0 else 0
    win_rate = len(wins) / (len(wins) + len(losses)) if (len(wins) + len(losses)) > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = (sum(wins) / sum(losses)) if losses and sum(losses) > 0 else 0
    expectancy = (win_rate * avg_win - (1 - win_rate) * avg_loss) if (wins or losses) else 0
    
    return {
        "accuracy": accuracy,
        "total_predictions": total,
        "skipped": skipped,
        "coverage": total / (total + skipped) if (total + skipped) > 0 else 0,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
    }


def test_model_config(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_config: Dict,
    eval_config: Dict,
) -> Dict:
    """测试特定模型配置."""
    try:
        model = StockTradingModel(model_config)
        model.train(
            train_df,
            forward_days=1,
            threshold=eval_config.get("threshold", 0.005),
            use_composite_labels=eval_config.get("use_composite", True),
            trend_weight=eval_config.get("trend_weight", 0.30),
            momentum_weight=eval_config.get("momentum_weight", 0.30),
            market_weight=eval_config.get("market_weight", 0.20),
        )
        
        return evaluate_predictions(
            model, test_df,
            threshold=eval_config.get("threshold", 0.005),
            confidence_gap=eval_config.get("confidence_gap", 0.0),
            min_confidence=eval_config.get("min_confidence", 0.0),
            use_trend_filter=eval_config.get("use_trend_filter", False),
            trend_window=eval_config.get("trend_window", 5),
        )
    except Exception as e:
        return {"error": str(e)}


def test_multi_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    eval_config: Dict,
) -> Dict:
    """测试多模型集成."""
    try:
        ensemble = MultiModelEnsemble({
            "catboost": {
                "iterations": 300,
                "depth": 4,
                "learning_rate": 0.03,
                "n_estimators": 3,
            },
            "lightgbm": {
                "n_estimators": 200,
                "max_depth": 4,
            },
            "xgboost": {
                "n_estimators": 200,
                "max_depth": 4,
            },
        })
        ensemble.train(
            train_df,
            forward_days=1,
            threshold=eval_config.get("threshold", 0.005),
        )
        
        return evaluate_predictions(
            ensemble, test_df,
            threshold=eval_config.get("threshold", 0.005),
            confidence_gap=eval_config.get("confidence_gap", 0.0),
            min_confidence=eval_config.get("min_confidence", 0.0),
            use_trend_filter=eval_config.get("use_trend_filter", False),
        )
    except Exception as e:
        return {"error": str(e)}


def run_optimization(code: str) -> Dict:
    """运行优化测试."""
    df = load_cached_data(code)
    if df is None:
        return {"error": f"No data for {code}"}
    
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    results = {}
    
    # ============================================================
    # 测试1: 不同阈值
    # ============================================================
    print(f"\n  测试1: 不同阈值")
    print(f"  {'─' * 50}")
    for threshold in [0.003, 0.005, 0.008, 0.010]:
        model_config = {"iterations": 200, "depth": 4, "learning_rate": 0.03, "n_estimators": 3}
        eval_config = {"threshold": threshold}
        r = test_model_config(train_df, test_df, model_config, eval_config)
        if "error" not in r:
            print(f"    阈值 {threshold:.3f}: 准确率 {r['accuracy']:.1%}, 覆盖率 {r['coverage']:.1%}, 期望收益 {r['expectancy']:.4f}")
            results[f"threshold_{threshold}"] = r
    
    # ============================================================
    # 测试2: 不同模型复杂度
    # ============================================================
    print(f"\n  测试2: 不同模型复杂度")
    print(f"  {'─' * 50}")
    configs = [
        (100, 3, "简单"),
        (200, 4, "中等"),
        (300, 5, "复杂"),
        (500, 6, "深度"),
    ]
    for iterations, depth, name in configs:
        model_config = {"iterations": iterations, "depth": depth, "learning_rate": 0.03, "n_estimators": 3}
        eval_config = {"threshold": 0.005}
        r = test_model_config(train_df, test_df, model_config, eval_config)
        if "error" not in r:
            print(f"    {name} ({iterations}iter, {depth}depth): 准确率 {r['accuracy']:.1%}, 期望收益 {r['expectancy']:.4f}")
            results[f"model_{name}"] = r
    
    # ============================================================
    # 测试3: 高置信度过滤
    # ============================================================
    print(f"\n  测试3: 高置信度过滤")
    print(f"  {'─' * 50}")
    for min_conf in [0.0, 0.40, 0.45, 0.50, 0.55]:
        model_config = {"iterations": 200, "depth": 4, "learning_rate": 0.03, "n_estimators": 3}
        eval_config = {"threshold": 0.005, "min_confidence": min_conf}
        r = test_model_config(train_df, test_df, model_config, eval_config)
        if "error" not in r:
            print(f"    最低置信度 {min_conf:.2f}: 准确率 {r['accuracy']:.1%}, 覆盖率 {r['coverage']:.1%}, 期望收益 {r['expectancy']:.4f}")
            results[f"min_conf_{min_conf}"] = r
    
    # ============================================================
    # 测试4: 信号差距过滤
    # ============================================================
    print(f"\n  测试4: 信号差距过滤")
    print(f"  {'─' * 50}")
    for gap in [0.0, 0.05, 0.10, 0.15, 0.20]:
        model_config = {"iterations": 200, "depth": 4, "learning_rate": 0.03, "n_estimators": 3}
        eval_config = {"threshold": 0.005, "confidence_gap": gap}
        r = test_model_config(train_df, test_df, model_config, eval_config)
        if "error" not in r:
            print(f"    信号差距 {gap:.2f}: 准确率 {r['accuracy']:.1%}, 覆盖率 {r['coverage']:.1%}, 期望收益 {r['expectancy']:.4f}")
            results[f"gap_{gap}"] = r
    
    # ============================================================
    # 测试5: 趋势过滤
    # ============================================================
    print(f"\n  测试5: 趋势过滤")
    print(f"  {'─' * 50}")
    for window in [3, 5, 10]:
        model_config = {"iterations": 200, "depth": 4, "learning_rate": 0.03, "n_estimators": 3}
        eval_config = {"threshold": 0.005, "use_trend_filter": True, "trend_window": window}
        r = test_model_config(train_df, test_df, model_config, eval_config)
        if "error" not in r:
            print(f"    趋势窗口 {window}: 准确率 {r['accuracy']:.1%}, 覆盖率 {r['coverage']:.1%}, 期望收益 {r['expectancy']:.4f}")
            results[f"trend_{window}"] = r
    
    # ============================================================
    # 测试6: 多模型集成
    # ============================================================
    print(f"\n  测试6: 多模型集成")
    print(f"  {'─' * 50}")
    eval_config = {"threshold": 0.005}
    r = test_multi_model(train_df, test_df, eval_config)
    if "error" not in r:
        print(f"    多模型集成: 准确率 {r['accuracy']:.1%}, 期望收益 {r['expectancy']:.4f}")
        results["multi_model"] = r
    
    # ============================================================
    # 测试7: 最佳组合
    # ============================================================
    print(f"\n  测试7: 最佳组合 (趋势过滤 + 高置信度)")
    print(f"  {'─' * 50}")
    model_config = {"iterations": 300, "depth": 5, "learning_rate": 0.025, "n_estimators": 4}
    eval_config = {
        "threshold": 0.005,
        "min_confidence": 0.45,
        "use_trend_filter": True,
        "trend_window": 5,
    }
    r = test_model_config(train_df, test_df, model_config, eval_config)
    if "error" not in r:
        print(f"    最佳组合: 准确率 {r['accuracy']:.1%}, 覆盖率 {r['coverage']:.1%}, 期望收益 {r['expectancy']:.4f}")
        results["best_combo"] = r
    
    return results


def find_best_config(results: Dict) -> Tuple[str, Dict]:
    """找到最佳配置."""
    best_name = None
    best_score = -float("inf")
    
    for name, r in results.items():
        if "error" in r:
            continue
        
        # 综合评分: 准确率 * 0.4 + 期望收益 * 0.4 + 覆盖率 * 0.2
        score = (
            r["accuracy"] * 0.4 +
            r["expectancy"] * 100 * 0.4 +  # 放大期望收益
            r["coverage"] * 0.2
        )
        
        if score > best_score:
            best_score = score
            best_name = name
    
    return best_name, results.get(best_name, {})


def main():
    parser = argparse.ArgumentParser(description="预测效果优化测试")
    parser.add_argument("--stock", type=str, help="单只股票")
    parser.add_argument("--market", choices=["hk", "cn", "us"], default="hk")
    parser.add_argument("--max-stocks", type=int, default=3)
    args = parser.parse_args()
    
    print("=" * 60)
    print("  预测效果优化测试")
    print("=" * 60)
    
    # 获取测试股票
    if args.stock:
        stocks = [args.stock]
    else:
        stocks = TEST_STOCKS.get(args.market, [])[:args.max_stocks]
    
    print(f"  测试股票: {stocks}")
    
    # 收集所有结果
    all_results = {}
    
    for code in stocks:
        print(f"\n{'=' * 60}")
        print(f"  测试: {code}")
        print(f"{'=' * 60}")
        
        results = run_optimization(code)
        if "error" not in results:
            all_results[code] = results
            
            # 找到最佳配置
            best_name, best_config = find_best_config(results)
            if best_config:
                print(f"\n  ✓ 最佳配置: {best_name}")
                print(f"    准确率: {best_config['accuracy']:.1%}")
                print(f"    覆盖率: {best_config['coverage']:.1%}")
                print(f"    期望收益: {best_config['expectancy']:.4f}")
                print(f"    盈亏比: {best_config['profit_factor']:.2f}")
    
    # 汇总分析
    if all_results:
        print(f"\n{'=' * 60}")
        print("  汇总分析")
        print(f"{'=' * 60}")
        
        # 收集所有最佳配置的指标
        best_accuracies = []
        best_coverages = []
        best_expectancies = []
        
        for code, results in all_results.items():
            best_name, best_config = find_best_config(results)
            if best_config:
                best_accuracies.append(best_config["accuracy"])
                best_coverages.append(best_config["coverage"])
                best_expectancies.append(best_config["expectancy"])
        
        if best_accuracies:
            print(f"\n  最佳配置平均指标:")
            print(f"    准确率: {np.mean(best_accuracies):.1%}")
            print(f"    覆盖率: {np.mean(best_coverages):.1%}")
            print(f"    期望收益: {np.mean(best_expectancies):.4f}")
        
        # 分析哪些优化方法最有效
        print(f"\n  优化方法效果分析:")
        
        # 收集各方法的准确率提升
        method_scores = {}
        for code, results in all_results.items():
            baseline = results.get("threshold_0.005", {}).get("accuracy", 0)
            for name, r in results.items():
                if "error" in r:
                    continue
                if name not in method_scores:
                    method_scores[name] = []
                method_scores[name].append(r["accuracy"] - baseline)
        
        # 按平均提升排序
        sorted_methods = sorted(
            method_scores.items(),
            key=lambda x: np.mean(x[1]),
            reverse=True,
        )
        
        print(f"    {'方法':<25} {'平均准确率提升':>15}")
        print(f"    {'─' * 40}")
        for name, improvements in sorted_methods[:10]:
            avg_imp = np.mean(improvements)
            sign = "+" if avg_imp > 0 else ""
            print(f"    {name:<25} {sign}{avg_imp:.1%}")
    
    print(f"\n{'=' * 60}")
    print("  优化测试完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
