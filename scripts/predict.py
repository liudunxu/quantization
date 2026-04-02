#!/usr/bin/env python
"""
股票下个交易日涨跌预测脚本

预测股票下个交易日的涨跌方向，输出概率和置信度。
结合ML模型、技术分析和动量分析进行综合预测。

Usage:
    python scripts/predict.py --stock 000001.SZ
    python scripts/predict.py --stock 0700.HK --train-days 365
    python scripts/predict.py --stock AAPL --threshold 0.01
    python scripts/predict.py --stock 000001.SZ --output json
    python scripts/predict.py --stock 000001.SZ --output csv

Features:
    - 使用 CatBoost 模型预测下个交易日涨跌
    - 结合技术分析信号（均线、RSI、MACD、布林带等）
    - 结合动量分析和市场情绪
    - 输出详细的看涨/看跌因素解释
    - 支持多种输出格式：text (默认), json, csv
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_cache, get_config, get_param_manager, StockInfoResolver
from src.pipelines import DataPipeline, ModelPipeline
from src.predictors import EnsemblePredictor
from src.display import PredictionFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="股票下个交易日涨跌预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/predict.py --stock 000001.SZ
  python scripts/predict.py --stock 0700.HK --train-days 365
  python scripts/predict.py --stock AAPL --output json
        """,
    )
    parser.add_argument(
        "--stock",
        type=str,
        required=True,
        help="股票代码 (如 000001.SZ, 0700.HK, AAPL)",
    )
    parser.add_argument(
        "--train-days", type=int, default=365, help="训练天数 (默认: 365)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.008, help="涨跌阈值 (默认: 0.008, 即0.8%)"
    )
    parser.add_argument(
        "--output",
        choices=["text", "json", "csv"],
        default="text",
        help="输出格式 (默认: text)",
    )
    parser.add_argument("--refresh", action="store_true", help="强制刷新数据缓存")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument(
        "--multi-model",
        action="store_true",
        help="使用多模型集成 (CatBoost + LightGBM + XGBoost)",
    )
    parser.add_argument(
        "--ml-weight", type=float, default=0.35, help="ML模型权重 (默认: 0.35)"
    )
    parser.add_argument(
        "--technical-weight", type=float, default=0.25, help="技术分析权重 (默认: 0.25)"
    )
    parser.add_argument(
        "--momentum-weight", type=float, default=0.15, help="动量分析权重 (默认: 0.15)"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.65, help="最低置信度阈值 (默认: 0.65)"
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # 创建日志目录
    Path("logs").mkdir(exist_ok=True)

    print("=" * 60)
    print("  STOCK PREDICTION SYSTEM (Enhanced)")
    print("=" * 60)

    # 1. 解析股票信息
    stock_code = args.stock
    try:
        stock_info = StockInfoResolver.resolve(stock_code)
        market = stock_info.market.replace("_share", "")
    except ValueError as e:
        print(f"  Error: {e}")
        return

    print(f"  Stock Code : {stock_code}")
    print(f"  Market     : {market}")

    # 2. 初始化流水线
    config = get_config()
    cache = get_cache(config.get("data.cache_dir", "cache"))
    data_pipeline = DataPipeline(cache, config)
    model_pipeline = ModelPipeline(config)

    # 2.1 读取优化后的参数（如果有）
    # 只有当用户没有显式指定参数时才使用优化参数
    param_manager = get_param_manager()
    optimized_params = param_manager.get_strategy_params(
        "prediction", market, stock_code
    )

    if optimized_params:
        print(f"\n  Found optimized parameters for {stock_code}")
        # 只有当用户没有显式指定时才使用优化参数
        # 检查是否是默认值
        if args.threshold == 0.008 and "threshold" in optimized_params:
            args.threshold = optimized_params["threshold"]
            print(f"    Using optimized threshold: {args.threshold}")
        else:
            print(f"    Using command-line threshold: {args.threshold}")
        if args.ml_weight == 0.35 and "ml_weight" in optimized_params:
            args.ml_weight = optimized_params["ml_weight"]
        if args.technical_weight == 0.25 and "technical_weight" in optimized_params:
            args.technical_weight = optimized_params["technical_weight"]
        if args.momentum_weight == 0.15 and "momentum_weight" in optimized_params:
            args.momentum_weight = optimized_params["momentum_weight"]

    # 3. 获取数据
    print("\n  Fetching data...")
    total_days = args.train_days + 30  # 额外30天用于评估
    df = data_pipeline.fetch_features(stock_code, total_days, args.refresh)

    if df.empty or len(df) < 50:
        print("  Error: Insufficient data")
        return

    # 4. 划分数据
    train_df, eval_df = data_pipeline.split_train_eval(df, backtest_days=30)
    print(f"  Total samples  : {len(df)}")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Eval samples   : {len(eval_df)}")

    # 5. 训练模型
    print("\n  Training model (forward_days=1)...")

    train_result = {}
    if args.multi_model:
        print("  Using multi-model ensemble (CatBoost + LightGBM + XGBoost)...")
        from src.models import MultiModelEnsemble

        ensemble = MultiModelEnsemble()
        train_result = ensemble.train(
            train_df,
            forward_days=1,
            threshold=args.threshold,
        )
        model = ensemble
        print(f"  Models trained: {train_result.get('models_trained', [])}")
        print(f"  Average accuracy: {train_result.get('train_accuracy', 0):.1%}")
    else:
        model = model_pipeline.train(
            train_df,
            forward_days=1,
            threshold=args.threshold,
            use_composite_labels=True,
            trend_weight=0.30,
            momentum_weight=0.30,
            market_weight=0.20,
        )
    print("  Model training completed")

    # 6. 模型评估
    print("\n  Evaluating model...")
    if args.multi_model:
        # MultiModelEnsemble 使用内置评估
        accuracy = train_result.get("train_accuracy", 0)
        eval_metrics = {
            "accuracy": accuracy,
            "precision_up": 0,
            "recall_up": 0,
            "f1_up": 0,
            "precision_down": 0,
            "recall_down": 0,
            "f1_down": 0,
        }
        print(f"  Models trained : {train_result.get('models_trained', [])}")
        print(f"  Avg Accuracy   : {accuracy:.1%}")
        for name, result in train_result.get("model_results", {}).items():
            print(f"  {name} Accuracy : {result.get('train_accuracy', 0):.1%}")
    else:
        eval_metrics = model_pipeline.evaluate_metrics(
            model, eval_df, threshold=args.threshold
        )
        accuracy = eval_metrics["accuracy"]
        print(f"  Accuracy      : {accuracy:.1%}")
        print(f"  UP Precision  : {eval_metrics['precision_up']:.1%}")
        print(f"  UP Recall     : {eval_metrics['recall_up']:.1%}")
        print(f"  UP F1         : {eval_metrics['f1_up']:.1%}")
        print(f"  DOWN Precision: {eval_metrics['precision_down']:.1%}")
        print(f"  DOWN Recall   : {eval_metrics['recall_down']:.1%}")
        print(f"  DOWN F1       : {eval_metrics['f1_down']:.1%}")

    # 7. 获取实时价格
    print("\n  Getting real-time price...")
    current_price = data_pipeline.get_realtime_price(stock_code)

    if current_price is None:
        current_price = df["close"].iloc[-1]
        print(f"  (Using last close price)")

    # 8. 使用集成预测器进行综合预测
    print("\n  Running ensemble prediction...")
    ensemble_predictor = EnsemblePredictor(
        {
            "ml_weight": args.ml_weight,
            "technical_weight": args.technical_weight,
            "momentum_weight": args.momentum_weight,
            "model_accuracy": accuracy,  # 传递模型准确率用于置信度校准
        }
    )

    prediction = ensemble_predictor.predict(model, df, current_price)

    # 9. 添加元数据
    prediction["stock_code"] = stock_code
    prediction["market"] = market
    prediction["prediction_date"] = datetime.now().strftime("%Y-%m-%d")
    prediction["target_date"] = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    prediction["model_accuracy"] = accuracy
    prediction["precision_up"] = eval_metrics["precision_up"]
    prediction["recall_up"] = eval_metrics["recall_up"]
    prediction["f1_up"] = eval_metrics["f1_up"]
    prediction["precision_down"] = eval_metrics["precision_down"]
    prediction["recall_down"] = eval_metrics["recall_down"]
    prediction["f1_down"] = eval_metrics["f1_down"]

    # 10. 输出结果
    formatter = PredictionFormatter()
    output = formatter.format(prediction, args.output)
    print("\n" + output)

    # 11. 详细输出
    if args.verbose:
        print("\n  === Verbose Info ===")
        print(f"  Threshold: {args.threshold}")
        print(f"  Train days: {args.train_days}")
        print(f"  Data range: {df['date'].min()} to {df['date'].max()}")
        print(f"  ML weight: {args.ml_weight}")
        print(f"  Technical weight: {args.technical_weight}")
        print(f"  Momentum weight: {args.momentum_weight}")

    print("\n" + "=" * 60)
    print("  PREDICTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
