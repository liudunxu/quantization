#!/usr/bin/env python
"""
股票下个交易日涨跌预测脚本

预测股票下个交易日的涨跌方向，输出概率和置信度。

Usage:
    python scripts/predict.py --stock 000001.SZ
    python scripts/predict.py --stock 0700.HK --train-days 365
    python scripts/predict.py --stock AAPL --threshold 0.01
    python scripts/predict.py --stock 000001.SZ --output json
    python scripts/predict.py --stock 000001.SZ --output csv

Features:
    - 使用 CatBoost 模型预测下个交易日涨跌
    - 输出方向：UP (看涨) / DOWN (看跌) / NEUTRAL (中性)
    - 支持多种输出格式：text (默认), json, csv
    - 每次重新训练模型，针对当前股票优化
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

from src.utils import get_cache, get_config, StockInfoResolver
from src.pipelines import DataPipeline, ModelPipeline
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
        "--threshold", type=float, default=0.005, help="涨跌阈值 (默认: 0.005, 即0.5%%)"
    )
    parser.add_argument(
        "--output",
        choices=["text", "json", "csv"],
        default="text",
        help="输出格式 (默认: text)",
    )
    parser.add_argument("--refresh", action="store_true", help="强制刷新数据缓存")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # 创建日志目录
    Path("logs").mkdir(exist_ok=True)

    print("=" * 60)
    print("  STOCK PREDICTION SYSTEM")
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
    model = model_pipeline.train(
        train_df,
        forward_days=1,
        threshold=args.threshold,
    )
    print("  Model training completed")

    # 6. 模型评估
    print("\n  Evaluating model...")
    accuracy = model_pipeline.evaluate_accuracy(
        model, eval_df, threshold=args.threshold
    )
    print(f"  Accuracy: {accuracy:.1%}")

    # 7. 获取最新特征并预测
    print("\n  Predicting next day...")
    latest_df = data_pipeline.get_latest(df)
    current_price = data_pipeline.get_realtime_price(stock_code)

    if current_price is None:
        current_price = df["close"].iloc[-1]
        print(f"  (Using last close price)")

    prediction = model_pipeline.predict_direction(model, latest_df, current_price)

    # 8. 添加元数据
    prediction["stock_code"] = stock_code
    prediction["market"] = market
    prediction["prediction_date"] = datetime.now().strftime("%Y-%m-%d")
    prediction["target_date"] = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    prediction["model_accuracy"] = accuracy

    # 9. 获取特征重要性
    importance = model.get_feature_importance()
    prediction["top_features"] = importance.head(5).to_dict("records")

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

    print("\n" + "=" * 60)
    print("  PREDICTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
