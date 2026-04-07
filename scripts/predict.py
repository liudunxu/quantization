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
    python scripts/predict.py --serve                    # Start HTTP API server
    python scripts/predict.py --serve --host 0.0.0.0 --port 8000

Features:
    - 使用 CatBoost 模型预测下个交易日涨跌
    - 结合技术分析信号（均线、RSI、MACD、布林带等）
    - 结合动量分析和市场情绪
    - 输出详细的看涨/看跌因素解释
    - 支持多种输出格式：text (默认), json, csv
    - 支持 HTTP API 服务模式 (FastAPI)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.display import PredictionFormatter
from src.features.index_features import extract_index_features, get_index_name
from src.pipelines import DataPipeline, ModelPipeline
from src.predictors import EnsemblePredictor
from src.utils import (
    StockInfoResolver,
    get_cache,
    get_config,
    get_important_dates_manager,
    get_param_manager,
)

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
        help="股票代码 (如 000001.SZ, 0700.HK, AAPL)",
    )
    parser.add_argument(
        "--index",
        type=str,
        help="A股指数代码 (如 000001=上证指数, 000300=沪深300, 399001=深证成指, 399006=创业板指)",
    )
    parser.add_argument(
        "--train-days", type=int, default=365, help="训练天数 (默认: 365)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.008, help="涨跌阈值 (默认: 0.008, 即0.8%%)"
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
        default=True,
        help="使用多模型集成 (CatBoost + LightGBM + XGBoost) (默认开启)",
    )
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="仅使用单个 CatBoost 模型",
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
    parser.add_argument(
        "--exclude-dates",
        action="store_true",
        help="排除极端波动日期以降低异常值影响",
    )
    parser.add_argument(
        "--exclude-threshold",
        type=float,
        default=2.0,
        help="极端波动检测阈值(标准差倍数, 默认: 2.0)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="启动 HTTP API 服务 (FastAPI)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="HTTP 服务监听地址 (默认: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP 服务监听端口 (默认: 8000)",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    if not args.stock and not args.index:
        print("  Error: Please specify --stock or --index")
        return
    if args.stock and args.index:
        print("  Error: Please specify only one of --stock or --index")
        return

    is_index = args.index is not None
    code = args.index if is_index else args.stock

    Path("logs").mkdir(exist_ok=True)

    print("=" * 60)
    print("  STOCK PREDICTION SYSTEM (Enhanced)")
    print("=" * 60)

    if is_index:
        index_name = get_index_name(code)
        print(f"  Index Code   : {code}")
        print(f"  Index Name   : {index_name}")
        market = "a_share"
    else:
        try:
            stock_info = StockInfoResolver.resolve(code)
            market = stock_info.market.replace("_share", "")
        except ValueError as e:
            print(f"  Error: {e}")
            return
        print(f"  Stock Code : {code}")
        print(f"  Market     : {market}")

    # 2. 初始化流水线
    config = get_config()
    cache = get_cache(config.get("data.cache_dir", "cache"))

    if is_index:
        # Use simplified index feature extraction
        print(f"\n  Fetching index data for {code}...")
        total_days = args.train_days + 30
        df = extract_index_features(code, days=total_days)
        if df.empty or len(df) < 50:
            print("  Error: Insufficient index data")
            return
    else:
        data_pipeline = DataPipeline(cache, config)
        model_pipeline = ModelPipeline(config)

        # 2.1 读取优化后的参数（如果有）
        param_manager = get_param_manager()
        optimized_params = param_manager.get_strategy_params("prediction", market, code)

        if optimized_params:
            print(f"\n  Found optimized parameters for {code}")
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
        total_days = args.train_days + 30
        df = data_pipeline.fetch_features(code, total_days, args.refresh)

        if df.empty or len(df) < 50:
            print("  Error: Insufficient data")
            return

    # 3.1 处理重要日期（如果启用）
    excluded_dates = []
    if args.exclude_dates:
        print("\n  Processing important dates...")
        dates_manager = get_important_dates_manager()

        start_date = (
            df["date"].min().strftime("%Y-%m-%d") if "date" in df.columns else None
        )
        end_date = (
            df["date"].max().strftime("%Y-%m-%d") if "date" in df.columns else None
        )

        excluded_dates = dates_manager.get_or_detect_dates(
            df=df,
            market=market,
            start_date=start_date,
            end_date=end_date,
            auto_detect=True,
        )

        if excluded_dates:
            print(f"  Found {len(excluded_dates)} extreme volatility dates to exclude")
            for d in excluded_dates[:5]:
                print(f"    - {d}")
            if len(excluded_dates) > 5:
                print(f"    ... and {len(excluded_dates) - 5} more")

            df_dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            mask = ~df_dates.isin(excluded_dates)
            df = df[mask].reset_index(drop=True)
            print(f"  Remaining samples after filtering: {len(df)}")

            if len(df) < 50:
                print(
                    "  Warning: Insufficient data after filtering, disabling date exclusion"
                )
                if is_index:
                    df = extract_index_features(code, days=total_days)
                else:
                    df = data_pipeline.fetch_features(code, total_days, args.refresh)
                excluded_dates = []
        else:
            print("  No extreme volatility dates detected")

    # 4. 划分数据
    if is_index:
        train_df = df.iloc[:-30]
        eval_df = df.iloc[-30:]
    else:
        train_df, eval_df = data_pipeline.split_train_eval(df, backtest_days=30)
    print(f"  Total samples  : {len(df)}")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Eval samples   : {len(eval_df)}")

    # 5. 训练模型
    print("\n  Training model (forward_days=1)...")

    # --single-model 覆盖默认的多模型设置
    use_multi_model = args.multi_model and not args.single_model

    train_result = {}
    if use_multi_model:
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
        print("  Using single CatBoost model...")
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
    if use_multi_model:
        # 多模型也使用eval_df进行完整评估
        eval_metrics = model_pipeline.evaluate_metrics(
            model, eval_df, threshold=args.threshold
        )
        accuracy = eval_metrics["accuracy"]
        print(f"  Models trained : {train_result.get('models_trained', [])}")
        print(f"  Eval Accuracy  : {accuracy:.1%}")
        print(f"  UP Precision   : {eval_metrics['precision_up']:.1%}")
        print(f"  UP Recall      : {eval_metrics['recall_up']:.1%}")
        print(f"  UP F1          : {eval_metrics['f1_up']:.1%}")
        print(f"  DOWN Precision : {eval_metrics['precision_down']:.1%}")
        print(f"  DOWN Recall    : {eval_metrics['recall_down']:.1%}")
        print(f"  DOWN F1        : {eval_metrics['f1_down']:.1%}")
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
    if is_index:
        current_price = df["close"].iloc[-1]
        print("  (Using last close price for index)")
    else:
        current_price = data_pipeline.get_realtime_price(code)
        if current_price is None:
            current_price = df["close"].iloc[-1]
            print("  (Using last close price)")

    # 8. 使用集成预测器进行综合预测
    print("\n  Running ensemble prediction...")
    ensemble_predictor = EnsemblePredictor(
        {
            "ml_weight": args.ml_weight,
            "technical_weight": args.technical_weight,
            "momentum_weight": args.momentum_weight,
            "model_accuracy": accuracy,
        }
    )

    prediction = ensemble_predictor.predict(model, df, current_price)

    # 9. 添加元数据
    prediction["stock_code"] = code
    prediction["market"] = market
    if is_index:
        prediction["index_name"] = get_index_name(code)

    if "date" in df.columns and not df.empty:
        last_date = pd.to_datetime(df["date"].max())
        prediction["prediction_date"] = last_date.strftime("%Y-%m-%d")
        target = last_date + pd.Timedelta(days=1)
        while target.weekday() >= 5:
            target += pd.Timedelta(days=1)
        prediction["target_date"] = target.strftime("%Y-%m-%d")
    else:
        now = pd.Timestamp.now()
        prediction["prediction_date"] = now.strftime("%Y-%m-%d")
        prediction["target_date"] = (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

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
        if is_index:
            print(f"  Index: {code} ({get_index_name(code)})")

    print("\n" + "=" * 60)
    print("  PREDICTION COMPLETE")
    print("=" * 60)


def run_prediction(
    code: str,
    is_index: bool = False,
    train_days: int = 365,
    threshold: float = 0.008,
    refresh: bool = False,
    multi_model: bool = True,
    ml_weight: float = 0.35,
    technical_weight: float = 0.25,
    momentum_weight: float = 0.15,
    exclude_dates: bool = False,
    exclude_threshold: float = 2.0,
    fast_mode: bool = False,
    skip_training: bool = False,
    skip_eval: bool = False,
    skip_realtime: bool = False,
    skip_params: bool = False,
) -> dict:
    """Run prediction and return result as dict.

    Args:
        code: Stock or index code
        is_index: Whether code is an index
        train_days: Training days
        threshold: Up/down threshold
        refresh: Force refresh data cache
        multi_model: Use multi-model ensemble
        ml_weight: ML model weight
        technical_weight: Technical analysis weight
        momentum_weight: Momentum analysis weight
        exclude_dates: Exclude extreme volatility dates
        exclude_threshold: Extreme volatility detection threshold
        fast_mode: Fast mode (skip training, eval, realtime price, use cache)
        skip_training: Skip model training (use cached model)
        skip_eval: Skip model evaluation
        skip_realtime: Skip realtime price (use last close)
        skip_params: Skip optimized params lookup

    Returns:
        Prediction result dict
    """
    market = "a_share"
    if not is_index:
        try:
            stock_info = StockInfoResolver.resolve(code)
            market = stock_info.market.replace("_share", "")
        except ValueError as e:
            return {"error": str(e)}

    config = get_config()
    cache = get_cache(config.get("data.cache_dir", "cache"))

    if is_index:
        total_days = train_days + 30
        df = extract_index_features(code, days=total_days)
        if df.empty or len(df) < 50:
            return {"error": "Insufficient index data"}
        model_pipeline = ModelPipeline(config)
    else:
        data_pipeline = DataPipeline(cache, config)
        model_pipeline = ModelPipeline(config)

        if not skip_params:
            param_manager = get_param_manager()
            optimized_params = param_manager.get_strategy_params(
                "prediction", market, code
            )

            if optimized_params:
                if threshold == 0.008 and "threshold" in optimized_params:
                    threshold = optimized_params["threshold"]
                if ml_weight == 0.35 and "ml_weight" in optimized_params:
                    ml_weight = optimized_params["ml_weight"]
                if technical_weight == 0.25 and "technical_weight" in optimized_params:
                    technical_weight = optimized_params["technical_weight"]
                if momentum_weight == 0.15 and "momentum_weight" in optimized_params:
                    momentum_weight = optimized_params["momentum_weight"]

        total_days = train_days + 30
        df = data_pipeline.fetch_features(code, total_days, refresh)

        if df.empty or len(df) < 50:
            return {"error": "Insufficient data"}

    excluded_dates = []
    if exclude_dates:
        dates_manager = get_important_dates_manager()
        start_date = (
            df["date"].min().strftime("%Y-%m-%d") if "date" in df.columns else None
        )
        end_date = (
            df["date"].max().strftime("%Y-%m-%d") if "date" in df.columns else None
        )

        excluded_dates = dates_manager.get_or_detect_dates(
            df=df,
            market=market,
            start_date=start_date,
            end_date=end_date,
            auto_detect=True,
        )

        if excluded_dates:
            df_dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            mask = ~df_dates.isin(excluded_dates)
            df = df[mask].reset_index(drop=True)

            if len(df) < 50:
                if is_index:
                    df = extract_index_features(code, days=total_days)
                else:
                    df = data_pipeline.fetch_features(code, total_days, refresh)
                excluded_dates = []

    if is_index:
        train_df = df.iloc[:-30]
        eval_df = df.iloc[-30:]
    else:
        train_df, eval_df = data_pipeline.split_train_eval(df, backtest_days=30)

    should_train = not skip_training and not fast_mode

    if should_train:
        if multi_model:
            from src.models import MultiModelEnsemble

            ensemble = MultiModelEnsemble()
            train_result = ensemble.train(
                train_df,
                forward_days=1,
                threshold=threshold,
            )
            model = ensemble
        else:
            model = model_pipeline.train(
                train_df,
                forward_days=1,
                threshold=threshold,
                use_composite_labels=True,
                trend_weight=0.30,
                momentum_weight=0.30,
                market_weight=0.20,
            )
    else:
        model = model_pipeline.train(
            train_df,
            forward_days=1,
            threshold=threshold,
            use_composite_labels=True,
            trend_weight=0.30,
            momentum_weight=0.30,
            market_weight=0.20,
        )

    if skip_eval or fast_mode:
        eval_metrics = {
            "accuracy": 0.5,
            "precision_up": 0.5,
            "recall_up": 0.5,
            "f1_up": 0.5,
            "precision_down": 0.5,
            "recall_down": 0.5,
            "f1_down": 0.5,
        }
    else:
        eval_metrics = model_pipeline.evaluate_metrics(
            model, eval_df, threshold=threshold
        )

    accuracy = eval_metrics["accuracy"]

    if is_index:
        current_price = df["close"].iloc[-1]
    else:
        if skip_realtime or fast_mode:
            current_price = df["close"].iloc[-1]
        else:
            current_price = data_pipeline.get_realtime_price(code)
            if current_price is None:
                current_price = df["close"].iloc[-1]

    ensemble_predictor = EnsemblePredictor(
        {
            "ml_weight": ml_weight,
            "technical_weight": technical_weight,
            "momentum_weight": momentum_weight,
            "model_accuracy": accuracy,
        }
    )

    prediction = ensemble_predictor.predict(model, df, current_price)

    prediction["stock_code"] = code
    prediction["market"] = market
    if is_index:
        prediction["index_name"] = get_index_name(code)

    if "date" in df.columns and not df.empty:
        last_date = pd.to_datetime(df["date"].max())
        prediction["prediction_date"] = last_date.strftime("%Y-%m-%d")
        target = last_date + pd.Timedelta(days=1)
        while target.weekday() >= 5:
            target += pd.Timedelta(days=1)
        prediction["target_date"] = target.strftime("%Y-%m-%d")
    else:
        now = pd.Timestamp.now()
        prediction["prediction_date"] = now.strftime("%Y-%m-%d")
        prediction["target_date"] = (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    prediction["model_accuracy"] = accuracy
    prediction["precision_up"] = eval_metrics["precision_up"]
    prediction["recall_up"] = eval_metrics["recall_up"]
    prediction["f1_up"] = eval_metrics["f1_up"]
    prediction["precision_down"] = eval_metrics["precision_down"]
    prediction["recall_down"] = eval_metrics["recall_down"]
    prediction["f1_down"] = eval_metrics["f1_down"]

    clean_data = {}
    for key, value in prediction.items():
        if isinstance(value, (list, dict)):
            clean_data[key] = value
        elif hasattr(value, "item"):
            clean_data[key] = value.item()
        elif hasattr(value, "tolist"):
            clean_data[key] = value.tolist()
        else:
            clean_data[key] = value

    return clean_data


def start_server(host: str = "127.0.0.1", port: int = 8000):
    """Start FastAPI HTTP server."""
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import JSONResponse
        import uvicorn
    except ImportError:
        print(
            "Error: fastapi and uvicorn are required. Install with: pip install fastapi uvicorn"
        )
        sys.exit(1)

    app = FastAPI(
        title="Stock Prediction API",
        description="股票预测 HTTP API - 基于 CatBoost 集成模型",
        version="1.0.0",
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "stock-prediction-api"}

    @app.get("/predict")
    async def predict(
        stock: str = Query(None, description="股票代码 (如 000001.SZ, 0700.HK, AAPL)"),
        index: str = Query(None, description="A股指数代码 (如 000001, 000300)"),
        train_days: int = Query(365, description="训练天数"),
        threshold: float = Query(0.008, description="涨跌阈值"),
        refresh: bool = Query(False, description="强制刷新数据"),
        multi_model: bool = Query(True, description="使用多模型集成"),
        ml_weight: float = Query(0.35, description="ML模型权重"),
        technical_weight: float = Query(0.25, description="技术分析权重"),
        momentum_weight: float = Query(0.15, description="动量分析权重"),
        exclude_dates: bool = Query(False, description="排除极端波动日期"),
        fast_mode: bool = Query(False, description="快速模式 (跳过训练/评估/实时价格)"),
        skip_training: bool = Query(False, description="跳过模型训练"),
        skip_eval: bool = Query(False, description="跳过模型评估"),
        skip_realtime: bool = Query(False, description="跳过实时价格查询"),
        skip_params: bool = Query(False, description="跳过优化参数查询"),
    ):
        if not stock and not index:
            raise HTTPException(
                status_code=400, detail="Please provide stock or index parameter"
            )
        if stock and index:
            raise HTTPException(
                status_code=400, detail="Please provide only one of stock or index"
            )

        is_index = index is not None
        code = index if is_index else stock

        try:
            result = run_prediction(
                code=code,
                is_index=is_index,
                train_days=train_days,
                threshold=threshold,
                refresh=refresh,
                multi_model=multi_model,
                ml_weight=ml_weight,
                technical_weight=technical_weight,
                momentum_weight=momentum_weight,
                exclude_dates=exclude_dates,
                fast_mode=fast_mode,
                skip_training=skip_training,
                skip_eval=skip_eval,
                skip_realtime=skip_realtime,
                skip_params=skip_params,
            )
        except Exception as e:
            logger.exception(f"Prediction failed for {code}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    @app.post("/predict")
    async def predict_post(request: dict):
        stock = request.get("stock")
        index = request.get("index")
        train_days = request.get("train_days", 365)
        threshold = request.get("threshold", 0.008)
        refresh = request.get("refresh", False)
        multi_model = request.get("multi_model", True)
        ml_weight = request.get("ml_weight", 0.35)
        technical_weight = request.get("technical_weight", 0.25)
        momentum_weight = request.get("momentum_weight", 0.15)
        exclude_dates = request.get("exclude_dates", False)
        fast_mode = request.get("fast_mode", False)
        skip_training = request.get("skip_training", False)
        skip_eval = request.get("skip_eval", False)
        skip_realtime = request.get("skip_realtime", False)
        skip_params = request.get("skip_params", False)

        if not stock and not index:
            raise HTTPException(
                status_code=400, detail="Please provide stock or index in request body"
            )
        if stock and index:
            raise HTTPException(
                status_code=400, detail="Please provide only one of stock or index"
            )

        is_index = index is not None
        code = index if is_index else stock

        try:
            result = run_prediction(
                code=code,
                is_index=is_index,
                train_days=train_days,
                threshold=threshold,
                refresh=refresh,
                multi_model=multi_model,
                ml_weight=ml_weight,
                technical_weight=technical_weight,
                momentum_weight=momentum_weight,
                exclude_dates=exclude_dates,
                fast_mode=fast_mode,
                skip_training=skip_training,
                skip_eval=skip_eval,
                skip_realtime=skip_realtime,
                skip_params=skip_params,
            )
        except Exception as e:
            logger.exception(f"Prediction failed for {code}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    @app.get("/stocks/{code}/info")
    async def stock_info(code: str):
        try:
            info = StockInfoResolver.resolve(code)
            return {
                "code": code,
                "name": info.name,
                "market": info.market,
                "exchange": info.exchange,
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    print(f"\n  Starting Stock Prediction API...")
    print(f"  Server: http://{host}:{port}")
    print(f"  Docs:   http://{host}:{port}/docs")
    print(f"  Health: http://{host}:{port}/health")
    print(f"  Predict (GET):  http://{host}:{port}/predict?stock=000001.SZ")
    print(f"  Predict (POST): http://{host}:{port}/predict")
    print()

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    args = parse_args()

    if args.serve:
        start_server(host=args.host, port=args.port)
    else:
        main()
