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
import asyncio
import json
import logging
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Any

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
    StockInfo,
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

# Module-level caches (initialized lazily, thread-safe)
_config_cache = None
_param_manager_cache = None
_dates_manager_cache = None
_stock_info_cache = {}
_cache_lock = threading.Lock()

# Prediction result cache (key: code, value: (result, timestamp))
_pred_cache = {}
_pred_ttl = 300  # 5 minutes TTL

# Thread pool for async execution
_executor = ThreadPoolExecutor(max_workers=4)


def _get_config_cached():
    """Get config with module-level caching (thread-safe)."""
    global _config_cache
    if _config_cache is None:
        with _cache_lock:
            if _config_cache is None:
                _config_cache = get_config()
    return _config_cache


def _get_param_manager_cached():
    """Get param manager with module-level caching (thread-safe)."""
    global _param_manager_cache
    if _param_manager_cache is None:
        with _cache_lock:
            if _param_manager_cache is None:
                _param_manager_cache = get_param_manager()
    return _param_manager_cache


def _get_dates_manager_cached():
    """Get dates manager with module-level caching (thread-safe)."""
    global _dates_manager_cache
    if _dates_manager_cache is None:
        with _cache_lock:
            if _dates_manager_cache is None:
                _dates_manager_cache = get_important_dates_manager()
    return _dates_manager_cache


def _get_stock_info_cached(code: str) -> StockInfo:
    """Get stock info with module-level caching (thread-safe)."""
    if code not in _stock_info_cache:
        with _cache_lock:
            if code not in _stock_info_cache:
                _stock_info_cache[code] = StockInfoResolver.resolve(code)
    return _stock_info_cache[code]


def _get_cache_cached():
    """Get cache with module-level caching (thread-safe)."""
    config = _get_config_cached()
    return get_cache(config.get("data.cache_dir", "cache"))


def _get_cached_prediction(code: str) -> Optional[dict]:
    """Get cached prediction result if still valid (5 min TTL)."""
    if code in _pred_cache:
        result, timestamp = _pred_cache[code]
        if time.time() - timestamp < _pred_ttl:
            return result
        else:
            del _pred_cache[code]
    return None


def _cache_prediction(code: str, result: dict):
    """Cache prediction result with 5 min TTL."""
    _pred_cache[code] = (result, time.time())

# Stock name mapping for predefined list
STOCK_NAMES = {
    # A股 (CN)
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "000688.SH": "科创50",
    "399006.SZ": "创业板指",
    "159632.SZ": "纳斯达克ETF",
    "159870.SZ": "化工ETF",
    "159567.SZ": "港股创新药ETF",
    "600111.SH": "北方华创",
    "603986.SH": "兆易创新",
    "601138.SH": "工业富联",
    "002475.SZ": "立讯精密",
    "002156.SZ": "通富微电",
    "000021.SZ": "深科技",
    "601020.SH": "华钰矿业",
    "600036.SH": "招商银行",
    "000333.SZ": "美的集团",
    "603191.SH": "望变电气",
    "600089.SH": "特变电工",
    "601288.SH": "农业银行",
    "600887.SH": "伊利股份",
    "600900.SH": "长江电力",
    "600362.SH": "江西铜业",
    "000807.SZ": "云铝股份",
    "000792.SZ": "盐湖股份",
    # 港股 (HK)
    "HSTECH.HK": "恒生科技指数",
    "9988.HK": "阿里巴巴",
    "1024.HK": "快手",
    "0981.HK": "中芯国际",
    "9961.HK": "携程集团",
    "3690.HK": "美团",
    "1810.HK": "小米集团",
    "3750.HK": "携程-S",
    "9880.HK": "小鹏汽车",
    "0700.HK": "腾讯控股",
    "2097.HK": "蜜雪集团",
    "9868.HK": "零跑汽车",
    "1357.HK": "美图",
    "0100.HK": "MiniMax",
    "6082.HK": "壁仞科技",
    "2577.HK": "英诺赛科",
    "2020.HK": "安踏体育",
    "0522.HK": "ASM太平洋",
    "1347.HK": "华虹半导体",
    "9626.HK": "贝壳-W",
    # 美股 (US)
    "AMZN": "亚马逊",
    "MSFT": "微软",
    "TSLA": "特斯拉",
    "AAPL": "苹果",
    "ASML": "阿斯麦",
    "TSM": "台积电",
    "SE": "Sea Ltd",
    "SMR": "NuScale Power",
    "CRDO": "Credo Technology",
    "OKLO": "Oklo Inc",
    "QCOM": "高通",
    "AMD": "超威半导体",
    "INTC": "英特尔",
    "GOOGL": "谷歌",
    "AVGO": "博通",
    "NVDA": "英伟达",
    "PONY": "小马智行",
    "PDD": "拼多多",
    "CRWV": "Coreweave",
    "MU": "美光科技",
    "SNDK": "西部数据",
    "UNH": "联合健康",
    "TCEHY": "腾讯ADR",
    "NBIS": "Nebius Group",
}

# Zone to stock code suffix mapping
ZONE_SUFFIX = {
    "cn": [".SH", ".SZ"],
    "hk": [".HK"],
    "us": [],
}


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

    # 加载差异化策略
    try:
        from scripts.prediction_strategies import get_strategy_for_stock
        stock_strategy = get_strategy_for_stock(code)
        print(f"  Strategy: {stock_strategy.name}")
        print(f"  Description: {stock_strategy.description}")
        
        # 应用策略参数
        if args.threshold == 0.008:  # 仅在使用默认值时覆盖
            args.threshold = stock_strategy.threshold
        if args.ml_weight == 0.35:
            args.ml_weight = stock_strategy.ml_weight
        if args.technical_weight == 0.25:
            args.technical_weight = stock_strategy.technical_weight
        if args.momentum_weight == 0.15:
            args.momentum_weight = stock_strategy.momentum_weight
        if not args.exclude_dates and stock_strategy.exclude_dates:
            args.exclude_dates = True
    except Exception as e:
        stock_strategy = None
        logger.debug(f"Could not load strategy: {e}")

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
            trend_weight=stock_strategy.trend_label_weight if stock_strategy else 0.30,
            momentum_weight=stock_strategy.momentum_label_weight if stock_strategy else 0.30,
            market_weight=stock_strategy.market_label_weight if stock_strategy else 0.20,
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
    # Check prediction cache first (fast_mode benefits most)
    if fast_mode or skip_training:
        cached = _get_cached_prediction(code)
        if cached is not None:
            logger.info(f"Returning cached prediction for {code}")
            return cached

    market = "a_share"
    if not is_index:
        try:
            stock_info = _get_stock_info_cached(code)
            market = stock_info.market.replace("_share", "")
        except ValueError as e:
            return {"error": str(e)}

    config = _get_config_cached()
    cache = _get_cache_cached()

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
            param_manager = _get_param_manager_cached()
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
        dates_manager = _get_dates_manager_cached()
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
        # Cache trained model for fast_mode reuse
        _cache_model(code, model)
    else:
        # Load cached model instead of retraining
        model = _load_cached_model(code, model_pipeline)
        if model is None:
            # Fallback: train if no cached model available
            logger.info("No cached model found, training new model...")
            model = model_pipeline.train(
                train_df,
                forward_days=1,
                threshold=threshold,
                use_composite_labels=True,
                trend_weight=0.30,
                momentum_weight=0.30,
                market_weight=0.20,
            )
            _cache_model(code, model)

    if skip_eval or fast_mode:
        # Use cached eval metrics if available, otherwise default
        eval_metrics = _get_cached_eval_metrics(code) or {
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
        # Cache eval metrics for fast_mode requests
        _cache_eval_metrics(code, eval_metrics)

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

    # Cache prediction result for fast_mode reuse
    if fast_mode or skip_training:
        _cache_prediction(code, clean_data)

    return clean_data


def _get_model_dir() -> Path:
    """Get the models directory path."""
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    return model_dir


def _load_cached_model(code: str, model_pipeline) -> Optional[Any]:
    """Load cached model for a stock.

    Args:
        code: Stock code
        model_pipeline: ModelPipeline instance

    Returns:
        Cached model or None if not found
    """
    model_dir = _get_model_dir()
    model_path = model_dir / f"{code.replace('.', '_')}_model.pkl"

    if not model_path.exists():
        return None

    try:
        import pickle

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded cached model for {code}")
        return model
    except Exception as e:
        logger.warning(f"Failed to load cached model for {code}: {e}")
        return None


def _cache_model(code: str, model: Any) -> bool:
    """Cache trained model to disk.

    Args:
        code: Stock code
        model: Trained model

    Returns:
        True if successful
    """
    model_dir = _get_model_dir()
    model_path = model_dir / f"{code.replace('.', '_')}_model.pkl"

    try:
        import pickle

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Cached model for {code}")
        return True
    except Exception as e:
        logger.warning(f"Failed to cache model for {code}: {e}")
        return False


def _get_cached_eval_metrics(code: str) -> Optional[dict]:
    """Get cached evaluation metrics for a stock.

    Args:
        code: Stock code

    Returns:
        Cached metrics or None
    """
    model_dir = _get_model_dir()
    metrics_path = model_dir / f"{code.replace('.', '_')}_metrics.json"

    if not metrics_path.exists():
        return None

    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _cache_eval_metrics(code: str, metrics: dict) -> bool:
    """Cache evaluation metrics to disk.

    Args:
        code: Stock code
        metrics: Evaluation metrics dict

    Returns:
        True if successful
    """
    model_dir = _get_model_dir()
    metrics_path = model_dir / f"{code.replace('.', '_')}_metrics.json"

    try:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        return True
    except Exception as e:
        logger.warning(f"Failed to cache metrics for {code}: {e}")
        return False


def _quick_predict(code: str, is_index: bool = False) -> dict:
    """轻量预测：只用缓存数据 + 技术信号，不训练模型，不调外部API

    Args:
        code: Stock or index code
        is_index: Whether code is an index

    Returns:
        Lightweight prediction result
    """
    import numpy as np

    # 1. 先检查预测缓存
    cached = _get_cached_prediction(code)
    if cached:
        logger.info(f"Quick predict: returning cached prediction for {code}")
        return cached

    # 2. 从缓存获取特征数据（只读SQLite，不调外部API）
    config = _get_config_cached()
    cache = _get_cache_cached()

    if is_index:
        # 指数没有缓存特征，只能返回错误
        return {"error": "Quick predict not available for index, use /predict instead"}

    try:
        stock_info = _get_stock_info_cached(code)
        market = stock_info.market.replace("_share", "")
    except ValueError as e:
        return {"error": str(e)}

    # 尝试从缓存读取特征
    from src.features import get_feature_combinator
    combinator = get_feature_combinator(cache)

    # 不强制刷新，只用缓存
    try:
        df = combinator.get_combined_features(code, days=365, force_refresh=False)
    except Exception as e:
        logger.warning(f"Quick predict: cache miss for {code}: {e}")
        return {"error": f"No cached data for {code}, use /predict first to fetch data"}

    if df is None or df.empty or len(df) < 30:
        return {"error": f"Insufficient cached data for {code}, use /predict first"}

    latest = df.iloc[-1]
    close = latest.get("close", 0)

    # 3. 纯技术分析，不需要模型
    up_signals = 0
    down_signals = 0
    total = 0
    bullish_factors = []
    bearish_factors = []

    # RSI
    rsi = latest.get("rsi", None)
    if rsi is not None and not pd.isna(rsi):
        total += 1
        if rsi < 30:
            up_signals += 1
            bullish_factors.append(f"RSI超卖({rsi:.0f})")
        elif rsi > 70:
            down_signals += 1
            bearish_factors.append(f"RSI超买({rsi:.0f})")

    # 均线排列
    ma5 = latest.get("ma_5", None)
    ma10 = latest.get("ma_10", None)
    ma20 = latest.get("ma_20", None)
    if ma5 and ma10 and ma20 and not pd.isna(ma5) and not pd.isna(ma10) and not pd.isna(ma20):
        total += 1
        if ma5 > ma10 > ma20:
            up_signals += 1
            bullish_factors.append("多头排列")
        elif ma5 < ma10 < ma20:
            down_signals += 1
            bearish_factors.append("空头排列")

    # MACD
    macd_hist = latest.get("macd_hist", None)
    if macd_hist is not None and not pd.isna(macd_hist) and len(df) >= 2:
        prev_hist = df.iloc[-2].get("macd_hist", 0)
        if not pd.isna(prev_hist):
            total += 1
            if prev_hist < 0 and macd_hist > 0:
                up_signals += 1
                bullish_factors.append("MACD金叉")
            elif prev_hist > 0 and macd_hist < 0:
                down_signals += 1
                bearish_factors.append("MACD死叉")

    # 布林带
    bb_pos = latest.get("bb_position", None)
    if bb_pos is not None and not pd.isna(bb_pos):
        total += 1
        if bb_pos < 0.1:
            up_signals += 1
            bullish_factors.append("触及布林下轨")
        elif bb_pos > 0.9:
            down_signals += 1
            bearish_factors.append("触及布林上轨")

    # 动量
    mom5 = latest.get("momentum_5", 0)
    if not pd.isna(mom5):
        total += 1
        if mom5 > 0.02:
            up_signals += 1
            bullish_factors.append(f"5日动量强({mom5:.1%})")
        elif mom5 < -0.02:
            down_signals += 1
            bearish_factors.append(f"5日动量弱({mom5:.1%})")

    # 成交量
    vol_ratio = latest.get("volume_ratio", None)
    returns = latest.get("returns", 0)
    if vol_ratio is not None and not pd.isna(vol_ratio):
        total += 1
        if vol_ratio > 2.0 and returns > 0.01:
            up_signals += 1
            bullish_factors.append("放量上涨")
        elif vol_ratio > 2.0 and returns < -0.01:
            down_signals += 1
            bearish_factors.append("放量下跌")

    # 判断方向
    if total == 0:
        direction = "NEUTRAL"
        confidence = 0.45
    elif up_signals > down_signals:
        direction = "UP"
        confidence = 0.5 + min(up_signals / total, 0.5) * 0.3
    elif down_signals > up_signals:
        direction = "DOWN"
        confidence = 0.5 + min(down_signals / total, 0.5) * 0.3
    else:
        direction = "NEUTRAL"
        confidence = 0.45

    # 获取日期
    last_date = pd.to_datetime(df["date"].max()).strftime("%Y-%m-%d") if "date" in df.columns else pd.Timestamp.now().strftime("%Y-%m-%d")
    target = pd.to_datetime(df["date"].max()) + pd.Timedelta(days=1) if "date" in df.columns else pd.Timestamp.now() + pd.Timedelta(days=1)
    while target.weekday() >= 5:
        target += pd.Timedelta(days=1)
    target_date = target.strftime("%Y-%m-%d")

    result = {
        "stock_code": code,
        "market": market,
        "direction": direction,
        "confidence": round(confidence, 3),
        "current_price": close,
        "prediction_date": last_date,
        "target_date": target_date,
        "mode": "quick",
        "up_signals": up_signals,
        "down_signals": down_signals,
        "total_signals": total,
        "bullish_factors": bullish_factors,
        "bearish_factors": bearish_factors,
    }

    # 缓存结果
    _cache_prediction(code, result)
    return result


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
        fast_mode: bool = Query(True, description="快速模式 (跳过训练/评估/实时价格)"),
        skip_training: bool = Query(True, description="跳过模型训练"),
        skip_eval: bool = Query(True, description="跳过模型评估"),
        skip_realtime: bool = Query(True, description="跳过实时价格查询"),
        skip_params: bool = Query(True, description="跳过优化参数查询"),
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

        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    _executor,
                    run_prediction,
                    code, is_index, train_days, threshold, refresh,
                    multi_model, ml_weight, technical_weight, momentum_weight,
                    exclude_dates, 2.0, fast_mode, skip_training, skip_eval,
                    skip_realtime, skip_params
                ),
                timeout=50,
            )
        except asyncio.TimeoutError:
            logger.error(f"Prediction timeout for {code}")
            raise HTTPException(status_code=504, detail="Prediction timed out, try /predict/cache or /predict/quick")
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
        fast_mode = request.get("fast_mode", True)
        skip_training = request.get("skip_training", True)
        skip_eval = request.get("skip_eval", True)
        skip_realtime = request.get("skip_realtime", True)
        skip_params = request.get("skip_params", True)

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

        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    _executor,
                    run_prediction,
                    code, is_index, train_days, threshold, refresh,
                    multi_model, ml_weight, technical_weight, momentum_weight,
                    exclude_dates, 2.0, fast_mode, skip_training, skip_eval,
                    skip_realtime, skip_params
                ),
                timeout=50,
            )
        except asyncio.TimeoutError:
            logger.error(f"Prediction timeout for {code}")
            raise HTTPException(status_code=504, detail="Prediction timed out, try /predict/cache or /predict/quick")
        except Exception as e:
            logger.exception(f"Prediction failed for {code}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    @app.get("/predict/cache")
    async def predict_cache(
        stock: str = Query(None, description="股票代码"),
        index: str = Query(None, description="指数代码"),
    ):
        """返回上次缓存的预测结果，无计算"""
        if not stock and not index:
            raise HTTPException(status_code=400, detail="Please provide stock or index")
        code = index if index else stock
        cached = _get_cached_prediction(code)
        if cached:
            return cached
        raise HTTPException(status_code=404, detail=f"No cached prediction for {code}, call /predict first")

    @app.get("/predict/quick")
    async def predict_quick(
        stock: str = Query(None, description="股票代码"),
        index: str = Query(None, description="指数代码"),
    ):
        """轻量预测：只用缓存数据 + 技术信号，不训练模型，不调外部API"""
        if not stock and not index:
            raise HTTPException(status_code=400, detail="Please provide stock or index")
        if stock and index:
            raise HTTPException(status_code=400, detail="Please provide only one of stock or index")

        code = index if index else stock
        is_index = index is not None

        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(_executor, _quick_predict, code, is_index),
                timeout=30,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Quick prediction timed out")
        except Exception as e:
            logger.exception(f"Quick prediction failed for {code}")
            raise HTTPException(status_code=500, detail=str(e))

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result

    @app.get("/stocks/{code}/info")
    async def stock_info(code: str):
        try:
            info = _get_stock_info_cached(code)
            return {
                "code": code,
                "name": info.name,
                "market": info.market,
                "exchange": info.exchange,
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/stocks")
    async def list_stocks(zone: str = Query(..., description="区域: cn, hk, us")):
        """Get predefined stock list by zone.

        Args:
            zone: cn (A股), hk (港股), us (美股)

        Returns:
            JSON with stock list including code and name
        """
        zone = zone.lower()
        if zone not in ZONE_SUFFIX:
            raise HTTPException(
                status_code=400, detail="Invalid zone. Must be one of: cn, hk, us"
            )

        suffixes = ZONE_SUFFIX[zone]
        stocks = []

        for code, name in STOCK_NAMES.items():
            if zone == "us":
                # US stocks have no suffix
                if "." not in code:
                    stocks.append({"code": code, "name": name})
            else:
                # CN and HK stocks have suffix
                if any(code.endswith(s) for s in suffixes):
                    stocks.append({"code": code, "name": name})

        return {"zone": zone, "count": len(stocks), "stocks": stocks}

    @app.post("/stocks")
    async def list_stocks_post(request: dict):
        """Get predefined stock list by zone (POST version).

        Args:
            zone: cn (A股), hk (港股), us (美股)

        Returns:
            JSON with stock list including code and name
        """
        zone = request.get("zone", "").lower()
        if not zone:
            raise HTTPException(status_code=400, detail="Please provide zone parameter")
        if zone not in ZONE_SUFFIX:
            raise HTTPException(
                status_code=400, detail="Invalid zone. Must be one of: cn, hk, us"
            )

        suffixes = ZONE_SUFFIX[zone]
        stocks = []

        for code, name in STOCK_NAMES.items():
            if zone == "us":
                if "." not in code:
                    stocks.append({"code": code, "name": name})
            else:
                if any(code.endswith(s) for s in suffixes):
                    stocks.append({"code": code, "name": name})

        return {"zone": zone, "count": len(stocks), "stocks": stocks}

    @app.on_event("startup")
    async def startup_event():
        """预热：加载配置和依赖，减少首次请求延迟"""
        logger.info("Warming up: preloading config and dependencies...")
        try:
            _get_config_cached()
            _get_cache_cached()
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")

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
