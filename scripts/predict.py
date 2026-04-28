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
    python scripts/predict.py --list cn
    python scripts/predict.py --batch "000001.SZ,0700.HK,AAPL"
    python scripts/predict.py --serve                    # Start HTTP API server
    python scripts/predict.py --serve --host 0.0.0.0 --port 8000
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.display import PredictionFormatter
from src.features.index_features import extract_index_features, get_index_name
from src.pipelines import DataPipeline, ModelPipeline
from src.predictors import EnsemblePredictor
from src.utils import (
    StockInfo,
    StockInfoResolver,
    get_cache,
    get_config,
    get_important_dates_manager,
    get_param_manager,
)
from src.utils.stock_info import STOCK_NAMES, ZONE_SUFFIX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class PredictionService:
    """Centralized prediction service with caching and pipeline reuse.

    Consolidates module-level caches into a single thread-safe service.
    Reuses DataPipeline/ModelPipeline instances across requests.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._config = None
        self._cache = None
        self._param_manager = None
        self._dates_manager = None
        self._data_pipeline = None
        self._model_pipeline = None
        self._stock_info_cache: Dict[str, StockInfo] = {}
        self._pred_cache: Dict[str, tuple] = {}
        self._pred_ttl = 14 * 24 * 3600  # 14 days
        self._model_cache: Dict[str, Any] = {}
        self._model_timestamp: Dict[str, float] = {}
        self._model_ttl = 14 * 24 * 3600  # 14 days

    @property
    def config(self):
        if self._config is None:
            with self._lock:
                if self._config is None:
                    self._config = get_config()
        return self._config

    @property
    def cache(self):
        if self._cache is None:
            with self._lock:
                if self._cache is None:
                    self._cache = get_cache(self.config.get("data.cache_dir", "cache"))
        return self._cache

    @property
    def param_manager(self):
        if self._param_manager is None:
            with self._lock:
                if self._param_manager is None:
                    self._param_manager = get_param_manager()
        return self._param_manager

    @property
    def dates_manager(self):
        if self._dates_manager is None:
            with self._lock:
                if self._dates_manager is None:
                    self._dates_manager = get_important_dates_manager()
        return self._dates_manager

    @property
    def data_pipeline(self):
        if self._data_pipeline is None:
            with self._lock:
                if self._data_pipeline is None:
                    self._data_pipeline = DataPipeline(self.cache, self.config)
        return self._data_pipeline

    @property
    def model_pipeline(self):
        if self._model_pipeline is None:
            with self._lock:
                if self._model_pipeline is None:
                    self._model_pipeline = ModelPipeline(self.config)
        return self._model_pipeline

    def resolve_stock(self, code: str) -> StockInfo:
        if code not in self._stock_info_cache:
            with self._lock:
                if code not in self._stock_info_cache:
                    self._stock_info_cache[code] = StockInfoResolver.resolve(code)
        return self._stock_info_cache[code]

    def get_cached_prediction(self, code: str) -> Optional[dict]:
        if code in self._pred_cache:
            result, ts = self._pred_cache[code]
            if time.time() - ts < self._pred_ttl:
                return result
            del self._pred_cache[code]
        return None

    def cache_prediction(self, code: str, result: dict):
        self._pred_cache[code] = (result, time.time())

    def get_cached_model(self, code: str):
        now = time.time()
        if code in self._model_cache:
            if now - self._model_timestamp.get(code, 0) < self._model_ttl:
                return self._model_cache[code]
            del self._model_cache[code]
            self._model_timestamp.pop(code, None)
        return None

    def cache_model(self, code: str, model):
        self._model_cache[code] = model
        self._model_timestamp[code] = time.time()

    def _get_model_dir(self) -> Path:
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        return model_dir

    def load_disk_model(self, code: str):
        model_path = self._get_model_dir() / f"{code.replace('.', '_')}_model.pkl"
        if not model_path.exists():
            return None
        try:
            import pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Loaded disk model for {code}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load disk model for {code}: {e}")
            return None

    def save_disk_model(self, code: str, model) -> bool:
        model_path = self._get_model_dir() / f"{code.replace('.', '_')}_model.pkl"
        try:
            import pickle
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"Saved disk model for {code}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save disk model for {code}: {e}")
            return False

    def load_disk_metrics(self, code: str) -> Optional[dict]:
        metrics_path = self._get_model_dir() / f"{code.replace('.', '_')}_metrics.json"
        if not metrics_path.exists():
            return None
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def save_disk_metrics(self, code: str, metrics: dict) -> bool:
        metrics_path = self._get_model_dir() / f"{code.replace('.', '_')}_metrics.json"
        try:
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)
            return True
        except Exception as e:
            logger.warning(f"Failed to save metrics for {code}: {e}")
            return False

    def resolve_strategy(self, code: str):
        try:
            from scripts.prediction_strategies import get_strategy_for_stock
            return get_strategy_for_stock(code)
        except Exception as e:
            logger.debug(f"Could not load strategy: {e}")
            return None

    def warmup(self):
        logger.info("Warming up PredictionService...")
        try:
            _ = self.config
            _ = self.cache
            _ = self.data_pipeline
            _ = self.model_pipeline
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")


_service = PredictionService()


def _apply_strategy(args, stock_strategy, code: str, market: str):
    if stock_strategy is None:
        return args

    defaults = {
        "threshold": (0.008, stock_strategy.threshold),
        "ml_weight": (0.35, stock_strategy.ml_weight),
        "technical_weight": (0.25, stock_strategy.technical_weight),
        "momentum_weight": (0.15, stock_strategy.momentum_weight),
    }
    for attr, (default_val, strategy_val) in defaults.items():
        if getattr(args, attr, default_val) == default_val:
            setattr(args, attr, strategy_val)

    if not getattr(args, "exclude_dates", False) and stock_strategy.exclude_dates:
        args.exclude_dates = True

    return args


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
        fast_mode: Fast mode (skip training, eval, realtime, use cache)
        skip_training: Skip model training (use cached model)
        skip_eval: Skip model evaluation
        skip_realtime: Skip realtime price (use last close)
        skip_params: Skip optimized params lookup

    Returns:
        Prediction result dict
    """
    if fast_mode or skip_training:
        cached = _service.get_cached_prediction(code)
        if cached is not None:
            logger.info(f"Returning cached prediction for {code}")
            return cached

    # 极速模式减少数据天数，降低特征计算和模型加载开销
    if fast_mode and train_days > 90:
        train_days = 90

    market = "a_share"
    if not is_index:
        try:
            stock_info = _service.resolve_stock(code)
            market = stock_info.market.replace("_share", "")
        except ValueError as e:
            return {"error": str(e)}

    stock_strategy = None
    if not is_index and not skip_params:
        stock_strategy = _service.resolve_strategy(code)
        if stock_strategy:
            if threshold == 0.008:
                threshold = stock_strategy.threshold
            if ml_weight == 0.35:
                ml_weight = stock_strategy.ml_weight
            if technical_weight == 0.25:
                technical_weight = stock_strategy.technical_weight
            if momentum_weight == 0.15:
                momentum_weight = stock_strategy.momentum_weight
            if not exclude_dates and stock_strategy.exclude_dates:
                exclude_dates = True

    if is_index:
        total_days = train_days + 30
        df = extract_index_features(code, days=total_days)
        if df.empty or len(df) < 50:
            return {"error": "Insufficient index data"}
    else:
        df = _service.data_pipeline.fetch_features(code, train_days + 30, refresh)
        if df.empty or len(df) < 50:
            return {"error": "Insufficient data"}

    excluded_dates = []
    if exclude_dates:
        start_date = df["date"].min().strftime("%Y-%m-%d") if "date" in df.columns else None
        end_date = df["date"].max().strftime("%Y-%m-%d") if "date" in df.columns else None

        excluded_dates = _service.dates_manager.get_or_detect_dates(
            df=df, market=market, start_date=start_date, end_date=end_date, auto_detect=True,
        )

        if excluded_dates:
            df_dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            mask = ~df_dates.isin(excluded_dates)
            df = df[mask].reset_index(drop=True)

            if len(df) < 50:
                if is_index:
                    df = extract_index_features(code, days=total_days)
                else:
                    df = _service.data_pipeline.fetch_features(code, train_days + 30, refresh)
                excluded_dates = []

    if is_index:
        train_df = df.iloc[:-30]
        eval_df = df.iloc[-30:]
    else:
        train_df, eval_df = _service.data_pipeline.split_train_eval(df, backtest_days=30)

    should_train = not skip_training and not fast_mode
    trend_weight = stock_strategy.trend_label_weight if stock_strategy else 0.30
    momentum_label_weight = stock_strategy.momentum_label_weight if stock_strategy else 0.30
    market_label_weight = stock_strategy.market_label_weight if stock_strategy else 0.20

    if should_train:
        if multi_model:
            from src.models import MultiModelEnsemble
            ensemble = MultiModelEnsemble()
            train_result = ensemble.train(train_df, forward_days=1, threshold=threshold)
            model = ensemble
            logger.info(f"Models trained: {train_result.get('models_trained', [])}")
        else:
            model = _service.model_pipeline.train(
                train_df, forward_days=1, threshold=threshold,
                use_composite_labels=True,
                trend_weight=trend_weight,
                momentum_weight=momentum_label_weight,
                market_weight=market_label_weight,
            )
        _service.cache_model(code, model)
        _service.save_disk_model(code, model)
    else:
        model = _service.get_cached_model(code) or _service.load_disk_model(code)
        if model is None:
            if fast_mode:
                logger.info("Fast mode: no cached model, skipping ML prediction")
                model = None
            else:
                logger.info("No cached model found, training new model...")
                if multi_model:
                    from src.models import MultiModelEnsemble
                    ensemble = MultiModelEnsemble()
                    ensemble.train(train_df, forward_days=1, threshold=threshold)
                    model = ensemble
                else:
                    model = _service.model_pipeline.train(
                        train_df, forward_days=1, threshold=threshold,
                        use_composite_labels=True,
                        trend_weight=trend_weight,
                        momentum_weight=momentum_label_weight,
                        market_weight=market_label_weight,
                    )
                _service.cache_model(code, model)
                _service.save_disk_model(code, model)

    if skip_eval or fast_mode:
        eval_metrics = _service.load_disk_metrics(code) or {
            "accuracy": 0.5, "precision_up": 0.5, "recall_up": 0.5, "f1_up": 0.5,
            "precision_down": 0.5, "recall_down": 0.5, "f1_down": 0.5,
        }
    else:
        eval_metrics = _service.model_pipeline.evaluate_metrics(model, eval_df, threshold=threshold)
        _service.save_disk_metrics(code, eval_metrics)

    accuracy = eval_metrics["accuracy"]

    if is_index:
        current_price = df["close"].iloc[-1]
    else:
        if skip_realtime or fast_mode:
            current_price = df["close"].iloc[-1]
        else:
            current_price = _service.data_pipeline.get_realtime_price(code)
            if current_price is None:
                current_price = df["close"].iloc[-1]

    ensemble_predictor = EnsemblePredictor({
        "ml_weight": ml_weight,
        "technical_weight": technical_weight,
        "momentum_weight": momentum_weight,
        "model_accuracy": accuracy,
    })

    prediction = ensemble_predictor.predict(model, df, current_price, fast_mode=fast_mode)

    prediction["stock_code"] = code
    prediction["market"] = market
    prediction["stock_name"] = STOCK_NAMES.get(code, code)
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

    if fast_mode or skip_training:
        _service.cache_prediction(code, clean_data)

    return clean_data


def run_batch_prediction(codes: List[str], **kwargs) -> List[dict]:
    """Run prediction for multiple stocks sequentially."""
    results = []
    for code in codes:
        code = code.strip()
        if not code:
            continue
        try:
            result = run_prediction(code=code, **kwargs)
            results.append(result)
        except Exception as e:
            results.append({"error": f"Prediction failed for {code}: {str(e)}", "stock_code": code})
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="股票下个交易日涨跌预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/predict.py --stock 000001.SZ
  python scripts/predict.py --stock 0700.HK --train-days 365
  python scripts/predict.py --stock AAPL --output json
  python scripts/predict.py --list cn
  python scripts/predict.py --batch "000001.SZ,0700.HK,AAPL"
  python scripts/predict.py --serve
        """,
    )
    parser.add_argument("--stock", type=str, help="股票代码 (如 000001.SZ, 0700.HK, AAPL)")
    parser.add_argument("--index", type=str, help="A股指数代码")
    parser.add_argument("--train-days", type=int, default=365, help="训练天数 (默认: 365)")
    parser.add_argument("--threshold", type=float, default=0.008, help="涨跌阈值 (默认: 0.008)")
    parser.add_argument("--output", choices=["text", "json", "csv"], default="text", help="输出格式 (默认: text)")
    parser.add_argument("--refresh", action="store_true", help="强制刷新数据缓存")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--multi-model", action="store_true", default=True, help="使用多模型集成 (默认开启)")
    parser.add_argument("--single-model", action="store_true", help="仅使用单个 CatBoost 模型")
    parser.add_argument("--ml-weight", type=float, default=0.35, help="ML模型权重 (默认: 0.35)")
    parser.add_argument("--technical-weight", type=float, default=0.25, help="技术分析权重 (默认: 0.25)")
    parser.add_argument("--momentum-weight", type=float, default=0.15, help="动量分析权重 (默认: 0.15)")
    parser.add_argument("--min-confidence", type=float, default=0.65, help="最低置信度阈值 (默认: 0.65)")
    parser.add_argument("--exclude-dates", action="store_true", help="排除极端波动日期")
    parser.add_argument("--exclude-threshold", type=float, default=2.0, help="极端波动检测阈值(标准差倍数)")
    parser.add_argument("--list", type=str, choices=["cn", "hk", "us"], help="列出指定区域的股票列表")
    parser.add_argument("--batch", type=str, help="批量预测，逗号分隔股票代码 (如 '000001.SZ,0700.HK')")
    parser.add_argument("--serve", action="store_true", help="启动 HTTP API 服务 (FastAPI)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="HTTP 服务监听地址")
    parser.add_argument("--port", type=int, default=8000, help="HTTP 服务监听端口")
    return parser.parse_args()


def print_stock_list(zone: str):
    suffixes = ZONE_SUFFIX[zone]
    stocks = []
    for code, name in STOCK_NAMES.items():
        if zone == "us":
            if "." not in code:
                stocks.append((code, name))
        else:
            if any(code.endswith(s) for s in suffixes):
                stocks.append((code, name))
    print(f"\n  {zone.upper()} Stocks ({len(stocks)}):")
    print("  " + "-" * 50)
    for code, name in sorted(stocks, key=lambda x: x[0]):
        print(f"  {code:<15} {name}")
    print()


def main():
    args = parse_args()

    if args.list:
        print_stock_list(args.list)
        return

    if not args.stock and not args.index and not args.batch:
        print("  Error: Please specify --stock, --index, --batch, or --list")
        return

    Path("logs").mkdir(exist_ok=True)

    if args.batch:
        codes = [c.strip() for c in args.batch.split(",") if c.strip()]
        if not codes:
            print("  Error: No valid stock codes in batch")
            return
        print("=" * 60)
        print("  BATCH PREDICTION")
        print("=" * 60)
        results = run_batch_prediction(
            codes=codes,
            train_days=args.train_days,
            threshold=args.threshold,
            refresh=args.refresh,
            multi_model=args.multi_model and not args.single_model,
            ml_weight=args.ml_weight,
            technical_weight=args.technical_weight,
            momentum_weight=args.momentum_weight,
            exclude_dates=args.exclude_dates,
            exclude_threshold=args.exclude_threshold,
        )
        formatter = PredictionFormatter()
        for result in results:
            output = formatter.format(result, args.output)
            print("\n" + output)
        print("\n" + "=" * 60)
        return

    if args.stock and args.index:
        print("  Error: Please specify only one of --stock or --index")
        return

    is_index = args.index is not None
    code = args.index if is_index else args.stock

    print("=" * 60)
    print("  STOCK PREDICTION SYSTEM (Enhanced)")
    print("=" * 60)

    if is_index:
        print(f"  Index Code   : {code}")
        print(f"  Index Name   : {get_index_name(code)}")
        market = "a_share"
    else:
        try:
            stock_info = StockInfoResolver.resolve(code)
            market = stock_info.market.replace("_share", "")
            name = STOCK_NAMES.get(code, code)
            print(f"  Stock Code : {code}")
            print(f"  Stock Name : {name}")
            print(f"  Market     : {market}")
        except ValueError as e:
            print(f"  Error: {e}")
            return

    stock_strategy = _service.resolve_strategy(code)
    if stock_strategy and not is_index:
        print(f"\n  Strategy: {stock_strategy.name}")
        print(f"  Description: {stock_strategy.description}")
        args = _apply_strategy(args, stock_strategy, code, market)

    use_multi_model = args.multi_model and not args.single_model

    print("\n  Fetching data...")
    result = run_prediction(
        code=code,
        is_index=is_index,
        train_days=args.train_days,
        threshold=args.threshold,
        refresh=args.refresh,
        multi_model=use_multi_model,
        ml_weight=args.ml_weight,
        technical_weight=args.technical_weight,
        momentum_weight=args.momentum_weight,
        exclude_dates=args.exclude_dates,
        exclude_threshold=args.exclude_threshold,
    )

    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    formatter = PredictionFormatter()
    output = formatter.format(result, args.output)
    print("\n" + output)

    if args.verbose:
        print("\n  === Verbose Info ===")
        print(f"  Threshold: {args.threshold}")
        print(f"  Train days: {args.train_days}")
        print(f"  Data range: {result.get('prediction_date', 'N/A')}")
        print(f"  ML weight: {args.ml_weight}")
        print(f"  Technical weight: {args.technical_weight}")
        print(f"  Momentum weight: {args.momentum_weight}")
        if is_index:
            print(f"  Index: {code} ({get_index_name(code)})")

    print("\n" + "=" * 60)
    print("  PREDICTION COMPLETE")
    print("=" * 60)


_executor = ThreadPoolExecutor(max_workers=8)


def start_server(host: str = "127.0.0.1", port: int = 8000):
    try:
        import uvicorn
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        print("Error: fastapi and uvicorn are required. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(
        title="Stock Prediction API",
        description="股票预测 HTTP API - 基于 CatBoost 集成模型",
        version="2.0.0",
    )

    # Internal auth middleware - protects all routes except /health
    INTERNAL_SECRET = os.environ.get("INTERNAL_API_SECRET")

    @app.middleware("http")
    async def internal_auth_middleware(request, call_next):
        if request.url.path == "/health" or request.url.path == "/docs" or request.url.path == "/openapi.json":
            return await call_next(request)
        if INTERNAL_SECRET:
            auth = request.headers.get("X-Internal-Auth")
            if auth != INTERNAL_SECRET:
                from fastapi.responses import JSONResponse
                return JSONResponse({"error": "Forbidden"}, status_code=403)
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "stock-prediction-api", "version": "2.0.0"}

    @app.get("/predict")
    async def predict(
        stock: str = Query(None, description="股票代码 (如 000001.SZ, 0700.HK, AAPL)"),
        index: str = Query(None, description="A股指数代码"),
        train_days: int = Query(365, description="训练天数"),
        threshold: float = Query(0.008, description="涨跌阈值"),
        refresh: bool = Query(False, description="强制刷新数据"),
        multi_model: bool = Query(True, description="使用多模型集成"),
        ml_weight: float = Query(0.35, description="ML模型权重"),
        technical_weight: float = Query(0.25, description="技术分析权重"),
        momentum_weight: float = Query(0.15, description="动量分析权重"),
        exclude_dates: bool = Query(False, description="排除极端波动日期"),
        fast_mode: bool = Query(True, description="快速模式"),
        skip_training: bool = Query(True, description="跳过模型训练"),
        skip_eval: bool = Query(True, description="跳过模型评估"),
        skip_realtime: bool = Query(True, description="跳过实时价格"),
        skip_params: bool = Query(False, description="跳过优化参数查找"),
    ):
        if not stock and not index:
            raise HTTPException(status_code=400, detail="Please provide stock or index parameter")
        if stock and index:
            raise HTTPException(status_code=400, detail="Please provide only one of stock or index")

        code = index if index else stock
        is_index = index is not None

        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    _executor, run_prediction,
                    code, is_index, train_days, threshold, refresh,
                    multi_model, ml_weight, technical_weight, momentum_weight,
                    exclude_dates, 2.0, fast_mode, skip_training, skip_eval,
                    skip_realtime, skip_params,
                ),
                timeout=120,
            )
        except asyncio.TimeoutError:
            logger.error(f"Prediction timeout for {code}")
            raise HTTPException(status_code=504, detail="Prediction timed out")
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
        if not stock and not index:
            raise HTTPException(status_code=400, detail="Please provide stock or index")
        code = index if index else stock
        cached = _service.get_cached_prediction(code)
        if cached:
            return cached
        raise HTTPException(status_code=404, detail=f"No cached prediction for {code}")

    @app.get("/predict/quick")
    async def predict_quick(
        stock: str = Query(None, description="股票代码"),
        index: str = Query(None, description="指数代码"),
    ):
        if not stock and not index:
            raise HTTPException(status_code=400, detail="Please provide stock or index")
        if stock and index:
            raise HTTPException(status_code=400, detail="Please provide only one of stock or index")

        code = index if index else stock
        is_index = index is not None

        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    _executor, run_prediction,
                    code, is_index, 90, 0.008, False,
                    False, 0.35, 0.25, 0.15,
                    False, 2.0, True, True, True, True, True,
                ),
                timeout=60,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Quick prediction timed out")
        except Exception as e:
            logger.exception(f"Quick prediction failed for {code}")
            raise HTTPException(status_code=500, detail=str(e))

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result

    @app.get("/predict/batch")
    async def predict_batch(
        stocks: str = Query(..., description="逗号分隔的股票代码 (如 000001.SZ,0700.HK)"),
        train_days: int = Query(365, description="训练天数"),
        threshold: float = Query(0.008, description="涨跌阈值"),
        fast_mode: bool = Query(True, description="快速模式"),
    ):
        codes = [c.strip() for c in stocks.split(",") if c.strip()]
        if not codes:
            raise HTTPException(status_code=400, detail="No valid stock codes provided")

        loop = asyncio.get_event_loop()
        try:
            results = await asyncio.wait_for(
                loop.run_in_executor(
                    _executor, run_batch_prediction, codes,
                ),
                timeout=120,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Batch prediction timed out")

        return {"results": results, "count": len(results)}

    @app.get("/stocks/{code}/info")
    async def stock_info(code: str):
        try:
            info = _service.resolve_stock(code)
            name = STOCK_NAMES.get(code.upper(), code)
            return {"code": code, "name": name, "market": info.market, "exchange": info.exchange}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/stocks")
    async def list_stocks(zone: str = Query(..., description="区域: cn, hk, us")):
        zone = zone.lower()
        if zone not in ZONE_SUFFIX:
            raise HTTPException(status_code=400, detail="Invalid zone. Must be one of: cn, hk, us")

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
        _service.warmup()

    @app.on_event("shutdown")
    async def shutdown_event():
        _executor.shutdown(wait=False)

    print("\n  Starting Stock Prediction API v2.0...")
    print(f"  Server : http://{host}:{port}")
    print(f"  Docs   : http://{host}:{port}/docs")
    print(f"  Health : http://{host}:{port}/health")
    print(f"  Predict: http://{host}:{port}/predict?stock=000001.SZ")
    print(f"  Quick  : http://{host}:{port}/predict/quick?stock=000001.SZ")
    print(f"  Batch  : http://{host}:{port}/predict/batch?stocks=000001.SZ,0700.HK")
    print(f"  Stocks : http://{host}:{port}/stocks?zone=cn")
    print()

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    args = parse_args()

    if args.serve:
        start_server(host=args.host, port=args.port)
    else:
        main()
