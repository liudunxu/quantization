"""场景定义 - 支持不同优化场景

场景类型：
- decision: 交易决策场景（用于 decide.py）
- prediction: 涨跌预测场景（用于 predict.py）
"""

from typing import Any, Dict, List

# 场景定义
SCENARIOS = {
    # ===== decide.py 场景 =====
    "decision_default": {
        "description": "默认交易决策场景",
        "script": "decide",
        "scenario_type": "decision",
        "forward_days": 5,
        "threshold": 0.01,
        "metric": "composite",
        "strategies": ["ma_golden_cross", "box_oscillation", "bull_trend"],
        "optimize_params": ["forward_days", "threshold"],
    },
    "decision_aggressive": {
        "description": "激进交易决策场景 - 更多信号",
        "script": "decide",
        "scenario_type": "decision",
        "forward_days": 3,
        "threshold": 0.005,
        "metric": "total_return",
        "strategies": ["ma_golden_cross", "volume_breakout"],
        "optimize_params": ["forward_days", "threshold"],
    },
    "decision_conservative": {
        "description": "保守交易决策场景 - 更高准确率",
        "script": "decide",
        "scenario_type": "decision",
        "forward_days": 10,
        "threshold": 0.02,
        "metric": "sharpe_ratio",
        "strategies": ["bull_trend", "macd_divergence"],
        "optimize_params": ["forward_days", "threshold"],
    },
    # ===== predict.py 场景 =====
    "prediction_default": {
        "description": "默认预测场景 - 预测下个交易日涨跌",
        "script": "predict",
        "scenario_type": "prediction",
        "forward_days": 1,
        "threshold": 0.005,
        "metric": "accuracy",
        "optimize_params": [
            "threshold",
            "forward_days",
            "ml_weight",
            "technical_weight",
        ],
    },
    "prediction_aggressive": {
        "description": "激进预测场景 - 更多交易信号",
        "script": "predict",
        "scenario_type": "prediction",
        "forward_days": 1,
        "threshold": 0.003,
        "metric": "accuracy",
        "optimize_params": ["threshold", "forward_days"],
    },
    "prediction_conservative": {
        "description": "保守预测场景 - 更高准确率",
        "script": "predict",
        "scenario_type": "prediction",
        "forward_days": 2,
        "threshold": 0.01,
        "metric": "accuracy",
        "optimize_params": ["threshold", "forward_days"],
    },
    "prediction_2day": {
        "description": "2日预测场景 - 预测未来2个交易日",
        "script": "predict",
        "scenario_type": "prediction",
        "forward_days": 2,
        "threshold": 0.008,
        "metric": "accuracy",
        "optimize_params": ["threshold", "forward_days"],
    },
    "prediction_3day": {
        "description": "3日预测场景 - 预测未来3个交易日",
        "script": "predict",
        "scenario_type": "prediction",
        "forward_days": 3,
        "threshold": 0.01,
        "metric": "accuracy",
        "optimize_params": ["threshold", "forward_days"],
    },
    "prediction_f1": {
        "description": "F1优化预测场景 - 平衡精准率和召回率",
        "script": "predict",
        "scenario_type": "prediction",
        "forward_days": 1,
        "threshold": 0.005,
        "metric": "f1",
        "optimize_params": ["threshold", "forward_days"],
    },
}


def get_scenario(name: str) -> Dict[str, Any]:
    """获取场景配置

    Args:
        name: 场景名称

    Returns:
        场景配置字典，如果不存在返回默认决策场景
    """
    return SCENARIOS.get(name, SCENARIOS["decision_default"])


def list_scenarios(scenario_type: str = None) -> List[str]:
    """列出场景

    Args:
        scenario_type: 场景类型过滤 ('decision' 或 'prediction')

    Returns:
        场景名称列表
    """
    if scenario_type:
        return [
            k for k, v in SCENARIOS.items() if v.get("scenario_type") == scenario_type
        ]
    return list(SCENARIOS.keys())


def get_scenario_params(scenario_name: str) -> Dict[str, Any]:
    """获取场景的模型参数

    Args:
        scenario_name: 场景名称

    Returns:
        模型参数字典
    """
    scenario = get_scenario(scenario_name)
    return {
        "forward_days": scenario.get("forward_days", 1),
        "threshold": scenario.get("threshold", 0.01),
    }
