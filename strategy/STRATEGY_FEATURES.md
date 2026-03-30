# 策略特征文档 (Strategy Features for CatBoost)

本文档基于 `references/daily_stock_analysis/` 目录下的交易策略，定义适合 CatBoost 模型的特征体系。

> **图例**: ✅ = 已完成 (代码已实现) | 🔄 = 待开发 | ⚙️ = 部分实现

---

## 1. 均线与趋势特征 (MA & Trend Features)

### 1.1 均线价格特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `ma5` | 5日简单移动平均 | ma_golden_cross, bull_trend, shrink_pullback | ✅ technical.py |
| `ma10` | 10日简单移动平均 | ma_golden_cross, bull_trend, shrink_pullback | ✅ technical.py |
| `ma20` | 20日简单移动平均 | ma_golden_cross, bull_trend | ✅ technical.py |
| `ma60` | 60日简单移动平均 | box_oscillation, chan_theory | ✅ technical.py |
| `ma120` | 120日简单移动平均 | wave_theory | ✅ technical.py |
| `ma_5_ratio` | close / ma_5 | - | ✅ technical.py |
| `ma_10_ratio` | close / ma_10 | - | ✅ technical.py |
| `ma_20_ratio` | close / ma_20 | - | ✅ technical.py |
| `ma_60_ratio` | close / ma_60 | - | ✅ technical.py |

### 1.2 均线排列特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `ma_bullish_arrange` | MA5 >= MA10 >= MA20 | bull_trend, shrink_pullback | ✅ technical.py |
| `ma_bearish_arrange` | MA5 <= MA10 <= MA20 | - | ✅ technical.py |
| `ma5_above_ma10` | MA5 > MA10 为 1 | ma_golden_cross | ✅ technical.py |
| `ma10_above_ma20` | MA10 > MA20 为 1 | ma_golden_cross | ✅ technical.py |
| `ma5_above_20` | MA5 > MA20 为 1 | - | ✅ technical.py |
| `ma_slope_20` | MA20 斜率 | bull_trend | ✅ technical.py |

### 1.3 金叉死叉特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `golden_cross_5_10` | MA5 上穿 MA10 | ma_golden_cross | ✅ technical.py |
| `death_cross_5_10` | MA5 下穿 MA10 | ma_golden_cross | ✅ technical.py |
| `golden_cross_10_20` | MA10 上穿 MA20 | ma_golden_cross | ✅ technical.py |
| `death_cross_10_20` | MA10 下穿 MA20 | ma_golden_cross | ✅ technical.py |
| `golden_cross_5_20` | MA5 上穿 MA20 | - | ✅ technical.py |
| `death_cross_5_20` | MA5 下穿 MA20 | - | ✅ technical.py |
| `ma_cross_days` | 距离最近金叉交易日数 | ma_golden_cross | ⚙️ 可选实现 |

### 1.4 乖离率特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `deviation_ma5` | (close - MA5) / MA5 * 100 | shrink_pullback, dragon_head | ⚙️ 可用 ma_5_ratio 计算 |
| `deviation_ma10` | (close - MA10) / MA10 * 100 | shrink_pullback | ⚙️ 可用 ma_10_ratio 计算 |
| `deviation_ma20` | (close - MA20) / MA20 * 100 | bull_trend, emotion_cycle | ⚙️ 可用 ma_20_ratio 计算 |
| `deviation_ma5_abs` | 乖离率绝对值 | shrink_pullback | ✅ technical.py |

---

## 2. 量能特征 (Volume Features)

### 2.1 成交量基础特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `volume` | 当日成交量 (手) | bottom_volume, volume_breakout | ✅ technical.py |
| `volume_ma5` | 5日均量 | bottom_volume, ma_golden_cross | ✅ technical.py |
| `volume_ratio` | volume / volume_ma20 | bottom_volume, volume_breakout | ✅ technical.py |
| `volume_ma20` | 20日均量 | emotion_cycle | ✅ technical.py |
| `volume_change` | volume.pct_change() | - | ✅ technical.py |
| `volume_shrink_ratio` | volume / volume_ma5 | shrink_pullback | ⚙️ 可用 volume_ma5 计算 |

### 2.2 换手率特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `turnover_rate` | 当日换手率 (%) | emotion_cycle, dragon_head | ⚙️ yfinance暂无，需akshare/tushare |
| `turnover_ma20` | 20日平均换手率 | emotion_cycle | ⚙️ yfinance暂无，需akshare/tushare |
| `turnover_level` | 换手率分位 (0-1) | emotion_cycle | ⚙️ yfinance暂无，需akshare/tushare |
| `turnover_trend` | 近20日换手率趋势 | emotion_cycle | ⚙️ yfinance暂无，需akshare/tushare |

### 2.3 量能形态特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `volume_breakout_flag` | volume_ratio > 2.0 | volume_breakout | ✅ technical.py |
| `bottom_volume_flag` | volume_ratio > 3.0 且价格低位 | bottom_volume | ✅ technical.py |
| `shrink_pullback_flag` | volume_ratio < 0.7 且价格在MA附近 | shrink_pullback | ✅ technical.py |
| `volume_increasing` | 近3日量能连续放大 | emotion_cycle | ✅ technical.py |

---

## 3. MACD 与背驰特征 (MACD & Divergence Features)

### 3.1 MACD 基础指标
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `macd` | EMA12 - EMA26 | chan_theory, wave_theory | ✅ technical.py |
| `macd_signal` | MACD的EMA9 | chan_theory | ✅ technical.py |
| `macd_hist` | macd - macd_signal | chan_theory, wave_theory | ✅ technical.py |
| `macd_histogram_area` | MACD红绿柱面积累计 | chan_theory | 🔄 待开发 |

### 3.2 背驰信号特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `top_divergence` | 顶背驰: 价格新高但MACD红柱缩小 | chan_theory | ✅ technical.py |
| `bottom_divergence` | 底背驰: 价格新低但MACD绿柱缩小 | chan_theory | ✅ technical.py |
| `divergence_strength` | 背驰强度 (0-1) | chan_theory | ✅ technical.py |

### 3.3 MACD 交叉特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `macd_cross_up` | MACD 金叉 | ma_golden_cross | ✅ technical.py |
| `macd_cross_above_zero` | MACD 在零轴上方金叉 | ma_golden_cross | ✅ technical.py |
| `macd_position` | MACD 在零轴上方为 1 | chan_theory | ✅ technical.py |

---

## 4. K线形态特征 (Candlestick Pattern Features)

### 4.1 单K线特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `body_ratio` | K线实体长度 / 当日振幅 | one_yang_three_yin | ✅ technical.py |
| `upper_shadow_ratio` | 上影线长度 / 当日振幅 | bottom_volume | ✅ technical.py |
| `lower_shadow_ratio` | 下影线长度 / 当日振幅 | bottom_volume | ✅ technical.py |
| `is_bullish` | 收盘价 > 开盘价 | bottom_volume | ✅ technical.py |
| `close_position` | (close - low) / (high - low) | volume_breakout | ✅ technical.py |

### 4.2 一阳三阴形态特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `oyty_pattern` | 最近5日是否符合一阳三阴形态 | one_yang_three_yin | ✅ technical.py |
| `oyty_bullish_body` | 第1日大阳线实体 > 2% | one_yang_three_yin | ✅ technical.py |
| `oyty_shrink_volume` | 第2-4日量能 < 0.8倍 | one_yang_three_yin | ✅ technical.py |
| `oyty_support_hold` | 第2-4日最低价不破第1日开盘价 | one_yang_three_yin | ✅ technical.py |
| `oyty_breakout` | 第5日阳线突破第1日收盘价 | one_yang_three_yin | ✅ technical.py |

### 4.3 底部形态特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `bottom_volume_surge` | 跌幅>15%后放量>3倍 | bottom_volume | ✅ technical.py |
| `long_lower_shadow` | lower_shadow_ratio > 0.6 | bottom_volume | ✅ technical.py |
| `price_stabilize` | 阳线收盘且守住近期低点 | bottom_volume | ✅ technical.py |

---

## 5. 箱体与支撑阻力特征 (Box & Support/Resistance Features)

### 5.1 箱体识别特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `box_top` | 箱体顶部价位 | box_oscillation | ✅ technical.py |
| `box_bottom` | 箱体底部价位 | box_oscillation | ✅ technical.py |
| `box_width_pct` | (box_top - box_bottom) / box_bottom * 100 | box_oscillation | ✅ technical.py |
| `box_touch_top_count` | 近20日触碰箱顶次数 | box_oscillation | ✅ technical.py |
| `box_touch_bottom_count` | 近20日触碰箱底次数 | box_oscillation | ✅ technical.py |

### 5.2 位置与距离特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `distance_to_support` | (close - support) / support * 100 | box_oscillation, shrink_pullback | ✅ technical.py |
| `distance_to_resistance` | (resistance - close) / resistance * 100 | volume_breakout, box_oscillation | ✅ technical.py |
| `near_box_bottom` | 距箱底 <= 5% | box_oscillation | ✅ technical.py |
| `near_box_top` | 距箱顶 <= 5% | box_oscillation | ✅ technical.py |
| `in_box_middle` | 处于箱体中间1/3区域 | box_oscillation | ✅ technical.py |

### 5.3 突破信号特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `breakout_up` | 收盘价突破箱体顶部 | box_oscillation | ✅ technical.py |
| `breakout_down` | 收盘价跌破箱体底部 | box_oscillation | ✅ technical.py |
| `breakout_volume_confirm` | 突破时量能 > 2倍均量 | volume_breakout | ✅ technical.py |
| `false_breakout` | 盘中触及但收盘回到箱内 | box_oscillation | 🔄 需要盘中high数据 |

---

## 6. 波浪理论特征 (Elliott Wave Features)

### 6.1 浪型识别特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `wave_position` | 当前所处浪型 (1-5, A-C) | wave_theory | 🔄 待开发 |
| `wave_confidence` | 波浪计数置信度 (0-1) | wave_theory | 🔄 待开发 |
| `wave_3_strength` | 第3浪强度指标 | wave_theory | 🔄 待开发 |
| `wave_5_end` | 第5浪末端信号 | wave_theory | 🔄 待开发 |

### 6.2 斐波那契特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `fib_382_support` | 38.2% 斐波那契支撑位 | wave_theory | 🔄 待开发 |
| `fib_618_support` | 61.8% 斐波那契支撑位 | wave_theory | 🔄 待开发 |
| `fib_1618_target` | 161.8% 斐波那契目标位 | wave_theory | 🔄 待开发 |
| `wave2_retrace_pct` | 第2浪回调比例 | wave_theory | 🔄 待开发 |

### 6.3 调整浪特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `wave_a_in_progress` | A浪进行中信号 | wave_theory | 🔄 待开发 |
| `wave_b_rebound` | B浪反弹信号 | wave_theory | 🔄 待开发 |
| `wave_c_decline` | C浪下跌信号 | wave_theory | 🔄 待开发 |
| `flat_correction` | 平台型调整标识 | wave_theory | 🔄 待开发 |

---

## 7. 情绪周期特征 (Sentiment Cycle Features)

### 7.1 换手率情绪指标
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `sentiment_cold` | 换手率 < 0.5% | emotion_cycle | ⚙️ 需要turnover_rate数据 |
| `sentiment_normal` | 换手率 0.5%-2% | emotion_cycle | ⚙️ 需要turnover_rate数据 |
| `sentiment_hot` | 换手率 2%-5% | emotion_cycle | ⚙️ 需要turnover_rate数据 |
| `sentiment_overheat` | 换手率 > 5% | emotion_cycle | ⚙️ 需要turnover_rate数据 |
| `sentiment_extreme` | 换手率 > 10% | emotion_cycle | ⚙️ 需要turnover_rate数据 |

### 7.2 情绪底部/顶部特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `emotion_bottom_1` | 近20日换手率处于近一年低位 | emotion_cycle | ⚙️ 需要turnover_rate数据 |
| `emotion_bottom_2` | 成交量持续萎缩 < 均量50% | emotion_cycle | ✅ technical.py |
| `emotion_bottom_3` | 新闻以中性或负面为主 | emotion_cycle | 🔄 新闻数据不支持 |
| `emotion_bottom_4` | 股价在MA20附近或以下 | emotion_cycle | ✅ technical.py |
| `emotion_top_1` | 近5日换手率 > 20日均值2倍 | emotion_cycle | ⚙️ 需要turnover_rate数据 |
| `emotion_top_2` | 成交量脉冲式放大 | emotion_cycle | ✅ technical.py |
| `emotion_top_3` | 价格偏离MA5超过8% | emotion_cycle | ✅ technical.py |
| `emotion_top_4` | MACD出现顶背离 | emotion_cycle | ✅ technical.py |

### 7.3 均线收缩与波动率特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `ma_convergence` | MA5/MA10/MA20 三线粘合程度 | emotion_cycle | ✅ technical.py |
| `atr_shrinking` | ATR降至低位 | emotion_cycle | ✅ technical.py |
| `low_volatility_flag` | 波动率历史分位 < 20% | emotion_cycle | ✅ technical.py |

---

## 8. 龙头股特征 (Dragon Head Features)

### 8.1 板块排名特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `sector_rank` | 个股所在板块排名 | dragon_head | 🔄 待开发 |
| `sector_rotation_active` | 板块正处于轮动期 | dragon_head | 🔄 待开发 |
| `stock_leads_sector` | 个股涨幅 - 板块涨幅 | dragon_head | 🔄 待开发 |

### 8.2 龙头强度特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `turnover_rate_leader` | 换手率 > 5% | dragon_head | 🔄 待开发 |
| `volume_ratio_leader` | 量比 > 1.5 | dragon_head | 🔄 待开发 |
| `relative_strength` | 个股涨幅 - 板块涨幅 | dragon_head | 🔄 待开发 |
| `limit_up_signal` | 涨停信号 | dragon_head | 🔄 待开发 |

---

## 9. 缠论特征 (Chan Theory Features)

### 9.1 中枢结构特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `中枢数量` | 近期中枢个数 | chan_theory | 🔄 待开发 |
| `中枢区间_top` | 中枢区间最高价 | chan_theory | 🔄 待开发 |
| `中枢区间_bottom` | 中枢区间最低价 | chan_theory | 🔄 待开发 |
| `in_中枢` | 价格在中枢内 | chan_theory | 🔄 待开发 |
| `above_中枢` | 价格在中枢上方 | chan_theory | 🔄 待开发 |
| `below_中枢` | 价格在中枢下方 | chan_theory | 🔄 待开发 |

### 9.2 买卖点特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `buy_point_1` | 一买信号 (底背驰+中枢) | chan_theory | 🔄 待开发 |
| `buy_point_2` | 二买信号 (回调不破中枢高点) | chan_theory | 🔄 待开发 |
| `buy_point_3` | 三买信号 (向上突破不回中枢) | chan_theory | 🔄 待开发 |
| `sell_point_1` | 一卖信号 | chan_theory | 🔄 待开发 |
| `sell_point_2` | 二卖信号 | chan_theory | 🔄 待开发 |
| `sell_point_3` | 三卖信号 | chan_theory | 🔄 待开发 |

### 9.3 趋势状态
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `trend_up` | 上涨趋势 | chan_theory | 🔄 待开发 |
| `trend_down` | 下跌趋势 | chan_theory | 🔄 待开发 |
| `trend_sideways` | 中枢震荡 | chan_theory | 🔄 待开发 |

---

## 10. 市场与基本面特征 (Market & Fundamental Features)

### 10.1 市场数据
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `index_close` | 对应指数收盘价 | bull_trend | ✅ market.py |
| `index_returns` | 指数涨跌幅 | bull_trend | ✅ market.py |
| `index_rsi` | 指数RSI | - | ✅ market.py |
| `index_momentum_5` | 指数5日动能 | - | ✅ market.py |
| `index_momentum_10` | 指数10日动能 | - | ✅ market.py |
| `index_momentum_20` | 指数20日动能 | - | ✅ market.py |
| `index_volatility_5` | 指数5日波动率 | - | ✅ market.py |
| `index_volatility_20` | 指数20日波动率 | - | ✅ market.py |
| `index_ma_5` | 指数5日均线 | - | ✅ market.py |
| `index_ma_10` | 指数10日均线 | - | ✅ market.py |
| `index_ma_20` | 指数20日均线 | - | ✅ market.py |
| `market_sentiment` | 市场整体情绪 | emotion_cycle | ⚙️ 可用index_rsi等综合计算 |

### 10.2 基本面特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `pe_ratio` | 市盈率 | volume_breakout | ✅ fundamental.py |
| `forward_pe` | 滚动市盈率 | - | ✅ fundamental.py |
| `peg_ratio` | PEG比率 | - | ✅ fundamental.py |
| `pb_ratio` | 市净率 | - | ✅ fundamental.py |
| `ps_ratio` | 市销率 | - | ✅ fundamental.py |
| `roe` | 净资产收益率 | - | ✅ fundamental.py |
| `roa` | 资产收益率 | - | ✅ fundamental.py |
| `gross_margin` | 毛利率 | - | ✅ fundamental.py |
| `operating_margin` | 营业利润率 | - | ✅ fundamental.py |
| `net_margin` | 净利润率 | - | ✅ fundamental.py |
| `revenue_growth` | 营收增长率 | - | ✅ fundamental.py |
| `earnings_growth` | 盈利增长率 | - | ✅ fundamental.py |
| `earnings_quarterly_growth` | 季度盈利增长 | - | ✅ fundamental.py |
| `debt_to_equity` | 负债权益比 | - | ✅ fundamental.py |
| `current_ratio` | 流动比率 | - | ✅ fundamental.py |
| `quick_ratio` | 速动比率 | - | ✅ fundamental.py |
| `dividend_yield` | 股息率 | - | ✅ fundamental.py |
| `dividend_rate` | 股息 | - | ✅ fundamental.py |
| `payout_ratio` | 派息比率 | - | ✅ fundamental.py |
| `market_cap` | 市值 | - | ✅ fundamental.py |
| `enterprise_value` | 企业价值 | - | ✅ fundamental.py |
| `target_mean_price` | 分析师目标均价 | - | ✅ fundamental.py |
| `target_high_price` | 分析师目标最高价 | - | ✅ fundamental.py |
| `target_low_price` | 分析师目标最低价 | - | ✅ fundamental.py |
| `price_to_target` | 现价/目标价 | - | ✅ fundamental.py |
| `week_52_high` | 52周最高价 | - | ✅ fundamental.py |
| `week_52_low` | 52周最低价 | - | ✅ fundamental.py |
| `week_52_high_ratio` | 现价/52周高点 | - | ✅ fundamental.py |
| `week_52_low_ratio` | 现价/52周低点 | - | ✅ fundamental.py |
| `news_sentiment` | 新闻情绪 | emotion_cycle | 🔄 待开发 |

---

## 11. 综合信号特征 (Composite Signals)

### 11.1 相对表现特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `alpha` | 个股收益 - 市场收益 | - | ✅ combinator.py |
| `alpha_5d` | 5日Alpha | - | ✅ combinator.py |
| `alpha_10d` | 10日Alpha | - | ✅ combinator.py |
| `sector_relative` | 个股收益 - 板块收益 | dragon_head | ✅ combinator.py |
| `sector_relative_5d` | 5日板块相对收益 | dragon_head | ✅ combinator.py |
| `market_corr_5` | 5日市场相关性 | - | ✅ combinator.py |
| `market_corr_10` | 10日市场相关性 | - | ✅ combinator.py |
| `beta_5` | 5日Beta | - | ✅ combinator.py |
| `beta_20` | 20日Beta | - | ✅ combinator.py |
| `volume_vs_market` | 个股量能 - 市场量能 | emotion_cycle | ✅ combinator.py |

### 11.2 趋势与风险指标
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `trend_score` | 综合趋势得分 | bull_trend | ✅ technical.py |
| `signal_buy` | 买入信号综合 | ma_golden_cross, shrink_pullback | ✅ technical.py |
| `signal_sell` | 卖出信号综合 | volume_breakout, emotion_cycle | ✅ technical.py |
| `signal_hold` | 观望信号 | box_oscillation | ✅ technical.py |
| `max_drawdown_20d` | 近20日最大回撤 | emotion_cycle | ✅ technical.py |
| `volatility_20d` | 近20日波动率 (年化) | emotion_cycle | ✅ technical.py |
| `atr_14` | 14日平均真实波幅 | wave_theory, shrink_pullback | ✅ technical.py |
| `risk_level` | 风险等级 (低/中/高) | - | ✅ technical.py |

---

## 12. 行业板块特征 (Industry & Sector Features)

### 12.1 板块数据
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `sector` | 行业板块 | dragon_head | ✅ industry.py |
| `industry` | 细分行业 | dragon_head | ✅ industry.py |
| `sector_close` | 板块ETF收盘价 | dragon_head | ✅ industry.py |
| `sector_returns` | 板块收益率 | dragon_head | ✅ industry.py |
| `sector_volume` | 板块成交量 | - | ✅ industry.py |
| `sector_ma20` | 板块20日均线 | - | ✅ industry.py |
| `sector_ma_ratio` | 板块价格/均线 | - | ✅ industry.py |
| `sector_rsi` | 板块RSI | - | ✅ industry.py |

---

## 13. 技术指标补充特征

### 13.1 动量与趋势强度
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `momentum_5` | 5日动量 | - | ✅ technical.py |
| `momentum_10` | 10日动量 | - | ✅ technical.py |
| `momentum_20` | 20日动量 | - | ✅ technical.py |
| `momentum_acceleration` | 动量加速 (5日-10日) | - | ✅ technical.py |
| `return_2d` | 2日收益率 | - | ✅ technical.py |
| `return_3d` | 3日收益率 | - | ✅ technical.py |

### 13.2 波动率特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `volatility_5` | 5日波动率 | - | ✅ technical.py |
| `volatility_10` | 10日波动率 | - | ✅ technical.py |
| `volatility_20` | 20日波动率 | - | ✅ technical.py |
| `sharpe_like` | 类夏普比率 (momentum_5/vol_20d) | - | ✅ technical.py |
| `cv` | 变异系数 | - | ✅ technical.py |

### 13.3 超买超卖指标
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `rsi` | 相对强弱指标 | - | ✅ technical.py |
| `stoch_k` | 随机指标K值 | - | ✅ technical.py |
| `stoch_d` | 随机指标D值 | - | ✅ technical.py |
| `mfi` | 资金流量指数 | - | ✅ technical.py |
| `cci` | 商品通道指数 | - | ✅ technical.py |
| `adx` | 平均趋向指数 | - | ✅ technical.py |
| `bb_position` | 布林带位置 | - | ✅ technical.py |
| `bb_width` | 布林带宽度 | - | ✅ technical.py |

### 13.4 价格位置特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `high_20d` | 20日最高价 | - | ✅ technical.py |
| `low_20d` | 20日最低价 | - | ✅ technical.py |
| `high_low_ratio` | close / high_20d | - | ✅ technical.py |
| `close_low_ratio` | close / low_20d | - | ✅ technical.py |
| `price_position` | 价格在20日高低点位置 | - | ✅ technical.py |

### 13.5 连续涨跌特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `streak_up_2` | 连续上涨2天 | - | ✅ technical.py |
| `streak_up_3` | 连续上涨3天 | - | ✅ technical.py |
| `streak_down_2` | 连续下跌2天 | - | ✅ technical.py |
| `streak_down_3` | 连续下跌3天 | - | ✅ technical.py |

### 13.6 形态模式特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `pattern_drop_then_rise` | 大跌后大涨 (>5%跌, >5%涨) | - | ✅ technical.py |
| `pattern_rise_then_drop` | 大涨后大跌 (>5%涨, >5%跌) | - | ✅ technical.py |
| `pattern_vol_surge_rise` | 放量上涨 (量增2倍+价格涨) | - | ✅ technical.py |
| `pattern_vol_surge_drop` | 放量下跌 (量增2倍+价格跌) | - | ✅ technical.py |
| `pattern_reversal_up` | 近3日反转上涨 | - | ✅ technical.py |
| `pattern_reversal_down` | 近3日反转向下 | - | ✅ technical.py |

---

## 14. 高级技术指标特征

### 14.1 VWAP 指标
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `vwap` | 成交量加权平均价 = cumsum(price*volume) / cumsum(volume) | - | ✅ technical.py |
| `price_to_vwap` | close / vwap | - | ✅ technical.py |
| `price_vs_vwap` | close > vwap 为 1 | - | ✅ technical.py |

### 14.2 Aroon 指标
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `aroon_up` | (period - 距最高价天数) / period * 100 | - | ✅ technical.py |
| `aroon_down` | (period - 距最低价天数) / period * 100 | - | ✅ technical.py |
| `aroon_oscillator` | aroon_up - aroon_down | - | ✅ technical.py |
| `aroon_trend` | aroon_up > aroon_down 为 1 | - | ✅ technical.py |
| `aroon_dmi_bullish` | aroon看涨 + DMI看涨组合信号 | - | ✅ technical.py |
| `aroon_dmi_bearish` | aroon看跌 + DMI看跌组合信号 | - | ✅ technical.py |

### 14.3 Accumulation/Distribution (A/D)
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `ad_line` | 累计资金流量线 | - | ✅ technical.py |
| `ad_oscillator` | A/D线 - 5日A/D均线 | - | ✅ technical.py |

### 14.4 ROC (Rate of Change)
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `roc_5` | 5日变化率 = (close - close_5d_ago) / close_5d_ago * 100 | - | ✅ technical.py |
| `roc_10` | 10日变化率 | - | ✅ technical.py |
| `roc_20` | 20日变化率 | - | ✅ technical.py |

### 14.5 DMI (Directional Movement Index)
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `dmi_plus_di` | +DI 方向指标 | - | ✅ technical.py |
| `dmi_minus_di` | -DI 方向指标 | - | ✅ technical.py |
| `dmi_di_diff` | +DI - (-DI) | - | ✅ technical.py |
| `dmi_adx` | ADX 平均趋向指数 | - | ✅ technical.py |

### 14.6 滞后特征 (Lag Features)
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `return_lag_1` | 1日前收益率 | - | ✅ technical.py |
| `return_lag_2` | 2日前收益率 | - | ✅ technical.py |
| `return_lag_3` | 3日前收益率 | - | ✅ technical.py |
| `return_lag_5` | 5日前收益率 | - | ✅ technical.py |
| `volume_change_lag_1` | 1日前成交量变化 | - | ✅ technical.py |
| `volume_change_lag_2` | 2日前成交量变化 | - | ✅ technical.py |
| `volume_change_lag_3` | 3日前成交量变化 | - | ✅ technical.py |
| `volume_change_lag_5` | 5日前成交量变化 | - | ✅ technical.py |
| `rsi_lag_1` | 1日前RSI | - | ✅ technical.py |
| `rsi_lag_2` | 2日前RSI | - | ✅ technical.py |
| `rsi_lag_3` | 3日前RSI | - | ✅ technical.py |
| `macd_lag_1` | 1日前MACD | - | ✅ technical.py |
| `macd_lag_2` | 2日前MACD | - | ✅ technical.py |
| `macd_hist_lag_1` | 1日前MACD柱状图 | - | ✅ technical.py |
| `macd_hist_lag_2` | 2日前MACD柱状图 | - | ✅ technical.py |

### 14.7 滚动统计特征
| 特征名 | 计算方式 | 来源策略 | 已接入 |
|--------|----------|----------|--------|
| `returns_std_5` | 5日收益率标准差 | - | ✅ technical.py |
| `returns_std_10` | 10日收益率标准差 | - | ✅ technical.py |
| `returns_skew_10` | 10日收益偏度 | - | ✅ technical.py |
| `returns_skew_20` | 20日收益偏度 | - | ✅ technical.py |
| `expanding_drawdown` | 回撤 (从历史最高点) | - | ✅ technical.py |

---

## 15. ML模型优化 (Model Optimization)

### 15.1 复合标签系统 (Composite Labels)
| 组件 | 权重 | 说明 |
|------|------|------|
| 未来收益 | 20% | 基础条件 - 必须满足 |
| 趋势对齐 | 30% | MA多头排列/空头排列 |
| 动量确认 | 30% | RSI、MACD方向确认 |
| 市场环境 | 20% | 大盘涨跌配合 |

**买入条件**: 复合分数 >= 0.2 且 未来收益为正
**卖出条件**: 复合分数 <= -0.2 且 未来收益为负
**观望**: 其他情况

```python
# trainer.py 中的 _create_labels() 实现
# 复合标签比简单阈值标签产生更少但更高质量的信号
```

### 15.2 特征选择 (Feature Selection)
基于CatBoost初步模型的特征重要性筛选:
1. 训练50次迭代的快速模型获取重要性
2. 保留top 60%重要性的特征
3. 移除低方差特征 (< 0.0001)
4. 移除高度相关特征 (> 0.95)
5. 最终限制80个特征

```python
# trainer.py 中的 _select_features() 实现
```

### 15.3 类权重平衡 (Class Weight Balancing)
使用平方根倒频权重代替auto_class_weights:
```python
weight = min(sqrt(n / (n_classes * class_count)), max_weight=5.0)
```
这比Balanced权重更温和，避免过度拟合少数类。

### 15.4 模型集成 (Ensemble Bagging)
训练3个CatBoost模型，使用不同随机种子:
- 模型1: seed + 0
- 模型2: seed + 111
- 模型3: seed + 222

预测时使用软投票(平均概率):
```python
avg_probabilities = mean(all_model_probabilities, axis=0)
```

### 15.5 市场环境过滤器 (Market Regime Filter)
ML策略增加市场环境判断:
- **看涨市场**: 大盘日收益率 > 0.5%
- **看跌市场**: 大盘日收益率 < 0.5%
- **只在看涨/中性市场买入** (除非是非常确定的卖出信号)

```python
bear_market_threshold=0.005  # 大盘收益 > 0.5% 才算看涨
require_bull_market_for_buy=True  # 看跌市场只允许卖出
```

### 15.6 混合策略 (Hybrid Strategy)
结合ML和HighSellLowBuy的混合策略:
- **信号一致时**: 使用ML信号 (需要置信度 > 阈值)
- **信号不一致时**: 退回到简单策略 (更保守)
- **市场过滤**: 看跌市场禁止买入

```python
class HybridStrategy(Strategy):
    def __init__(self, model, lookback=10, threshold=0.10,
                 ml_confidence_threshold=0.50, bear_market_threshold=0.005):
        ...
```

**适用场景**:
- 港股等熊市环境: 混合策略优于纯ML
- A股等牛市环境: 纯ML策略表现更好

---

## 16. 已接入代码位置汇总

| 模块 | 文件路径 | 特征数量 |
|------|----------|----------|
| 技术指标 | `src/features/technical.py` | ~180个 |
| 基本面 | `src/features/fundamental.py` | ~30个 |
| 市场数据 | `src/features/market.py` | ~25个 |
| 行业板块 | `src/features/industry.py` | ~8个 |
| 特征组合 | `src/features/combinator.py` | ~10个 (组合特征) |
| **合计** | | **~253个** |

---

## 17. 待开发特征优先级

### 高优先级 (策略核心)
1. **换手率相关** - turnover_rate, sentiment_cold/hot 等 (需要akshare/tushare数据源)
2. **乖离率特征** - deviation_ma5/10/20_abs ✅ 已完成
3. **量能形态** - volume_breakout_flag, shrink_pullback_flag ✅ 已完成
4. **MACD背驰** - top_divergence, bottom_divergence ✅ 已完成

### 中优先级 (增强信号)
5. **K线形态** - body_ratio, lower_shadow_ratio, oyty_pattern ✅ 已完成
6. **箱体特征** - box_top, box_bottom, breakout_up/down ✅ 已完成
7. **情绪周期** - emotion_bottom/top 系列 ⚙️ 部分完成

### 低优先级 (高级特性)
8. **波浪理论** - wave_position, fib 系列 🔄 待开发
9. **缠论中枢** - 中枢结构, 买卖点 🔄 待开发
10. **龙头股** - sector_rank, limit_up_signal 🔄 待开发

---

## 18. 特征工程注意事项

### 18.1 缺失值处理
- 均线数据: 使用前向填充 (forward fill) 再后向填充
- 量比/换手率: 0 填充
- MACD: 使用 0 填充 (零轴位置)

### 18.2 特征标准化
- 百分比特征 (乖离率, 涨跌幅): 无需标准化
- 绝对值特征 (价格, 成交量): 取对数或分位数标准化
- 比率特征 (量比): 可直接使用

### 18.3 标签定义

**当前实现**: 复合多维度标签 (见15.1节)

| 标签 | 条件 |
|------|------|
| **买入 (1)** | 复合分数 >= 0.2 且 未来5日收益 > threshold*0.5 |
| **卖出 (-1)** | 复合分数 <= -0.2 且 未来5日收益 < -threshold*0.5 |
| **观望 (0)** | 其他情况 |

**参数配置**:
```yaml
training:
  forward_days: 5       # 预测周期
  threshold: 0.01        # 收益阈值 (1%)
  use_composite_labels: true
  trend_weight: 0.30     # 趋势权重
  momentum_weight: 0.30  # 动量权重
  market_weight: 0.20    # 市场权重
```
