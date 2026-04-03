# Stock Trading ML Decision System - 项目路线图

## 📅 版本历史

### v1.0 - 核心功能 (已完成)
- [x] 多数据源支持（baostock, akshare, yfinance, openbb）
- [x] 技术指标特征工程（100+ 特征）
- [x] CatBoost 机器学习模型
- [x] 回测引擎（9种规则策略 + 4种ML策略）
- [x] 参数优化系统
- [x] 交易决策脚本（decide.py）
- [x] 回测对比脚本（backtest.py）

### v1.1 - 预测功能 (已完成)
- [x] 涨跌预测脚本（predict.py）
- [x] 技术信号生成器（8种指标）
- [x] 集成预测器（多信号源投票）
- [x] 详细预测解释（看涨/看跌因素）
- [x] 支持多种输出格式（text/json/csv）

### v1.2 - 优化增强 (已完成)
- [x] 策略参数 SQLite 存储
- [x] 参数优先级系统（股票 > 市场 > 默认）
- [x] 场景化参数优化（decision/prediction）
- [x] 核心流水线模块（pipelines）

### v1.3 - 预测系统增强 (已完成)
- [x] 18种技术信号（ADX, MFI, CCI, DMI, RSI背离, 量价背离, Ichimoku, Williams %R, OBV, 连续涨跌等）
- [x] 7信号源集成（ML, 技术, 动量, 趋势, Alpha, 策略叠加, 支撑阻力）
- [x] 多时间框架趋势确认（短期5日/中期20日/长期60日）
- [x] 共识机制（≥3信号同意加成置信度）
- [x] Top3投票机制（按权重排序前3投票）
- [x] 市场状态过滤（震荡市场降低信号置信度）
- [x] Composite labels训练（结合returns/trend/momentum/market）
- [x] Precision/Recall/F1评估指标
- [x] 高置信度过滤机制
- [x] 15种规则策略信号叠加

---

## 🚀 未来计划

### v1.4 - 多模型集成 (已完成)
- [x] 支持 LightGBM 模型
- [x] 支持 XGBoost 模型
- [x] 模型投票集成 (MultiModelEnsemble)
- [x] 模型权重自动优化
- [x] explore_params 支持多模型参数探索

### v1.5 - 实时预测 (计划中)
- [ ] WebSocket 实时数据接入
- [ ] 实时信号推送
- [ ] 价格预警系统
- [ ] 盘中信号更新

### v1.6 - 组合优化 (计划中)
- [ ] 多股票组合管理
- [ ] 资金分配优化
- [ ] 风险预算控制
- [ ] 组合再平衡策略

### v2.0 - Web 界面 (规划中)
- [ ] Web Dashboard
- [ ] 可视化回测报告
- [ ] 策略性能监控
- [ ] 参数调优界面

---

## 🎯 当前重点

### 短期目标 (1-2周)
1. **多模型支持** ✅ 已完成
   - LightGBM 集成 ✅
   - XGBoost 集成 ✅
   - 模型对比功能 ✅

2. **参数优化增强**
   - 贝叶斯优化
   - 交叉验证优化
   - 在线学习支持

3. **文档完善**
   - 更新 API 文档
   - 添加更多代码示例
   - 完善故障排除指南

### 中期目标 (1-2月)
1. **实时预测增强**
   - WebSocket 实时数据接入
   - 实时信号推送
   - 价格预警系统

2. **组合管理**
   - 多股票组合支持
   - 资金分配优化
   - 风险预算控制

### 长期目标 (3-6月)
1. **实时交易支持**
   - 实时数据接入
   - 信号自动推送
   - 交易执行接口

2. **组合管理**
   - 多股票支持
   - 资金管理
   - 风险控制

---

## 📊 技术债务

### 已改进 ✅
- [x] 增加单元测试覆盖率 (+15 tests)
- [x] 优化内存使用（FeatureCombinator 批量合并 + gc）
- [x] 改进错误处理和日志（替换 bare except）
- [x] 添加性能监控（Timer, PerformanceTracker）

### 仍需改进
- [ ] 统一代码风格
- [ ] 添加更多类型注解
- [ ] 完善 docstring
- [ ] 重构长函数 (>50 行)

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 如何贡献
1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 提交规范
```
feat: 添加新功能
fix: 修复bug
docs: 文档更新
style: 代码格式调整
refactor: 代码重构
test: 测试相关
chore: 构建/工具相关
```

---

## 📝 备注

- 本项目仅供学习研究使用，不构成投资建议
- 股市有风险，投资需谨慎
- 模型预测结果仅供参考

---

**最后更新**: 2026-04-03
