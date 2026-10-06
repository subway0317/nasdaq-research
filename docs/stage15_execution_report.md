# Stage 15 — PIT 基本面信息增量价值多年 Walk-Forward 消融实验

正式结论：**NO_ROBUST_INCREMENTAL_FUNDAMENTAL_VALUE**；**NO_ROBUST_MULTI_YEAR_SIGNAL**。完整工程与协议验收：**PASS**。

加入完整已登记 PIT 基本面 block 后，两个固定模型均显著劣于 Relative Market Control 和 naive baselines。Treatment OLS 有 1/22 folds 改善，Ridge 有 2/22 folds 改善。所有 Treatment 设计矩阵均秩亏，同时出现新的大幅外推与极端预测；没有非有限预测或模型拟合失败。该负面结果在数据、PIT、purge、训练预处理、复现与 Final-Test-lock 验收通过的前提下成立。

## 1. Scope / 执行范围

本阶段只检验完整已登记 PIT 基本面信息的增量价值。固定 OLS、Ridge(alpha=1)、22 个季度 expanding folds、5d target 和 Stage 10 预处理。没有新模型、新特征、基本面 family search、missingness indicators、tuning、network、Final Test 或 FDE 开发。没有统计显著性检验、交易回测或 alpha 声称。

新增 [fundamental_ablation.py](../src/nasdaq_research/fundamental_ablation.py) 和 [test_fundamental_ablation.py](../tests/test_fundamental_ablation.py)。成熟历史模块全部保持原字节。Control 直接复用 Stage 11 `fit_arm`；Treatment 通过双射列名 adapter 复用未修改的 Stage 10 `fit_linear(ols_all/ridge_all)`。Adapter 只把 Relative SMA 名称临时映射回已有 whitelist 名称，值仍然是相对 SMA；fit 后恢复真实标签。

先保存初始 repository state，再冻结 [NVDA_stage15_protocol.json](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_protocol.json) 和上游哈希；随后加载 authority、重建 Control、通过复现/PIT/覆盖率审计后才运行 Treatment。Protocol SHA-256：`a74822e331126644ec219a645ba4946fa175533c6df6e2bb633c3656807892fa`。Registration 绑定 protocol 与上游 manifest 的 hashes。

## 2. Initial Repository State / 初始状态

真实初始 cwd：`/home/zbw21/projects/nasdaq-research`；branch：`main`；HEAD：`9e8b2448bd11a520b271d44eee03b52333b9fce5`；origin/main 相同；origin：`https://github.com/subway0317/nasdaq-research.git`。Working tree clean，`git diff --check` exit code 0、无输出。研究执行日期按用户上下文为 2026-10-06。

```text
9e8b244 Complete stage 14 multi-year walk-forward validation
c3da396 Approve stage 13 canonical market data authority
04e2789 Complete legacy evidence recovery audit
d6c4b2c Record stage 13 history expansion and unresolved source reconciliation
bde8ed9 Complete stage 12 post-representation signal diagnostics
```

完整命令、stdout、stderr 和 exit codes：[NVDA_stage15_initial_repository_state.json](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_initial_repository_state.json)。无未知用户修改需要回退或覆盖。

## 3. Inherited Research State / 继承的研究状态

Stage 14：`NO_ROBUST_MULTI_YEAR_SIGNAL` 与 `BROAD_REPRESENTATION_STABILITY_SUPPORT`。Stage 13=`INCOMPLETE_OR_BLOCKED`；Stage 13.1=`UNRESOLVED_BLOCKER`，root cause=`UNRESOLVED_SOURCE_DISCREPANCY`；Legacy Recovery=`LEGACY_EVIDENCE_NOT_RECOVERABLE`。这些历史状态完全保留，未重新调查 legacy 根因。

## 4. Canonical Data Authority / 数据权威

唯一使用 Stage 13.2 批准的 `NVDA_YAHOO_STAGE13_2020_2026_V1`，approval status 为 `APPROVED_WITH_DOCUMENTED_NONMATERIAL_CROSS_VINTAGE_DIFFERENCE`，expanded dataset approved 与 Stage14 modeling eligible 均为 true。`require_canonical_vintage` 核对 24 个 source/code hashes 与完整 Treatment scope。

Canonical raw market 4304 rows，research matrix 1695 sessions。仅 1620 个 development rows（2020-01-02–2026-06-12）进入本阶段 feature/label matrix；只加载 numerical 5d target。未来 filing records 和 gap/Final-Test/tail X/Y 在数值解析前过滤。原始交易日历读取日期用于 presample PIT 生效时点，不将 Volume 或未来价格载入模型。

数据源与完整范围证据：[NVDA_stage15_canonical_input_audit.json](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_canonical_input_audit.json)。没有下载、刷新、patch、hybrid vintage 或 source fallback。

## 5. Stage 14 Control Authority / 冻结对照

Control authority 包括 Stage14 protocol、fold registration、fold inventory、feature manifest、target purge、OOF predictions、saved model parameters 与 fit membership。读取并核对冻结 hashes，重新构造的 inventory、purges、feature manifest 与历史 CSV 完全同字节。

复用 22 folds：2021Q1–2026Q2 partial，末 validation date 为 2026-06-12；1367 OOS 日期。每个 fold 训练 label 的 5d exit 必须严格早于 validation 首个 session，每 fold purge 5 rows，共 110 次边界排除，0 violations。没有重新设计 fold、删除早期年份或改变样本。

[NVDA_stage15_fold_inventory.csv](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_fold_inventory.csv)；[NVDA_stage15_target_purge_audit.csv](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_target_purge_audit.csv)。训练规模从 248 扩大到 1564，validation 共 1367 rows。

## 6. Control Reproduction / 完整复现

检查 2734 个 learned prediction values（1367 dates × OLS/Ridge），coverage=100%，missing=0，duplicates=0，最大 absolute difference=0.0，预注册 tolerance `rtol=0, atol=1e-12`，tolerance violations=0。两项 baselines 完全重现。训练 membership、全部 saved preprocessing statistics、coefficients/intercepts 也与 Stage14 一致。

[NVDA_stage15_stage14_control_reproduction.csv](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_stage14_control_reproduction.csv)；[NVDA_stage15_control_reproduction_summary.json](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_control_reproduction_summary.json)。失败门控测试确认复现失败时不会调用任何 Treatment fit。

## 7. Fundamental Feature Manifest / 完整白名单

实际为 13 quarterly、13 annual、4 balance-sheet，共 30 fundamentals；加 9 Relative Market 后为 39 initial candidates，与预期一致。顺序继承 `research_features.FEATURE_COLUMNS`，并与 Stage13 protocol 和 Stage13.2 scope 交叉核对。下表按 family 展示，实际模型列顺序以 manifest 为准。

**Quarterly（13）**：`q_gross_margin`、`q_operating_margin`、`q_net_margin`、`q_rd_to_revenue`、`q_sga_to_revenue`、`q_ocf_margin`、`q_fcf_margin`、`q_revenue_growth_yoy`、`q_gross_profit_growth_yoy`、`q_operating_income_growth_yoy`、`q_net_income_growth_yoy`、`q_rd_growth_yoy`、`q_operating_cash_flow_growth_yoy`。

**Annual（13）**：`fy_gross_margin`、`fy_operating_margin`、`fy_net_margin`、`fy_rd_to_revenue`、`fy_sga_to_revenue`、`fy_ocf_margin`、`fy_fcf_margin`、`fy_revenue_growth_yoy`、`fy_gross_profit_growth_yoy`、`fy_operating_income_growth_yoy`、`fy_net_income_growth_yoy`、`fy_rd_growth_yoy`、`fy_operating_cash_flow_growth_yoy`。

**Balance sheet（4）**：`current_ratio`、`cash_to_assets`、`liabilities_to_assets`、`debt_to_assets`。

Relative Market：`simple_return`、`log_return`、`intraday_return`、`daily_range`、`close_to_sma_5`、`close_to_sma_20`、`close_to_sma_60`、`rolling_volatility_20`、`rolling_volatility_60`。

[NVDA_stage15_control_feature_manifest.csv](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_control_feature_manifest.csv)；[NVDA_stage15_fundamental_feature_manifest.csv](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_fundamental_feature_manifest.csv)；[NVDA_stage15_treatment_feature_manifest.csv](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_treatment_feature_manifest.csv)。没有新增 ratio、valuation、interaction、age 或 missingness feature。

## 8. PIT Semantics / 信息可得性

延续 filing date → 严格之后第一个 observed trading session 的生效规则。Quarterly、annual 与 balance-sheet 分别继承 whole-observation 状态选择；filing、period、quarterly/annual、amendment 与 accession 排序规则均由原 Stage5/6 code 执行，未重定义。Period end 不用于提前回填。YoY reference 必须在 source filing 时已可知，并保留原可比期间 guards。

完整 raw 日历自 2009 年起提供早于 2020 的真实 observed activation dates。冻结状态进入 2020 后保留其原生效日期，不能把 sample 起点当作 filing 生效时点。

## 9. PIT Leakage Audit / 审计结果

逐日×逐特征 48600 checks；全部 48600 个 cell 与 frozen snapshot 精确相同，包括 missingness；future information violations=0。Independent raw-source state/BS formula oracle：56700 cell checks，0 violations；YoY reference/growth oracle：546 checks，0 violations。沿原代码重新 replay 48600 feature cells，numeric tolerance `rtol=1e-10, atol=1e-12` 内 0 violations。

[NVDA_stage15_pit_leakage_audit.csv](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_pit_leakage_audit.csv) 保存 feature date、source filing、filing/effective dates、next-session oracle、eligibility、availability、reference chronology 与 exact canonical state；[NVDA_stage15_pit_state_replay_audit.json](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_pit_state_replay_audit.json) 保存总体审计。

Adversarial tests 在隔离内存副本中改变 future filing 的 R&D，验证生效前状态及预测逐值不变、effective date 后 feature state 才改变；注入未来 filing metadata 或 period-end activation 均被识别。Validation 第一日也逐 fold 核对 then-known state。

## 10. Fundamental Coverage / 训练覆盖率

筛选只用每 fold effective train：coverage ≥0.50 保留；training median fill；exact zero variance removal；training StandardScaler，Y 不标准化。没有 complete-case filtering。Validation 缺失使用冻结 training median，既不删 validation rows，也不改变 feature selection。

| Fold | Train | Validation | Fundamental selected | Low coverage dropped | Zero variance dropped | Treatment final count |
| --- | --- | --- | --- | --- | --- | --- |
| 2021Q1 | 248 | 61 | 20 | 10 | 0 | 29 |
| 2021Q2 | 309 | 63 | 23 | 7 | 0 | 32 |
| 2021Q3 | 372 | 64 | 23 | 7 | 0 | 32 |
| 2021Q4 | 436 | 64 | 23 | 7 | 0 | 32 |
| 2022Q1 | 500 | 62 | 21 | 9 | 0 | 30 |
| 2022Q2 | 562 | 62 | 24 | 6 | 0 | 33 |
| 2022Q3 | 624 | 64 | 24 | 6 | 0 | 33 |
| 2022Q4 | 688 | 63 | 26 | 4 | 0 | 35 |
| 2023Q1 | 751 | 62 | 26 | 4 | 0 | 35 |
| 2023Q2 | 813 | 62 | 26 | 4 | 0 | 35 |
| 2023Q3 | 875 | 63 | 26 | 4 | 0 | 35 |
| 2023Q4 | 938 | 63 | 26 | 4 | 0 | 35 |
| 2024Q1 | 1001 | 61 | 26 | 4 | 0 | 35 |
| 2024Q2 | 1062 | 63 | 26 | 4 | 0 | 35 |
| 2024Q3 | 1125 | 64 | 26 | 4 | 0 | 35 |
| 2024Q4 | 1189 | 64 | 26 | 4 | 0 | 35 |
| 2025Q1 | 1253 | 60 | 26 | 4 | 0 | 35 |
| 2025Q2 | 1313 | 62 | 26 | 4 | 0 | 35 |
| 2025Q3 | 1375 | 64 | 26 | 4 | 0 | 35 |
| 2025Q4 | 1439 | 64 | 26 | 4 | 0 | 35 |
| 2026Q1 | 1503 | 61 | 26 | 4 | 0 | 35 |
| 2026Q2 | 1564 | 51 | 26 | 4 | 0 | 35 |

660 个 fold×fundamental exposures 中，548 个 selected，112 个低覆盖率剔除，fundamental exact zero variance 剔除为 0。每 fold OLS/Ridge feature lists 和 scaler statistics 相同。

| 被剔除的 feature | 低覆盖率 folds |
| --- | --- |
| debt_to_assets | 4 |
| fy_fcf_margin | 22 |
| q_fcf_margin | 22 |
| q_gross_profit_growth_yoy | 2 |
| q_net_income_growth_yoy | 2 |
| q_ocf_margin | 22 |
| q_operating_cash_flow_growth_yoy | 22 |
| q_operating_income_growth_yoy | 7 |
| q_rd_growth_yoy | 7 |
| q_revenue_growth_yoy | 2 |

Selected feature exposures 的 validation missing fraction 平均为 1.0725%，最大为 100%；即使一个已选 feature 在 validation 中完全缺失，仍按训练 median 预测全部日期。first_available_date 仅为 descriptive development metadata，不参与 selector。

[NVDA_stage15_fundamental_coverage.csv](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_fundamental_coverage.csv)；[NVDA_stage15_preprocessing_audit.csv](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_preprocessing_audit.csv) 保存 nonmissing counts、coverage、selected/drop reason、median、scaler means/scales 和 validation missingness。

## 11. Model Specifications / 固定规格

| 模型 | 输入 / 规则 |
| --- | --- |
| relative_ols | Relative Market；OLS with intercept |
| relative_ridge_alpha1 | Relative Market；Ridge alpha=1，solver=svd |
| relative_fundamentals_ols | Relative Market + full registered PIT block；OLS |
| relative_fundamentals_ridge_alpha1 | Relative Market + full registered PIT block；Ridge alpha=1，solver=svd |
| zero_return | prediction=0.0 |
| historical_mean | 当前 effective train 的 5d Y 均值 |

共 88 learned fits（22×4）；没有 Original Market competitor、alpha tuning、family search 或 secondary-target evaluation。唯一 target 为 `Close[t+5]/Open[t+1]-1`，observed-session positions，与 Stage14 相同。

## 12. Fold-Level Results / 每折 MAE

| Fold | Zero | Hist mean | C OLS | T OLS | C Ridge(1) | T Ridge(1) |
| --- | --- | --- | --- | --- | --- | --- |
| 2021Q1 | 0.048664 | 0.049180 | 0.048751 | 0.049827 | 0.048038 | 0.049003 |
| 2021Q2 | 0.049450 | 0.044624 | 0.056105 | 0.095613 | 0.055585 | 0.092955 |
| 2021Q3 | 0.035466 | 0.039659 | 0.034458 | 0.098634 | 0.034975 | 0.082036 |
| 2021Q4 | 0.058283 | 0.055816 | 0.056687 | 0.066986 | 0.056341 | 0.062951 |
| 2022Q1 | 0.069981 | 0.072662 | 0.084775 | 0.131612 | 0.083025 | 0.127398 |
| 2022Q2 | 0.075286 | 0.080015 | 0.090709 | 0.178342 | 0.090181 | 0.173087 |
| 2022Q3 | 0.059257 | 0.060884 | 0.061130 | 0.121498 | 0.060963 | 0.115329 |
| 2022Q4 | 0.063921 | 0.062242 | 0.059161 | 0.074759 | 0.059636 | 0.074327 |
| 2023Q1 | 0.060075 | 0.055542 | 0.060260 | 0.085635 | 0.060164 | 0.080094 |
| 2023Q2 | 0.048950 | 0.046089 | 0.047197 | 0.070558 | 0.046278 | 0.067026 |
| 2023Q3 | 0.044464 | 0.045253 | 0.045904 | 0.332625 | 0.045955 | 0.317984 |
| 2023Q4 | 0.040274 | 0.039943 | 0.041746 | 0.179793 | 0.041805 | 0.155992 |
| 2024Q1 | 0.059697 | 0.053917 | 0.053289 | 0.490648 | 0.053508 | 0.270125 |
| 2024Q2 | 0.053478 | 0.049492 | 0.048169 | 0.312795 | 0.048098 | 0.103789 |
| 2024Q3 | 0.070751 | 0.071623 | 0.072631 | 0.104527 | 0.072644 | 0.094594 |
| 2024Q4 | 0.042542 | 0.042452 | 0.041840 | 0.047414 | 0.041855 | 0.044166 |
| 2025Q1 | 0.070349 | 0.073353 | 0.075478 | 0.224005 | 0.075786 | 0.082717 |
| 2025Q2 | 0.055102 | 0.049077 | 0.048453 | 0.168924 | 0.048679 | 0.088848 |
| 2025Q3 | 0.029312 | 0.027852 | 0.027132 | 0.036309 | 0.027110 | 0.029456 |
| 2025Q4 | 0.036841 | 0.040063 | 0.041389 | 0.036400 | 0.041320 | 0.038535 |
| 2026Q1 | 0.033795 | 0.035404 | 0.034998 | 0.037109 | 0.034992 | 0.034907 |
| 2026Q2 | 0.047798 | 0.045890 | 0.046278 | 0.060753 | 0.046273 | 0.053649 |

[NVDA_stage15_fold_metrics.csv](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_fold_metrics.csv)；[NVDA_stage15_oof_predictions.csv](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_oof_predictions.csv)。每模型、每日期恰好一个预测，无 duplicate 或 unexpected missing。

## 13. Incremental Comparison / 增量比较

| 模型 | Mean T MAE | Mean C MAE | Mean Δ | Median Δ | Wins | Losses | Ties | Win rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| relative_fundamentals_ols | 0.136580 | 0.053479 | 0.083101 | 0.035702 | 1 | 21 | 0 | 4.55% |
| relative_fundamentals_ridge_alpha1 | 0.101771 | 0.053328 | 0.048443 | 0.021349 | 2 | 20 | 0 | 9.09% |

Δ=Treatment MAE−对应 Control MAE，负数才表示改善；采用 equal-weight mean fold aggregation，tie 不算 win。[NVDA_stage15_incremental_comparison.csv](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_incremental_comparison.csv) 同时保存逐 fold 和 aggregate pairs。

## 14. Incremental Fundamental Value Classification / 第一轴

结论：**NO_ROBUST_INCREMENTAL_FUNDAMENTAL_VALUE**。

| 模型 | Mean Δ<0 | Median Δ<0 | Win rate>50% | 三项全通过 |
| --- | --- | --- | --- | --- |
| relative_fundamentals_ols | 未通过 | 未通过 | 未通过 | 未通过 |
| relative_fundamentals_ridge_alpha1 | 未通过 | 未通过 | 未通过 | 未通过 |

Protocol 在结果前将 broad OOS improvement 操作化为 mean paired delta<0 且 strict fold win rate>50%，继承 Stage14 的 mean-and-majority 思路；显式 robust 数值谓词优先于 prompt 的示例性 mixed 描述。两模型均无 broad improvement，均未满足完整三项，按预注册 precedence 判为 NO_ROBUST；不存在结果后改分类标准。

## 15. Absolute Baseline Comparison / 绝对预测比较

| Treatment | Baseline | Mean baseline MAE | Mean Δ | Median Δ | Wins/losses/ties | Win rate |
| --- | --- | --- | --- | --- | --- | --- |
| relative_fundamentals_ols | zero_return | 0.052443 | 0.084138 | 0.039970 | 1/21/0 | 4.55% |
| relative_fundamentals_ols | historical_mean | 0.051865 | 0.084715 | 0.041947 | 1/21/0 | 4.55% |
| relative_fundamentals_ridge_alpha1 | zero_return | 0.052443 | 0.049329 | 0.021931 | 0/22/0 | 0.00% |
| relative_fundamentals_ridge_alpha1 | historical_mean | 0.051865 | 0.049906 | 0.023761 | 3/19/0 | 13.64% |

[NVDA_stage15_absolute_baseline_comparison.csv](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_absolute_baseline_comparison.csv)。Zero mean fold MAE=0.052443，Historical Mean=0.051865。两项 baseline 只依赖 effective train 或常数，不依赖 validation Y。

## 16. Absolute Predictive Signal Classification / 第二轴

结论：**NO_ROBUST_MULTI_YEAR_SIGNAL**。直接调用 Stage14 的 `predictive_assessment`，仅双射更换 Treatment model identity，保留六项 gates 与 classification precedence。

**relative_fundamentals_ols**：两个 mean-fold baseline 优胜、两个 median delta<0、两个 strict fold win rate>50%，全部六项均未通过。

**relative_fundamentals_ridge_alpha1**：两个 mean-fold baseline 优胜、两个 median delta<0、两个 strict fold win rate>50%，全部六项均未通过。

Neither model 同时满足两项 mean-fold baseline wins 与两项多数 fold wins，因此按 Stage14 NO_ROBUST predicate 分类；不能将个别 fold 改善解释为 robust alpha。两个研究轴独立评定。

## 17. Numerical Stability / 线性代数与系数

Rank 和 condition number 使用训练标准化设计矩阵（含 intercept 列）。所有 44 个 Treatment fits 的设计矩阵均秩亏（两模型共享 X），OLS 有效特征 29–35，rank 14–31，rank deficiency 5–18；condition number 最大约 7.43e31。Control 每 fold rank=10（9 features+intercept），无秩亏，最大 condition number 97.80。

| Fold | Initial | Coverage kept | Zero variance | Final | Rank(+intercept) | Deficiency | Condition | OLS coef L2 | Ridge coef L2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021Q1 | 39 | 29 | 0 | 29 | 14 | 16 | 7.4255e+31 | 0.383017 | 0.071027 |
| 2021Q2 | 39 | 32 | 0 | 32 | 15 | 18 | 3.0683e+16 | 0.344729 | 0.071490 |
| 2021Q3 | 39 | 32 | 0 | 32 | 16 | 17 | 2.6918e+16 | 0.343027 | 0.153922 |
| 2021Q4 | 39 | 32 | 0 | 32 | 17 | 16 | 2.4288e+16 | 0.330253 | 0.163829 |
| 2022Q1 | 39 | 30 | 0 | 30 | 18 | 13 | 2.4364e+16 | 0.306102 | 0.129824 |
| 2022Q2 | 39 | 33 | 0 | 33 | 19 | 15 | 2.2417e+16 | 0.326234 | 0.103406 |
| 2022Q3 | 39 | 33 | 0 | 33 | 20 | 14 | 2.3839e+16 | 0.323600 | 0.134862 |
| 2022Q4 | 39 | 35 | 0 | 35 | 21 | 15 | 2.1731e+16 | 0.297700 | 0.112831 |
| 2023Q1 | 39 | 35 | 0 | 35 | 22 | 14 | 2.4488e+16 | 0.299565 | 0.120749 |
| 2023Q2 | 39 | 35 | 0 | 35 | 23 | 13 | 2.3008e+16 | 0.264392 | 0.121899 |
| 2023Q3 | 39 | 35 | 0 | 35 | 24 | 12 | 1.9799e+16 | 0.193377 | 0.098915 |
| 2023Q4 | 39 | 35 | 0 | 35 | 25 | 11 | 1.9922e+16 | 0.246867 | 0.153443 |
| 2024Q1 | 39 | 35 | 0 | 35 | 26 | 10 | 2.0024e+16 | 0.443615 | 0.202163 |
| 2024Q2 | 39 | 35 | 0 | 35 | 27 | 9 | 1.7818e+16 | 0.825646 | 0.192923 |
| 2024Q3 | 39 | 35 | 0 | 35 | 28 | 8 | 2.1641e+16 | 0.859742 | 0.179317 |
| 2024Q4 | 39 | 35 | 0 | 35 | 29 | 7 | 1.9864e+16 | 3.378952 | 0.166377 |
| 2025Q1 | 39 | 35 | 0 | 35 | 29 | 7 | 2.3330e+16 | 1.447270 | 0.187686 |
| 2025Q2 | 39 | 35 | 0 | 35 | 30 | 6 | 2.2830e+16 | 3.141213 | 0.169680 |
| 2025Q3 | 39 | 35 | 0 | 35 | 30 | 6 | 2.0274e+16 | 0.778214 | 0.173712 |
| 2025Q4 | 39 | 35 | 0 | 35 | 30 | 6 | 2.0805e+16 | 0.621541 | 0.175243 |
| 2026Q1 | 39 | 35 | 0 | 35 | 30 | 6 | 1.9160e+16 | 0.362390 | 0.165146 |
| 2026Q2 | 39 | 35 | 0 | 35 | 31 | 5 | 1.4918e+16 | 0.351371 | 0.169407 |

OLS coefficient L2 最大 3.378952，max absolute coefficient 最大 2.071611；Ridge 对应最大值 0.202163 与 0.115506。绝对 intercept 最大约 0.016038。Fit failures=0；coefficients、intercepts、transformed inputs 与 predictions 均有限。

[NVDA_stage15_numerical_stability.csv](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_numerical_stability.csv)；[NVDA_stage15_model_parameters.csv](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_model_parameters.csv)。没有依 condition number、rank 或系数幅度新增 fail threshold，未删除通过训练 coverage/variance 筛选的共线特征。保存的统计量、系数与 intercept 在全部 88 fits 上重建 OOS predictions，与正式值在严格 tolerance 内一致。

## 18. Prediction Stability / 极端预测

| 模型 | Max |prediction| | Folds with |p|>.20 | OOF count >.10 | OOF fraction >.10 | OOF count >.20 | OOF fraction >.20 |
| --- | --- | --- | --- | --- | --- | --- |
| zero_return | 0.000000 | 0 | 0 | 0.00% | 0 | 0.00% |
| historical_mean | 0.016038 | 0 | 0 | 0.00% | 0 | 0.00% |
| relative_ols | 0.103117 | 0 | 1 | 0.07% | 0 | 0.00% |
| relative_ridge_alpha1 | 0.092829 | 0 | 0 | 0.00% | 0 | 0.00% |
| relative_fundamentals_ols | 1.052852 | 7 | 335 | 24.51% | 163 | 11.92% |
| relative_fundamentals_ridge_alpha1 | 0.818442 | 5 | 268 | 19.60% | 108 | 7.90% |

Treatment OLS 最大预测幅度约 105.3%，Ridge 约 81.8%，而 Control 最大幅度约 10.3% 与 9.3%；Treatment 分别在 7 与 5 folds 出现 |prediction|>20%。这是新的 prediction pathology，不能沿用 Stage14“未出现新 catastrophic pathology”的判断。Ridge 在固定 alpha=1 下减轻部分幅度，但没有消除问题或恢复预测价值。这里为 descriptive 诊断，不是新的调参依据。

[NVDA_stage15_prediction_stability.csv](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_prediction_stability.csv) 保存各 fold/model 的 mean、std、min、max 与阈值 fractions。

## 19. Fundamental Extrapolation Diagnostics / 基本面外推

| Fold | Selected fundamentals | Max |z| | Raw OOR fraction | Mean feature fraction |z|>3 | Mean feature fraction |z|>5 |
| --- | --- | --- | --- | --- | --- |
| 2021Q1 | 20 | 5.914486 | 0.150820 | 0.113115 | 0.056557 |
| 2021Q2 | 23 | 3.374585 | 0.069717 | 0.130435 | 0.000000 |
| 2021Q3 | 23 | 2.961387 | 0.109375 | 0.000000 | 0.000000 |
| 2021Q4 | 23 | 1.957987 | 0.079353 | 0.000000 | 0.000000 |
| 2022Q1 | 21 | 9.735699 | 0.069124 | 0.034562 | 0.020737 |
| 2022Q2 | 24 | 7.522201 | 0.029570 | 0.166667 | 0.125000 |
| 2022Q3 | 24 | 10.570794 | 0.095703 | 0.082031 | 0.027344 |
| 2022Q4 | 26 | 5.527315 | 0.102564 | 0.213675 | 0.038462 |
| 2023Q1 | 26 | 4.703690 | 0.155087 | 0.084988 | 0.000000 |
| 2023Q2 | 26 | 3.888679 | 0.014268 | 0.129653 | 0.000000 |
| 2023Q3 | 26 | 23.374694 | 0.168498 | 0.080586 | 0.042125 |
| 2023Q4 | 26 | 9.837585 | 0.158730 | 0.147131 | 0.108669 |
| 2024Q1 | 26 | 11.819858 | 0.213115 | 0.334174 | 0.049180 |
| 2024Q2 | 26 | 5.982254 | 0.102564 | 0.333333 | 0.115385 |
| 2024Q3 | 26 | 3.353468 | 0.052885 | 0.140625 | 0.000000 |
| 2024Q4 | 26 | 2.681523 | 0.048678 | 0.000000 | 0.000000 |
| 2025Q1 | 26 | 2.765595 | 0.132692 | 0.000000 | 0.000000 |
| 2025Q2 | 26 | 2.485817 | 0.040943 | 0.000000 | 0.000000 |
| 2025Q3 | 26 | 2.191101 | 0.027644 | 0.000000 | 0.000000 |
| 2025Q4 | 26 | 2.203317 | 0.067308 | 0.000000 | 0.000000 |
| 2026Q1 | 26 | 2.379444 | 0.060530 | 0.000000 | 0.000000 |
| 2026Q2 | 26 | 2.268070 | 0.072398 | 0.000000 | 0.000000 |

基本面最大 validation |z|=23.374694。z 使用训练 median-imputed validation 及训练 mean/scale；raw OOR 仅比较 non-null validation 与训练 min/max，等于边界算 within。表中 fraction |z|>3/5 是 selected features 的等权平均，raw OOR 按 non-null feature exposure 加总；两者分母不同。每 fold/representation 只计一次 exposure，避免重复 OLS/Ridge。

[NVDA_stage15_extrapolation_audit.csv](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_extrapolation_audit.csv) 包含每个 selected feature 的 max/median |z|、阈值 fractions 和范围。没有基于诊断做 clipping、winsorization、删特征或 feature engineering。

## 20. Year-Level Diagnostics / 年度描述

| Year | Zero | Hist mean | C OLS | C Ridge | T OLS | T Ridge | OLS T−C | Ridge T−C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021 | 0.047952 | 0.047308 | 0.048975 | 0.048716 | 0.078027 | 0.071923 | 0.029052 | 0.023207 |
| 2022 | 0.067036 | 0.068860 | 0.073783 | 0.073297 | 0.126306 | 0.122286 | 0.052523 | 0.048989 |
| 2023 | 0.048392 | 0.046674 | 0.048737 | 0.048513 | 0.167865 | 0.155928 | 0.119128 | 0.107415 |
| 2024 | 0.056593 | 0.054396 | 0.054013 | 0.054056 | 0.235555 | 0.126575 | 0.181541 | 0.072519 |
| 2025 | 0.047484 | 0.047162 | 0.047672 | 0.047779 | 0.114268 | 0.059292 | 0.066595 | 0.011513 |
| 2026 | 0.040171 | 0.040179 | 0.040134 | 0.040129 | 0.047875 | 0.043441 | 0.007741 | 0.003313 |

[NVDA_stage15_yearly_diagnostics.csv](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_yearly_diagnostics.csv)。2026 只到 2026-06-12，其余均为相应 OOF calendar years；年度 pooled MAE 是 descriptive only，主分类仍按 equal-weight fold MAE，未生成 regime selection。5d labels 重叠、fundamental state 重复、expanding fits 相关，不做 IID t-test、naive p-value 或独立样本 CI。

## 21. Volume Exclusion / 字段范围

**Volume not used；Volume-dependent research not approved。** 所有 Control/fundamental/Treatment manifest、actual model matrices、scaler feature names 和 dependency graph 均程序化排除直接/间接 Volume dependency。Canonical source 中该字段保持原样，不被选作 predictors；没有 turnover、OBV、VWAP 或 liquidity proxy。

## 22. Final Test State / 锁定状态

```text
final_test_locked = true
final_test_predictions_generated = false
final_test_metrics_computed = false
final_training_pool_used = false
```

Frozen gap：2026-06-15–2026-07-14，20 sessions；Final Test：2026-07-15–2026-09-23，50 rows。没有 gap training/validation feature observation，没有 Final Test prediction/metric/final-model fit。末尾 5 个 development validation labels 按 Stage14 已有协议可在 gap 中 realized；独立 target source oracle 仅读取实现这些既有标签所需 terminal Open/Close，不把 gap rows 当特征，也不重算截断 label。

[NVDA_stage15_final_test_lock_audit.json](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_final_test_lock_audit.json) 所有 10 项程序化 boundary/label/pool checks 通过。

## 23. Tests / 真实测试结果

历史 679 项全部继续通过；新增 Stage15 60 项全部通过；完整 suite **739 passed，0 failed，0 skipped**。Targeted suite：60 passed，0 failed，0 skipped。

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_fundamental_ablation.py -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

覆盖 Control predictions/stats/membership、22 folds 与 1367 dates、purge、全 PIT/replay、future-filing mutations、validation missingness 与 future-fold isolation、training median/scaler、完整 OOF、no complete-case、Volume 排除、fixed OLS/Ridge、88 saved-model reconstruction、baseline 与 validation-Y isolation、Final Test、offline transport guard、fail-closed gates、immutable writes 和隔离目录全 pipeline byte reproduction。

原始日志：[targeted_tests.txt](../data/research/stage15_pit_fundamental_ablation/execution_logs/targeted_tests.txt)；[full_suite_tests.txt](../data/research/stage15_pit_fundamental_ablation/execution_logs/full_suite_tests.txt)。

实现过程中修正的工程问题均发生在新增 Stage15 wrapper/auditor：① Calendar year int32 与 CSV int64 的 dtype 差别被误当 identity 差别，改为严格 cell-value identity，prediction difference 始终为 0；② 初次 PIT next-session oracle 误用 2020 起始的研究 calendar，改用冻结 raw presample dates，初次误报 1020 cell/102 state checks，所有 canonical values 原本已精确相同；③ augmented matrix 的 Control 重建审计改为显式只选原 Control source schema。首次 preliminary test run 有 2 个工程 errors（未完成的参数文件及该 schema），修复后重新执行 targeted/full suites 全通过。无成熟历史 bug、协议改动或预测驱动的 specification 改动。Preliminary diagnostics 保留在 execution_logs。

## 24. Reproducibility / 连续正式运行

完整 pipeline 在 tests 通过后连续正式执行两次，31 个 authoritative CSV/JSON 全部 byte-for-byte identical，difference files=0；两次 output file set 也完全相同。Freeze writer 对已有正式文件只允许同字节重现。Protocol、initial state、registration、upstream manifest 也包含在 repeat set。

OLS/Ridge 不使用 randomness；BLAS threads=1；正式文件没有 current timestamp、temporary path 或随机 row order。执行后 documentary `execution_verification.json` 单独生成，沿用 Stage14 post-run verification pattern，不作为模型/协议 authority、不纳入两次 pipeline repeat set；其重复 serialization/write 也保持同字节。Runtime/test logs 不作为 formal deterministic artifacts。

完整 run1/run2 hashes、测试证据、最终验收：[NVDA_stage15_execution_verification.json](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_execution_verification.json)。

## 25. Upstream Immutability / 上游不变性

开始前保护 299 个 files：全部现有 data（包括 ignored sources）、tracked code/tests/docs/config 与 Stage1–14 artifacts。尤其包含 Stage13 canonical raw、Stage13.2 authority、Stage14 protocol/folds/OOF/report。完成后 mutation violations=0，changed files=[]；canonical 24 个 source/code hashes 也重新通过。

[NVDA_stage15_upstream_sha256.json](../data/research/stage15_pit_fundamental_ablation/NVDA_stage15_upstream_sha256.json)。所有新增写入仅为 Stage15 code/tests/report/output，未修改历史 statuses 或 artifacts。

## 26. Git Safety / 操作边界

未执行 reset、restore、clean、stash、rebase、commit、push 或任何 staging/index mutation。所有检查为本地 read-only Git commands；没有网络。修改只新增 Stage15 文件，保留用户可 review 的未提交结果。

## 27. Git Status / 结束状态

实际执行 `git status`、`git diff --check`、`git log -1 --oneline`；diff check exit 0，无输出。HEAD 保持 `9e8b244`，origin/main 未改变。

```text
On branch main
Your branch is up to date with 'origin/main'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	data/research/stage15_pit_fundamental_ablation/
	docs/stage15_execution_report.md
	src/nasdaq_research/fundamental_ablation.py
	tests/test_fundamental_ablation.py

nothing added to commit but untracked files present (use "git add" to track)

9e8b244 Complete stage 14 multi-year walk-forward validation
```

## 28. Final Integrated Conclusion / 综合回答

**A. Robust incremental OOS value：否。** 加入完整已登记 PIT block 后，OLS mean fold MAE 从 0.053479 上升至 0.136580，Ridge 从 0.053328 上升至 0.101771；paired mean/median delta 均为正，多数 folds 变差，正式分类为 `NO_ROBUST_INCREMENTAL_FUNDAMENTAL_VALUE`。

**B. Robust multi-year predictive signal：否。** Treatment 未稳定超过 Zero 与 Historical Mean，所有六项 robust signal gates 均失败；正式分类为 `NO_ROBUST_MULTI_YEAR_SIGNAL`。

**C. 高维 OLS 不稳定/新 prediction pathology：是。** 全部 Treatment designs 秩亏、condition numbers 很高；OLS 出现大幅 prediction extrapolation，Ridge(1) 也有异常幅度。系数和预测均有限、fit failure=0，所以这些属于被如实保留的研究/数值诊断，工程没有通过 post-hoc feature adjustment 去掩盖它们。

**D. PIT、leakage、reproducibility、immutability 与 Final-Test-lock：全部通过。** Control 逐值重现、OOF 完整、target purge 和 PIT violations 均为 0；training-only preprocessing、739 tests、31 formal artifacts 双运行同字节、299 upstream files 不变、Final Test locked 均满足，Stage15 engineering/protocol=PASS。

本阶段为单一 NVDA quant 主线的自然停止点。成熟到可供后续单独评估的组件包括 frozen-vintage authority/hash gate、PIT alignment、target provenance、walk-forward engine、training-only preprocessing、saved-model reconstruction 与 research audit/report contracts。本阶段没有构建 API、agent orchestration、UI、deployment 或 multi-ticker system，也没有设计或执行 Stage16。完成后停止，等待用户审核。

