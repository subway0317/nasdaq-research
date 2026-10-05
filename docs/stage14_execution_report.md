# Stage 14 — Relative Market 线性模型多年预注册 Walk-Forward 验证

正式双轴结论：**NO_ROBUST_MULTI_YEAR_SIGNAL**；**BROAD_REPRESENTATION_STABILITY_SUPPORT**。工程/protocol/leakage/reproducibility/Final-Test-lock：**PASS**。

Relative OLS 与 Ridge 在多年季度验证中改善 Original representation 的多数 folds 和 SMA extrapolation diagnostics；它们仍未满足同时优于 zero-return 与 historical-mean 的预注册 robust signal criteria。两个问题分别判断，不由 representation improvement 推导 predictive alpha。

## 1. Stage 14 Scope

本阶段仅做固定 Stage 10/11 specifications 的多年 walk-forward OOS validation、temporal stability 与 representation ablation validation。唯一 target 为 `forward_return_5d`；Market-only，不使用 fundamentals 或 Volume；没有 tuning、feature/model/target search、network、final-model fit 或 Final Test evaluation。

新增 [walk_forward.py](../src/nasdaq_research/walk_forward.py) 和 [test_walk_forward.py](../tests/test_walk_forward.py)。未修改成熟历史模块；直接复用 Stage 11 `representation`、`fit_arm` 和 Stage 10 `fit_linear`、`fit_preprocessor`、`predict_linear`、scalar preprocessing audit。新 outputs 独立位于 `data/research/stage14_multi_year_walk_forward/`。

先冻结 [NVDA_stage14_protocol.json](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_protocol.json)（SHA-256 `e6787c175a077a3e422d9533df00369e0450c35532b7c67276fbaeee21446908`），再冻结 upstream hashes、加载 canonical authority、构造 inventory/purge/features，并保存 [NVDA_stage14_fold_registration.json](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_fold_registration.json)；之后才执行 models。首个工程 smoke、正式运行、isolated tests 与 repeat 均未改变 protocol 或正式 predictions。

## 2. Initial Repository State

开始前真实执行 `pwd`、`git status`、`git log -5 --oneline`、`git diff --check`。cwd 为 `/home/zbw21/projects/nasdaq-research`；branch=`main`；HEAD 与 origin/main=`c3da396f929ec6be0e61e0842d6661b003851dbf`；working tree clean；diff check 通过。origin 为 `https://github.com/subway0317/nasdaq-research.git`。

```text
c3da396 Approve stage 13 canonical market data authority
04e2789 Complete legacy evidence recovery audit
d6c4b2c Record stage 13 history expansion and unresolved source reconciliation
bde8ed9 Complete stage 12 post-representation signal diagnostics
dbe6689 Complete stage 11 market representation ablation
```

初始真实状态见 [NVDA_stage14_initial_repository_state.json](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_initial_repository_state.json)。Stage 13=`INCOMPLETE_OR_BLOCKED`、Stage 13.1=`UNRESOLVED_BLOCKER`、Recovery=`LEGACY_EVIDENCE_NOT_RECOVERABLE`、Stage 13.2=`APPROVED_WITH_DOCUMENTED_NONMATERIAL_CROSS_VINTAGE_DIFFERENCE` 保持原字节。Legacy root cause 继续为 `UNRESOLVED_SOURCE_DISCREPANCY`，未重新调查。

## 3. Canonical Data Authority

使用 Stage 13.2 批准的 **`NVDA_YAHOO_STAGE13_2020_2026_V1`**，expanded_dataset_approved_for_downstream_research 和 stage14_modeling_eligible 均为 true。入口调用 `require_canonical_vintage`，核对 approved status、显式 ID、registered feature/target scope 和全部 **24 个 source/code hashes**，没有 cache-missing network fallback。

Canonical raw market **4304 rows**；研究矩阵 **1695 sessions，2020-01-02–2026-09-30**。主要 preserved source identity：

| 文件 | SHA-256 |
|---|---|
| `NVDA_stage13_research_matrix.csv` | `85d9297de51d4fa930af08d6b0aedea748c2915ee795c472c9a229d49b471ce6` |
| `NVDA_stage13_targets.csv` | `7c40adbde262d8f763c255a1b84b95ba4d3c096fda9d163a886e639979ff27ff` |
| `NVDA_acquisition.json` | `7e398ad570b0c8ebfdd6212763e1a60917b7f63db060c6782bb54c31fe9bdc03` |
| `NVDA_companyfacts.json` | `19ef503a5770f5660964b3c3aea6937579d9b359da344afe6a9adf59c63d26ff` |
| `NVDA_market.csv` | `019077bc0f1708ac4ac2ffb73613d28836b8b315070046e13c9ec59208b1fc14` |
| `NVDA_yahoo_chart_c8925631bb56ef3700f384bd325454dad7d03548493c7b6ddf4942f8f5074227.json` | `c8925631bb56ef3700f384bd325454dad7d03548493c7b6ddf4942f8f5074227` |
| `NVDA_yahoo_chart_d2306ab048d4171827c527cc46cc44aa6011145eb3902091ba451399b10aa6ae.json` | `d2306ab048d4171827c527cc46cc44aa6011145eb3902091ba451399b10aa6ae` |
| `NVDA_yahoo_source.csv` | `df45bf7400f08dc0e0e92c5cde50606a14004bfbcd9e3e9ccf3a0fcb9282c6a2` |

完整 authority/input evidence 见 [NVDA_stage14_canonical_input_audit.json](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_canonical_input_audit.json)。本阶段没有下载、刷新、覆盖、patch 或 merge cell。2026-09-30 Volume independent discrepancy 及五个 manifestations 原样保留；该日期也不在本阶段 development universe。

## 4. Development / Locked Boundaries

| 区间 | Frozen bounds | Observed sessions | 本阶段角色 |
|---|---|---:|---|
| Development | 2020-01-02–2026-06-12 | 1620 | 2020 initial training；2021 起 expanding train/quarterly validation |
| Pre-test gap | 2026-06-15–2026-07-14 | 20 | 不进入任何 training/validation feature observation |
| Final Test | 2026-07-15–2026-09-23 | 50 | 完全 LOCKED |
| Post-test tail | 2026-09-24–2026-09-30 | 5 | 不参与 training/validation/diagnostic decisions |

`read_selected` 在解析 numeric payload 之前以 lexical dates 过滤，只把 development rows 和 whitelisted columns 构成 modeling matrix；被排除 rows 的数值 X/Y 不返回给模型。没有读取 final-training-pool artifact。

**边界与 label realization 区分：**末尾五个 development feature dates 的既有 5d labels 于 2026-06-15–2026-06-22 realized。协议在模型运行前明确允许 retained canonical labels，且所有 exits 必须严格早于 Final Test；不因截断价格窗口重新定义 label 或删除 validation rows。仅 price-provenance oracle 读取这些标签必需的 terminal Open/Close；gap 仍为 20-session frozen gap，gap feature observations used=0。

## 5. Fold Design

**EXPANDING WINDOW，calendar quarters，22 folds**。首 fold 2021Q1、末 fold 2026Q2 partial；2020 不设置 OOS fold。Training 从 2020-01-02 累积，使用严格早于 validation first observed session 且 label 已 realized 的 observations。没有比较 rolling windows，没有改变边界。

完整 inventory：[NVDA_stage14_fold_inventory.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_fold_inventory.csv)。模型运行前保存真实 observed boundaries、raw/effective train ranges/counts、validation count 与 purge count；registration 绑定其 hashes。

| Fold | Observed validation | Raw train candidates | Effective train | Validation rows | 5d purged | Violations |
|---|---|---:|---:|---:|---:|---:|
| 2021Q1 | 2021-01-04–2021-03-31 | 253 | 248 | 61 | 5 | 0 |
| 2021Q2 | 2021-04-01–2021-06-30 | 314 | 309 | 63 | 5 | 0 |
| 2021Q3 | 2021-07-01–2021-09-30 | 377 | 372 | 64 | 5 | 0 |
| 2021Q4 | 2021-10-01–2021-12-31 | 441 | 436 | 64 | 5 | 0 |
| 2022Q1 | 2022-01-03–2022-03-31 | 505 | 500 | 62 | 5 | 0 |
| 2022Q2 | 2022-04-01–2022-06-30 | 567 | 562 | 62 | 5 | 0 |
| 2022Q3 | 2022-07-01–2022-09-30 | 629 | 624 | 64 | 5 | 0 |
| 2022Q4 | 2022-10-03–2022-12-30 | 693 | 688 | 63 | 5 | 0 |
| 2023Q1 | 2023-01-03–2023-03-31 | 756 | 751 | 62 | 5 | 0 |
| 2023Q2 | 2023-04-03–2023-06-30 | 818 | 813 | 62 | 5 | 0 |
| 2023Q3 | 2023-07-03–2023-09-29 | 880 | 875 | 63 | 5 | 0 |
| 2023Q4 | 2023-10-02–2023-12-29 | 943 | 938 | 63 | 5 | 0 |
| 2024Q1 | 2024-01-02–2024-03-28 | 1006 | 1001 | 61 | 5 | 0 |
| 2024Q2 | 2024-04-01–2024-06-28 | 1067 | 1062 | 63 | 5 | 0 |
| 2024Q3 | 2024-07-01–2024-09-30 | 1130 | 1125 | 64 | 5 | 0 |
| 2024Q4 | 2024-10-01–2024-12-31 | 1194 | 1189 | 64 | 5 | 0 |
| 2025Q1 | 2025-01-02–2025-03-31 | 1258 | 1253 | 60 | 5 | 0 |
| 2025Q2 | 2025-04-01–2025-06-30 | 1318 | 1313 | 62 | 5 | 0 |
| 2025Q3 | 2025-07-01–2025-09-30 | 1380 | 1375 | 64 | 5 | 0 |
| 2025Q4 | 2025-10-01–2025-12-31 | 1444 | 1439 | 64 | 5 | 0 |
| 2026Q1 | 2026-01-02–2026-03-31 | 1508 | 1503 | 61 | 5 | 0 |
| 2026Q2 | 2026-04-01–2026-06-12 | 1569 | 1564 | 51 | 5 | 0 |

Effective training 从 **248** 增长至 **1564**，每个后续 training set 包含前一个的全部 eligible sessions。Validation windows 无重叠；总 OOF **1367 sessions**。

## 6. Target Purge

Formal definition 仍为 `Close[t+5]/Open[t+1]-1`，observed session positions，target builder 未修改。只从冻结 targets artifact 取 existing Y；entry/exit/source prices 经 independent scalar source oracle 核对，**1620 checks，0 violations**。1d/20d 只核对 exit-date provenance：**3240 date checks，0 violations**，未加载其 numeric Y。

[NVDA_stage14_target_purge_audit.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_target_purge_audit.csv) 为每个 fold×raw-train-candidate 保存 feature date、entry date、target end、独立 observed-position oracle dates、validation start、eligible/reason。Training label 的 exit 必须 **严格早于 first observed validation session**，不能仅按 feature date 判定。

每 fold purge **5 rows**，共 **110 个 fold-boundary purged observations**；other exclusions=0，violations=0。普通 folds 使用 primary 5d realization；Final Test 既有 20-session pre-gap 维持不变。Validation labels 在季度边界的正常 overlap 不被事后删除，也不被当作独立样本。

## 7. Model Specifications

| Specification | Representation | 实现 |
|---|---|---|
| original_ols | Original Market | inherited LinearRegression(fit_intercept=True,n_jobs=1) |
| original_ridge_alpha1 | Original Market | inherited Ridge(alpha=1,fit_intercept=True,solver=svd) |
| relative_ols | Relative Market | Stage 11 name adapter + same OLS |
| relative_ridge_alpha1 | Relative Market | Stage 11 name adapter + same Ridge(1) |
| zero_return | 无 predictors | 0.0；不 fit |
| historical_mean | 无 predictors | mean(effective train primary y) |

共 **88 learned fits**，6 个 fixed specifications，未选 winner 或修改模型。Original 初始 whitelist 9：simple_return、log_return、intraday_return、daily_range、sma_5/20/60、rolling_volatility_20/60。

Relative 初始 whitelist 9，只替换 sma_5/20/60 为 close_to_sma_5/20/60，使用 Stage 11 正式 `Close/SMA_k-1` 实现；其它六项精确相同。[NVDA_stage14_feature_manifest.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_feature_manifest.csv) 18 rows，包含 paired source name、substitution、actual definition、dependency source/code hash。没有加入 raw Close 或任何 SEC/Volume-derived feature 到 predictors。

## 8. Preprocessing

每 fold/representation/model 继承 Stage 10：effective-training coverage ≥0.50 → training median fill → imputed-training exact zero-variance removal → training-only StandardScaler(ddof=0) → fit with intercept。Y 不 scaled。Validation 仅应用已 fitted statistics，不计算 validation/global medians 或独立 scaler。

[NVDA_stage14_preprocessing_audit.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_preprocessing_audit.csv) 保存 **792 fold×model×candidate rows**。真实数据中所有 candidate training coverage=1.0，九项均 retained；没有 global complete-case filtering，没有因 sparse fundamentals 删除任何 row。coverage/median/zero-variance/scaler 的 independent scalar audit、fit membership 和 scaler n_samples 均通过；全部违规计数为 0。

验证 validation X/Y mutation 不改变同一 fold fit/preprocessing state、future development mutations 不改变 first-fold predictions；edge fixtures 确认 exact 50% KEEP、低于 50% DROP、training median、population std、constant-column removal。正式代码仍只调用既有 fit semantics。

## 9. Fold-Level MAE

[NVDA_stage14_fold_metrics.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_fold_metrics.csv) 132 rows，保存 fold、model、primary target、train/validation count、MAE、prediction bias（prediction−actual）与 stability endpoints。下表 MAE 均为 return decimal units；没有 change primary metric。

| Fold | Zero | Historical mean | Original OLS | Original Ridge(1) | Relative OLS | Relative Ridge(1) |
|---|---:|---:|---:|---:|---:|---:|
| 2021Q1 | 0.04866447 | 0.04917968 | 0.04778218 | 0.04563513 | 0.04875095 | 0.04803763 |
| 2021Q2 | 0.04945000 | 0.04462364 | 0.06945596 | 0.06524843 | 0.05610510 | 0.05558451 |
| 2021Q3 | 0.03546635 | 0.03965852 | 0.03402276 | 0.03371549 | 0.03445827 | 0.03497526 |
| 2021Q4 | 0.05828274 | 0.05581560 | 0.05762729 | 0.05698013 | 0.05668740 | 0.05634083 |
| 2022Q1 | 0.06998100 | 0.07266217 | 0.10998459 | 0.09986912 | 0.08477453 | 0.08302544 |
| 2022Q2 | 0.07528601 | 0.08001472 | 0.08240002 | 0.08202180 | 0.09070870 | 0.09018087 |
| 2022Q3 | 0.05925743 | 0.06088391 | 0.06002641 | 0.05996310 | 0.06113040 | 0.06096286 |
| 2022Q4 | 0.06392091 | 0.06224209 | 0.05838461 | 0.05897845 | 0.05916130 | 0.05963621 |
| 2023Q1 | 0.06007515 | 0.05554230 | 0.06597954 | 0.06567749 | 0.06025998 | 0.06016409 |
| 2023Q2 | 0.04895043 | 0.04608862 | 0.05932917 | 0.05701144 | 0.04719738 | 0.04627843 |
| 2023Q3 | 0.04446387 | 0.04525296 | 0.04850483 | 0.04780883 | 0.04590424 | 0.04595451 |
| 2023Q4 | 0.04027408 | 0.03994330 | 0.04494206 | 0.04441395 | 0.04174615 | 0.04180470 |
| 2024Q1 | 0.05969742 | 0.05391729 | 0.06132222 | 0.06160248 | 0.05328872 | 0.05350806 |
| 2024Q2 | 0.05347791 | 0.04949221 | 0.04973571 | 0.05024234 | 0.04816950 | 0.04809830 |
| 2024Q3 | 0.07075080 | 0.07162254 | 0.07742538 | 0.07556524 | 0.07263058 | 0.07264389 |
| 2024Q4 | 0.04254239 | 0.04245245 | 0.04088907 | 0.04103920 | 0.04183976 | 0.04185520 |
| 2025Q1 | 0.07034903 | 0.07335287 | 0.07713279 | 0.07648727 | 0.07547794 | 0.07578650 |
| 2025Q2 | 0.05510243 | 0.04907664 | 0.05612868 | 0.05565056 | 0.04845314 | 0.04867862 |
| 2025Q3 | 0.02931156 | 0.02785156 | 0.02877132 | 0.02861838 | 0.02713234 | 0.02710959 |
| 2025Q4 | 0.03684066 | 0.04006299 | 0.03775399 | 0.03779348 | 0.04138881 | 0.04132036 |
| 2026Q1 | 0.03379491 | 0.03540376 | 0.03396180 | 0.03390590 | 0.03499841 | 0.03499205 |
| 2026Q2 | 0.04779825 | 0.04589036 | 0.04706290 | 0.04694079 | 0.04627766 | 0.04627299 |

## 10. Aggregate Predictive Results

Primary aggregation 是 **equal-weight mean fold MAE**，每 quarter 权重相同；最后 partial quarter 同样一个 fold。Pooled OOF MAE 与 median fold MAE 为预注册 secondary summaries。

| Model | Mean fold MAE（primary） | Pooled OOF MAE | Median fold MAE |
|---|---:|---:|---:|
| zero_return | 0.05244263 | 0.05240644 | 0.05146395 |
| historical_mean | 0.05186501 | 0.05184500 | 0.04912816 |
| original_ols | 0.05675560 | 0.05672544 | 0.05687799 |
| original_ridge_alpha1 | 0.05568950 | 0.05565768 | 0.05631534 |
| relative_ols | 0.05347915 | 0.05345291 | 0.04860205 |
| relative_ridge_alpha1 | 0.05332777 | 0.05330172 | 0.04838846 |

Formal [NVDA_stage14_model_summary.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_model_summary.csv) 包含 full precision values。展示的 rounding 不参与 classification。OOF table=[NVDA_stage14_oof_predictions.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_oof_predictions.csv)，**1367 dates × 6 models = 8202 prediction values**；missing sessions=0，duplicates=0，unexpected missing/nonfinite=0。

## 11. Baseline Comparisons

Delta=`MAE_model−MAE_baseline`，negative 为改善。Win 严格为 `<`；exact ties 不算 wins。[NVDA_stage14_baseline_comparison.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_baseline_comparison.csv) 同时保存所有 learned specs；这里展示两个 Relative models：

| Model | Baseline | Mean delta | Median delta | Wins/Losses/Ties | Fold win rate |
|---|---|---:|---:|---:|---:|
| relative_ols | zero_return | 0.00103652 | 0.00013565 | 10/12/0 | 45.4545% |
| relative_ols | historical_mean | 0.00161414 | 0.00051929 | 9/13/0 | 40.9091% |
| relative_ridge_alpha1 | zero_return | 0.00088514 | -0.00020107 | 11/11/0 | 50.0000% |
| relative_ridge_alpha1 | historical_mean | 0.00146276 | 0.00028622 | 9/13/0 | 40.9091% |

Relative OLS 对 zero/historical mean 胜率分别 **10/22**、**9/22**；Relative Ridge 分别 **11/22**、**9/22**。Ridge vs zero 的 50% 不满足严格 >50%。两个 Relative models 的 mean fold MAE 都高于两个 baselines。

## 12. Predictive Signal Classification

**`NO_ROBUST_MULTI_YEAR_SIGNAL`**。

每个 Relative model 的六项 gate：

| Pre-registered criterion | Relative OLS | Relative Ridge(1) |
|---|---|---|
| `mean_MAE_beats_historical_mean` | FAIL | FAIL |
| `mean_MAE_beats_zero_return` | FAIL | FAIL |
| `median_delta_vs_historical_mean_negative` | FAIL | FAIL |
| `median_delta_vs_zero_return_negative` | FAIL | PASS |
| `win_rate_vs_historical_mean_majority` | FAIL | FAIL |
| `win_rate_vs_zero_return_majority` | FAIL | FAIL |

至少一个 model 必须全部六项通过才为 ROBUST；当前两个都失败。冻结 precedence 按任务明确 quantitative conditions 执行：先 all-six ROBUST；若两个均不能同时满足两项 mean wins 与两项 majority wins，则 NO_ROBUST；其它才为 MIXED。任务中 MIXED 的示例不覆盖其明确 numeric NO predicate。这一 operational choice 在任何 Stage 14 结果之前已写入 protocol。

该结果是有效 negative validation result，工程执行通过；没有从单个好季度或某一年结果重新选择模型、变换、window 或 criteria。

## 13. Original vs Relative Representation

严格 paired comparisons 为 Relative OLS vs Original OLS，以及 Relative Ridge(1) vs Original Ridge(1)。Delta=`MAE_relative−MAE_original`。[NVDA_stage14_representation_comparison.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_representation_comparison.csv) 保存 44 个 fold pairs 与 2 个 aggregates。

| Family | Mean paired delta | Median paired delta | Relative wins / Original wins / ties | Relative win rate |
|---|---:|---:|---:|---:|
| ols | -0.00327646 | -0.00160260 | 14/8/0 | 63.6364% |
| ridge_alpha1 | -0.00236173 | -0.00110479 | 14/8/0 | 63.6364% |

两个 family 均 **14/22 folds（63.6364%）**改善。该 paired improvement 描述 representation 的影响；它没有使 Relative 同时打败两种 simple baselines，也不构成 alpha 声明。

## 14. Extrapolation Diagnostics

[NVDA_stage14_extrapolation_audit.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_extrapolation_audit.csv) 按 fold×representation×selected feature 保存 raw train/validation min/max、training mean/scale、validation non-null counts、out-of-range counts/fraction、max/median |z|、fraction |z|>3 与 >5。z 来自 **actual training median-imputed validation data 和 fitted training StandardScaler**；不是 validation 自己 center/scale。

Range fraction 分母为非空 raw validation values，严格低于/高于训练 min/max，等于边界算 within。同一 representation 的 OLS/Ridge preprocessing 精确相同，诊断只保存一次 exposure，避免 duplicate model-family weighting。All feature rows 共 **396**；SMA pairs 见 [NVDA_stage14_paired_extrapolation_summary.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_paired_extrapolation_summary.csv)。

| Fold | Original SMA max abs(z) | Relative SMA max abs(z) | Original outside fraction | Relative outside fraction | Relative max-z 更低 |
|---|---:|---:|---:|---:|---|
| 2021Q1 | 1.813166 | 3.177719 | 33.8798% | 0.0000% | 否 |
| 2021Q2 | 3.039092 | 2.238995 | 76.1905% | 0.0000% | 是 |
| 2021Q3 | 3.429968 | 2.479921 | 100.0000% | 0.0000% | 是 |
| 2021Q4 | 4.360829 | 3.758823 | 80.7292% | 4.6875% | 是 |
| 2022Q1 | 2.918315 | 3.448958 | 19.3548% | 1.0753% | 否 |
| 2022Q2 | 1.714038 | 3.583329 | 0.0000% | 3.2258% | 否 |
| 2022Q3 | 0.542763 | 2.820314 | 0.0000% | 0.0000% | 否 |
| 2022Q4 | 0.684507 | 2.509098 | 0.0000% | 0.0000% | 否 |
| 2023Q1 | 1.736535 | 2.466899 | 0.0000% | 0.0000% | 否 |
| 2023Q2 | 4.245272 | 4.728341 | 33.3333% | 6.4516% | 否 |
| 2023Q3 | 4.320502 | 1.608542 | 91.0053% | 0.0000% | 是 |
| 2023Q4 | 3.243036 | 1.819015 | 43.9153% | 0.0000% | 是 |
| 2024Q1 | 6.226673 | 2.903917 | 97.2678% | 0.5464% | 是 |
| 2024Q2 | 6.778978 | 2.895016 | 79.8942% | 0.0000% | 是 |
| 2024Q3 | 4.936005 | 2.969660 | 45.3125% | 0.0000% | 是 |
| 2024Q4 | 3.945720 | 1.618837 | 78.1250% | 0.0000% | 是 |
| 2025Q1 | 3.061377 | 4.409851 | 12.2222% | 0.0000% | 否 |
| 2025Q2 | 2.717114 | 3.630598 | 3.2258% | 0.0000% | 否 |
| 2025Q3 | 3.146430 | 1.269046 | 93.2292% | 0.0000% | 是 |
| 2025Q4 | 3.050707 | 1.840438 | 85.9375% | 0.0000% | 是 |
| 2026Q1 | 2.492520 | 1.855692 | 8.7432% | 0.0000% | 是 |
| 2026Q2 | 2.721983 | 1.749742 | 60.7843% | 0.0000% | 是 |

跨 folds：Original/Relative median SMA-group max |z| 为 **3.056042 / 2.664706**；equal-weight mean SMA outside fraction 为 **47.4159% / 0.7267%**；**13/22** folds 的 Relative SMA max-z 更低。

Original SMA 的最大 |z|=**6.778978**，Relative 最大=**4.728341**。Mean fold SMA fraction |z|>5 为 **2.5304% / 0.0000%**。

Relative 在九个 folds 的 SMA max-z 没有更低：2021Q1、2022Q1/Q2/Q3/Q4、2023Q1/Q2、2025Q1/Q2。没有宣称每 fold 的 representation 或所有特征都更稳定。详细每个 absolute SMA/relative SMA min/max、z 及 counts 保留在正式表中。

## 15. Prediction Stability

[NVDA_stage14_prediction_stability.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_prediction_stability.csv) 保存各 fold×model 的 prediction mean/std(ddof=1)/min/max/max absolute value 与两个固定 threshold fractions。Thresholds **|prediction|>0.10、>0.20** 仅 descriptive，未用于改模型或参数。

| Model | 全部 OOF 最大 abs(prediction) | 含 >0.10 predictions 的 folds | 含 >0.20 predictions 的 folds |
|---|---:|---:|---:|
| zero_return | 0.00000000 | 0 | 0 |
| historical_mean | 0.01603809 | 0 | 0 |
| original_ols | 0.14523975 | 2 | 0 |
| original_ridge_alpha1 | 0.12653434 | 1 | 0 |
| relative_ols | 0.10311695 | 1 | 0 |
| relative_ridge_alpha1 | 0.09282867 | 0 | 0 |

Original OLS >0.10 出现在 2022Q1 与 2023Q2；Original Ridge 在 2022Q1；Relative OLS 在 2022Q1；Relative Ridge 无 >0.10。2022Q1 的 Original OLS/Ridge max absolute predictions 分别约 **0.14524 / 0.12653**，Relative OLS 约 **0.10312**。2023Q2 Original OLS 的 prediction minimum 约 **−0.11527**。

所有六个 specifications 的所有 folds 都没有 **|prediction|>0.20**，没有重现任务所述 catastrophic large-negative prediction behavior。预注册 severe extrapolation count=0；Relative 比 Original 更严重的 severe-prediction fold count：OLS=0、Ridge=0。它们是 fixed-threshold observations，不是投资风险保证。

## 16. Representation Conclusion

**`BROAD_REPRESENTATION_STABILITY_SUPPORT`**。

预注册 gates 逐项通过：

| Criterion | Result |
|---|---|
| OLS paired fold MAE majority >50% | PASS，14/22 |
| Ridge paired fold MAE majority >50% | PASS，14/22 |
| Median fold SMA max-z lower | PASS |
| Mean fold SMA outside fraction lower | PASS |
| Majority folds lower SMA max-z | PASS，13/22 |
| Mean fold SMA fraction abs(z)>5 no higher | PASS |
| No new systematic severe pathology | PASS |

“总体机制方向”和“systematic”在 protocol 中于结果前定量化：SMA max-z median、outside fraction mean、strict majority improvement、>5 fraction；systematic severe extrapolation 指 strict majority folds 同时 Relative max-z>5 且高于 Original、Relative >5 fraction 更高；systematic severe prediction 指某一 family 的 strict majority folds 同时 Relative >0.20 fraction 更高、max absolute prediction 更高且>0.20。没有结果后增加阈值或逐个尝试替代规则。

BROAD 是对两个 family 和 formal extrapolation tables 的联合判断；没有使用 baseline performance 来替代 mechanism criteria，也没有将其改名为 predictive success。

## 17. Year-Level Diagnostics

2021–2026 pre-test descriptive aggregation，2026 仅覆盖 2026-01-02–2026-06-12。[NVDA_stage14_yearly_diagnostics.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_yearly_diagnostics.csv) 36 rows，包含 pooled yearly MAE、delta vs zero/historical_mean、row counts 与 descriptive_only=true。

| Year | Zero | Historical mean | Original OLS | Original Ridge(1) | Relative OLS | Relative Ridge(1) |
|---|---:|---:|---:|---:|---:|---:|
| 2021 | 0.04795169 | 0.04730791 | 0.05220651 | 0.05039252 | 0.04897521 | 0.04871567 |
| 2022 | 0.06703605 | 0.06885972 | 0.07748114 | 0.07502198 | 0.07378274 | 0.07329679 |
| 2023 | 0.04839231 | 0.04667392 | 0.05462518 | 0.05366699 | 0.04873732 | 0.04851306 |
| 2024 | 0.05659292 | 0.05439589 | 0.05732591 | 0.05708612 | 0.05401346 | 0.05405606 |
| 2025 | 0.04748414 | 0.04716182 | 0.04946226 | 0.04915972 | 0.04767250 | 0.04777913 |
| 2026 | 0.04017143 | 0.04017890 | 0.03992748 | 0.03984143 | 0.04013449 | 0.04012890 |

Relative 的 baseline comparisons 在年份之间有差异，2024 年改善、2022 年较弱；仅描述表中的 behavior。没有据此创建 regime filter、按年份选模型、删除坏年份或调整 window。没有以 yearly diagnostics 修改预注册双轴结论。

## 18. Coefficient / Model Parameter Audit

[NVDA_stage14_model_parameters.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_model_parameters.csv) 保存 **792 个 fold×learned-model×candidate rows**：selected feature/order、used/drop_reason、coverage/non-null count、training median、training imputed mean/std、actual scaler scale、coefficient、intercept、ridge alpha、fit target。

每个 estimator 与 preprocessing 的 fit dates 等于同一 effective-training membership，完整保存于 [NVDA_stage14_fit_membership.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_fit_membership.csv)；per-model-fold leakage audit 保存在 [NVDA_stage14_leakage_audit.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_leakage_audit.csv)。全部 **88** model-fold predictions 从 saved stats/coefficient/intercept 独立重建并通过 rtol=1e-10、atol=1e-12；CSV roundtrip 后代表性 2021Q1、2023Q4、2026Q2 再次重建通过。

OLS/Ridge 截距和正则语义继承旧代码，训练 y 未 standardized；coefficients 未用于 feature selection。本阶段仅保存参数，未额外设计 coefficient stability 或下一阶段研究。

## 19. Volume Exclusion

**Volume not used**。读取 modeling payload 的 gateway 从 frozen matrix 显式选择 date/ticker/open/close、原九项 features、三项已存 relative columns、primary label/provenance；Volume 和 fundamentals 不进入 modeling matrix。输入 schema 多出 Volume/turnover/SEC/secondary-Y 会 fail。

Graph gate 核对 Stage 13.2 每个 registered market/relative research object 的 explicit Volume nondependency evidence，missing proof 或 used_directly/used_indirectly=true 均 fail。Scaler/estimator 只能接收 Stage 11 精确九列 whitelist；formal parameter/scaler inputs 均不含 Volume、raw Close、fundamentals、targets 或 provenance。

Stage 13.2 的 Volume validity-guard limitation 仍保留于历史数据构建，但本阶段读冻结的已批准 artifacts，不重建或修改该 preprocessing。没有引入 Volume-dependent feature 或覆盖未来 revalidation requirement。

## 20. Final Test State

[NVDA_stage14_final_test_lock_audit.json](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_final_test_lock_audit.json) 全部 checks 通过：

```text
final_test_locked = true
final_test_predictions_generated = false
final_test_metrics_computed = false
final_training_pool_used = false
```

No Stage14 validation feature date ≥2026-06-15；training/validation observations 不进入 gap、Final Test 或 post-test tail；training target exits 早于对应 validation first session；全部 validation 5d exits 早于 2026-07-15。

Final Test numeric y 未载入 evaluation matrix，1d/20d numeric y 未载入；没有预测、MAE、correlation、distribution inspection、final model 或 full-pre-test fit。本阶段 5 个 validation labels 的 realization 在预注册允许的 pre-test gap 内；gap feature observations used=0，边界未缩短。

## 21. Tests

| Suite | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Historical Stage 1–13.2 + Recovery | 620 | 0 | 0 |
| New Stage 14 | 59 | 0 | 0 |
| Full suite | 679 | 0 | 0 |

Targeted 实际日志结果：`Ran 59 tests in 3.516s / OK`。Full 实际结果：`Ran 679 tests in 275.094s / OK`。未修改旧 tests、未 skip 或削弱 checks。

新增 tests 覆盖 quarter boundaries/22 folds、expanding set inclusion、5d observed-position purge、no gap/test/tail feature observations、training-only coverage/median/zero-variance/scaler、validation X/Y/future mutation isolation、baselines independence、88 fits 的 prediction reconstruction、formal CSV representative reconstruction、OOF uniqueness/coverage、exact 9-vs-9/three substitutions、Volume dependency/scaler exclusions、locked-y lexical rejection、tamper/frozen-file failures，以及两套 classification 的 strict thresholds/tie/majority semantics。

可复查 [targeted test log](../data/research/stage14_multi_year_walk_forward/execution_logs/targeted_tests.log) 与 [full-suite log](../data/research/stage14_multi_year_walk_forward/execution_logs/full_suite_tests.log)。仅 isolated regression/mutation tests 在 copies/fixtures 上生成 diagnostic outputs，未替换正式 OOF predictions。

```bash
PYTHONPATH=src GIT_OPTIONAL_LOCKS=0 .venv/bin/python -m unittest discover -s tests -p test_walk_forward.py -v
PYTHONPATH=src GIT_OPTIONAL_LOCKS=0 .venv/bin/python -m unittest discover -s tests -v
```

## 22. Reproducibility

最终 implementation 的首次正式运行后执行 targeted 和 full suites，再执行第二次正式 pipeline。**全部 24 个 formal authoritative CSV/JSON 实际 bytes 相同，mismatch_count=0**；hash maps 与 actual_bytes_compared=true 保存在 [NVDA_stage14_execution_verification.json](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_execution_verification.json)。

`freeze` 对 formal artifacts 为 idempotent：相同 bytes 允许复核，不同 bytes 直接 fail，不覆盖任何已冻结 predictions、folds 或 protocol。正式集合包含 16 个 CSV 与 8 个 JSON（protocol/upstream/initial registration、canonical input audit、fold registration、lock、summary、validation），没有 runtime current timestamp。

[NVDA_stage14_execution_verification.json](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_execution_verification.json) 属于预注册单独 post-run execution evidence；包含真实 test timings/log hashes，连同 execution logs 明确为非 authoritative、排除于 formal repeat 集合。protocol 在模型结果前已经固定此划分。Formal runtime versions：numpy `2.5.3`、pandas `3.0.6`、scikit-learn `1.9.1`；既有单线程 semantics 复用。

Formal artifacts：

- [NVDA_stage14_baseline_comparison.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_baseline_comparison.csv)
- [NVDA_stage14_canonical_input_audit.json](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_canonical_input_audit.json)
- [NVDA_stage14_extrapolation_audit.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_extrapolation_audit.csv)
- [NVDA_stage14_feature_manifest.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_feature_manifest.csv)
- [NVDA_stage14_final_test_lock_audit.json](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_final_test_lock_audit.json)
- [NVDA_stage14_fit_membership.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_fit_membership.csv)
- [NVDA_stage14_fold_inventory.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_fold_inventory.csv)
- [NVDA_stage14_fold_metrics.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_fold_metrics.csv)
- [NVDA_stage14_fold_registration.json](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_fold_registration.json)
- [NVDA_stage14_initial_repository_state.json](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_initial_repository_state.json)
- [NVDA_stage14_leakage_audit.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_leakage_audit.csv)
- [NVDA_stage14_model_parameters.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_model_parameters.csv)
- [NVDA_stage14_model_summary.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_model_summary.csv)
- [NVDA_stage14_oof_predictions.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_oof_predictions.csv)
- [NVDA_stage14_paired_extrapolation_summary.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_paired_extrapolation_summary.csv)
- [NVDA_stage14_prediction_stability.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_prediction_stability.csv)
- [NVDA_stage14_preprocessing_audit.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_preprocessing_audit.csv)
- [NVDA_stage14_protocol.json](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_protocol.json)
- [NVDA_stage14_representation_comparison.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_representation_comparison.csv)
- [NVDA_stage14_summary.json](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_summary.json)
- [NVDA_stage14_target_purge_audit.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_target_purge_audit.csv)
- [NVDA_stage14_upstream_sha256.json](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_upstream_sha256.json)
- [NVDA_stage14_validation.json](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_validation.json)
- [NVDA_stage14_yearly_diagnostics.csv](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_yearly_diagnostics.csv)

## 23. Upstream Immutability

预注册 [NVDA_stage14_upstream_sha256.json](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_upstream_sha256.json) 保护 **267 个 files**：所有 tracked source/tests/docs，加所有现存 data artifacts（包括 ignored legacy data）、Stage 13 candidate/cache、Stage 13.1、Recovery、Stage 13.2 authority/evidence/policy/reports。保护 manifest SHA-256=`6b9a4b082207d8ca86ebb2ea960a51e95b4d36df97fed805e8f30a87d84891bf`。

完成后全部重新计算：**mutation_violations=0；changed_files=[]**。另行复核 canonical manifest 绑定的 **24 source/code files**，canonical source mutation violations=0。没有更改 legacy、canonical Volume、targets/PIT/market definitions 或历史结论。

## 24. Git Safety

仅执行 initial/final 只读 status、log、ref 和 diff-check 检查，`GIT_OPTIONAL_LOCKS=0`。没有执行 add、commit、push、tag、reset、restore、clean、gc、prune、rebase、checkout 或 stash；没有 Git forensic search。所有工作仅新增 Stage 14 files，未修改旧 tracked files。

## 25. Git Status

完成后真实执行 `git status`、`git diff --check`、`git log -1 --oneline`。HEAD 与 origin/main 仍为开始时的 **c3da396**；branch main 与 origin 同步；diff check 通过。

```text
?? data/research/stage14_multi_year_walk_forward/
?? docs/stage14_execution_report.md
?? src/nasdaq_research/walk_forward.py
?? tests/test_walk_forward.py
```

```text
c3da396 Approve stage 13 canonical market data authority
```

完整 status、HEAD/origin、implementation/test/report hashes、logs、reproducibility maps、canonical-source check 与 no-partial-file check 见 [NVDA_stage14_execution_verification.json](../data/research/stage14_multi_year_walk_forward/NVDA_stage14_execution_verification.json)。没有 commit 或 push。

## 26. Final Integrated Conclusion

**A. Predictive signal：`NO_ROBUST_MULTI_YEAR_SIGNAL`。** Relative OLS 和 Ridge 都没有同时打败 zero-return/historical-mean 的 equal-weight mean fold MAE，也没有对两个 baselines 同时取得 strict majority fold wins；不满足 robust multi-year signal。该 negative result 合法，并已按原 criteria 保存。

**B. Representation：`BROAD_REPRESENTATION_STABILITY_SUPPORT`。** 两个 family 均在 14/22 folds 改善 Original MAE；全部冻结的 SMA mechanism gates 通过，且没有新 systematic extrapolation/prediction pathology。该结论支持当前 representation 的 temporal stability；不表示 predictive alpha。

**C. Catastrophic extrapolation：未触发新的 systematic pathology；未出现 |prediction|>0.20。** 原始 absolute SMA 仍有较高 range extrapolation；Relative 也存在个别 >0.10 prediction 和 max-z 更高 folds。这些 observed diagnostics 已逐 fold 保存，不作所有未来市场情景的保证。

**D. Engineering/protocol/leakage/reproducibility/Final-Test-lock：PASS。** Protocol/folds 先冻结、canonical authority 固定、purge/leakage/OOF checks 0 violations、历史+新增 679 tests 通过、24 formal artifacts bytes 一致、267 个 upstream files 0 mutations、Final Test 继续锁定。

研究限制保持明确：单一 NVDA、overlapping 5d labels、successive expanding fits 有依赖；季度或每日 OOF 不视为 IID。未做 t-tests、naive standard errors、IID CI、HAC/bootstrap 或 p-values。该研究是冻结 Stage 11/12 后的 retrospective multi-year walk-forward follow-up；各 fold fit 严格 chronological OOS，但不是从未被研究过的独立 confirmatory sample，更不等同于 untouched Final Test。Yahoo fixed split-adjusted coordinate、archived SEC source 和 Stage 13.2 metadata limitations 原样继承；不认证 historical as-traded vendor archive，也不作盈利/交易成本/经济 alpha 声明。

没有 tuning、网络下载、Volume/fundamental modeling、历史 root-cause 重新调查、Final Test 使用或下一阶段设计。完成后停止，等待用户审核。
