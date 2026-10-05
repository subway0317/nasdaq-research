Stage 11 — Pre-Registered Extrapolation-Stable Market Feature Representation Ablation

中文执行报告。结论：**PARTIALLY_SUPPORTED_OR_INCONCLUSIVE**。工程与协议执行完成，Final Test 保持 locked。以下数字来自正式 development CV artifacts，表格保留六位小数，CSV 保留完整精度。

1. Stage 11 scope

仅对 Market specification 的三项 SMA 进行预注册表征替换，运行 Original/Relative × OLS/Ridge 四个实验臂。没有进行其它公式搜索、All/fundamental 干预、alpha tuning、新模型、特征选择、PCA、clipping、替代 scaler/imputation、target/split 修改、final training、Final Test evaluation 或 backtest。Stage 10/10.1 development CV 已经看过，本实验属于 diagnostic-informed pre-registered follow-up，不能称为独立 confirmatory experiment。

2. 修改 / 新增文件

修改 `README.md`：新增 Stage 11 使用说明与实验约束。新增 `src/nasdaq_research/market_representation.py`：协议、局部表征、Stage 10 名称适配、CV、诊断、独立审计与 CLI。新增 `tests/test_market_representation.py`：41 项独立 oracle、adversarial、篡改检测与 pipeline 测试。新增 `docs/stage11_execution_report.md`：本报告。旧 Stage 1–10.1 源码未修改。新增以下 16 个正式 artifacts，全部位于 `data/research/modeling/stage11_market_representation/`：

| 新增文件 | 用途 |
| --- | --- |
| NVDA_stage11_protocol.json | 拟合前冻结的完整实验协议、唯一公式与机制结论规则 |
| NVDA_stage11_upstream_sha256.json | 拟合前冻结的 81 个 Stage 1–10.1 上游数据文件哈希 |
| NVDA_stage11_feature_manifest.csv | 候选/实际使用特征、coverage、过滤理由、训练 median/mean/std/scale |
| NVDA_stage11_oof_predictions.csv | 75 个日期 × 4 个实验臂，共 300 行 prediction/actual/error/absolute_error |
| NVDA_stage11_fold_metrics.csv | 12 个 fold/arm/model 的全部六项预测指标 |
| NVDA_stage11_feature_shift.csv | used features 的 raw training support、越界计数及 z 汇总 |
| NVDA_stage11_zscores.csv | 每个实际 used feature 的训练 scaler 与逐行 validation z |
| NVDA_stage11_prediction_summary.csv | 预测分布、绝对预测大小与固定 0.10/0.30 阈值 |
| NVDA_stage11_coefficients.csv | 标准化系数及截距 |
| NVDA_stage11_coefficient_summary.csv | 截距、系数 L2 和最大绝对系数 |
| NVDA_stage11_contributions.csv | 全部逐行 feature contribution 与 reconstruction 所需数值 |
| NVDA_stage11_ablation_summary.csv | 等权 mean-fold metrics、sample MAE std、pooled OOF metrics；不选模型 |
| NVDA_stage11_benchmark_metrics.csv | 定义不变且与 Stage 10 校验一致的两个 naive anchors |
| NVDA_stage11_stability_summary.csv | 按折及 pooled 的 all-used/SMA-related endpoint 汇总 |
| NVDA_stage11_summary.json | 机制判断与正式实验结果、Final Test 安全状态 |
| NVDA_stage11_validation.json | Control 复现、PIT/隔离/训练统计、round-trip/immutability 审计 |

3. Protocol

Control：`simple_return, log_return, intraday_return, daily_range, sma_5, sma_20, sma_60, rolling_volatility_20, rolling_volatility_60`。

Treatment：`simple_return, log_return, intraday_return, daily_range, close_to_sma_5, close_to_sma_20, close_to_sma_60, rolling_volatility_20, rolling_volatility_60`。

唯一公式为 `close_to_sma_k(t) = Close(t) / SMA_k(t) - 1`，k = 5/20/60。只使用同日已知 Close 与 trailing SMA；保留 warm-up NaN；Close 不进入 X。每臂恰好 9 个 candidates，Treatment 不追加原始 SMA。

Target：`forward_return_5d = Close[t+5] / Open[t+1] - 1`，feature 时点为 day t after close。Stage 9.1 manifest 是唯一 split authority，文件 SHA-256：`28568dd154031d4f0386dfeda826b2cf36a89c872bea7a9c0246266619c6dd4e`。

| Fold | Validation dates | Effective train | Validation |
| --- | --- | --- | --- |
| cv_1 | 2026-02-26 至 2026-04-01 | 96 | 25 |
| cv_2 | 2026-04-02 至 2026-05-07 | 121 | 25 |
| cv_3 | 2026-05-08 至 2026-06-12 | 146 | 25 |

OLS 使用原 Stage 10 LinearRegression；Ridge alpha=1.0、solver=svd，均带截距。每折/模型/臂独立执行 effective-training coverage >=0.50（恰好 50% 保留）、training median fill、training-only exact constant filter、training StandardScaler（ddof=0），Y 不缩放。名称适配仅用于复用未修改的 Stage 10 函数，处理的数据已经是相对值；随后恢复所有 Treatment feature/scaler labels，不修改全局白名单。

Primary predictive metric 为 MAE，summary 使用等权 mean fold MAE；secondary 为 RMSE、R²、Pearson、Spearman、Directional Accuracy。std 使用 ddof=1。Raw range 分母为非缺失 validation 值，等于训练 min/max 不算越界；z 使用该折 training scaler。

对用户给定的 qualitative success framework，本实现在首次真实 Stage 11 拟合前将“一致、实质性改善”固定为：cv_3 的三个 SMA-related/预测 endpoint 在两个模型均至少降低 25%，pooled endpoint 全部改善，并且 cv_1/cv_2 无新增极端不稳定。新增不稳定规则：Treatment 全 used-feature max |z| >5 且大于 Control，或 0.10/0.30 prediction 超阈值计数增加。混合证据报告 PARTIALLY_SUPPORTED_OR_INCONCLUSIVE，MAE 不参与分类。协议运行后未修改。

4. Control reproduction

Stage 10 OLS Market、Ridge Market 均成功复现：OOF 日期、ticker、fold、actual/prediction、fold metrics、全部候选特征使用/过滤行为、训练 coverage/median/mean/std/scale 及 standardized coefficients/intercepts 全部一致。浮点审计使用 rtol=0、atol=1e-12；将 Control 投影到 Stage 10 schema 后，两个模型的 prediction、metrics、feature usage、coefficients CSV 均逐字节一致。Reproduction violations = 0。

`cv_1` 两臂都因 training coverage 排除 60 日 SMA 维度和 60 日 volatility，实际 used=7；`cv_2/cv_3` used=9。候选数始终为 9，未依据预测表现删除特征。

5. Fold-level results

收益和误差均用原始 return units，Directional Accuracy 为 0–1。

| Fold | Arm | Model | MAE | RMSE | R² | Pearson | Spearman | DA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cv_1 | control | ols | 0.032433 | 0.042140 | -0.268730 | 0.339442 | 0.204615 | 0.520000 |
| cv_1 | control | ridge | 0.032344 | 0.041947 | -0.257145 | 0.332634 | 0.196923 | 0.440000 |
| cv_1 | treatment | ols | 0.033185 | 0.041736 | -0.244552 | 0.074067 | -0.063077 | 0.400000 |
| cv_1 | treatment | ridge | 0.033914 | 0.041829 | -0.250104 | 0.046403 | -0.140000 | 0.400000 |
| cv_2 | control | ols | 0.065026 | 0.087790 | -2.032023 | 0.224549 | -0.026923 | 0.600000 |
| cv_2 | control | ridge | 0.065839 | 0.088434 | -2.076641 | 0.227031 | -0.010000 | 0.600000 |
| cv_2 | treatment | ols | 0.075468 | 0.083388 | -1.735584 | 0.645906 | 0.661538 | 0.440000 |
| cv_2 | treatment | ridge | 0.077156 | 0.085021 | -1.843763 | 0.667313 | 0.677692 | 0.440000 |
| cv_3 | control | ols | 0.229550 | 0.239560 | -40.315115 | 0.098207 | 0.078462 | 0.600000 |
| cv_3 | control | ridge | 0.226452 | 0.236109 | -39.133532 | 0.115018 | 0.093846 | 0.600000 |
| cv_3 | treatment | ols | 0.050455 | 0.057536 | -1.383172 | 0.226112 | 0.070769 | 0.520000 |
| cv_3 | treatment | ridge | 0.049477 | 0.056077 | -1.263896 | 0.248031 | 0.143077 | 0.520000 |

| Arm | Model | Mean fold MAE | MAE std |
| --- | --- | --- | --- |
| control | ols | 0.109003 | 0.105661 |
| control | ridge | 0.108212 | 0.103760 |
| treatment | ols | 0.053036 | 0.021259 |
| treatment | ridge | 0.053516 | 0.021902 |

Naive anchors mean fold MAE：zero_return = 0.042271；historical_mean = 0.043501。Treatment 的平均 MAE 仍高于两个 anchors。cv_1/cv_2 MAE 均变差，平均改善主要来自 cv_3；不能据此宣布更好泛化或 alpha。

6. Stability diagnostics

Range/z 的预处理与模型无关；同臂 OLS/Ridge 数值相同，以下以 OLS 显示一次。

| Fold | Arm | All used 越界 fraction | All max |z| | All >3 | All >5 | SMA 越界 fraction | SMA >3 | SMA >5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cv_1 | control | 0.062857 | 2.691400 | 0 | 0 | 0.180000 | 0 | 0 |
| cv_1 | treatment | 0.022857 | 2.513659 | 0 | 0 | 0.040000 | 0 | 0 |
| cv_2 | control | 0.217778 | 5.217990 | 35 | 2 | 0.520000 | 27 | 2 |
| cv_2 | treatment | 0.133333 | 8.623852 | 33 | 9 | 0.266667 | 25 | 9 |
| cv_3 | control | 0.351111 | 19.250139 | 77 | 51 | 0.946667 | 71 | 51 |
| cv_3 | treatment | 0.053333 | 5.804598 | 18 | 1 | 0.053333 | 12 | 1 |
| pooled | control | 0.222400 | 19.250139 | 112 | 53 | 0.595000 | 98 | 53 |
| pooled | treatment | 0.073600 | 8.623852 | 51 | 10 | 0.130000 | 37 | 10 |

三项 SMA-related dimensions 的逐特征结果（cv_1 的 60 日维度未通过 training coverage，所以不伪造 z）：

| Fold | Arm | Feature | 越界 /25 | 越界 fraction | max |z| | >3 | >5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cv_1 | control | sma_5 | 5 | 0.200000 | 2.691400 | 0 | 0 |
| cv_1 | control | sma_20 | 4 | 0.160000 | 2.415990 | 0 | 0 |
| cv_1 | treatment | close_to_sma_5 | 1 | 0.040000 | 2.513659 | 0 | 0 |
| cv_1 | treatment | close_to_sma_20 | 1 | 0.040000 | 2.222905 | 0 | 0 |
| cv_2 | control | sma_5 | 7 | 0.280000 | 4.413351 | 8 | 0 |
| cv_2 | control | sma_20 | 16 | 0.640000 | 5.217990 | 6 | 1 |
| cv_2 | control | sma_60 | 16 | 0.640000 | 5.083045 | 13 | 1 |
| cv_2 | treatment | close_to_sma_5 | 0 | 0.000000 | 2.413010 | 0 | 0 |
| cv_2 | treatment | close_to_sma_20 | 3 | 0.120000 | 4.153856 | 8 | 0 |
| cv_2 | treatment | close_to_sma_60 | 17 | 0.680000 | 8.623852 | 17 | 9 |
| cv_3 | control | sma_5 | 21 | 0.840000 | 5.287219 | 21 | 4 |
| cv_3 | control | sma_20 | 25 | 1.000000 | 8.709776 | 25 | 23 |
| cv_3 | control | sma_60 | 25 | 1.000000 | 19.250139 | 25 | 24 |
| cv_3 | treatment | close_to_sma_5 | 0 | 0.000000 | 2.377428 | 0 | 0 |
| cv_3 | treatment | close_to_sma_20 | 1 | 0.040000 | 2.789840 | 0 | 0 |
| cv_3 | treatment | close_to_sma_60 | 3 | 0.120000 | 5.804598 | 12 | 1 |

Prediction distribution（std 为 ddof=1）：

| Fold | Arm | Model | Mean | Std | Min | Max | Mean |pred| | Median |pred| | Count >0.10 | Count >0.30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cv_1 | control | ols | 0.021737 | 0.023549 | -0.030285 | 0.067806 | 0.024160 | 0.020406 | 0 | 0 |
| cv_1 | control | ridge | 0.021325 | 0.023006 | -0.028292 | 0.067815 | 0.023775 | 0.020190 | 0 | 0 |
| cv_1 | treatment | ols | 0.012024 | 0.018258 | -0.015240 | 0.058976 | 0.016023 | 0.013359 | 0 | 0 |
| cv_1 | treatment | ridge | 0.011284 | 0.017826 | -0.019039 | 0.058504 | 0.015335 | 0.012275 | 0 | 0 |
| cv_2 | control | ols | -0.019197 | 0.048349 | -0.083047 | 0.065182 | 0.046989 | 0.046860 | 0 | 0 |
| cv_2 | control | ridge | -0.019973 | 0.048725 | -0.084394 | 0.066142 | 0.047296 | 0.049925 | 0 | 0 |
| cv_2 | treatment | ols | -0.029504 | 0.024721 | -0.065802 | 0.011141 | 0.031716 | 0.036121 | 0 | 0 |
| cv_2 | treatment | ridge | -0.031804 | 0.025643 | -0.073189 | 0.009350 | 0.033695 | 0.035305 | 0 | 0 |
| cv_3 | control | ols | -0.244740 | 0.062543 | -0.327749 | -0.107495 | 0.244740 | 0.260978 | 25 | 5 |
| cv_3 | control | ridge | -0.241642 | 0.061168 | -0.320563 | -0.107418 | 0.241642 | 0.257315 | 25 | 5 |
| cv_3 | treatment | ols | 0.017335 | 0.039800 | -0.039407 | 0.089525 | 0.031841 | 0.020605 | 0 | 0 |
| cv_3 | treatment | ridge | 0.016091 | 0.039410 | -0.043383 | 0.086655 | 0.031230 | 0.019740 | 0 | 0 |

各 fold 都有 25 行；count fraction = count/25。cv_1/cv_2 两臂两个模型均为 0/0；cv_3 Control 的 >0.10 为 25/25=1.00、>0.30 为 5/25=0.20，Treatment 均降为 0。CSV 单独保存全部 fraction。

cv_3 SMA-related 越界从 71/75（94.6667%）降至 4/75（5.3333%），max |z| 从 19.250139 降至 5.804598；SMA >3 从71降至12，>5从51降至1。系统性巨大负预测明显缓解。cv_2 的 close_to_sma_60 则更极端：越界 16→17、max |z| 5.083045→8.623852、>5 1→9。所有 used features 的 cv_2 max |z| 5.217990→8.623852。

7. Coefficient / contribution diagnostics

| Fold | Arm | Model | Intercept | L2 | Max |coef| |
| --- | --- | --- | --- | --- | --- |
| cv_1 | control | ols | -0.001128 | 0.141704 | 0.101352 |
| cv_1 | control | ridge | -0.001128 | 0.024602 | 0.023570 |
| cv_1 | treatment | ols | -0.001128 | 0.187537 | 0.136308 |
| cv_1 | treatment | ridge | -0.001128 | 0.025204 | 0.020329 |
| cv_2 | control | ols | -0.005046 | 0.239648 | 0.172728 |
| cv_2 | control | ridge | -0.005046 | 0.022465 | 0.019023 |
| cv_2 | treatment | ols | -0.005046 | 0.287878 | 0.205771 |
| cv_2 | treatment | ridge | -0.005046 | 0.020030 | 0.017402 |
| cv_3 | control | ols | 0.001917 | 0.194488 | 0.139698 |
| cv_3 | control | ridge | 0.001917 | 0.029519 | 0.017515 |
| cv_3 | treatment | ols | 0.001917 | 0.154197 | 0.109894 |
| cv_3 | treatment | ridge | 0.001917 | 0.022912 | 0.016357 |

全部 300 个 learned-model OOF rows 使用 intercept + sum(z × coefficient) 重构，prediction reconstruction violations = 0。逐项 contributions 也与独立 training statistics 重算值相符。2026-06-11 的 SMA60 contribution：OLS Control -0.322223 → Treatment -0.001225；Ridge Control -0.310317 → Treatment -0.001379。系数与贡献仅用于诊断，没有据此筛特征或调参。

8. Mechanism assessment

**PARTIALLY_SUPPORTED_OR_INCONCLUSIVE**。cv_3 的 6/6 预注册 endpoint 达到至少 25% 改善，pooled endpoint 全部改善；两个模型的 cv_3 mean |prediction| 分别从 0.244740→0.031841、0.241642→0.031230。但是 cv_2 的相对 60 日维度出现更极端 standardized extrapolation，触发预注册跨折一致性限制，故不能报告 SUPPORTED。该结果与 absolute SMA level representation 是 cv_3 不稳定的重要 contributor 的描述性判断相容，但该 intervention 未在全部 development folds 一致解决外推不稳定。

9. Leakage / adversarial tests

| 检查 | 结果 | Violations |
| --- | --- | --- |
| Final Test X mutation | PASS | 0 |
| Final Test Y/provenance mutation | PASS | 0 |
| Final Test metadata mutation | PASS | 0 |
| Pre-Test Gap mutation | PASS | 0 |
| Unlabeled Tail mutation | PASS | 0 |
| Secondary targets/provenance mutation | PASS | 0 |
| Training-only coverage | PASS | 0 |
| Training-only median imputation | PASS | 0 |
| Training-only constant/variance filtering | PASS | 0 |
| Training-only scaler | PASS | 0 |
| Same-fold validation X fit invariance | PASS | 0 |
| Same-fold validation Y fit invariance | PASS | 0 |
| Independent exact-formula oracle | PASS | 0 |
| Truncation invariance | PASS | 0 |
| Independent past-only SMA oracle | PASS | 0 |
| Future dependency | PASS | 0 |
| Feature whitelist | PASS | 0 |
| Exactly 9 candidate features | PASS | 0 |
| Forbidden rows used | PASS | 0 |

Final Test / Gap / Tail 的 X、Y、provenance、ticker/其它 metadata 在 in-memory copy 大幅变异后，全部 development tables、fit records、summary 和 mechanism/conclusion inputs 精确不变；未写回任何 upstream artifact。date 作为 immutable manifest join identity 保留。次要目标及相关 provenance 变异后结果也精确不变。同折 validation X/Y 变异不改变其训练参数；早期 validation 合法进入后续 expanding training。

10. Final Test state

```text
final_test_locked = true
final_test_predictions_generated = false
final_test_metrics_computed = false
final_training_pool_used = false
```

Final Test feature/target/metadata mutation dependency violations 均为 0。未计算或查看 Final Test performance / feature distribution。

11. Tests

Stage 11 tests: **41 passed, 0 failed**。独立历史 Stage 1–10.1 suite：**357 passed, 0 failed**。Full suite: **398 passed, 0 failed**。数量来自实际 unittest 执行，不包含 subTest 额外计数。未删除、弱化或修改旧 tests。

12. Reproducibility

正式 CLI 连续运行 2 次；run 1/run 2 **16/16 files byte-for-byte identical**：12 CSV + 4 JSON，包括 protocol、summary、validation、upstream manifest。未生成 PNG。没有 .tmp、half-written 或 partial output。当前环境：scikit-learn 1.9.1、pandas 3.0.6、numpy 2.5.3；BLAS threads=1、无随机性、无运行时间戳。

两次运行相同的 SHA-256：

| Artifact | Run 1 = Run 2 SHA-256 | Byte equality |
| --- | --- | --- |
| NVDA_stage11_ablation_summary.csv | acf34389977accb18a578276a13d9c7d6e226938483ef0a740b964c2f47b7f12 | true |
| NVDA_stage11_benchmark_metrics.csv | 7ba3696b115cad506fb18097d02bd7cf3f097aa8606bbd41eea53382f5943292 | true |
| NVDA_stage11_coefficient_summary.csv | 4e29ad78b3c9ad3b9f6c6906533a79d29d7831a066000cea78dcd36396e19b66 | true |
| NVDA_stage11_coefficients.csv | 51fdd3e33756b7047350f7d83158921d6370f987217ac016299bfbea18c36eef | true |
| NVDA_stage11_contributions.csv | 11b898aee71f59359b36506be85bf88eb27da85306ec8fdb60df9e93b9e40a60 | true |
| NVDA_stage11_feature_manifest.csv | baed71999c52b9b8b548608adba2fa912795bb5e7136e7fdeb155255685874e8 | true |
| NVDA_stage11_feature_shift.csv | d71e8fc13099c863a046a0201dae5a43fd7ca11bbe3ab6a6250d9fb57dabb822 | true |
| NVDA_stage11_fold_metrics.csv | c0a6324b3125d9d946e69c72ccee5b148967f6fde9f5a1ed397fa1271f2e48f3 | true |
| NVDA_stage11_oof_predictions.csv | 16bd4a2e7c2b72bba81afeb73d70b76518757098752a3ae5b4b461f5304217a1 | true |
| NVDA_stage11_prediction_summary.csv | 8bd119f7cc803732e42732053ac9e3809685388ba6e62d0b60cc6e99eed58075 | true |
| NVDA_stage11_protocol.json | 65abcffe46ded9948d06ccd6dddfb8ead67f1ce9c6c9c55f1c0fd8db23fb2c35 | true |
| NVDA_stage11_stability_summary.csv | f1807e4331100cd4734f6569ae69d864898fe174a1fe0acb5bfbec419c18204c | true |
| NVDA_stage11_summary.json | a1fe43ee7e08573851ab8982647cc98f302e4135de48d90d3a911bf46459798e | true |
| NVDA_stage11_upstream_sha256.json | b18f2d37e821d7eb018d165aeadcee16314940ca14dde14aba2fb643931bdf9c | true |
| NVDA_stage11_validation.json | cdac0441992758307187e807d691fea61a4be49f09570f862b8589ffb743ba00 | true |
| NVDA_stage11_zscores.csv | 03624def6cd92b95e171e672c4fc329a3a25311029783aa728d36a548c4eddb6 | true |

13. Upstream immutability

开始时保存全部现有 data 文件：**81 files**，覆盖 Stage 1–9.1、Stage 10 baseline 与 Stage 10.1 CSV/JSON/PNG。Stage 11 protocol 前保存的正式 manifest 与初始 manifest 完全一致。正式运行结束和全部测试后再次检查同一清单：**SHA-256 mutation violations = 0**。Validation 的 before/after map 相同，未修改 upstream outputs 或原 SMA 定义。

14. Git status

初始状态：main 与 origin/main 同步，working tree clean。最终 `git diff --check` 通过（exit code 0）。最终状态：

```text
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   README.md

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	data/research/modeling/stage11_market_representation/
	docs/
	src/nasdaq_research/market_representation.py
	tests/test_market_representation.py

no changes added to commit (use "git add" and/or "git commit -a")
```

没有执行 git add、commit、push 或 tag。

15. Final conclusion

预注册的 Close/SMA_k−1 表征显著降低了 Stage 10.1 所见的 cv_3 extrapolation / prediction pathology，并改善 pooled 稳定性；但是 cv_2 仍有更极端的 standardized extrapolation，且前两折 MAE 变差。因此正式结论为 PARTIALLY_SUPPORTED_OR_INCONCLUSIVE，不能声称全面解决 instability、证明因果机制、证明更好泛化或 alpha。仅基于 development CV；Final Test 未使用。Stage 11 到此结束，未进入 Stage 12。
