Stage 12 — Post-Representation Linear Signal and Calibration Diagnostics

中文执行报告。正式结论：**MIXED_OR_INCONCLUSIVE**。工程与协议检查通过。本阶段只诊断 Stage 11 固定预测，未拟合新模型，未生成任何校准后预测。表格保留六位小数，CSV 保存完整精度。

1. Scope

完成固定预测相对 naive baselines 的逐行优势、bias、dispersion、validation calibration a/b、三折 coefficient sign/cosine、feature-target association、target overlap/lag dependence，以及固定 |z|>5 context。Relative Market 是 primary specification，Original Market 仅为 historical reference。没有新 learned-model fit/prediction、调参、表征或特征干预、selection、校准修正、calibrated MAE/RMSE、fundamental intervention、clipping、替代 scaler/imputation、backtest 或 Final Test evaluation。

2. 修改 / 新增文件

没有修改任何历史文件。新增 `src/nasdaq_research/signal_diagnostics.py`：独立诊断 CLI/pipeline。新增 `tests/test_signal_diagnostics.py`：39 项 Stage 12 tests。新增 `docs/stage12_execution_report.md`：本报告。全部 17 个正式 artifacts 位于 `data/research/modeling/stage12_signal_diagnostics/`：

| 新增文件 | 用途 |
| --- | --- |
| NVDA_stage12_protocol.json | 正式诊断前固定的定义、约定、结论框架与限制 |
| NVDA_stage12_upstream_sha256.json | 138 个历史代码、测试、数据、protocol/report 等文件的起始哈希 |
| NVDA_stage12_row_baseline_comparison.csv | 300 行固定预测及 zero/historical-mean 逐行 AE、delta、比较结果 |
| NVDA_stage12_baseline_advantage_summary.csv | fold/pooled × arm/model 的逐行胜负、tie、平均/中位 delta |
| NVDA_stage12_prediction_calibration.csv | mean bias、median error/AE、terminal calibration a/b、Pearson/Spearman |
| NVDA_stage12_prediction_dispersion.csv | 样本 std、dispersion ratio、actual/prediction range 与固定阈值 |
| NVDA_stage12_coefficient_stability.csv | Relative 每特征/折/模型冻结系数、missing/drop、sign 与共同特征 summary |
| NVDA_stage12_coefficient_similarity.csv | 共同 used features 上的 cosine，含明确 feature names/count |
| NVDA_stage12_feature_target_associations.csv | 冻结训练 median/scaler 后的 train/validation Pearson/Spearman |
| NVDA_stage12_feature_target_stability.csv | 按 fold 及 feature-fold pooled 的符号匹配/翻转和相关性变化 |
| NVDA_stage12_target_windows.csv | 75 个 OOF date 的既有 entry/exit provenance 与相邻 overlap |
| NVDA_stage12_target_overlap.csv | 各 fold/pooled 的全 unordered pair 与 adjacent pair overlap |
| NVDA_stage12_target_dependence.csv | 各 fold lag 1–5 target Pearson dependence 与 pair denominator |
| NVDA_stage12_fold_diagnostics.csv | Relative 综合诊断及原 Stage 11 固定 MAE/RMSE/R²/DA |
| NVDA_stage12_extrapolation_context.csv | Stage 11 固定 row max \|z\|>5 / <=5 的原始 loss/bias 描述 |
| NVDA_stage12_summary.json | 综合诊断、预注册机制类别与安全状态 |
| NVDA_stage12_validation.json | authority、PIT、directional mutation、non-feedback、round-trip、immutability 审计 |

3. Protocol

Prediction authority：Stage 11 `NVDA_stage11_oof_predictions.csv`。Coefficient authority：Stage 11 `NVDA_stage11_coefficients.csv`。Preprocessing authority：Stage 11 `NVDA_stage11_feature_manifest.csv` 的实际 used/filter 状态、training median、mean、scale。Split authority：Stage 9.1 frozen manifest/protocol，其 manifest SHA-256 为 `28568dd154031d4f0386dfeda826b2cf36a89c872bea7a9c0246266619c6dd4e`。

Primary 为 Relative Market 的九个既定 candidates：`simple_return, log_return, intraday_return, daily_range, close_to_sma_5, close_to_sma_20, close_to_sma_60, rolling_volatility_20, rolling_volatility_60`。唯一相对公式仍为 `Close(t)/SMA_k(t)-1`；没有新公式。Original Market 是 reference，未重新分类 Stage 11 mechanism。Target 仍为 `forward_return_5d = Close[t+5]/Open[t+1]-1`，feature timestamp 为 t after close。

| Fold | Validation dates | Effective train | Validation |
| --- | --- | --- | --- |
| cv_1 | 2026-02-26 至 2026-04-01 | 96 | 25 |
| cv_2 | 2026-04-02 至 2026-05-07 | 121 | 25 |
| cv_3 | 2026-05-08 至 2026-06-12 | 146 | 25 |

Naive predictions 直接来自 Stage 10 正式 row-level OOF：zero_return=0；historical_mean=该折 effective training primary Y 的均值，并独立核对。没有用 validation Y 构造 historical mean。

Stage 12 error=prediction−actual，与 Stage 11 signed error 方向相反；AE 不变。delta_AE=AE_model−AE_baseline，负值为 learned model 胜。Tie 和 sign-zero tolerance 均固定为 1e-15。std/covariance/variance 使用 ddof=1。Calibration 仅输出 a/b：b=Cov(pred,actual)/Var(pred)，a=mean(actual)−b·mean(pred)，prediction constant 时 a/b 为 NaN。没有构造 a+b·prediction 向量。

结论框架在正式诊断前落盘，运行后未改：两模型均满足稳定 association direction、至少两折有稳健的多数 row advantage、明显 calibration mismatch，并有 coefficient/feature association 辅助支持，才给 SIGNAL_WITH_CALIBRATION_INSTABILITY；需多数 row failure、slope direction 不稳定及辅助 instability 的一致证据才给 WEAK_OR_TEMPORALLY_UNSTABLE_SIGNAL；其余给 MIXED_OR_INCONCLUSIVE。多数边界一行（1/25）以内优先 inconclusive，不能把阈值当科学证明。详细操作化阈值保存在 protocol。

4. Prediction / coefficient immutability

Stage 11 prediction consistency violations = **0**；Stage 11 coefficient consistency violations = **0**。全部 300 行 learned predictions/actual/date/fold/arm/model 投影到 authority schema 后精确一致且 projected CSV byte-equivalent。Relative 的 50 行 actually-used coefficients/intercepts 投影后也精确一致且 byte-equivalent；另有 4 个 dropped feature-fold/model 记录保持 NaN。new learned prediction rows = **0**；modified learned prediction rows = **0**。Stage 11 prediction、coefficient、preprocessing/input tables 和原始 artifacts 均未修改。

5. Row-level baseline advantage

Relative vs zero_return：

| Fold | Model | N | Mean delta | Median delta | Model wins | Win fraction | Ties | Tie fraction | Zero wins | Zero win fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cv_1 | ols | 25 | 0.001154 | 0.005893 | 9 | 0.360000 | 0 | 0.000000 | 16 | 0.640000 |
| cv_1 | ridge | 25 | 0.001884 | 0.006488 | 9 | 0.360000 | 0 | 0.000000 | 16 | 0.640000 |
| cv_2 | ols | 25 | 0.015838 | 0.013111 | 8 | 0.320000 | 0 | 0.000000 | 17 | 0.680000 |
| cv_2 | ridge | 25 | 0.017525 | 0.018312 | 8 | 0.320000 | 0 | 0.000000 | 17 | 0.680000 |
| cv_3 | ols | 25 | 0.015302 | 0.016699 | 9 | 0.360000 | 0 | 0.000000 | 16 | 0.640000 |
| cv_3 | ridge | 25 | 0.014324 | 0.017283 | 9 | 0.360000 | 0 | 0.000000 | 16 | 0.640000 |
| pooled | ols | 75 | 0.010764 | 0.008928 | 26 | 0.346667 | 0 | 0.000000 | 49 | 0.653333 |
| pooled | ridge | 75 | 0.011244 | 0.008306 | 26 | 0.346667 | 0 | 0.000000 | 49 | 0.653333 |

Relative vs historical_mean：

| Fold | Model | Mean delta | Median delta | Model wins | Win fraction | Ties | Tie fraction | Hist wins | Hist win fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cv_1 | ols | 0.001019 | 0.006981 | 8 | 0.320000 | 0 | 0.000000 | 17 | 0.680000 |
| cv_1 | ridge | 0.001748 | 0.006224 | 8 | 0.320000 | 0 | 0.000000 | 17 | 0.680000 |
| cv_2 | ols | 0.012666 | 0.013975 | 9 | 0.360000 | 0 | 0.000000 | 16 | 0.640000 |
| cv_2 | ridge | 0.014354 | 0.019115 | 9 | 0.360000 | 0 | 0.000000 | 16 | 0.640000 |
| cv_3 | ols | 0.014918 | 0.014783 | 10 | 0.400000 | 0 | 0.000000 | 15 | 0.600000 |
| cv_3 | ridge | 0.013940 | 0.015366 | 10 | 0.400000 | 0 | 0.000000 | 15 | 0.600000 |
| pooled | ols | 0.009534 | 0.007604 | 27 | 0.360000 | 0 | 0.000000 | 48 | 0.640000 |
| pooled | ridge | 0.010014 | 0.007615 | 27 | 0.360000 | 0 | 0.000000 | 48 | 0.640000 |

两个模型在全部三折均只胜过 zero-return 32%–36% 的 observations；pooled 为 26/75=34.6667%。相对 historical mean pooled 为 27/75=36%。所有 fold/pooled mean delta 都为正，两个基线的 MAE 优势具有逐行支持，不是只由单一极端 observation 导致。

6. Bias diagnostics

正 mean bias=系统性高估，负 mean bias=系统性低估。

| Fold | Model | Mean actual | Mean prediction | Mean bias | Median error | Median AE |
| --- | --- | --- | --- | --- | --- | --- |
| cv_1 | ols | 0.001006 | 0.012024 | 0.011018 | 0.000210 | 0.026499 |
| cv_1 | ridge | 0.001006 | 0.011284 | 0.010278 | 0.001178 | 0.030571 |
| cv_2 | ols | 0.043998 | -0.029504 | -0.073502 | -0.076495 | 0.076495 |
| cv_2 | ridge | 0.043998 | -0.031804 | -0.075802 | -0.079383 | 0.079383 |
| cv_3 | ols | -0.015190 | 0.017335 | 0.032525 | 0.054955 | 0.055185 |
| cv_3 | ridge | -0.015190 | 0.016091 | 0.031281 | 0.054376 | 0.054614 |

cv_2 actual mean 为 +0.043998，两个模型的 prediction mean 为 −0.029504/−0.031804，因此有明显 level underprediction。cv_1/cv_3 则 overprediction，bias 方向随时间变化。没有执行 bias correction。

7. Dispersion diagnostics

| Fold | Model | Std actual | Std prediction | Dispersion ratio | Pred min | Pred max | Mean \|pred\| | Median \|pred\| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cv_1 | ols | 0.038183 | 0.018258 | 0.478176 | -0.015240 | 0.058976 | 0.016023 | 0.013359 |
| cv_1 | ridge | 0.038183 | 0.017826 | 0.466856 | -0.019039 | 0.058504 | 0.015335 | 0.012275 |
| cv_2 | ols | 0.051457 | 0.024721 | 0.480419 | -0.065802 | 0.011141 | 0.031716 | 0.036121 |
| cv_2 | ridge | 0.051457 | 0.025643 | 0.498345 | -0.073189 | 0.009350 | 0.033695 | 0.035305 |
| cv_3 | ols | 0.038039 | 0.039800 | 1.046311 | -0.039407 | 0.089525 | 0.031841 | 0.020605 |
| cv_3 | ridge | 0.038039 | 0.039410 | 1.036053 | -0.043383 | 0.086655 | 0.031230 | 0.019740 |

std 统一 ddof=1。实际收益 min/max：cv_1 −0.056129/0.068126，cv_2 −0.076164/0.106605，cv_3 −0.081609/0.052700。Relative 所有 fold/model 的 |prediction|>0.10 和 >0.30 计数均为 0。前两折预测 dispersion 约为 actual 的一半，cv_3 dispersion ratio 接近 1；matching marginal dispersion 不能证明预测 calibration 或 association 良好。

8. Calibration regression

| Fold | Model | Intercept a | Slope b | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- |
| cv_1 | ols | -0.000856 | 0.154895 | 0.074067 | -0.063077 |
| cv_1 | ridge | -0.000115 | 0.099394 | 0.046403 | -0.140000 |
| cv_2 | ols | 0.083665 | 1.344465 | 0.645906 | 0.661538 |
| cv_2 | ridge | 0.086586 | 1.339058 | 0.667313 | 0.677692 |
| cv_3 | ols | -0.018936 | 0.216104 | 0.226112 | 0.070769 |
| cv_3 | ridge | -0.019042 | 0.239400 | 0.248031 | 0.143077 |

三个 folds 的 slope 全为正，但幅度明显变化：cv_1 约 0.10–0.15，cv_2 约 1.34，cv_3 约 0.22–0.24。cv_2 的 association 较强而 location calibration 明显异常；cv_1 association 很弱且 Spearman 为负，cv_3 为较弱正 association。这不支持把所有失败归结为纯 calibration。

**No calibrated prediction generated；no calibrated MAE/RMSE computed。** a/b 使用 validation Y，是 terminal diagnostics，没有反馈到 prediction/error/原 Stage 11 metrics。未做统计显著性声称。

9. Coefficient stability

三折共同实际 used 的 7 个 features 才进行 sign consistency：

| Model | Feature | Positive folds | Negative folds | Zero folds | All-fold consistent |
| --- | --- | --- | --- | --- | --- |
| ols | simple_return | 3.000000 | 0.000000 | 0.000000 | True |
| ols | log_return | 0.000000 | 3.000000 | 0.000000 | True |
| ols | intraday_return | 0.000000 | 3.000000 | 0.000000 | True |
| ols | daily_range | 0.000000 | 3.000000 | 0.000000 | True |
| ols | close_to_sma_5 | 0.000000 | 3.000000 | 0.000000 | True |
| ols | close_to_sma_20 | 0.000000 | 3.000000 | 0.000000 | True |
| ols | rolling_volatility_20 | 0.000000 | 3.000000 | 0.000000 | True |
| ridge | simple_return | 3.000000 | 0.000000 | 0.000000 | True |
| ridge | log_return | 3.000000 | 0.000000 | 0.000000 | True |
| ridge | intraday_return | 1.000000 | 2.000000 | 0.000000 | False |
| ridge | daily_range | 0.000000 | 3.000000 | 0.000000 | True |
| ridge | close_to_sma_5 | 0.000000 | 3.000000 | 0.000000 | True |
| ridge | close_to_sma_20 | 0.000000 | 3.000000 | 0.000000 | True |
| ridge | rolling_volatility_20 | 0.000000 | 3.000000 | 0.000000 | True |

| Model | Fold A | Fold B | Common features | Cosine |
| --- | --- | --- | --- | --- |
| ols | cv_1 | cv_2 | 7 | 0.997488 |
| ols | cv_1 | cv_3 | 7 | 0.997913 |
| ols | cv_2 | cv_3 | 9 | 0.994021 |
| ridge | cv_1 | cv_2 | 7 | 0.968450 |
| ridge | cv_1 | cv_3 | 7 | 0.892219 |
| ridge | cv_2 | cv_3 | 9 | 0.728618 |

cv_1/cv_2、cv_1/cv_3 的 intersection 为：simple_return、log_return、intraday_return、daily_range、close_to_sma_5、close_to_sma_20、rolling_volatility_20。cv_2/cv_3 另外共有 close_to_sma_60 和 rolling_volatility_60，共9个。CSV 单独保存 common_feature_names。OLS 7/7 共同 feature sign stable；Ridge 6/7，intraday_return 为1正/2负。未将 dropped coefficients 填0；zero vector norm 的 cosine 为 NaN。不同 folds 的 standardized coefficients 使用各自 scaler，方向稳定是描述性信息，不能单独证明稳定预测能力。

10. Feature-target association stability

只使用 Stage 11 冻结的 training-median/scaler transforms；两模型 preprocessing 一致，association 不重复计算。

| Fold | Used feature-fold pairs | Pearson matches | Pearson flips | Pearson match fraction | Spearman matches | Spearman flips | Spearman match fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cv_1 | 7 | 4 | 3 | 0.571429 | 4 | 3 | 0.571429 |
| cv_2 | 9 | 7 | 2 | 0.777778 | 5 | 4 | 0.555556 |
| cv_3 | 9 | 5 | 4 | 0.555556 | 4 | 5 | 0.444444 |
| pooled | 25 | 16 | 9 | 0.640000 | 13 | 12 | 0.520000 |

逐特征 train → validation 相关性：

| Fold | Feature | Train Pearson | Validation Pearson | Train Spearman | Validation Spearman |
| --- | --- | --- | --- | --- | --- |
| cv_1 | simple_return | -0.103753 | -0.022761 | -0.090295 | -0.020769 |
| cv_1 | log_return | -0.103427 | -0.026964 | -0.090295 | -0.020769 |
| cv_1 | intraday_return | -0.091817 | 0.136227 | -0.092566 | 0.150000 |
| cv_1 | daily_range | -0.052562 | 0.092068 | -0.091630 | 0.079231 |
| cv_1 | close_to_sma_5 | -0.287567 | -0.041674 | -0.264652 | -0.066923 |
| cv_1 | close_to_sma_20 | -0.411452 | -0.287114 | -0.440202 | -0.221538 |
| cv_1 | rolling_volatility_20 | -0.139994 | 0.191062 | -0.142349 | 0.186923 |
| cv_2 | simple_return | -0.102085 | -0.030328 | -0.079366 | 0.111538 |
| cv_2 | log_return | -0.102528 | -0.026916 | -0.079366 | 0.111538 |
| cv_2 | intraday_return | -0.063864 | -0.253660 | -0.054606 | -0.176154 |
| cv_2 | daily_range | -0.010102 | -0.233582 | -0.025281 | -0.039231 |
| cv_2 | close_to_sma_5 | -0.271487 | -0.254616 | -0.252860 | -0.163846 |
| cv_2 | close_to_sma_20 | -0.396118 | -0.703592 | -0.390166 | -0.708462 |
| cv_2 | close_to_sma_60 | -0.255491 | -0.603336 | -0.256942 | -0.527692 |
| cv_2 | rolling_volatility_20 | -0.052591 | 0.619885 | -0.044820 | 0.483077 |
| cv_2 | rolling_volatility_60 | 0.059177 | -0.433146 | 0.082370 | -0.405385 |
| cv_3 | simple_return | -0.073091 | 0.012575 | -0.045725 | 0.116923 |
| cv_3 | log_return | -0.073006 | 0.020713 | -0.045725 | 0.116923 |
| cv_3 | intraday_return | -0.051716 | 0.007610 | -0.034679 | 0.084615 |
| cv_3 | daily_range | -0.076829 | -0.355086 | -0.088093 | -0.422308 |
| cv_3 | close_to_sma_5 | -0.172204 | -0.155513 | -0.128448 | -0.183846 |
| cv_3 | close_to_sma_20 | -0.205495 | -0.301208 | -0.248075 | -0.196923 |
| cv_3 | close_to_sma_60 | -0.187632 | -0.371128 | -0.206435 | -0.271538 |
| cv_3 | rolling_volatility_20 | -0.074833 | 0.009977 | -0.109799 | 0.009231 |
| cv_3 | rolling_volatility_60 | 0.205372 | 0.003270 | 0.180573 | -0.075385 |

close_to_sma_5/20 的关联方向在三折均为负，close_to_sma_60 在实际 used 的两折也为负。rolling_volatility_20 则在三折均由 training 负关联转为 validation 正关联；cv_2 Pearson −0.052591→+0.619885 是明显变化。Pooled 指25个 feature-fold 诊断对，不是合并后的全样本相关性。Undefined correlation 保存 NaN 与原因，绝不填0。没有根据这些结果筛特征或重拟合。

11. Target overlap

Overlap 使用 existing primary target entry/exit provenance 的闭区间交集，接触同一 session 也算 overlap；没有假定5个calendar days。

| Fold | Windows | All-pair denominator | Overlapping pairs | Overlap fraction | Adjacent denominator | Adjacent overlaps |
| --- | --- | --- | --- | --- | --- | --- |
| cv_1 | 25 | 300 | 90 | 0.300000 | 24 | 24 |
| cv_2 | 25 | 300 | 90 | 0.300000 | 24 | 24 |
| cv_3 | 25 | 300 | 90 | 0.300000 | 24 | 24 |
| pooled | 75 | 2775 | 290 | 0.104505 | 74 | 74 |

Pooled denominator 75×74/2=2775，包含 cross-fold pairs；每个 OOF date 只出现一次。各 fold 相邻 overlap 为24/24，pooled 为74/74。Lag-1..5 target dependence 定义为对应截断序列的 Pearson，各 fold 分别计算：

| Lag observations | cv_1 | cv_2 | cv_3 |
| --- | --- | --- | --- |
| 1.000000 | 0.818552 | 0.812965 | 0.662478 |
| 2.000000 | 0.641396 | 0.516129 | 0.265771 |
| 3.000000 | 0.363368 | 0.106088 | -0.139121 |
| 4.000000 | 0.166998 | -0.290173 | -0.539112 |
| 5.000000 | -0.150919 | -0.457026 | -0.709530 |

Lag pair counts 为24/23/22/21/20；lag1–4 的对应窗口均 overlap，lag5不overlap。**75 OOF rows 不是75个独立样本**，也没有据此计算有效样本量或进行显著性检验。

12. Integrated mechanism assessment

**MIXED_OR_INCONCLUSIVE**。两个模型均有正 calibration slopes/Pearson，且系数方向总体稳定；cv_2 有较强正 association，同时存在严重 mean/location bias。然而没有一折在多数 observations 上优于 zero-return，cv_1 rank association 为负、cv_3 association 较弱，feature-target signs/magnitudes 也混合。因此既不满足 SIGNAL_WITH_CALIBRATION_INSTABILITY 的一致 incremental advantage，也不满足 WEAK_OR_TEMPORALLY_UNSTABLE_SIGNAL 所要求的一致方向不稳定证据。

cv_2 extreme z 的描述性 context（固定 Stage 11 row max |z| >5 / <=5）：

| Model | Frozen z stratum | N | Mean bias | Original MAE | Mean delta AE zero |
| --- | --- | --- | --- | --- | --- |
| ols | abs_z_gt_5 | 9 | -0.061305 | 0.066766 | 0.007951 |
| ols | abs_z_le_5 | 16 | -0.080363 | 0.080363 | 0.020274 |
| ridge | abs_z_gt_5 | 9 | -0.066307 | 0.070065 | 0.011251 |
| ridge | abs_z_le_5 | 16 | -0.081144 | 0.081144 | 0.021055 |

max |z| >5 的9行并未呈现更差平均 loss；<=5的16行低估更明显。这说明剩余失败不能仅凭 extreme z 数字归因，不能据此做因果结论或删除/截断这些行。Stage 12 未解决或修改 cv_2 z-score。

13. Leakage / adversarial tests

| 检查 | 状态 | Violations |
| --- | --- | --- |
| Final Test X | PASS | 0 |
| Final Test Y/provenance | PASS | 0 |
| Final Test metadata | PASS | 0 |
| Pre-Test Gap | PASS | 0 |
| Unlabeled Tail | PASS | 0 |
| Secondary target/provenance | PASS | 0 |
| Validation-Y directional dependency | PASS | 0 |
| Validation-X frozen prediction | PASS | 0 |
| Calibration non-feedback | PASS | 0 |
| Frozen preprocessing authority | PASS | 0 |
| Frozen validation z reconstruction | PASS | 0 |
| PIT exact formula | PASS | 0 |
| Truncation invariance | PASS | 0 |
| Past-only SMA oracle | PASS | 0 |
| Future dependency | PASS | 0 |

Final Test/Gap/Tail/secondary-target 的大幅 in-memory mutations 后，全部 development tables、summary、结论 inputs 与类别精确不变，未写回 upstream。date 是固定 manifest join identity。Validation-Y mutation 正确改变 error、calibration、validation associations，但不改变固定 predictions/coefficients、saved preprocessing 或 feature values。Validation-X mutation 可改变 associations，但固定 predictions/coefficients、原始 errors/calibration 不变。Calibration a/b 中间结果被大幅改写后，原 prediction、row errors、baseline advantage、原 MAE/RMSE 仍不变。Target overlap 有 touching/irregular-session 独立 oracle，lag dependence 和300行原始 AE 也有独立测试。

14. Final Test state

```text
final_test_locked = true
final_test_predictions_generated = false
final_test_metrics_computed = false
final_training_pool_used = false
calibrated_predictions_generated = false
calibrated_metrics_computed = false
new_learned_model_predictions_generated = false
```

Final Test X/Y/provenance/metadata mutation dependency violations 全部0；没有计算或查看任何 Final Test performance、calibration、dispersion 或 feature-target association。

15. Tests

Stage 12 tests: **39 passed, 0 failed**。独立历史 suite: **398 passed, 0 failed**。Full suite: **437 passed, 0 failed**。数量取自真实 unittest 运行，不把 subTest 计为独立测试。全部历史 tests 未修改或弱化。

16. Reproducibility

正式 Stage 12 CLI 连续运行2次：**17/17 artifacts byte-for-byte identical**，包括13 CSV与4 JSON。无PNG，无 .tmp/partial/half-written outputs。诊断无随机性，运行时间戳省略。软件版本：numpy 2.5.3，pandas 3.0.6。重复运行哈希证据：

| Artifact | Run1 = Run2 SHA-256 | Byte equality |
| --- | --- | --- |
| NVDA_stage12_baseline_advantage_summary.csv | 59c9a2234e845a9b17f053f473c3836d0cab78e0efd1fbdb2efe0f05354c4f7a | true |
| NVDA_stage12_coefficient_similarity.csv | 7cc3ec5cafc981a7faa7aca6aada47fc4e57b940660822cd99eff9408ba9d5ce | true |
| NVDA_stage12_coefficient_stability.csv | afa3037c8fa28065da88b83c96669e8c31fa4cf326dc38e30b803a34bdec5b3d | true |
| NVDA_stage12_extrapolation_context.csv | be1e1e45951d9854a19758d93f8c068e6260fda22d6198ec2ff4b84ca1cab80a | true |
| NVDA_stage12_feature_target_associations.csv | fa72a5bc6c79536b7bc6912c850abbe87ba6b71729fdeb0b258f9a9adab7c792 | true |
| NVDA_stage12_feature_target_stability.csv | 8ea4348d1ddfa93bd0b8d8fca49fbf6f8e88f2eeabec1f3b684659afc351c163 | true |
| NVDA_stage12_fold_diagnostics.csv | 8a48b154964a8432b897b1b72cf6646859e1a49bc997398ffaa8212d2510acdd | true |
| NVDA_stage12_prediction_calibration.csv | d4b9bf3a94edc5d5d6b028db15f2bd225bfba3c829c375365f71ab8033dae47b | true |
| NVDA_stage12_prediction_dispersion.csv | c1121b05e3c09a6e36ec582a58ca67884234c71e169054652c7d8ce05df7f9c0 | true |
| NVDA_stage12_protocol.json | bf8ec040bee2527fa83531b2a710e6a8ab1fc7f3643c17c61fe8ce2249aaa0df | true |
| NVDA_stage12_row_baseline_comparison.csv | e0129de0a1b7ad8344eecd193185a4a89c0943e0312ba973247b207d63e604d7 | true |
| NVDA_stage12_summary.json | c21bf1c4efc6510eda5e1487b5b6ed9b77be7617f72372fa32a637e5d8f9927e | true |
| NVDA_stage12_target_dependence.csv | 39346c1f875643770c23e10bf057dd1ab2ef809e8f7a311c2d8d43f4f351294d | true |
| NVDA_stage12_target_overlap.csv | 3bb267587b57fd2397f72cc86fc727fba51ae29e1d483055a0b206f6267fbb7f | true |
| NVDA_stage12_target_windows.csv | a41383d0f50b08bb9c82c11dfbaef19c10419ef31d542eac2c3f5db9b3ee5d43 | true |
| NVDA_stage12_upstream_sha256.json | 468d405d9aa9301a58b35d5d3af76b4f77ca4f509a6ef667f0ad64009cd15f6a | true |
| NVDA_stage12_validation.json | 2f64129fb2fa6fc6cc89ca9657561104ec23cd232bd80cfe6241c5e65bc3852c | true |

17. Upstream immutability

初始、正式 protocol 注册、两次运行及全部测试之后核对同一清单：**138 upstream files**，其中97个历史数据文件、41个历史代码/测试/报告/配置等文件。实际数量由文件清单得到，没有沿用上一阶段81的假设。Stage 1–11 的 SHA-256 mutation violations = **0**，Stage 11 正式数据、protocol、validation、report与源码均未修改。

18. Git status

初始 main 与 origin/main 同步、working tree clean；HEAD 为 `dbe6689 Complete stage 11 market representation ablation`，符合指定 checkpoint。最终 git diff --check 通过，exit code=0。没有修改 tracked 历史文件，没有执行 git add/commit/push/tag。最终实际 git status：

```text
On branch main
Your branch is up to date with 'origin/main'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	data/research/modeling/stage12_signal_diagnostics/
	docs/stage12_execution_report.md
	src/nasdaq_research/signal_diagnostics.py
	tests/test_signal_diagnostics.py

nothing added to commit but untracked files present (use "git add" to track)
```

19. Final conclusion

当前证据仍然混合。Relative Market 在 cv_2 存在较强正 association，并有明显 level/calibration 异常；但这种 association 强度未稳定延续，所有 folds 的多数 observations 都输给 naive baseline。系数方向稳定与部分 feature associations 提供一些支持，validation ranking、baseline advantage 与 bias/dispersion 又提供反向或不一致证据。故正式结论为 MIXED_OR_INCONCLUSIVE，不能宣称 stable predictive advantage、可交易 alpha 或已证明泛化。**Final Test 未使用。** Stage 12 到此结束，未进入 Stage 13。
