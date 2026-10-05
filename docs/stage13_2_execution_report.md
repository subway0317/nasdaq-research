# Stage 13.2 — 前瞻性行情权威版本冻结与下游研究批准

最终分类：**APPROVED_WITH_DOCUMENTED_NONMATERIAL_CROSS_VINTAGE_DIFFERENCE**。

Stage 13 frozen vintage 对当前已登记、无需 Volume 数值的研究用途具备足够的可审计性、内部一致性与可复现性，可作为前瞻性统一 authority。批准不代表历史 root cause 已解决，也不代表 vendor 数据是绝对市场真值。

## 1. Scope

本阶段执行 prospective data governance、source authority 和 fit-for-purpose approval。仅评估已经保存的 Stage 13 candidate；全过程离线，不创建新 acquisition。没有重新搜索 Git forensic evidence、恢复旧采集时间或推测 legacy discrepancy 的原因。没有对 Stage 13 candidate 训练或评价模型，没有设计 Stage 14 folds。

新增 [data_authority.py](../src/nasdaq_research/data_authority.py) 与 [test_data_authority.py](../tests/test_data_authority.py)。既有核心 pipeline、features、targets、PIT、历史报告均未修改。输出固定在 `data/research/historical_expansion/stage13_2_canonical_authority/`。

正式分析前先保存 [NVDA_stage13_2_protocol.json](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_protocol.json)、[NVDA_stage13_2_upstream_sha256.json](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_upstream_sha256.json) 与 [NVDA_stage13_2_initial_repository_state.json](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_initial_repository_state.json)。协议 SHA-256 为 `e15de0f4de24783ead998fa2ba14ea3648ccd36579a75f40db15333c2e00e9bc`；执行入口核对该 hash，未按观察结果更改规则。

## 2. Initial repository state

实际执行 `pwd`、`git status`、`git log -5 --oneline`、`git diff --check`。初始 cwd 为 `/home/zbw21/projects/nasdaq-research`，branch 为 `main`，HEAD 与 `origin/main` 均为 `04e2789f781f5a9a851d7be722c2dd23b6fc4680`，working tree clean，diff check 通过。origin 为 `https://github.com/subway0317/nasdaq-research.git`。

初始 log：

```text
04e2789 Complete legacy evidence recovery audit
d6c4b2c Record stage 13 history expansion and unresolved source reconciliation
bde8ed9 Complete stage 12 post-representation signal diagnostics
dbe6689 Complete stage 11 market representation ablation
4b232ac Complete stage 10.1 baseline model stability diagnostics
```

真实初始状态保存在 [NVDA_stage13_2_initial_repository_state.json](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_initial_repository_state.json)。记录中 `current_status_after_registration` 的新增目录来自完成预注册后的状态，预注册前工作区确为 clean。

## 3. Historical status preservation

| 阶段 | 永久保留的正式状态 |
|---|---|
| Stage 13 | `INCOMPLETE_OR_BLOCKED` |
| Stage 13.1 | `UNRESOLVED_BLOCKER` |
| Legacy Evidence Recovery | `LEGACY_EVIDENCE_NOT_RECOVERABLE` |
| Legacy root cause | `UNRESOLVED_SOURCE_DISCREPANCY` |

Stage 13 原 strict legacy-overlap rule 的 **5 unexpected overlap violations** 仍然存在于原 validation/report，未回写为通过。Stage 13.2 作出新的前瞻性治理决定；Stage 13.1 的两个 downstream flags 与 Recovery 的原拒绝状态均保持原字节。新的 true flags 仅属于 Stage 13.2。

## 4. Candidate vintage

唯一 candidate 与 canonical ID：`NVDA_YAHOO_STAGE13_2020_2026_V1`。研究请求范围为 **2020-01-01–2026-09-30**，实际研究 session 为 **2020-01-02–2026-09-30，1695 行**。完整 raw market 为 **2009-08-20–2026-09-30，4304 行**，包含 **2609 行** pre-sample history。

冻结查询边界为 start `2009-08-20`、end-exclusive `2026-10-01`；Yahoo chart 显示 NVDA、`1d`、USD、NMS、`America/New_York`。没有使用 bar timestamp 推断 retrieval timestamp。

| 角色 | 已保存文件 | SHA-256 |
|---|---|---|
| `acquisition_manifest` | `NVDA_acquisition.json` | `7e398ad570b0c8ebfdd6212763e1a60917b7f63db060c6782bb54c31fe9bdc03` |
| `SEC_frozen_input` | `NVDA_companyfacts.json` | `19ef503a5770f5660964b3c3aea6937579d9b359da344afe6a9adf59c63d26ff` |
| `normalized_market` | `NVDA_market.csv` | `019077bc0f1708ac4ac2ffb73613d28836b8b315070046e13c9ec59208b1fc14` |
| `original_vendor_chart` | `NVDA_yahoo_chart_c8925631bb56ef3700f384bd325454dad7d03548493c7b6ddf4942f8f5074227.json` | `c8925631bb56ef3700f384bd325454dad7d03548493c7b6ddf4942f8f5074227` |
| `original_vendor_chart` | `NVDA_yahoo_chart_d2306ab048d4171827c527cc46cc44aa6011145eb3902091ba451399b10aa6ae.json` | `d2306ab048d4171827c527cc46cc44aa6011145eb3902091ba451399b10aa6ae` |
| `original_yfinance_table` | `NVDA_yahoo_source.csv` | `df45bf7400f08dc0e0e92c5cde50606a14004bfbcd9e3e9ccf3a0fcb9282c6a2` |

完整 path/bytes/identity 见 [NVDA_stage13_2_candidate_source_inventory.csv](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_candidate_source_inventory.csv)。主 chart 4304 个 timestamp 与 normalized session dates 一致；原 chart、yfinance table、normalized Volume 全序列内部一致。原始 yfinance 表经现有 `standardize_history` 重放，OHLCV/date 逐列精确一致。

## 5. Governance audit

Axis A 共 **25 项关键检查通过，0 failure**；另有 1 项 `DOCUMENTED_LIMITATION`。各检查的 source、notes、violations 见 [NVDA_stage13_2_governance_audit.csv](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_governance_audit.csv)。

| Check | Status | Violations |
|---|---|---:|
| `raw_source_frozen` | PASS | 0 |
| `raw_and_normalized_hash_identity` | PASS | 0 |
| `provider_identity_and_query_bounds` | PASS | 0 |
| `original_table_normalization_exact_replay` | PASS | 0 |
| `raw_chart_source_identity_and_sessions` | PASS | 0 |
| `raw_chart_volume_internal_coherence` | PASS | 0 |
| `critical_OHLC_valid` | PASS | 0 |
| `Volume_legal_values` | PASS | 0 |
| `market_rows_ranges_unique_order` | PASS | 0 |
| `presample_maturity` | PASS | 0 |
| `calendar_audit` | PASS | 0 |
| `raw_calendar_audit` | PASS | 0 |
| `corporate_actions` | PASS | 0 |
| `PIT_activation_presample_future_filing_amendment` | PASS | 0 |
| `targets_definition_and_independent_oracle` | PASS | 0 |
| `Stage13_two_run_reproducibility_and_current_identity` | PASS | 0 |
| `research_exact_semantic_replay` | PASS | 0 |
| `snapshots_exact_semantic_replay` | PASS | 0 |
| `targets_exact_replay_outside_Final_Test` | PASS | 0 |
| `historical_acceptance_failures_preserved` | PASS | 0 |
| `legacy_nonidentifiability_and_historical_statuses` | PASS | 0 |
| `known_discrepancy_registry_complete` | PASS | 0 |
| `code_and_authoritative_manifest_dependency_evidence` | PASS | 0 |
| `upstream_immutability` | PASS | 0 |
| `Final_Test_locked_and_no_modeling` | PASS | 0 |
| `acquisition_timestamp_version_endpoint_limitation` | DOCUMENTED_LIMITATION | 0 |

日历检查分别为 raw **4304/4304**、research **1695/1695**，violations 均为 0。重复日期、非有限/非正 critical OHLC、OHLC bounds 均为 0。没有 drop、repair 或 re-adjustment。

现存 corporate-action evidence 确认 **2021-07-20 4:1** 与 **2024-06-10 10:1**，OHLC consistency、return continuity、SMA scalar oracle、cross-split target checks 均保持 0 violations。沿用 frozen vendor split-adjusted OHLC coordinate，不重新应用 dividends，不认证 as-traded absolute price levels。

PIT 保留严格 filing 后首个 observed trading session 激活、完整 pre-sample 初始化、独立 q/fy/bs whole-observation states。已冻结的 45 个 prefix dates、future SEC isolation、state/YoY oracle、amendment timing fixture 及 post-effective positive control 均通过；真实源内无 amendment 的既有限制保留。

Stage 13 run1/run2 原 **20 formal + 6 raw/cache** 的 26 个 current hashes 全部与原两次记录一致。本阶段按 Stage 13 原 default CSV parser boundaries 做只读 semantic replay：1695 行研究矩阵和 79 条 snapshots 精确一致；锁定区间外 targets 精确一致。未重新执行旧 artifact writers。

## 6. Research dependency manifest

[NVDA_stage13_2_research_dependency_manifest.csv](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_research_dependency_manifest.csv) 共 **595 行、173 个 research objects**，包含 research_object、dependency_type、source_field、used_directly、used_indirectly、description，以及实际 source/function/line/hash。[NVDA_stage13_2_dependency_code_evidence.json](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_dependency_code_evidence.json) 保留原函数源码、AST graph、source-owned fundamental maps 与 authoritative protocol identity。

真实实现文件为 `features.py`、`targets.py`、`alignment.py`、`research_features.py`、`data.py` 与 `historical_expansion.py`。仓库没有 prompt 示例中的 `integration.py` 或 `feature_matrix.py`。没有为配合示例而改变公式。

| Registered feature | 实际 AST expression | Raw market dependency |
|---|---|---|
| `close_to_sma_20` | `daily.close / daily[f'sma_{k}'] - 1` | close |
| `close_to_sma_5` | `daily.close / daily[f'sma_{k}'] - 1` | close |
| `close_to_sma_60` | `daily.close / daily[f'sma_{k}'] - 1` | close |
| `daily_range` | `(result['high'] - result['low']) / close` | close, high, low |
| `intraday_return` | `close / result['open'] - 1` | close, open |
| `log_return` | `np.log(close / close.shift(1))` | close |
| `rolling_volatility_20` | `result['simple_return'].rolling(window, min_periods=window, center=False).std(ddof=1)` | close |
| `rolling_volatility_60` | `result['simple_return'].rolling(window, min_periods=window, center=False).std(ddof=1)` | close |
| `simple_return` | `close.pct_change(fill_method=None)` | close |
| `sma_20` | `close.rolling(window, min_periods=window, center=False).mean()` | close |
| `sma_5` | `close.rolling(window, min_periods=window, center=False).mean()` | close |
| `sma_60` | `close.rolling(window, min_periods=window, center=False).mean()` | close |

SMA 为 trailing 5/20/60 observed closes，包含 t，固定 min_periods。Volatility 为 trailing 20/60 simple returns 的 sample std、ddof=1；其 Close 依赖通过 simple_return 传播。Relative SMA 的 Close 同时是直接分子依赖和通过 SMA 的间接依赖。

Targets 均沿用 `Close[t+h] / Open[t+1] - 1`，h=1/5/20；primary 为 `forward_return_5d`。entry/exit provenance 依赖 observed dates、Open 与 Close。

30 个 fundamental features 的字段来自正式 `RATIO_FIELDS`、`BALANCE_RATIOS`、`GROWTH_FIELDS`：q/fy 七种利润/费用/现金流占 revenue 比率；四种 balance ratios；q/fy 六种同口径 prior-FY growth。数值来自 SEC facts；whole-observation selection、reference timing、effective dates 和 ages 依赖 SEC provenance 与 observed session dates。全部 **155 个 daily fundamental/state/provenance columns** 与完整 snapshots 纳入 mutation 比较；排除的是携带的 raw market fields。

Stage 11 frozen protocol 的 control/treatment whitelists、三项 relative formulas、primary target 与 inherited Ridge alpha=1 均核对一致，只读取协议，不读取 predictive performance。本阶段没有 import 或执行 modeling module。

**Volume 有输入合法性依赖。** `standardize_history` 会 `to_numeric` 并按包含 Volume 的 required fields 执行 dropna；feature input validator 要求 OHLCV 有限且 Volume 非负。manifest 将两者单独列为 `preprocessing_validity_guard`，Volume 的 used_directly=true。所有当前数值 X/Y/PIT object 的 Volume used_directly/used_indirectly 均为 false。两个争议计数均是合法非负有限整数；不能把结论扩展到 missing/negative/nonfinite Volume。

AST tracer 对实际 Series assignment、alias、loop、f-string 建图；未知字段/符号/表达式 fail closed。广泛 DataFrame pass-through 与 integration helper semantics 由不可变代码 hashes、实际函数证据及全状态 mutation replay 共同约束。这是对当前固定实现的证据审计，不宣称对任意未来程序完成通用形式证明。

## 7. Known discrepancy registry

[NVDA_stage13_2_known_cross_vintage_discrepancies.csv](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_known_cross_vintage_discrepancies.csv) 从 Stage 13 原 unexpected overlap CSV 分组生成，完整保留：

| Ticker | Date | Field | Legacy | Candidate | Difference |
|---|---|---|---:|---:|---:|
| NVDA | 2026-09-30 | Volume | 121269300 | 121732200 | 462900 |

**Independent discrepancy count=1；downstream manifestations=5**，分别是 raw、market、PIT_stage5、research_stage61、labeled_stage8。原 5 violations 未删除或转换为 expected differences。

root_cause_status=`UNRESOLVED_SOURCE_DISCREPANCY`；legacy_evidence_recoverability=`LEGACY_EVIDENCE_NOT_RECOVERABLE`。第一因果来源仍是 preserved evidence 下 historically non-identifiable。

## 8. Materiality audit

Axis B 结果见 [NVDA_stage13_2_materiality_audit.csv](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_materiality_audit.csv)：

| 项目 | 结果 |
|---|---|
| direct feature dependency | false |
| indirect feature dependency | false |
| target dependency | false |
| PIT numerical/state dependency | false |
| preprocessing validity dependency | true，两个已知值都通过 |
| mutation/positive controls | 全部通过 |
| 分类 | `PROVENANCE_MATERIAL_MODEL_SEMANTIC_NONMATERIAL` |

当前 1 个 known discrepancy 属于该类别；`RESEARCH_SEMANTIC_MATERIAL` 与 `UNKNOWN_MATERIALITY` 均为 0。差异仍然 provenance-material。结论依据 code-derived dependency evidence、authoritative whitelists、实际 pipeline mutations 与 positive controls，未仅凭 feature 名单判断 Volume 无用。

## 9. Volume mutation adversarial tests

正式 evidence：[NVDA_stage13_2_volume_mutation_audit.csv](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_volume_mutation_audit.csv)。共 **11 个 isolated-copy cases**：

- 2026-09-30：121732200 → legacy 值 **121269300**。
- 2026-09-30：121732200 → **1217322000**，即乘 10。
- 2010-06-30、2020-06-30、2021-07-20、2022-06-30、2023-06-30、2024-06-10、2025-06-30、2026-06-30：各行 Volume 乘 10 再加 123。
- 全部 **4304 行** Volume 向量同时乘 10 再加 123。

每个 case 都运行现有 Stage 3 → Stage 5 → Stage 6.1 → inherited relative formula → target builders，核对完整 raw-history 下的 **9 market features、3 relative features、155 PIT columns、79 snapshots**、research **1d/5d/20d target 与 provenance**，以及 normalization/pipeline date-row eligibility。X/PIT 比较覆盖 4304 行；target 比较排除 locked 50 target rows。所有比较为 exact cells，paired missing sentinels 允许相等，不使用新 tolerance。

市场特征差异、relative 差异、PIT/snapshot 差异、锁定区间外 target 差异、row eligibility changes、总 violations 均为 **0**。原文件未修改；general historical 和全向量 mutation 排除了只靠最后一天自然 target tail 得出结论的可能。

## 10. Positive controls

[NVDA_stage13_2_positive_controls.csv](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_positive_controls.csv) 在 **2023-06-30** 分别对 Open、High、Low、Close 作合法的 isolated modifications，保证价格正且仍处于 OHLC bounds。结果：

| Mutated field | Market changed cells | Relative changed cells | Targets/provenance changed cells（锁外） | PIT changed cells | Violations |
|---|---:|---:|---:|---:|---:|
| open | 1 | 0 | 4 | 0 | 0 |
| high | 1 | 0 | 0 | 0 | 0 |
| low | 1 | 0 | 0 | 0 | 0 |
| close | 399 | 93 | 6 | 0 | 0 |

Open 改变 intraday_return、entry Open 与全部三种 return；High/Low 各改变 daily_range，targets 保持相同。Close 改变全部九种 market features、全部三种 relative features，以及三种 exit Close/return。所有 controls 的 PIT、snapshots、date eligibility 均相同。该证据确认测试实际进入了现有 dependency pipeline。

## 11. Canonical vintage manifest

[NVDA_stage13_2_canonical_vintage_manifest.json](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_canonical_vintage_manifest.json) 指定 canonical_vintage_id=`NVDA_YAHOO_STAGE13_2020_2026_V1`，approved=true，source_sha256 同时绑定 6 个 frozen raw/cache、既有 normalized fundamentals、关键 Stage 13 formal/provenance files 与现有 builder/helper code。

Authority scope 为 NVDA 的现有 research window、9 market features、3 relative features、30 fundamental features 与不变的 1d/5d/20d target semantics。未来目前登记的研究方向是 Relative Market OLS、inherited Ridge alpha=1、zero_return、historical_mean、primary 5d；本阶段未定义 fold 或训练流程。

manifest 明文保留 `legacy_cross_vintage_root_cause=historically_non_identifiable_from_preserved_evidence`，以及全部 historical statuses、metadata gaps、Volume limits 与 no-cell-patching/no-silent-refresh rules。

新增 future-facing `require_canonical_vintage` 要求 explicit vintage ID、已批准状态、scope 与每个 source hash；缺失、hash mismatch、path escape、未知 feature/target 均 fail loudly。它没有 network fallback。该新增入口需由未来正式 pipeline 显式调用，历史 downloader 未重构。

## 12. Future vintage policy

[NVDA_stage13_2_future_vintage_policy.json](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_future_vintage_policy.json) 已冻结以下规则：新 acquisition 必须新 vintage ID 和新 immutable namespace；使用过的旧 vintage、legacy 与 Stage 13 都不得覆盖；禁止 cell-level patch、手工 splice 或静默 refresh。

未来必须保存 provider、ticker/universe、可靠 UTC retrieval timestamp、query start/end、interval、endpoint/request semantics、library name/version、original vendor response/raw SHA-256、normalized output/normalized SHA-256、adjustment semantics、timezone/session semantics 与 corporate-action semantics。missing metadata 意味着 incomplete，Stage 13 的旧 metadata exception 不延续为新 acquisition 的许可。

新 vintage 采用前必须完成 internal audit、完整 cross-vintage diff、当前用途 materiality 审计，以及新的 explicit governance decision。当前 canonical 不是 latest forever；替换需经过新决定。

## 13. Limitations

- Legacy root cause 在当前 preserved evidence 下 historically non-identifiable，未被本阶段解决。
- Stage 13 的可靠 retrieval_timestamp_utc、library_version_at_acquisition、exact endpoint/request metadata 与 back_adjust_at_acquisition 未可靠保存，manifest 均保留 null；没有用当前环境版本、file mtime 或 bar timestamp 补造。
- 根据预注册 limitation rule，已保存 original payload、固定 hash、query boundaries、精确 normalization/semantic replay、26 个旧 run hashes 和通过的 critical internal audits 足以支持当前 prospective authority。它们无法回答缺失的历史采集元数据。
- 未证明 121732200 为绝对市场真值，也未证明 legacy 121269300 错误。价格/PIT 结论以既有 frozen Yahoo split-adjusted coordinate 与 archived SEC Company Facts vintage 为条件；不认证 as-traded price history 或完备 historical SEC disclosure-vintage archive。
- `volume_field_approved_for_current_modeling=false`；两个 future Volume revalidation flags 均为 true。Volume、turnover、volume change/z-score 或 liquidity proxy 等未来用途必须新增字段验证。
- 当前合法 Volume count 的非实质性不覆盖 missing/negative/nonfinite count 导致的输入失败或 row selection 变化。新增 feature、target、PIT semantics 或 source implementation 要重新治理。

## 14. Downstream approval

```text
expanded_dataset_approved_for_downstream_research = true
```

批准限于 manifest 中当前已有、无需 Volume 数值的用途。Stage 1–12 legacy snapshots 的未来角色为 `HISTORICAL_RESEARCH_RECORD_ONLY`，保留旧成果，不与 canonical cells 合并。

## 15. Stage 14 eligibility

```text
stage14_modeling_eligible = true
stage14_designed = false
stage14_executed = false
```

本结果打开 data governance gate。Stage 14 未设计，也未执行；不存在新增 folds、candidate training、predictions、MAE、CV 或 backtest artifacts。

## 16. Final Test

原 **2026-07-15–2026-09-23，50 rows** 继续锁定：

```text
final_test_locked = true
final_test_predictions_generated = false
final_test_metrics_computed = false
final_training_pool_used = false
predictive_performance_read = false
```

全部 target oracle/replay/mutation numeric comparisons 排除这 50 个 target rows。仅使用 source-date/calendar governance，不读取 Final Test predictive performance、target distribution 或训练池。既有 targets 文件只读，未生成 predictions。

## 17. Tests

| Suite | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Historical Stage 1–13.1 + Recovery | 558 | 0 | 0 |
| New Stage 13.2 | 62 | 0 | 0 |
| Full suite | 620 | 0 | 0 |

Targeted 实际结果：`Ran 62 tests in 36.676s / OK`。Full suite 实际结果：`Ran 620 tests in 276.503s / OK`。两份执行日志 SHA-256 保存在 [NVDA_stage13_2_execution_verification.json](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_execution_verification.json)。

可复查实际 [targeted test log](../data/research/historical_expansion/stage13_2_canonical_authority/execution_logs/targeted_tests.log) 与 [full-suite log](../data/research/historical_expansion/stage13_2_canonical_authority/execution_logs/full_suite_tests.log)。这些含执行耗时的日志属于非 authoritative execution evidence，不计入正式 deterministic CSV/JSON 集合。

覆盖真实 candidate 的四项 Volume X/Y/PIT tests、历史/全向量 mutations、OHLC positive controls、missing/negative Volume validity negative cases、AST direct/indirect/unknown dependencies、approval failure mapping、metadata policy、explicit-vintage/hash/no-refresh failures、scope exclusions 与 Final Test numeric exclusion。全套也包含用户要求的既有 regression tests；没有修改或弱化旧 tests，没有基于本阶段 candidate 进行模型研究。

执行命令：

```bash
PYTHONPATH=src GIT_OPTIONAL_LOCKS=0 .venv/bin/python -m unittest discover -s tests -p test_data_authority.py -v
PYTHONPATH=src GIT_OPTIONAL_LOCKS=0 .venv/bin/python -m unittest discover -s tests -v
```

## 18. Reproducibility

按最终实现连续运行两次 `PYTHONPATH=src .venv/bin/python -m nasdaq_research.data_authority` 的同一 `run_pipeline`；每次实际读取 frozen inputs、执行 builders、mutations、controls 并写新的 Stage 13.2 outputs。

**全部 15 个正式 CSV/JSON 实际 bytes 相同；mismatch_count=0**。比较包含 protocol、upstream/initial-state registration、policy、manifest、summary、validation，以及全部七份 CSV。没有 authoritative runtime current timestamps。两次完整 SHA-256 maps 见 [NVDA_stage13_2_execution_verification.json](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_execution_verification.json)；该 post-run verification 是单独的 execution evidence，不计入 formal repeat 集合。

正式集合：

- [NVDA_stage13_2_candidate_source_inventory.csv](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_candidate_source_inventory.csv)
- [NVDA_stage13_2_canonical_vintage_manifest.json](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_canonical_vintage_manifest.json)
- [NVDA_stage13_2_dependency_code_evidence.json](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_dependency_code_evidence.json)
- [NVDA_stage13_2_future_vintage_policy.json](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_future_vintage_policy.json)
- [NVDA_stage13_2_governance_audit.csv](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_governance_audit.csv)
- [NVDA_stage13_2_initial_repository_state.json](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_initial_repository_state.json)
- [NVDA_stage13_2_known_cross_vintage_discrepancies.csv](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_known_cross_vintage_discrepancies.csv)
- [NVDA_stage13_2_materiality_audit.csv](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_materiality_audit.csv)
- [NVDA_stage13_2_positive_controls.csv](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_positive_controls.csv)
- [NVDA_stage13_2_protocol.json](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_protocol.json)
- [NVDA_stage13_2_research_dependency_manifest.csv](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_research_dependency_manifest.csv)
- [NVDA_stage13_2_summary.json](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_summary.json)
- [NVDA_stage13_2_upstream_sha256.json](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_upstream_sha256.json)
- [NVDA_stage13_2_validation.json](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_validation.json)
- [NVDA_stage13_2_volume_mutation_audit.csv](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_volume_mutation_audit.csv)

## 19. Upstream immutability

开始前保护 **246 个 files**：tracked files 加所有现存 data artifacts，含 ignored legacy data、Stage 13 raw/cache/blocked report、Stage 13.1 reports/artifacts、Recovery reports/artifacts与既有源码/tests。

最终重新计算 SHA-256：**mutation_violations=0，changed_files=[]**。没有改 Stage 13 Volume、删除最后一天、改缓存、覆盖旧归档或回写历史结论。保护集 hash 为 `441a2544eef7c2d9f3ecb8bb7dfa435a9eb3af9021d049ab3a7be0fc44ba4966`。

## 20. Git safety

仅使用用户要求的只读 status/log/ref/diff-check 检查，并设置 `GIT_OPTIONAL_LOCKS=0`。未执行 `git add/commit/push/tag/reset/restore/clean/gc/prune/stash`，未再次进行 Git forensic search。无 commit 或 push；HEAD 与 origin/main 仍为原 checkpoint。

## 21. Git status

完成后实际执行 `git status`、`git diff --check`、`git log -1 --oneline`。结果：main 与 origin/main 同步；全部变化为新增 Stage 13.2 文件，没有 tracked modification；diff check 通过。

```text
?? data/research/historical_expansion/stage13_2_canonical_authority/
?? docs/stage13_2_execution_report.md
?? src/nasdaq_research/data_authority.py
?? tests/test_data_authority.py
```

```text
04e2789 Complete legacy evidence recovery audit
```

完整 status、HEAD、origin/main、source/test/report hashes、无 `.tmp` 残留的检查见 [NVDA_stage13_2_execution_verification.json](../data/research/historical_expansion/stage13_2_canonical_authority/NVDA_stage13_2_execution_verification.json)。本报告 hash 在执行证据生成时最终登记。

## 22. Completion classification

```text
APPROVED_WITH_DOCUMENTED_NONMATERIAL_CROSS_VINTAGE_DIFFERENCE
```

Axis A 通过；known registry 完整；唯一 independent discrepancy 的当前研究语义非实质性已由程序化证据支持；全部 mutation/positive controls、tests、formal repeats、upstream immutability 均通过。因为 unresolved cross-vintage discrepancy 仍存在，不使用 `APPROVED_CANONICAL_VINTAGE`。

## 23. Final conclusion

**是。** 在不声明 vendor snapshot 为绝对市场真值的前提下，Stage 13 frozen `NVDA_YAHOO_STAGE13_2020_2026_V1` 已具备足够的审计性、内部一致性、可复现性及当前登记用途适用性，可以成为未来 Stage 14+ 统一使用的 canonical NVDA market-data vintage；Volume-dependent research 仍须新字段验证和治理。

没有修改 Stage 13 / Stage 13.1 / Legacy Recovery 历史结论；没有继续追查 legacy root cause；没有重新下载行情；没有对 candidate 进行模型评价；没有打开 Final Test；没有设计或执行 Stage 14。至此停止。
