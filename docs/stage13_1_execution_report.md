# Stage 13.1 — Yahoo 历史数据快照差异溯源与权威快照策略冻结

本阶段已完成限定范围内的调查、证据冻结、策略映射、回归测试和离线复现。正式状态为 **UNRESOLVED_BLOCKER**：缺少 legacy 原始 Yahoo/yfinance 输入及可靠获取时间，不能可靠定位第一因果来源。因此 expanded dataset 未获得 downstream approval，Stage 14 modeling eligibility 为 false。工程审计通过不等于根因解决。

## 1. Scope

唯一 primary investigation object：**NVDA / 2026-09-30 / Volume**。已知 legacy normalized value 为 121269300，Stage 13 expanded value 为 121732200，差值为 +462900。

邻近 2026-09-29 和 2026-10-01 仅用于请求边界、响应和 session/timezone context。未重新比较全部 4304 条历史记录，未重新设计 Stage 13 的数据质量、PIT、corporate-action 或 target audits。未修改 features、targets、CV/splits，未开展新的 modeling、模型评价或 backtest，未进行 Final Test evaluation。

先创建并冻结 `NVDA_stage13_1_protocol.json`，再读取调查数据；先冻结 local evidence 和 processing trace，再进行一次 current Yahoo acquisition。正式复现仅使用冻结证据。协议的四类根因、证据优先级、分类与 policy mapping 未按结果改写。

新增独立模块 `src/nasdaq_research/source_reconciliation.py`、测试 `tests/test_source_reconciliation.py` 和本报告。新证据全部写入独立目录：

```text
data/research/historical_expansion/stage13_1_yahoo_reconciliation/
```

## 2. Initial repository state

开始时执行了 `pwd`、`git status`、`git log -5 --oneline` 和 `git diff --check`。

```text
cwd    = /home/zbw21/projects/nasdaq-research
branch = main
HEAD   = bde8ed98c34c7a3acb4b8911ae629cdbd99e363f
```

HEAD 为 Stage 12 checkpoint，**Stage 13 尚未 commit**。初始 working tree：

```text
 M src/nasdaq_research/data.py
?? data/research/historical_expansion/
?? docs/stage13_execution_report.md
?? src/nasdaq_research/historical_expansion.py
?? tests/test_historical_expansion.py
```

其中初始 historical_expansion 内容为 Stage 13。上述修改及未跟踪文件均为本阶段开始前已经存在的研究输入；本阶段按当时实际字节冻结保护，包括已修改的 `data.py`。初始 `git diff --check` 通过。初始状态保存在 `NVDA_stage13_1_initial_repository_state.json`。

## 3. Source inventory

正式 inventory 共 27 行，逐项记录存在状态、SHA-256、已知获取时间、实际观察日期范围、已记录请求和 authority role。

| Source side | AVAILABLE | PARTIAL | NOT_AVAILABLE | 主要解释 |
|---|---:|---:|---:|---|
| legacy | 5 | 1 | 3 | 五份 normalized/downstream CSV 可用；旧 downloader 代码只能提供配置默认值；原始响应、原始 yfinance table、获取 manifest 缺失 |
| stage13 | 8 | 3 | 1 | 原始 chart、原始 yfinance、normalized/downstream 和冻结审计可用；获取元数据不完整，Stage 8 只有间接 audit evidence |
| current_external | 4 | 1 | 0 | primary chart、yfinance、normalized、acquisition manifest 可用；辅助时区请求不含 primary date |
| third_party | 0 | 0 | 1 | 未查询；无法弥补 legacy 原始输入与 timing 缺口 |

检查了现有 data 文件的 88 个 CSV headers 和 28 个 market JSON signatures，用于识别原始源，不进行全历史 observation 审计。发现的原始 market source candidates 仅属于 Stage 13；未发现 legacy 原始 Yahoo HTTP response 或原始 yfinance representation。搜索过程另存 `evidence/NVDA_source_discovery_audit.json`。

**`data/raw/NVDA.csv` 已经过标准化，是 Level 3 normalized representation。目录名 raw 不能使它成为 Level 1 原始 vendor evidence。** 未从该文件反推或伪造 legacy raw source。

Stage 13 六个 raw/cache 文件均保留：SEC copy、normalized market、yfinance original table、acquisition manifest、两个 Yahoo chart responses。SEC copy 不参与本次 Volume 因果判断；它作为 Stage 13 原始证据接受 hash 保护。

证据优先级依次为：Level 1 冻结原始输入、Level 2 同期获取元数据、Level 3 normalized/downstream、Level 4 current Yahoo、Level 5 第三方 corroboration。

## 4. Acquisition metadata

| 字段 | legacy | Stage 13 expanded | 本次 current Yahoo |
|---|---|---|---|
| retrieval/request timestamp | unknown / unavailable | unknown / unavailable | 有明确 UTC 开始、完成和 response-received 时间 |
| query start | unknown / unavailable | 2009-08-20 | 2026-09-29 |
| query end | unknown / unavailable | 2026-10-01，manifest 标记 exclusive | 2026-10-02 exclusive |
| interval | 旧代码默认 1d；实际参数 unknown | 原始响应确认 1d | 1d |
| exact endpoint | unknown / unavailable | unknown / unavailable | `https://query2.finance.yahoo.com/v8/finance/chart/NVDA` |
| acquisition-time yfinance version | unknown / unavailable | unknown / unavailable | 1.7.0 |
| auto_adjust | 旧代码配置 False；实际请求 unknown | manifest 记录 False | False |
| back_adjust | unknown / unavailable | unknown / unavailable | False，当前安装版本 signature default |
| repair | unknown / unavailable | manifest 记录 False | False |
| 原始 timezone/session 证据 | unknown / unavailable | America/New_York，EDT，gmtoffset=-14400 | 原始 chart 与请求参数可核对 |

旧 HEAD downloader 配置 `period=1y`、`interval=1d`、`auto_adjust=False`，但没有保存实际调用记录。legacy normalized 文件观察范围为 2025-10-01 至 2026-09-30，不能当作查询边界或 retrieval timestamp。

Stage 13 chart 的 `regularMarketTime` 为 `2026-10-05T14:06:30+00:00`，属于报价时间；primary bar epoch 为 1790775000，对应 `2026-09-30T13:30:00+00:00` / `2026-09-30T09:30:00-04:00`，属于日线 session-open timestamp。二者均不是 HTTP 获取时间，也不能证明 legacy session completeness。Stage 13 响应中的 primary session 已位于报价时间之前，但这不能恢复 legacy timing。

可用 expanded/current chart 的 5 条邻近日期 context 中，UTC 与 New York session date 一致，观察到的 timezone/date identity issues 为 0。legacy 原始 session identity 因缺证据不能认证。

本次 current acquisition：

```text
acquisition_started_UTC  = 2026-10-05T14:58:04.773972+00:00
acquisition_finished_UTC = 2026-10-05T14:58:05.054492+00:00
primary_response_received_UTC = 2026-10-05T14:58:05.042647+00:00
primary_HTTP_Date = Mon, 05 Oct 2026 14:58:04 GMT
```

Primary GET 参数：`period1=1790654400`、`period2=1790913600`、`interval=1d`、`events=div,splits,capitalGains`、`includePrePost=false`。请求还冻结了 actions/keepna/repair/adjustment、threads/progress 和 session impersonation 配置，未归档 cookies 或 crumb。yfinance 为 timezone bootstrap 发起的辅助 `range=1d` 响应也原样保存，但没有将其当日 Volume 纳入调查。

当前官方 [yfinance.download 文档](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html) 说明 start inclusive、end exclusive；本次记录按当前行为解释。**当前文档或当前安装版本不能认证旧版本和未记录的 legacy 请求语义。** 文件 mtime、Git 时间未被用作获取时间。

## 5. Primary source comparison

| Evidence | 2026-09-30 Volume | Authority | 状态与限制 |
|---|---:|---|---|
| Legacy raw Yahoo response | unknown / unavailable | Level 1 | 缺失，不补值 |
| Legacy original yfinance table | unknown / unavailable | Level 1 | 缺失，不补值 |
| Legacy normalized `data/raw/NVDA.csv` | 121269300 | Level 3 | 可验证的历史 normalized record |
| Expanded frozen original Yahoo chart | 121732200 | Level 1 | 保存原始 JSON 与 hash |
| Expanded frozen original yfinance table | 121732200 | Level 1 | 保存原始 multi-index CSV 与 hash |
| Expanded normalized market | 121732200 | Level 3 | 与本侧原始输入一致 |
| Current Yahoo chart / yfinance / normalized | 121732200 | Level 4 | 仅代表本次查询的 vendor state |
| Third-party Volume | unknown / unavailable | Level 5 | 未查询 |

Current context Volume：2026-09-29 为 101494900，2026-09-30 为 121732200，2026-10-01 为 98591400；仅 primary date 进入 discrepancy 分类。current 与 expanded 相符不能证明 legacy 原始值或根因。

## 6. Processing trace

`NVDA_stage13_1_processing_trace.csv` 共 14 行，记录 side、layer、source_file、source_hash、date、field、value、available、authority_level 和 representation。缺失的 legacy 原始层 `available=false`，value 留空。

| Layer | Legacy | Expanded | 比较 |
|---|---:|---:|---|
| Raw Yahoo chart | unavailable | 121732200 | 无法比较原始输入 |
| Raw yfinance table | unavailable | 121732200 | 无法比较原始输入 |
| Normalized market | 121269300 | 121732200 | +462900 |
| Stage 3 | 121269300 | 121732200 | +462900 |
| Stage 5 | 121269300 | 121732200 | +462900 |
| Stage 6.1 | 121269300 | 121732200 | +462900 |
| Stage 8 labeled | 121269300 | 121732200* | +462900 |

Legacy 五层来自 `data/raw/NVDA.csv`、`data/processed/NVDA.csv`、`data/research/NVDA_research.csv`、`NVDA_features.csv`、`NVDA_labeled.csv`。Expanded 各层来自 Stage 13 原始 chart/table、`raw/NVDA_market.csv`、market_history、pit_states、research_matrix 和冻结 overlap audit。

*Stage 13 未保存独立 expanded labeled CSV。Stage 8 expanded value 取自原 `NVDA_stage13_unexpected_overlap_differences.csv` 对当时内存 labeled join 的冻结审计；明确记作 `frozen_audit_of_in_memory_join`，是 PARTIAL/间接证据。没有伪造一个独立 labeled artifact。

可用证据共核对 10 条传播关系，`propagation_consistency_violations=0`。legacy normalized → downstream 已验证；legacy original source → normalized 未验证。expanded original source → normalized → downstream 在可用直接/间接证据范围内一致。

仅对 frozen expanded yfinance 的邻近两行重放 `nasdaq_research.data.standardize_history`：原始 Volume、replay Volume、记录的 normalized Volume 均为 121732200，violations 为 0。该路径使用 `pd.to_numeric` 和已有日期标准化，不对 Volume 做 arithmetic。未重建 features/targets 或 rerun 全部 Stage 13 数据生产。

## 7. First divergence

```text
first_divergence_layer = NOT_IDENTIFIABLE_WITH_AVAILABLE_EVIDENCE
first_observed_divergence_layer = normalized_market
causal_first_divergence_identified = false
```

两侧最早共同可用且不一致的证据是 normalized market：legacy 121269300、expanded 121732200。其更早的两个 legacy 原始层均缺失，因此不能判断原始输入已不同，还是在无法恢复的 source → normalized boundary 出现变化。不能把最早可观察的层误写成已定位的第一因果层。

证据文件和 hash 记录在 `NVDA_stage13_1_first_divergence.json`；legacy normalized SHA-256 为 `066133d6cbaa0496cd6e7cfe8626542e269d7da013862b2bbeaced2db4fbd920`，expanded normalized SHA-256 为 `019077bc0f1708ac4ac2ffb73613d28836b8b315070046e13c9ec59208b1fc14`。

## 8. Root cause evaluation

| 候选 | 结论 | 证据与限制 |
|---|---|---|
| PIPELINE_PROCESSING_BUG | NOT_SUPPORTED | expanded 原始输入、标准化重放及传播一致；legacy 已保存层传播一致。没有定位到导致本次差异的 exact file/function/transformation bug。缺少 legacy 原始输入，不能完全排除未恢复的早期处理、cache 或 session boundary 问题 |
| LEGACY_INCOMPLETE_SESSION_SNAPSHOT | NOT_ESTABLISHED | 缺少可靠 legacy request/retrieval/source-finalization 证据。较小 Volume 和日线开盘 timestamp 均不能证明获取时 session 未结束 |
| VENDOR_SOURCE_VINTAGE_DRIFT | NOT_ESTABLISHED | current Yahoo 与 expanded 一致，但缺少 legacy frozen original source，且 incomplete-session possibility 未解决，不能认证两次完整 source vintage 间的历史修订 |
| UNRESOLVED_SOURCE_DISCREPANCY | SUPPORTED_FALLBACK | 原始输入及同期 timing 缺失，仍有多个可解释 discrepancy 的路径，符合预注册 unresolved fallback |

前三类未达到证据门槛，并非全部被彻底排除。仍无法区分 legacy 未完成 session/source 的获取、vendor historical revision，以及未恢复的 pre-normalized source/session/cache 差异。

## 9. Final root cause classification

```text
root_cause = UNRESOLVED_SOURCE_DISCREPANCY
```

观察到的独立 discrepancy 为 1，已解决为 0；原 normalized 层加四个 downstream manifestations 共 5 个历史 violations。当前响应补充了 Level 4 corroboration，但不能补齐 Level 1 legacy 输入或 Level 2 contemporaneous timing。

## 10. Canonical snapshot policy

四类自动 policy mapping 在调查前固定：processing bug 对应独立 corrected namespace 且须修复/revalidate；incomplete-session 对应经证据认证的 Stage 13 complete-session frozen snapshot；vendor drift 对应 Stage 13 frozen multi-year vintage 和明确 vintage transition；unresolved 对应不批准任何 expanded authority。

实际由 unresolved 自动映射：

```text
canonical_policy_frozen = true
future_stage14_plus_market_data_authority = null
candidate_authority_approved = false
legacy_vendor_vintage_preserved = true
stage13_vendor_vintage_preserved = true
manual_value_override_allowed = false
```

**已冻结可审计的“不批准”策略；尚未批准一个供 Stage 14+ 使用的 canonical market snapshot。** Stage 13 `raw/NVDA_market.csv` 仅记录为 frozen candidate，path/hash 可审计，不获得 authority。保留两侧历史数字，不 splice，不通过 override、tolerance、删字段、删日期或 post-hoc expected 分类制造 overlap equality。策略保存在 `NVDA_stage13_1_canonical_snapshot_policy.json`。

## 11. Stage 13 historical status

```text
Stage 13 remains INCOMPLETE_OR_BLOCKED as historical record
original unexpected_overlap_violations = 5
```

Stage 13 原 summary、validation、报告和 overlap audit 均保持原字节。未将旧状态写为 COMPLETE，未将 violations 改为 0；本次 unresolved 同时意味着原 blocker 仍未解除。

## 12. Expanded dataset downstream approval

```text
expanded_dataset_approved_for_downstream_research = false
```

未达到 root cause resolved + canonical authority approved 的验收条件。工程 trace/hash/reproducibility checks 全部通过，不能代替缺失的因果证据。

## 13. Stage 14 eligibility

```text
stage14_modeling_eligible = false
stage14_designed = false
stage14_executed = false
```

未设计、进入或执行 Stage 14。

## 14. Immutability

| 保护范围 | Files checked | Mutation violations |
|---|---:|---:|
| Stage 1–12 当前输入及已有代码 | 158 | 0 |
| Stage 13 原始 evidence/code/tests/report | 30 | 0 |
| 合计 | 188 | 0 |

Stage 1–12 的 158 项比 Stage 13 原先的 157 项多保护了本阶段开始前已经修改的 `data.py`，不存在减少原保护范围的情况。Stage 13 的 30 项包括 27 个 data/raw artifacts 和原模块、测试、报告。原 Stage 13 upstream manifest 的 157 项也全部一致。未修改已有代码、旧 tests 或任何 Stage 1–13 原 artifact；新模块只增加独立 investigation/policy audit。

## 15. Final Test state

```text
final_test_locked = true
final_test_predictions_generated = false
final_test_metrics_computed = false
final_training_pool_used = false
final_test_performance_dependency = false
new_model_fit = false
new_CV = false
feature_changes = false
target_changes = false
```

Final Test 仍为 50 行、2026-07-15 至 2026-09-23；primary date 2026-09-30 在其范围外。本次没有生成新的 Final Test 预测、计算 Final Test metrics 或使用 final training pool。读取 lineage CSV 时只读取 primary date/Volume；隔离测试在临时 fixture 上更改 target/performance 字段，确认输出不依赖这些字段。旧回归测试使用原测试 fixtures，并未对 expanded research dataset 开展新的模型研究。

## 16. Tests

| 范围 | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Stage 1–12 historical tests | 437 | 0 | 0 |
| Stage 13 tests | 39 | 0 | 0 |
| 新 Stage 13.1 tests | 43 | 0 | 0 |
| Full suite | **519** | **0** | **0** |

Full suite 实际日志：`Ran 519 tests in 240.253s / OK`。最后只补充 inventory 元数据呈现后，再次执行 43 项 targeted tests：`Ran 43 tests in 0.681s / OK`；随后完成最终两次正式复现。旧 tests 未修改或弱化。

新 tests 覆盖 trace schema/completeness、raw-same/processed-different、raw-different、缺原始源时区分 first observable 与 first causal layer、incomplete-session 与 vendor-drift evidence guards、unresolved fallback、四类 policy mapping、未修复 bug/工程失败不批准、exact integer handling、传播异常、旧版本未知字段、保护 hash、Stage 13 原 blocked status、Final Test isolation、冻结 current evidence 和禁止联网的离线复现。

执行命令：

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_source_reconciliation.py -v
```

日志位于 `/tmp/nasdaq-stage13-1-full-tests.txt`、`/tmp/nasdaq-stage13-1-final-tests.txt`；执行 verification 记录实际计数和日志 SHA-256。独立导入 reconciliation 模块时 sklearn/xgboost 未载入，正式 pipeline 未 fit 模型。

## 17. Reproducibility

最后版本连续两次执行：

```bash
PYTHONPATH=src .venv/bin/python -m nasdaq_research.source_reconciliation
```

两次均返回 `UNRESOLVED_BLOCKER`，approval 和 eligibility 均 false。**14 个根目录正式 deterministic CSV/JSON 全部逐字节一致，8 个 frozen evidence/external files 全部一致，合计 22/22，mismatches=0。** current acquisition 只冻结一次；复现不联网、不更新 retrieval timestamp、不重写 Stage 13 sources。

14 个正式文件：

```text
NVDA_stage13_1_protocol.json
NVDA_stage13_1_protected_sha256.json
NVDA_stage13_1_initial_repository_state.json
NVDA_stage13_1_local_evidence.json
NVDA_stage13_1_source_inventory.csv
NVDA_stage13_1_acquisition_metadata.csv
NVDA_stage13_1_processing_trace.csv
NVDA_stage13_1_timezone_session_context.csv
NVDA_stage13_1_first_divergence.json
NVDA_stage13_1_source_comparison.csv
NVDA_stage13_1_root_cause_evidence.csv
NVDA_stage13_1_canonical_snapshot_policy.json
NVDA_stage13_1_summary.json
NVDA_stage13_1_validation.json
```

8 个冻结辅助文件包括 local trace bundle、source discovery audit、旧 HEAD downloader 副本、current acquisition manifest、current yfinance/normalized CSV 和两个 current chart JSON。raw chart 文件名含内容 SHA-256。

另新增 `NVDA_stage13_1_execution_verification.json` 作为两次运行结束后的验收记录，保存两次 hash 集、tests、Git 和最终保护审计；该文件不冒充两次 pipeline 复现的第 15 个 artifact。其 JSON serialization/roundtrip 单独验证一致。报告及该记录生成后，前述 22 文件再次确认未变。

正式输出目录 tmp/partial files 为 0。`/tmp` 中保留测试日志、两次 hash 清单和本次下载的运行时 cache，均不属于 authoritative research output。

## 18. Git status

最终执行 `git status` 和 `git diff --check`，HEAD/branch 未变，diff check 通过。实际变更区分如下：

| 范围 | 路径 | 性质 |
|---|---|---|
| Pre-existing Stage 13 | `src/nasdaq_research/data.py` | 已有 tracked modification，字节未变 |
| Pre-existing Stage 13 | `data/research/historical_expansion/stage13_nvda/` | 已有未跟踪 artifacts/raw，字节未变 |
| Pre-existing Stage 13 | `docs/stage13_execution_report.md` | 已有未跟踪报告，字节未变 |
| Pre-existing Stage 13 | `src/nasdaq_research/historical_expansion.py` | 已有未跟踪模块，字节未变 |
| Pre-existing Stage 13 | `tests/test_historical_expansion.py` | 已有未跟踪测试，字节未变 |
| New Stage 13.1 | `src/nasdaq_research/source_reconciliation.py` | 新未跟踪模块 |
| New Stage 13.1 | `tests/test_source_reconciliation.py` | 新未跟踪测试 |
| New Stage 13.1 | `docs/stage13_1_execution_report.md` | 新未跟踪报告 |
| New Stage 13.1 | `data/research/historical_expansion/stage13_1_yahoo_reconciliation/` | 新未跟踪 protocol、证据、策略与验证 |

最终普通 short status 会把两个阶段的未跟踪 data 子目录汇总为 `?? data/research/historical_expansion/`，完整文件清单由 execution verification 保存。未执行 add、commit、push、tag，也未 reset、checkout、restore、stash 或 clean。

## 19. Completion classification

```text
completion_status = UNRESOLVED_BLOCKER
root_cause_resolved = false
investigation_engineering_checks_passed = true
validation.valid = false
```

`validation.valid=false` 表示正式 root-cause resolution/acceptance 条件未满足；不是 trace、hash、tests 或 reproducibility 失败。完成了授权范围内可执行的调查，但不能将根因未解决写为 resolved。

## 20. Final conclusion

**2026-09-30 NVDA Volume discrepancy 的根因尚未被可靠定位。** 最早可观察到差异的层为 normalized market；真正的 source/processing/timing 因果起点因 legacy 原始输入和同期 metadata 缺失而无法确认。没有发现已证实的处理 bug，也没有足够证据区分 incomplete-session acquisition 与 vendor-vintage drift。

**Stage 13 expanded 2020–2026 dataset 已拥有明确、可审计的冻结“不批准”策略，但尚未获得 canonical market-data authority，不能用于未来 multi-year walk-forward research。** 两个历史快照均原样保留。解除该 blocker 需要可恢复的 legacy original Yahoo/yfinance input，以及可靠 acquisition/query/version/session 元数据，才能在原预注册规则内判断因果起点；重复查询 current Yahoo 或引用第三方数字不能单独解除。

没有修改 Stage 13 历史结论；没有进行新的模型评价；没有打开 Final Test；没有进入或执行 Stage 14。本阶段到此停止。
