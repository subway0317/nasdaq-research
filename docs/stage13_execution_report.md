# Stage 13 — NVDA 多周期历史证据库扩展执行报告

最终分类：**INCOMPLETE_OR_BLOCKED**。数据、PIT、公式、测试与复现工作已完成，但正式 overlap acceptance 未通过：Yahoo 新旧快照的 **2026-09-30 成交量**存在一项独立来源差异，在五个旧层中形成 **5 个 unexpected overlap violations**。它不是 pre-sample expansion effect，未静默替换、修复或豁免。

## 1. Scope

固定 research window：NVDA，2020-01-01 through 2026-09-30。实际 observed sessions 为 2020-01-02 至 2026-09-30，共 **1695 行**。本阶段仅构建多年数据基础及 data/PIT/provenance/compatibility audits，没有对 expanded dataset 进行模型拟合、CV、MAE、相关性、Sharpe、PnL 或模型排名；没有设计新的 splits。

raw window 根据已保留 SEC filing history 的最早 filing 与真实 60-session lookback 需求确定，起点为 2009-08-20，未硬编码“2020 减 90 天”。保留 **2609 个 pre-sample market sessions**，用于历史 activation calendar 和预热；保留全部 SEC 历史，没有在 2020 sample boundary 删除此前已公开信息。

## 2. 修改 / 新增文件

唯一修改的旧代码：[src/nasdaq_research/data.py](../src/nasdaq_research/data.py)：给既有 downloader 增加可选 keyword-only `start/end`、archiving `session`、source snapshot 参数。默认 download kwargs 与旧版完全一致；既有 source snapshot 在联网前拒绝覆盖。未新增第二套 downloader，未改 price/feature/accounting/target 公式。

新增 [src/nasdaq_research/historical_expansion.py](../src/nasdaq_research/historical_expansion.py)：预注册、冻结 acquisition、复用 Stage 3/5/6.1/8 builders、独立 oracle、coverage、corporate actions、overlap、验收与 deterministic 输出。保留原阶段的 CSV/parser 边界，避免跳过边界造成 OHLC 末位浮点差异。

新增 [tests/test_historical_expansion.py](../tests/test_historical_expansion.py)：39 项独立 Stage 13 测试；synthetic prices/filings 仅作为 adversarial test fixtures，正式数据全部来自冻结真实响应。

新增本报告 `docs/stage13_execution_report.md`。其余新增文件全部位于 `data/research/historical_expansion/stage13_nvda/`：

| 文件 | 用途 |
| --- | --- |
| `NVDA_stage13_PIT_truncation_oracles.csv` | 45 个预注册规则选出的日期及 legal-prefix 比较结果 |
| `NVDA_stage13_acquisition_api_audit.json` | 唯一旧代码修改的 HEAD/hash/diff 及兼容性证据 |
| `NVDA_stage13_calendar_violations.csv` | 研究窗口交易日缺口/非 session 异常清单，当前只有 header |
| `NVDA_stage13_corporate_action_audit.csv` | 研究窗口 2 次 split / 27 次 dividend 的价格和公式审计 |
| `NVDA_stage13_corporate_action_events.csv` | 冻结 Yahoo raw response 的所有 dividend / split 记录 |
| `NVDA_stage13_expected_expansion_differences.csv` | 2400 个合法扩展差异的日期、字段、旧/新值及具体证据 |
| `NVDA_stage13_feature_coverage.csv` | 42 个原/relative 特征逐列 coverage、missing/infinity、首次有效日期 |
| `NVDA_stage13_fundamental_history.csv` | 全部 114 个 Stage 4 标准化 observations，含 pre-sample 与 YTD |
| `NVDA_stage13_fundamental_snapshots.csv` | 79 个 annual / standalone quarterly 快照、YoY 参考与各流选择身份 |
| `NVDA_stage13_market_history.csv` | 研究窗口 OHLCV 与原 9 个 market features |
| `NVDA_stage13_overlap_compatibility.csv` | 六层 × 字段的 exact/tolerance/expected/unexpected 分类计数 |
| `NVDA_stage13_pit_states.csv` | Stage 5 整体 observation、filing/effective dates、概念来源及信息年龄 |
| `NVDA_stage13_protocol.json` | 正式计算前锁定的窗口、公式、审计规则与禁止事项 |
| `NVDA_stage13_research_matrix.csv` | 1695 × 174；39 原特征、3 relative-SMA、独立 q/fy/bs 与原始来源/provenance |
| `NVDA_stage13_summary.json` | 实际 coverage、first-valid dates、验收分类与锁定状态 |
| `NVDA_stage13_targets.csv` | 1d/5d/20d 标签与 next-open / exit-close 完整 provenance；尾部保留 |
| `NVDA_stage13_unexpected_overlap_differences.csv` | 同一成交量差异在五个旧层中的 5 个非预期单元格 |
| `NVDA_stage13_upstream_sha256.json` | 157 个旧文件的运行前 SHA-256 manifest |
| `NVDA_stage13_validation.json` | 各项布尔验收、独立 oracle、raw/formal hashes 和失败门槛 |
| `NVDA_stage13_yearly_coverage.csv` | 2020–2026 年度 cell coverage、state coverage、目标可用数和质量计数 |
| `NVDA_stage13_execution_verification.json` | 完整测试、两次运行 hash 比较、最终 Git/immutability 验证证据 |

| raw/cache 文件 | 用途 |
| --- | --- |
| `raw/NVDA_acquisition.json` | 请求边界、价格参数与 raw SHA-256 manifest |
| `raw/NVDA_companyfacts.json` | SEC Company Facts 原响应逐字节副本 |
| `raw/NVDA_market.csv` | 标准化 OHLCV 冻结输入 |
| `raw/NVDA_yahoo_chart_c8925631bb56ef3700f384bd325454dad7d03548493c7b6ddf4942f8f5074227.json` | 成功 Yahoo chart HTTP response 原始 JSON bytes；不存 cookie/crumb |
| `raw/NVDA_yahoo_chart_d2306ab048d4171827c527cc46cc44aa6011145eb3902091ba451399b10aa6ae.json` | 成功 Yahoo chart HTTP response 原始 JSON bytes；不存 cookie/crumb |
| `raw/NVDA_yahoo_source.csv` | yfinance 原始表，保留 Adj Close 与 corporate-action 列 |

原 Stage 1–12 artifacts 没有覆盖或重建，没有修改历史 tests、README、映射、阈值、splits 或协议。

## 3. Historical acquisition

- Market raw：**2009-08-20 至 2026-09-30，4304 行**；请求 end-exclusive 为 2026-10-01，没有加载更晚市场 rows 来填充研究尾部 targets。
- Research：**2020-01-02 至 2026-09-30，1695 行**；2026 仅截至 9 月 30 日。
- SEC filing range：**2009-08-20 至 2026-08-26**；period_end range：**2009-07-26 至 2026-07-26**。
- SEC observations：**114**；其中 **73** 个 filing 早于 2020。复用既有 Company Facts vintage，Stage 13 raw copy 与原响应 SHA-256 相同，没有再次访问 SEC。
- 沙箱内首次行情请求因 DNS 解析失败；获准网络访问后的真实下载成功。保留 HTTP chart 原始 bytes、source table、标准化 OHLCV 与 hashes；正式运行均离线复用缓存。
- 完整 raw 响应有 **4304 timestamps**，标准化后仍为 **4304 行**；独立 raw calendar 也期望 **4304 sessions**。未发现标准化丢行。

下载参数沿用 `auto_adjust=False`；未用 Adj Close 替换 Close，未执行 repair、dividend re-adjustment、插值或 source splicing。[yfinance 官方 download 文档](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)记录了 start-inclusive、end-exclusive、adjustment 与 actions 参数。

## 4. Market feature construction

复用原 `build_features`：simple/log return、intraday return、daily range、SMA 5/20/60、volatility 20/60，共 **9** 个特征。Rolling volatility 是 simple return 的 **sample std、ddof=1、未年化**；SMA/volatility 均只使用当前及过去 observed rows。

raw 历史首次同时成熟 SMA60/vol60 的日期为 **2009-11-13**；研究窗口内首次成熟日期为 **2020-01-02**。研究窗口 market 与 relative-market cell coverage 均为 **100%**，未填充 warm-up NaNs。原 SMA 全部保留，新增三个下游 relative-SMA 均为 `Close / SMA_k - 1`。

## 5. Corporate-action audit

真实 raw response 包含研究窗口内 **2 次拆股、27 次 dividend**。两次拆股分别由 [NVIDIA 2021 官方公告](https://nvidianews.nvidia.com/news/nvidia-announces-four-for-one-stock-split-pending-stockholder-approval-at-annual-meeting-set-for-june-3)与 [2024 SEC filing](https://www.sec.gov/Archives/edgar/data/1045810/000104581024000144/nvda-20240607.htm)核对日期和比例。

| 拆股 session | 比例 | 相邻 Close ratio | 相邻 log return | 跨事件最大 absolute target return | violations |
| --- | --- | --- | --- | --- | --- |
| 2021-07-20 | 4:1 | 0.9910674367392721 | -0.0089726977854803 | 0.12686112939648897 | 0 |
| 2024-06-10 | 10:1 | 1.0074614554265338 | 0.0074337564657012 | 0.4485197838184680 | 0 |

全部 raw OHLC bounds violations **0**；registered ratio bounds 下 unexplained adjacent-close discontinuities **0**；split-event OHLC adjustment consistency、独立 SMA/volatility 以及跨拆股 label/source-position 检查 violations 均 **0**。没有人工 inverse-split jump。测试故意破坏 split convention 后会触发失败。

action metadata 仅用于审计，不进入 feature builders。需要保留原口径的实际限制：固定 Yahoo vintage 使用 split-adjusted price coordinate，不能据此宣称具备历史 as-traded price archive，也不能认证 absolute SMA 对供应商未来 retroactive price rescaling 不变。正式 leakage 结论是**给定冻结价格/SEC vintage 的计算 PIT 正确性**。

## 6. SEC / PIT history

- Annual observations：**17**；standalone quarterly：**62**；YTD：**35**；eligible snapshots：**79**。
- YTD 保留在 source history，但不进入默认 q/fy/bs streams；没有 YTD differencing，没有 accounting concept 替换。
- 保留每条 period、filing、accession、form、field concept 及 reference provenance。严格在 filing 后的下一 observed session 激活，filing day 不可使用该 filing。
- 三个 state streams 在全部 **1695** 个 research rows 均有身份；state coverage **100%** 不等于其所有 ratios/growth 全部可用。
- archived source 中真实 amendment observations **0**。新增独立 amendment fixture 验证 filing day/next session 边界、amendment 前不变，以及 effective 后能够实际改变状态的 positive control。

2020-01-02 的 q/bs 状态来自 `0001045810-19-000170`，filing 2019-11-14，effective **2019-11-15**；annual 状态来自 `0001045810-19-000023`，filing 2019-02-21，effective **2019-02-22**。它们正确使用研究边界前已公开的信息。q YoY reference 为 `0001045810-18-000150`；FY reference 为 `0001045810-18-000010`。

SEC Company Facts 的来源范围与当前 vintage 限制继续沿用原定义，不宣称涵盖每一历史公开 disclosure version。[SEC 官方 API 文档](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)说明 Company Facts 的概念/单位组织与提取范围。

## 7. Research feature history

原 whitelist 仍为 **39**：9 market、13 quarterly、13 annual、4 balance-sheet；另保留三个 relative-SMA，未将它们改成原 whitelist 或 production model。

first research row、mature 60-session market features、meaningful q state、annual state、BS state、valid 1d/5d/20d targets，均为 **2020-01-02**。

`fy_fcf_margin` **0/1695** 可用；`q_fcf_margin` **62/1695**；`q_ocf_margin` **443/1695**；`q_operating_cash_flow_growth_yoy` **321/1695**。Quarterly cash-flow margins 首次有效 **2020-05-22**，quarterly OCF YoY 首次有效 **2022-05-31**；`debt_to_assets` 首次有效 **2020-11-19**。其余逐字段 dates/missing counts 在 feature coverage artifact 中。

**不存在 39 个 features 同时非空的日期**，因此按 all-39 complete-case + primary-target 可用这个结构性定义，earliest date 为 **null**。这是 availability 报告，没有拟合模型，没有决定未来 preprocessing/feature retention，也没有删除早期年份或 sparse features。

## 8. Target history

定义不变：`Open[t+1]` 入场，`Close[t+1/t+5/t+20]` 退出；t+n 指 observed-row position。Primary identity 仍为 `forward_return_5d`。全部 research rows 保留。

| Horizon | Valid count | Tail NaNs |
| --- | --- | --- |
| 1d | 1694 | 1 |
| 5d | 1690 | 5 |
| 20d | 1675 | 20 |

entry date/open、每个 horizon 的 exit date/close 全部保留。独立 target oracle 检查 **4935 个 horizon cases**，violations **0**；有意跳过 **50 个 locked Final Test rows 的 numerical target payload**。只记录标签 availability/count，不输出 Final Test target distributions、预测或 performance。

## 9. Yearly coverage table

下表 coverage 为 registered feature cells 的非空比例；q/FY/BS state-identity coverage 在各年均为 **100%**。NaN 单独作为 missing cells 报告，未与 infinity 合并或填充。

| 年 | Rows | Market | Relative | Quarterly | Annual | BS | 39 features | 1d/5d/20d valid | Missing cells | Infinity | Duplicates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 253 | 100.00% | 100.00% | 51.81% | 92.31% | 77.87% | 79.10% | 253/253/253 | 2062 | 0 | 0 |
| 2021 | 252 | 100.00% | 100.00% | 59.68% | 92.31% | 100.00% | 83.99% | 252/252/252 | 1573 | 0 | 0 |
| 2022 | 251 | 100.00% | 100.00% | 80.91% | 92.31% | 100.00% | 91.07% | 251/251/251 | 874 | 0 | 0 |
| 2023 | 250 | 100.00% | 100.00% | 80.80% | 92.31% | 100.00% | 91.04% | 250/250/250 | 874 | 0 | 0 |
| 2024 | 252 | 100.00% | 100.00% | 80.77% | 92.31% | 100.00% | 91.03% | 252/252/252 | 882 | 0 | 0 |
| 2025 | 250 | 100.00% | 100.00% | 80.80% | 92.31% | 100.00% | 91.04% | 250/250/250 | 874 | 0 | 0 |
| 2026 | 187 | 100.00% | 100.00% | 82.44% | 92.31% | 100.00% | 91.58% | 186/182/167 | 614 | 0 | 0 |

独立研究交易日 calendar：expected/observed **1695/1695**，missing sessions、unexpected non-sessions 均 **0**。完整 raw calendar：**4304/4304**，violations **0**。包括 Juneteenth 自 2022 生效、Saturday New Year 特殊规则、Sandy/Bush 闭市和 [2025-01-09 Nasdaq 悼念闭市](https://ir.nasdaq.com/news-releases/news-release-details/nasdaq-announces-closure-its-us-markets-honor-national-day-0)。Calendar audit 不参与 target offsets，target 仍用 observed rows。

## 10. Overlap compatibility

共同 dates：**251**。比较 legacy raw、Stage 3 market、Stage 5 PIT、Stage 6.1 features、Stage 8 targets、Stage 8 labeled，共 **447 个 layer/field comparisons**；所有单元格均得到分类。

- Exact matches：**103030**。
- Derived strict-tolerance matches：**6062**（rtol=0、atol=1e-12；raw OHLC/SEC/metadata 仍 exact）。
- Expected expansion differences：**2400**，逐项记录 old/new values 和独立证据。
- Unexpected violations：**5**，对应**一项独立成交量来源差异**。
- Locked numerical target cells skipped：**700**；日期及 availability 不用于 performance decisions。

Expected differences 仅有两类：

1. **656** 个 registered market warm-up cells：simple/log return 各 4；SMA5/20/60 为 16/76/236；vol20/60 为 80/240。每个差异均属于原始 sample position 的既定 warm-up，original-boundary reconstruction 仍 NaN，full-history independent scalar oracle 正确。这些 counts 包含四个 downstream 层的重复记录。
2. **1744** 个旧 pre-window filing 的 effective date / truncation flag / basis / effective-age cells：accession 和真实 filing identity 相同；原 first-session 截断可独立重建；expanded effective date 由完整 observed calendar 恢复。q、FY、BS/global 的相应 metadata 全部逐单元格保存，不将 raw financial values 或不同 state identity 豁免为 expected。

唯一独立非预期差异：

```text
Date:          2026-09-30
Field:         volume
Legacy:        121269300
Expanded:      121732200
Difference:    +462900
```

raw、Stage 3、Stage 5、Stage 6.1、Stage 8 labeled 各出现一次，故 violations = **5**。OHLC source 在相应序列化/parser 边界完全一致；targets/provenance 的非锁定数值没有 unexpected mismatch。成交量更新的根因未从 source response 中获得独立证实；没有猜测其原因，没有 source splicing，没有把它标记为 expected，没有修改旧数据。**`overlap_compatible = false`，`validation.valid = false`**。

## 11. PIT / leakage adversarial tests

| Audit | 代表 dates / checked cells | Violations |
| --- | --- | --- |
| `PIT_truncation` | 45 | 0 |
| `amendment_timing_isolation` | — | 0 |
| `deterministic_source_order` | — | 0 |
| `future_SEC_isolation` | — | 0 |
| `future_market_isolation` | — | 0 |
| `future_split_metadata_isolation` | — | 0 |
| `future_target_isolation` | — | 0 |
| `independent_YoY_oracle` | 553 | 0 |
| `independent_state_oracle` | 1575 | 0 |
| `market_rolling_oracle` | 405 | 0 |
| `relative_SMA_oracle` | 135 | 0 |

Truncation 的 **45** 个 dates 依赖年度边界、真实 state transitions 与 split 附近 observed sessions 选出，不依赖 targets/performance。比较完整历史 feature(t) 与仅使用 market <=t、filing <t 的重建结果；所有列、state provenance、YoY 与 relative-SMA 都匹配。Feature oracle 与 target oracle 分离。

Future SEC、future market、target/provenance mutation、source-order shuffle、same-day accession ties、collision-safe metadata、YTD exclusion、amendment timing tests 均通过。Future action metadata 不进入 construction；给定 raw vintage 的 metadata isolation 通过，未将供应商价格 vintage 不变性伪报为已认证。

## 12. Final Test state

原 **50 rows，2026-07-15 至 2026-09-23** 完全保留既有 lock；Stage 9.1 splits、Pre-Test Gap 与 Final Training Pool artifacts 均未修改或使用。

```text
final_test_locked = true
final_test_predictions_generated = false
final_test_metrics_computed = false
final_training_pool_used = false
no_final_test_performance_dependency = true
```

Expanded matrix 可以包含这些 dates；数据构建与 allowed source consistency 不构成 evaluation。没有预测、Final Test performance metrics、distribution-driven intervention 或窗口/feature/cleaning 选择。

## 13. Tests

- Stage 13：**39 passed，0 failed**。
- 历史 tests：**437 passed，0 failed**。
- Full suite：**476 passed，0 failed**；实际日志 `Ran 476 tests in 237.535s / OK`。
- acquisition API 最终兼容性复核：**8 passed，0 failed**。

全套回归测试使用既有阶段的测试 fixture / 临时输出目录验证既有行为；它们没有以 expanded data 进行模型研究，也没有重写权威旧 artifacts。Stage 13 formal module 不导入 sklearn/modeling/evaluation modules。测试通过表示审计实现正确，**不将真实来源差异导致的验收失败变成成功**。

复现命令：

```bash
PYTHONPATH=src .venv/bin/python -m nasdaq_research.historical_expansion --register-only
PYTHONPATH=src .venv/bin/python -m nasdaq_research.historical_expansion --acquire-only
PYTHONPATH=src .venv/bin/python -m nasdaq_research.historical_expansion
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_historical_expansion.py' -v
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

已存在 cache 时 `--acquire-only` 只验证/复用，不重复下载。既有 protocol 或 cache identity 不一致会拒绝，不自动覆盖。

## 14. Reproducibility

实现稳定后连续进行 **两次正式 offline runs**：**20 个正式 CSV/JSON** 全部 SHA-256 相同，即 **byte-for-byte identical**；**6 个 frozen raw/cache** 文件也未变化，mismatches **0**。Protocol 与 acquisition API audit 包含在这 20 个比较文件中。运行耗时/运行时间戳未进入正式 artifacts。

`NVDA_stage13_execution_verification.json` 是这两次运行之后生成的 verification record，包含完整 20+6 个 before/after hash；它不是被计入“两次生成”的第 21 个 pipeline artifact。该 record 自身采用 deterministic JSON 生成并校验重复序列化字节一致。没有 `.tmp`、`.partial` 或 half-written files。

## 15. Upstream immutability

运行前 manifest 覆盖 **157 个旧受保护文件**（包含 **114 个旧 data 文件**及旧 tracked code/docs/config）；implementation、formal runs、tests 完成后重新校验，**SHA-256 mutation violations = 0**。

`data.py` 是预先声明的唯一必要兼容 API 修改，单独保存 original HEAD SHA-256、current SHA-256 和 exact diff，未将这项 code change 隐藏为“所有旧代码都没变”。Stage 1–12 authoritative artifacts 全部 immutable。

## 16. Git status

初始 main clean；HEAD **bde8ed98c34c7a3acb4b8911ae629cdbd99e363f**（`bde8ed9 Complete stage 12 post-representation signal diagnostics`），与本地 `origin/main` reference 的 ahead/behind 为 **0/0**。开始时无用户未说明的修改。没有使用 restore、stash、delete 或覆盖用户工作。

最终预期状态由 verification record 实际记录：

```text
 M src/nasdaq_research/data.py
?? data/research/historical_expansion/
?? docs/stage13_execution_report.md
?? src/nasdaq_research/historical_expansion.py
?? tests/test_historical_expansion.py
```

`git diff --check` **通过**；HEAD/branch 未改变。**没有 git add / commit / push / tag**。

## 17. Completion classification

**INCOMPLETE_OR_BLOCKED**。目标窗口、原语义、PIT、预热、corporate actions、年度 coverage、leakage tests、full test suite、复现和 upstream immutability 均完成并通过相应检查；但预注册要求 unexpected overlap violations = 0，而真实结果为 **5**，因此不能使用两个 COMPLETE 类别。

所有 expected differences 均有 pre-sample truncation 证据；新旧成交量差异不能通过扩大 expected 定义、豁免非特征字段或修改 protocol 来消除。保留了可复核的 expanded artifacts 和失败证据，未把未验收的数据基础发布为已通过。

## 18. Final conclusion

**当前还不能宣布 Stage 1–12 pipeline 已成功扩展为通过全部验收、可信且可复现的 2020–2026 多年历史研究数据基础。** 实际 expanded foundation 已构建，给定 frozen source vintage 的 PIT correctness、tests 和 reproducibility 均通过，但 strict overlap acceptance 被一项真实 vendor volume 差异阻断；历史 as-traded/完整 SEC disclosure vintages 的继承限制也继续明确保留。

**没有进行新的模型评价。没有打开 Final Test。没有进入下一 Stage。没有自动设计或执行 Stage 14。**
