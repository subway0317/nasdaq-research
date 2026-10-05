# Legacy 原始行情证据一次性恢复审计

正式任务：**Legacy Evidence Recovery Audit**。这是只读 forensic evidence recovery audit，不是 Stage 13.2 或 Stage 14。最终分类：**LEGACY_EVIDENCE_NOT_RECOVERABLE**。

在预注册的 preserved Git/project 搜索范围内，没有恢复足以 materially improve legacy first-divergence identifiability 的原始 Yahoo/yfinance input 或可靠 acquisition/runtime provenance。该结论限制在本次搜索过的保存证据集，不表示原则上不可能恢复。

## 1. Scope

唯一 primary target 为 NVDA / 2026-09-30 / Volume / legacy normalized value 121269300。只寻找原始 vendor/library input、实际 query/endpoint/request/runtime records 和可靠获取时间。

本阶段无网络请求、无新的模型研究或评价、无 data remediation、无新 Yahoo/SEC/第三方 acquisition。没有修改 Stage 13/13.1 classification、canonical policy 或 downstream approval。Historical tests 只运行原 fixtures；没有以 expanded dataset 开展模型操作。

允许的写入仅为新增 Audit module/tests/artifacts/report，旧研究文件未修改。项目新增模块为 `src/nasdaq_research/legacy_evidence_recovery.py`，测试为 `tests/test_legacy_evidence_recovery.py`。

## 2. Initial repository state

开始执行 `pwd`、`git status`、`git log -3 --oneline`、`git diff --check` 并核对 refs/remote。

```text
cwd         = /home/zbw21/projects/nasdaq-research
branch      = main
HEAD        = d6c4b2ce51dd3a38e4fc4847c8a7c1d1ac870fbd
origin/main = d6c4b2ce51dd3a38e4fc4847c8a7c1d1ac870fbd
origin URL  = https://github.com/subway0317/nasdaq-research.git
working tree = clean
```

实际 checkpoint 与用户记录一致，Stage 13/13.1 已提交。开始时无用户修改。初始状态保存为 `NVDA_legacy_recovery_initial_repository_state.json`。

搜索前冻结 201 个 tracked files 和 501 个 `.git` physical files 的 SHA/size。再以该 checkpoint 已提交且 hash 受保护的 Stage 13.1 manifest 核对其 188 项旧输入，其中 13 项不在本次 tracked list；合计覆盖 214 个独立研究文件。该补核仅用于 preservation，不把旧 Stage 13.1 context 当作 recovery。

## 3. Search protocol

正式搜索前写入 `NVDA_legacy_recovery_protocol.json`，SHA-256：`4c71dc61e82a403934af9f9b3a45b2a9ac2d65cf2e3e2a0506aa74e513bad93f`。之后没有改写搜索顺序、成功标准或证据规则。

固定顺序：

1. Current reachable Git history。
2. Deleted files in reachable history。
3. Git reflog。
4. Unreachable Git objects。
5. Current repository-local legacy files。
6. Limited `/home/zbw21/projects` files/clones/archives。
7. Historical implementation evidence。

边界只包括当前 repository 和有限 projects 名称/相关文件搜索，projects directory depth 上限 6；单个文本/blob 上限 32 MiB，相关 zip/tar archives 上限 100 MiB。未扩展到整个 home、根文件系统、Windows filesystem 或全局 shell history，未读取无关项目内容，未运行其它项目代码。

使用 `GIT_OPTIONAL_LOCKS=0` 执行只读 Git commands。未执行 reset/restore/checkout/clean/gc/prune/repack/reflog expire/rebase/add/commit/push/tag；fsck 未使用 `--lost-found`。未改变原路径或覆盖当前文件。

已知 Stage 13/13.1 data directories 和原报告不算新 legacy raw evidence；Audit 自己的目录、module/tests/report 排除。普通代码、测试、processed data、报告和 mtime 均不满足 recovery criterion。

## 4. Reachable Git history

实际检查 **40 个 reachable commits**；`git rev-list --objects --all` 搜索 **372 条 object/path records**。对各 commit 使用 ls-tree，得到 **121 个不同 matching blob/path pairs**，覆盖历史 NVDA/raw/cache/market/acquisition 等候选名称。

排除 known context 和 performance payload 后，对 **18 个 candidate paths** 阅读完整 `git log --all --name-status -- <path>` history。历史 normalized/downstream artifacts 存在，但没有找到新的 Yahoo chart response、原始 yfinance table 或实际 acquisition/runtime record。

输出 `git_reachable_inventory.csv`、`commit_inventory.csv`、`reachable_path_history.json` 和 `reachable_git_log.txt`。Inventory 包含 object SHA、历史 commit availability、current path existence 和 exclusion/classification role；具体 representation 由 candidate table 记录。

## 5. Deleted file history

执行 `git log --all --diff-filter=D --summary` 及 name-status 审计。

```text
all deleted path events = 0
relevant deleted blob paths = 0
recovered deleted-file raw/runtime evidence = 0
```

未发现可由 deletion parent 读取的相关旧原始文件。`NVDA_legacy_recovery_deleted_history.csv` 为 header-only，未 checkout 或 restore 文件。

## 6. Reflog audit

`git reflog --all --date=iso` 检查 **37 条 entries**、**13 个 distinct commits**；这些 commits 全部已经在 reachable refs history 中，outside-reachable commits 为 **0**。

对这 13 个 commit 的树检查得到 362 个 matching path occurrences，去重候选仍为已知历史 artifacts，未恢复新 raw/runtime evidence。Reflog 时间只证明 ref movement，不是 acquisition timestamp。

## 7. Unreachable object audit

执行只读 `git fsck --full --no-reflogs --unreachable`，先 inventory type/OID/size，再检查合理大小的内容，不做大规模 binary dump。

以**开始时冻结的 checkpoint object set**为正式 recovery universe：

| Object type | Preserved unreachable count | Inspected |
|---|---:|---:|
| commit | 0 | 0 |
| tree | 23 | 23 |
| blob | 60 | 60 |
| 合计 | **83** | **83** |

没有恢复原始 source/runtime evidence。原始 blob 的关键词命中均继续作 representation/provenance 分类，而非直接认作 raw。

**必须保留的 Git diagnostics 与边界：**

- fsck 非零退出并报告 `missing tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904`。这是 canonical empty tree 的 OID，可由 `SHA1("tree 0\0")` 认证，定义上没有条目；遍历时按零条目处理，没有合成或写入 Git object，也没有修复 Git。
- 在新增审计文件期间，`.git/objects` 出现了开始后新增的 loose objects。原 501 个 Git files 全部保持原字节，但**整个 object database 不再 byte-identical**。审计调用的 Git commands 均为只读；新增对象的具体来源未据此推断。不能把这些 audit-era objects 算作旧 legacy evidence。
- 最终第二次正式运行的实时 fsck 为 **0 commits / 23 trees / 98 blobs**，共 121 个 unreachable objects，其中 38 个 objects 不在初始 checkpoint snapshot。正式 inventory 保留的是初始 **0 / 23 / 60**。完整实际 stdout、return code 2 和新增对象差异另存 execution logs，没有用 83 冒充实时总数。

上述变化不涉及 HEAD/ref/reflog 重写或已有对象删除。保存的原始对象均已检查；canonical empty tree 的缺失未阻止检查非空 preserved trees/blobs。结论仍限定于已保存且本次搜索的 evidence set。

## 8. Repository-local search

排除 Git/runtime dependency caches、known Stage 13/13.1 data/report 及 Audit 自身后，看到 **162 个文件名**，inventory **138 个文本/相关文件候选**，实际内容读取 **75 个**。

`121269300` 的命中均被分类为 normalized/downstream data、implementation、report/documentation 或 test fixture。没有把 `data/raw/NVDA.csv` 认作 vendor response；它是 project-normalized lowercase OHLCV CSV。未发现足够强的 repository-local acquisition logs、notebook output、timestamped request 或 manifest。

可能含 performance 的 artifacts 只作路径 inventory，排除内容检查；未读取 Final Test performance。旧 Stage 13/13.1 context 出现的数字不算恢复。

## 9. Limited ~/projects search

在 repository 未找到 qualifying evidence 后，扩展到预注册 `/home/zbw21/projects` 范围，只检查明确与 nasdaq/NVDA/Yahoo/yfinance 相关的名称和文件。

```text
related old clones = 0
related backups/archives = 0
candidate files = 0
archive members inspected = 0
unrelated project contents read = 0
```

没有进一步明确 evidence pointer，因此未扩展到 home、Windows 或全局 history。没有修改其它 repository。

## 10. Historical implementation evidence

最后检查 downloader path history：`src/nasdaq_research/data.py` 共 **8 个 path-change commits**、**4 个 distinct blobs**，并只读核对相关 config defaults。

有效 legacy implementation 可描述 `yf.download()`、`period=1y`、`interval=1d`、`auto_adjust=False`、`progress=False`、`threads=False`，以及 reset_index、rename、Volume 的 `pd.to_numeric` 和 normalized CSV serialization。Stage 13 checkpoint 版本增加原始 table capture；这不意味着旧运行也保存了原始 table。

这些均为 configured possible behavior，strength 为 **WEAK**。未发现 embedded actual runtime record；actual historical request、retrieval timestamp、query、endpoint、installed acquisition-time version 和原始返回值仍不能由代码确认。Implementation evidence 不改变 recovery classification。

## 11. Candidate evidence table

完整 `NVDA_legacy_recovery_candidate_evidence.csv` 为 **194 行**，对应 **86 个不同 payload SHA-256**；同一内容可能在多个 source/tree/path 中出现，不把这些重复行算作独立新证据。以下是代表行，完整 SHA/commit/provenance 在 CSV 中：

| evidence_id | source | original_path | git/object SHA | category | strength | date hit | value hit | new_information |
|---|---|---|---|---|---|---|---|---|
| d97d384e5d15ad8593c7 | repository_local | data/raw/NVDA.csv | filesystem; SHA256见manifest | NORMALIZED_DATA | WEAK | True | True | 无新的因果信息 |
| 0c3cda3cc58644e3ba05 | repository_local | data/processed/NVDA.csv | filesystem; SHA256见manifest | DOWNSTREAM_PROCESSED_DATA | WEAK | True | True | 无新的因果信息 |
| ad31fa12ff6480068603 | reachable_git | data/research/NVDA_research.csv | e56cbfb343c4 | DOWNSTREAM_PROCESSED_DATA | WEAK | True | True | 无新的因果信息 |
| ff1bc87d91e532d7e242 | reachable_git | data/research/NVDA_features.csv | 73e9a07cfaa5 | DOWNSTREAM_PROCESSED_DATA | WEAK | True | True | 无新的因果信息 |
| 7f1ef68dea9555db6c29 | reachable_git | data/research/NVDA_labeled.csv | e0d6ff77c2ee | DOWNSTREAM_PROCESSED_DATA | WEAK | True | True | 无新的因果信息 |
| e4de90104c8e38161914 | repository_local | src/nasdaq_research/source_reconciliation.py | filesystem; SHA256见manifest | IMPLEMENTATION_EVIDENCE | WEAK | True | True | 无新的因果信息 |
| 18277b560f54e043d80b | repository_local | tests/test_source_reconciliation.py | filesystem; SHA256见manifest | TEST_FIXTURE | WEAK | True | True | 无新的因果信息 |

所有候选 representation 分类：

| Category | Rows |
|---|---:|
| RAW_VENDOR_EVIDENCE | 0 |
| RAW_LIBRARY_EVIDENCE | 0 |
| RUNTIME_ACQUISITION_METADATA | 0 |
| NORMALIZED_DATA | 5 |
| DOWNSTREAM_PROCESSED_DATA | 40 |
| IMPLEMENTATION_EVIDENCE | 38 |
| REPORT_OR_DOCUMENTATION | 27 |
| TEST_FIXTURE | 22 |
| UNKNOWN | 62 |

UNKNOWN 保留给无法认证 representation 的弱信息/占位或其它项目格式，没有升级为原始证据；其中 `121269300` 的 UNKNOWN 命中为 0。原始输入/runtime 的 qualifying new evidence 总数为 **0**。

mtime/ctime 如有记录，一律标为 `WEAK_AUXILIARY_ONLY`。它们不证明获取时间或 incomplete session。

## 12. Recovered raw/runtime evidence

```text
legacy raw Yahoo response recovered = false
legacy raw yfinance data recovered = false
legacy reliable acquisition timestamp recovered = false
legacy acquisition manifest recovered = false
```

`NVDA_legacy_recovery_recovered_evidence_manifest.csv` 为 header-only。没有生成伪 raw，没有恢复到旧路径，也没有复制大量 processed/无关 blobs。

## 13. New causal information

**没有恢复 Stage 13.1 当时不存在、足以 materially improve first-divergence identifiability 的新原始输入或可靠 runtime provenance。**

本次新增的是保存范围搜索与缺失情况的审计记录，而非 acquisition 当时 vendor value 或 request time。历史 normalized 121269300 的重复副本、旧代码、test fixture 和 Git/ref 时间不构成新的因果信息。

## 14. Completion classification

```text
completion_classification = LEGACY_EVIDENCE_NOT_RECOVERABLE
qualifying_new_evidence_count = 0
```

选择依据为七个预注册步骤全部完成且没有 qualifying STRONG/MODERATE new legacy raw/runtime evidence。没有重新推测 vendor drift、incomplete session 或 processing bug。

## 15. If recovered

本次未触发 recovered 分支，因此没有可供独立 closure stage 使用的新增原始证据。即使未来找到证据，也需由新的独立任务评估，不能由本 Audit 回写 Stage 13.1 root cause 或批准 Stage 13 dataset。本次没有设计该任务。

## 16. If not recoverable

> 在预注册搜索范围内，没有恢复出能够识别 legacy first causal divergence 的原始 vendor/library input 或可靠 runtime provenance。

因此，legacy root cause is historically non-identifiable **under the currently preserved evidence set**。这不是“原则上不可能”，而是“无法从本次搜索过的保存证据恢复”。

不建议继续对同一份 preserved evidence 重复 forensic investigation。建议停止追查这次历史 root cause，把未来工作方向转向 prospective canonical data-authority policy；该建议不批准当前 expanded dataset，也未自动设计或执行新 stage。

## 17. Stage 13 / 13.1 status

```text
Stage 13 = INCOMPLETE_OR_BLOCKED
Stage 13 original unexpected overlap violations = 5
Stage 13.1 = UNRESOLVED_BLOCKER
Stage 13.1 root cause = UNRESOLVED_SOURCE_DISCREPANCY
expanded_dataset_approved_for_downstream_research = false
stage14_modeling_eligible = false
```

以上原 artifacts/summary/report/policy 全部保持原字节。本 Audit 没有修改 root-cause classification 或 canonical policy。

## 18. Final Test state

```text
final_test_locked = true
final_test_predictions_generated = false
final_test_metrics_computed = false
final_training_pool_used = false
final_test_performance_read = false
```

Final Test 原 50 行、2026-07-15 至 2026-09-23 的锁定范围未改变。没有生成研究预测、计算研究 metrics 或使用 final training pool。

## 19. Tests

| Tests | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Stage 1–12 | 437 | 0 | 0 |
| Stage 13 | 39 | 0 | 0 |
| Stage 13.1 | 43 | 0 | 0 |
| Historical subtotal | **519** | **0** | **0** |
| New recovery audit | **39** | **0** | **0** |
| Full suite | **558** | **0** | **0** |

Full suite 实际日志：`Ran 558 tests in 226.109s / OK`。最后完善 audit classification、historical defaults 呈现和 ignored legacy preservation 核对后，39 项 targeted tests 再次通过：`Ran 39 tests in 0.002s / OK`。已有 tests 未修改或弱化。

New tests 覆盖 raw/processed 区分、unmapped blobs、代码/fixture/report 不算恢复、mtime/quote timestamp 不算 runtime、timestamp timezone guard、actual request representation、recovery 两类映射与 materiality/provenance gate、projects scope/depth、Git ls-tree/fsck/reflog parsing、post-registration object exclusion、Git mutation command rejection 和固定搜索顺序。

```bash
PYTHONPATH=src GIT_OPTIONAL_LOCKS=0 .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src GIT_OPTIONAL_LOCKS=0 .venv/bin/python -m unittest discover -s tests -p test_legacy_evidence_recovery.py -v
```

日志与 SHA 留在 execution verification 中；历史 tests 使用原 fixtures，没有对 expanded research dataset 开展新模型研究。

## 20. Reproducibility

最后版本连续两次正式执行，每次均按同一七步顺序进行只读搜索：

```bash
PYTHONPATH=src GIT_OPTIONAL_LOCKS=0 .venv/bin/python -m nasdaq_research.legacy_evidence_recovery
```

**19 个正式 deterministic CSV/JSON 全部逐字节一致；3 个文本 Git inventories 也一致，合计 22/22，mismatches=0。** 两次均返回 LEGACY_EVIDENCE_NOT_RECOVERABLE。

正式数据按注册时冻结的 checkpoint object scope 报告。Audit-era loose objects 和实时 fsck 的变化不是 legacy 输入；其实际全量 stdout/returncode/新增对象清单放入 non-authoritative execution log，不伪造 byte-identical 整个 Git database，也不改变 recovery criteria。没有把当前执行时间写入正式 outputs。

正式文件共 10 CSV + 9 JSON：protocol、initial state、protected research/Git manifests、reachable/deleted/reflog/unreachable/project inventories、candidate/recovered manifests、commit/path history、implementation、exclusions、command inventory、search steps、summary、validation。另有 reachable/deleted/fsck 三份文本 inventory。`execution_logs/` 中的运行后 verification 是明确排除的非-authoritative 执行记录，不冒充第 20 个两次复现 artifact。

正式目录 tmp/partial files 为 0。`/tmp` 中的完整实时 fsck、tests 和 run1/run2 hash files 是执行证据，不属于 authoritative deterministic output。

## 21. Git safety

```text
git reset run = false
git restore run = false
git clean run = false
git gc run = false
git prune run = false
reflog expire run = false
checkout modifying working tree = false
git add/commit/push/tag run = false
objects intentionally deleted = false
```

HEAD、branch、origin/main 和 remote URL 未变；原 refs/reflog/object/index 等 501 个 Git files 无改动或删除。214 个独立研究文件无 mutation violations。必须同时保留第 7 节的真实限制：发现 additional loose objects，整个 Git directory 不是 byte-identical，fsck 有 canonical empty-tree diagnostic。没有进行修复或清理。

## 22. Git status

最终执行 `git status`、`git diff --check`、`git log -1 --oneline`。Tracked working tree 无修改，diff check 通过；HEAD 仍为 `d6c4b2c`，main 与 origin/main 未移动。

允许新增的未跟踪内容只有：

```text
data/research/historical_expansion/legacy_evidence_recovery/
docs/legacy_evidence_recovery_report.md
src/nasdaq_research/legacy_evidence_recovery.py
tests/test_legacy_evidence_recovery.py
```

旧 Stage 1–13.1 文件未改变。完整最终 status 和保护校验保存于 `execution_logs/NVDA_legacy_recovery_execution_verification.json`。没有 commit。

## 23. Final conclusion

**Legacy 2026-09-30 NVDA Volume 的原始 Yahoo/yfinance evidence 或可靠 acquisition provenance 未被恢复。** 在当前保存的 Git 历史、Git object database、project-local files 和有限 projects 搜索范围内，没有找到足以识别 legacy first causal divergence 的新证据。

**当前没有足够新证据值得开启独立 root-cause closure stage。** 建议停止对同一保存证据集重复追查这次历史 root cause，将未来方向转向 prospective canonical data-authority policy。该方向尚未设计或执行；当前 expanded dataset 的 approval 和 Stage 14 eligibility 仍为 false。

Stage 13 / Stage 13.1 原状态未修改，Final Test 保持锁定，没有联网，没有进行新的模型研究或 data remediation，没有自动设计 Stage 13.2/Stage 14，没有 commit。本 Audit 到此停止。
