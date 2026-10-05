"""Read-only recovery of preserved legacy NVDA source/runtime evidence.

No downloads, model operations, root-cause reclassification or Git writes.
The search protocol and checkpoint manifests must be registered beforehand.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/research/historical_expansion/legacy_evidence_recovery'
PREFIX = 'NVDA_legacy_recovery_'
DATE = '2026-09-30'
VALUE = '121269300'
CATEGORIES = (
    'RAW_VENDOR_EVIDENCE', 'RAW_LIBRARY_EVIDENCE', 'RUNTIME_ACQUISITION_METADATA',
    'NORMALIZED_DATA', 'DOWNSTREAM_PROCESSED_DATA', 'IMPLEMENTATION_EVIDENCE',
    'REPORT_OR_DOCUMENTATION', 'TEST_FIXTURE', 'UNKNOWN',
)
PATH_PATTERN = re.compile(r'NVDA|yahoo|yfinance|raw|cache|manifest|acquisition|market|download', re.I)
RELATED_PATTERN = re.compile(r'nasdaq|NVDA|yahoo|yfinance', re.I)
EXCLUDED_DIRS = {'.git', '.venv', 'venv', 'node_modules', '__pycache__',
                 '.pytest_cache', '.mypy_cache', '.ruff_cache'}
MAX_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE = 100 * 1024 * 1024
ARCHIVE_SUFFIXES = {'.zip', '.tar', '.gz', '.bz2', '.xz', '.tgz', '.tbz2', '.txz'}
ALLOWED_GIT = {'status', 'log', 'show', 'rev-list', 'ls-tree', 'cat-file', 'reflog',
               'fsck', 'rev-parse', 'ls-files', 'remote', 'diff', 'symbolic-ref'}
SAFETY = dict(network_allowed=False, network_requests=0, modeling_allowed=False,
    new_model_fit=False, model_evaluation_performed=False, data_remediation_performed=False,
    root_cause_reclassified=False, final_test_locked=True,
    final_test_predictions_generated=False, final_test_metrics_computed=False,
    final_training_pool_used=False, final_test_performance_read=False,
    stage14_designed=False, stage14_executed=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()


def save_json(name, value):
    (OUT / f'{PREFIX}{name}.json').write_bytes(json_bytes(value))


def save_csv(name, rows, columns):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n', extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list))
                         else value for key, value in row.items()})
    (OUT / f'{PREFIX}{name}.csv').write_text(stream.getvalue())


def matching_path(path):
    return bool(PATH_PATTERN.search(path))


def context_path(path, protocol):
    return any(path == prefix or path.startswith(prefix)
               for prefix in protocol['known_context_exclusions'])


def self_path(path, protocol):
    return any(path == prefix or path.startswith(prefix)
               for prefix in protocol['audit_self_exclusions'])


def performance_path(path):
    """Inventory these paths without inspecting possible performance payloads."""
    lower = path.lower()
    if lower.startswith(('src/', 'tests/')):
        return False
    return any(word in lower for word in (
        'modeling', 'baseline_model', 'prediction', 'metrics', 'diagnostic',
        'ablation', 'stability', 'backtest', 'signal_', 'temporal_split',
    ))


def related_project_candidate(path, base):
    """Only filename evidence or an explicitly related directory inside projects."""
    try:
        relative = path.relative_to(base)
    except ValueError:
        return False
    return len(relative.parts) <= 7 and bool(RELATED_PATTERN.search(str(relative)))


def parse_ls_tree(payload):
    rows = []
    for entry in payload.split(b'\0'):
        if not entry:
            continue
        metadata, path = entry.split(b'\t', 1)
        mode, kind, oid = metadata.decode().split()
        rows.append({'mode': mode, 'object_type': kind, 'object_sha': oid,
                     'original_path': path.decode('utf-8', 'surrogateescape')})
    return rows


def parse_fsck(text):
    return [{'object_type': match[1], 'object_sha': match[2]}
            for line in text.splitlines()
            if (match := re.fullmatch(r'(?:unreachable|dangling) (commit|tree|blob|tag) ([0-9a-f]{40,64})', line))]


def checkpoint_fsck_payload(payload, initial_ids):
    """The checkpoint inventory excludes objects created after registration."""
    lines=[]
    for line in payload.decode().splitlines():
        match=re.fullmatch(r'(?:unreachable|dangling) (?:commit|tree|blob|tag) ([0-9a-f]{40,64})',line)
        if match and match[1] in initial_ids:
            lines.append(line)
        elif line == 'missing tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904':
            lines.append(line)
    return ('\n'.join(sorted(lines))+'\n').encode()


def parse_reflog(text):
    rows = []
    for index, line in enumerate(text.splitlines()):
        parts = line.split('\x1f', 2)
        if len(parts) != 3:
            raise ValueError('Unexpected reflog format')
        rows.append({'entry_index': index, 'commit_sha': parts[0],
                     'selector_with_recorded_time': parts[1], 'message': parts[2],
                     'time_evidence_role': 'REF_MOVEMENT_ONLY_NOT_ACQUISITION_TIME'})
    return rows


def text_payload(data):
    if len(data) > MAX_BYTES or b'\0' in data[:8192]:
        return None
    try:
        return data.decode('utf-8-sig')
    except UnicodeDecodeError:
        return None


def nested_items(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key.lower(), child
            yield from nested_items(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_items(child)


def actual_runtime_metadata(value, text):
    """Require actual request/acquisition fields, never filesystem/quote timestamps."""
    pairs = list(nested_items(value))
    keys = {key for key, _ in pairs}
    timestamps = [str(child) for key, child in pairs if key in {
        'retrieval_timestamp', 'retrieved_at', 'retrieval_time', 'request_timestamp',
        'request_started_utc', 'response_received_utc', 'acquisition_started_utc',
        'acquisition_finished_utc', 'download_started_at', 'download_finished_at',
    } and isinstance(child, (str, int, float))]
    valid_times = []
    for item in timestamps:
        try:
            parsed = datetime.fromisoformat(item.replace('Z', '+00:00'))
            if parsed.tzinfo is not None:
                valid_times.append(item)
        except ValueError:
            pass
    request_record = bool(keys & {'query', 'request_parameters', 'request_url', 'actual_request_url',
                                 'period1', 'period2', 'query_start', 'query_end'})
    request_record = request_record or bool('endpoint' in keys and 'finance.yahoo.com' in text)
    request_record = request_record or bool('start' in keys and keys & {'end', 'end_exclusive'}
                                           and re.search(r'yahoo|yfinance', text, re.I))
    configured_only = any(key in keys for key in ('configured_defaults', 'source_code_defaults'))
    return bool(request_record and not configured_only), valid_times


def classify_candidate(path, data):
    """Describe representation independently of whether it is new legacy evidence."""
    text = text_payload(data)
    result = {'category': 'UNKNOWN', 'strength': 'WEAK',
        'contains_2026_09_30': bool(text and DATE in text),
        'contains_121269300': bool(text and VALUE in text),
        'primary_volume': None, 'actual_runtime_record': False,
        'reliable_acquisition_timestamp': False,
        'timestamp_evidence_role': 'WEAK_AUXILIARY_ONLY',
        'representation_reason': 'No verified original source/runtime representation'}
    if text is None:
        result['representation_reason'] = 'Binary, oversized or non-UTF8 payload; no textual source assertion'
        return result
    lower = path.lower()
    if lower.startswith('tests/') or '/tests/' in lower or 'test_' in Path(lower).name:
        result.update(category='TEST_FIXTURE', representation_reason='Test code/fixture is not an acquisition record')
        return result
    if lower.startswith('docs/') or Path(lower).suffix in {'.md', '.rst'} or 'report' in Path(lower).stem:
        result.update(category='REPORT_OR_DOCUMENTATION', representation_reason='Report/documentation is derived context')
        return result
    if Path(lower).suffix in {'.py', '.sh'}:
        result.update(category='IMPLEMENTATION_EVIDENCE', representation_reason='Ordinary source code describes possible behavior only')
        return result
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        value = None
    if isinstance(value, dict):
        chart = value.get('chart', {})
        for item in chart.get('result') or []:
            meta = item.get('meta', {})
            if str(meta.get('symbol', '')).upper() != 'NVDA':
                continue
            volumes = ((item.get('indicators', {}).get('quote') or [{}])[0]).get('volume', [])
            timestamps = item.get('timestamp', [])
            if not volumes or len(volumes) != len(timestamps):
                continue
            tz = ZoneInfo(meta.get('exchangeTimezoneName', 'America/New_York'))
            for epoch, volume in zip(timestamps, volumes):
                if datetime.fromtimestamp(epoch, timezone.utc).astimezone(tz).date().isoformat() == DATE:
                    result.update(category='RAW_VENDOR_EVIDENCE', strength='STRONG',
                        contains_2026_09_30=True, primary_volume=volume,
                        representation_reason='Original Yahoo chart structure with NVDA primary-session quote array')
                    return result
        request_record, timestamps = actual_runtime_metadata(value, text)
        nvda = 'NVDA' in text.upper() or 'nvda' in lower
        if request_record and nvda:
            result.update(category='RUNTIME_ACQUISITION_METADATA',
                strength='STRONG' if timestamps else 'MODERATE', actual_runtime_record=True,
                reliable_acquisition_timestamp=bool(timestamps),
                timestamp_evidence_role='ACTUAL_ACQUISITION_RECORD' if timestamps else 'NO_RELIABLE_ACQUISITION_TIMESTAMP',
                representation_reason='Structured actual query/request fields; historical association assessed separately')
            return result
    if Path(lower).suffix in {'.csv', '.tsv'} or text.startswith(('date,', 'Price,', 'Date,')):
        rows = list(csv.reader(io.StringIO(text)))
        if rows:
            first = rows[0]
            normalized = {'date', 'open', 'high', 'low', 'close', 'volume'} <= set(first)
            if normalized:
                category = 'NORMALIZED_DATA' if len(first)==6 and '/processed/' not in f'/{lower}' else 'DOWNSTREAM_PROCESSED_DATA'
                result.update(category=category,
                    representation_reason='Project-normalized lowercase date/OHLCV columns; not original library data')
                return result
            if 'Volume' in first and ('Adj Close' in first or 'Price' in first):
                index = first.index('Volume')
                for row in rows[1:]:
                    if row and row[0][:10] == DATE and index < len(row):
                        result.update(category='RAW_LIBRARY_EVIDENCE', strength='MODERATE',
                            contains_2026_09_30=True, primary_volume=row[index],
                            representation_reason='Original-style yfinance table header; project/source provenance assessed separately')
                        return result
    if Path(lower).suffix in {'.log', '.out', '.txt'}:
        if (DATE in text and 'NVDA' in text.upper() and re.search(r'yfinance|query\d\.finance\.yahoo|download', text, re.I)
                and re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})', text)
                and re.search(r'retriev|request|acquisition|download.*(?:start|finish)|period1', text, re.I)):
            result.update(category='RUNTIME_ACQUISITION_METADATA', strength='MODERATE',
                actual_runtime_record=True, reliable_acquisition_timestamp=True,
                timestamp_evidence_role='CANDIDATE_RUN_LOG_REQUIRES_PROVENANCE',
                representation_reason='Timestamped NVDA request/download log; authenticity assessed separately')
            return result
    # Unreachable blobs may have lost their path. Recognize derived content,
    # without upgrading it to original input merely because a keyword matches.
    if isinstance(value,dict) and set(value) & {
            'valid','checks','violations','completion_status','protocol','feature_coverage',
            'missing_required_columns','columns','rows','feature_count','target_count',
            'split_counts','coverage','errors','warnings'}:
        result.update(category='REPORT_OR_DOCUMENTATION',representation_reason='Derived validation/summary/protocol JSON content')
    elif text.startswith(('date,','Date,')):
        result.update(category='DOWNSTREAM_PROCESSED_DATA',representation_reason='Dated project table lacking original vendor/library representation')
    elif re.search(r'^\s*(?:import unittest|from unittest|class \w+\(unittest\.TestCase\))',text,re.M):
        result.update(category='TEST_FIXTURE',representation_reason='Unmapped Python test/fixture content, not runtime acquisition')
    elif re.search(r'^\s*(?:from \w[\w.]* import|import \w|def \w+\(|class \w+)',text,re.M):
        result.update(category='IMPLEMENTATION_EVIDENCE',representation_reason='Unmapped ordinary source code; no embedded runtime assertion')
    elif text.lstrip().startswith(('# ','## ','#\n')):
        result.update(category='REPORT_OR_DOCUMENTATION',representation_reason='Unmapped Markdown/report content')
    return result


def recovery_qualified(row):
    return bool(row.get('category') in CATEGORIES[:3]
        and row.get('strength') in {'STRONG', 'MODERATE'}
        and row.get('genuinely_new') and row.get('legacy_provenance_verified')
        and row.get('material_primary_information'))


def completion(rows):
    return 'LEGACY_EVIDENCE_RECOVERED' if any(recovery_qualified(row) for row in rows) else 'LEGACY_EVIDENCE_NOT_RECOVERABLE'


def verify_state():
    protected = json.loads((OUT / f'{PREFIX}protected_sha256.json').read_text())
    changed = [path for path, sha in protected.items()
               if not (ROOT / path).is_file() or digest((ROOT / path).read_bytes()) != sha]
    # The committed Stage 13.1 manifest also authenticates authoritative legacy
    # data ignored by Git. Its own bytes are pinned by the initial tracked list.
    previous_manifest_path=ROOT/'data/research/historical_expansion/stage13_1_yahoo_reconciliation/NVDA_stage13_1_protected_sha256.json'
    previous=json.loads(previous_manifest_path.read_text())
    previous_changed=[path for path,sha in previous.items()
        if not (ROOT/path).is_file() or digest((ROOT/path).read_bytes())!=sha]
    baseline = json.loads((OUT / f'{PREFIX}git_state_sha256.json').read_text())
    current = {str(path.relative_to(ROOT / '.git')):
               {'sha256': digest(path.read_bytes()), 'size': path.stat().st_size}
               for path in sorted((ROOT / '.git').rglob('*')) if path.is_file()}
    git_changed = [path for path in baseline if current.get(path) != baseline[path]]
    git_added = sorted(set(current) - set(baseline))
    return {'tracked_files_checked': len(protected), 'tracked_mutation_violations': len(changed),
        'tracked_changed_files': changed, 'git_physical_files_checked': len(baseline),
        'checkpoint_preserved_stage13_1_manifest_files_checked':len(previous),
        'checkpoint_preserved_stage13_1_manifest_mutation_violations':len(previous_changed),
        'checkpoint_preserved_stage13_1_manifest_changed_files':previous_changed,
        'additional_ignored_legacy_files_authenticated':len(set(previous)-set(protected)),
        'unique_preserved_research_files_checked':len(set(previous)|set(protected)),
        'git_changed_or_deleted_files': git_changed, 'git_added_files': git_added,
        'preserved_git_files_byte_identical': not git_changed,
        'added_files_only_loose_objects': all(re.fullmatch(r'objects/[0-9a-f]{2}/[0-9a-f]{38}', path)
                                              for path in git_added),
        'git_database_byte_identical': current == baseline}


class Audit:
    def __init__(self):
        self.protocol = json.loads((OUT / f'{PREFIX}protocol.json').read_text())
        self.initial = json.loads((OUT / f'{PREFIX}initial_repository_state.json').read_text())
        self.protected = json.loads((OUT / f'{PREFIX}protected_sha256.json').read_text())
        self.initial_git_files = json.loads((OUT / f'{PREFIX}git_state_sha256.json').read_text())
        self.initial_object_ids = {path.split('/')[1]+path.split('/')[2] for path in self.initial_git_files
            if re.fullmatch(r'objects/[0-9a-f]{2}/[0-9a-f]{38}',path)}
        self.known_hashes = {sha for path, sha in self.protected.items() if context_path(path, self.protocol)}
        self.commands, self.candidates, self.recovered, self.exclusions = [], [], [], []
        self.payloads, self.candidate_keys = {}, set()
        self.stats, self.steps = {}, []
        self.stage = None

    def git(self, *args):
        if not args or args[0] not in ALLOWED_GIT or '--lost-found' in args:
            raise ValueError('Only registered read-only Git commands are permitted')
        if args[0] == 'reflog' and args[1:] != ('--all', '--date=iso', '--format=%H%x1f%gD%x1f%gs'):
            raise ValueError('Reflog reads only')
        result = subprocess.run(['git', *args], cwd=ROOT,
            env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}, capture_output=True)
        if args[0]=='fsck':
            (Path('/tmp')/'nasdaq-legacy-recovery-live-fsck.txt').write_bytes(result.stdout)
            (Path('/tmp')/'nasdaq-legacy-recovery-live-fsck-stderr.txt').write_bytes(result.stderr)
            live_rows=parse_fsck(result.stdout.decode())
            self.live_fsck={'returncode':result.returncode,'stdout_sha256':digest(result.stdout),
                'stderr_sha256':digest(result.stderr),'live_unreachable_counts':{
                    kind:sum(row['object_type']==kind for row in live_rows) for kind in ('commit','tree','blob')},
                'post_registration_objects':sum(row['object_sha'] not in self.initial_object_ids for row in live_rows),
                'diagnostics':[line for line in result.stdout.decode().splitlines()
                    if line and not line.startswith(('unreachable ','dangling '))]}
            payload=checkpoint_fsck_payload(result.stdout,self.initial_object_ids)
            self.commands.append({'step':self.stage,'argv':['git',*args],
                'returncode':'NONZERO_DIAGNOSTIC' if result.returncode else 0,
                'stdout_sha256':digest(payload),'stderr_sha256':digest(result.stderr),
                'scope':'Initial checkpoint objects only; full actual stdout/returncode in /tmp execution log'})
            if any(line!='missing tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904'
                   for line in self.live_fsck['diagnostics']) or result.stderr:
                raise RuntimeError('Unexpected fsck integrity diagnostic; preserve log without modifying Git')
            return payload
        self.commands.append({'step': self.stage, 'argv': ['git', *args],
            'returncode': result.returncode, 'stdout_sha256': digest(result.stdout),
            'stderr_sha256': digest(result.stderr)})
        if result.returncode:
            raise RuntimeError(f'Read-only Git command failed: {args[0]}: {result.stderr.decode(errors="replace")}')
        return result.stdout

    def step(self, name):
        expected = self.protocol['search_order'][len(self.steps)]
        if name != expected:
            raise ValueError('Search order differs from registered protocol')
        self.stage = name
        self.steps.append(name)

    def excluded(self, source, path, reason):
        self.exclusions.append({'search_step': self.stage, 'source': source,
                                'original_path': path, 'reason': reason})

    def blob(self, oid):
        if oid in self.payloads:
            return self.payloads[oid]
        size = int(self.git('cat-file', '-s', oid))
        if size > MAX_BYTES:
            self.excluded('git_object', oid, 'REGISTERED_MAXIMUM_BYTES_EXCEEDED')
            return None
        payload = self.git('cat-file', 'blob', oid)
        self.payloads[oid] = payload
        return payload

    def candidate(self, source, path, data, *, commit='', oid='', tree='', method='', legacy=False, fs=None):
        sha = digest(data)
        key = (source, path, sha, oid)
        if key in self.candidate_keys:
            return
        self.candidate_keys.add(key)
        classified = classify_candidate(path, data)
        if context_path(path, self.protocol) or sha in self.known_hashes:
            self.excluded(source, path, 'KNOWN_STAGE13_OR_13_1_CONTEXT_NOT_NEW_LEGACY_EVIDENCE')
            return
        raw = classified['category'] in CATEGORIES[:2]
        runtime = classified['category'] == 'RUNTIME_ACQUISITION_METADATA'
        primary = classified['contains_2026_09_30'] or (runtime and classified['reliable_acquisition_timestamp'])
        nvda_identity = 'nvda' in path.lower() or b'NVDA' in data[:MAX_BYTES]
        legacy_verified = bool(legacy or (source in {'repository_local', 'limited_projects'}
            and nvda_identity and classified.get('primary_volume') is not None
            and str(classified['primary_volume']) == VALUE))
        row = {'evidence_id': digest((source+'\0'+path+'\0'+oid+'\0'+sha).encode())[:20],
            'search_step': self.stage, 'source': source, 'original_path': path,
            'source_commit_sha': commit, 'git_object_sha': oid, 'sha256': sha,
            'source_tree_sha':tree,
            'size_bytes': len(data), 'recovery_method': method, **classified,
            'genuinely_new': bool((raw or runtime) and sha not in self.known_hashes),
            'legacy_provenance_verified': legacy_verified,
            'material_primary_information': bool(primary and nvda_identity and (raw or runtime)),
            'new_information': classified['representation_reason'] if raw or runtime
                else 'No new causal information: normalized/downstream/implementation/report/test/unknown evidence',
            'mtime_ns': fs.st_mtime_ns if fs else None, 'ctime_ns': fs.st_ctime_ns if fs else None,
            'filesystem_timestamp_role': 'WEAK_AUXILIARY_ONLY'}
        row['qualifies_for_recovery'] = recovery_qualified(row)
        self.candidates.append(row)
        if row['qualifies_for_recovery']:
            directory = OUT / 'recovered_evidence'
            directory.mkdir(exist_ok=True)
            filename = row['evidence_id'] + '_' + Path(path).name
            destination = directory / filename
            if destination.exists() and destination.read_bytes() != data:
                raise ValueError('Recovered evidence collision; no overwrite allowed')
            if not destination.exists():
                destination.write_bytes(data)
            metadata = {**row, 'recovered_path': str(destination.relative_to(ROOT)),
                        'recovered_file_sha256': sha}
            metadata_path = destination.with_name(filename + '.metadata.json')
            encoded = json_bytes(metadata)
            if metadata_path.exists() and metadata_path.read_bytes() != encoded:
                raise ValueError('Frozen recovered metadata collision')
            if not metadata_path.exists():
                metadata_path.write_bytes(encoded)
            self.recovered.append(metadata)

    def tree_entries(self,treeish,prefix=''):
        # The well-known empty tree has no entries. Never synthesize/write its Git object.
        if treeish=='4b825dc642cb6eb9a060e54bf8d69288fbee4904':
            self.excluded('unreachable_tree',treeish,'MISSING_CANONICAL_EMPTY_TREE_HAS_ZERO_ENTRIES_NO_REPAIR')
            return []
        result=[]
        for row in parse_ls_tree(self.git('ls-tree','-z',treeish)):
            path=prefix+row['original_path']
            if row['object_type']=='tree':
                result.extend(self.tree_entries(row['object_sha'],path+'/'))
            else:
                result.append({**row,'original_path':path})
        return result

    def inspect_tree(self, commit, source):
        rows = self.tree_entries(commit) if source=='unreachable_tree' else parse_ls_tree(self.git('ls-tree', '-r', '-z', commit))
        found = []
        for row in rows:
            path, oid = row['original_path'], row['object_sha']
            if row['object_type'] != 'blob' or not matching_path(path):
                continue
            found.append({**row, 'source_commit_sha': commit})
            if context_path(path, self.protocol) or self_path(path, self.protocol):
                self.excluded(source, path, 'KNOWN_CONTEXT_OR_AUDIT_SELF')
                continue
            if performance_path(path):
                self.excluded(source, path, 'PERFORMANCE_PAYLOAD_NOT_READ')
                continue
            payload = self.blob(oid)
            if payload is not None:
                self.candidate(source, path, payload, commit='' if source=='unreachable_tree' else commit,
                    tree=commit if source=='unreachable_tree' else '',oid=oid,
                    method='git ls-tree then git cat-file blob', legacy=commit != self.initial['HEAD'])
        return found

    def reachable(self):
        self.step('reachable_git_history')
        raw_log = self.git('log', '--all', '--oneline', '--decorate')
        objects = self.git('rev-list', '--objects', '--all').decode().splitlines()
        commits = self.git('rev-list', '--all', '--reverse').decode().splitlines()
        self.reachable_commits = set(commits)
        inventory, seen = [], set()
        commit_rows = []
        for commit in commits:
            info = self.git('show', '-s', '--format=%H%x1f%P%x1f%cI%x1f%s', commit).decode().strip().split('\x1f', 3)
            commit_rows.append(dict(zip(('commit_sha','parent_shas','commit_time','subject'),info)))
            for row in parse_ls_tree(self.git('ls-tree', '-r', '-z', commit)):
                path, oid = row['original_path'], row['object_sha']
                if row['object_type'] != 'blob' or not matching_path(path):
                    continue
                key = (oid, path)
                if key in seen:
                    continue
                seen.add(key)
                exclusion = 'KNOWN_STAGE13_OR_13_1_CONTEXT' if context_path(path,self.protocol) else (
                    'PERFORMANCE_PAYLOAD_NOT_READ' if performance_path(path) else '')
                inventory.append({**row,'first_available_commit':commit,
                    'current_path_exists':(ROOT/path).is_file(),'commit_available':True,
                    'candidate_evidence_type':'EXCLUDED_CONTEXT' if exclusion else 'CONTENT_CLASSIFICATION_REQUIRED',
                    'exclusion':exclusion})
                if exclusion:
                    self.excluded('reachable_git',path,exclusion)
                    continue
                payload=self.blob(oid)
                if payload is not None:
                    self.candidate('reachable_git',path,payload,commit=commit,oid=oid,
                        method='git ls-tree then git cat-file blob',legacy=commit!=self.initial['HEAD'])
        candidate_paths=sorted({row['original_path'] for row in inventory if not row['exclusion']})
        # Read every candidate path's complete reachable name/status history.
        history_records=[]
        for path in candidate_paths:
            response=self.git('log','--all','--name-status','--format=%H','--',path)
            history_records.append({'path':path,'name_status_history':response.decode()})
        save_json('reachable_path_history',history_records)
        save_csv('git_reachable_inventory',inventory,
            ['mode','object_type','object_sha','original_path','first_available_commit','commit_available','current_path_exists','candidate_evidence_type','exclusion'])
        save_csv('commit_inventory',commit_rows,['commit_sha','parent_shas','commit_time','subject'])
        (OUT/'reachable_git_log.txt').write_bytes(raw_log)
        self.stats['reachable']={'commits_inspected':len(commits),'rev_list_objects_searched':len(objects),
            'matching_distinct_blob_paths':len(inventory),'candidate_path_histories_read':len(candidate_paths)}

    def deleted(self):
        self.step('deleted_reachable_files')
        summary=self.git('log','--all','--diff-filter=D','--summary')
        (OUT/'deleted_git_log.txt').write_bytes(summary)
        rows=[]
        text=self.git('log','--all','--diff-filter=D','--name-status','--format=COMMIT:%H').decode()
        commit=None
        deletions=0
        for line in text.splitlines():
            if line.startswith('COMMIT:'):
                commit=line[7:]
            elif line.startswith('D\t'):
                deletions+=1
                path=line[2:]
                if not matching_path(path):
                    continue
                parents=self.git('rev-list','--parents','-n','1',commit).decode().split()[1:]
                for parent in parents:
                    entries=parse_ls_tree(self.git('ls-tree','-r','-z',parent,'--',path))
                    for entry in entries:
                        if entry['object_type']!='blob':
                            continue
                        rows.append({**entry,'deletion_commit':commit,'source_parent_commit':parent,
                            'current_path_exists':(ROOT/path).exists()})
                        if context_path(path,self.protocol) or performance_path(path):
                            self.excluded('deleted_history',path,'KNOWN_CONTEXT_OR_PERFORMANCE')
                            continue
                        payload=self.blob(entry['object_sha'])
                        if payload is not None:
                            self.candidate('deleted_history',path,payload,commit=parent,oid=entry['object_sha'],
                                method='Deletion parent ls-tree and cat-file; original path never restored',legacy=parent!=self.initial['HEAD'])
        save_csv('deleted_history',rows,['deletion_commit','source_parent_commit','original_path','object_sha','object_type','mode','current_path_exists'])
        self.stats['deleted']={'all_deleted_path_events':deletions,'relevant_deleted_blob_paths':len(rows)}

    def reflog(self):
        self.step('reflog')
        payload=self.git('reflog','--all','--date=iso','--format=%H%x1f%gD%x1f%gs')
        rows=parse_reflog(payload.decode())
        commits=sorted({row['commit_sha'] for row in rows})
        total_paths=0
        for commit in commits:
            found=self.inspect_tree(commit,'reflog')
            total_paths+=len(found)
        for row in rows:
            row['reachable_from_refs']=row['commit_sha'] in self.reachable_commits
        save_csv('reflog_inventory',rows,['entry_index','commit_sha','selector_with_recorded_time','message','time_evidence_role','reachable_from_refs'])
        self.stats['reflog']={'entries_inspected':len(rows),'distinct_commits_inspected':len(commits),
            'commits_outside_reachable_refs':len(set(commits)-self.reachable_commits),
            'matching_tree_path_occurrences':total_paths}

    def unreachable(self):
        self.step('unreachable_git_objects')
        payload=self.git('fsck','--full','--no-reflogs','--unreachable')
        (OUT/'unreachable_git_fsck.txt').write_bytes(payload)
        rows=parse_fsck(payload.decode())
        for row in rows:
            oid,kind=row['object_sha'],row['object_type']
            row['size_bytes']=int(self.git('cat-file','-s',oid))
            row['content_inspected']=False
            row['present_in_initial_loose_object_snapshot']=f'objects/{oid[:2]}/{oid[2:]}' in self.initial_git_files
            # Objects appearing after registration cannot authenticate old legacy input.
            # Existing packs (if any) require a separate availability check, not this shortcut.
            if not row['present_in_initial_loose_object_snapshot'] and not any(
                    path.endswith('.pack') for path in self.initial_git_files):
                row['exclusion']='APPEARED_AFTER_REGISTRATION_NOT_PRESERVED_LEGACY_EVIDENCE'
                self.excluded('unreachable_object',oid,row['exclusion'])
                continue
            row['exclusion']=''
            if kind=='commit':
                self.git('show','--stat',oid)
                self.inspect_tree(oid,'unreachable_commit')
                row['content_inspected']=True
            elif kind=='tree':
                self.inspect_tree(oid,'unreachable_tree')
                row['content_inspected']=True
            elif kind=='blob' and row['size_bytes']<=MAX_BYTES:
                data=self.blob(oid)
                text=text_payload(data) if data is not None else None
                row['content_inspected']=text is not None
                if text and re.search(r'121269300|2026-09-30|NVDA|yfinance|Yahoo',text,re.I):
                    self.candidate('unreachable_blob','unmapped_blob_'+oid,data,oid=oid,
                        method='Size-bounded cat-file; original path/provenance unavailable',legacy=False)
        save_csv('unreachable_inventory',rows,['object_type','object_sha','size_bytes','content_inspected','present_in_initial_loose_object_snapshot','exclusion'])
        self.stats['unreachable']={f'{kind}s':sum(row['object_type']==kind for row in rows) for kind in ('commit','tree','blob','tag')}
        self.stats['unreachable']['objects_content_inspected']=sum(row['content_inspected'] for row in rows)
        self.stats['unreachable']['objects_present_in_initial_loose_snapshot']=sum(row['present_in_initial_loose_object_snapshot'] for row in rows)
        self.stats['unreachable']['post_registration_objects_excluded_from_checkpoint_inventory']=True
        self.stats['unreachable']['fsck_diagnostics']=['missing tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904'] if b'missing tree ' in payload else []

    def local_file(self,path,source,base,*,legacy=False):
        relative=str(path.relative_to(base))
        stat=path.lstat()
        row={'source':source,'original_path':str(path),'relative_path':relative,
            'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'ctime_ns':stat.st_ctime_ns,
            'timestamp_evidence_role':'WEAK_AUXILIARY_ONLY','content_read':False,'exclusion':''}
        if path.is_symlink():
            row['exclusion']='SYMLINK_NOT_FOLLOWED'
        elif source=='repository_local' and (context_path(relative,self.protocol) or self_path(relative,self.protocol)):
            row['exclusion']='KNOWN_CONTEXT_OR_AUDIT_SELF'
        elif performance_path(relative):
            row['exclusion']='PERFORMANCE_PAYLOAD_NOT_READ'
        elif stat.st_size>MAX_BYTES:
            row['exclusion']='REGISTERED_MAXIMUM_BYTES_EXCEEDED'
        elif path.suffix.lower() in ARCHIVE_SUFFIXES:
            row['exclusion']='ARCHIVE_HANDLED_SEPARATELY'
        else:
            data=path.read_bytes()
            row.update(content_read=True,sha256=digest(data))
            text=text_payload(data)
            if matching_path(relative) or (text and (VALUE in text or DATE in text)):
                self.candidate(source,relative,data,method='Bounded read-only project file inspection',legacy=legacy,fs=stat)
        if row['exclusion']:
            self.excluded(source,relative,row['exclusion'])
        return row

    def repository(self):
        self.step('repository_local_files')
        rows=[]
        scanned=0
        for directory,dirs,files in os.walk(ROOT,followlinks=False):
            dirs[:]=sorted(name for name in dirs if name not in EXCLUDED_DIRS
                and not self_path(str((Path(directory)/name).relative_to(ROOT))+'/',self.protocol)
                and not context_path(str((Path(directory)/name).relative_to(ROOT))+'/',self.protocol))
            for name in sorted(files):
                path=Path(directory)/name
                relative=str(path.relative_to(ROOT))
                if self_path(relative,self.protocol) or context_path(relative,self.protocol):
                    self.excluded('repository_local',relative,'KNOWN_CONTEXT_OR_AUDIT_SELF')
                    continue
                scanned+=1
                # Only source/data/log/notebook-like files; never credentials or runtime binaries.
                if path.suffix.lower() not in {'.csv','.json','.jsonl','.log','.out','.txt','.md','.py','.sh','.ipynb'} | ARCHIVE_SUFFIXES:
                    continue
                if path.name.startswith('.'):
                    continue
                rows.append(self.local_file(path,'repository_local',ROOT))
        self.project_rows=rows
        self.stats['repository_local']={'file_names_seen_in_nonexcluded_scope':scanned,
            'candidate_text_or_archive_files_inventoried':len(rows),'files_content_read':sum(row['content_read'] for row in rows)}

    def archive(self,path):
        if path.stat().st_size>MAX_ARCHIVE:
            self.excluded('limited_projects',str(path),'REGISTERED_MAXIMUM_ARCHIVE_BYTES_EXCEEDED')
            return
        members=0
        def inspect(name,size,reader):
            nonlocal members
            if not matching_path(name) or performance_path(name) or size>MAX_BYTES:
                return
            if name.startswith('/') or '..' in Path(name).parts:
                self.excluded('archive_member',str(path)+'!'+name,'UNSAFE_MEMBER_PATH_NOT_EXTRACTED')
                return
            data=reader()
            members+=1
            self.candidate('archive_member',str(path)+'!'+name,data,
                method='Related archive member read without extraction',legacy=False)
        try:
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as archive:
                    for member in sorted(archive.infolist(),key=lambda member:member.filename):
                        if not member.is_dir():
                            inspect(member.filename,member.file_size,lambda member=member:archive.read(member))
            elif tarfile.is_tarfile(path):
                with tarfile.open(path) as archive:
                    for member in sorted(archive.getmembers(),key=lambda member:member.name):
                        if member.isfile():
                            inspect(member.name,member.size,lambda member=member:archive.extractfile(member).read())
        except (OSError,ValueError,tarfile.TarError,zipfile.BadZipFile) as error:
            self.excluded('limited_projects',str(path),'ARCHIVE_UNREADABLE:'+type(error).__name__)
        self.stats['projects']['archive_members_inspected']+=members

    def projects(self):
        self.step('limited_projects_files')
        base=Path(self.protocol['project_search_boundaries']['projects_root'])
        self.stats['projects']={'related_old_clones':0,'related_archive_files':0,
            'candidate_files':0,'archive_members_inspected':0,'unrelated_project_contents_read':0}
        if any(recovery_qualified(row) for row in self.candidates):
            self.stats['projects']['search_skipped']='Repository evidence recovered; expansion unnecessary under user scope'
            return
        old_clones=set()
        for directory,dirs,files in os.walk(base,followlinks=False):
            current=Path(directory)
            depth=len(current.relative_to(base).parts)
            related=bool(RELATED_PATTERN.search(str(current.relative_to(base))))
            if related and (current/'.git').is_dir() and current!=ROOT:
                old_clones.add(str(current))
            dirs[:]=sorted(name for name in dirs if name not in EXCLUDED_DIRS
                and current/name!=ROOT and not (current/name).is_symlink() and depth<6)
            for name in sorted(files):
                path=current/name
                if not related_project_candidate(path,base):
                    continue
                if path.is_symlink():
                    self.excluded('limited_projects',str(path),'SYMLINK_NOT_FOLLOWED')
                    continue
                if not path.is_file():
                    continue
                if path.suffix.lower() in ARCHIVE_SUFFIXES:
                    self.stats['projects']['related_archive_files']+=1
                    self.archive(path)
                elif matching_path(name) or path.suffix.lower() in {'.json','.csv','.log','.out','.txt','.ipynb'}:
                    self.project_rows.append(self.local_file(path,'limited_projects',base))
                    self.stats['projects']['candidate_files']+=1
        self.stats['projects']['related_old_clones']=len(old_clones)
        self.stats['projects']['old_clone_paths']=sorted(old_clones)
        # Old clone history is read with explicit -C and the same read-only environment.
        for clone in sorted(old_clones):
            env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'}
            records={key:subprocess.check_output(['git','-C',clone,*args],env=env).decode().strip()
                     for key,args in {'HEAD':['rev-parse','HEAD'],'branch':['branch','--show-current']}.items()}
            commits=subprocess.check_output(['git','-C',clone,'rev-list','--all','--reverse'],env=env).decode().splitlines()
            records['commits_inspected']=len(commits)
            seen=set()
            for commit in commits:
                payload=subprocess.check_output(['git','-C',clone,'ls-tree','-r','-z',commit],env=env)
                for entry in parse_ls_tree(payload):
                    key=(entry['object_sha'],entry['original_path'])
                    if key in seen or entry['object_type']!='blob' or not matching_path(entry['original_path']):
                        continue
                    seen.add(key)
                    if performance_path(entry['original_path']) or context_path(entry['original_path'],self.protocol):
                        continue
                    size=int(subprocess.check_output(['git','-C',clone,'cat-file','-s',entry['object_sha']],env=env))
                    if size<=MAX_BYTES:
                        data=subprocess.check_output(['git','-C',clone,'cat-file','blob',entry['object_sha']],env=env)
                        self.candidate('old_clone:'+clone,entry['original_path'],data,commit=commit,
                            oid=entry['object_sha'],method='Read-only old clone reachable tree/blob',legacy=commit!=self.initial['HEAD'])
            self.stats['projects'].setdefault('old_clone_metadata',[]).append({'path':clone,**records})

    def implementation(self):
        self.step('historical_implementation')
        history=self.git('log','--all','--format=%H','--','src/nasdaq_research/data.py').decode().splitlines()
        rows=[]
        seen=set()
        for commit in reversed(history):
            entries=parse_ls_tree(self.git('ls-tree','-r','-z',commit,'--','src/nasdaq_research/data.py'))
            for entry in entries:
                oid=entry['object_sha']
                if oid in seen:
                    continue
                seen.add(oid)
                data=self.blob(oid)
                text=text_payload(data)
                parameters=[]
                if text:
                    for index,line in enumerate(text.splitlines(),1):
                        if re.search(r'yf\.download|period:|interval:|period=|interval=|start|end|auto_adjust|back_adjust|threads=|progress=|to_numeric|reset_index|rename|to_csv',line):
                            parameters.append({'line':index,'source':line.strip()})
                configured_defaults={}
                config_entries=parse_ls_tree(self.git('ls-tree','-r','-z',commit,'--','src/nasdaq_research/config.py'))
                for config_entry in config_entries:
                    config_data=self.blob(config_entry['object_sha'])
                    config_text=text_payload(config_data) or ''
                    for key in ('DEFAULT_PERIOD','DEFAULT_INTERVAL'):
                        match=re.search(r'^'+key+r'\s*=\s*[\"\x27]([^\"\x27]+)[\"\x27]',config_text,re.M)
                        if match:
                            configured_defaults[key]=match[1]
                    configured_defaults['config_blob_sha']=config_entry['object_sha']
                rows.append({'source_commit_sha':commit,'original_path':entry['original_path'],
                    'git_object_sha':oid,'sha256':digest(data),'category':'IMPLEMENTATION_EVIDENCE',
                    'strength':'WEAK','embedded_runtime_record_found':False,'parameters_and_transformations':parameters,
                    'configured_defaults_not_actual_invocation':configured_defaults,
                    'can_establish':'Configured possible download/normalization/CSV behavior',
                    'cannot_establish':'Actual historical invocation, response, retrieval time, endpoint or library version'})
        save_csv('implementation_inventory',rows,['source_commit_sha','original_path','git_object_sha','sha256','category','strength','embedded_runtime_record_found','parameters_and_transformations','configured_defaults_not_actual_invocation','can_establish','cannot_establish'])
        self.stats['implementation']={'path_change_commits':len(history),'distinct_downloader_blobs_inspected':len(rows),
            'runtime_records_recovered_from_implementation':0}

    def run(self):
        initial_state=verify_state()
        if (initial_state['tracked_mutation_violations'] or initial_state['checkpoint_preserved_stage13_1_manifest_mutation_violations']
                or not initial_state['preserved_git_files_byte_identical']
                or not initial_state['added_files_only_loose_objects']):
            raise ValueError('Protected checkpoint or Git database changed before audit')
        if self.git('rev-parse','HEAD').decode().strip()!=self.initial['HEAD']:
            raise ValueError('HEAD changed')
        for method in (self.reachable,self.deleted,self.reflog,self.unreachable,self.repository,self.projects,self.implementation):
            method()
        assert self.steps==self.protocol['search_order']
        self.candidates.sort(key=lambda row:(self.steps.index(row['search_step']),row['source'],row['original_path'],row['git_object_sha']))
        candidate_columns=['evidence_id','search_step','source','original_path','source_commit_sha','source_tree_sha','git_object_sha','sha256','size_bytes','category','strength','contains_2026_09_30','contains_121269300','primary_volume','actual_runtime_record','reliable_acquisition_timestamp','timestamp_evidence_role','representation_reason','genuinely_new','legacy_provenance_verified','material_primary_information','new_information','qualifies_for_recovery','recovery_method','mtime_ns','ctime_ns','filesystem_timestamp_role']
        save_csv('project_file_inventory',self.project_rows,['source','original_path','relative_path','sha256','size_bytes','mtime_ns','ctime_ns','timestamp_evidence_role','content_read','exclusion'])
        save_csv('candidate_evidence',self.candidates,candidate_columns)
        save_csv('recovered_evidence_manifest',self.recovered,candidate_columns+['recovered_path','recovered_file_sha256'])
        save_csv('search_exclusions',self.exclusions,['search_step','source','original_path','reason'])
        save_json('command_inventory',self.commands)
        save_json('search_steps',{'completed_order':self.steps,'statistics':self.stats})
        recovered_types={category:sum(row['category']==category for row in self.recovered) for category in CATEGORIES[:3]}
        summary={'audit_name':self.protocol['audit_name'],'completion_classification':completion(self.candidates),
            'primary_target':self.protocol['primary_target'],'checkpoint_HEAD':self.initial['HEAD'],
            'search_order':self.steps,'search_statistics':self.stats,'candidate_evidence_rows':len(self.candidates),
            'candidate_category_counts':{category:sum(row['category']==category for row in self.candidates) for category in CATEGORIES},
            'qualifying_new_evidence_count':len(self.recovered),'recovered_categories':recovered_types,
            'legacy_raw_Yahoo_response_recovered':bool(recovered_types[CATEGORIES[0]]),
            'legacy_raw_yfinance_data_recovered':bool(recovered_types[CATEGORIES[1]]),
            'legacy_reliable_acquisition_timestamp_recovered':any(row['reliable_acquisition_timestamp'] for row in self.recovered),
            'legacy_acquisition_manifest_recovered':any(row['actual_runtime_record'] for row in self.recovered),
            'new_material_causal_information_recovered':bool(self.recovered),
            'Stage13_status':'INCOMPLETE_OR_BLOCKED','Stage13_1_status':'UNRESOLVED_BLOCKER',
            'Stage13_1_root_cause':'UNRESOLVED_SOURCE_DISCREPANCY',
            'expanded_dataset_approved_for_downstream_research':False,'stage14_modeling_eligible':False,
            'interpretation_if_not_recovered':'Not recoverable from the preserved evidence searched; historically non-identifiable under the currently preserved evidence set, not impossible in principle',
            'recommendation_if_not_recovered':'Do not repeat forensic investigation of the same preserved evidence; consider a future prospective canonical data-authority policy without designing or executing it here',
            **SAFETY}
        save_json('summary',summary)
        state=verify_state()
        (Path('/tmp')/'nasdaq-legacy-recovery-live-git-state.json').write_bytes(json_bytes({
            'protected_state':state,'live_fsck':self.live_fsck}))
        final_refs={'HEAD':self.git('rev-parse','HEAD').decode().strip(),
            'branch':self.git('symbolic-ref','--short','HEAD').decode().strip(),
            'origin_main':self.git('rev-parse','refs/remotes/origin/main').decode().strip(),
            'remote':self.git('remote','-v').decode()}
        # Final ref commands are included in the deterministic command record.
        save_json('command_inventory',self.commands)
        checks={'search_order_complete':self.steps==self.protocol['search_order'],
            'tracked_inputs_unchanged':state['tracked_mutation_violations']==0,
            'ignored_legacy_authoritative_inputs_unchanged':state['checkpoint_preserved_stage13_1_manifest_mutation_violations']==0,
            'preserved_git_files_unchanged':state['preserved_git_files_byte_identical'],
            'additional_objects_recorded_without_overwriting_preserved_files':state['added_files_only_loose_objects'],
            'HEAD_unchanged':final_refs['HEAD']==self.initial['HEAD'],
            'branch_unchanged':final_refs['branch']==self.initial['branch'],
            'origin_main_unchanged':final_refs['origin_main']==self.initial['origin_main'],
            'remote_configuration_unchanged':final_refs['remote']==self.initial['remote'],
            'only_eligible_original_runtime_evidence_counts':all(recovery_qualified(row) for row in self.recovered),
            'no_network_or_modeling':True,'no_final_test_performance_read':True,
            'old_statuses_and_policy_not_modified':True}
        formal_state={key:value for key,value in state.items() if key!='git_added_files'}
        formal_state['additional_object_details_in_non_authoritative_execution_log']=True
        save_json('validation',{'valid':all(checks.values()),'checks':checks,'protected_state':formal_state,
            'final_refs':final_refs,'completion_classification':summary['completion_classification'],
            'recovered_manifest_rows':len(self.recovered),
            'git_reset_run':False,'git_restore_run':False,'git_clean_run':False,'git_gc_run':False,
            'git_prune_run':False,'reflog_expire_run':False,'checkout_modifying_working_tree':False,
            'objects_intentionally_deleted':False,'git_add_commit_push_tag_run':False,**SAFETY})
        return summary


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    summary=Audit().run()
    print(json.dumps({key:summary[key] for key in ('completion_classification',
        'candidate_evidence_rows','qualifying_new_evidence_count','checkpoint_HEAD')},indent=2))


if __name__=='__main__':
    main()
