from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import re
import shutil
import threading
import time

import akshare
import akshare.stock.stock_hk_sina as sina
import numpy as np
import pandas as pd
import requests


START, END = '20100101', '20260831'
PREFIX = 'vectaix-cross-hk-v1:'
MODES = ('raw', 'hfq-factor', 'qfq-factor')
ADJUST = {'raw': '', 'hfq-factor': 'hfq-factor', 'qfq-factor': 'qfq-factor'}
URLS = {
    'raw': sina.hk_sina_stock_hist_url,
    'hfq-factor': sina.hk_sina_stock_hist_hfq_url,
    'qfq-factor': sina.hk_sina_stock_hist_qfq_url,
}
IDENTITIES = {
    'L': Path('data/raw/cross-market-discovery/20260908T145950Z/0-hk_basic.json'),
    'D': Path('data/raw/cross-market-discovery/20260908T145950Z/1-hk_basic.json'),
    'P': Path('data/raw/cross-market-support/hk_basic_P.json'),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def freeze(root: Path) -> dict:
    if akshare.__version__ != '1.18.94':
        raise ValueError('This collection requires the fixed AKShare 1.18.94 decoder.')
    rows, evidence, unknown_lifecycle, outside = [], [], [], []
    for status, source in IDENTITIES.items():
        payload = json.loads(source.read_text())
        if payload['code'] != 0 or payload['data']['has_more'] is not False:
            raise ValueError(f'Incomplete identity source: {source}')
        data = payload['data']
        if len(set(data['fields'])) != len(data['fields']):
            raise ValueError('Duplicate identity fields')
        source_rows = [dict(zip(data['fields'], r, strict=True)) for r in data['items']]
        if any(r['list_status'] != status for r in source_rows):
            raise ValueError(f'Status mismatch: {source}')
        dest = root / 'universe' / f'{status}.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha(dest) != sha(source):
            raise ValueError('Frozen identity source changed')
        if not dest.exists():
            shutil.copyfile(source, dest)
        evidence.append({'status': status, 'originalFile': str(source), 'file': str(dest.relative_to(root)),
                         'sha256': sha(dest), 'rows': len(source_rows)})
        rows.extend(source_rows)
    if len({r['ts_code'] for r in rows}) != len(rows):
        raise ValueError('Duplicate Tushare identities across status lists')
    candidates = []
    for r in rows:
        for k in ('list_date', 'delist_date'):
            if r[k] is not None and not re.fullmatch(r'\d{8}', r[k]):
                raise ValueError(f'Unexpected lifecycle date: {r["ts_code"]} {k}')
        if not r['list_date']:
            unknown_lifecycle.append(r)
            continue
        if r['list_date'] > END or (r['delist_date'] and r['delist_date'] < START):
            outside.append(r['ts_code'])
            continue
        h = hashlib.sha256((PREFIX + r['ts_code']).encode()).hexdigest()
        if int(h, 16) % 7:
            continue
        direct = re.fullmatch(r'\d{5}\.HK', r['ts_code']) is not None
        candidates.append({**r, 'symbol': r['ts_code'], 'samplingHash': h,
                           'sinaSymbol': r['ts_code'][:-3] if direct else None,
                           'candidateEligibility': 'period_intersection_and_fixed_hash_only',
                           'ordinaryShareEligibility': 'unverified', 'issuerIdentityEligibility': 'unverified',
                           'sinaMappingStatus': 'direct_five_digit_code_requires_identity_review' if direct else 'unsupported_identity_suffix'})
    candidates.sort(key=lambda r: r['samplingHash'])
    sample = candidates[:5]
    if not any(r['list_status'] == 'D' for r in sample):
        sample[-1] = next(r for r in candidates[5:] if r['list_status'] == 'D')
    source_text = inspect.getsource(sina.stock_hk_daily)
    source_dir = root / 'source'
    source_dir.mkdir(parents=True, exist_ok=True)
    sources = {'stock_hk_daily.py': source_text, 'hk_js_decode.js': sina.hk_js_decode}
    for filename, text in sources.items():
        target = source_dir / filename
        if target.exists() and target.read_text() != text:
            raise ValueError('Fixed decoder source changed')
        if not target.exists():
            target.write_text(text)
    spec = {
        'schemaVersion': 1, 'start': START, 'end': END, 'akshareVersion': akshare.__version__,
        'collectorSha256': sha(Path(__file__)),
        'interface': 'stock_hk_daily', 'modes': list(MODES), 'adjustArguments': ADJUST, 'urlTemplates': URLS,
        'transport': 'Direct official-source URLs from fixed AKShare source; independent raw/hfq-factor/qfq-factor requests.',
        'rawDecoder': 'AKShare 1.18.94 MiniRacer hk_js_decode without price filling, rounding, adjustment or deduplication.',
        'factorDecoder': 'Original d/f/c or d/f event objects parsed with ast.literal_eval; fields match independent factor branches; total checked.',
        'sourceFiles': {str((source_dir / name).relative_to(root)): sha(source_dir / name) for name in sources},
        'identitySources': evidence, 'fullIdentityRows': len(rows),
        'unknownLifecycleIdentities': unknown_lifecycle, 'outsidePeriodSymbols': outside,
        'universeRule': 'Known listing date <= 20260831 and missing delisting date or delisting date >= 20100101; L/D/P all retained.',
        'samplingRule': "int(sha256('vectaix-cross-hk-v1:'+ts_code),16)%7==0",
        'stockCount': len(candidates), 'stocks': candidates, 'sampleSymbols': [r['symbol'] for r in sample],
        'sampleRule': 'First five sampling hashes; if none D, replace fifth with first subsequent D before any price request.',
        'workers': 2, 'minimumHttpRequestSpacingSeconds': 0.5, 'automaticRetry': False,
        'sourceSubstitution': False, 'unknownValuesFilled': False, 'adjustedPricesComputed': False,
        'researchReady': False,
        'qualification': 'Candidate acquisition only. Ordinary shares, currency history, issuer identity, calendar coverage and price adjustment require independent admission.',
        'identitySuffixPolicy': 'Never remove Tushare identity suffixes to request an unrelated reused Sina code; all three unsupported attempts remain explicit no-request records.',
        'rawCoverage': 'Save complete provider response, including observations outside research dates; no assertion of complete listing history.',
    }
    path = root / 'specification.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError('Frozen collection specification changed')
    if not path.exists():
        save(path, spec)
    return spec


def decode(mode: str, text: str) -> pd.DataFrame:
    if mode == 'raw':
        engine = sina.MiniRacer()
        engine.eval(sina.hk_js_decode)
        values = engine.call('d', text.split('=')[1].split(';')[0].replace('"', ''))
        frame = pd.DataFrame(values)
        if frame.empty:
            return frame
        if not {'date', 'open', 'high', 'low', 'close', 'volume'}.issubset(frame.columns):
            raise ValueError(f'Unexpected raw columns: {list(frame.columns)}')
        frame['date'] = pd.to_datetime(frame['date']).dt.strftime('%Y-%m-%d')
        for k in frame.columns.drop('date'):
            frame[k] = frame[k].astype(float)
        return frame
    payload = ast.literal_eval(text.split('=', 1)[1].split('\n')[0].strip().rstrip(';'))
    fields = ['date', 'hfq_factor', 'cash'] if mode == 'hfq-factor' else ['date', 'qfq_factor']
    values = payload['data']
    source_fields = ['d', 'f', 'c'] if mode == 'hfq-factor' else ['d', 'f']
    if not isinstance(values, list) or any(not isinstance(row, dict) or set(row) != set(source_fields) for row in values):
        raise ValueError('Unexpected factor event objects')
    if payload['total'] != len(values):
        raise ValueError('Factor response total differs from event count')
    frame = pd.DataFrame(values, columns=source_fields)
    frame.columns = fields
    frame['date'] = pd.to_datetime(frame['date']).dt.strftime('%Y-%m-%d')
    if frame['date'].isna().any():
        raise ValueError('Factor event date missing')
    frame.attrs['providerTotal'] = payload['total']
    return frame


def quality(frame: pd.DataFrame, mode: str) -> dict:
    if frame.empty:
        return {'rows': 0, 'empty': True}
    q = {'rows': len(frame), 'columns': list(frame.columns), 'firstDate': str(frame.date.min()),
         'lastDate': str(frame.date.max()), 'duplicateDates': [str(d) if pd.notna(d) else None for d in frame.loc[frame.date.duplicated(False), 'date']],
         'nullCounts': {k: int(v) for k, v in frame.isna().sum().items()},
         'researchWindowRows': int(frame.date.between('2010-01-01', '2026-08-31').sum())}
    q['nonfiniteCounts'] = {}
    for col in frame.columns.drop('date'):
        values = pd.to_numeric(frame[col], errors='raise').to_numpy(float)
        q['nonfiniteCounts'][col] = int((~np.isfinite(values)).sum())
    if mode == 'raw':
        q['amountPresent'] = 'amount' in frame.columns
        for col in ('open', 'high', 'low', 'close', 'volume', 'amount'):
            if col in frame:
                q[f'{col}NonpositiveDates'] = frame.loc[frame[col].le(0), 'date'].tolist()
        q['highBelowOpenOrCloseDates'] = frame.loc[frame.high.lt(frame[['open', 'close']].max(axis=1) - 1e-4), 'date'].tolist()
        q['lowAboveOpenOrCloseDates'] = frame.loc[frame.low.gt(frame[['open', 'close']].min(axis=1) + 1e-4), 'date'].tolist()
        q['ohlcEnvelopePolicy'] = 'Audit only; HK source-defined high/low need not enclose official open/close. No price edited or rejected here.'
    else:
        factor = 'hfq_factor' if mode == 'hfq-factor' else 'qfq_factor'
        q['nonpositiveFactorDates'] = frame.loc[pd.to_numeric(frame[factor]).le(0), 'date'].tolist()
        q['eventDatesNotDailyCoverage'] = True
        q['providerTotal'] = frame.attrs['providerTotal']
    return q


class Collector:
    def __init__(self, root: Path, spec: dict):
        self.root, self.spec = root, spec
        self.lock = threading.Lock()
        self.last_start = 0.0

    def fetch(self, stock: dict, mode: str) -> dict:
        symbol = stock['symbol']
        target = self.root / 'receipts' / mode / f'{symbol}.json'
        request = {'interface': 'stock_hk_daily', 'params': {'symbol': stock['sinaSymbol'], 'adjust': ADJUST[mode]},
                   'mode': mode, 'ts_code': symbol, 'url': URLS[mode].format(stock['sinaSymbol']) if stock['sinaSymbol'] else None}
        if target.exists():
            saved = json.loads(target.read_text())
            if saved['request'] != request:
                raise ValueError('Existing attempt differs; no retry permitted')
            for key, hash_key in [('rawFile', 'rawSha256'), ('file', 'sha256')]:
                if key in saved and sha(self.root / saved[key]) != saved[hash_key]:
                    raise ValueError('Saved source modified')
            return saved
        receipt = {'request': request, 'akshareVersion': self.spec['akshareVersion'],
                   'interfaceSourceSha256': self.spec['sourceFiles']['source/stock_hk_daily.py'],
                   'researchReady': False, 'httpAttempted': False}
        if stock['sinaSymbol'] is None:
            receipt.update(status='unsupported_identity_mapping', recordedAt=now(),
                           reason='Tushare historical identity suffix has no verified Sina endpoint mapping; not stripped.')
            save(target, receipt)
            return receipt
        with self.lock:
            time.sleep(max(0.0, 0.5 - (time.monotonic() - self.last_start)))
            self.last_start = time.monotonic()
            receipt.update(requestedAt=now(), httpAttempted=True)
        try:
            response = requests.get(request['url'], timeout=(15, 30), allow_redirects=False)
            raw = self.root / 'raw' / mode / f'{symbol}.txt'
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(response.content)
            receipt.update(rawFile=str(raw.relative_to(self.root)), rawSha256=sha(raw),
                           statusCode=response.status_code, contentType=response.headers.get('Content-Type'),
                           responseEncoding=response.encoding, responseBytes=len(response.content), receivedAt=now())
            receipt['status'] = 'received_unparsed'
            save(target, receipt)
            if response.status_code != 200:
                receipt.update(status='http_error', reason='Unexpected HTTP status; no retry or redirect followed.')
            else:
                try:
                    frame = decode(mode, response.text)
                    decoded = self.root / 'decoded' / mode / f'{symbol}.csv'
                    decoded.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_csv(decoded, index=False)
                    receipt.update(status='ok' if len(frame) else 'empty', rows=len(frame), columns=list(frame.columns),
                                   file=str(decoded.relative_to(self.root)), sha256=sha(decoded), quality=quality(frame, mode))
                except Exception as exc:
                    receipt.update(status='decode_error', errorType=type(exc).__name__, error=str(exc)[:1000])
        except requests.RequestException as exc:
            receipt.update(status='transport_error', errorType=type(exc).__name__, error=str(exc)[:1000], receivedAt=now())
        receipt['completedAt'] = now()
        save(target, receipt)
        return receipt


def manifest(root: Path, spec: dict, phase: str) -> dict:
    receipts = []
    hashes = {'specification.json': sha(root / 'specification.json'), **spec['sourceFiles']}
    hashes.update({r['file']: r['sha256'] for r in spec['identitySources']})
    for stock in spec['stocks']:
        for mode in MODES:
            file = root / 'receipts' / mode / f'{stock["symbol"]}.json'
            if not file.exists():
                continue
            r = json.loads(file.read_text()); receipts.append(r)
            hashes[str(file.relative_to(root))] = sha(file)
            for key, hash_key in [('rawFile', 'rawSha256'), ('file', 'sha256')]:
                if key in r:
                    hashes[r[key]] = r[hash_key]
    status_counts = {s: sum(r['status'] == s for r in receipts) for s in sorted({r['status'] for r in receipts})}
    result = {'schemaVersion': 1, 'collectionComplete': len(receipts) == len(spec['stocks']) * len(MODES),
              'downloadComplete': len(receipts) == len(spec['stocks']) * len(MODES) and all(r['status'] == 'ok' for r in receipts),
              'researchReady': False, 'phase': phase, 'start': START, 'end': END,
              'stockCount': spec['stockCount'], 'stocks': spec['stocks'], 'specificationSha256': sha(root / 'specification.json'),
              'requests': receipts, 'plannedRequests': len(spec['stocks']) * len(MODES), 'completedRequests': len(receipts),
              'actualHttpRequests': sum(r['httpAttempted'] for r in receipts), 'statusCounts': status_counts,
              'failures': [{'symbol': r['request']['ts_code'], 'mode': r['request']['mode'], 'status': r['status']}
                           for r in receipts if r['status'] != 'ok'], 'fileHashes': hashes, 'updatedAt': now(),
              'qualification': spec['qualification'], 'adjustedPricesComputed': False,
              'automaticRetry': False, 'sourceSubstitution': False, 'unknownValuesFilled': False}
    save(root / 'manifest.json', result)
    save(root / 'progress.json', {k: result[k] for k in ('phase', 'collectionComplete', 'plannedRequests', 'completedRequests', 'actualHttpRequests', 'statusCounts', 'updatedAt')})
    return result


def run(root: Path, phase: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = freeze(root)
        stocks = [s for s in spec['stocks'] if s['symbol'] in spec['sampleSymbols']] if phase == 'samples' else spec['stocks']
        if phase == 'all' and not (root / 'sample-review.json').exists():
            raise ValueError('Five source samples must be independently reviewed before the full collection.')
        c = Collector(root, spec)
        jobs = [(s, mode) for s in stocks for mode in MODES]
        manifest(root, spec, phase)
        print(f'Frozen {len(spec["stocks"])} candidates; phase={phase}; jobs={len(jobs)}', flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(c.fetch, s, mode) for s, mode in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                receipt = future.result()
                if phase == 'samples' or i % 30 == 0 or i == len(jobs):
                    result = manifest(root, spec, phase)
                    print(f'{i}/{len(jobs)}; completed={result["completedRequests"]}; {result["statusCounts"]}; latest={receipt["request"]["ts_code"]}/{receipt["request"]["mode"]}', flush=True)
        manifest(root, spec, phase)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/cross-market-hk'))
    parser.add_argument('--phase', choices=('samples', 'all'), required=True)
    args = parser.parse_args()
    run(args.output, args.phase)
