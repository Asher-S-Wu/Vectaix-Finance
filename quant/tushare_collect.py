from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time

import requests


START = '20100101'
END = '20260831'
ENDPOINT = 'https://api.tushare.pro'
PREFIX = 'vectaix-general-v1:'
HOLDOUT_PREFIX = 'vectaix-general-holdout-v1:'
FIELDS = {
    'stock_basic': 'ts_code,symbol,name,market,exchange,curr_type,list_status,list_date,delist_date',
    'trade_cal': 'exchange,cal_date,is_open,pretrade_date',
    'index_daily': 'ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount',
    'daily': 'ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount',
    'adj_factor': 'ts_code,trade_date,adj_factor',
    'daily_basic': 'ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,total_share,float_share,free_share,total_mv,circ_mv',
    'stk_limit': 'ts_code,trade_date,pre_close,up_limit,down_limit',
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    pending.replace(path)


def records(payload: dict) -> list[dict]:
    data = payload['data']
    return [dict(zip(data['fields'], row, strict=True)) for row in data['items']]


class Collector:
    def __init__(self, root: Path, credential: Path):
        self.root = root
        self.token = credential.read_text().strip()
        if not self.token:
            raise ValueError('Empty credential file')
        self.lock = threading.Lock()
        self.last_start = 0.0
        self.attempts = 0

    def fetch(self, api: str, params: dict, key: str) -> dict:
        raw = self.root / 'raw' / f'{key}.json'
        receipt = self.root / 'receipts' / f'{key}.json'
        request = {'api': api, 'params': params, 'fields': FIELDS[api]}
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved['request'] != request or saved['status'] != 'ok':
                raise ValueError(f'Existing request is failed or different; requires explicit review: {key}')
            if digest(raw) != saved['sha256']:
                raise ValueError(f'Saved source hash mismatch: {key}')
            return saved
        with self.lock:
            delay = max(0.0, 0.35 - (time.monotonic() - self.last_start))
            time.sleep(delay)
            self.last_start = time.monotonic()
            self.attempts += 1
            if self.attempts > 3500:
                raise ValueError('Explicit request budget exceeded')
        record = {'request': request, 'requestedAt': datetime.now(timezone.utc).isoformat()}
        try:
            response = requests.post(ENDPOINT, json={
                'api_name': api, 'token': self.token, 'params': params, 'fields': FIELDS[api],
            }, timeout=30)
            response.raise_for_status()
            if self.token in response.text:
                raise ValueError('Provider response unexpectedly contains credential; not stored')
            payload = response.json()
            if payload.get('code') != 0 or not isinstance(payload.get('data'), dict):
                raise ValueError(f"Provider rejected request: {payload.get('code')} {payload.get('msg')}")
            data = payload['data']
            if set(data['fields']) != set(FIELDS[api].split(',')):
                raise ValueError(f'Unexpected returned fields: {data["fields"]}')
            rows = records(payload)
            if len(rows) >= params['limit']:
                raise ValueError('Response reached requested row cap; not assumed complete')
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(response.content)
            record.update(status='ok', rows=len(rows), file=str(raw.relative_to(self.root)),
                          sha256=digest(raw), responseRequestId=payload.get('request_id'))
        except Exception as exc:
            record.update(status='error', errorType=type(exc).__name__,
                          error=str(exc).replace(self.token, '[REDACTED]')[:1000])
        record['receivedAt'] = datetime.now(timezone.utc).isoformat()
        write(receipt, record)
        return record

    def required(self, api: str, params: dict, key: str) -> dict:
        receipt = self.fetch(api, params, key)
        if receipt['status'] != 'ok':
            raise ValueError(f'Cannot continue with unsuccessful required source: {key}')
        return receipt


def run(root: Path, credential: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c = Collector(root, credential)
        base, stocks = [], []
        for status in ('L', 'D', 'P'):
            receipt = c.required('stock_basic', {'list_status': status, 'limit': 6000}, f'stock_basic/{status}')
            base.append(receipt)
            stocks.extend(records(json.loads((root / receipt['file']).read_bytes())))
        if len({s['ts_code'] for s in stocks}) != len(stocks):
            raise ValueError('Stock status lists contain duplicate securities')
        eligible = []
        for s in stocks:
            if s['exchange'] not in ('SSE', 'SZSE') or s['curr_type'] != 'CNY':
                continue
            if not s['list_date'] or s['list_date'] > END:
                continue
            if s['delist_date'] and s['delist_date'] < START:
                continue
            if int(hashlib.sha256((PREFIX + s['ts_code']).encode()).hexdigest(), 16) % 10 == 0:
                s = dict(s)
                s['split'] = ('heldout' if int(hashlib.sha256((HOLDOUT_PREFIX + s['ts_code']).encode()).hexdigest(), 16) % 5 == 0 else 'train')
                eligible.append(s)
        eligible.sort(key=lambda s: s['ts_code'])
        if len(eligible) < 300 or len(eligible) > 800:
            raise ValueError('Unexpected sample size; source universe needs review')
        specification = {
            'schemaVersion': 1, 'start': START, 'end': END, 'source': 'Tushare Pro HTTPS',
            'universeRule': 'All L/D/P mainland CNY SSE/SZSE stocks existing at any time in interval; no return/size/current-status filter.',
            'samplingRule': "int(sha256('vectaix-general-v1:'+ts_code),16)%10==0",
            'holdoutRule': "int(sha256('vectaix-general-holdout-v1:'+ts_code),16)%5==0",
            'selectionUsesOutcomes': False, 'stocks': eligible, 'universeRows': len(stocks),
            'fields': FIELDS, 'workers': 3, 'minimumRequestSpacingSeconds': 0.35,
            'requestBudget': 3500, 'automaticRetry': False, 'unknownMissingValuesFilled': False,
            'statusAndNamesUsage': 'Metadata only, never historical predictive features.',
            'units': {'daily.vol': '100 shares', 'daily.amount': '1000 CNY', 'daily.pre_close': 'ex-rights previous close, not raw previous row close'},
        }
        spec_path = root / 'specification.json'
        if spec_path.exists() and json.loads(spec_path.read_text()) != specification:
            raise ValueError('Frozen universe or specification changed')
        write(spec_path, specification)
        print(f'Frozen {len(eligible)} stocks, {sum(s["split"]=="heldout" for s in eligible)} stock holdouts', flush=True)
        for exchange in ('SSE', 'SZSE'):
            base.append(c.required('trade_cal', {'exchange': exchange, 'start_date': START, 'end_date': END, 'limit': 7000}, f'trade_cal/{exchange}'))
        base.append(c.required('index_daily', {'ts_code': '000300.SH', 'start_date': START, 'end_date': END, 'limit': 6000}, 'index_daily/000300.SH'))
        jobs = [(api, {'ts_code': s['ts_code'], 'start_date': START, 'end_date': END, 'limit': 5800}, f'{api}/{s["ts_code"]}')
                for s in eligible for api in ('daily', 'adj_factor', 'daily_basic', 'stk_limit')]
        completed = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            for i, receipt in enumerate(pool.map(lambda args: c.fetch(*args), jobs), 1):
                completed.append(receipt)
                if i % 50 == 0 or i == len(jobs):
                    write(root / 'progress.json', {'completed': i, 'total': len(jobs), 'errors': sum(r['status'] != 'ok' for r in completed), 'updatedAt': datetime.now(timezone.utc).isoformat()})
                    print(f'Collected {i}/{len(jobs)}; errors {sum(r["status"]!="ok" for r in completed)}', flush=True)
        receipts = base + completed
        hashes = {r['file']: r['sha256'] for r in receipts if r['status'] == 'ok'}
        hashes['specification.json'] = digest(spec_path)
        for path in (root / 'receipts').rglob('*.json'):
            hashes[str(path.relative_to(root))] = digest(path)
        failures = [r for r in receipts if r['status'] != 'ok']
        manifest = {'schemaVersion': 1, 'downloadComplete': not failures, 'researchReady': False,
                    'scope': 'Source acquisition only; training loader must independently validate tables and coverage.',
                    'start': START, 'end': END, 'stockCount': len(eligible), 'stocks': eligible,
                    'requests': receipts, 'failures': failures, 'fileHashes': hashes,
                    'collectorSha256': digest(Path(__file__)), 'completedAt': datetime.now(timezone.utc).isoformat()}
        write(root / 'manifest.json', manifest)
        print(f'Acquisition finished: {len(failures)} failed source calls; {root / "manifest.json"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--credential-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/general-tushare'))
    args = parser.parse_args()
    run(args.output, args.credential_file)
