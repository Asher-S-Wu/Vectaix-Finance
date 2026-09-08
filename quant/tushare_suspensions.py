from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import time

import requests


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_suffix('.pending')
    stage.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    stage.replace(path)


def collect(credential: Path, universe: Path, root: Path):
    token = credential.read_text().strip()
    spec = json.loads(universe.read_bytes())
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for i, stock in enumerate(spec['stocks'], 1):
            code = stock['ts_code']
            raw, receipt = root / 'raw' / f'{code}.json', root / 'receipts' / f'{code}.json'
            request = {'api': 'suspend_d', 'params': {'ts_code': code, 'start_date': spec['start'], 'end_date': spec['end'], 'limit': 6000},
                       'fields': 'ts_code,trade_date,suspend_timing,suspend_type'}
            if receipt.exists():
                item = json.loads(receipt.read_bytes())
                if item['request'] != request or item['status'] != 'ok' or sha(raw) != item['sha256']:
                    raise ValueError(f'Existing suspension request needs explicit review: {code}')
            else:
                item = {'request': request, 'requestedAt': datetime.now(timezone.utc).isoformat()}
                try:
                    response = requests.post('https://api.tushare.pro', json={
                        'api_name': request['api'], 'params': request['params'], 'fields': request['fields'], 'token': token,
                    }, timeout=30)
                    response.raise_for_status()
                    if token in response.text:
                        raise ValueError('Credential appeared in provider response; not stored')
                    payload = response.json()
                    if payload.get('code') != 0:
                        raise ValueError(f"Provider rejected request: {payload.get('code')} {payload.get('msg')}")
                    data = payload['data']
                    if set(data['fields']) != set(request['fields'].split(',')) or len(data['items']) >= 6000:
                        raise ValueError('Unexpected fields or unproven response completeness')
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    raw.write_bytes(response.content)
                    item.update(status='ok', rows=len(data['items']), file=str(raw.relative_to(root)), sha256=sha(raw))
                except Exception as exc:
                    item.update(status='error', errorType=type(exc).__name__, error=str(exc).replace(token, '[REDACTED]')[:1000])
                item['receivedAt'] = datetime.now(timezone.utc).isoformat()
                save(receipt, item)
                time.sleep(.65)
            rows.append(item)
            if i % 50 == 0 or i == len(spec['stocks']):
                print(f'Suspensions {i}/{len(spec["stocks"])}; errors {sum(x["status"]!="ok" for x in rows)}', flush=True)
        hashes = {x['file']: x['sha256'] for x in rows if x['status'] == 'ok'}
        hashes.update({str(x.relative_to(root)): sha(x) for x in (root / 'receipts').glob('*.json')})
        save(root / 'manifest.json', {'schemaVersion': 1, 'downloadComplete': all(x['status'] == 'ok' for x in rows),
             'researchReady': False, 'start': spec['start'], 'end': spec['end'], 'stockCount': len(spec['stocks']),
             'universeSpecificationSha256': sha(universe), 'requests': rows, 'fileHashes': hashes,
             'collectorSha256': sha(Path(__file__)), 'completedAt': datetime.now(timezone.utc).isoformat(),
             'interpretation': 'Original S/R dates and intraday timing retained. No synthetic suspension dates or prices.',
             'automaticRetry': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--credential-file', type=Path, required=True)
    p.add_argument('--universe', type=Path, default=Path('data/general-tushare/specification.json'))
    p.add_argument('--output', type=Path, default=Path('data/general-tushare-suspensions'))
    args = p.parse_args()
    collect(args.credential_file, args.universe, args.output)
