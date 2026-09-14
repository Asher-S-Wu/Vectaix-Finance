import importlib.util
import json
from pathlib import Path

import pandas as pd


def test_catalog_match_does_not_certify_archive_amount_publication(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'tmp/build_catalog_cash_action_candidate.py'
    spec = importlib.util.spec_from_file_location('cash_catalog_candidate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    catalog = tmp_path / 'catalog'
    archive = tmp_path / 'archive'
    output = tmp_path / 'output'
    for directory in (catalog, archive, output):
        directory.mkdir()
    monkeypatch.setattr(module, 'CATALOG', catalog)
    monkeypatch.setattr(module, 'ARCHIVE', archive)
    monkeypatch.setattr(module, 'OUT', output)
    for name in ('dividend', 'dividend_form', 'payment_change'):
        records = []
        if name == 'dividend_form':
            records = [dict(stock_codes=['778'], published_at='2026-03-10T16:31:00+08:00',
                            title='Final distribution', pdf_url='https://example.test/form.pdf')]
        (catalog / (name + '.json')).write_text(json.dumps(dict(records=records)), encoding='utf-8')
    pd.DataFrame([dict(IssueID=1, StockCode='00778', FirstTradeDate='2010-04-20',
                       FinalTradeDate=None, DelistDate=None)]).to_parquet(archive / 'identity_periods.parquet')
    pd.DataFrame([dict(security_id='00778.HK', source_issue_id=1,
                       announced_date='2026-03-10', effective_date='2026-03-26',
                       payment_date='2026-04-24', detail='Fin', cash_per_share=0.1681,
                       amount_known_at='2026-03-10T16:31:00+08:00')]).to_parquet(output / 'cash_action_candidates.parquet')

    module.main()

    actual = pd.read_parquet(output / 'cash_actions_catalog_timestamp_candidate.parquet').iloc[0]
    assert pd.isna(actual.amount_known_at)
    assert actual.catalog_published_at == pd.Timestamp('2026-03-10T08:31:00Z')
    assert actual.catalog_source_url == 'https://example.test/form.pdf'
    assert actual.cash_per_share == 0.1681
    assert not actual.verified
    assert not actual.formal_replay_eligible
    audit = json.loads((output / 'catalog_timestamp_candidate_audit.json').read_text())
    assert audit['catalog_date_matches'] == 1
    assert audit['amount_known_at_rows'] == 0
