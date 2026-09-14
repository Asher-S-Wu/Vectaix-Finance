import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hk_quant.transfer_labels import build_transfer_label_patches


def transfer_files(tmp_path):
    root = tmp_path / 'data'
    (root / 'bars').mkdir(parents=True)
    (root / 'features').mkdir()
    source_url = 'https://issuer.test/transfer.pdf'
    pdf = tmp_path / 'transfer.pdf'
    pdf.write_bytes(b'%PDF-1.7\nreviewed original source')
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps({'records': [dict(pdf_url=source_url,
        published_at='2019-12-09T20:42:00+08:00', stock_codes=['08199'])]}))
    listings_path = tmp_path / 'stocklistings.parquet'
    pd.DataFrame([
        dict(ID=10, IssueID=25221, StockCode='8199', FirstTradeDate='2017-09-27',
             FinalTradeDate='2019-12-16', DelistDate='2019-12-17'),
        dict(ID=11, IssueID=25221, StockCode='6966', FirstTradeDate='2019-12-17',
             FinalTradeDate=None, DelistDate=None),
    ]).to_parquet(listings_path, index=False)
    pd.DataFrame([
        dict(security_id='08199.HK', exchange_code='08199.HK', isin='SAME', currency='HKD',
             identity_status='verified', identity_valid_from=pd.Timestamp('2017-09-27'), identity_valid_to=pd.Timestamp('2019-12-17')),
        dict(security_id='06966.HK', exchange_code='06966.HK', isin='SAME', currency='HKD',
             identity_status='verified', identity_valid_from=pd.Timestamp('2019-12-17'), identity_valid_to=pd.NaT),
    ]).to_parquet(root / 'securities.parquet', index=False)
    bars = []
    for security, day, price, factor in [
        ('08199.HK', '2019-12-10', 100., 2.), ('08199.HK', '2019-12-16', 110., 2.),
        ('06966.HK', '2019-12-16', 110., .5), ('06966.HK', '2019-12-17', 121., .5),
    ]:
        bars.append(dict(security_id=security, date=pd.Timestamp(day), raw_close=price,
            cum_adjfactor=factor, adj_close_hkd=price * factor, fx_to_hkd=1., currency='HKD',
            data_valid=True, quote_present=True, identity_date_verified=False))
    pd.DataFrame(bars).to_parquet(root / 'bars/2019.parquet', index=False)
    pd.DataFrame([dict(security_id='08199.HK', date=pd.Timestamp('2019-12-10'),
        adj_close_hkd=200., fwd_return_5=np.nan, label_end_5=pd.Timestamp('2019-12-17'))]).to_parquet(
            root / 'features/08199.HK.parquet', index=False)
    events = tmp_path / 'events.parquet'
    pd.DataFrame([dict(event_id='transfer', event_type='listing_transfer_existing_shares_continue',
        old_security_id='08199.HK', new_security_id='06966.HK', source_issue_id=25221,
        old_source_listing_id=10, new_source_listing_id=11, old_listing_start=pd.Timestamp('2017-09-27'),
        old_listing_end_exclusive=pd.Timestamp('2019-12-17'), new_listing_start=pd.Timestamp('2019-12-17'),
        new_listing_end_exclusive=pd.NaT, old_last_trading_date=pd.Timestamp('2019-12-16'),
        listing_transfer_effective_date=pd.Timestamp('2019-12-17'), published_at='2019-12-09T20:42:00+08:00',
        continuing_shares_per_old_share=1., currency='HKD', isin='SAME', verified=True,
        source_identity_verified=True, source_url=source_url, source_document=str(pdf),
        source_catalog=str(catalog), source_identity_file=str(listings_path),
        source_quote_file=str(root / 'bars/2019.parquet'))]).to_parquet(events, index=False)
    queue = tmp_path / 'queue.parquet'
    pd.DataFrame([dict(date=pd.Timestamp('2019-12-10'), security_id='08199.HK', horizon=5,
        label_end=pd.Timestamp('2019-12-17'), gap_reason='listing_exit_requires_verified_terminal_or_transfer_outcome')]).to_parquet(queue, index=False)
    return root, events, queue


def change(path, column, value, row=0):
    frame = pd.read_parquet(path)
    frame.loc[row, column] = value
    frame.to_parquet(path, index=False)


def write_text_pdf(path, text):
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
        NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
    stream = DecodedStreamObject()
    stream.set_data(('BT /F1 10 Tf 10 700 Td (' + text + ') Tj ET').encode())
    page[NameObject('/Contents')] = writer._add_object(stream)
    with path.open('wb') as output:
        writer.write(output)


def test_bridge_preserves_actual_transfer_price_move_without_mutating_sources(tmp_path):
    args = transfer_files(tmp_path)
    root, events, queue = args
    paths = [root / 'bars/2019.parquet', root / 'features/08199.HK.parquet', events, queue]
    before = {path: path.read_bytes() for path in paths}
    patches, audit = build_transfer_label_patches(*args)
    row = patches.iloc[0]
    assert len(patches) == 1
    assert row.new_fwd_return == pytest.approx(.21)
    assert pd.isna(row.old_fwd_return)
    assert row.adjustment_scale_bridge == 4.
    assert row.anchor_date == pd.Timestamp('2019-12-16')
    assert row.successor_anchor_role == 'scale_calibration_only_not_listed_counter'
    assert row.source_url == 'https://issuer.test/transfer.pdf'
    assert row.date == pd.Timestamp('2019-12-10') and row.label_end == pd.Timestamp('2019-12-17')
    assert audit['patches'] == 1 and audit['excluded_rows'] == 0
    assert {path: path.read_bytes() for path in paths} == before


@pytest.mark.parametrize('column,value', [('continuing_shares_per_old_share', 2.), ('currency', 'USD')])
def test_non_unit_or_non_hkd_transfer_is_rejected(tmp_path, column, value):
    args = transfer_files(tmp_path)
    change(args[1], column, value)
    with pytest.raises(ValueError, match='1:1|HKD'):
        build_transfer_label_patches(*args)


@pytest.mark.parametrize('column,value', [('raw_close', 109.), ('data_valid', False), ('quote_present', False)])
def test_mismatched_or_invalid_prelisting_anchor_is_rejected(tmp_path, column, value):
    args = transfer_files(tmp_path)
    change(args[0] / 'bars/2019.parquet', column, value, row=2)
    with pytest.raises(ValueError, match='锚点'):
        build_transfer_label_patches(*args)


def test_reused_code_with_other_source_issue_cannot_bridge(tmp_path):
    args = transfer_files(tmp_path)
    change(tmp_path / 'stocklistings.parquet', 'IssueID', 3485, row=0)
    with pytest.raises(ValueError, match='源证券'):
        build_transfer_label_patches(*args)


@pytest.mark.parametrize('old_isin', ['OLD_NAME_ISIN', None])
@pytest.mark.parametrize('certificate_clause', [
    'and will not involve any transfer or exchange.',
    'and be valid for trading. No change will be made to the share certificate.',
])
def test_different_current_isin_requires_original_share_continuity_evidence(tmp_path, old_isin, certificate_clause):
    args = transfer_files(tmp_path)
    change(args[0] / 'securities.parquet', 'isin', old_isin, row=0)
    with pytest.raises(ValueError, match='源证券|股证'):
        build_transfer_label_patches(*args)

    excerpt = 'The existing share certificates continue to be good evidence of legal title ' + certificate_clause
    write_text_pdf(tmp_path / 'transfer.pdf', excerpt)
    events = pd.read_parquet(args[1])
    events['old_isin_at_master_snapshot'] = old_isin
    events['share_continuity_verified'] = True
    events['share_continuity_evidence_page'] = 1
    events['share_continuity_evidence_excerpt'] = excerpt
    events.to_parquet(args[1], index=False)
    patches, audit = build_transfer_label_patches(*args)
    assert len(patches) == 1 and audit['excluded_rows'] == 0
    assert patches.new_fwd_return.iloc[0] == pytest.approx(.21)
    change(args[1], 'share_continuity_evidence_excerpt', 'Shares were exchanged for another company.')
    with pytest.raises(ValueError, match='股证'):
        build_transfer_label_patches(*args)


def test_final_date_clarification_can_reference_earlier_original_share_terms(tmp_path):
    args = transfer_files(tmp_path)
    change(args[0] / 'securities.parquet', 'isin', 'OLD_ISIN', row=0)
    excerpt = 'The existing share certificates continue to be good evidence of legal title and will not involve any transfer or exchange.'
    original = tmp_path / 'original.pdf'
    write_text_pdf(original, excerpt)
    write_text_pdf(tmp_path / 'transfer.pdf', 'The last trading day is corrected. All other information remains unchanged.')
    events = pd.read_parquet(args[1])
    events['old_isin_at_master_snapshot'] = 'OLD_ISIN'
    events['share_continuity_verified'] = True
    events['share_continuity_evidence_page'] = 1
    events['share_continuity_evidence_excerpt'] = excerpt
    events['share_continuity_source_document'] = str(original)
    events['share_continuity_source_url'] = 'https://issuer.test/original.pdf'
    events['share_continuity_published_at'] = '2019-12-08T18:00:00+08:00'
    events.to_parquet(args[1], index=False)
    catalog_path = tmp_path / 'catalog.json'
    catalog = json.loads(catalog_path.read_text())
    catalog['records'].append(dict(pdf_url='https://issuer.test/original.pdf',
        published_at='2019-12-08T18:00:00+08:00', stock_codes=['08199']))
    catalog_path.write_text(json.dumps(catalog))
    patches, audit = build_transfer_label_patches(*args)
    assert len(patches) == 1 and audit['excluded_rows'] == 0
    assert patches.event_published_at.iloc[0] == pd.Timestamp('2019-12-09T20:42:00+08:00')
    assert patches.new_fwd_return.iloc[0] == pytest.approx(.21)
    catalog['records'][1]['stock_codes'] = ['99999']
    catalog_path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match='源证券'):
        build_transfer_label_patches(*args)
    catalog['records'][1]['stock_codes'] = ['08199']
    later = '2019-12-10T18:00:00+08:00'
    change(args[1], 'share_continuity_published_at', later)
    catalog['records'][1]['published_at'] = later
    catalog_path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match='公开时间'):
        build_transfer_label_patches(*args)


def test_reused_security_identity_uses_its_verified_exchange_code(tmp_path):
    args = transfer_files(tmp_path)
    root, events_path, queue_path = args
    old = '08199.HK'
    reused = '08199!AA.HK'
    for path in [root / 'securities.parquet', root / 'bars/2019.parquet',
                 root / 'features/08199.HK.parquet', queue_path]:
        frame = pd.read_parquet(path)
        frame.loc[frame.security_id.eq(old), 'security_id'] = reused
        frame.to_parquet(path, index=False)
    (root / 'features/08199.HK.parquet').rename(root / 'features/08199!AA.HK.parquet')
    change(events_path, 'old_security_id', reused)
    patches, audit = build_transfer_label_patches(*args)
    assert len(patches) == 1 and audit['excluded_rows'] == 0
    assert patches.security_id.iloc[0] == reused
    assert patches.new_fwd_return.iloc[0] == pytest.approx(.21)


def test_missing_original_document_and_catalog_time_conflict_are_rejected(tmp_path):
    args = transfer_files(tmp_path)
    (tmp_path / 'transfer.pdf').unlink()
    with pytest.raises(ValueError, match='源文件'):
        build_transfer_label_patches(*args)
    (tmp_path / 'transfer.pdf').write_bytes(b'%PDF-1.7\nsource')
    catalog = tmp_path / 'catalog.json'
    payload = json.loads(catalog.read_text())
    payload['records'][0]['published_at'] = '2019-12-08T20:42:00+08:00'
    catalog.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='公告时间'):
        build_transfer_label_patches(*args)


def test_existing_label_cannot_be_overwritten(tmp_path):
    args = transfer_files(tmp_path)
    change(args[0] / 'features/08199.HK.parquet', 'fwd_return_5', .12)
    patches, audit = build_transfer_label_patches(*args)
    assert patches.empty
    assert audit['excluded'][0]['reason'] == 'existing_label_present'


def test_missing_successor_quote_does_not_use_anchor_as_endpoint(tmp_path):
    args = transfer_files(tmp_path)
    path = args[0] / 'bars/2019.parquet'
    frame = pd.read_parquet(path)
    frame = frame.loc[~(frame.security_id.eq('06966.HK') & frame.date.eq('2019-12-17'))]
    frame.to_parquet(path, index=False)
    patches, audit = build_transfer_label_patches(*args)
    assert patches.empty
    assert audit['excluded'][0]['reason'] == 'invalid_signal_or_successor_observation'


def test_signal_feature_must_match_its_own_raw_price_and_adjustment(tmp_path):
    args = transfer_files(tmp_path)
    change(args[0] / 'features/08199.HK.parquet', 'adj_close_hkd', 150.)
    patches, audit = build_transfer_label_patches(*args)
    assert patches.empty
    assert audit['excluded'][0]['reason'] == 'source_signal_adjustment_mismatch'


@pytest.mark.parametrize('problem,reason', [
    ('signal_period', 'signal_outside_old_listing'),
    ('end_period', 'endpoint_outside_new_listing'),
    ('confirmation', 'endpoint_outside_development'),
    ('publication', 'event_not_public_by_label_cutoff'),
])
def test_row_must_belong_to_own_listing_and_development_time(tmp_path, problem, reason):
    args = transfer_files(tmp_path)
    if problem == 'signal_period':
        change(args[2], 'date', pd.Timestamp('2017-09-26'))
    elif problem == 'end_period':
        change(args[2], 'label_end', pd.Timestamp('2019-12-16'))
    elif problem == 'confirmation':
        change(args[2], 'label_end', pd.Timestamp('2024-01-02'))
    else:
        published = '2019-12-17T19:01:00+08:00'
        change(args[1], 'published_at', published)
        catalog = tmp_path / 'catalog.json'
        payload = json.loads(catalog.read_text())
        payload['records'][0]['published_at'] = published
        catalog.write_text(json.dumps(payload))
    patches, audit = build_transfer_label_patches(*args)
    assert patches.empty
    assert audit['excluded'][0]['reason'] == reason
