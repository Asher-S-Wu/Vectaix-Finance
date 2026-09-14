import json

import pandas as pd
import pytest

from tmp import review_v5_lot_documents as reviewer


def run_review(tmp_path,monkeypatch,*,effective='2021-08-20',master_isin='OLD'):
    source=tmp_path/'universal';v5=tmp_path/'v5';out=v5/'references/historical_lot_original_review'
    for path in [source/'references/historical_lot_source_audit',source/'references/webb_archive/parquet',
                 source/'hkex/extracted/board_lot',source/'hkex/catalog',v5]:path.mkdir(parents=True,exist_ok=True)
    url='https://www1.hkexnews.hk/listedco/listconews/sehk/2021/0730/test.pdf'
    pd.DataFrame([dict(status='parser_candidate_matches_curated_history',source_url=url,code='06908',
                       archive_issue_id=22195,effective_date=effective,old_lot=5000,new_lot=1000)]).to_parquet(
        source/'references/historical_lot_source_audit/announcement_candidate_audit.parquet',index=False)
    (source/'hkex/catalog/board_lot.json').write_text(json.dumps({'records':[{'pdf_url':url,'published_at':'2021-07-30T19:26:00+08:00'}]}),encoding='utf-8')
    day=pd.Timestamp(effective)
    (source/'hkex/extracted/board_lot/cache_test.txt').write_text(
        f'Stock Code: 6908\nThe board lot size will be changed from 5,000 Shares to 1,000 Shares '
        f'with effect from 9:00 a.m. on {day.day} {day.strftime("%B")} {day.year}.',encoding='utf-8')
    master=pd.DataFrame([
        dict(security_id='08343.HK',exchange_code='08343.HK',isin='OLD',asset_type='equity',identity_status='verified',
             identity_valid_from='2016-12-30',identity_valid_to='2019-11-13'),
        dict(security_id='06908.HK',exchange_code='06908.HK',isin=master_isin,asset_type='equity',identity_status='verified',
             identity_valid_from='2019-11-13',identity_valid_to=None)])
    master.to_parquet(v5/'securities.parquet',index=False)
    pd.DataFrame([dict(security_id='08343.HK',IssueID=22195,matched_isin='OLD',mapping_status='matched_unique_isin')]).to_parquet(
        source/'references/webb_archive/identity_mapping.parquet',index=False)
    pd.DataFrame([
        dict(ID=8025,IssueID=22195,StockCode='8343',StockExID=20,FirstTradeDate='2016-12-30',FinalTradeDate='2019-11-12',DelistDate='2019-11-13',isin=None),
        dict(ID=10803,IssueID=22195,StockCode='6908',StockExID=1,FirstTradeDate='2019-11-13',FinalTradeDate=None,DelistDate=None,isin='OLD')]).to_parquet(
        source/'references/webb_archive/parquet/stocklistings.parquet',index=False)
    monkeypatch.setattr(reviewer,'ROOT',source);monkeypatch.setattr(reviewer,'V5',v5);monkeypatch.setattr(reviewer,'OUT',out)
    reviewer.main()
    return pd.read_parquet(out/'reviewed_original_documents.parquet'),pd.read_parquet(out/'verified_change_facts.parquet')


def test_announcement_code_and_period_override_an_unrelated_unique_issue_mapping(tmp_path,monkeypatch):
    reviewed,facts=run_review(tmp_path,monkeypatch)
    assert reviewed.iloc[0].security_id=='06908.HK'
    assert facts.security_id.tolist()==['06908.HK']
    assert facts.verified.all()


def test_current_isin_conflict_cannot_be_relabelled_as_a_verified_historical_entity(tmp_path,monkeypatch):
    reviewed,facts=run_review(tmp_path,monkeypatch,master_isin='NEW')
    assert pd.isna(reviewed.iloc[0].security_id)
    assert not reviewed.iloc[0].security_id_mapping_unique
    assert facts.empty


def test_a_prelisting_announcement_does_not_attach_to_the_old_counter(tmp_path,monkeypatch):
    reviewed,facts=run_review(tmp_path,monkeypatch,effective='2019-10-01')
    assert pd.isna(reviewed.iloc[0].security_id)
    assert facts.empty
