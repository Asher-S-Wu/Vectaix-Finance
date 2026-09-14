import json

import pandas as pd
import pytest

from tmp import build_snapshot_lot_facts as builder


def inputs(tmp_path,monkeypatch,protocol):
    data=tmp_path/'v5';root=data/'references/webb_board_lot_snapshots'
    (root/'parsed').mkdir(parents=True);(root/'parsed_reit').mkdir();(data/'bars').mkdir()
    rows=[]
    for day,observed in [('2022-06-29','2022-06-23'),('2022-07-04','2022-07-04')]:
        rows.append(dict(date=pd.Timestamp(day),source_observation_date=pd.Timestamp(observed),
                         source_lot_size=6000 if day=='2022-06-29' else 2000,
                         security_id='01049.HK',source_issue_id=218,source_listing_id=579,
                         stock_code='01049',source_stock_code='01049',lot_size=6000 if day=='2022-06-29' else 2000,
                         status='ok',identity_mapping_verified=True,historical_value_date_verified=True,
                         parse_protocol=protocol,source_document=f'/raw/{day}.html.gz',source_url='https://example.org/source'))
    pd.DataFrame(rows).to_parquet(root/'parsed/2022.parquet',index=False)
    pd.DataFrame({'date':pd.to_datetime(['2022-06-29','2022-07-04']),'security_id':['01049.HK']*2,'quote_present':[False,True]}).to_parquet(data/'bars/2022.parquet',index=False)
    (data/'partial_replay_inputs_audit.json').write_text(json.dumps({'corporate_actions_rows':123}),encoding='utf-8')
    monkeypatch.setattr(builder,'DATA',data);monkeypatch.setattr(builder,'ROOT',root)
    return data,root


def test_facts_require_source_observation_date_even_if_flags_are_incorrect(tmp_path,monkeypatch):
    data,root=inputs(tmp_path,monkeypatch,'source-issue-counter-date-v3')
    builder.main()
    facts=pd.read_parquet(data/'historical_lots.parquet')
    assert len(facts)==1 and facts.iloc[0].lot_size==2000
    assert facts.source_observation_date.eq(facts.lot_valid_from).all()
    audit=json.loads((data/'partial_replay_inputs_audit.json').read_text(encoding='utf-8'))
    assert audit['corporate_actions_rows']==123


def test_v2_year_cannot_enter_new_facts(tmp_path,monkeypatch):
    data,_=inputs(tmp_path,monkeypatch,'source-issue-counter-date-v2')
    with pytest.raises(ValueError,match='观察|协议|重建'):
        builder.main()
    assert not (data/'historical_lots.parquet').exists()
