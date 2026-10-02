import importlib.util
import numpy as np
import pandas as pd


def test_build_features_retains_same_37_baseline_factors(tmp_path):
    assert importlib.util.find_spec('ashare_quant.dataset') is not None, 'dataset builder not implemented'
    from ashare_quant.dataset import build_features
    from tests.test_ashare_core import bars_fixture
    from ashare_quant.data import normalize_securities
    bars,calendar=bars_fixture()
    root=tmp_path/'cn'; (root/'bars_by_security'/'600000.SH').mkdir(parents=True)
    bars.to_parquet(root/'bars_by_security'/'600000.SH'/'2018.parquet',index=False)
    pd.DataFrame({'cal_date':calendar,'is_open':1}).to_parquet(root/'calendar.parquet',index=False)
    normalize_securities(pd.DataFrame([dict(ts_code='600000.SH',name='A',list_date='19910101',delist_date=None)])).to_parquet(root/'securities.parquet',index=False)
    result=build_features(root)
    assert result['status']=='complete' and len(result['features'])==37
    assert result['market']=='CN' and result['return_basis'].startswith('CNY')
    latest=pd.read_parquet(root/'latest_inputs.parquet')
    assert latest.date.nunique()==1 and latest.date.iloc[0]==calendar[-1]
    assert not any('hkd' in c for c in latest.columns)
    assert (root/'risk_returns.parquet').is_file()


def test_alias_mapping_does_not_merge_conflicting_quotes():
    assert importlib.util.find_spec('ashare_quant.dataset') is not None
    from ashare_quant.dataset import canonicalize_codes
    import pytest
    mapping=pd.DataFrame({'o_code':['830001.BJ'],'n_code':['920001.BJ']})
    good=pd.DataFrame({'security_id':['830001.BJ'],'date':pd.to_datetime(['20200101']),'raw_close':[1.]})
    out=canonicalize_codes(good,mapping)
    assert out.security_id.iloc[0]=='920001.BJ' and out.source_security_id.iloc[0]=='830001.BJ'
    both=pd.concat([good,good.assign(security_id='920001.BJ',raw_close=2.)])
    with pytest.raises(ValueError,match='alias'):
        canonicalize_codes(both,mapping)


def test_incomplete_source_year_cannot_be_normalized(tmp_path):
    from ashare_quant import dataset
    import pytest
    assert hasattr(dataset,'prepare_data'),'raw preparation not implemented'
    with pytest.raises(ValueError,match='calendar'):
        dataset.prepare_data(tmp_path)


def test_partial_collection_cannot_be_labeled_complete_features(tmp_path):
    from ashare_quant.dataset import build_features
    import json,pytest
    (tmp_path/'data_audit.json').write_text(json.dumps({'status':'partial'}))
    with pytest.raises(ValueError,match='incomplete'):
        build_features(tmp_path)


def test_feature_manifest_tracks_mature_missing_labels(tmp_path):
    from ashare_quant.dataset import build_features
    from tests.test_ashare_core import bars_fixture
    from ashare_quant.data import normalize_securities
    bars,calendar=bars_fixture();bars=bars.drop(index=[350])
    root=tmp_path/'data';(root/'bars_by_security'/'600000.SH').mkdir(parents=True)
    bars.to_parquet(root/'bars_by_security'/'600000.SH'/'2018.parquet',index=False)
    pd.DataFrame({'cal_date':calendar,'is_open':1}).to_parquet(root/'calendar.parquet',index=False)
    normalize_securities(pd.DataFrame([dict(ts_code='600000.SH',name='A',list_date='19910101',delist_date=None)])).to_parquet(root/'securities.parquet',index=False)
    result=build_features(root)
    assert 'label_coverage' in result,'mature missing outcomes must be reported'
    assert result['label_coverage']['20']['eligible_mature_missing']>=1


def test_alias_is_applied_before_cross_source_join(tmp_path):
    from ashare_quant.dataset import save_normalized_year
    from ashare_quant.data import normalize_securities
    daily=pd.DataFrame([dict(ts_code='830001.BJ',trade_date='20220104',open=10.,high=11.,low=9.,close=10.,pre_close=10.,vol=1000.,amount=1000.)])
    basic=pd.DataFrame([dict(ts_code='920001.BJ',trade_date='20220104',total_mv=100.,total_share=10.,free_share=5.,turnover_rate=1.)])
    factors=pd.DataFrame([dict(ts_code='920001.BJ',trade_date='20220104',adj_factor=2.)])
    limits=pd.DataFrame([dict(ts_code='920001.BJ',trade_date='20220104',up_limit=13.,down_limit=7.)])
    master=normalize_securities(pd.DataFrame([dict(ts_code='920001.BJ',name='Fixture',list_date='20211115',delist_date=None)]))
    aliases=pd.DataFrame(dict(o_code=['830001.BJ'],n_code=['920001.BJ']))
    save_normalized_year(tmp_path,2022,daily,basic,factors,limits,master,aliases)
    result=pd.read_parquet(tmp_path/'bars'/'2022.parquet').iloc[0]
    assert result.data_valid and result.adj_close==20.
    assert result.total_mv==1e6
    assert result.source_security_id=='830001.BJ'


def test_equivalent_alias_rows_can_be_audited_without_double_counting():
    from ashare_quant.dataset import canonicalize_codes
    aliases=pd.DataFrame(dict(o_code=['830001.BJ'],n_code=['920001.BJ']))
    frame=pd.DataFrame(dict(security_id=['830001.BJ','920001.BJ'],date=pd.to_datetime(['20220104']*2),raw_close=[10.,10.],unused_dividend_yield=[1.,2.]))
    out=canonicalize_codes(frame,aliases,allow_equivalent=True,used_columns=['raw_close'])
    assert len(out)==1 and out.attrs['equivalent_alias_rows_removed']==1
    import pytest
    frame.loc[1,'raw_close']=11.
    with pytest.raises(ValueError,match='alias'):
        canonicalize_codes(frame,aliases,allow_equivalent=True,used_columns=['raw_close'])


def test_st_flag_aggregates_duplicate_status_types_and_aliases(tmp_path):
    from ashare_quant.dataset import save_normalized_year
    from ashare_quant.data import normalize_securities
    daily=pd.DataFrame([dict(ts_code='920001.BJ',trade_date='20220104',open=10.,high=11.,low=9.,close=10.,pre_close=10.,vol=1000.,amount=1000.)])
    master=normalize_securities(pd.DataFrame([dict(ts_code='920001.BJ',name='Fixture',list_date='20211115',delist_date=None)]))
    mapping=pd.DataFrame({'o_code':['830001.BJ'],'n_code':['920001.BJ']})
    st=pd.DataFrame({'ts_code':['830001.BJ','920001.BJ','920001.BJ'],'trade_date':['20220104']*3,'type':['ST','ST','*ST']})
    save_normalized_year(tmp_path,2022,daily,pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),master,mapping,st)
    assert bool(pd.read_parquet(tmp_path/'bars'/'2022.parquet').is_st.iloc[0])


def test_adj_factor_alias_rounding_prefers_more_precise_source_with_audit():
    from ashare_quant.dataset import canonicalize_codes
    import pytest
    mapping=pd.DataFrame({'o_code':['000022.SZ'],'n_code':['001872.SZ'],'effective_date':['2018-12-26']})
    frame=pd.DataFrame({'security_id':['000022.SZ','001872.SZ'],'date':['20180816']*2,'adj_factor':[4.189,4.1885]})
    result=canonicalize_codes(frame,mapping,allow_equivalent=True,used_columns=['adj_factor'])
    assert result.adj_factor.iloc[0]==4.1885
    assert len(result.attrs['precision_reconciliations'])==1
    frame.loc[1,'adj_factor']=4.18
    with pytest.raises(ValueError,match='conflicting'):
        canonicalize_codes(frame,mapping,allow_equivalent=True,used_columns=['adj_factor'])


def test_conflicting_alias_factors_use_explicit_verified_date_active_source():
    from ashare_quant.dataset import canonicalize_codes
    import pytest
    mapping=pd.DataFrame({'o_code':['000043.SZ'],'n_code':['001914.SZ'],'effective_date':['2019-12-16'],'verified':[True],'source_url':['https://static.cninfo.com.cn/finalpage/2019-12-16/1207164397.PDF']})
    frame=pd.DataFrame({'security_id':['000043.SZ','001914.SZ'],'date':['20190606']*2,'adj_factor':[8.333,8.334]})
    result=canonicalize_codes(frame,mapping,allow_equivalent=True,used_columns=['adj_factor'],prefer_dated_factor=True)
    assert result.adj_factor.iloc[0]==8.333
    assert result.attrs['dated_factor_choices'][0]['policy']=='documented date-active source code; alternative retained in raw audit'
    with pytest.raises(ValueError,match='conflicting'):
        canonicalize_codes(frame,mapping.assign(verified=False),allow_equivalent=True,used_columns=['adj_factor'],prefer_dated_factor=True)
    with pytest.raises(ValueError,match='conflicting'):
        canonicalize_codes(frame.rename(columns={'adj_factor':'raw_close'}),mapping,allow_equivalent=True,used_columns=['raw_close'],prefer_dated_factor=True)
