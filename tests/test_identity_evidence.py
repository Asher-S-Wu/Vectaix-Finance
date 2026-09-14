import pandas as pd
from hk_quant.identity_evidence import compare_identity_evidence


def data():
    master=pd.DataFrame([{'security_id':'00096.HK','isin':'KYG9883K1013','asset_type':'equity'}])
    current=pd.DataFrame([{'Stock Code':'00096','ISIN':'KYG9883K1013','Category':'Equity','Sub-Category':'Ordinary Shares'}])
    issues=pd.DataFrame([{'ID1':253,'issuer':10,'typeID':0},{'ID1':4650,'issuer':20,'typeID':0}])
    listings=pd.DataFrame([{'stockExID':1,'shortName':'HK Main','longName':'SEHK main board'},{'stockExID':20,'shortName':'HK GEM','longName':'SEHK Growth Enterprise Market'},{'stockExID':38,'shortName':'HKCIS','longName':'SEHK Collective Investment Schemes'}])
    types=pd.DataFrame([{'typeID':0,'typeShort':'O','typeLong':'Ordinary shares'},{'typeID':5,'typeShort':'Pref','typeLong':'Preference'},{'typeID':10,'typeShort':'Unit','typeLong':'Unit'}])
    periods=pd.DataFrame([
        {'ID':1,'IssueID':253,'StockCode':'0096','StockExID':1,'FirstTradeDate':'1980-03-27','FinalTradeDate':'2008-10-27','DelistDate':'2009-01-16','isin':None,'2ndCtr':0},
        {'ID':2,'IssueID':4650,'StockCode':'8319','StockExID':20,'FirstTradeDate':'2005-10-13','FinalTradeDate':'2010-12-14','DelistDate':'2010-12-15','isin':None,'2ndCtr':0},
        {'ID':3,'IssueID':4650,'StockCode':'0096','StockExID':1,'FirstTradeDate':'2010-12-15','FinalTradeDate':None,'DelistDate':None,'isin':'KYG9883K1013','2ndCtr':0}])
    return master,current,issues,periods,types,listings


def test_unique_isin_excludes_reused_code_and_preserves_board_transfer():
    mapping,periods,audit=compare_identity_evidence(*data())
    row=mapping.iloc[0]
    assert row.mapping_status=='matched_unique_isin' and row.IssueID==4650
    assert row.same_code_other_issue_ids==[253]
    assert set(periods.IssueID)=={4650} and periods.StockCode.tolist()==['8319','0096']
    assert len(periods)==2 and not audit['all_historical_identity_verified']
    assert audit['transitions'][0]['relation']=='matching_transfer_boundary'
    assert audit['early_quote_attribution_verified'] is False


def test_noncontiguous_periods_are_not_filled_by_min_max():
    args=list(data());args[3].loc[2,'FirstTradeDate']='2011-01-03'
    mapping,periods,audit=compare_identity_evidence(*args)
    assert len(periods)==2
    transition=audit['transitions'][0]
    assert transition['relation']=='gap'
    assert transition['left_end']=='2010-12-15' and transition['right_start']=='2011-01-03'


def test_missing_or_ambiguous_isin_does_not_match_by_code_or_name():
    args=list(data());args[0]['isin']=None;args[1]['ISIN']=None
    mapping,periods,_=compare_identity_evidence(*args)
    assert mapping.iloc[0].mapping_status=='missing_isin' and periods.empty
    args=list(data());args[3].loc[0,'isin']='KYG9883K1013'
    mapping,periods,_=compare_identity_evidence(*args)
    assert mapping.iloc[0].mapping_status=='ambiguous_isin' and periods.empty
    assert mapping.iloc[0].candidate_issue_ids==[253,4650]


def test_current_hkex_isin_conflict_is_explicit():
    args=list(data());args[1]['ISIN']='OTHER'
    mapping,periods,_=compare_identity_evidence(*args)
    assert mapping.iloc[0].mapping_status=='isin_conflict' and periods.empty


def test_current_reit_on_board_38_is_identified_by_official_isin():
    args=list(data());args[0]['security_id']='01503.HK';args[1]['Stock Code']='01503';args[1]['Category']='Real Estate Investment Trusts'
    args[2].loc[1,'typeID']=10;args[3].loc[2,['StockCode','StockExID']]=['1503',38]
    mapping,_,_=compare_identity_evidence(*args)
    assert mapping.iloc[0].asset_class=='reit' and mapping.iloc[0].source_typeID==10


def test_preference_share_does_not_inherit_hkex_equity_as_ordinary():
    args=list(data());args[2].loc[1,'typeID']=5
    mapping,_,_=compare_identity_evidence(*args)
    assert mapping.iloc[0].asset_class=='preference'
    assert not mapping.iloc[0].is_ordinary_share


def test_named_ordinary_share_classes_are_not_unknown_assets():
    for short in ['C ord','X ord','Y ord','Z ord']:
        args=list(data());args[2].loc[1,'typeID']=51
        args[4]=pd.concat([args[4],pd.DataFrame([{'typeID':51,'typeShort':short,'typeLong':short.replace('ord','ordinary')}])],ignore_index=True)
        mapping,_,_=compare_identity_evidence(*args)
        assert mapping.iloc[0].asset_class=='ordinary_share'


def test_archive_id_never_takes_current_code_isin():
    args=list(data());args[0]['security_id']='00096!AA.HK';args[0]['isin']=None
    mapping,periods,_=compare_identity_evidence(*args)
    row=mapping[mapping.security_id.eq('00096!AA.HK')].iloc[0]
    assert row.mapping_status=='missing_isin'


def test_missing_evidence_serializes_as_json_null_not_nan():
    import json
    args=list(data())
    args[0]=pd.concat([args[0],pd.DataFrame([{'security_id':'00096!AA.HK','isin':None,'asset_type':'unresolved'}])],ignore_index=True)
    _,_,audit=compare_identity_evidence(*args)
    encoded=json.dumps(audit,allow_nan=False)
    assert json.loads(encoded)['gap_rows'][0]['v4_isin'] is None
