import pandas as pd
import pytest
from hk_quant.distributions import parse_history,reconcile_histories


def test_true_ex_date_and_payment_are_parsed_without_inventing_announcement_time():
    en=parse_history(pd.DataFrame([[2025,'0.3522','0.1841','20 Aug 2025','19 Sep 2025','0.1681','26 Mar 2026','24 Apr 2026']]))
    zh=parse_history(pd.DataFrame([['2025年','0.3522','0.1841','2025年8月20日','2025年9月19日','0.1681','2026年3月26日','2026年4月24日']]))
    result=reconcile_histories(en,zh,'Fortune REIT',['https://issuer/en','https://issuer/zh'])
    assert result.status.eq('matched_issuer_records').all()
    assert result.announced_at.isna().all() and not result.usable_for_prediction_features.any()
    assert result.loc[result.distribution_type.eq('interim'),'payment_date'].iloc[0]==pd.Timestamp('2025-09-19')


def test_conflicting_payment_dates_are_preserved_but_not_chosen():
    en=parse_history(pd.DataFrame([[2025,'-','0.1841','20 Aug 2025','21 Sep 2025','-','-','-']]))
    zh=en.copy();zh['payment_date']=pd.Timestamp('2025-09-19')
    result=reconcile_histories(en,zh,'Fortune REIT',['https://issuer/en','https://issuer/zh'])
    assert result.iloc[0].status=='source_conflict' and pd.isna(result.iloc[0].payment_date)
    assert result.iloc[0].english_evidence['payment_date']!=result.iloc[0].chinese_evidence['payment_date']


def test_missing_schema_or_invalid_payment_chronology_is_rejected():
    with pytest.raises(ValueError):parse_history(pd.DataFrame([[2025,1]]))
    with pytest.raises(ValueError,match='早于'):
        parse_history(pd.DataFrame([[2025,'-','1','20 Aug 2025','19 Aug 2025','-','-','-']]))
