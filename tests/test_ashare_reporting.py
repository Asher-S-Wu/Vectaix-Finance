import importlib.util
import pandas as pd
import pytest


def test_benchmark_uses_identical_wealth_dates_and_price_returns():
    assert importlib.util.find_spec('ashare_quant.report') is not None,'reporting not implemented'
    from ashare_quant.report import benchmark_curves
    source=pd.DataFrame(dict(ts_code=['000300.SH']*3,trade_date=['20250102','20250103','20250106'],close=[100.,105.,90.]))
    curve,summary=benchmark_curves(source,pd.to_datetime(['2025-01-02','2025-01-03','2025-01-06']))
    assert curve['000300.SH'].tolist()==[1.,1.05,.9]
    assert summary['000300.SH']['total_return']==pytest.approx(-.1)
    assert summary['000300.SH']['basis']=='price index, excluding dividends and trading costs'


def test_missing_benchmark_day_is_not_silently_filled():
    assert importlib.util.find_spec('ashare_quant.report') is not None
    from ashare_quant.report import benchmark_curves
    source=pd.DataFrame(dict(ts_code=['000300.SH']*2,trade_date=['20250102','20250106'],close=[100.,90.]))
    curve,summary=benchmark_curves(source,pd.to_datetime(['2025-01-02','2025-01-03','2025-01-06']))
    assert pd.isna(curve['000300.SH'].iloc[1])
    assert summary['000300.SH']['status']=='incomplete' and summary['000300.SH']['missing_sessions']==1


def test_report_cannot_call_untrained_dataset_complete(tmp_path):
    assert importlib.util.find_spec('ashare_quant.report') is not None
    from ashare_quant.report import build_report
    with pytest.raises(ValueError,match='training'):
        build_report(tmp_path,tmp_path,tmp_path)
