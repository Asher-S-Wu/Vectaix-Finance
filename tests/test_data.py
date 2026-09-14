import unittest

import pandas as pd

from hk_quant.data import join_market_sources, financial_asof, security_master


class MarketDataTests(unittest.TestCase):
    def test_rmb_and_cny_are_same_currency_not_identity_conflict(self):
        basic=pd.DataFrame([dict(ts_code='87001.HK',name='REIT',isin='HK0000078516',
            list_date='20110429',delist_date=None,list_status='L',curr_type='CNY')])
        observed=pd.DataFrame([dict(security_id='87001.HK',first_quote=pd.Timestamp('2020-01-02'),last_quote=pd.Timestamp('2026-09-10'))])
        hkex=pd.DataFrame([{'Stock Code':'87001','Name of Securities':'REIT','ISIN':'HK0000078516',
            'Trading Currency':'RMB','Board Lot':'1,000','Category':'Real Estate Investment Trusts'}])
        result=security_master(basic,observed,hkex,'2026-09-11').iloc[0]
        self.assertEqual(result.identity_status,'verified')
        self.assertEqual(result.currency,'CNY')
        self.assertEqual(result.currency_source_label,'RMB')

    def test_vwap_uses_turnover_and_volume_not_rounded_display_price(self):
        date = pd.Timestamp('2020-01-02')
        quote = pd.DataFrame({'ts_code': ['00022.HK'], 'trade_date': [date],
                              'close': [.13], 'open': [.13], 'high': [.14], 'low': [.12],
                              'vol': [1020000.], 'amount': [129200.], 'vwap': [.13],
                              'adj_factor': [1.], 'turnover_ratio': [.1], 'free_share': [1e7],
                              'total_share': [1e7], 'free_mv': [1.3e6], 'total_mv': [1.3e6]})
        factor = pd.DataFrame({'ts_code': ['00022.HK'], 'trade_date': [date],
                               'cum_adjfactor': [1.], 'close_price': [.13]})
        result, audit = join_market_sources(quote, factor)
        self.assertAlmostEqual(result.vwap.iloc[0], 129200/1020000)
        self.assertEqual(audit['vwap_unit_mismatch_rows'], 0)
        self.assertTrue(result.data_valid.iloc[0])

    def test_reused_exchange_code_does_not_merge_two_issuers(self):
        basic = pd.DataFrame({'ts_code': ['00020!.HK','00020.HK'], 'name': ['OLD','NEW'],
                              'isin': ['OLD_ISIN','NEW_ISIN'], 'list_date': ['19830103','20211230'],
                              'delist_date': ['20200727',None], 'list_status': ['D','L'],
                              'curr_type': ['HKD','HKD']})
        observed = pd.DataFrame({'security_id': ['00020!AA.HK','00020.HK'],
                                 'first_quote': pd.to_datetime(['2010-01-04','2021-12-30']),
                                 'last_quote': pd.to_datetime(['2020-07-24','2026-09-09'])})
        current = pd.DataFrame({'Stock Code': ['00020'], 'Name of Securities': ['NEW'],
                                'ISIN': ['NEW_ISIN'], 'Trading Currency': ['HKD'],
                                'Board Lot': ['1,000'], 'Category': ['Equity']})
        result = security_master(basic, observed, current, '2026-09-11').set_index('security_id')
        self.assertEqual(result.loc['00020!AA.HK', 'isin'], 'OLD_ISIN')
        self.assertEqual(result.loc['00020.HK', 'isin'], 'NEW_ISIN')
        self.assertEqual(result.loc['00020.HK', 'lot_size'], 1000.)
        self.assertTrue(pd.isna(result.loc['00020!AA.HK', 'lot_valid_from']))

    def test_reference_price_does_not_create_a_trade(self):
        date = pd.Timestamp('2026-09-01')
        quote = pd.DataFrame(columns=['ts_code', 'trade_date', 'close', 'open', 'high', 'low',
                                      'vol', 'amount', 'vwap', 'adj_factor', 'turnover_ratio',
                                      'free_share', 'total_share', 'free_mv', 'total_mv'])
        factor = pd.DataFrame({'ts_code': ['02252.HK'], 'trade_date': [date],
                               'cum_adjfactor': [1.], 'close_price': [21.18]})
        result, audit = join_market_sources(quote, factor)
        self.assertFalse(result.quote_present.iloc[0])
        self.assertAlmostEqual(result.raw_close.iloc[0], 21.18)
        self.assertTrue(pd.isna(result.volume.iloc[0]))
        self.assertEqual(audit['reference_only_rows'], 1)

    def test_adjustment_uses_raw_price_once_and_preserves_identity(self):
        date = pd.Timestamp('2020-01-02')
        quote = pd.DataFrame({'ts_code': ['00020!AA.HK'], 'trade_date': [date],
                              'close': [50.], 'open': [49.], 'high': [51.], 'low': [48.],
                              'vol': [1000.], 'amount': [100000.], 'vwap': [100.],
                              'adj_factor': [.5], 'turnover_ratio': [.1], 'free_share': [1e6],
                              'total_share': [1e6], 'free_mv': [1e8], 'total_mv': [1e8]})
        factor = pd.DataFrame({'ts_code': ['00020!AA.HK'], 'trade_date': [date],
                               'cum_adjfactor': [.5], 'close_price': [100.]})
        result, _ = join_market_sources(quote, factor)
        self.assertEqual(result.security_id.iloc[0], '00020!AA.HK')
        self.assertEqual(result.adj_close.iloc[0], 50.)

    def test_future_financial_revision_is_not_visible(self):
        records = pd.DataFrame({'security_id': ['A']*3, 'metric': ['roe']*3,
                                'period_end': pd.to_datetime(['2023-12-31','2023-12-31','2024-06-30']),
                                'published_at': pd.to_datetime(['2024-03-20T18:00:00+08:00','2025-03-20T18:00:00+08:00','2024-08-20T18:00:00+08:00']),
                                'value': [10., 99., 12.], 'verified': [True]*3,
                                'version': ['1','2','1'], 'source_url': ['https://issuer.test/a']*3})
        dates = pd.DataFrame({'date': pd.to_datetime(['2024-03-19','2024-03-21','2025-03-21']),
                              'security_id': ['A']*3})
        result = financial_asof(dates, records)
        self.assertTrue(pd.isna(result.roe.iloc[0]))
        self.assertEqual(result.roe.iloc[1], 10.)
        self.assertEqual(result.roe.iloc[2], 12.)

    def test_unverified_financial_values_are_not_used(self):
        records = pd.DataFrame({'security_id': ['A'], 'metric': ['roe'],
                                'period_end': [pd.Timestamp('2023-12-31')],
                                'published_at': [pd.Timestamp('2024-03-20')], 'value': [99.],
                                'verified': [False], 'version': ['1'], 'source_url': ['']})
        dates = pd.DataFrame({'date': [pd.Timestamp('2024-04-01')], 'security_id': ['A']})
        self.assertTrue(pd.isna(financial_asof(dates, records).roe.iloc[0]))


if __name__ == '__main__':
    unittest.main()


def test_current_transfer_listing_verified_despite_earlier_source_history():
    basic=pd.DataFrame({'ts_code':['08319!.HK','00096.HK'],'name':['OLD BOARD','CURRENT BOARD'],'isin':['KYG9883K1013']*2,'list_date':['20051013','20101215'],'delist_date':['20101215',None],'list_status':['D','L'],'curr_type':['HKD']*2})
    observed=pd.DataFrame({'security_id':['00096.HK'],'first_quote':pd.to_datetime(['20100118']),'last_quote':pd.to_datetime(['20260909'])})
    hkex=pd.DataFrame({'Stock Code':['00096'],'Name of Securities':['CURRENT BOARD'],'ISIN':['KYG9883K1013'],'Trading Currency':['HKD'],'Board Lot':['2000'],'Category':['Equity']})
    result=security_master(basic,observed,hkex,'2026-09-11').set_index('security_id').loc['00096.HK']
    assert result.identity_status=='verified'
    assert result.list_date==pd.Timestamp('2010-12-15')
    assert result.identity_valid_from==pd.Timestamp('2010-12-15')
    assert result.identity_period_status=='source_outside_verified_period'
    assert result.source_first_quote==pd.Timestamp('2010-01-18')
    assert result.lot_valid_from==pd.Timestamp('2026-09-11')


def test_early_reused_code_history_never_changes_current_isin_or_start():
    basic=pd.DataFrame({'ts_code':['08321!.HK','00048.HK'],'name':['SAME NAME']*2,'isin':['OLD_ISIN','NEW_ISIN'],'list_date':['20030101','20140825'],'delist_date':['20140825',None],'list_status':['D','L'],'curr_type':['HKD']*2})
    observed=pd.DataFrame({'security_id':['00048.HK','99999.HK'],'first_quote':pd.to_datetime(['20100929','20200101']),'last_quote':pd.to_datetime(['20260909','20260909'])})
    hkex=pd.DataFrame({'Stock Code':['00048'],'Name of Securities':['SAME NAME'],'ISIN':['NEW_ISIN'],'Trading Currency':['HKD'],'Board Lot':['1000'],'Category':['Equity']})
    result=security_master(basic,observed,hkex,'2026-09-11').set_index('security_id')
    assert result.loc['00048.HK','identity_status']=='verified'
    assert result.loc['00048.HK','isin']=='NEW_ISIN'
    assert result.loc['00048.HK','identity_valid_from']==pd.Timestamp('2014-08-25')
    assert result.loc['99999.HK','identity_status']=='unresolved'
    assert set(result.index)=={'00048.HK','99999.HK'}


def test_current_basic_and_hkex_identity_conflict_cannot_verify_history():
    basic=pd.DataFrame({'ts_code':['00048.HK'],'name':['SAME NAME'],'isin':['OLD_ISIN'],'list_date':['20140825'],'delist_date':[None],'list_status':['L'],'curr_type':['HKD']})
    observed=pd.DataFrame({'security_id':['00048.HK'],'first_quote':pd.to_datetime(['20140825']),'last_quote':pd.to_datetime(['20260909'])})
    hkex=pd.DataFrame({'Stock Code':['00048'],'Name of Securities':['SAME NAME'],'ISIN':['NEW_ISIN'],'Trading Currency':['HKD'],'Board Lot':['1000'],'Category':['Equity']})
    result=security_master(basic,observed,hkex,'2026-09-11').iloc[0]
    assert result.identity_status=='unresolved'
