import unittest

import numpy as np
import pandas as pd

from hk_quant.features import security_features, offer_features, reit_features_asof, reit_forward_labels


class FeatureTests(unittest.TestCase):
    def test_nullable_trade_status_preserves_missing_day_without_crashing(self):
        bars=self.bars();calendar=pd.DatetimeIndex(bars.date)
        bars['quote_present']=bars.quote_present.astype('boolean')
        bars['data_valid']=bars.data_valid.astype('boolean')
        result=security_features(bars.drop(index=70),calendar)
        self.assertEqual(result.loc[70,'status'],'data_issue')
        self.assertTrue(pd.isna(result.loc[70,'raw_close']))
        self.assertTrue(pd.isna(result.loc[71,'return_1']))
        self.assertEqual(result.loc[71,'history_sessions'],71)

    def test_reit_labels_use_each_horizon_public_information(self):
        bars=self.bars()
        ex=bars.date.iloc[103]
        original=dict(security_id='A',event_id='d',event_type='cash_distribution',
            published_at=bars.date.iloc[99].tz_localize('Asia/Hong_Kong'),ex_date=ex,
            cash_per_unit_decimal='1',cash_currency='HKD',conditional_cash_amount=True,
            cash_amount_status='conditional_pending_units',announcement_status='New')
        revision={**original,'published_at':bars.date.iloc[108].tz_localize('Asia/Hong_Kong'),
                  'cash_per_unit_decimal':'2','conditional_cash_amount':False,
                  'cash_amount_status':'declared_payment_currency_amount'}
        versions=pd.DataFrame([original,revision])
        signal=bars.date.iloc[100]
        early=reit_forward_labels(bars,pd.DatetimeIndex(bars.date),versions,signal,
             bars.date.iloc[101].tz_localize('Asia/Hong_Kong')+pd.Timedelta(hours=19),'HKD')
        later=reit_forward_labels(bars,pd.DatetimeIndex(bars.date),versions,signal,
             bars.date.iloc[-1].tz_localize('Asia/Hong_Kong')+pd.Timedelta(hours=19),'HKD')
        self.assertAlmostEqual(early.fwd_return_1.iloc[0],201/200-1)
        self.assertTrue(pd.isna(early.fwd_return_20.iloc[0]))
        self.assertTrue(pd.isna(later.fwd_return_5.iloc[0]))
        self.assertAlmostEqual(later.fwd_return_20.iloc[0],220/(200*(202-2)/202)-1)
        self.assertTrue(pd.isna(later.fwd_return_60.iloc[0]))
        self.assertEqual(later.label_end_20.iloc[0],bars.date.iloc[120])

    def test_old_unresolved_distribution_does_not_block_unaffected_recent_window(self):
        bars=pd.concat([self.bars()]*3,ignore_index=True)
        bars['date']=pd.bdate_range('2020-01-01',periods=len(bars))
        for column in ['raw_close','adj_close','adj_close_hkd','open_adj','high_adj','low_adj']:
            bars[column]=np.arange(100.,100.+len(bars))
        bars['raw_open']=bars.open_adj;bars['high']=bars.high_adj;bars['low']=bars.low_adj
        versions=pd.DataFrame([dict(security_id='A',event_id='old_dividend',event_type='cash_distribution',
            published_at=bars.date.iloc[1].tz_localize('Asia/Hong_Kong'),ex_date=bars.date.iloc[10],
            cash_per_unit_decimal='1',cash_currency='HKD',conditional_cash_amount=True,
            cash_amount_status='conditional_pending_units',announcement_status='New')])
        cutoff=bars.date.iloc[-1].tz_localize('Asia/Hong_Kong')+pd.Timedelta(hours=19)
        current=reit_features_asof(bars,pd.DatetimeIndex(bars.date),versions,cutoff,'HKD')
        self.assertEqual(current.status.iloc[0],'ok')
        self.assertAlmostEqual(current.momentum_252.iloc[0],579/327-1)
        self.assertTrue(current.attrs['adjustment_gaps'])
        self.assertFalse(current.attrs['coverage_complete'])

    def test_reit_snapshot_uses_public_version_and_never_future_labels(self):
        bars=self.bars()
        bars['raw_open']=bars.open_adj;bars['high']=bars.high_adj;bars['low']=bars.low_adj
        ex=bars.date.iloc[130]
        versions=pd.DataFrame([dict(security_id='A',event_id='dividend',event_type='cash_distribution',
            published_at=(bars.date.iloc[120].tz_localize('Asia/Hong_Kong')+pd.Timedelta(hours=18)),
            ex_date=ex,cash_per_unit_decimal='1',cash_currency='HKD',conditional_cash_amount=False,
            cash_amount_status='declared_payment_currency_amount',announcement_status='New')])
        cutoff=bars.date.iloc[135].tz_localize('Asia/Hong_Kong')+pd.Timedelta(hours=19)
        first=reit_features_asof(bars,pd.DatetimeIndex(bars.date),versions,cutoff,'HKD')
        future=versions.iloc[[0]].copy();future['published_at']=cutoff+pd.Timedelta(days=1);future['cash_per_unit_decimal']='2'
        changed=bars.copy();changed.loc[changed.date.gt(cutoff.tz_localize(None)),['raw_close','raw_open','high','low']]=9999.
        second=reit_features_asof(changed,pd.DatetimeIndex(bars.date),pd.concat([versions,future]),cutoff,'HKD')
        pd.testing.assert_frame_equal(first,second)
        self.assertEqual(len(first),1)
        self.assertFalse(any(c.startswith(('fwd_return_','label_end_')) for c in first))
        self.assertAlmostEqual(first.momentum_20.iloc[0],235/(215*(229-1)/229)-1)
        self.assertFalse(first.attrs['approved_for_training'])

    def test_resumption_jump_uses_observed_reference_value_without_inventing_trade(self):
        bars=self.bars()
        bars.loc[70,['raw_close','adj_close','adj_close_hkd']]=169.
        bars.loc[70,'quote_present']=False
        bars.loc[70,['volume','amount','amount_hkd']]=0.
        bars.loc[71,['raw_close','adj_close','adj_close_hkd']]=84.5
        out=security_features(bars,pd.DatetimeIndex(bars.date))
        self.assertEqual(out.loc[70,'status'],'no_trade_quote')
        self.assertAlmostEqual(out.loc[70,'return_1'],0.)
        self.assertAlmostEqual(out.loc[71,'return_1'],-.5)
        self.assertGreater(out.loc[71,'volatility_20'],out.loc[69,'volatility_20'])
        bars.loc[70,['raw_close','adj_close','adj_close_hkd']]=np.nan
        missing=security_features(bars,pd.DatetimeIndex(bars.date))
        self.assertTrue(pd.isna(missing.loc[71,'return_1']))

    def test_event_coverage_absence_is_not_encoded_as_no_event(self):
        frame = pd.DataFrame({'date':pd.to_datetime(['2015-01-02','2018-01-02'])})
        offers = pd.DataFrame({'code':pd.Series(dtype='str'),'offer_start':pd.Series(dtype='datetime64[ns]'),
                               'offer_end':pd.Series(dtype='datetime64[ns]')})
        result = offer_features(frame, offers, '00001.HK')
        self.assertTrue(pd.isna(result.offer_active.iloc[0]))
        self.assertEqual(result.offer_active.iloc[1], 0.)

    def bars(self):
        dates = pd.bdate_range('2020-01-01', periods=160)
        price = np.arange(100., 260.)
        return pd.DataFrame({'date': dates, 'security_id': ['A']*len(dates),
                             'adj_close_hkd': price, 'adj_close': price, 'raw_close': price,
                             'quote_present': True, 'data_valid': True, 'fx_to_hkd': 1.,
                             'volume': 10000., 'amount': price*10000, 'amount_hkd': price*10000,
                             'open_adj': price-.5, 'high_adj': price+1, 'low_adj': price-1,
                             'total_mv': price*1e6, 'free_mv': price*5e5,
                             'total_share': 1e6, 'free_share': 5e5, 'turnover_ratio': .1})

    def test_each_horizon_uses_its_actual_market_label_end(self):
        bars = self.bars()
        result = security_features(bars, pd.DatetimeIndex(bars.date))
        row = result.iloc[70]
        for horizon in (1, 5, 20, 60):
            self.assertEqual(row[f'label_end_{horizon}'], bars.date.iloc[70+horizon])
            self.assertAlmostEqual(row[f'fwd_return_{horizon}'], (170+horizon)/170-1)
        self.assertTrue(pd.isna(result.iloc[-1].fwd_return_1))

    def test_future_price_change_cannot_change_earlier_factors(self):
        bars = self.bars()
        before = security_features(bars, pd.DatetimeIndex(bars.date))
        bars.loc[120:, ['adj_close_hkd','adj_close','raw_close']] *= 2
        after = security_features(bars, pd.DatetimeIndex(bars.date))
        cols = [c for c in before if not c.startswith(('fwd_', 'label_end_'))]
        pd.testing.assert_frame_equal(before.loc[:119, cols], after.loc[:119, cols])

    def test_missing_market_day_is_not_skipped_or_filled(self):
        bars = self.bars()
        dates = pd.DatetimeIndex(bars.date)
        result = security_features(bars.drop(index=71), dates)
        self.assertTrue(pd.isna(result.iloc[70].fwd_return_1))
        self.assertEqual(result.iloc[70].label_end_1, dates[71])
        self.assertEqual(result.iloc[71].status, 'data_issue')


if __name__ == '__main__':
    unittest.main()


def test_earlier_unverified_prices_do_not_seed_current_features_or_labels():
    from hk_quant.features import identity_period_features
    bars=FeatureTests().bars()
    metadata=pd.Series({'security_id':'A','identity_status':'verified','identity_valid_from':bars.date.iloc[80],'identity_valid_to':pd.NaT})
    before=identity_period_features(bars,pd.DatetimeIndex(bars.date),metadata)
    changed=bars.copy();changed.loc[:79,['raw_close','adj_close','adj_close_hkd','amount_hkd','total_share']]*=1000
    after=identity_period_features(changed,pd.DatetimeIndex(bars.date),metadata)
    pd.testing.assert_frame_equal(before,after)
    assert len(before)==len(bars)
    assert before.loc[:79,'status'].eq('identity_period_unresolved').all()
    assert before.loc[:79,['fwd_return_1','fwd_return_5','fwd_return_20','fwd_return_60']].isna().all().all()
    assert before.history_sessions.iloc[80]==1
    assert pd.isna(before.momentum_60.iloc[80])
    assert before.status.iloc[138]=='insufficient_history'
    assert before.status.iloc[139]=='ok'
    assert np.isclose(before.momentum_60.iloc[140],240/180-1)


def test_after_identity_end_cannot_enter_last_valid_return_label():
    from hk_quant.features import identity_period_features
    bars=FeatureTests().bars()
    metadata=pd.Series({'security_id':'A','identity_status':'verified','identity_valid_from':bars.date.iloc[0],'identity_valid_to':bars.date.iloc[100]})
    result=identity_period_features(bars,pd.DatetimeIndex(bars.date),metadata)
    assert pd.isna(result.fwd_return_1.iloc[99])
    assert result.loc[100:,'status'].eq('identity_period_unresolved').all()
    assert result.loc[100:,'raw_close'].isna().all()


def test_unresolved_entities_are_retained_but_cannot_pollute_market_history():
    from hk_quant.features import isolate_identity_periods
    from hk_quant.market_context import market_context_from_inputs
    bars=FeatureTests().bars()
    master=pd.DataFrame([{'security_id':'A','identity_status':'unresolved','identity_valid_from':pd.NaT,'identity_valid_to':pd.NaT}]).set_index('security_id')
    isolated=isolate_identity_periods(bars,master)
    assert len(isolated)==len(bars)
    assert isolated.adj_close_hkd.isna().all() and isolated.quote_present.eq(False).all()
    inputs=security_features(isolated,pd.DatetimeIndex(bars.date))
    inputs['quote_present']=isolated.quote_present.to_numpy()
    market=market_context_from_inputs(inputs,pd.DatetimeIndex(bars.date))
    assert market.market_momentum_60.isna().all()
