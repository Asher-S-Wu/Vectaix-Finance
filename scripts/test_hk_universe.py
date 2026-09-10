import unittest
import numpy as np
import pandas as pd

from hk_universe import quarterly_pool, filter_candidates
from portfolio_scores import smooth_scores


class QuarterlyPoolTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range('2022-01-03', '2025-07-10')
        self.close = pd.DataFrame(10., index=self.dates, columns=['A', 'B', 'C'])
        self.amt = self.close * 2_000_000
        self.volume = self.close * 200_000
        self.members = pd.DataFrame([
            {'as_of': pd.Timestamp(d), 'code': c, 'market_cap_hkd': cap}
            for d in ['2025-03-31', '2025-06-30']
            for c, cap in [('A', 300.), ('B', 200.), ('C', 100.)]
        ])

    def pool(self, date, **kwargs):
        return quarterly_pool(self.close, self.amt, self.volume,
                              self.members, pd.Timestamp(date), **kwargs)

    def test_future_changes_do_not_change_quarter_pool(self):
        before = self.pool('2025-05-30', target_size=2)
        self.amt.loc['2025-04-01':, 'A'] = 0
        self.close.loc['2025-04-01':, 'A'] = np.nan
        pd.testing.assert_frame_equal(before, self.pool('2025-05-30', target_size=2))

    def test_liquidity_and_history_are_checked_at_quarter_boundary(self):
        self.amt.loc[:'2025-03-31', 'B'] = 9_000_000
        self.close.loc[:'2024-01-01', 'C'] = np.nan
        pool = self.pool('2025-05-30')
        self.assertEqual(pool.loc[pool.eligible, 'code'].tolist(), ['A'])
        self.assertFalse(pool.set_index('code').loc['B', 'liquidity_ok'])
        self.assertFalse(pool.set_index('code').loc['C', 'history_ok'])

    def test_missing_liquidity_is_not_filled(self):
        self.amt.loc['2025-03-20', 'A'] = np.nan
        self.assertFalse(self.pool('2025-05-30').set_index('code').loc['A', 'eligible'])

    def test_missing_quarter_is_an_error(self):
        self.members = self.members[self.members.as_of.ne(pd.Timestamp('2025-03-31'))]
        with self.assertRaisesRegex(ValueError, '季度.*2025-03-31'):
            self.pool('2025-05-30')

    def test_membership_changes_only_at_next_quarter(self):
        self.members = self.members[~(self.members.as_of.eq(pd.Timestamp('2025-06-30')) & self.members.code.eq('A'))]
        self.assertIn('A', self.pool('2025-06-30').code.tolist())
        self.assertNotIn('A', self.pool('2025-07-01').code.tolist())

    def test_missing_stock_is_reported_not_silently_dropped(self):
        self.close = self.close.drop(columns='A')
        row = self.pool('2025-05-30').set_index('code').loc['A']
        self.assertFalse(row.data_available)
        self.assertFalse(row.eligible)

    def test_signal_filter_excludes_missing_factors_and_nontrading(self):
        scores = pd.DataFrame({'date': [pd.Timestamp('2025-05-30')]*3,
                               'code': ['A', 'B', 'C'], 'factor': [1., np.nan, 2.]})
        self.volume.loc['2025-05-30', 'C'] = 0
        result = filter_candidates(scores, self.close, self.amt, self.volume,
                                   self.members, required_factors=['factor'])
        self.assertEqual(result.code.tolist(), ['A'])

    def test_reentry_does_not_use_a_stale_quarter_score(self):
        frame = pd.DataFrame({'code': ['A']*3, 'date': pd.to_datetime(['2025-01-31', '2025-04-30', '2025-05-30']),
                              'base_score': [10., 2., 4.]})
        scores = smooth_scores(frame, .5).score
        self.assertTrue(pd.isna(scores.iloc[1]))
        self.assertEqual(scores.iloc[2], 3.)


if __name__ == '__main__':
    unittest.main()
