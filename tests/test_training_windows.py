import unittest
import tempfile
from pathlib import Path
import json

import numpy as np
import pandas as pd

from hk_quant.training import cross_sectional_inputs, split_window, initialize_run, dataset_lineage, complete_market_snapshot
from hk_quant.prediction_tasks import TASK_STATUS_COLUMNS,PREDICTION_SCHEMA,validate_task_outputs


class WindowTests(unittest.TestCase):
    def test_snapshot_retains_unscored_reit_with_all_horizons(self):
        sec = pd.DataFrame({'security_id':['equity','reit','retired'],'exchange_code':['1','2','3'],
            'name':['A','B','C'],'isin':['I1','I2','I3'],'currency':'HKD','asset_type':['equity','reit','equity'],
            'identity_status':'verified','list_date':pd.to_datetime(['2010-01-01']*3),
            'delist_date':pd.to_datetime([None,None,'2020-01-01'])})
        forecasts = pd.DataFrame({'date':pd.to_datetime(['2026-09-10']*4),'security_id':'equity',
            'horizon':[1,5,20,60],'status':'ok','q10':-.1,'q50':0.,'q90':.1,
            'score':.2,'probability_up':.6,'expected_return':.02,'baseline_probability':.5,
            'baseline_q10':-.2,'baseline_q50':0.,'baseline_q90':.2,
            'prediction_schema':PREDICTION_SCHEMA,**{field:'ok' for field in TASK_STATUS_COLUMNS.values()}})
        result = complete_market_snapshot(forecasts,sec,'2026-09-10','v3','2026-08-31')
        missing = result[result.security_id.eq('reit')]
        self.assertEqual(set(missing.horizon),{1,5,20,60})
        self.assertTrue(missing.status.eq('insufficient_model_inputs').all())
        self.assertTrue(missing[list(TASK_STATUS_COLUMNS.values())].eq('insufficient_model_inputs').all().all())
        self.assertTrue(missing.prediction_schema.eq(PREDICTION_SCHEMA).all())
        self.assertTrue(missing[['q10','q50','q90']].isna().all().all())
        self.assertNotIn('retired',set(result.security_id))
        validate_task_outputs(result)
        partial=complete_market_snapshot(forecasts.iloc[:3],sec,'2026-09-10','v3','2026-08-31')
        self.assertTrue(partial.groupby('security_id').horizon.apply(lambda values:set(values)=={1,5,20,60}).all())
        last=partial.loc[partial.security_id.eq('equity') & partial.horizon.eq(60)].iloc[0]
        self.assertEqual(last.status,'insufficient_model_inputs')
        self.assertTrue(pd.isna(last.score))

    def test_changed_feature_generation_cannot_reuse_normalized_or_training_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'build_status.json').write_text(json.dumps({'status':'complete'}))
            manifest = root/'feature_manifest.json'
            manifest.write_text(json.dumps({'data_as_of':'20260910','rows':20,'features':['x']}))
            normalized = root/'normalization_manifest.json'
            normalized.write_text(json.dumps(dataset_lineage(root)))
            initialize_run(root,root/'results')
            manifest.write_text(json.dumps({'data_as_of':'20260910','rows':21,'features':['x']}))
            with self.assertRaisesRegex(ValueError,'标准化'):
                initialize_run(root,root/'results')
            normalized.write_text(json.dumps(dataset_lineage(root)))
            with self.assertRaisesRegex(ValueError,'独立结果目录'):
                initialize_run(root,root/'results')

    def test_normalization_does_not_use_future_return_availability(self):
        frame = pd.DataFrame({'date': pd.to_datetime(['2020-01-02']*3), 'security_id':['A','B','C'],
                              'status':['ok']*3, 'volatility_20':[.1,.2,.3], 'factor':[1.,2.,3.],
                              'market_momentum_60':[.2]*3, 'fwd_return_20':[np.nan,.1,.2]})
        result = cross_sectional_inputs(frame, ['factor','market_momentum_60'])
        np.testing.assert_allclose(result.factor,[1/3,2/3,1.])
        np.testing.assert_allclose(result.market_momentum_60,[.2]*3)
        np.testing.assert_allclose(result.sigma_daily,np.array([.1,.2,.3])/np.sqrt(252))

    def test_every_horizon_is_purged_at_training_validation_boundary(self):
        dates = pd.bdate_range('2017-01-02','2020-12-31')
        frame = pd.DataFrame({'date':dates,'security_id':'A','status':'ok','sigma_daily':.01,'factor':1.})
        for h in (1,5,20,60):
            frame[f'fwd_return_{h}']=.01
            frame[f'label_end_{h}']=pd.Series(dates).shift(-h)
        tr,cal=split_window(frame,['factor'],'2020-12-31')
        self.assertTrue(tr.label_end.lt(pd.Timestamp('2019-12-31')).all())
        self.assertTrue(cal.label_end.le(pd.Timestamp('2020-12-31')).all())
        self.assertEqual(set(tr.horizon),(set((1,5,20,60))))


if __name__ == '__main__':
    unittest.main()
