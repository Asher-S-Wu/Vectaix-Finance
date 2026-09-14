import tempfile
import shutil
from pathlib import Path
import unittest

import pandas as pd

from hk_quant.registry import publish_snapshot
from hk_quant.prediction_tasks import TASK_STATUS_COLUMNS,PREDICTION_SCHEMA
from hk_quant.contracts import FORECAST_RETURN_BASIS


class PublicationTests(unittest.TestCase):
    def forecasts(self, path, version='candidate'):
        pd.DataFrame({'date':pd.to_datetime(['2026-09-10']*4),'security_id':['A']*4,
                      'horizon':[1,5,20,60],'model_version':[version]*4,
                      'data_as_of':['2026-09-10']*4,'status':['ok']*4,'score':[.1]*4,
                      'probability_up':[.6]*4,'expected_return':[.02]*4,
                      'q10':[-.1]*4,'q50':[.01]*4,'q90':[.12]*4,
                      'baseline_probability':.5,'baseline_q10':-.2,'baseline_q50':0.,'baseline_q90':.2,
                      'prediction_schema':PREDICTION_SCHEMA,
                      'reasons':'','explanations':[[] for _ in range(4)],'return_basis':FORECAST_RETURN_BASIS,
                      **{field:'ok' for field in TASK_STATUS_COLUMNS.values()}}).to_parquet(path,index=False)

    def test_old_output_schema_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'forecast.parquet';self.forecasts(path)
            frame=pd.read_parquet(path).drop(columns='score_status')
            frame.to_parquet(path,index=False)
            with self.assertRaisesRegex(ValueError,'字段|任务'):
                publish_snapshot(path,'candidate','2026-09-10',{'eligible':True},
                                 {'critical_gap_count':0,'freshness_passed':True},root)
            self.assertFalse((root/'active.json').exists())

    def test_unqualified_model_never_creates_active_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=publish_snapshot(root/'missing.parquet','candidate','2026-09-10',{'eligible':False},
                                    {'critical_gap_count':0,'freshness_passed':True},root)
            self.assertFalse(result['eligible'])
            self.assertFalse((root/'active.json').exists())

    def test_snapshot_version_is_bound_to_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'forecast.parquet';self.forecasts(path,'wrong')
            with self.assertRaisesRegex(ValueError,'版本或日期'):
                publish_snapshot(path,'candidate','2026-09-10',{'eligible':True},
                                 {'critical_gap_count':0,'freshness_passed':True},root)
            self.assertFalse((root/'active.json').exists())

    def test_complete_approved_snapshot_can_be_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'forecast.parquet';self.forecasts(path)
            active=publish_snapshot(path,'candidate','2026-09-10',{'eligible':True},
                                    {'critical_gap_count':0,'freshness_passed':True},root)
            self.assertTrue(active['eligible'])
            self.assertFalse(Path(active['acceptance_path']).is_absolute())
            self.assertTrue((root/active['acceptance_path']).is_file())

    def test_publication_survives_moving_to_another_server_directory(self):
        from hk_quant.service import HKQuantService
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/'training';root.mkdir()
            path=base/'forecast.parquet';self.forecasts(path)
            active=publish_snapshot(path,'candidate','2026-09-10',{'eligible':True},
                                    {'critical_gap_count':0,'freshness_passed':True},root)
            deployed=base/'deployed';shutil.copytree(root,deployed)
            path.unlink()
            self.assertEqual(HKQuantService(data=base/'unused',models=deployed).status()['model_version'],'candidate')
            self.assertTrue((deployed/active['forecast_path']).is_file())


if __name__=='__main__':unittest.main()
