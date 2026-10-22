import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from hk_quant.collect import collect_month, collect_references


class CollectionTests(unittest.TestCase):
    def test_overlapping_pages_are_rejected(self):
        class Client:
            def query(self, api, params, fields):
                return {'code': 0, 'source': 'fixture', 'data': {
                    'fields': fields.split(','), 'items': [['00001.HK', '20200102', 1., 10.]]}}
        with tempfile.TemporaryDirectory() as directory, patch('hk_quant.collect.PAGE_SIZE', 1):
            with self.assertRaisesRegex(ValueError, '分页重复'):
                collect_month(Client(), 'hk_adjfactor', '20200101', '20200131', Path(directory))

    def test_last_short_page_and_archive_identity_are_preserved(self):
        class Client:
            def query(self, api, params, fields):
                rows = ([['00020!AA.HK', '20200102', .8, 20.]] if params['offset'] == 0 else [])
                return {'code': 0, 'source': 'fixture', 'data': {'fields': fields.split(','), 'items': rows}}
        with tempfile.TemporaryDirectory() as directory, patch('hk_quant.collect.PAGE_SIZE', 1):
            result = collect_month(Client(), 'hk_adjfactor', '20200101', '20200131', Path(directory))
            self.assertEqual(result['rows'], 1)
            self.assertEqual(len(result['requests']), 2)

    def test_changed_start_date_recollects_month(self):
        class Client:
            def query(self, api, params, fields):
                rows = [['00001.HK', '20200120', 1., 10.]]
                if params['start_date'] == '20200101':
                    rows.insert(0, ['00001.HK', '20200102', 1., 10.])
                return {'code': 0, 'data': {'fields': fields.split(','), 'items': rows}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collect_month(Client(), 'hk_adjfactor', '20200116', '20200131', root)
            result = collect_month(Client(), 'hk_adjfactor', '20200101', '20200131', root)
            self.assertEqual(result['start_date'], '20200101')
            self.assertEqual(result['rows'], 2)

    def test_reference_business_errors_are_rejected(self):
        for failing_api in ('hk_basic', 'hk_tradecal'):
            with self.subTest(api=failing_api), tempfile.TemporaryDirectory() as directory:
                class Client:
                    def query(self, api, params, fields=None):
                        columns = ['ts_code'] if api == 'hk_basic' else fields.split(',')
                        return {'code': -1 if api == failing_api else 0,
                                'data': {'fields': columns, 'items': []}}

                with self.assertRaisesRegex(ValueError, failing_api):
                    collect_references(Client(), '20200101', '20200131', Path(directory))


if __name__ == '__main__':
    unittest.main()
