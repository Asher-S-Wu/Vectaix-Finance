import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from hk_quant.collect import FIELDS, collect_month


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


if __name__ == '__main__':
    unittest.main()
