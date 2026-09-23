import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent_real import RealForecast


class RealAdapterTests(unittest.TestCase):
    def provider(self):
        df = pd.DataFrame({'target_time': ['2026-01-30 23:00', '2026-01-31 00:00'],
                           'ens_ws100': [6., 30.], 'power_clean': [.2, 1.], 'day_offset': [1, 1]})
        with patch('agent_real.pd.read_parquet', return_value=df), \
             patch('agent_real.Path.read_bytes', return_value=b'train'), \
             patch('model.baseline_nwp.fit_curves', return_value={'all': {'ws': [0, 10], 'p': [0, 1]}}) as fit:
            provider = RealForecast('baseline', '2026-01-31T00:00Z')
        self.assertEqual(len(fit.call_args.args[0]), 1)
        self.assertEqual(provider.provenance['training_available_through'], '2026-01-31T00:00:00+00:00')
        return provider

    def test_training_excludes_future_hour(self):
        self.provider()

    def test_missing_wind_rejected_before_model(self):
        provider = self.provider()
        with patch('agent_real.fetch_forecast'), patch('agent_real.check_no_leakage'), \
             patch('agent_real.add_features', return_value=pd.DataFrame({'ens_ws100': [np.nan]})):
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                provider('2026-01-31T00:00Z')

    def test_weather_leakage_is_not_silenced(self):
        provider = self.provider()
        with patch('agent_real.fetch_forecast'), \
             patch('agent_real.check_no_leakage', side_effect=AssertionError('future run')):
            with self.assertRaises(AssertionError):
                provider('2026-01-31T00:00Z')

    def test_missing_lightgbm_does_not_fallback(self):
        with patch('agent_real.Path.exists', return_value=False):
            with self.assertRaises(FileNotFoundError):
                RealForecast('lightgbm', '2026-01-31T00:00Z')

    def test_issue_before_training_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cutoff'):
            self.provider()('2026-01-30T00:00Z')


if __name__ == '__main__':
    unittest.main()
