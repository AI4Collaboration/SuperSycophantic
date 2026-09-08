import inspect
import math
import sys
import unittest
from unittest.mock import patch

import plot_trigger_figures as plot


class ConfidenceAggregationTests(unittest.TestCase):
    def test_complete_moments_match_raw_sample_interval(self):
        rows = [dict(turn=0, n=2, mean_confidence=2, sum_confidence=4, sum_sq_confidence=10),
                dict(turn=0, n=2, mean_confidence=4, sum_confidence=8, sum_sq_confidence=34)]
        observed = plot.combine_confidence_rows(rows, lambda row: True, 0)
        expected = plot.normal_ci_from_values([1, 3, 3, 5])
        for actual, target in zip(observed, expected):
            self.assertAlmostEqual(actual, target)
        self.assertLess(observed[1], observed[0])
        self.assertGreater(observed[2], observed[0])

    def test_mean_only_rows_never_manufacture_an_interval(self):
        rows = [dict(turn=0, n=3, mean_confidence=2), dict(turn=0, n=1, mean_confidence=4)]
        self.assertEqual(plot.combine_confidence_rows(rows, lambda row: True, 0, intervals=False), (2.5, None, None))
        with self.assertRaisesRegex(ValueError, 'require sum_confidence and sum_sq_confidence'):
            plot.combine_confidence_rows(rows, lambda row: True, 0)

    def test_either_missing_moment_rejects_intervals(self):
        row = dict(turn=0, n=100, mean_confidence=3, sum_confidence=300, sum_sq_confidence=1000)
        for key in ['sum_confidence', 'sum_sq_confidence']:
            candidate = dict(row)
            del candidate[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                plot.combine_confidence_rows([candidate], lambda row: True, 0)

    def test_real_zero_variance_remains_a_point_interval(self):
        row = dict(turn=0, n=100, mean_confidence=3, sum_confidence=300, sum_sq_confidence=900)
        self.assertEqual(plot.combine_confidence_rows([row], lambda row: True, 0), (3, 3, 3))

    def test_invalid_moments_fail_instead_of_clamping_to_zero(self):
        row = dict(turn=0, n=100, mean_confidence=3, sum_confidence=300, sum_sq_confidence=800)
        with self.assertRaisesRegex(ValueError, 'negative variance'):
            plot.combine_confidence_rows([row], lambda row: True, 0)
        row['sum_sq_confidence'] = math.nan
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            plot.combine_confidence_rows([row], lambda row: True, 0)

    def test_empty_groups_do_not_create_missing_moments(self):
        rows = [dict(turn=0, n=0, mean_confidence=0),
                dict(turn=0, n=2, mean_confidence=3, sum_confidence=6, sum_sq_confidence=20)]
        self.assertEqual(plot.combine_confidence_rows(rows, lambda row: True, 0),
                         plot.normal_ci_from_values([2, 4]))
        with self.assertRaisesRegex(ValueError, 'Missing confidence rows'):
            plot.combine_confidence_rows(rows, lambda row: False, 0)

    def test_existing_positional_helper_api_is_preserved(self):
        row = dict(turn=0, n=2, mean_confidence=3, sum_confidence=6, sum_sq_confidence=20)
        self.assertEqual(plot.combine_confidence_rows([row], lambda row: True, 0, 1), (3, 2, 4))

    def test_plot_and_cli_default_to_mean_only(self):
        signature = inspect.signature(plot.figure_confidence_trajectory)
        self.assertIs(signature.parameters['legacy_response_intervals'].default, False)
        with patch.object(sys, 'argv', ['plot_trigger_figures.py', '--run-id', 'fixture']):
            self.assertIs(plot.parse_args().legacy_response_intervals, False)
        with patch.object(sys, 'argv', ['plot_trigger_figures.py', '--run-id', 'fixture', '--legacy-response-intervals']):
            self.assertIs(plot.parse_args().legacy_response_intervals, True)


if __name__ == '__main__':
    unittest.main()
