"""Unit tests for wavelet_metrics.py."""

import copy
import unittest
import numpy
from ml4convection.machine_learning import wavelet_metrics

THESE_DIM = (
    wavelet_metrics.NUM_ROWS_BEFORE_PADDING,
    wavelet_metrics.NUM_ROWS_BEFORE_PADDING
)
DUMMY_MASK_MATRIX = numpy.full(THESE_DIM, 1, dtype=bool)

FIRST_MIN_RESOLUTION_DEG = 0.
FIRST_MAX_RESOLUTION_DEG = 1.
FIRST_KEEP_MEAN_FLAGS = numpy.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
FIRST_KEEP_DETAIL_FLAGS = numpy.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)

SECOND_MIN_RESOLUTION_DEG = 0.0125
SECOND_MAX_RESOLUTION_DEG = 0.2
SECOND_KEEP_MEAN_FLAGS = numpy.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
SECOND_KEEP_DETAIL_FLAGS = numpy.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)

THIRD_MIN_RESOLUTION_DEG = 0.1
THIRD_MAX_RESOLUTION_DEG = 0.2
THIRD_KEEP_MEAN_FLAGS = numpy.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
THIRD_KEEP_DETAIL_FLAGS = numpy.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=bool)

FOURTH_MIN_RESOLUTION_DEG = 0.0125
FOURTH_MAX_RESOLUTION_DEG = 0.09
FOURTH_KEEP_MEAN_FLAGS = numpy.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
FOURTH_KEEP_DETAIL_FLAGS = numpy.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)

FIFTH_MIN_RESOLUTION_DEG = 0.03
FIFTH_MAX_RESOLUTION_DEG = 0.09
FIFTH_KEEP_MEAN_FLAGS = numpy.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
FIFTH_KEEP_DETAIL_FLAGS = numpy.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=bool)


class WaveletMetricsTests(unittest.TestCase):
    """Each method is a unit test for wavelet_metrics.py."""

    def test_check_input_args_first(self):
        """Ensures correct output from _check_input_args.

        In this case, using first set of desired resolutions.
        """

        these_keep_mean_flags, these_keep_detail_flags = (
            wavelet_metrics._check_input_args(
                min_resolution_deg=FIRST_MIN_RESOLUTION_DEG,
                max_resolution_deg=FIRST_MAX_RESOLUTION_DEG,
                mask_matrix=copy.deepcopy(DUMMY_MASK_MATRIX),
                function_name='foo'
            )[1:]
        )

        self.assertTrue(numpy.array_equal(
            FIRST_KEEP_MEAN_FLAGS, these_keep_mean_flags
        ))
        self.assertTrue(numpy.array_equal(
            FIRST_KEEP_DETAIL_FLAGS, these_keep_detail_flags
        ))

    def test_check_input_args_second(self):
        """Ensures correct output from _check_input_args.

        In this case, using second set of desired resolutions.
        """

        these_keep_mean_flags, these_keep_detail_flags = (
            wavelet_metrics._check_input_args(
                min_resolution_deg=SECOND_MIN_RESOLUTION_DEG,
                max_resolution_deg=SECOND_MAX_RESOLUTION_DEG,
                mask_matrix=copy.deepcopy(DUMMY_MASK_MATRIX),
                function_name='foo'
            )[1:]
        )

        self.assertTrue(numpy.array_equal(
            SECOND_KEEP_MEAN_FLAGS, these_keep_mean_flags
        ))
        self.assertTrue(numpy.array_equal(
            SECOND_KEEP_DETAIL_FLAGS, these_keep_detail_flags
        ))

    def test_check_input_args_third(self):
        """Ensures correct output from _check_input_args.

        In this case, using third set of desired resolutions.
        """

        these_keep_mean_flags, these_keep_detail_flags = (
            wavelet_metrics._check_input_args(
                min_resolution_deg=THIRD_MIN_RESOLUTION_DEG,
                max_resolution_deg=THIRD_MAX_RESOLUTION_DEG,
                mask_matrix=copy.deepcopy(DUMMY_MASK_MATRIX),
                function_name='foo'
            )[1:]
        )

        self.assertTrue(numpy.array_equal(
            THIRD_KEEP_MEAN_FLAGS, these_keep_mean_flags
        ))
        self.assertTrue(numpy.array_equal(
            THIRD_KEEP_DETAIL_FLAGS, these_keep_detail_flags
        ))

    def test_check_input_args_fourth(self):
        """Ensures correct output from _check_input_args.

        In this case, using fourth set of desired resolutions.
        """

        these_keep_mean_flags, these_keep_detail_flags = (
            wavelet_metrics._check_input_args(
                min_resolution_deg=FOURTH_MIN_RESOLUTION_DEG,
                max_resolution_deg=FOURTH_MAX_RESOLUTION_DEG,
                mask_matrix=copy.deepcopy(DUMMY_MASK_MATRIX),
                function_name='foo'
            )[1:]
        )

        self.assertTrue(numpy.array_equal(
            FOURTH_KEEP_MEAN_FLAGS, these_keep_mean_flags
        ))
        self.assertTrue(numpy.array_equal(
            FOURTH_KEEP_DETAIL_FLAGS, these_keep_detail_flags
        ))

    def test_check_input_args_fifth(self):
        """Ensures correct output from _check_input_args.

        In this case, using fifth set of desired resolutions.
        """

        these_keep_mean_flags, these_keep_detail_flags = (
            wavelet_metrics._check_input_args(
                min_resolution_deg=FIFTH_MIN_RESOLUTION_DEG,
                max_resolution_deg=FIFTH_MAX_RESOLUTION_DEG,
                mask_matrix=copy.deepcopy(DUMMY_MASK_MATRIX),
                function_name='foo'
            )[1:]
        )

        self.assertTrue(numpy.array_equal(
            FIFTH_KEEP_MEAN_FLAGS, these_keep_mean_flags
        ))
        self.assertTrue(numpy.array_equal(
            FIFTH_KEEP_DETAIL_FLAGS, these_keep_detail_flags
        ))


if __name__ == '__main__':
    unittest.main()
