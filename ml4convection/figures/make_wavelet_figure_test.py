"""Unit tests for make_wavelet_figure.py."""

import copy
import unittest
import numpy
import tensorflow
from keras import backend as K
from ml4convection.figures import make_wavelet_figure

TOLERANCE = 1e-6

COEFF_MATRIX_BY_LEVEL_ORIG = [
    numpy.random.uniform(low=-1., high=1., size=(10, 8, 8, 4)),
    numpy.random.uniform(low=-1., high=1., size=(10, 4, 4, 4)),
    numpy.random.uniform(low=-1., high=1., size=(10, 2, 2, 4)),
    numpy.random.uniform(low=-1., high=1., size=(10, 1, 1, 4))
]
COEFF_TENSOR_BY_LEVEL_ORIG = [
    tensorflow.constant(x, dtype=tensorflow.float64)
    for x in COEFF_MATRIX_BY_LEVEL_ORIG
]

# DETAIL_RESOLUTION_BY_LEVEL_DEG = numpy.array([0.0125, 0.025, 0.05, 0.1])
# MEAN_RESOLUTION_BY_LEVEL_DEG = numpy.array([0.025, 0.05, 0.1, 0.2])

FIRST_MIN_RESOLUTION_DEG = 0.
FIRST_MAX_RESOLUTION_DEG = 1.
FIRST_COEFF_MATRIX_BY_LEVEL = copy.deepcopy(COEFF_MATRIX_BY_LEVEL_ORIG)

SECOND_MIN_RESOLUTION_DEG = 0.0125
SECOND_MAX_RESOLUTION_DEG = 0.2
SECOND_COEFF_MATRIX_BY_LEVEL = copy.deepcopy(COEFF_MATRIX_BY_LEVEL_ORIG)

THIRD_MIN_RESOLUTION_DEG = 0.1
THIRD_MAX_RESOLUTION_DEG = 0.2
THIRD_COEFF_MATRIX_BY_LEVEL = copy.deepcopy(COEFF_MATRIX_BY_LEVEL_ORIG)
THIRD_COEFF_MATRIX_BY_LEVEL[0][..., 1:] = 0.
THIRD_COEFF_MATRIX_BY_LEVEL[1][..., 1:] = 0.
THIRD_COEFF_MATRIX_BY_LEVEL[2][..., 1:] = 0.
THIRD_COEFF_MATRIX_BY_LEVEL[0][..., 0] = numpy.nan
THIRD_COEFF_MATRIX_BY_LEVEL[1][..., 0] = numpy.nan

FOURTH_MIN_RESOLUTION_DEG = 0.0125
FOURTH_MAX_RESOLUTION_DEG = 0.09
FOURTH_COEFF_MATRIX_BY_LEVEL = copy.deepcopy(COEFF_MATRIX_BY_LEVEL_ORIG)
FOURTH_COEFF_MATRIX_BY_LEVEL[2][..., 0] = 0.
FOURTH_COEFF_MATRIX_BY_LEVEL[3][..., 0] = 0.
FOURTH_COEFF_MATRIX_BY_LEVEL[0][..., 0] = numpy.nan
FOURTH_COEFF_MATRIX_BY_LEVEL[1][..., 0] = numpy.nan

FIFTH_MIN_RESOLUTION_DEG = 0.03
FIFTH_MAX_RESOLUTION_DEG = 0.09
FIFTH_COEFF_MATRIX_BY_LEVEL = copy.deepcopy(COEFF_MATRIX_BY_LEVEL_ORIG)
FIFTH_COEFF_MATRIX_BY_LEVEL[2][..., 0] = 0.
FIFTH_COEFF_MATRIX_BY_LEVEL[3][..., 0] = 0.
FIFTH_COEFF_MATRIX_BY_LEVEL[0][..., 0] = numpy.nan
FIFTH_COEFF_MATRIX_BY_LEVEL[1][..., 0] = numpy.nan
FIFTH_COEFF_MATRIX_BY_LEVEL[0][..., 1:] = 0.
FIFTH_COEFF_MATRIX_BY_LEVEL[1][..., 1:] = 0.


class MakeWaveletFigureTests(unittest.TestCase):
    """Each method is a unit test for make_wavelet_figure.py."""

    def test_filter_wavelet_coeffs_first(self):
        """Ensures correct output from _filter_wavelet_coeffs.

        In this case, using first set of desired resolutions.
        """

        this_coeff_tensor_by_level = make_wavelet_figure._filter_wavelet_coeffs(
            coeff_tensor_by_level=copy.deepcopy(COEFF_TENSOR_BY_LEVEL_ORIG),
            min_resolution_deg=FIRST_MIN_RESOLUTION_DEG,
            max_resolution_deg=FIRST_MAX_RESOLUTION_DEG
        )

        this_coeff_matrix_by_level = [
            K.eval(x) for x in this_coeff_tensor_by_level
        ]

        for i in range(len(this_coeff_matrix_by_level)):
            self.assertTrue(numpy.allclose(
                this_coeff_matrix_by_level[i], FIRST_COEFF_MATRIX_BY_LEVEL[i],
                atol=TOLERANCE
            ))

    def test_filter_wavelet_coeffs_second(self):
        """Ensures correct output from _filter_wavelet_coeffs.

        In this case, using second set of desired resolutions.
        """

        this_coeff_tensor_by_level = make_wavelet_figure._filter_wavelet_coeffs(
            coeff_tensor_by_level=copy.deepcopy(COEFF_TENSOR_BY_LEVEL_ORIG),
            min_resolution_deg=SECOND_MIN_RESOLUTION_DEG,
            max_resolution_deg=SECOND_MAX_RESOLUTION_DEG
        )

        this_coeff_matrix_by_level = [
            K.eval(x) for x in this_coeff_tensor_by_level
        ]

        for i in range(len(this_coeff_matrix_by_level)):
            self.assertTrue(numpy.allclose(
                this_coeff_matrix_by_level[i], SECOND_COEFF_MATRIX_BY_LEVEL[i],
                atol=TOLERANCE
            ))

    def test_filter_wavelet_coeffs_third(self):
        """Ensures correct output from _filter_wavelet_coeffs.

        In this case, using third set of desired resolutions.
        """

        this_coeff_tensor_by_level = make_wavelet_figure._filter_wavelet_coeffs(
            coeff_tensor_by_level=copy.deepcopy(COEFF_TENSOR_BY_LEVEL_ORIG),
            min_resolution_deg=THIRD_MIN_RESOLUTION_DEG,
            max_resolution_deg=THIRD_MAX_RESOLUTION_DEG
        )

        actual_coeff_matrix_by_level = [
            K.eval(x) for x in this_coeff_tensor_by_level
        ]
        expected_coeff_matrix_by_level = THIRD_COEFF_MATRIX_BY_LEVEL

        for i in range(len(expected_coeff_matrix_by_level)):
            nan_flag_matrix = numpy.isnan(expected_coeff_matrix_by_level[i])
            expected_coeff_matrix_by_level[i][nan_flag_matrix] = (
                actual_coeff_matrix_by_level[i][nan_flag_matrix]
            )

            self.assertTrue(numpy.allclose(
                actual_coeff_matrix_by_level[i],
                expected_coeff_matrix_by_level[i],
                atol=TOLERANCE
            ))

    def test_filter_wavelet_coeffs_fourth(self):
        """Ensures correct output from _filter_wavelet_coeffs.

        In this case, using fourth set of desired resolutions.
        """

        this_coeff_tensor_by_level = make_wavelet_figure._filter_wavelet_coeffs(
            coeff_tensor_by_level=copy.deepcopy(COEFF_TENSOR_BY_LEVEL_ORIG),
            min_resolution_deg=FOURTH_MIN_RESOLUTION_DEG,
            max_resolution_deg=FOURTH_MAX_RESOLUTION_DEG
        )

        actual_coeff_matrix_by_level = [
            K.eval(x) for x in this_coeff_tensor_by_level
        ]
        expected_coeff_matrix_by_level = FOURTH_COEFF_MATRIX_BY_LEVEL

        for i in range(len(expected_coeff_matrix_by_level)):
            nan_flag_matrix = numpy.isnan(expected_coeff_matrix_by_level[i])
            expected_coeff_matrix_by_level[i][nan_flag_matrix] = (
                actual_coeff_matrix_by_level[i][nan_flag_matrix]
            )

            self.assertTrue(numpy.allclose(
                actual_coeff_matrix_by_level[i],
                expected_coeff_matrix_by_level[i],
                atol=TOLERANCE
            ))

    def test_filter_wavelet_coeffs_fifth(self):
        """Ensures correct output from _filter_wavelet_coeffs.

        In this case, using fifth set of desired resolutions.
        """

        this_coeff_tensor_by_level = make_wavelet_figure._filter_wavelet_coeffs(
            coeff_tensor_by_level=copy.deepcopy(COEFF_TENSOR_BY_LEVEL_ORIG),
            min_resolution_deg=FIFTH_MIN_RESOLUTION_DEG,
            max_resolution_deg=FIFTH_MAX_RESOLUTION_DEG
        )

        actual_coeff_matrix_by_level = [
            K.eval(x) for x in this_coeff_tensor_by_level
        ]
        expected_coeff_matrix_by_level = FIFTH_COEFF_MATRIX_BY_LEVEL

        for i in range(len(expected_coeff_matrix_by_level)):
            nan_flag_matrix = numpy.isnan(expected_coeff_matrix_by_level[i])
            expected_coeff_matrix_by_level[i][nan_flag_matrix] = (
                actual_coeff_matrix_by_level[i][nan_flag_matrix]
            )

            self.assertTrue(numpy.allclose(
                actual_coeff_matrix_by_level[i],
                expected_coeff_matrix_by_level[i],
                atol=TOLERANCE
            ))


if __name__ == '__main__':
    unittest.main()
