"""Unit tests for wavelet_utils.py."""

import copy
import unittest
import numpy
import tensorflow
from keras import backend as K
from ml4convection.utils import wavelet_utils

TOLERANCE = 1e-6

# The following constants are used to test taper_spatial_data and
# untaper_spatial_data.
FIRST_DATA_MATRIX_UNTAPERED = numpy.random.normal(
    loc=0., scale=1., size=(10, 15, 20)
)
FIRST_PADDING_ARG = ((0, 0), (8, 9), (6, 6))

SECOND_DATA_MATRIX_UNTAPERED = numpy.random.normal(
    loc=0., scale=1., size=(10, 100, 50)
)
SECOND_PADDING_ARG = ((0, 0), (14, 14), (39, 39))

THIRD_DATA_MATRIX_UNTAPERED = numpy.random.normal(
    loc=0., scale=1., size=(10, 200, 200)
)
THIRD_PADDING_ARG = ((0, 0), (28, 28), (28, 28))

FOURTH_DATA_MATRIX_UNTAPERED = numpy.random.normal(
    loc=0., scale=1., size=(10, 300, 500)
)
FOURTH_PADDING_ARG = ((0, 0), (106, 106), (6, 6))

FIFTH_DATA_MATRIX_UNTAPERED = numpy.random.normal(
    loc=0., scale=1., size=(10, 300, 1000)
)
FIFTH_PADDING_ARG = ((0, 0), (362, 362), (12, 12))

# The following constants are used to test filter_coefficients.
GRID_SPACING_DEG = 0.0125

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


class WaveletUtilsTests(unittest.TestCase):
    """Each method is a unit test for wavelet_utils.py."""

    def test_taper_spatial_data_first(self):
        """Ensures correct output from taper_spatial_data

        In this case, using first input matrix.
        """

        this_data_matrix, this_padding_arg = wavelet_utils.taper_spatial_data(
            FIRST_DATA_MATRIX_UNTAPERED + 0.
        )

        self.assertTrue(numpy.isclose(
            numpy.sum(this_data_matrix), numpy.sum(FIRST_DATA_MATRIX_UNTAPERED),
            atol=TOLERANCE
        ))
        self.assertTrue(this_padding_arg == FIRST_PADDING_ARG)

    def test_taper_spatial_data_second(self):
        """Ensures correct output from taper_spatial_data

        In this case, using second input matrix.
        """

        this_data_matrix, this_padding_arg = wavelet_utils.taper_spatial_data(
            SECOND_DATA_MATRIX_UNTAPERED + 0.
        )

        self.assertTrue(numpy.isclose(
            numpy.sum(this_data_matrix), numpy.sum(SECOND_DATA_MATRIX_UNTAPERED),
            atol=TOLERANCE
        ))
        self.assertTrue(this_padding_arg == SECOND_PADDING_ARG)

    def test_taper_spatial_data_third(self):
        """Ensures correct output from taper_spatial_data

        In this case, using third input matrix.
        """

        this_data_matrix, this_padding_arg = wavelet_utils.taper_spatial_data(
            THIRD_DATA_MATRIX_UNTAPERED + 0.
        )

        self.assertTrue(numpy.isclose(
            numpy.sum(this_data_matrix), numpy.sum(THIRD_DATA_MATRIX_UNTAPERED),
            atol=TOLERANCE
        ))
        self.assertTrue(this_padding_arg == THIRD_PADDING_ARG)

    def test_taper_spatial_data_fourth(self):
        """Ensures correct output from taper_spatial_data

        In this case, using fourth input matrix.
        """

        this_data_matrix, this_padding_arg = wavelet_utils.taper_spatial_data(
            FOURTH_DATA_MATRIX_UNTAPERED + 0.
        )

        self.assertTrue(numpy.isclose(
            numpy.sum(this_data_matrix), numpy.sum(FOURTH_DATA_MATRIX_UNTAPERED),
            atol=TOLERANCE
        ))
        self.assertTrue(this_padding_arg == FOURTH_PADDING_ARG)

    def test_taper_spatial_data_fifth(self):
        """Ensures correct output from taper_spatial_data

        In this case, using fifth input matrix.
        """

        this_data_matrix, this_padding_arg = wavelet_utils.taper_spatial_data(
            FIFTH_DATA_MATRIX_UNTAPERED + 0.
        )

        self.assertTrue(numpy.isclose(
            numpy.sum(this_data_matrix), numpy.sum(FIFTH_DATA_MATRIX_UNTAPERED),
            atol=TOLERANCE
        ))
        self.assertTrue(this_padding_arg == FIFTH_PADDING_ARG)

    def test_untaper_spatial_data_first(self):
        """Ensures correct output from untaper_spatial_data

        In this case, using first input matrix.
        """

        this_data_matrix, this_padding_arg = wavelet_utils.taper_spatial_data(
            FIRST_DATA_MATRIX_UNTAPERED + 0.
        )
        this_data_matrix = wavelet_utils.untaper_spatial_data(
            spatial_data_matrix=this_data_matrix,
            numpy_pad_width=this_padding_arg
        )

        self.assertTrue(numpy.allclose(
            this_data_matrix, FIRST_DATA_MATRIX_UNTAPERED, atol=TOLERANCE
        ))

    def test_untaper_spatial_data_second(self):
        """Ensures correct output from untaper_spatial_data

        In this case, using second input matrix.
        """

        this_data_matrix, this_padding_arg = wavelet_utils.taper_spatial_data(
            SECOND_DATA_MATRIX_UNTAPERED + 0.
        )
        this_data_matrix = wavelet_utils.untaper_spatial_data(
            spatial_data_matrix=this_data_matrix,
            numpy_pad_width=this_padding_arg
        )

        self.assertTrue(numpy.allclose(
            this_data_matrix, SECOND_DATA_MATRIX_UNTAPERED, atol=TOLERANCE
        ))

    def test_untaper_spatial_data_third(self):
        """Ensures correct output from untaper_spatial_data

        In this case, using third input matrix.
        """

        this_data_matrix, this_padding_arg = wavelet_utils.taper_spatial_data(
            THIRD_DATA_MATRIX_UNTAPERED + 0.
        )
        this_data_matrix = wavelet_utils.untaper_spatial_data(
            spatial_data_matrix=this_data_matrix,
            numpy_pad_width=this_padding_arg
        )

        self.assertTrue(numpy.allclose(
            this_data_matrix, THIRD_DATA_MATRIX_UNTAPERED, atol=TOLERANCE
        ))

    def test_untaper_spatial_data_fourth(self):
        """Ensures correct output from untaper_spatial_data

        In this case, using fourth input matrix.
        """

        this_data_matrix, this_padding_arg = wavelet_utils.taper_spatial_data(
            FOURTH_DATA_MATRIX_UNTAPERED + 0.
        )
        this_data_matrix = wavelet_utils.untaper_spatial_data(
            spatial_data_matrix=this_data_matrix,
            numpy_pad_width=this_padding_arg
        )

        self.assertTrue(numpy.allclose(
            this_data_matrix, FOURTH_DATA_MATRIX_UNTAPERED, atol=TOLERANCE
        ))

    def test_untaper_spatial_data_fifth(self):
        """Ensures correct output from untaper_spatial_data

        In this case, using fifth input matrix.
        """

        this_data_matrix, this_padding_arg = wavelet_utils.taper_spatial_data(
            FIFTH_DATA_MATRIX_UNTAPERED + 0.
        )
        this_data_matrix = wavelet_utils.untaper_spatial_data(
            spatial_data_matrix=this_data_matrix,
            numpy_pad_width=this_padding_arg
        )

        self.assertTrue(numpy.allclose(
            this_data_matrix, FIFTH_DATA_MATRIX_UNTAPERED, atol=TOLERANCE
        ))

    def test_filter_coefficients_first(self):
        """Ensures correct output from filter_coefficients.

        In this case, using first set of desired resolutions.
        """

        this_coeff_tensor_by_level = wavelet_utils.filter_coefficients(
            coeff_tensor_by_level=copy.deepcopy(COEFF_TENSOR_BY_LEVEL_ORIG),
            grid_spacing_metres=GRID_SPACING_DEG,
            min_resolution_metres=FIRST_MIN_RESOLUTION_DEG,
            max_resolution_metres=FIRST_MAX_RESOLUTION_DEG
        )

        this_coeff_matrix_by_level = [
            K.eval(x) for x in this_coeff_tensor_by_level
        ]

        for i in range(len(this_coeff_matrix_by_level)):
            self.assertTrue(numpy.allclose(
                this_coeff_matrix_by_level[i], FIRST_COEFF_MATRIX_BY_LEVEL[i],
                atol=TOLERANCE
            ))

    def test_filter_coefficients_second(self):
        """Ensures correct output from filter_coefficients.

        In this case, using second set of desired resolutions.
        """

        this_coeff_tensor_by_level = wavelet_utils.filter_coefficients(
            coeff_tensor_by_level=copy.deepcopy(COEFF_TENSOR_BY_LEVEL_ORIG),
            grid_spacing_metres=GRID_SPACING_DEG,
            min_resolution_metres=SECOND_MIN_RESOLUTION_DEG,
            max_resolution_metres=SECOND_MAX_RESOLUTION_DEG
        )

        this_coeff_matrix_by_level = [
            K.eval(x) for x in this_coeff_tensor_by_level
        ]

        for i in range(len(this_coeff_matrix_by_level)):
            self.assertTrue(numpy.allclose(
                this_coeff_matrix_by_level[i], SECOND_COEFF_MATRIX_BY_LEVEL[i],
                atol=TOLERANCE
            ))

    def test_filter_coefficients_third(self):
        """Ensures correct output from filter_coefficients.

        In this case, using third set of desired resolutions.
        """

        this_coeff_tensor_by_level = wavelet_utils.filter_coefficients(
            coeff_tensor_by_level=copy.deepcopy(COEFF_TENSOR_BY_LEVEL_ORIG),
            grid_spacing_metres=GRID_SPACING_DEG,
            min_resolution_metres=THIRD_MIN_RESOLUTION_DEG,
            max_resolution_metres=THIRD_MAX_RESOLUTION_DEG
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

    def test_filter_coefficients_fourth(self):
        """Ensures correct output from filter_coefficients.

        In this case, using fourth set of desired resolutions.
        """

        this_coeff_tensor_by_level = wavelet_utils.filter_coefficients(
            coeff_tensor_by_level=copy.deepcopy(COEFF_TENSOR_BY_LEVEL_ORIG),
            grid_spacing_metres=GRID_SPACING_DEG,
            min_resolution_metres=FOURTH_MIN_RESOLUTION_DEG,
            max_resolution_metres=FOURTH_MAX_RESOLUTION_DEG
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

    def test_filter_coefficients_fifth(self):
        """Ensures correct output from filter_coefficients.

        In this case, using fifth set of desired resolutions.
        """

        this_coeff_tensor_by_level = wavelet_utils.filter_coefficients(
            coeff_tensor_by_level=copy.deepcopy(COEFF_TENSOR_BY_LEVEL_ORIG),
            grid_spacing_metres=GRID_SPACING_DEG,
            min_resolution_metres=FIFTH_MIN_RESOLUTION_DEG,
            max_resolution_metres=FIFTH_MAX_RESOLUTION_DEG
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
