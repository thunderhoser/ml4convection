"""Unit tests for fourier_utils.py."""

import unittest
import numpy
from ml4convection.utils import fourier_utils

TOLERANCE = 1e-6

KM_TO_METRES = 1000.
GRID_SPACING_METRES = 1250.

# The following constants are used to test _get_spatial_resolutions.
FIRST_NUM_GRID_ROWS = 7
FIRST_NUM_GRID_COLUMNS = 7

FIRST_X_RESOLUTION_MATRIX_METRES = KM_TO_METRES * numpy.array([
    [numpy.inf, 3.75, 1.875, 1.25, 1.25, 1.875, 3.75],
    [numpy.inf, 3.75, 1.875, 1.25, 1.25, 1.875, 3.75],
    [numpy.inf, 3.75, 1.875, 1.25, 1.25, 1.875, 3.75],
    [numpy.inf, 3.75, 1.875, 1.25, 1.25, 1.875, 3.75],
    [numpy.inf, 3.75, 1.875, 1.25, 1.25, 1.875, 3.75],
    [numpy.inf, 3.75, 1.875, 1.25, 1.25, 1.875, 3.75],
    [numpy.inf, 3.75, 1.875, 1.25, 1.25, 1.875, 3.75]
])

FIRST_Y_RESOLUTION_MATRIX_METRES = KM_TO_METRES * numpy.array([
    [numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf,
     numpy.inf],
    [3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75],
    [1.875, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875],
    [1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25],
    [1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25],
    [1.875, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875],
    [3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75]
])

SECOND_NUM_GRID_ROWS = 7
SECOND_NUM_GRID_COLUMNS = 9

SECOND_X_RESOLUTION_MATRIX_METRES = KM_TO_METRES * numpy.array([
    [numpy.inf, 5, 2.5, 5. / 3, 1.25, 1.25, 5. / 3, 2.5, 5],
    [numpy.inf, 5, 2.5, 5. / 3, 1.25, 1.25, 5. / 3, 2.5, 5],
    [numpy.inf, 5, 2.5, 5. / 3, 1.25, 1.25, 5. / 3, 2.5, 5],
    [numpy.inf, 5, 2.5, 5. / 3, 1.25, 1.25, 5. / 3, 2.5, 5],
    [numpy.inf, 5, 2.5, 5. / 3, 1.25, 1.25, 5. / 3, 2.5, 5],
    [numpy.inf, 5, 2.5, 5. / 3, 1.25, 1.25, 5. / 3, 2.5, 5],
    [numpy.inf, 5, 2.5, 5. / 3, 1.25, 1.25, 5. / 3, 2.5, 5]
])

SECOND_Y_RESOLUTION_MATRIX_METRES = KM_TO_METRES * numpy.array([
    [numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf,
     numpy.inf, numpy.inf, numpy.inf],
    [3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75],
    [1.875, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875],
    [1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25],
    [1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25],
    [1.875, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875, 1.875],
    [3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75]
])

# print(numpy.sqrt(
#     SECOND_X_RESOLUTION_MATRIX_METRES ** 2 +
#     SECOND_Y_RESOLUTION_MATRIX_METRES ** 2
# ))

# The following constants are used to test apply_rectangular_filter.
COEFFICIENT_MATRIX = numpy.full((7, 9), 1.)

FIRST_MIN_RESOLUTION_METRES = 2500.
FIRST_MAX_RESOLUTION_METRES = 5000.
FIRST_COEFFICIENT_MATRIX_RECT = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0]
], dtype=float)

SECOND_MIN_RESOLUTION_METRES = 2500.
SECOND_MAX_RESOLUTION_METRES = numpy.inf
SECOND_COEFFICIENT_MATRIX_RECT = numpy.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
], dtype=float)

THIRD_MIN_RESOLUTION_METRES = 0.
THIRD_MAX_RESOLUTION_METRES = 5000.
THIRD_COEFFICIENT_MATRIX_RECT = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0]
], dtype=float)

# The following constants are used to test apply_butterworth_filter.
FILTER_ORDER = 2.

THIS_X_WAVENUMBER_MATRIX_METRES01 = numpy.power(
    2 * SECOND_X_RESOLUTION_MATRIX_METRES, -1
)
THIS_Y_WAVENUMBER_MATRIX_METRES01 = numpy.power(
    2 * SECOND_Y_RESOLUTION_MATRIX_METRES, -1
)
THIS_WAVENUMBER_MATRIX_METRES01 = numpy.sqrt(
    THIS_X_WAVENUMBER_MATRIX_METRES01 ** 2 +
    THIS_Y_WAVENUMBER_MATRIX_METRES01 ** 2
)

FIRST_MIN_WAVENUMBER_METRES01 = numpy.power(2 * FIRST_MAX_RESOLUTION_METRES, -1)
THIS_RATIO_MATRIX = (
    THIS_WAVENUMBER_MATRIX_METRES01 / FIRST_MIN_WAVENUMBER_METRES01
)
FIRST_COEFFICIENT_MATRIX_BUTTER = 1. - numpy.power(
    1 + THIS_RATIO_MATRIX ** (2 * FILTER_ORDER), -1
)

FIRST_MAX_WAVENUMBER_METRES01 = numpy.power(2 * FIRST_MIN_RESOLUTION_METRES, -1)
THIS_RATIO_MATRIX = (
    THIS_WAVENUMBER_MATRIX_METRES01 / FIRST_MAX_WAVENUMBER_METRES01
)
FIRST_COEFFICIENT_MATRIX_BUTTER *= numpy.power(
    1 + THIS_RATIO_MATRIX ** (2 * FILTER_ORDER), -1
)

SECOND_MAX_WAVENUMBER_METRES01 = numpy.power(
    2 * SECOND_MIN_RESOLUTION_METRES, -1
)
THIS_RATIO_MATRIX = (
    THIS_WAVENUMBER_MATRIX_METRES01 / SECOND_MAX_WAVENUMBER_METRES01
)
SECOND_COEFFICIENT_MATRIX_BUTTER = numpy.power(
    1 + THIS_RATIO_MATRIX ** (2 * FILTER_ORDER), -1
)

THIRD_MIN_WAVENUMBER_METRES01 = numpy.power(2 * THIRD_MAX_RESOLUTION_METRES, -1)
THIS_RATIO_MATRIX = (
    THIS_WAVENUMBER_MATRIX_METRES01 / THIRD_MIN_WAVENUMBER_METRES01
)
THIRD_COEFFICIENT_MATRIX_BUTTER = 1. - numpy.power(
    1 + THIS_RATIO_MATRIX ** (2 * FILTER_ORDER), -1
)

# The following constants are used to test taper_spatial_data and
# untaper_spatial_data.
FIRST_DATA_MATRIX = numpy.full((7, 7), 1.)

FIRST_DATA_MATRIX_TAPERED = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

SECOND_DATA_MATRIX = numpy.full((8, 7), 1.)

SECOND_DATA_MATRIX_TAPERED = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)


class FourierUtilsTests(unittest.TestCase):
    """Each method is a unit test for fourier_utils.py."""

    def test_get_spatial_resolutions_first(self):
        """Ensures correct output from _get_spatial_resolutions.

        In this case, using first set of input args.
        """

        this_x_res_matrix_metres, this_y_res_matrix_metres = (
            fourier_utils._get_spatial_resolutions(
                num_grid_rows=FIRST_NUM_GRID_ROWS,
                num_grid_columns=FIRST_NUM_GRID_COLUMNS,
                grid_spacing_metres=GRID_SPACING_METRES
            )
        )

        self.assertTrue(numpy.allclose(
            this_x_res_matrix_metres, FIRST_X_RESOLUTION_MATRIX_METRES,
            atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            this_y_res_matrix_metres, FIRST_Y_RESOLUTION_MATRIX_METRES,
            atol=TOLERANCE
        ))

    def test_get_spatial_resolutions_second(self):
        """Ensures correct output from _get_spatial_resolutions.

        In this case, using second set of input args.
        """

        this_x_res_matrix_metres, this_y_res_matrix_metres = (
            fourier_utils._get_spatial_resolutions(
                num_grid_rows=SECOND_NUM_GRID_ROWS,
                num_grid_columns=SECOND_NUM_GRID_COLUMNS,
                grid_spacing_metres=GRID_SPACING_METRES
            )
        )

        self.assertTrue(numpy.allclose(
            this_x_res_matrix_metres, SECOND_X_RESOLUTION_MATRIX_METRES,
            atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            this_y_res_matrix_metres, SECOND_Y_RESOLUTION_MATRIX_METRES,
            atol=TOLERANCE
        ))

    def test_apply_rectangular_filter_first(self):
        """Ensures correct output from apply_rectangular_filter.

        In this case, using first set of input args.
        """

        this_coeff_matrix = fourier_utils.apply_rectangular_filter(
            coefficient_matrix=COEFFICIENT_MATRIX + 0.,
            grid_spacing_metres=GRID_SPACING_METRES,
            min_resolution_metres=FIRST_MIN_RESOLUTION_METRES,
            max_resolution_metres=FIRST_MAX_RESOLUTION_METRES
        )

        self.assertTrue(numpy.allclose(
            this_coeff_matrix, FIRST_COEFFICIENT_MATRIX_RECT, atol=TOLERANCE
        ))

    def test_apply_rectangular_filter_second(self):
        """Ensures correct output from apply_rectangular_filter.

        In this case, using second set of input args.
        """

        this_coeff_matrix = fourier_utils.apply_rectangular_filter(
            coefficient_matrix=COEFFICIENT_MATRIX + 0.,
            grid_spacing_metres=GRID_SPACING_METRES,
            min_resolution_metres=SECOND_MIN_RESOLUTION_METRES,
            max_resolution_metres=SECOND_MAX_RESOLUTION_METRES
        )

        self.assertTrue(numpy.allclose(
            this_coeff_matrix, SECOND_COEFFICIENT_MATRIX_RECT, atol=TOLERANCE
        ))

    def test_apply_rectangular_filter_third(self):
        """Ensures correct output from apply_rectangular_filter.

        In this case, using third set of input args.
        """

        this_coeff_matrix = fourier_utils.apply_rectangular_filter(
            coefficient_matrix=COEFFICIENT_MATRIX + 0.,
            grid_spacing_metres=GRID_SPACING_METRES,
            min_resolution_metres=THIRD_MIN_RESOLUTION_METRES,
            max_resolution_metres=THIRD_MAX_RESOLUTION_METRES
        )

        self.assertTrue(numpy.allclose(
            this_coeff_matrix, THIRD_COEFFICIENT_MATRIX_RECT, atol=TOLERANCE
        ))

    def test_apply_butterworth_filter_first(self):
        """Ensures correct output from apply_butterworth_filter.

        In this case, using first set of input args.
        """

        this_coeff_matrix = fourier_utils.apply_butterworth_filter(
            coefficient_matrix=COEFFICIENT_MATRIX + 0.,
            filter_order=FILTER_ORDER, grid_spacing_metres=GRID_SPACING_METRES,
            min_resolution_metres=FIRST_MIN_RESOLUTION_METRES,
            max_resolution_metres=FIRST_MAX_RESOLUTION_METRES
        )

        self.assertTrue(numpy.allclose(
            this_coeff_matrix, FIRST_COEFFICIENT_MATRIX_BUTTER, atol=TOLERANCE
        ))

    def test_apply_butterworth_filter_second(self):
        """Ensures correct output from apply_butterworth_filter.

        In this case, using second set of input args.
        """

        this_coeff_matrix = fourier_utils.apply_butterworth_filter(
            coefficient_matrix=COEFFICIENT_MATRIX + 0.,
            filter_order=FILTER_ORDER, grid_spacing_metres=GRID_SPACING_METRES,
            min_resolution_metres=SECOND_MIN_RESOLUTION_METRES,
            max_resolution_metres=SECOND_MAX_RESOLUTION_METRES
        )

        self.assertTrue(numpy.allclose(
            this_coeff_matrix, SECOND_COEFFICIENT_MATRIX_BUTTER, atol=TOLERANCE
        ))

    def test_apply_butterworth_filter_third(self):
        """Ensures correct output from apply_butterworth_filter.

        In this case, using third set of input args.
        """

        this_coeff_matrix = fourier_utils.apply_butterworth_filter(
            coefficient_matrix=COEFFICIENT_MATRIX + 0.,
            filter_order=FILTER_ORDER, grid_spacing_metres=GRID_SPACING_METRES,
            min_resolution_metres=THIRD_MIN_RESOLUTION_METRES,
            max_resolution_metres=THIRD_MAX_RESOLUTION_METRES
        )

        self.assertTrue(numpy.allclose(
            this_coeff_matrix, THIRD_COEFFICIENT_MATRIX_BUTTER, atol=TOLERANCE
        ))

    def test_taper_spatial_data_first(self):
        """Ensures correct output from taper_spatial_data.

        In this case, using first set of input args.
        """

        this_data_matrix = fourier_utils.taper_spatial_data(
            FIRST_DATA_MATRIX + 0.
        )
        self.assertTrue(numpy.allclose(
            this_data_matrix, FIRST_DATA_MATRIX_TAPERED, atol=TOLERANCE
        ))

    def test_taper_spatial_data_second(self):
        """Ensures correct output from taper_spatial_data.

        In this case, using second set of input args.
        """

        this_data_matrix = fourier_utils.taper_spatial_data(
            SECOND_DATA_MATRIX + 0.
        )
        self.assertTrue(numpy.allclose(
            this_data_matrix, SECOND_DATA_MATRIX_TAPERED, atol=TOLERANCE
        ))

    def test_untaper_spatial_data_first(self):
        """Ensures correct output from untaper_spatial_data.

        In this case, using first set of input args.
        """

        this_data_matrix = fourier_utils.untaper_spatial_data(
            FIRST_DATA_MATRIX_TAPERED + 0.
        )
        self.assertTrue(numpy.allclose(
            this_data_matrix, FIRST_DATA_MATRIX, atol=TOLERANCE
        ))

    def test_untaper_spatial_data_second(self):
        """Ensures correct output from untaper_spatial_data.

        In this case, using second set of input args.
        """

        this_data_matrix = fourier_utils.untaper_spatial_data(
            SECOND_DATA_MATRIX_TAPERED + 0.
        )
        self.assertTrue(numpy.allclose(
            this_data_matrix, SECOND_DATA_MATRIX, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
