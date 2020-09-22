"""Unit tests for general_utils.py."""

import unittest
import numpy
from ml4convection.utils import general_utils

TOLERANCE = 1e-6

# The following constants are used to test get_previous_date and get_next_date.
CURRENT_DATE_STRINGS = [
    '20191231', '20200101', '20200228', '20200229', '20200301'
]
PREVIOUS_DATE_STRINGS = [
    '20191230', '20191231', '20200227', '20200228', '20200229'
]
NEXT_DATE_STRINGS = [
    '20200101', '20200102', '20200229', '20200301', '20200302'
]

# The following constants are used to test create_mean_filter.
FIRST_HALF_NUM_ROWS = 0
FIRST_HALF_NUM_COLUMNS = 1
FIRST_NUM_CHANNELS = 3

FIRST_WEIGHT_MATRIX = numpy.full((1, 3), 1. / 3)
FIRST_WEIGHT_MATRIX = numpy.expand_dims(FIRST_WEIGHT_MATRIX, axis=-1)
FIRST_WEIGHT_MATRIX = numpy.expand_dims(FIRST_WEIGHT_MATRIX, axis=-1)
FIRST_WEIGHT_MATRIX = numpy.repeat(FIRST_WEIGHT_MATRIX, repeats=3, axis=-2)
FIRST_WEIGHT_MATRIX = numpy.repeat(FIRST_WEIGHT_MATRIX, repeats=3, axis=-1)

SECOND_HALF_NUM_ROWS = 2
SECOND_HALF_NUM_COLUMNS = 2
SECOND_NUM_CHANNELS = 1

SECOND_WEIGHT_MATRIX = numpy.full((5, 5), 1. / 25)
SECOND_WEIGHT_MATRIX = numpy.expand_dims(SECOND_WEIGHT_MATRIX, axis=-1)
SECOND_WEIGHT_MATRIX = numpy.expand_dims(SECOND_WEIGHT_MATRIX, axis=-1)

# The following constants are used to test fill_nans.
MATRIX_WITH_NANS_1D = numpy.array([1, 2, 3, numpy.nan])
MATRIX_WITHOUT_NANS_1D = numpy.array([1, 2, 3, 3], dtype=float)

MATRIX_WITH_NANS_2D = numpy.array([
    [1, 2, 3, 4, 5],
    [6, 7, numpy.nan, numpy.nan, 10],
    [numpy.nan, 12, numpy.nan, numpy.nan, 15]
])
MATRIX_WITHOUT_NANS_2D = numpy.array([
    [1, 2, 3, 4, 5],
    [6, 7, 7, 4, 10],
    [6, 12, 12, 15, 15]
])

MATRIX_WITH_NANS_3D = numpy.stack(
    (MATRIX_WITH_NANS_2D, MATRIX_WITH_NANS_2D), axis=0
)
MATRIX_WITHOUT_NANS_3D = numpy.stack(
    (MATRIX_WITHOUT_NANS_2D, MATRIX_WITHOUT_NANS_2D), axis=0
)


class GeneralUtilsTests(unittest.TestCase):
    """Each method is a unit test for general_utils.py."""

    def test_get_previous_date(self):
        """Ensures correct output from get_previous_date."""

        these_previous_date_strings = [
            general_utils.get_previous_date(d) for d in CURRENT_DATE_STRINGS
        ]
        self.assertTrue(these_previous_date_strings == PREVIOUS_DATE_STRINGS)

    def test_get_next_date(self):
        """Ensures correct output from get_next_date."""

        these_next_date_strings = [
            general_utils.get_next_date(d) for d in CURRENT_DATE_STRINGS
        ]
        self.assertTrue(these_next_date_strings == NEXT_DATE_STRINGS)

    def test_create_mean_filter_first(self):
        """Ensures correct output from _create_mean_filter.

        In this case, using first set of inputs.
        """

        this_weight_matrix = general_utils.create_mean_filter(
            half_num_rows=FIRST_HALF_NUM_ROWS,
            half_num_columns=FIRST_HALF_NUM_COLUMNS,
            num_channels=FIRST_NUM_CHANNELS
        )

        self.assertTrue(numpy.allclose(
            this_weight_matrix, FIRST_WEIGHT_MATRIX, atol=TOLERANCE
        ))

    def test_create_mean_filter_second(self):
        """Ensures correct output from _create_mean_filter.

        In this case, using second set of inputs.
        """

        this_weight_matrix = general_utils.create_mean_filter(
            half_num_rows=SECOND_HALF_NUM_ROWS,
            half_num_columns=SECOND_HALF_NUM_COLUMNS,
            num_channels=SECOND_NUM_CHANNELS
        )

        self.assertTrue(numpy.allclose(
            this_weight_matrix, SECOND_WEIGHT_MATRIX, atol=TOLERANCE
        ))

    def test_fill_nans_1d(self):
        """Ensures correct output from fill_nans (with 1-D array)."""

        this_matrix_without_nans = general_utils.fill_nans(MATRIX_WITH_NANS_1D)
        self.assertTrue(numpy.allclose(
            this_matrix_without_nans, MATRIX_WITHOUT_NANS_1D, atol=TOLERANCE
        ))

    def test_fill_nans_2d(self):
        """Ensures correct output from fill_nans (with 2-D matrix)."""

        this_matrix_without_nans = general_utils.fill_nans(MATRIX_WITH_NANS_2D)
        self.assertTrue(numpy.allclose(
            this_matrix_without_nans, MATRIX_WITHOUT_NANS_2D, atol=TOLERANCE
        ))

    def test_fill_nans_3d(self):
        """Ensures correct output from fill_nans (with 3-D matrix)."""

        this_matrix_without_nans = general_utils.fill_nans(MATRIX_WITH_NANS_3D)
        self.assertTrue(numpy.allclose(
            this_matrix_without_nans, MATRIX_WITHOUT_NANS_3D, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
