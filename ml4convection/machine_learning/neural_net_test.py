"""Unit tests for neural_net.py."""

import copy
import unittest
import numpy
from ml4convection.io import example_io
from ml4convection.machine_learning import neural_net

TOLERANCE = 1e-6

# The following constants are used to test _reshape_predictor_matrix.
ORIG_PREDICTOR_MATRIX_1LAG_TIME = numpy.random.normal(
    loc=0., scale=5., size=(30, 50, 100, 7)
)
NEW_PREDICTOR_MATRIX_1LAG_TIME = ORIG_PREDICTOR_MATRIX_1LAG_TIME + 0.

PREDICTOR_MATRIX_FIRST_LAG = numpy.random.normal(
    loc=0., scale=5., size=(10, 50, 100, 7)
)
PREDICTOR_MATRIX_SECOND_LAG = numpy.random.normal(
    loc=1., scale=10., size=(10, 50, 100, 7)
)
PREDICTOR_MATRIX_THIRD_LAG = numpy.random.normal(
    loc=-3., scale=3., size=(10, 50, 100, 7)
)

ORIG_PREDICTOR_MATRIX_3LAG_TIMES = numpy.full((0, 50, 100, 7), numpy.nan)
NEW_PREDICTOR_MATRIX_3LAG_TIMES = numpy.full((10, 50, 100, 21), numpy.nan)

for i in range(10):
    ORIG_PREDICTOR_MATRIX_3LAG_TIMES = numpy.concatenate((
        ORIG_PREDICTOR_MATRIX_3LAG_TIMES, PREDICTOR_MATRIX_FIRST_LAG[[i], ...]
    ), axis=0)

    ORIG_PREDICTOR_MATRIX_3LAG_TIMES = numpy.concatenate((
        ORIG_PREDICTOR_MATRIX_3LAG_TIMES, PREDICTOR_MATRIX_SECOND_LAG[[i], ...]
    ), axis=0)

    ORIG_PREDICTOR_MATRIX_3LAG_TIMES = numpy.concatenate((
        ORIG_PREDICTOR_MATRIX_3LAG_TIMES, PREDICTOR_MATRIX_THIRD_LAG[[i], ...]
    ), axis=0)

    NEW_PREDICTOR_MATRIX_3LAG_TIMES[i, ..., 0::3] = (
        PREDICTOR_MATRIX_FIRST_LAG[i, ...]
    )
    NEW_PREDICTOR_MATRIX_3LAG_TIMES[i, ..., 1::3] = (
        PREDICTOR_MATRIX_SECOND_LAG[i, ...]
    )
    NEW_PREDICTOR_MATRIX_3LAG_TIMES[i, ..., 2::3] = (
        PREDICTOR_MATRIX_THIRD_LAG[i, ...]
    )

# The following constants are used to test _find_days_with_both_inputs.
TOP_PREDICTOR_DIR_NAME = 'foo'
TOP_TARGET_DIR_NAME = 'bar'

PREDICTOR_DATE_STRINGS = [
    '20200105', '20200103', '20200110', '20200115', '20200109', '20200104',
    '20200112', '20200113', '20200102', '20200106'
]
TARGET_DATE_STRINGS = [
    '20200109', '20200107', '20200111', '20200108', '20200113', '20200103',
    '20200112', '20200101', '20200110', '20200115'
]

PREDICTOR_FILE_NAMES = [
    example_io.find_predictor_file(
        top_directory_name=TOP_PREDICTOR_DIR_NAME, date_string=d,
        raise_error_if_missing=False
    ) for d in PREDICTOR_DATE_STRINGS
]

TARGET_FILE_NAMES = [
    example_io.find_target_file(
        top_directory_name=TOP_TARGET_DIR_NAME, date_string=d,
        raise_error_if_missing=False
    ) for d in TARGET_DATE_STRINGS
]

VALID_DATE_STRINGS_ZERO_LEAD = [
    '20200109', '20200113', '20200103', '20200112', '20200110', '20200115'
]
VALID_DATE_STRINGS_NONZERO_LEAD = ['20200113', '20200103', '20200110']

# The following constants are used to test _get_input_px_for_partial_grid.
FIRST_PARTIAL_GRID_DICT_BEFORE = {
    neural_net.NUM_FULL_ROWS_KEY: 881,
    neural_net.NUM_FULL_COLUMNS_KEY: 921,
    neural_net.NUM_PARTIAL_ROWS_KEY: 205,
    neural_net.NUM_PARTIAL_COLUMNS_KEY: 205,
    neural_net.OVERLAP_SIZE_KEY: 50,
    neural_net.LAST_INPUT_ROW_KEY: -1,
    neural_net.LAST_INPUT_COLUMN_KEY: -1
}

FIRST_PARTIAL_GRID_DICT_AFTER = {
    neural_net.NUM_FULL_ROWS_KEY: 881,
    neural_net.NUM_FULL_COLUMNS_KEY: 921,
    neural_net.NUM_PARTIAL_ROWS_KEY: 205,
    neural_net.NUM_PARTIAL_COLUMNS_KEY: 205,
    neural_net.OVERLAP_SIZE_KEY: 50,
    neural_net.FIRST_INPUT_ROW_KEY: 0,
    neural_net.LAST_INPUT_ROW_KEY: 204,
    neural_net.FIRST_INPUT_COLUMN_KEY: 0,
    neural_net.LAST_INPUT_COLUMN_KEY: 204
}

SECOND_PARTIAL_GRID_DICT_BEFORE = {
    neural_net.NUM_FULL_ROWS_KEY: 881,
    neural_net.NUM_FULL_COLUMNS_KEY: 921,
    neural_net.NUM_PARTIAL_ROWS_KEY: 205,
    neural_net.NUM_PARTIAL_COLUMNS_KEY: 205,
    neural_net.OVERLAP_SIZE_KEY: 50,
    neural_net.LAST_INPUT_ROW_KEY: 204,
    neural_net.LAST_INPUT_COLUMN_KEY: 309
}

SECOND_PARTIAL_GRID_DICT_AFTER = {
    neural_net.NUM_FULL_ROWS_KEY: 881,
    neural_net.NUM_FULL_COLUMNS_KEY: 921,
    neural_net.NUM_PARTIAL_ROWS_KEY: 205,
    neural_net.NUM_PARTIAL_COLUMNS_KEY: 205,
    neural_net.OVERLAP_SIZE_KEY: 50,
    neural_net.FIRST_INPUT_ROW_KEY: 0,
    neural_net.LAST_INPUT_ROW_KEY: 204,
    neural_net.FIRST_INPUT_COLUMN_KEY: 210,
    neural_net.LAST_INPUT_COLUMN_KEY: 414
}

THIRD_PARTIAL_GRID_DICT_BEFORE = {
    neural_net.NUM_FULL_ROWS_KEY: 881,
    neural_net.NUM_FULL_COLUMNS_KEY: 921,
    neural_net.NUM_PARTIAL_ROWS_KEY: 205,
    neural_net.NUM_PARTIAL_COLUMNS_KEY: 205,
    neural_net.OVERLAP_SIZE_KEY: 50,
    neural_net.LAST_INPUT_ROW_KEY: 204,
    neural_net.LAST_INPUT_COLUMN_KEY: 920
}

THIRD_PARTIAL_GRID_DICT_AFTER = {
    neural_net.NUM_FULL_ROWS_KEY: 881,
    neural_net.NUM_FULL_COLUMNS_KEY: 921,
    neural_net.NUM_PARTIAL_ROWS_KEY: 205,
    neural_net.NUM_PARTIAL_COLUMNS_KEY: 205,
    neural_net.OVERLAP_SIZE_KEY: 50,
    neural_net.FIRST_INPUT_ROW_KEY: 105,
    neural_net.LAST_INPUT_ROW_KEY: 309,
    neural_net.FIRST_INPUT_COLUMN_KEY: 0,
    neural_net.LAST_INPUT_COLUMN_KEY: 204
}

FOURTH_PARTIAL_GRID_DICT_BEFORE = {
    neural_net.NUM_FULL_ROWS_KEY: 881,
    neural_net.NUM_FULL_COLUMNS_KEY: 921,
    neural_net.NUM_PARTIAL_ROWS_KEY: 205,
    neural_net.NUM_PARTIAL_COLUMNS_KEY: 205,
    neural_net.OVERLAP_SIZE_KEY: 50,
    neural_net.LAST_INPUT_ROW_KEY: 880,
    neural_net.LAST_INPUT_COLUMN_KEY: 920
}

FOURTH_PARTIAL_GRID_DICT_AFTER = {
    neural_net.NUM_FULL_ROWS_KEY: 881,
    neural_net.NUM_FULL_COLUMNS_KEY: 921,
    neural_net.NUM_PARTIAL_ROWS_KEY: 205,
    neural_net.NUM_PARTIAL_COLUMNS_KEY: 205,
    neural_net.OVERLAP_SIZE_KEY: 50,
    neural_net.FIRST_INPUT_ROW_KEY: -205,
    neural_net.LAST_INPUT_ROW_KEY: -1,
    neural_net.FIRST_INPUT_COLUMN_KEY: -205,
    neural_net.LAST_INPUT_COLUMN_KEY: -1
}


class NeuralNetTests(unittest.TestCase):
    """Each method is a unit test for neural_net.py."""

    def test_reshape_predictor_matrix_1lag_time(self):
        """Ensures correct output from _reshape_predictor_matrix.

        In this case, there is only one lag time.
        """

        this_predictor_matrix = neural_net._reshape_predictor_matrix(
            predictor_matrix=ORIG_PREDICTOR_MATRIX_1LAG_TIME + 0.,
            num_lag_times=1
        )

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, NEW_PREDICTOR_MATRIX_1LAG_TIME,
            atol=TOLERANCE
        ))

    def test_reshape_predictor_matrix_3lag_times(self):
        """Ensures correct output from _reshape_predictor_matrix.

        In this case, there are 3 lag times.
        """

        this_predictor_matrix = neural_net._reshape_predictor_matrix(
            predictor_matrix=ORIG_PREDICTOR_MATRIX_3LAG_TIMES + 0.,
            num_lag_times=3
        )

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, NEW_PREDICTOR_MATRIX_3LAG_TIMES,
            atol=TOLERANCE
        ))

    def test_find_days_zero_lead(self):
        """Ensures correct output from _find_days_with_both_inputs.

        In this case, lead time is zero.
        """

        these_date_strings = neural_net._find_days_with_both_inputs(
            predictor_file_names=PREDICTOR_FILE_NAMES,
            target_file_names=TARGET_FILE_NAMES,
            lead_time_seconds=0,
            lag_times_seconds=numpy.array([0], dtype=int)
        )

        self.assertTrue(these_date_strings == VALID_DATE_STRINGS_ZERO_LEAD)

    def test_find_days_nonzero_lead(self):
        """Ensures correct output from _find_days_with_both_inputs.

        In this case, lead time is non-zero.
        """

        these_date_strings = neural_net._find_days_with_both_inputs(
            predictor_file_names=PREDICTOR_FILE_NAMES,
            target_file_names=TARGET_FILE_NAMES,
            lead_time_seconds=600,
            lag_times_seconds=numpy.array([0], dtype=int)
        )

        self.assertTrue(these_date_strings == VALID_DATE_STRINGS_NONZERO_LEAD)

    def test_get_input_px_for_partial_grid_first(self):
        """Ensures correct output from _get_input_px_for_partial_grid.

        In this case, using first set of inputs.
        """

        this_partial_grid_dict = neural_net._get_input_px_for_partial_grid(
            copy.deepcopy(FIRST_PARTIAL_GRID_DICT_BEFORE)
        )

        self.assertTrue(this_partial_grid_dict == FIRST_PARTIAL_GRID_DICT_AFTER)

    def test_get_input_px_for_partial_grid_second(self):
        """Ensures correct output from _get_input_px_for_partial_grid.

        In this case, using second set of inputs.
        """

        this_partial_grid_dict = neural_net._get_input_px_for_partial_grid(
            copy.deepcopy(SECOND_PARTIAL_GRID_DICT_BEFORE)
        )

        self.assertTrue(
            this_partial_grid_dict == SECOND_PARTIAL_GRID_DICT_AFTER
        )

    def test_get_input_px_for_partial_grid_third(self):
        """Ensures correct output from _get_input_px_for_partial_grid.

        In this case, using third set of inputs.
        """

        this_partial_grid_dict = neural_net._get_input_px_for_partial_grid(
            copy.deepcopy(THIRD_PARTIAL_GRID_DICT_BEFORE)
        )

        self.assertTrue(
            this_partial_grid_dict == THIRD_PARTIAL_GRID_DICT_AFTER
        )

    def test_get_input_px_for_partial_grid_fourth(self):
        """Ensures correct output from _get_input_px_for_partial_grid.

        In this case, using fourth set of inputs.
        """

        this_partial_grid_dict = neural_net._get_input_px_for_partial_grid(
            copy.deepcopy(FOURTH_PARTIAL_GRID_DICT_BEFORE)
        )

        self.assertTrue(
            this_partial_grid_dict == FOURTH_PARTIAL_GRID_DICT_AFTER
        )


if __name__ == '__main__':
    unittest.main()
