"""Unit tests for neural_net.py."""

import copy
import unittest
import numpy
from ml4convection.io import example_io
from ml4convection.machine_learning import neural_net

TOLERANCE = 1e-6

# The following constants are used to test predictor_matrix_to_keras and
# predictor_matrix_from_keras.
ORIG_PREDICTOR_MATRIX_1LAG = numpy.random.normal(
    loc=0., scale=5., size=(30, 50, 100, 7)
)
NEW_PREDICTOR_MATRIX_1LAG_SANS_TIME = ORIG_PREDICTOR_MATRIX_1LAG + 0.
NEW_PREDICTOR_MATRIX_1LAG_WITH_TIME = numpy.expand_dims(
    ORIG_PREDICTOR_MATRIX_1LAG, axis=-2
)

PREDICTOR_MATRIX_FIRST_LAG = numpy.random.normal(
    loc=0., scale=5., size=(10, 50, 100, 7)
)
PREDICTOR_MATRIX_SECOND_LAG = numpy.random.normal(
    loc=1., scale=10., size=(10, 50, 100, 7)
)
PREDICTOR_MATRIX_THIRD_LAG = numpy.random.normal(
    loc=-3., scale=3., size=(10, 50, 100, 7)
)

ORIG_PREDICTOR_MATRIX_3LAGS = numpy.full((0, 50, 100, 7), numpy.nan)
NEW_PREDICTOR_MATRIX_3LAGS_SANS_TIME = numpy.full((10, 50, 100, 21), numpy.nan)
NEW_PREDICTOR_MATRIX_3LAGS_WITH_TIME = numpy.full(
    (10, 50, 100, 3, 7), numpy.nan
)

for i in range(10):
    ORIG_PREDICTOR_MATRIX_3LAGS = numpy.concatenate((
        ORIG_PREDICTOR_MATRIX_3LAGS, PREDICTOR_MATRIX_FIRST_LAG[[i], ...]
    ), axis=0)

    ORIG_PREDICTOR_MATRIX_3LAGS = numpy.concatenate((
        ORIG_PREDICTOR_MATRIX_3LAGS, PREDICTOR_MATRIX_SECOND_LAG[[i], ...]
    ), axis=0)

    ORIG_PREDICTOR_MATRIX_3LAGS = numpy.concatenate((
        ORIG_PREDICTOR_MATRIX_3LAGS, PREDICTOR_MATRIX_THIRD_LAG[[i], ...]
    ), axis=0)

    NEW_PREDICTOR_MATRIX_3LAGS_SANS_TIME[i, ..., 0::3] = (
        PREDICTOR_MATRIX_FIRST_LAG[i, ...]
    )
    NEW_PREDICTOR_MATRIX_3LAGS_SANS_TIME[i, ..., 1::3] = (
        PREDICTOR_MATRIX_SECOND_LAG[i, ...]
    )
    NEW_PREDICTOR_MATRIX_3LAGS_SANS_TIME[i, ..., 2::3] = (
        PREDICTOR_MATRIX_THIRD_LAG[i, ...]
    )

    THIS_MATRIX = numpy.stack((
        PREDICTOR_MATRIX_FIRST_LAG[i, ...],
        PREDICTOR_MATRIX_SECOND_LAG[i, ...],
        PREDICTOR_MATRIX_THIRD_LAG[i, ...]
    ), axis=-2)

    NEW_PREDICTOR_MATRIX_3LAGS_WITH_TIME[i, ...] = THIS_MATRIX

# The following constants are used to test _add_coords_to_predictors.
PREDICTOR_DICT = {
    example_io.LATITUDES_KEY: numpy.array([19, 21, 23, 25, 27], dtype=float),
    example_io.LONGITUDES_KEY: numpy.array(
        [117.5, 118.5, 119.5, 120.5, 121.5, 122.5, 123.5], dtype=float
    )
}

# LATITUDES_DEG_N = numpy.array([
#     [19, 19, 19, 19, 19, 19, 19],
#     [21, 21, 21, 21, 21, 21, 21],
#     [23, 23, 23, 23, 23, 23, 23],
#     [25, 25, 25, 25, 25, 25, 25],
#     [27, 27, 27, 27, 27, 27, 27]
# ], dtype=float)
#
# LONGITUDES_DEG_E = numpy.array([
#     [117.5, 118.5, 119.5, 120.5, 121.5, 122.5, 123.5],
#     [117.5, 118.5, 119.5, 120.5, 121.5, 122.5, 123.5],
#     [117.5, 118.5, 119.5, 120.5, 121.5, 122.5, 123.5],
#     [117.5, 118.5, 119.5, 120.5, 121.5, 122.5, 123.5],
#     [117.5, 118.5, 119.5, 120.5, 121.5, 122.5, 123.5]
# ], dtype=float)

NUM_EXAMPLES = 32
NUM_GRID_ROWS = 5
NUM_GRID_COLUMNS = 7
NUM_INPUT_CHANNELS_SANS_TIME = 14

THESE_DIM = (
    NUM_EXAMPLES, NUM_GRID_ROWS, NUM_GRID_COLUMNS, NUM_INPUT_CHANNELS_SANS_TIME
)
INPUT_MATRIX_SANS_TIME = numpy.random.uniform(low=5., high=6., size=THESE_DIM)

Y_COORD_MATRIX_SANS_TIME = numpy.array([
    [1, 1, 1, 1, 1, 1, 1],
    [3, 3, 3, 3, 3, 3, 3],
    [5, 5, 5, 5, 5, 5, 5],
    [7, 7, 7, 7, 7, 7, 7],
    [9, 9, 9, 9, 9, 9, 9]
], dtype=float)

Y_COORD_MATRIX_SANS_TIME = -3 + Y_COORD_MATRIX_SANS_TIME * (6. / 11)

X_COORD_MATRIX_SANS_TIME = numpy.array([
    [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
    [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
    [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
    [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
    [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
])

X_COORD_MATRIX_SANS_TIME = -3 + X_COORD_MATRIX_SANS_TIME * (6. / 11.5)

Y_COORD_MATRIX_SANS_TIME = numpy.expand_dims(Y_COORD_MATRIX_SANS_TIME, axis=0)
Y_COORD_MATRIX_SANS_TIME = numpy.repeat(
    Y_COORD_MATRIX_SANS_TIME, axis=0, repeats=NUM_EXAMPLES
)
Y_COORD_MATRIX_SANS_TIME = numpy.expand_dims(Y_COORD_MATRIX_SANS_TIME, axis=-1)

X_COORD_MATRIX_SANS_TIME = numpy.expand_dims(X_COORD_MATRIX_SANS_TIME, axis=0)
X_COORD_MATRIX_SANS_TIME = numpy.repeat(
    X_COORD_MATRIX_SANS_TIME, axis=0, repeats=NUM_EXAMPLES
)
X_COORD_MATRIX_SANS_TIME = numpy.expand_dims(X_COORD_MATRIX_SANS_TIME, axis=-1)

OUTPUT_MATRIX_SANS_TIME = numpy.concatenate((
    INPUT_MATRIX_SANS_TIME, X_COORD_MATRIX_SANS_TIME, Y_COORD_MATRIX_SANS_TIME
), axis=-1)

NUM_TIMES = 2
NUM_INPUT_CHANNELS_WITH_TIME = 7

THESE_DIM = (
    NUM_EXAMPLES, NUM_GRID_ROWS, NUM_GRID_COLUMNS, NUM_TIMES,
    NUM_INPUT_CHANNELS_WITH_TIME
)
INPUT_MATRIX_WITH_TIME = numpy.random.uniform(low=5., high=6., size=THESE_DIM)

Y_COORD_MATRIX_WITH_TIME = numpy.expand_dims(Y_COORD_MATRIX_SANS_TIME, axis=-2)
Y_COORD_MATRIX_WITH_TIME = numpy.repeat(
    Y_COORD_MATRIX_WITH_TIME, axis=-2, repeats=NUM_TIMES
)

X_COORD_MATRIX_WITH_TIME = numpy.expand_dims(X_COORD_MATRIX_SANS_TIME, axis=-2)
X_COORD_MATRIX_WITH_TIME = numpy.repeat(
    X_COORD_MATRIX_WITH_TIME, axis=-2, repeats=NUM_TIMES
)

OUTPUT_MATRIX_WITH_TIME = numpy.concatenate((
    INPUT_MATRIX_WITH_TIME, X_COORD_MATRIX_WITH_TIME, Y_COORD_MATRIX_WITH_TIME
), axis=-1)

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

# The following constants are used to test metric_params_to_name,
# metric_name_to_params, and get_metrics.
SCORE_NAMES = [
    neural_net.FSS_NAME, neural_net.BRIER_SCORE_NAME,
    neural_net.CSI_NAME, neural_net.FREQUENCY_BIAS_NAME,
    neural_net.IOU_NAME, neural_net.DICE_COEFF_NAME,
    neural_net.FSS_NAME, neural_net.BRIER_SCORE_NAME,
    neural_net.CSI_NAME, neural_net.FREQUENCY_BIAS_NAME,
    neural_net.IOU_NAME, neural_net.DICE_COEFF_NAME
]
HALF_WINDOW_SIZES_PX = [0, 1, 2, 3, 4, 5, None, None, None, None, None, None]
MIN_RESOLUTIONS_DEG = [
    None, None, None, None, None, None, 0., 0.1, 0.1, 0., 0.1, 0.1
]
MAX_RESOLUTIONS_DEG = [
    None, None, None, None, None, None, 0.5, 0.5, numpy.inf, 0.5, 0.5, numpy.inf
]

METRIC_NAMES = [
    neural_net.FSS_NAME + '_neigh0', neural_net.BRIER_SCORE_NAME + '_neigh1',
    neural_net.CSI_NAME + '_neigh2', neural_net.FREQUENCY_BIAS_NAME + '_neigh3',
    neural_net.IOU_NAME + '_neigh4', neural_net.DICE_COEFF_NAME + '_neigh5',
    neural_net.FSS_NAME + '_0.0000d_0.5000d',
    neural_net.BRIER_SCORE_NAME + '_0.1000d_0.5000d',
    neural_net.CSI_NAME + '_0.1000d_infd',
    neural_net.FREQUENCY_BIAS_NAME + '_0.0000d_0.5000d',
    neural_net.IOU_NAME + '_0.1000d_0.5000d',
    neural_net.DICE_COEFF_NAME + '_0.1000d_infd'
]

MASK_MATRIX = numpy.random.random_integers(low=0, high=1, size=(13, 11))
MASK_MATRIX = MASK_MATRIX.astype(bool)


class NeuralNetTests(unittest.TestCase):
    """Each method is a unit test for neural_net.py."""

    def test_predictor_matrix_to_keras_1lag_sans_time(self):
        """Ensures correct output from predictor_matrix_to_keras.

        In this case, there is one lag time and the output matrix will *not*
        include the time dimension.
        """

        this_predictor_matrix = neural_net.predictor_matrix_to_keras(
            predictor_matrix=ORIG_PREDICTOR_MATRIX_1LAG + 0.,
            num_lag_times=1, add_time_dimension=False
        )

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, NEW_PREDICTOR_MATRIX_1LAG_SANS_TIME,
            atol=TOLERANCE
        ))

    def test_predictor_matrix_from_keras_1lag_sans_time(self):
        """Ensures correct output from predictor_matrix_from_keras.

        In this case, there is one lag time and the input matrix does *not*
        include the time dimension.
        """

        this_predictor_matrix = neural_net.predictor_matrix_from_keras(
            predictor_matrix=NEW_PREDICTOR_MATRIX_1LAG_SANS_TIME + 0.,
            num_lag_times=1
        )

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, ORIG_PREDICTOR_MATRIX_1LAG, atol=TOLERANCE
        ))

    def test_predictor_matrix_to_keras_1lag_with_time(self):
        """Ensures correct output from predictor_matrix_to_keras.

        In this case, there is one lag time and the output matrix will include
        the time dimension.
        """

        this_predictor_matrix = neural_net.predictor_matrix_to_keras(
            predictor_matrix=ORIG_PREDICTOR_MATRIX_1LAG + 0.,
            num_lag_times=1, add_time_dimension=True
        )

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, NEW_PREDICTOR_MATRIX_1LAG_WITH_TIME,
            atol=TOLERANCE
        ))

    def test_predictor_matrix_from_keras_1lag_with_time(self):
        """Ensures correct output from predictor_matrix_from_keras.

        In this case, there is one lag time and the input matrix includes the
        time dimension.
        """

        this_predictor_matrix = neural_net.predictor_matrix_from_keras(
            predictor_matrix=NEW_PREDICTOR_MATRIX_1LAG_WITH_TIME + 0.,
            num_lag_times=1
        )

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, ORIG_PREDICTOR_MATRIX_1LAG, atol=TOLERANCE
        ))

    def test_predictor_matrix_to_keras_3lags_sans_time(self):
        """Ensures correct output from predictor_matrix_to_keras.

        In this case, there are 3 lag times and the output matrix will *not*
        include the time dimension.
        """

        this_predictor_matrix = neural_net.predictor_matrix_to_keras(
            predictor_matrix=ORIG_PREDICTOR_MATRIX_3LAGS + 0.,
            num_lag_times=3, add_time_dimension=False
        )

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, NEW_PREDICTOR_MATRIX_3LAGS_SANS_TIME,
            atol=TOLERANCE
        ))

    def test_predictor_matrix_from_keras_3lags_sans_time(self):
        """Ensures correct output from predictor_matrix_from_keras.

        In this case, there are 3 lag times and the input matrix does *not*
        include the time dimension.
        """

        this_predictor_matrix = neural_net.predictor_matrix_from_keras(
            predictor_matrix=NEW_PREDICTOR_MATRIX_3LAGS_SANS_TIME + 0.,
            num_lag_times=3
        )

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, ORIG_PREDICTOR_MATRIX_3LAGS, atol=TOLERANCE
        ))

    def test_predictor_matrix_to_keras_3lags_with_time(self):
        """Ensures correct output from predictor_matrix_to_keras.

        In this case, there are 3 lag times and the output matrix will include
        the time dimension.
        """

        this_predictor_matrix = neural_net.predictor_matrix_to_keras(
            predictor_matrix=ORIG_PREDICTOR_MATRIX_3LAGS + 0.,
            num_lag_times=3, add_time_dimension=True
        )

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, NEW_PREDICTOR_MATRIX_3LAGS_WITH_TIME,
            atol=TOLERANCE
        ))

    def test_predictor_matrix_from_keras_3lags_with_time(self):
        """Ensures correct output from predictor_matrix_from_keras.

        In this case, there are 3 lag times and the input matrix includes the
        time dimension.
        """

        this_predictor_matrix = neural_net.predictor_matrix_from_keras(
            predictor_matrix=NEW_PREDICTOR_MATRIX_3LAGS_WITH_TIME + 0.,
            num_lag_times=3
        )

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, ORIG_PREDICTOR_MATRIX_3LAGS, atol=TOLERANCE
        ))

    def test_add_coords_sans_time(self):
        """Ensures correct output from _add_coords_to_predictors.

        In this case, data contain no time dimension.
        """

        output_matrix = neural_net._add_coords_to_predictors(
            predictor_matrix=INPUT_MATRIX_SANS_TIME,
            predictor_dict=PREDICTOR_DICT, normalize=True
        )

        self.assertTrue(numpy.allclose(
            OUTPUT_MATRIX_SANS_TIME, output_matrix, atol=TOLERANCE
        ))

    def test_add_coords_with_time(self):
        """Ensures correct output from _add_coords_to_predictors.

        In this case, data contain the time dimension.
        """

        output_matrix = neural_net._add_coords_to_predictors(
            predictor_matrix=INPUT_MATRIX_WITH_TIME,
            predictor_dict=PREDICTOR_DICT, normalize=True
        )

        self.assertTrue(numpy.allclose(
            OUTPUT_MATRIX_WITH_TIME, output_matrix, atol=TOLERANCE
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

    def test_metric_params_to_name(self):
        """Ensures correct output from metric_params_to_name."""

        for k in range(len(METRIC_NAMES)):
            this_metric_name = neural_net.metric_params_to_name(
                score_name=SCORE_NAMES[k],
                half_window_size_px=HALF_WINDOW_SIZES_PX[k],
                min_resolution_deg=MIN_RESOLUTIONS_DEG[k],
                max_resolution_deg=MAX_RESOLUTIONS_DEG[k]
            )

            self.assertTrue(this_metric_name == METRIC_NAMES[k])

    def test_metric_name_to_params(self):
        """Ensures correct output from metric_name_to_params."""

        for k in range(len(METRIC_NAMES)):
            this_param_dict = neural_net.metric_name_to_params(METRIC_NAMES[k])

            self.assertTrue(
                this_param_dict[neural_net.SCORE_NAME_KEY] == SCORE_NAMES[k]
            )
            self.assertTrue(
                this_param_dict[neural_net.HALF_WINDOW_SIZE_KEY] ==
                HALF_WINDOW_SIZES_PX[k]
            )

            this_min_resolution_deg = (
                this_param_dict[neural_net.MIN_RESOLUTION_KEY]
            )
            this_max_resolution_deg = (
                this_param_dict[neural_net.MAX_RESOLUTION_KEY]
            )

            if this_min_resolution_deg is None:
                self.assertTrue(MIN_RESOLUTIONS_DEG[k] is None)
            else:
                self.assertTrue(numpy.isclose(
                    this_min_resolution_deg, MIN_RESOLUTIONS_DEG[k],
                    atol=TOLERANCE
                ))

            if this_max_resolution_deg is None:
                self.assertTrue(MAX_RESOLUTIONS_DEG[k] is None)
            else:
                self.assertTrue(numpy.isclose(
                    this_max_resolution_deg, MAX_RESOLUTIONS_DEG[k],
                    atol=TOLERANCE
                ))

    def test_get_metrics_as_loss(self):
        """Ensures correct output from get_metrics.

        In this case, returning metrics as loss functions.
        """

        neural_net.get_metrics(
            metric_names=METRIC_NAMES, mask_matrix=MASK_MATRIX,
            use_as_loss_function=True
        )

    def test_get_metrics_no_loss(self):
        """Ensures correct output from get_metrics.

        In this case, returning metrics as metrics rather than loss functions.
        """

        neural_net.get_metrics(
            metric_names=METRIC_NAMES, mask_matrix=MASK_MATRIX,
            use_as_loss_function=False
        )


if __name__ == '__main__':
    unittest.main()
