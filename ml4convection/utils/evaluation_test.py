"""Unit tests for evaluation.py."""

import copy
import unittest
import numpy
import xarray
from gewittergefahr.gg_utils import model_evaluation as gg_model_eval
from ml4convection.io import prediction_io
from ml4convection.utils import evaluation

TOLERANCE = 1e-6

# The following constants are used to test dilate_binary_matrix and
# erode_binary_matrix.
FIRST_DILATION_DISTANCE_PX = 35000. / 32463.
SECOND_DILATION_DISTANCE_PX = 50000. / 32463.
THIRD_DILATION_DISTANCE_PX = 100000. / 32463.
FOURTH_DILATION_DISTANCE_PX = 150000. / 32463.

ORIG_MASK_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
], dtype=int)

FIRST_DILATED_MASK_MATRIX = numpy.array([
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
    [0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
], dtype=int)

SECOND_DILATED_MASK_MATRIX = numpy.array([
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
], dtype=int)

THIRD_DILATED_MASK_MATRIX = numpy.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
], dtype=int)

FOURTH_DILATED_MASK_MATRIX = numpy.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
], dtype=int)

FIRST_ERODED_MASK_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
], dtype=int)

SECOND_ERODED_MASK_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
], dtype=int)

THIRD_ERODED_MASK_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=int)

# The following constants are used to test _match_actual_convection_one_time,
# _match_predicted_convection_one_time, _get_reliability_components_one_time,
# _get_fss_components_one_time.
ACTUAL_TARGET_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=int)

PREDICTED_TARGET_MATRIX = numpy.array([
    [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
], dtype=int)

PROBABILITY_MATRIX = numpy.array([
    [0.11, 0.21, 0.31, 0.41, 0.00, 0.00, 0.00, 0.00, 0.41, 0.31, 0.21, 0.11],
    [0.51, 0.61, 0.71, 0.81, 0.00, 0.00, 0.00, 0.00, 0.81, 0.71, 0.61, 0.51],
    [0.21, 0.41, 0.61, 1.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.61, 0.41, 0.21],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.21, 0.41, 0.61, 1.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.61, 0.41, 0.21],
    [0.51, 0.61, 0.71, 0.81, 0.00, 0.00, 0.00, 0.00, 0.81, 0.71, 0.61, 0.51],
    [0.11, 0.21, 0.31, 0.41, 0.00, 0.00, 0.00, 0.00, 0.41, 0.31, 0.21, 0.11]
])

MASK_MATRIX = numpy.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
], dtype=int)

FIRST_MATCHING_DISTANCE_PX = 0.
FIRST_NUM_ACTUAL_ORIENTED_TP = 0
FIRST_NUM_FALSE_NEGATIVES = 16
FIRST_NUM_PREDICTION_ORIENTED_TP = 0
FIRST_NUM_FALSE_POSITIVES = 36

SECOND_MATCHING_DISTANCE_PX = numpy.sqrt(2.)
SECOND_NUM_ACTUAL_ORIENTED_TP = 4
SECOND_NUM_FALSE_NEGATIVES = 12
SECOND_NUM_PREDICTION_ORIENTED_TP = 1
SECOND_NUM_FALSE_POSITIVES = 27

NUM_BINS_FOR_RELIABILITY = 10
FIRST_EXAMPLE_COUNTS = numpy.array([56, 3, 6, 3, 6, 3, 6, 3, 3, 3], dtype=int)
FIRST_MEAN_PROBS = numpy.array([
    0, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 1
])
FIRST_EVENT_FREQUENCIES = numpy.array([16. / 56, 0, 0, 0, 0, 0, 0, 0, 0, 0])

SECOND_EXAMPLE_COUNTS = numpy.array([42, 3, 4, 3, 4, 3, 4, 3, 3, 1], dtype=int)
SECOND_MEAN_PROBS = FIRST_MEAN_PROBS + 0.
SECOND_EVENT_FREQUENCIES = numpy.array([19. / 42, 0, 0, 0, 0, 0, 0, 0, 0, 1])

FIRST_TARGET_SSE = 16.
FIRST_PROBABILITY_SSE = numpy.sum(FIRST_EXAMPLE_COUNTS * FIRST_MEAN_PROBS ** 2)
FIRST_REFERENCE_SSE = FIRST_TARGET_SSE + FIRST_PROBABILITY_SSE
FIRST_ACTUAL_SSE = FIRST_REFERENCE_SSE + 0.

# The following constants are used to test _get_pod, _get_success_ratio,
# _get_csi, and _get_frequency_bias.
CONTINGENCY_TABLE_DICT = {
    evaluation.NUM_ACTUAL_ORIENTED_TP_KEY: 14,
    evaluation.NUM_FALSE_NEGATIVES_KEY: 15,
    evaluation.NUM_PREDICTION_ORIENTED_TP_KEY: 15,
    evaluation.NUM_FALSE_POSITIVES_KEY: 18
}

PROBABILITY_OF_DETECTION = 14. / 29
SUCCESS_RATIO = 15. / 33
CRITICAL_SUCCESS_INDEX = (29. / 14 + 33. / 15 - 1) ** -1
FREQUENCY_BIAS = float(14 * 33) / (29 * 15)

# The following constants are used to test get_basic_scores.
VALID_TIMES_UNIX_SEC = numpy.array([1000], dtype=int)

PREDICTION_DICT = {
    prediction_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC,
    prediction_io.TARGET_MATRIX_KEY:
        numpy.expand_dims(ACTUAL_TARGET_MATRIX, axis=0),
    prediction_io.PROBABILITY_MATRIX_KEY:
        numpy.expand_dims(PROBABILITY_MATRIX, axis=0)
}

MODEL_FILE_NAME = 'foo.bar'
MATCHING_DISTANCE_PX = 0.
TRAINING_EVENT_FREQUENCY = 0.01
SQUARE_FSS_FILTER = True
NUM_PROB_THRESHOLDS = 11

THESE_PROB_THRESHOLDS = numpy.array([
    0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1 + 1e-6
])
THESE_BIN_INDICES = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)

THESE_NUM_ACTUAL_ORIENTED_TP = numpy.array(
    [16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int
)
THESE_NUM_PREDICTION_ORIENTED_TP = numpy.array(
    [16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int
)
THESE_NUM_FALSE_POSITIVES = numpy.array(
    [76, 36, 33, 27, 24, 18, 15, 9, 6, 3, 3, 0], dtype=int
)
THESE_NUM_FALSE_NEGATIVES = numpy.array(
    [0, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16], dtype=int
)

THIS_METADATA_DICT = {
    evaluation.TIME_DIM: VALID_TIMES_UNIX_SEC,
    evaluation.PROBABILITY_THRESHOLD_DIM: THESE_PROB_THRESHOLDS,
    evaluation.RELIABILITY_BIN_DIM: THESE_BIN_INDICES
}

THESE_DIM = (evaluation.TIME_DIM, evaluation.PROBABILITY_THRESHOLD_DIM)
THIS_MAIN_DICT = {
    evaluation.NUM_ACTUAL_ORIENTED_TP_KEY: (
        THESE_DIM, numpy.expand_dims(THESE_NUM_ACTUAL_ORIENTED_TP, axis=0)
    ),
    evaluation.NUM_PREDICTION_ORIENTED_TP_KEY: (
        THESE_DIM, numpy.expand_dims(THESE_NUM_PREDICTION_ORIENTED_TP, axis=0)
    ),
    evaluation.NUM_FALSE_POSITIVES_KEY: (
        THESE_DIM, numpy.expand_dims(THESE_NUM_FALSE_POSITIVES, axis=0)
    ),
    evaluation.NUM_FALSE_NEGATIVES_KEY: (
        THESE_DIM, numpy.expand_dims(THESE_NUM_FALSE_NEGATIVES, axis=0)
    )
}

THESE_DIM = (evaluation.TIME_DIM, evaluation.RELIABILITY_BIN_DIM)
THIS_NEW_DICT = {
    evaluation.NUM_EXAMPLES_KEY: (
        THESE_DIM, numpy.expand_dims(FIRST_EXAMPLE_COUNTS, axis=0)
    ),
    evaluation.MEAN_FORECAST_PROB_KEY: (
        THESE_DIM, numpy.expand_dims(FIRST_MEAN_PROBS, axis=0)
    ),
    evaluation.EVENT_FREQUENCY_KEY: (
        THESE_DIM, numpy.expand_dims(FIRST_EVENT_FREQUENCIES, axis=0)
    )
}
THIS_MAIN_DICT.update(THIS_NEW_DICT)

THESE_DIM = (evaluation.TIME_DIM,)
THIS_NEW_DICT = {
    evaluation.ACTUAL_SSE_KEY: (
        THESE_DIM, numpy.array([FIRST_ACTUAL_SSE])
    ),
    evaluation.REFERENCE_SSE_KEY: (
        THESE_DIM, numpy.array([FIRST_REFERENCE_SSE])
    )
}
THIS_MAIN_DICT.update(THIS_NEW_DICT)

BASIC_SCORE_TABLE_XARRAY = xarray.Dataset(
    data_vars=THIS_MAIN_DICT, coords=THIS_METADATA_DICT
)

BASIC_SCORE_TABLE_XARRAY.attrs[evaluation.MODEL_FILE_KEY] = (
    MODEL_FILE_NAME
)
BASIC_SCORE_TABLE_XARRAY.attrs[evaluation.MATCHING_DISTANCE_KEY] = (
    MATCHING_DISTANCE_PX
)
BASIC_SCORE_TABLE_XARRAY.attrs[evaluation.SQUARE_FSS_FILTER_KEY] = (
    SQUARE_FSS_FILTER
)
BASIC_SCORE_TABLE_XARRAY.attrs[evaluation.TRAINING_EVENT_FREQ_KEY] = (
    TRAINING_EVENT_FREQUENCY
)

# The following constants are used to test concat_basic_score_tables.
SECOND_BASIC_SCORE_TABLE_XARRAY = copy.deepcopy(BASIC_SCORE_TABLE_XARRAY)
THIS_COORD_DICT = {
    evaluation.TIME_DIM: numpy.array([10000], dtype=int)
}
SECOND_BASIC_SCORE_TABLE_XARRAY = (
    SECOND_BASIC_SCORE_TABLE_XARRAY.assign_coords(THIS_COORD_DICT)
)

THIS_METADATA_DICT = {
    evaluation.TIME_DIM: numpy.array([1000, 10000], dtype=int),
    evaluation.PROBABILITY_THRESHOLD_DIM: THESE_PROB_THRESHOLDS,
    evaluation.RELIABILITY_BIN_DIM: THESE_BIN_INDICES
}

THESE_DIM = (evaluation.TIME_DIM, evaluation.PROBABILITY_THRESHOLD_DIM)
THIS_MAIN_DICT = {
    evaluation.NUM_ACTUAL_ORIENTED_TP_KEY: (
        THESE_DIM, numpy.vstack([THESE_NUM_ACTUAL_ORIENTED_TP] * 2)
    ),
    evaluation.NUM_PREDICTION_ORIENTED_TP_KEY: (
        THESE_DIM, numpy.vstack([THESE_NUM_PREDICTION_ORIENTED_TP] * 2)
    ),
    evaluation.NUM_FALSE_POSITIVES_KEY: (
        THESE_DIM, numpy.vstack([THESE_NUM_FALSE_POSITIVES] * 2)
    ),
    evaluation.NUM_FALSE_NEGATIVES_KEY: (
        THESE_DIM, numpy.vstack([THESE_NUM_FALSE_NEGATIVES] * 2)
    )
}

THESE_DIM = (evaluation.TIME_DIM, evaluation.RELIABILITY_BIN_DIM)
THIS_NEW_DICT = {
    evaluation.NUM_EXAMPLES_KEY: (
        THESE_DIM, numpy.vstack([FIRST_EXAMPLE_COUNTS] * 2)
    ),
    evaluation.MEAN_FORECAST_PROB_KEY: (
        THESE_DIM, numpy.vstack([FIRST_MEAN_PROBS] * 2)
    ),
    evaluation.EVENT_FREQUENCY_KEY: (
        THESE_DIM, numpy.vstack([FIRST_EVENT_FREQUENCIES] * 2)
    )
}
THIS_MAIN_DICT.update(THIS_NEW_DICT)

THESE_DIM = (evaluation.TIME_DIM,)
THIS_NEW_DICT = {
    evaluation.ACTUAL_SSE_KEY: (
        THESE_DIM, numpy.array([FIRST_ACTUAL_SSE] * 2)
    ),
    evaluation.REFERENCE_SSE_KEY: (
        THESE_DIM, numpy.array([FIRST_REFERENCE_SSE] * 2)
    )
}
THIS_MAIN_DICT.update(THIS_NEW_DICT)

CONCAT_BASIC_SCORE_TABLE_XARRAY = xarray.Dataset(
    data_vars=THIS_MAIN_DICT, coords=THIS_METADATA_DICT
)

CONCAT_BASIC_SCORE_TABLE_XARRAY.attrs[evaluation.MODEL_FILE_KEY] = (
    MODEL_FILE_NAME
)
CONCAT_BASIC_SCORE_TABLE_XARRAY.attrs[evaluation.MATCHING_DISTANCE_KEY] = (
    MATCHING_DISTANCE_PX
)
CONCAT_BASIC_SCORE_TABLE_XARRAY.attrs[evaluation.SQUARE_FSS_FILTER_KEY] = (
    SQUARE_FSS_FILTER
)
CONCAT_BASIC_SCORE_TABLE_XARRAY.attrs[evaluation.TRAINING_EVENT_FREQ_KEY] = (
    TRAINING_EVENT_FREQUENCY
)

# The following constants are used to test get_advanced_scores.
THIS_METADATA_DICT = {
    evaluation.TIME_DIM: VALID_TIMES_UNIX_SEC,
    evaluation.PROBABILITY_THRESHOLD_DIM: THESE_PROB_THRESHOLDS,
    evaluation.RELIABILITY_BIN_DIM: THESE_BIN_INDICES
}

THESE_CONTINGENCY_TABLES = [
    {
        evaluation.NUM_ACTUAL_ORIENTED_TP_KEY: a_a,
        evaluation.NUM_PREDICTION_ORIENTED_TP_KEY: a_p,
        evaluation.NUM_FALSE_POSITIVES_KEY: b,
        evaluation.NUM_FALSE_NEGATIVES_KEY: c,
    }
    for a_a, a_p, b, c in zip(
        THESE_NUM_ACTUAL_ORIENTED_TP,
        THESE_NUM_PREDICTION_ORIENTED_TP,
        THESE_NUM_FALSE_POSITIVES,
        THESE_NUM_FALSE_NEGATIVES
    )
]

THESE_POD = numpy.array([
    evaluation._get_pod(t) for t in THESE_CONTINGENCY_TABLES
])
THESE_SUCCESS_RATIOS = numpy.array([
    evaluation._get_success_ratio(t) for t in THESE_CONTINGENCY_TABLES
])
THESE_CSI = numpy.array([
    evaluation._get_csi(t) for t in THESE_CONTINGENCY_TABLES
])
THESE_BIASES = numpy.array([
    evaluation._get_frequency_bias(t) for t in THESE_CONTINGENCY_TABLES
])

THESE_DIM = (evaluation.PROBABILITY_THRESHOLD_DIM,)
THIS_MAIN_DICT = {
    evaluation.NUM_ACTUAL_ORIENTED_TP_KEY:
        (THESE_DIM, THESE_NUM_ACTUAL_ORIENTED_TP + 0),
    evaluation.NUM_PREDICTION_ORIENTED_TP_KEY:
        (THESE_DIM, THESE_NUM_PREDICTION_ORIENTED_TP + 0),
    evaluation.NUM_FALSE_POSITIVES_KEY:
        (THESE_DIM, THESE_NUM_FALSE_POSITIVES + 0),
    evaluation.NUM_FALSE_NEGATIVES_KEY:
        (THESE_DIM, THESE_NUM_FALSE_NEGATIVES + 0),
    evaluation.POD_KEY: (THESE_DIM, THESE_POD + 0.),
    evaluation.SUCCESS_RATIO_KEY: (THESE_DIM, THESE_SUCCESS_RATIOS + 0.),
    evaluation.CSI_KEY: (THESE_DIM, THESE_CSI + 0.),
    evaluation.FREQUENCY_BIAS_KEY: (THESE_DIM, THESE_BIASES + 0.)
}

THESE_DIM = (evaluation.RELIABILITY_BIN_DIM,)
THIS_NEW_DICT = {
    evaluation.NUM_EXAMPLES_KEY: (THESE_DIM, FIRST_EXAMPLE_COUNTS + 0),
    evaluation.MEAN_FORECAST_PROB_KEY: (THESE_DIM, FIRST_MEAN_PROBS + 0.),
    evaluation.EVENT_FREQUENCY_KEY: (THESE_DIM, FIRST_EVENT_FREQUENCIES + 0.)
}
THIS_MAIN_DICT.update(THIS_NEW_DICT)

ADVANCED_SCORE_TABLE_XARRAY = xarray.Dataset(
    data_vars=THIS_MAIN_DICT, coords=THIS_METADATA_DICT
)

ADVANCED_SCORE_TABLE_XARRAY.attrs[evaluation.MODEL_FILE_KEY] = (
    MODEL_FILE_NAME
)
ADVANCED_SCORE_TABLE_XARRAY.attrs[evaluation.MATCHING_DISTANCE_KEY] = (
    MATCHING_DISTANCE_PX
)
ADVANCED_SCORE_TABLE_XARRAY.attrs[evaluation.SQUARE_FSS_FILTER_KEY] = (
    SQUARE_FSS_FILTER
)
ADVANCED_SCORE_TABLE_XARRAY.attrs[evaluation.TRAINING_EVENT_FREQ_KEY] = (
    TRAINING_EVENT_FREQUENCY
)
ADVANCED_SCORE_TABLE_XARRAY.attrs[evaluation.FSS_KEY] = 0.

THIS_BSS_DICT = gg_model_eval.get_brier_skill_score(
    mean_forecast_prob_by_bin=FIRST_MEAN_PROBS,
    mean_observed_label_by_bin=FIRST_EVENT_FREQUENCIES,
    num_examples_by_bin=FIRST_EXAMPLE_COUNTS,
    climatology=TRAINING_EVENT_FREQUENCY
)

ADVANCED_SCORE_TABLE_XARRAY.attrs[evaluation.BRIER_SKILL_SCORE_KEY] = (
    THIS_BSS_DICT[gg_model_eval.BSS_KEY]
)
ADVANCED_SCORE_TABLE_XARRAY.attrs[evaluation.BRIER_SCORE_KEY] = (
    THIS_BSS_DICT[gg_model_eval.BRIER_SCORE_KEY]
)
ADVANCED_SCORE_TABLE_XARRAY.attrs[evaluation.RELIABILITY_KEY] = (
    THIS_BSS_DICT[gg_model_eval.RELIABILITY_KEY]
)
ADVANCED_SCORE_TABLE_XARRAY.attrs[evaluation.RESOLUTION_KEY] = (
    THIS_BSS_DICT[gg_model_eval.RESOLUTION_KEY]
)

THIS_METADATA_DICT = {
    evaluation.TIME_DIM: numpy.array([1000, 10000], dtype=int),
    evaluation.PROBABILITY_THRESHOLD_DIM: THESE_PROB_THRESHOLDS,
    evaluation.RELIABILITY_BIN_DIM: THESE_BIN_INDICES
}

THESE_DIM = (evaluation.PROBABILITY_THRESHOLD_DIM,)
THIS_MAIN_DICT = {
    evaluation.NUM_ACTUAL_ORIENTED_TP_KEY:
        (THESE_DIM, 2 * THESE_NUM_ACTUAL_ORIENTED_TP),
    evaluation.NUM_PREDICTION_ORIENTED_TP_KEY:
        (THESE_DIM, 2 * THESE_NUM_PREDICTION_ORIENTED_TP),
    evaluation.NUM_FALSE_POSITIVES_KEY:
        (THESE_DIM, 2 * THESE_NUM_FALSE_POSITIVES),
    evaluation.NUM_FALSE_NEGATIVES_KEY:
        (THESE_DIM, 2 * THESE_NUM_FALSE_NEGATIVES),
    evaluation.POD_KEY: (THESE_DIM, THESE_POD + 0.),
    evaluation.SUCCESS_RATIO_KEY: (THESE_DIM, THESE_SUCCESS_RATIOS + 0.),
    evaluation.CSI_KEY: (THESE_DIM, THESE_CSI + 0.),
    evaluation.FREQUENCY_BIAS_KEY: (THESE_DIM, THESE_BIASES + 0.)
}

THESE_DIM = (evaluation.RELIABILITY_BIN_DIM,)
THIS_NEW_DICT = {
    evaluation.NUM_EXAMPLES_KEY: (THESE_DIM, 2 * FIRST_EXAMPLE_COUNTS),
    evaluation.MEAN_FORECAST_PROB_KEY: (THESE_DIM, FIRST_MEAN_PROBS + 0.),
    evaluation.EVENT_FREQUENCY_KEY: (THESE_DIM, FIRST_EVENT_FREQUENCIES + 0.)
}
THIS_MAIN_DICT.update(THIS_NEW_DICT)

CONCAT_ADVANCED_SCORE_TABLE_XARRAY = xarray.Dataset(
    data_vars=THIS_MAIN_DICT, coords=THIS_METADATA_DICT
)

for THIS_KEY in [
        evaluation.MODEL_FILE_KEY, evaluation.MATCHING_DISTANCE_KEY,
        evaluation.SQUARE_FSS_FILTER_KEY, evaluation.TRAINING_EVENT_FREQ_KEY,
        evaluation.FSS_KEY, evaluation.BRIER_SKILL_SCORE_KEY,
        evaluation.BRIER_SCORE_KEY, evaluation.RELIABILITY_KEY,
        evaluation.RESOLUTION_KEY
]:
    CONCAT_ADVANCED_SCORE_TABLE_XARRAY.attrs[THIS_KEY] = (
        ADVANCED_SCORE_TABLE_XARRAY.attrs[THIS_KEY]
    )

# The following constants are used to test find_basic_score_file,
# basic_file_name_to_date, and find_advanced_score_file.
TOP_DIRECTORY_NAME = 'foo'
VALID_DATE_STRING = '19670502'
BASIC_SCORE_FILE_NAME = 'foo/1967/basic_scores_19670502.p'

ADVANCED_SCORE_FILE_NAME_ALL = 'foo/advanced_scores.p'
ADVANCED_SCORE_FILE_NAME_HOUR0 = 'foo/advanced_scores_hour=00.p'
ADVANCED_SCORE_FILE_NAME_MONTH1 = 'foo/advanced_scores_month=01.p'


def _compare_basic_score_tables(first_table, second_table):
    """Compares two xarray tables with basic scores.

    :param first_table: First table.
    :param second_table: Second table.
    :return: are_tables_equal: Boolean flag.
    """

    float_keys = [
        evaluation.EVENT_FREQUENCY_KEY, evaluation.MEAN_FORECAST_PROB_KEY,
        evaluation.ACTUAL_SSE_KEY, evaluation.REFERENCE_SSE_KEY
    ]
    integer_keys = [
        evaluation.NUM_ACTUAL_ORIENTED_TP_KEY,
        evaluation.NUM_PREDICTION_ORIENTED_TP_KEY,
        evaluation.NUM_FALSE_POSITIVES_KEY,
        evaluation.NUM_FALSE_NEGATIVES_KEY,
        evaluation.NUM_EXAMPLES_KEY
    ]

    for this_key in float_keys:
        if not numpy.allclose(
                first_table[this_key].values, second_table[this_key].values,
                atol=TOLERANCE
        ):
            return False

    for this_key in integer_keys:
        if not numpy.array_equal(
                first_table[this_key].values, second_table[this_key].values
        ):
            return False

    float_keys = [evaluation.PROBABILITY_THRESHOLD_DIM]
    integer_keys = [evaluation.TIME_DIM, evaluation.RELIABILITY_BIN_DIM]

    for this_key in float_keys:
        if not numpy.allclose(
                first_table.coords[this_key].values,
                second_table.coords[this_key].values,
                atol=TOLERANCE
        ):
            return False

    for this_key in integer_keys:
        if not numpy.array_equal(
                first_table.coords[this_key].values,
                second_table.coords[this_key].values
        ):
            return False

    float_keys = [
        evaluation.MATCHING_DISTANCE_KEY, evaluation.TRAINING_EVENT_FREQ_KEY
    ]
    exact_keys = [evaluation.MODEL_FILE_KEY, evaluation.SQUARE_FSS_FILTER_KEY]

    for this_key in float_keys:
        if not numpy.isclose(
                first_table.attrs[this_key], second_table.attrs[this_key],
                atol=TOLERANCE
        ):
            return False

    for this_key in exact_keys:
        if first_table.attrs[this_key] != second_table.attrs[this_key]:
            return False

    return True


def _compare_advanced_score_tables(first_table, second_table):
    """Compares two xarray tables with advanced scores.

    :param first_table: First table.
    :param second_table: Second table.
    :return: are_tables_equal: Boolean flag.
    """

    float_keys = [
        evaluation.POD_KEY, evaluation.SUCCESS_RATIO_KEY,
        evaluation.CSI_KEY, evaluation.FREQUENCY_BIAS_KEY,
        evaluation.EVENT_FREQUENCY_KEY, evaluation.MEAN_FORECAST_PROB_KEY
    ]
    integer_keys = [
        evaluation.NUM_ACTUAL_ORIENTED_TP_KEY,
        evaluation.NUM_PREDICTION_ORIENTED_TP_KEY,
        evaluation.NUM_FALSE_POSITIVES_KEY,
        evaluation.NUM_FALSE_NEGATIVES_KEY,
        evaluation.NUM_EXAMPLES_KEY
    ]

    for this_key in float_keys:
        if not numpy.allclose(
                first_table[this_key].values, second_table[this_key].values,
                atol=TOLERANCE, equal_nan=True
        ):
            return False

    for this_key in integer_keys:
        if not numpy.array_equal(
                first_table[this_key].values, second_table[this_key].values
        ):
            return False

    float_keys = [evaluation.PROBABILITY_THRESHOLD_DIM]
    integer_keys = [evaluation.TIME_DIM, evaluation.RELIABILITY_BIN_DIM]

    for this_key in float_keys:
        if not numpy.allclose(
                first_table.coords[this_key].values,
                second_table.coords[this_key].values,
                atol=TOLERANCE
        ):
            return False

    for this_key in integer_keys:
        if not numpy.array_equal(
                first_table.coords[this_key].values,
                second_table.coords[this_key].values
        ):
            return False

    float_keys = [
        evaluation.MATCHING_DISTANCE_KEY, evaluation.TRAINING_EVENT_FREQ_KEY,
        evaluation.BRIER_SKILL_SCORE_KEY, evaluation.BRIER_SCORE_KEY,
        evaluation.RELIABILITY_KEY, evaluation.RESOLUTION_KEY,
        evaluation.FSS_KEY
    ]
    exact_keys = [evaluation.MODEL_FILE_KEY, evaluation.SQUARE_FSS_FILTER_KEY]

    for this_key in float_keys:
        if not numpy.isclose(
                first_table.attrs[this_key], second_table.attrs[this_key],
                atol=TOLERANCE
        ):
            return False

    for this_key in exact_keys:
        if first_table.attrs[this_key] != second_table.attrs[this_key]:
            return False

    return True


class EvaluationTests(unittest.TestCase):
    """Each method is a unit test for evaluation.py."""

    def test_dilate_binary_matrix_first(self):
        """Ensures correct output from dilate_binary_matrix.

        In this case, using first dilation distance.
        """

        this_matrix = evaluation.dilate_binary_matrix(
            binary_matrix=ORIG_MASK_MATRIX,
            buffer_distance_px=FIRST_DILATION_DISTANCE_PX
        )

        self.assertTrue(numpy.array_equal(
            this_matrix, FIRST_DILATED_MASK_MATRIX
        ))

    def test_dilate_binary_matrix_second(self):
        """Ensures correct output from dilate_binary_matrix.

        In this case, using second dilation distance.
        """

        this_matrix = evaluation.dilate_binary_matrix(
            binary_matrix=ORIG_MASK_MATRIX,
            buffer_distance_px=SECOND_DILATION_DISTANCE_PX
        )

        self.assertTrue(numpy.array_equal(
            this_matrix, SECOND_DILATED_MASK_MATRIX
        ))

    def test_dilate_binary_matrix_third(self):
        """Ensures correct output from dilate_binary_matrix.

        In this case, using third dilation distance.
        """

        this_matrix = evaluation.dilate_binary_matrix(
            binary_matrix=ORIG_MASK_MATRIX,
            buffer_distance_px=THIRD_DILATION_DISTANCE_PX
        )

        self.assertTrue(numpy.array_equal(
            this_matrix, THIRD_DILATED_MASK_MATRIX
        ))

    def test_dilate_binary_matrix_fourth(self):
        """Ensures correct output from dilate_binary_matrix.

        In this case, using fourth dilation distance.
        """

        this_matrix = evaluation.dilate_binary_matrix(
            binary_matrix=ORIG_MASK_MATRIX,
            buffer_distance_px=FOURTH_DILATION_DISTANCE_PX
        )

        self.assertTrue(numpy.array_equal(
            this_matrix, FOURTH_DILATED_MASK_MATRIX
        ))

    def test_erode_binary_matrix_first(self):
        """Ensures correct output from erode_binary_matrix.

        In this case, using first erosion distance.
        """

        this_matrix = evaluation.erode_binary_matrix(
            binary_matrix=ORIG_MASK_MATRIX,
            buffer_distance_px=FIRST_DILATION_DISTANCE_PX
        )

        self.assertTrue(numpy.array_equal(
            this_matrix, FIRST_ERODED_MASK_MATRIX
        ))

    def test_erode_binary_matrix_second(self):
        """Ensures correct output from erode_binary_matrix.

        In this case, using second erosion distance.
        """

        this_matrix = evaluation.erode_binary_matrix(
            binary_matrix=ORIG_MASK_MATRIX,
            buffer_distance_px=SECOND_DILATION_DISTANCE_PX
        )

        self.assertTrue(numpy.array_equal(
            this_matrix, SECOND_ERODED_MASK_MATRIX
        ))

    def test_erode_binary_matrix_third(self):
        """Ensures correct output from erode_binary_matrix.

        In this case, using third erosion distance.
        """

        this_matrix = evaluation.erode_binary_matrix(
            binary_matrix=ORIG_MASK_MATRIX,
            buffer_distance_px=THIRD_DILATION_DISTANCE_PX
        )

        self.assertTrue(numpy.array_equal(
            this_matrix, THIRD_ERODED_MASK_MATRIX
        ))

    def test_match_actual_convection_one_time_first(self):
        """Ensures correct output from _match_actual_convection_one_time.

        In this case, using first matching distance.
        """

        this_num_actual_oriented_tp, this_num_false_negatives = (
            evaluation._match_actual_convection_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                predicted_target_matrix=PREDICTED_TARGET_MATRIX,
                matching_distance_px=FIRST_MATCHING_DISTANCE_PX,
                eroded_eval_mask_matrix=MASK_MATRIX
            )
        )

        self.assertTrue(
            this_num_actual_oriented_tp == FIRST_NUM_ACTUAL_ORIENTED_TP
        )
        self.assertTrue(this_num_false_negatives == FIRST_NUM_FALSE_NEGATIVES)

    def test_match_predicted_convection_one_time_first(self):
        """Ensures correct output from _match_predicted_convection_one_time.

        In this case, using first matching distance.
        """

        this_num_prediction_oriented_tp, this_num_false_positives = (
            evaluation._match_predicted_convection_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                predicted_target_matrix=PREDICTED_TARGET_MATRIX,
                matching_distance_px=FIRST_MATCHING_DISTANCE_PX,
                eroded_eval_mask_matrix=MASK_MATRIX
            )
        )

        self.assertTrue(
            this_num_prediction_oriented_tp == FIRST_NUM_PREDICTION_ORIENTED_TP
        )
        self.assertTrue(this_num_false_positives == FIRST_NUM_FALSE_POSITIVES)

    def test_match_actual_convection_one_time_second(self):
        """Ensures correct output from _match_actual_convection_one_time.

        In this case, using second matching distance.
        """

        this_num_actual_oriented_tp, this_num_false_negatives = (
            evaluation._match_actual_convection_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                predicted_target_matrix=PREDICTED_TARGET_MATRIX,
                matching_distance_px=SECOND_MATCHING_DISTANCE_PX,
                eroded_eval_mask_matrix=MASK_MATRIX
            )
        )

        self.assertTrue(
            this_num_actual_oriented_tp == SECOND_NUM_ACTUAL_ORIENTED_TP
        )
        self.assertTrue(this_num_false_negatives == SECOND_NUM_FALSE_NEGATIVES)

    def test_match_predicted_convection_one_time_second(self):
        """Ensures correct output from _match_predicted_convection_one_time.

        In this case, using second matching distance.
        """

        this_mask_matrix = evaluation.erode_binary_matrix(
            binary_matrix=MASK_MATRIX,
            buffer_distance_px=SECOND_MATCHING_DISTANCE_PX
        )

        this_num_prediction_oriented_tp, this_num_false_positives = (
            evaluation._match_predicted_convection_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                predicted_target_matrix=PREDICTED_TARGET_MATRIX,
                matching_distance_px=SECOND_MATCHING_DISTANCE_PX,
                eroded_eval_mask_matrix=this_mask_matrix
            )
        )

        self.assertTrue(
            this_num_prediction_oriented_tp == SECOND_NUM_PREDICTION_ORIENTED_TP
        )
        self.assertTrue(this_num_false_positives == SECOND_NUM_FALSE_POSITIVES)

    def test_get_reliability_components_first(self):
        """Ensures correct output from _get_reliability_components_one_time.

        In this case, using first matching distance.
        """

        these_example_counts, these_mean_probs, these_event_frequencies = (
            evaluation._get_reliability_components_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                probability_matrix=PROBABILITY_MATRIX,
                matching_distance_px=FIRST_MATCHING_DISTANCE_PX,
                num_bins=NUM_BINS_FOR_RELIABILITY,
                eroded_eval_mask_matrix=MASK_MATRIX
            )
        )

        self.assertTrue(numpy.array_equal(
            these_example_counts, FIRST_EXAMPLE_COUNTS
        ))
        self.assertTrue(numpy.allclose(
            these_mean_probs, FIRST_MEAN_PROBS, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_event_frequencies, FIRST_EVENT_FREQUENCIES, atol=TOLERANCE
        ))

    def test_get_reliability_components_second(self):
        """Ensures correct output from _get_reliability_components_one_time.

        In this case, using second matching distance.
        """

        this_mask_matrix = evaluation.erode_binary_matrix(
            binary_matrix=MASK_MATRIX,
            buffer_distance_px=SECOND_MATCHING_DISTANCE_PX
        )

        these_example_counts, these_mean_probs, these_event_frequencies = (
            evaluation._get_reliability_components_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                probability_matrix=PROBABILITY_MATRIX,
                matching_distance_px=SECOND_MATCHING_DISTANCE_PX,
                num_bins=NUM_BINS_FOR_RELIABILITY,
                eroded_eval_mask_matrix=this_mask_matrix
            )
        )

        self.assertTrue(numpy.array_equal(
            these_example_counts, SECOND_EXAMPLE_COUNTS
        ))
        self.assertTrue(numpy.allclose(
            these_mean_probs, SECOND_MEAN_PROBS, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_event_frequencies, SECOND_EVENT_FREQUENCIES, atol=TOLERANCE
        ))

    def test_get_fss_components_first(self):
        """Ensures correct output from _get_fss_components_one_time.

        In this case, using first matching distance.
        """

        this_actual_sse, this_reference_sse = (
            evaluation._get_fss_components_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                probability_matrix=PROBABILITY_MATRIX,
                matching_distance_px=FIRST_MATCHING_DISTANCE_PX,
                eroded_eval_mask_matrix=MASK_MATRIX, square_filter=True
            )
        )

        self.assertTrue(numpy.isclose(
            this_actual_sse, FIRST_ACTUAL_SSE, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_reference_sse, FIRST_REFERENCE_SSE, atol=TOLERANCE
        ))

    def test_get_pod(self):
        """Ensures correct output from _get_pod."""

        self.assertTrue(numpy.isclose(
            evaluation._get_pod(CONTINGENCY_TABLE_DICT),
            PROBABILITY_OF_DETECTION, atol=TOLERANCE
        ))

    def test_get_success_ratio(self):
        """Ensures correct output from _get_success_ratio."""

        self.assertTrue(numpy.isclose(
            evaluation._get_success_ratio(CONTINGENCY_TABLE_DICT),
            SUCCESS_RATIO, atol=TOLERANCE
        ))

    def test_get_csi(self):
        """Ensures correct output from _get_csi."""

        self.assertTrue(numpy.isclose(
            evaluation._get_csi(CONTINGENCY_TABLE_DICT),
            CRITICAL_SUCCESS_INDEX, atol=TOLERANCE
        ))

    def test_get_frequency_bias(self):
        """Ensures correct output from _get_frequency_bias."""

        self.assertTrue(numpy.isclose(
            evaluation._get_frequency_bias(CONTINGENCY_TABLE_DICT),
            FREQUENCY_BIAS, atol=TOLERANCE
        ))

    def test_get_basic_scores(self):
        """Ensures correct output from get_basic_scores."""

        this_score_table_xarray = evaluation.get_basic_scores(
            prediction_file_name=None,
            matching_distance_px=MATCHING_DISTANCE_PX,
            training_event_frequency=TRAINING_EVENT_FREQUENCY,
            square_fss_filter=SQUARE_FSS_FILTER,
            num_prob_thresholds=NUM_PROB_THRESHOLDS,
            num_bins_for_reliability=NUM_BINS_FOR_RELIABILITY,
            test_mode=True, prediction_dict=PREDICTION_DICT,
            eval_mask_matrix=MASK_MATRIX, model_file_name=MODEL_FILE_NAME
        )

        self.assertTrue(_compare_basic_score_tables(
            this_score_table_xarray, BASIC_SCORE_TABLE_XARRAY
        ))

    def test_concat_basic_score_tables(self):
        """Ensures correct output from concat_basic_score_tables."""

        this_score_table_xarray = evaluation.concat_basic_score_tables(
            [BASIC_SCORE_TABLE_XARRAY, SECOND_BASIC_SCORE_TABLE_XARRAY]
        )

        self.assertTrue(_compare_basic_score_tables(
            this_score_table_xarray, CONCAT_BASIC_SCORE_TABLE_XARRAY
        ))

    def test_subset_basic_scores_hour0(self):
        """Ensures correct output from subset_basic_scores_by_hour.

        In this case, desired hour is 0.
        """

        this_score_table_xarray = evaluation.subset_basic_scores_by_hour(
            basic_score_table_xarray=CONCAT_BASIC_SCORE_TABLE_XARRAY,
            desired_hour=0
        )

        self.assertTrue(_compare_basic_score_tables(
            this_score_table_xarray, BASIC_SCORE_TABLE_XARRAY
        ))

    def test_subset_basic_scores_hour2(self):
        """Ensures correct output from subset_basic_scores_by_hour.

        In this case, desired hour is 2.
        """

        this_score_table_xarray = evaluation.subset_basic_scores_by_hour(
            basic_score_table_xarray=CONCAT_BASIC_SCORE_TABLE_XARRAY,
            desired_hour=2
        )

        self.assertTrue(_compare_basic_score_tables(
            this_score_table_xarray, SECOND_BASIC_SCORE_TABLE_XARRAY
        ))

    def test_subset_basic_scores_hour23(self):
        """Ensures correct output from subset_basic_scores_by_hour.

        In this case, desired hour is 23.
        """

        this_score_table_xarray = evaluation.subset_basic_scores_by_hour(
            basic_score_table_xarray=CONCAT_BASIC_SCORE_TABLE_XARRAY,
            desired_hour=23
        )

        self.assertTrue(
            this_score_table_xarray.coords[evaluation.TIME_DIM].size == 0
        )

    def test_get_advanced_scores_1time(self):
        """Ensures correct output from get_advanced_scores.

        In this case, input table contains one time.
        """

        this_score_table_xarray = evaluation.get_advanced_scores(
            BASIC_SCORE_TABLE_XARRAY
        )

        self.assertTrue(_compare_advanced_score_tables(
            this_score_table_xarray, ADVANCED_SCORE_TABLE_XARRAY
        ))

    def test_get_advanced_scores_2times(self):
        """Ensures correct output from get_advanced_scores.

        In this case, input table contains two times.
        """

        this_score_table_xarray = evaluation.get_advanced_scores(
            CONCAT_BASIC_SCORE_TABLE_XARRAY
        )

        self.assertTrue(_compare_advanced_score_tables(
            this_score_table_xarray, CONCAT_ADVANCED_SCORE_TABLE_XARRAY
        ))

    def test_find_basic_score_file(self):
        """Ensures correct output from find_basic_score_file."""

        this_file_name = evaluation.find_basic_score_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == BASIC_SCORE_FILE_NAME)

    def test_basic_file_name_to_date(self):
        """Ensures correct output from basic_file_name_to_date."""

        self.assertTrue(
            evaluation.basic_file_name_to_date(BASIC_SCORE_FILE_NAME) ==
            VALID_DATE_STRING
        )

    def test_find_advanced_score_file_all(self):
        """Ensures correct output from find_advanced_score_file.

        In this case, file contains all hours and months.
        """

        this_file_name = evaluation.find_advanced_score_file(
            directory_name=TOP_DIRECTORY_NAME, month=None, hour=None,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == ADVANCED_SCORE_FILE_NAME_ALL)

    def test_find_advanced_score_file_hour0(self):
        """Ensures correct output from find_advanced_score_file.

        In this case, file contains only hour 0.
        """

        this_file_name = evaluation.find_advanced_score_file(
            directory_name=TOP_DIRECTORY_NAME, month=None, hour=0,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == ADVANCED_SCORE_FILE_NAME_HOUR0)

    def test_find_advanced_score_file_month1(self):
        """Ensures correct output from find_advanced_score_file.

        In this case, file contains only January.
        """

        this_file_name = evaluation.find_advanced_score_file(
            directory_name=TOP_DIRECTORY_NAME, month=1, hour=None,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == ADVANCED_SCORE_FILE_NAME_MONTH1)


if __name__ == '__main__':
    unittest.main()
