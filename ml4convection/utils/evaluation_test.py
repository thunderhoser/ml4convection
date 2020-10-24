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

TOY_MASK_MATRIX = numpy.array([
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

FIRST_EROSION_DISTANCE_PX = 35000. / 32463.
SECOND_EROSION_DISTANCE_PX = 50000. / 32463.
THIRD_EROSION_DISTANCE_PX = 100000. / 32463.

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
# _get_fss_components_one_time, _get_bss_components_one_time.
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

MEAN_PROB_MATRIX = numpy.array([
    [0.11, 0.21, 0.31, 0.41, 0.00, 0.00, 0.00, 0.00, -1.0, -1.0, -1.0, -1.0],
    [0.51, 0.61, 0.71, 0.81, 0.00, 0.00, 0.00, 0.00, -1.0, -1.0, -1.0, -1.0],
    [0.21, 0.41, 0.61, 1.00, 0.00, 0.00, 0.00, 0.00, -1.0, -1.0, -1.0, -1.0],
    [-1.0, -1.0, -1.0, -1.0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [-1.0, -1.0, -1.0, -1.0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [-1.0, -1.0, -1.0, -1.0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [-1.0, -1.0, -1.0, -1.0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.21, 0.41, 0.61, 1.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.61, 0.41, 0.21],
    [0.51, 0.61, 0.71, 0.81, 0.00, 0.00, 0.00, 0.00, 0.81, 0.71, 0.61, 0.51],
    [0.11, 0.21, 0.31, 0.41, 0.00, 0.00, 0.00, 0.00, 0.41, 0.31, 0.21, 0.11]
])

MEAN_PROB_MATRIX[MEAN_PROB_MATRIX < 0] = numpy.nan

FIRST_MATCHING_DISTANCE_PX = 0.
# FIRST_NUM_ACTUAL_ORIENTED_TP = 0
# FIRST_NUM_FALSE_NEGATIVES = 16
# FIRST_NUM_PREDICTION_ORIENTED_TP = 0
# FIRST_NUM_FALSE_POSITIVES = 36

FIRST_FANCY_PREDICTION_MATRIX = numpy.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1],
    [-1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1],
    [-1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1],
    [-1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
], dtype=int)

FIRST_FANCY_TARGET_MATRIX = numpy.array([
    [0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0]
], dtype=int)

SECOND_MATCHING_DISTANCE_PX = numpy.sqrt(2.)
# SECOND_NUM_ACTUAL_ORIENTED_TP = 1
# SECOND_NUM_FALSE_NEGATIVES = 10
# SECOND_NUM_PREDICTION_ORIENTED_TP = 1
# SECOND_NUM_FALSE_POSITIVES = 27

SECOND_FANCY_PREDICTION_MATRIX = numpy.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, 0, 0, 1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
], dtype=int)

SECOND_FANCY_TARGET_MATRIX = numpy.array([
    [0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 0, 0],
    [0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0]
], dtype=int)

# The following constants are used to test _get_reliability_components_one_time.
NUM_BINS_FOR_RELIABILITY = 10

THIS_BIN_INDEX_MATRIX = numpy.array([
    [1, 2, 3, 4, 0, 0, 0, 0, -1, -1, -1, -1],
    [5, 6, 7, 8, 0, 0, 0, 0, -1, -1, -1, -1],
    [2, 4, 6, 9, 0, 0, 0, 0, -1, -1, -1, -1],
    [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 4, 6, 9, 0, 0, 0, 0, 9, 6, 4, 2],
    [5, 6, 7, 8, 0, 0, 0, 0, 8, 7, 6, 5],
    [1, 2, 3, 4, 0, 0, 0, 0, 4, 3, 2, 1]
], dtype=int)

THESE_DIM = THIS_BIN_INDEX_MATRIX.shape + (NUM_BINS_FOR_RELIABILITY,)
FIRST_EXAMPLE_COUNT_MATRIX = numpy.full(THESE_DIM, 0, dtype=int)

for j in range(NUM_BINS_FOR_RELIABILITY):
    FIRST_EXAMPLE_COUNT_MATRIX[..., j] = (
        (THIS_BIN_INDEX_MATRIX == j).astype(int)
    )

THESE_MEAN_PROBS = numpy.array([
    0, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 1
])

FIRST_SUMMED_PROB_MATRIX = numpy.full(THESE_DIM, numpy.nan)
FIRST_POS_EXAMPLE_COUNT_MATRIX = numpy.full(THESE_DIM, 0, dtype=int)

for j in range(NUM_BINS_FOR_RELIABILITY):
    n = FIRST_EXAMPLE_COUNT_MATRIX[..., j]

    FIRST_SUMMED_PROB_MATRIX[..., j][n > 0] = THESE_MEAN_PROBS[j]
    FIRST_POS_EXAMPLE_COUNT_MATRIX[..., j][n > 0] = 0

FIRST_POS_EXAMPLE_COUNT_MATRIX[3:7, 4:8, 0] = 1

THIS_BIN_INDEX_MATRIX = numpy.array([
    [1, 2, 3, 4, 0, 0, 0, -1, -1, -1, -1, -1],
    [5, 6, 7, 8, 0, 0, 0, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, 0, 0, 0, 9, 6, 4, 2],
    [5, 6, 7, 8, 0, 0, 0, 0, 8, 7, 6, 5],
    [1, 2, 3, 4, 0, 0, 0, 0, 4, 3, 2, 1]
], dtype=int)

SECOND_EXAMPLE_COUNT_MATRIX = numpy.full(THESE_DIM, 0, dtype=int)

for j in range(NUM_BINS_FOR_RELIABILITY):
    SECOND_EXAMPLE_COUNT_MATRIX[..., j] = (
        (THIS_BIN_INDEX_MATRIX == j).astype(int)
    )

SECOND_SUMMED_PROB_MATRIX = numpy.full(THESE_DIM, numpy.nan)
SECOND_POS_EXAMPLE_COUNT_MATRIX = numpy.full(THESE_DIM, 0, dtype=int)

for j in range(NUM_BINS_FOR_RELIABILITY):
    n = SECOND_EXAMPLE_COUNT_MATRIX[..., j]

    SECOND_SUMMED_PROB_MATRIX[..., j][n > 0] = THESE_MEAN_PROBS[j]

    SECOND_POS_EXAMPLE_COUNT_MATRIX[..., j][n > 0] = (
        1 if j == NUM_BINS_FOR_RELIABILITY - 1 else 0
    )

SECOND_POS_EXAMPLE_COUNT_MATRIX[..., 0] = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=int)

# The following constants are used to test _get_fss_components_one_time.
FIRST_REFERENCE_SSE_MATRIX = PROBABILITY_MATRIX ** 2 + ACTUAL_TARGET_MATRIX ** 2
FIRST_REFERENCE_SSE_MATRIX[MASK_MATRIX == 0] = numpy.nan
FIRST_ACTUAL_SSE_MATRIX = FIRST_REFERENCE_SSE_MATRIX + 0.

# The following constants are used to test _get_bss_components_one_time.
TRAINING_EVENT_FREQUENCY = 0.01
FIRST_CLIMO_SSE_MATRIX = (TRAINING_EVENT_FREQUENCY - ACTUAL_TARGET_MATRIX) ** 2
FIRST_CLIMO_SSE_MATRIX[MASK_MATRIX == 0] = numpy.nan

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

# The following constants are used to test get_basic_scores_gridded.
VALID_TIMES_UNIX_SEC = numpy.array([1000], dtype=int)
LATITUDES_DEG_N = numpy.linspace(50, 59, num=10, dtype=float)
LONGITUDES_DEG_E = numpy.linspace(240, 251, num=12, dtype=float)

PREDICTION_DICT = {
    prediction_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC,
    prediction_io.LATITUDES_KEY: LATITUDES_DEG_N,
    prediction_io.LONGITUDES_KEY: LONGITUDES_DEG_E,
    prediction_io.TARGET_MATRIX_KEY:
        numpy.expand_dims(ACTUAL_TARGET_MATRIX, axis=0),
    prediction_io.PROBABILITY_MATRIX_KEY:
        numpy.expand_dims(PROBABILITY_MATRIX, axis=0)
}

MODEL_FILE_NAME = 'foo.bar'
MATCHING_DISTANCE_PX = 0.
SQUARE_FSS_FILTER = True

PROB_THRESHOLDS = numpy.array([
    0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1 + 1e-6
])
RELIABILITY_BIN_INDICES = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)

FIRST_METADATA_DICT = {
    evaluation.TIME_DIM: VALID_TIMES_UNIX_SEC,
    evaluation.LATITUDE_DIM: LATITUDES_DEG_N,
    evaluation.LONGITUDE_DIM: LONGITUDES_DEG_E,
    evaluation.PROBABILITY_THRESHOLD_DIM: PROB_THRESHOLDS,
    evaluation.RELIABILITY_BIN_DIM: RELIABILITY_BIN_INDICES
}

THESE_DIM = (
    1, len(LATITUDES_DEG_N), len(LONGITUDES_DEG_E), len(PROB_THRESHOLDS)
)
THIS_NUM_AO_TRUE_POS_MATRIX = numpy.full(THESE_DIM, 0, dtype=int)
THIS_NUM_AO_TRUE_POS_MATRIX[0, 3:7, 4:8, 0] = 1
THIS_NUM_PO_TRUE_POS_MATRIX = THIS_NUM_AO_TRUE_POS_MATRIX + 0

THIS_NUM_FALSE_NEG_MATRIX = numpy.full(THESE_DIM, 0, dtype=int)
THIS_NUM_FALSE_NEG_MATRIX[0, 3:7, 4:8, 1:] = 1

# TODO(thunderhoser): Work NaN's into these matrices.
THIS_NUM_FALSE_POS_MATRIX = numpy.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
], dtype=int)

THIS_NUM_FALSE_POS_MATRIX = numpy.expand_dims(
    THIS_NUM_FALSE_POS_MATRIX, axis=(0, -1)
)
THIS_NUM_FALSE_POS_MATRIX = numpy.repeat(
    THIS_NUM_FALSE_POS_MATRIX, axis=-1, repeats=len(PROB_THRESHOLDS)
)

for j in range(1, len(PROB_THRESHOLDS)):
    THIS_NUM_FALSE_POS_MATRIX[0, ..., j][
        PROBABILITY_MATRIX < PROB_THRESHOLDS[j]
    ] = 0

THESE_DIM = (
    evaluation.TIME_DIM, evaluation.LATITUDE_DIM, evaluation.LONGITUDE_DIM,
    evaluation.PROBABILITY_THRESHOLD_DIM
)
FIRST_MAIN_DICT = {
    evaluation.NUM_ACTUAL_ORIENTED_TP_KEY:
        (THESE_DIM, THIS_NUM_AO_TRUE_POS_MATRIX + 0),
    evaluation.NUM_PREDICTION_ORIENTED_TP_KEY:
        (THESE_DIM, THIS_NUM_PO_TRUE_POS_MATRIX + 0),
    evaluation.NUM_FALSE_POSITIVES_KEY:
        (THESE_DIM, THIS_NUM_FALSE_POS_MATRIX + 0),
    evaluation.NUM_FALSE_NEGATIVES_KEY:
        (THESE_DIM, THIS_NUM_FALSE_NEG_MATRIX + 0)
}

THIS_EXAMPLE_COUNT_MATRIX = numpy.invert(
    numpy.isnan(FIRST_CLIMO_SSE_MATRIX)
).astype(int)

THESE_DIM = (
    evaluation.TIME_DIM, evaluation.LATITUDE_DIM, evaluation.LONGITUDE_DIM
)
NEW_DICT = {
    evaluation.ACTUAL_SSE_FOR_FSS_KEY: (
        THESE_DIM, numpy.expand_dims(FIRST_ACTUAL_SSE_MATRIX, axis=0)
    ),
    evaluation.REFERENCE_SSE_FOR_FSS_KEY: (
        THESE_DIM, numpy.expand_dims(FIRST_REFERENCE_SSE_MATRIX, axis=0)
    ),
    evaluation.ACTUAL_SSE_FOR_BRIER_KEY: (
        THESE_DIM, numpy.expand_dims(FIRST_ACTUAL_SSE_MATRIX, axis=0)
    ),
    evaluation.CLIMO_SSE_FOR_BRIER_KEY: (
        THESE_DIM, numpy.expand_dims(FIRST_CLIMO_SSE_MATRIX, axis=0)
    ),
    evaluation.TOTAL_NUM_EXAMPLES_KEY: (
        THESE_DIM, numpy.expand_dims(THIS_EXAMPLE_COUNT_MATRIX, axis=0)
    ),
    evaluation.MEAN_FORECAST_PROBS_KEY: (
        THESE_DIM, numpy.expand_dims(MEAN_PROB_MATRIX, axis=0)
    )
}
FIRST_MAIN_DICT.update(NEW_DICT)

FIRST_TRAINING_FREQ_MATRIX = numpy.full(
    FIRST_CLIMO_SSE_MATRIX.shape, TRAINING_EVENT_FREQUENCY
)
FIRST_TRAINING_FREQ_MATRIX[MASK_MATRIX == 0] = numpy.nan

THESE_DIM = (evaluation.LATITUDE_DIM, evaluation.LONGITUDE_DIM)
NEW_DICT = {
    evaluation.TRAINING_EVENT_FREQ_KEY:
        (THESE_DIM, FIRST_TRAINING_FREQ_MATRIX + 0.)
}
FIRST_MAIN_DICT.update(NEW_DICT)

FIRST_BASIC_TABLE_GRIDDED = xarray.Dataset(
    data_vars=FIRST_MAIN_DICT, coords=FIRST_METADATA_DICT
)

FIRST_BASIC_TABLE_GRIDDED.attrs[evaluation.MODEL_FILE_KEY] = (
    MODEL_FILE_NAME
)
FIRST_BASIC_TABLE_GRIDDED.attrs[evaluation.MATCHING_DISTANCE_KEY] = (
    MATCHING_DISTANCE_PX
)
FIRST_BASIC_TABLE_GRIDDED.attrs[evaluation.SQUARE_FSS_FILTER_KEY] = (
    SQUARE_FSS_FILTER
)

# The following constants are used to test concat_basic_score_tables.
SECOND_BASIC_TABLE_GRIDDED = copy.deepcopy(FIRST_BASIC_TABLE_GRIDDED)
THIS_COORD_DICT = {
    evaluation.TIME_DIM: numpy.array([10000], dtype=int)
}
SECOND_BASIC_TABLE_GRIDDED = (
    SECOND_BASIC_TABLE_GRIDDED.assign_coords(THIS_COORD_DICT)
)

CONCAT_METADATA_DICT = {
    evaluation.TIME_DIM: numpy.array([1000, 10000], dtype=int),
    evaluation.LATITUDE_DIM: LATITUDES_DEG_N,
    evaluation.LONGITUDE_DIM: LONGITUDES_DEG_E,
    evaluation.PROBABILITY_THRESHOLD_DIM: PROB_THRESHOLDS,
    evaluation.RELIABILITY_BIN_DIM: RELIABILITY_BIN_INDICES
}

THESE_DIM = (
    evaluation.TIME_DIM, evaluation.LATITUDE_DIM, evaluation.LONGITUDE_DIM,
    evaluation.PROBABILITY_THRESHOLD_DIM
)
CONCAT_MAIN_DICT = {
    evaluation.NUM_ACTUAL_ORIENTED_TP_KEY: (
        THESE_DIM,
        numpy.repeat(THIS_NUM_AO_TRUE_POS_MATRIX, axis=0, repeats=2)
    ),
    evaluation.NUM_PREDICTION_ORIENTED_TP_KEY: (
        THESE_DIM,
        numpy.repeat(THIS_NUM_PO_TRUE_POS_MATRIX, axis=0, repeats=2)
    ),
    evaluation.NUM_FALSE_POSITIVES_KEY: (
        THESE_DIM, numpy.repeat(THIS_NUM_FALSE_POS_MATRIX, axis=0, repeats=2)
    ),
    evaluation.NUM_FALSE_NEGATIVES_KEY: (
        THESE_DIM, numpy.repeat(THIS_NUM_FALSE_NEG_MATRIX, axis=0, repeats=2)
    )
}

THIS_EXAMPLE_COUNT_MATRIX = numpy.invert(
    numpy.isnan(FIRST_CLIMO_SSE_MATRIX)
).astype(int)

THESE_DIM = (
    evaluation.TIME_DIM, evaluation.LATITUDE_DIM, evaluation.LONGITUDE_DIM
)
NEW_DICT = {
    evaluation.ACTUAL_SSE_FOR_FSS_KEY: (
        THESE_DIM, numpy.stack([FIRST_ACTUAL_SSE_MATRIX] * 2, axis=0)
    ),
    evaluation.REFERENCE_SSE_FOR_FSS_KEY: (
        THESE_DIM, numpy.stack([FIRST_REFERENCE_SSE_MATRIX] * 2, axis=0)
    ),
    evaluation.ACTUAL_SSE_FOR_BRIER_KEY: (
        THESE_DIM, numpy.stack([FIRST_ACTUAL_SSE_MATRIX] * 2, axis=0)
    ),
    evaluation.CLIMO_SSE_FOR_BRIER_KEY: (
        THESE_DIM, numpy.stack([FIRST_CLIMO_SSE_MATRIX] * 2, axis=0)
    ),
    evaluation.TOTAL_NUM_EXAMPLES_KEY: (
        THESE_DIM, numpy.stack([THIS_EXAMPLE_COUNT_MATRIX] * 2, axis=0)
    ),
    evaluation.MEAN_FORECAST_PROBS_KEY: (
        THESE_DIM, numpy.stack([MEAN_PROB_MATRIX] * 2, axis=0)
    )
}
CONCAT_MAIN_DICT.update(NEW_DICT)

THESE_DIM = (evaluation.LATITUDE_DIM, evaluation.LONGITUDE_DIM)
NEW_DICT = {
    evaluation.TRAINING_EVENT_FREQ_KEY:
        (THESE_DIM, FIRST_TRAINING_FREQ_MATRIX + 0.)
}
CONCAT_MAIN_DICT.update(NEW_DICT)

CONCAT_BASIC_TABLE_GRIDDED = xarray.Dataset(
    data_vars=CONCAT_MAIN_DICT, coords=CONCAT_METADATA_DICT,
    attrs=FIRST_BASIC_TABLE_GRIDDED.attrs
)

# The following constants are used to test subset_basic_scores_by_space.
FIRST_ROW_SMALL_GRID = 2
LAST_ROW_SMALL_GRID = 8
FIRST_COLUMN_SMALL_GRID = 3
LAST_COLUMN_SMALL_GRID = 8

SMALL_GRID_LATITUDES_DEG_N = numpy.linspace(52, 58, num=7, dtype=float)
SMALL_GRID_LONGITUDES_DEG_E = numpy.linspace(243, 248, num=6, dtype=float)

THESE_DIM = (
    1, len(SMALL_GRID_LATITUDES_DEG_N), len(SMALL_GRID_LONGITUDES_DEG_E),
    len(PROB_THRESHOLDS)
)
THIS_NUM_AO_TRUE_POS_MATRIX = numpy.full(THESE_DIM, 0, dtype=int)
THIS_NUM_AO_TRUE_POS_MATRIX[0, 1:5, 1:5, 0] = 1
THIS_NUM_PO_TRUE_POS_MATRIX = THIS_NUM_AO_TRUE_POS_MATRIX + 0

THIS_NUM_FALSE_NEG_MATRIX = numpy.full(THESE_DIM, 0, dtype=int)
THIS_NUM_FALSE_NEG_MATRIX[0, 1:5, 1:5, 1:] = 1

THIS_NUM_FALSE_POS_MATRIX = numpy.array([
    [1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1]
], dtype=int)

THIS_PROB_MATRIX = numpy.array([
    [1.00, 0.00, 0.00, 0.00, 0.00, 1.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [1.00, 0.00, 0.00, 0.00, 0.00, 1.00],
    [0.81, 0.00, 0.00, 0.00, 0.00, 0.81]
])

THIS_NUM_FALSE_POS_MATRIX = numpy.expand_dims(
    THIS_NUM_FALSE_POS_MATRIX, axis=(0, -1)
)
THIS_NUM_FALSE_POS_MATRIX = numpy.repeat(
    THIS_NUM_FALSE_POS_MATRIX, axis=-1, repeats=len(PROB_THRESHOLDS)
)
for j in range(1, len(PROB_THRESHOLDS)):
    THIS_NUM_FALSE_POS_MATRIX[0, ..., j][
        THIS_PROB_MATRIX < PROB_THRESHOLDS[j]
    ] = 0

METADATA_DICT_SMALL_GRID = {
    evaluation.TIME_DIM: VALID_TIMES_UNIX_SEC,
    evaluation.LATITUDE_DIM: SMALL_GRID_LATITUDES_DEG_N,
    evaluation.LONGITUDE_DIM: SMALL_GRID_LONGITUDES_DEG_E,
    evaluation.PROBABILITY_THRESHOLD_DIM: PROB_THRESHOLDS,
    evaluation.RELIABILITY_BIN_DIM: RELIABILITY_BIN_INDICES
}

THESE_DIM = (
    evaluation.TIME_DIM, evaluation.LATITUDE_DIM, evaluation.LONGITUDE_DIM,
    evaluation.PROBABILITY_THRESHOLD_DIM
)
MAIN_DICT_SMALL_GRID = {
    evaluation.NUM_ACTUAL_ORIENTED_TP_KEY:
        (THESE_DIM, THIS_NUM_AO_TRUE_POS_MATRIX + 0),
    evaluation.NUM_PREDICTION_ORIENTED_TP_KEY:
        (THESE_DIM, THIS_NUM_PO_TRUE_POS_MATRIX + 0),
    evaluation.NUM_FALSE_POSITIVES_KEY:
        (THESE_DIM, THIS_NUM_FALSE_POS_MATRIX + 0),
    evaluation.NUM_FALSE_NEGATIVES_KEY:
        (THESE_DIM, THIS_NUM_FALSE_NEG_MATRIX + 0)
}

THIS_EXAMPLE_COUNT_MATRIX = numpy.invert(
    numpy.isnan(FIRST_CLIMO_SSE_MATRIX)
).astype(int)

THESE_DIM = (
    evaluation.TIME_DIM, evaluation.LATITUDE_DIM, evaluation.LONGITUDE_DIM
)
NEW_DICT = {
    evaluation.ACTUAL_SSE_FOR_FSS_KEY: (
        THESE_DIM, numpy.expand_dims(FIRST_ACTUAL_SSE_MATRIX[2:9, 3:9], axis=0)
    ),
    evaluation.REFERENCE_SSE_FOR_FSS_KEY: (
        THESE_DIM,
        numpy.expand_dims(FIRST_REFERENCE_SSE_MATRIX[2:9, 3:9], axis=0)
    ),
    evaluation.ACTUAL_SSE_FOR_BRIER_KEY: (
        THESE_DIM,
        numpy.expand_dims(FIRST_ACTUAL_SSE_MATRIX[2:9, 3:9], axis=0)
    ),
    evaluation.CLIMO_SSE_FOR_BRIER_KEY: (
        THESE_DIM,
        numpy.expand_dims(FIRST_CLIMO_SSE_MATRIX[2:9, 3:9], axis=0)
    ),
    evaluation.TOTAL_NUM_EXAMPLES_KEY: (
        THESE_DIM,
        numpy.expand_dims(THIS_EXAMPLE_COUNT_MATRIX[2:9, 3:9], axis=0)
    ),
    evaluation.MEAN_FORECAST_PROBS_KEY: (
        THESE_DIM, numpy.expand_dims(MEAN_PROB_MATRIX[2:9, 3:9], axis=0)
    )
}
MAIN_DICT_SMALL_GRID.update(NEW_DICT)

THESE_DIM = (evaluation.LATITUDE_DIM, evaluation.LONGITUDE_DIM)
NEW_DICT = {
    evaluation.TRAINING_EVENT_FREQ_KEY:
        (THESE_DIM, FIRST_TRAINING_FREQ_MATRIX[2:9, 3:9])
}
MAIN_DICT_SMALL_GRID.update(NEW_DICT)

SMALL_BASIC_TABLE_GRIDDED = xarray.Dataset(
    data_vars=MAIN_DICT_SMALL_GRID, coords=METADATA_DICT_SMALL_GRID,
    attrs=FIRST_BASIC_TABLE_GRIDDED.attrs
)

# The following constants are used to test get_basic_scores_ungridded.
METADATA_DICT_UNGRIDDED = {
    evaluation.TIME_DIM: VALID_TIMES_UNIX_SEC,
    evaluation.PROBABILITY_THRESHOLD_DIM: PROB_THRESHOLDS,
    evaluation.RELIABILITY_BIN_DIM: RELIABILITY_BIN_INDICES
}

THIS_NUM_AO_TRUE_POS_MATRIX = numpy.array(
    [16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int
)
THIS_NUM_FALSE_NEG_MATRIX = numpy.array(
    [0, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16], dtype=int
)
THIS_NUM_FALSE_POS_MATRIX = numpy.array(
    [76, 36, 33, 27, 24, 18, 15, 9, 6, 3, 3, 0], dtype=int
)

THIS_NUM_AO_TRUE_POS_MATRIX = numpy.expand_dims(
    THIS_NUM_AO_TRUE_POS_MATRIX, axis=0
)
THIS_NUM_FALSE_NEG_MATRIX = numpy.expand_dims(
    THIS_NUM_FALSE_NEG_MATRIX, axis=0
)
THIS_NUM_FALSE_POS_MATRIX = numpy.expand_dims(
    THIS_NUM_FALSE_POS_MATRIX, axis=0
)
THIS_NUM_PO_TRUE_POS_MATRIX = THIS_NUM_AO_TRUE_POS_MATRIX + 0

THESE_DIM = (evaluation.TIME_DIM, evaluation.PROBABILITY_THRESHOLD_DIM)
MAIN_DICT_UNGRIDDED = {
    evaluation.NUM_ACTUAL_ORIENTED_TP_KEY:
        (THESE_DIM, THIS_NUM_AO_TRUE_POS_MATRIX + 0),
    evaluation.NUM_PREDICTION_ORIENTED_TP_KEY:
        (THESE_DIM, THIS_NUM_PO_TRUE_POS_MATRIX + 0),
    evaluation.NUM_FALSE_POSITIVES_KEY:
        (THESE_DIM, THIS_NUM_FALSE_POS_MATRIX + 0),
    evaluation.NUM_FALSE_NEGATIVES_KEY:
        (THESE_DIM, THIS_NUM_FALSE_NEG_MATRIX + 0)
}

TOTAL_COUNT_MATRIX_UNGRIDDED = numpy.array(
    [56, 3, 6, 3, 6, 3, 6, 3, 3, 3], dtype=int
)
MEAN_PROB_MATRIX_UNGRIDDED = numpy.array([
    0, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 1
])
SUMMED_PROB_MATRIX_UNGRIDDED = (
    TOTAL_COUNT_MATRIX_UNGRIDDED * MEAN_PROB_MATRIX_UNGRIDDED
)
POS_COUNT_MATRIX_UNGRIDDED = numpy.array(
    [16, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int
)
EVENT_FREQ_MATRIX_UNGRIDDED = numpy.array([
    16. / 56, 0, 0, 0, 0, 0, 0, 0, 0, 0
])

TOTAL_COUNT_MATRIX_UNGRIDDED = numpy.expand_dims(
    TOTAL_COUNT_MATRIX_UNGRIDDED, axis=0
)
MEAN_PROB_MATRIX_UNGRIDDED = numpy.expand_dims(
    MEAN_PROB_MATRIX_UNGRIDDED, axis=0
)
SUMMED_PROB_MATRIX_UNGRIDDED = numpy.expand_dims(
    SUMMED_PROB_MATRIX_UNGRIDDED, axis=0
)
POS_COUNT_MATRIX_UNGRIDDED = numpy.expand_dims(
    POS_COUNT_MATRIX_UNGRIDDED, axis=0
)
EVENT_FREQ_MATRIX_UNGRIDDED = numpy.expand_dims(
    EVENT_FREQ_MATRIX_UNGRIDDED, axis=0
)

REF_SSE_MATRIX_UNGRIDDED = numpy.full(
    1, numpy.nansum(FIRST_REFERENCE_SSE_MATRIX)
)
ACTUAL_SSE_MATRIX_UNGRIDDED = numpy.full(
    1, numpy.nansum(FIRST_ACTUAL_SSE_MATRIX)
)

THESE_DIM = (evaluation.TIME_DIM, evaluation.RELIABILITY_BIN_DIM)
NEW_DICT = {
    evaluation.BINNED_NUM_EXAMPLES_KEY:
        (THESE_DIM, TOTAL_COUNT_MATRIX_UNGRIDDED + 0),
    evaluation.BINNED_SUM_PROBS_KEY:
        (THESE_DIM, SUMMED_PROB_MATRIX_UNGRIDDED + 0.),
    evaluation.BINNED_NUM_POSITIVES_KEY:
        (THESE_DIM, POS_COUNT_MATRIX_UNGRIDDED + 0)
}
MAIN_DICT_UNGRIDDED.update(NEW_DICT)

THESE_DIM = (evaluation.TIME_DIM,)
NEW_DICT = {
    evaluation.ACTUAL_SSE_FOR_FSS_KEY:
        (THESE_DIM, ACTUAL_SSE_MATRIX_UNGRIDDED + 0.),
    evaluation.REFERENCE_SSE_FOR_FSS_KEY:
        (THESE_DIM, REF_SSE_MATRIX_UNGRIDDED + 0.),
}
MAIN_DICT_UNGRIDDED.update(NEW_DICT)

BASIC_SCORE_TABLE_UNGRIDDED = xarray.Dataset(
    data_vars=MAIN_DICT_UNGRIDDED, coords=METADATA_DICT_UNGRIDDED,
    attrs=FIRST_BASIC_TABLE_GRIDDED.attrs
)

# The following constants are used to test get_advanced_scores_ungridded.
METADATA_DICT_ADVANCED_UNGRIDDED = {
    evaluation.PROBABILITY_THRESHOLD_DIM: PROB_THRESHOLDS,
    evaluation.RELIABILITY_BIN_DIM: RELIABILITY_BIN_INDICES,
    evaluation.SINGLETON_DIM: numpy.array([0], dtype=int)
}

THIS_NUM_AO_TRUE_POS_MATRIX = numpy.sum(
    FIRST_MAIN_DICT[evaluation.NUM_ACTUAL_ORIENTED_TP_KEY][1], axis=(0, 1, 2)
)
THIS_NUM_PO_TRUE_POS_MATRIX = numpy.sum(
    FIRST_MAIN_DICT[evaluation.NUM_PREDICTION_ORIENTED_TP_KEY][1],
    axis=(0, 1, 2)
)
THIS_NUM_FALSE_POS_MATRIX = numpy.sum(
    FIRST_MAIN_DICT[evaluation.NUM_FALSE_POSITIVES_KEY][1], axis=(0, 1, 2)
)
THIS_NUM_FALSE_NEG_MATRIX = numpy.sum(
    FIRST_MAIN_DICT[evaluation.NUM_FALSE_NEGATIVES_KEY][1], axis=(0, 1, 2)
)

THESE_CONTINGENCY_TABLES = [
    {
        evaluation.NUM_ACTUAL_ORIENTED_TP_KEY: a_a,
        evaluation.NUM_PREDICTION_ORIENTED_TP_KEY: a_p,
        evaluation.NUM_FALSE_POSITIVES_KEY: b,
        evaluation.NUM_FALSE_NEGATIVES_KEY: c,
    }
    for a_a, a_p, b, c in zip(
        THIS_NUM_AO_TRUE_POS_MATRIX,
        THIS_NUM_PO_TRUE_POS_MATRIX,
        THIS_NUM_FALSE_POS_MATRIX,
        THIS_NUM_FALSE_NEG_MATRIX
    )
]

THIS_POD_MATRIX = numpy.array([
    evaluation._get_pod(t) for t in THESE_CONTINGENCY_TABLES
])
THIS_SUCCESS_RATIO_MATRIX = numpy.array([
    evaluation._get_success_ratio(t) for t in THESE_CONTINGENCY_TABLES
])
THIS_CSI_MATRIX = numpy.array([
    evaluation._get_csi(t) for t in THESE_CONTINGENCY_TABLES
])
THIS_BIAS_MATRIX = numpy.array([
    evaluation._get_frequency_bias(t) for t in THESE_CONTINGENCY_TABLES
])

THESE_DIM = (evaluation.PROBABILITY_THRESHOLD_DIM,)
MAIN_DICT_ADVANCED_UNGRIDDED = {
    evaluation.NUM_ACTUAL_ORIENTED_TP_KEY:
        (THESE_DIM, THIS_NUM_AO_TRUE_POS_MATRIX + 0),
    evaluation.NUM_PREDICTION_ORIENTED_TP_KEY:
        (THESE_DIM, THIS_NUM_PO_TRUE_POS_MATRIX + 0),
    evaluation.NUM_FALSE_POSITIVES_KEY:
        (THESE_DIM, THIS_NUM_FALSE_POS_MATRIX + 0),
    evaluation.NUM_FALSE_NEGATIVES_KEY:
        (THESE_DIM, THIS_NUM_FALSE_NEG_MATRIX + 0),
    evaluation.POD_KEY: (THESE_DIM, THIS_POD_MATRIX + 0.),
    evaluation.SUCCESS_RATIO_KEY: (THESE_DIM, THIS_SUCCESS_RATIO_MATRIX + 0.),
    evaluation.CSI_KEY: (THESE_DIM, THIS_CSI_MATRIX + 0.),
    evaluation.FREQUENCY_BIAS_KEY: (THESE_DIM, THIS_BIAS_MATRIX + 0.)
}

THESE_DIM = (evaluation.RELIABILITY_BIN_DIM,)
NEW_DICT = {
    evaluation.BINNED_NUM_EXAMPLES_KEY:
        (THESE_DIM, TOTAL_COUNT_MATRIX_UNGRIDDED[0, ...]),
    evaluation.BINNED_MEAN_PROBS_KEY:
        (THESE_DIM, MEAN_PROB_MATRIX_UNGRIDDED[0, ...]),
    evaluation.BINNED_EVENT_FREQS_KEY:
        (THESE_DIM, EVENT_FREQ_MATRIX_UNGRIDDED[0, ...])
}
MAIN_DICT_ADVANCED_UNGRIDDED.update(NEW_DICT)

THIS_BSS_DICT = gg_model_eval.get_brier_skill_score(
    mean_forecast_prob_by_bin=numpy.ravel(MEAN_PROB_MATRIX_UNGRIDDED),
    mean_observed_label_by_bin=numpy.ravel(EVENT_FREQ_MATRIX_UNGRIDDED),
    num_examples_by_bin=numpy.ravel(TOTAL_COUNT_MATRIX_UNGRIDDED),
    climatology=TRAINING_EVENT_FREQUENCY
)

THIS_BRIER_SCORE_MATRIX = numpy.full(
    1, THIS_BSS_DICT[gg_model_eval.BRIER_SCORE_KEY]
)
THIS_BSS_MATRIX = numpy.full(1, THIS_BSS_DICT[gg_model_eval.BSS_KEY])
THIS_RELIABILITY_MATRIX = numpy.full(
    1, THIS_BSS_DICT[gg_model_eval.RELIABILITY_KEY]
)
THIS_RESOLUTION_MATRIX = numpy.full(
    1, THIS_BSS_DICT[gg_model_eval.RESOLUTION_KEY]
)

THIS_FSS = numpy.ravel(
    1. - ACTUAL_SSE_MATRIX_UNGRIDDED / REF_SSE_MATRIX_UNGRIDDED
)
THIS_FSS_MATRIX = numpy.full(1, THIS_FSS)
THIS_TRAINING_FREQ_MATRIX = numpy.full(1, TRAINING_EVENT_FREQUENCY)

THESE_DIM = (evaluation.SINGLETON_DIM,)
NEW_DICT = {
    evaluation.BRIER_SCORE_KEY: (THESE_DIM, THIS_BRIER_SCORE_MATRIX + 0.),
    evaluation.BRIER_SKILL_SCORE_KEY: (THESE_DIM, THIS_BSS_MATRIX + 0.),
    evaluation.RELIABILITY_KEY: (THESE_DIM, THIS_RELIABILITY_MATRIX + 0.),
    evaluation.RESOLUTION_KEY: (THESE_DIM, THIS_RESOLUTION_MATRIX + 0.),
    evaluation.FSS_KEY: (THESE_DIM, THIS_FSS_MATRIX + 0.),
    evaluation.TRAINING_EVENT_FREQ_KEY:
        (THESE_DIM, THIS_TRAINING_FREQ_MATRIX + 0.)
}
MAIN_DICT_ADVANCED_UNGRIDDED.update(NEW_DICT)

ADVANCED_SCORE_TABLE_UNGRIDDED = xarray.Dataset(
    data_vars=MAIN_DICT_ADVANCED_UNGRIDDED,
    coords=METADATA_DICT_ADVANCED_UNGRIDDED,
    attrs=FIRST_BASIC_TABLE_GRIDDED.attrs
)

# The following constants are used to test get_advanced_scores_gridded.
METADATA_DICT_ADVANCED_GRIDDED = {
    evaluation.LATITUDE_DIM: LATITUDES_DEG_N,
    evaluation.LONGITUDE_DIM: LONGITUDES_DEG_E,
    evaluation.PROBABILITY_THRESHOLD_DIM: PROB_THRESHOLDS
}

THIS_NUM_AO_TRUE_POS_MATRIX = numpy.sum(
    CONCAT_MAIN_DICT[evaluation.NUM_ACTUAL_ORIENTED_TP_KEY][1], axis=0
)
THIS_NUM_PO_TRUE_POS_MATRIX = numpy.sum(
    CONCAT_MAIN_DICT[evaluation.NUM_PREDICTION_ORIENTED_TP_KEY][1], axis=0
)
THIS_NUM_FALSE_POS_MATRIX = numpy.sum(
    CONCAT_MAIN_DICT[evaluation.NUM_FALSE_POSITIVES_KEY][1], axis=0
)
THIS_NUM_FALSE_NEG_MATRIX = numpy.sum(
    CONCAT_MAIN_DICT[evaluation.NUM_FALSE_NEGATIVES_KEY][1], axis=0
)

THESE_CONTINGENCY_TABLES = [
    {
        evaluation.NUM_ACTUAL_ORIENTED_TP_KEY: a_a,
        evaluation.NUM_PREDICTION_ORIENTED_TP_KEY: a_p,
        evaluation.NUM_FALSE_POSITIVES_KEY: b,
        evaluation.NUM_FALSE_NEGATIVES_KEY: c,
    }
    for a_a, a_p, b, c in zip(
        numpy.ravel(THIS_NUM_AO_TRUE_POS_MATRIX),
        numpy.ravel(THIS_NUM_PO_TRUE_POS_MATRIX),
        numpy.ravel(THIS_NUM_FALSE_POS_MATRIX),
        numpy.ravel(THIS_NUM_FALSE_NEG_MATRIX)
    )
]

THIS_POD_MATRIX = numpy.array([
    evaluation._get_pod(t) for t in THESE_CONTINGENCY_TABLES
])
THIS_SUCCESS_RATIO_MATRIX = numpy.array([
    evaluation._get_success_ratio(t) for t in THESE_CONTINGENCY_TABLES
])
THIS_CSI_MATRIX = numpy.array([
    evaluation._get_csi(t) for t in THESE_CONTINGENCY_TABLES
])
THIS_BIAS_MATRIX = numpy.array([
    evaluation._get_frequency_bias(t) for t in THESE_CONTINGENCY_TABLES
])

THESE_DIM = THIS_NUM_AO_TRUE_POS_MATRIX.shape
THIS_POD_MATRIX = numpy.reshape(THIS_POD_MATRIX, THESE_DIM)
THIS_SUCCESS_RATIO_MATRIX = numpy.reshape(THIS_SUCCESS_RATIO_MATRIX, THESE_DIM)
THIS_CSI_MATRIX = numpy.reshape(THIS_CSI_MATRIX, THESE_DIM)
THIS_BIAS_MATRIX = numpy.reshape(THIS_BIAS_MATRIX, THESE_DIM)

THESE_DIM = (
    evaluation.LATITUDE_DIM, evaluation.LONGITUDE_DIM,
    evaluation.PROBABILITY_THRESHOLD_DIM
)
MAIN_DICT_ADVANCED_GRIDDED = {
    evaluation.NUM_ACTUAL_ORIENTED_TP_KEY:
        (THESE_DIM, THIS_NUM_AO_TRUE_POS_MATRIX + 0),
    evaluation.NUM_PREDICTION_ORIENTED_TP_KEY:
        (THESE_DIM, THIS_NUM_PO_TRUE_POS_MATRIX + 0),
    evaluation.NUM_FALSE_POSITIVES_KEY:
        (THESE_DIM, THIS_NUM_FALSE_POS_MATRIX + 0),
    evaluation.NUM_FALSE_NEGATIVES_KEY:
        (THESE_DIM, THIS_NUM_FALSE_NEG_MATRIX + 0),
    evaluation.POD_KEY: (THESE_DIM, THIS_POD_MATRIX + 0.),
    evaluation.SUCCESS_RATIO_KEY: (THESE_DIM, THIS_SUCCESS_RATIO_MATRIX + 0.),
    evaluation.CSI_KEY: (THESE_DIM, THIS_CSI_MATRIX + 0.),
    evaluation.FREQUENCY_BIAS_KEY: (THESE_DIM, THIS_BIAS_MATRIX + 0.)
}

THIS_ACTUAL_SSE_MATRIX = numpy.nansum(
    CONCAT_MAIN_DICT[evaluation.ACTUAL_SSE_FOR_BRIER_KEY][1], axis=0
)
THIS_CLIMO_SSE_MATRIX = numpy.nansum(
    CONCAT_MAIN_DICT[evaluation.CLIMO_SSE_FOR_BRIER_KEY][1], axis=0
)
THIS_EXAMPLE_COUNT_MATRIX = numpy.invert(
    numpy.isnan(CONCAT_MAIN_DICT[evaluation.ACTUAL_SSE_FOR_BRIER_KEY][1])
).astype(int)

THIS_EXAMPLE_COUNT_MATRIX = numpy.sum(THIS_EXAMPLE_COUNT_MATRIX, axis=0)

THESE_DIM = (evaluation.LATITUDE_DIM, evaluation.LONGITUDE_DIM)
NEW_DICT = {
    evaluation.ACTUAL_SSE_FOR_BRIER_KEY:
        (THESE_DIM, THIS_ACTUAL_SSE_MATRIX + 0.),
    evaluation.CLIMO_SSE_FOR_BRIER_KEY: (THESE_DIM, THIS_CLIMO_SSE_MATRIX + 0.),
    evaluation.TOTAL_NUM_EXAMPLES_KEY:
        (THESE_DIM, THIS_EXAMPLE_COUNT_MATRIX + 0),
    evaluation.MEAN_FORECAST_PROBS_KEY: (THESE_DIM, MEAN_PROB_MATRIX + 0.)
}
MAIN_DICT_ADVANCED_GRIDDED.update(NEW_DICT)

THIS_NUM_ROWS = len(LATITUDES_DEG_N)
THIS_NUM_COLUMNS = len(LONGITUDES_DEG_E)
THIS_BRIER_SCORE_MATRIX = numpy.full(
    (THIS_NUM_ROWS, THIS_NUM_COLUMNS), numpy.nan
)
THIS_BSS_MATRIX = numpy.full(
    (THIS_NUM_ROWS, THIS_NUM_COLUMNS), numpy.nan
)

for r in range(THIS_NUM_ROWS):
    for c in range(THIS_NUM_COLUMNS):
        if THIS_EXAMPLE_COUNT_MATRIX[r, c] == 0:
            continue

        THIS_BRIER_SCORE_MATRIX[r, c] = (
            THIS_ACTUAL_SSE_MATRIX[r, c] /
            THIS_EXAMPLE_COUNT_MATRIX[r, c]
        )
        THIS_BSS_MATRIX[r, c] = 1. - (
            THIS_ACTUAL_SSE_MATRIX[r, c] / THIS_CLIMO_SSE_MATRIX[r, c]
        )

THIS_FSS_MATRIX = 1. - FIRST_ACTUAL_SSE_MATRIX / FIRST_REFERENCE_SSE_MATRIX

THESE_DIM = (evaluation.LATITUDE_DIM, evaluation.LONGITUDE_DIM)
NEW_DICT = {
    evaluation.BRIER_SCORE_KEY: (THESE_DIM, THIS_BRIER_SCORE_MATRIX + 0.),
    evaluation.BRIER_SKILL_SCORE_KEY: (THESE_DIM, THIS_BSS_MATRIX + 0.),
    evaluation.FSS_KEY: (THESE_DIM, THIS_FSS_MATRIX + 0.),
    evaluation.TRAINING_EVENT_FREQ_KEY:
        (THESE_DIM, FIRST_TRAINING_FREQ_MATRIX + 0.)
}
MAIN_DICT_ADVANCED_GRIDDED.update(NEW_DICT)

ADVANCED_SCORE_TABLE_GRIDDED = xarray.Dataset(
    data_vars=MAIN_DICT_ADVANCED_GRIDDED, coords=METADATA_DICT_ADVANCED_GRIDDED,
    attrs=FIRST_BASIC_TABLE_GRIDDED.attrs
)

# The following constants are used to test find_basic_score_file,
# basic_file_name_to_date, and find_advanced_score_file.
TOP_DIRECTORY_NAME = 'foo'
VALID_DATE_STRING = '19670502'
BASIC_FILE_NAME_UNGRIDDED = 'foo/1967/basic_scores_gridded=0_19670502.nc'
BASIC_FILE_NAME_GRIDDED = 'foo/1967/basic_scores_gridded=1_19670502.nc'

ADVANCED_FILE_NAME_ALL_UNGRIDDED = 'foo/advanced_scores_gridded=0.p'
ADVANCED_FILE_NAME_ALL_GRIDDED = 'foo/advanced_scores_gridded=1.p'
ADVANCED_FILE_NAME_HOUR0_UNGRIDDED = 'foo/advanced_scores_hour=00_gridded=0.p'
ADVANCED_FILE_NAME_HOUR0_GRIDDED = 'foo/advanced_scores_hour=00_gridded=1.p'
ADVANCED_FILE_NAME_MONTH1_UNGRIDDED = 'foo/advanced_scores_month=01_gridded=0.p'
ADVANCED_FILE_NAME_MONTH1_GRIDDED = 'foo/advanced_scores_month=01_gridded=1.p'


def _compare_basic_score_tables(first_table, second_table):
    """Compares two xarray tables with basic scores.

    :param first_table: First table.
    :param second_table: Second table.
    :return: are_tables_equal: Boolean flag.
    """

    gridded = evaluation.LATITUDE_DIM in first_table.coords

    float_keys = [
        evaluation.ACTUAL_SSE_FOR_FSS_KEY, evaluation.REFERENCE_SSE_FOR_FSS_KEY
    ]
    integer_keys = [
        evaluation.NUM_ACTUAL_ORIENTED_TP_KEY,
        evaluation.NUM_PREDICTION_ORIENTED_TP_KEY,
        evaluation.NUM_FALSE_POSITIVES_KEY,
        evaluation.NUM_FALSE_NEGATIVES_KEY
    ]

    if gridded:
        float_keys += [
            evaluation.ACTUAL_SSE_FOR_BRIER_KEY,
            evaluation.CLIMO_SSE_FOR_BRIER_KEY,
            evaluation.TRAINING_EVENT_FREQ_KEY,
            evaluation.MEAN_FORECAST_PROBS_KEY
        ]
        integer_keys += [evaluation.TOTAL_NUM_EXAMPLES_KEY]
    else:
        float_keys += [evaluation.BINNED_SUM_PROBS_KEY]
        integer_keys += [
            evaluation.BINNED_NUM_POSITIVES_KEY,
            evaluation.BINNED_NUM_EXAMPLES_KEY
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
    integer_keys = [evaluation.TIME_DIM]

    if gridded:
        float_keys += [evaluation.LATITUDE_DIM, evaluation.LONGITUDE_DIM]
    else:
        integer_keys += [evaluation.RELIABILITY_BIN_DIM]

    for this_key in float_keys:
        if not numpy.allclose(
                first_table.coords[this_key].values,
                second_table.coords[this_key].values,
                atol=TOLERANCE, equal_nan=True
        ):
            return False

    for this_key in integer_keys:
        if not numpy.array_equal(
                first_table.coords[this_key].values,
                second_table.coords[this_key].values
        ):
            return False

    float_keys = [evaluation.MATCHING_DISTANCE_KEY]
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

    gridded = evaluation.LATITUDE_DIM in first_table.coords

    float_keys = [
        evaluation.POD_KEY, evaluation.SUCCESS_RATIO_KEY,
        evaluation.CSI_KEY, evaluation.FREQUENCY_BIAS_KEY,
        evaluation.BRIER_SCORE_KEY, evaluation.BRIER_SKILL_SCORE_KEY,
        evaluation.FSS_KEY, evaluation.TRAINING_EVENT_FREQ_KEY
    ]
    integer_keys = [
        evaluation.NUM_ACTUAL_ORIENTED_TP_KEY,
        evaluation.NUM_PREDICTION_ORIENTED_TP_KEY,
        evaluation.NUM_FALSE_POSITIVES_KEY,
        evaluation.NUM_FALSE_NEGATIVES_KEY
    ]

    if gridded:
        float_keys += [
            evaluation.ACTUAL_SSE_FOR_BRIER_KEY,
            evaluation.CLIMO_SSE_FOR_BRIER_KEY,
            evaluation.MEAN_FORECAST_PROBS_KEY
        ]
    else:
        float_keys += [
            evaluation.BINNED_EVENT_FREQS_KEY, evaluation.BINNED_MEAN_PROBS_KEY,
            evaluation.RELIABILITY_KEY, evaluation.RESOLUTION_KEY
        ]
        integer_keys += [evaluation.BINNED_NUM_EXAMPLES_KEY]

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
    integer_keys = []

    if gridded:
        float_keys += [evaluation.LATITUDE_DIM, evaluation.LONGITUDE_DIM]
    else:
        integer_keys = [evaluation.RELIABILITY_BIN_DIM]

    for this_key in float_keys:
        if not numpy.allclose(
                first_table.coords[this_key].values,
                second_table.coords[this_key].values,
                atol=TOLERANCE, equal_nan=True
        ):
            return False

    for this_key in integer_keys:
        if not numpy.array_equal(
                first_table.coords[this_key].values,
                second_table.coords[this_key].values
        ):
            return False

    float_keys = [evaluation.MATCHING_DISTANCE_KEY]
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
            binary_matrix=TOY_MASK_MATRIX,
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
            binary_matrix=TOY_MASK_MATRIX,
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
            binary_matrix=TOY_MASK_MATRIX,
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
            binary_matrix=TOY_MASK_MATRIX,
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
            binary_matrix=TOY_MASK_MATRIX,
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
            binary_matrix=TOY_MASK_MATRIX,
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
            binary_matrix=TOY_MASK_MATRIX,
            buffer_distance_px=THIRD_DILATION_DISTANCE_PX
        )

        self.assertTrue(numpy.array_equal(
            this_matrix, THIRD_ERODED_MASK_MATRIX
        ))

    def test_match_actual_convection_one_time_first(self):
        """Ensures correct output from _match_actual_convection_one_time.

        In this case, using first matching distance.
        """

        this_fancy_prediction_matrix = (
            evaluation._match_actual_convection_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                predicted_target_matrix=PREDICTED_TARGET_MATRIX,
                matching_distance_px=FIRST_MATCHING_DISTANCE_PX,
                eroded_eval_mask_matrix=MASK_MATRIX
            )
        )

        self.assertTrue(numpy.array_equal(
            this_fancy_prediction_matrix, FIRST_FANCY_PREDICTION_MATRIX
        ))

    def test_match_predicted_convection_one_time_first(self):
        """Ensures correct output from _match_predicted_convection_one_time.

        In this case, using first matching distance.
        """

        this_fancy_target_matrix = (
            evaluation._match_predicted_convection_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                predicted_target_matrix=PREDICTED_TARGET_MATRIX,
                matching_distance_px=FIRST_MATCHING_DISTANCE_PX,
                eroded_eval_mask_matrix=MASK_MATRIX
            )
        )

        self.assertTrue(numpy.array_equal(
            this_fancy_target_matrix, FIRST_FANCY_TARGET_MATRIX
        ))

    def test_match_actual_convection_one_time_second(self):
        """Ensures correct output from _match_actual_convection_one_time.

        In this case, using second matching distance.
        """

        this_mask_matrix = evaluation.erode_binary_matrix(
            binary_matrix=MASK_MATRIX,
            buffer_distance_px=SECOND_MATCHING_DISTANCE_PX
        )

        this_fancy_prediction_matrix = (
            evaluation._match_actual_convection_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                predicted_target_matrix=PREDICTED_TARGET_MATRIX,
                matching_distance_px=SECOND_MATCHING_DISTANCE_PX,
                eroded_eval_mask_matrix=this_mask_matrix
            )
        )

        self.assertTrue(numpy.array_equal(
            this_fancy_prediction_matrix, SECOND_FANCY_PREDICTION_MATRIX
        ))

    def test_match_predicted_convection_one_time_second(self):
        """Ensures correct output from _match_predicted_convection_one_time.

        In this case, using second matching distance.
        """

        this_mask_matrix = evaluation.erode_binary_matrix(
            binary_matrix=MASK_MATRIX,
            buffer_distance_px=SECOND_MATCHING_DISTANCE_PX
        )

        this_fancy_target_matrix = (
            evaluation._match_predicted_convection_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                predicted_target_matrix=PREDICTED_TARGET_MATRIX,
                matching_distance_px=SECOND_MATCHING_DISTANCE_PX,
                eroded_eval_mask_matrix=this_mask_matrix
            )
        )

        self.assertTrue(numpy.array_equal(
            this_fancy_target_matrix, SECOND_FANCY_TARGET_MATRIX
        ))

    def test_get_reliability_components_first(self):
        """Ensures correct output from _get_reliability_components_one_time.

        In this case, using first matching distance.
        """

        this_example_count_matrix, this_mean_prob_matrix, this_freq_matrix = (
            evaluation._get_reliability_components_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                probability_matrix=PROBABILITY_MATRIX,
                matching_distance_px=FIRST_MATCHING_DISTANCE_PX,
                num_bins=NUM_BINS_FOR_RELIABILITY,
                eroded_eval_mask_matrix=MASK_MATRIX
            )
        )

        self.assertTrue(numpy.array_equal(
            this_example_count_matrix, FIRST_EXAMPLE_COUNT_MATRIX
        ))
        self.assertTrue(numpy.allclose(
            this_mean_prob_matrix, FIRST_SUMMED_PROB_MATRIX,
            atol=TOLERANCE, equal_nan=True
        ))
        self.assertTrue(numpy.allclose(
            this_freq_matrix, FIRST_POS_EXAMPLE_COUNT_MATRIX,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_get_reliability_components_second(self):
        """Ensures correct output from _get_reliability_components_one_time.

        In this case, using second matching distance.
        """

        this_mask_matrix = evaluation.erode_binary_matrix(
            binary_matrix=MASK_MATRIX,
            buffer_distance_px=SECOND_MATCHING_DISTANCE_PX
        )

        this_example_count_matrix, this_mean_prob_matrix, this_freq_matrix = (
            evaluation._get_reliability_components_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                probability_matrix=PROBABILITY_MATRIX,
                matching_distance_px=SECOND_MATCHING_DISTANCE_PX,
                num_bins=NUM_BINS_FOR_RELIABILITY,
                eroded_eval_mask_matrix=this_mask_matrix
            )
        )

        self.assertTrue(numpy.array_equal(
            this_example_count_matrix, SECOND_EXAMPLE_COUNT_MATRIX
        ))
        self.assertTrue(numpy.allclose(
            this_mean_prob_matrix, SECOND_SUMMED_PROB_MATRIX,
            atol=TOLERANCE, equal_nan=True
        ))
        self.assertTrue(numpy.allclose(
            this_freq_matrix, SECOND_POS_EXAMPLE_COUNT_MATRIX,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_get_fss_components_first(self):
        """Ensures correct output from _get_fss_components_one_time.

        In this case, using first matching distance.
        """

        this_actual_sse_matrix, this_reference_sse_matrix = (
            evaluation._get_fss_components_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                probability_matrix=PROBABILITY_MATRIX,
                matching_distance_px=FIRST_MATCHING_DISTANCE_PX,
                eroded_eval_mask_matrix=MASK_MATRIX, square_filter=True
            )
        )

        self.assertTrue(numpy.allclose(
            this_actual_sse_matrix, FIRST_ACTUAL_SSE_MATRIX,
            atol=TOLERANCE, equal_nan=True
        ))
        self.assertTrue(numpy.allclose(
            this_reference_sse_matrix, FIRST_REFERENCE_SSE_MATRIX,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_get_bss_components_first(self):
        """Ensures correct output from _get_bss_components_one_time.

        In this case, using first matching distance.
        """

        this_event_freq_matrix = numpy.full(
            PROBABILITY_MATRIX.shape, TRAINING_EVENT_FREQUENCY
        )

        this_actual_sse_matrix, this_climo_sse_matrix = (
            evaluation._get_bss_components_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                probability_matrix=PROBABILITY_MATRIX,
                training_event_freq_matrix=this_event_freq_matrix,
                matching_distance_px=FIRST_MATCHING_DISTANCE_PX,
                eroded_eval_mask_matrix=MASK_MATRIX
            )
        )

        self.assertTrue(numpy.allclose(
            this_actual_sse_matrix, FIRST_ACTUAL_SSE_MATRIX,
            atol=TOLERANCE, equal_nan=True
        ))
        self.assertTrue(numpy.allclose(
            this_climo_sse_matrix, FIRST_CLIMO_SSE_MATRIX,
            atol=TOLERANCE, equal_nan=True
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

    def test_get_basic_scores_gridded(self):
        """Ensures correct output from get_basic_scores_gridded."""

        this_score_table_xarray = evaluation.get_basic_scores_gridded(
            prediction_file_name=None,
            matching_distance_px=MATCHING_DISTANCE_PX,
            probability_thresholds=PROB_THRESHOLDS,
            training_event_freq_matrix=FIRST_TRAINING_FREQ_MATRIX,
            square_fss_filter=SQUARE_FSS_FILTER,
            test_mode=True, prediction_dict=PREDICTION_DICT,
            eval_mask_matrix=MASK_MATRIX, model_file_name=MODEL_FILE_NAME
        )

        self.assertTrue(_compare_basic_score_tables(
            this_score_table_xarray, FIRST_BASIC_TABLE_GRIDDED
        ))

    def test_concat_basic_score_tables(self):
        """Ensures correct output from concat_basic_score_tables."""

        this_score_table_xarray = evaluation.concat_basic_score_tables(
            [FIRST_BASIC_TABLE_GRIDDED, SECOND_BASIC_TABLE_GRIDDED]
        )

        self.assertTrue(_compare_basic_score_tables(
            this_score_table_xarray, CONCAT_BASIC_TABLE_GRIDDED
        ))

    def test_subset_basic_scores_hour0(self):
        """Ensures correct output from subset_basic_scores_by_hour.

        In this case, desired hour is 0.
        """

        this_score_table_xarray = evaluation.subset_basic_scores_by_hour(
            basic_score_table_xarray=CONCAT_BASIC_TABLE_GRIDDED,
            desired_hour=0
        )

        self.assertTrue(_compare_basic_score_tables(
            this_score_table_xarray, FIRST_BASIC_TABLE_GRIDDED
        ))

    def test_subset_basic_scores_hour2(self):
        """Ensures correct output from subset_basic_scores_by_hour.

        In this case, desired hour is 2.
        """

        this_score_table_xarray = evaluation.subset_basic_scores_by_hour(
            basic_score_table_xarray=CONCAT_BASIC_TABLE_GRIDDED,
            desired_hour=2
        )

        self.assertTrue(_compare_basic_score_tables(
            this_score_table_xarray, SECOND_BASIC_TABLE_GRIDDED
        ))

    def test_subset_basic_scores_hour23(self):
        """Ensures correct output from subset_basic_scores_by_hour.

        In this case, desired hour is 23.
        """

        this_score_table_xarray = evaluation.subset_basic_scores_by_hour(
            basic_score_table_xarray=CONCAT_BASIC_TABLE_GRIDDED,
            desired_hour=23
        )

        self.assertTrue(
            this_score_table_xarray.coords[evaluation.TIME_DIM].size == 0
        )

    def test_subset_basic_scores_by_space(self):
        """Ensures correct output from subset_basic_scores_by_space."""

        this_score_table_xarray = evaluation.subset_basic_scores_by_space(
            basic_score_table_xarray=FIRST_BASIC_TABLE_GRIDDED,
            first_grid_row=FIRST_ROW_SMALL_GRID,
            last_grid_row=LAST_ROW_SMALL_GRID,
            first_grid_column=FIRST_COLUMN_SMALL_GRID,
            last_grid_column=LAST_COLUMN_SMALL_GRID
        )

        self.assertTrue(_compare_basic_score_tables(
            this_score_table_xarray, SMALL_BASIC_TABLE_GRIDDED
        ))

    def test_get_basic_scores_ungridded(self):
        """Ensures correct output from get_basic_scores_ungridded."""

        this_score_table_xarray = evaluation.get_basic_scores_ungridded(
            prediction_file_name=None,
            matching_distance_px=MATCHING_DISTANCE_PX,
            num_prob_thresholds=len(PROB_THRESHOLDS) - 1,
            square_fss_filter=SQUARE_FSS_FILTER,
            num_bins_for_reliability=NUM_BINS_FOR_RELIABILITY,
            test_mode=True, prediction_dict=PREDICTION_DICT,
            eval_mask_matrix=MASK_MATRIX, model_file_name=MODEL_FILE_NAME
        )

        self.assertTrue(_compare_basic_score_tables(
            this_score_table_xarray, BASIC_SCORE_TABLE_UNGRIDDED
        ))

    def test_get_advanced_scores_ungridded(self):
        """Ensures correct output from get_advanced_scores_ungridded."""

        this_score_table_xarray = evaluation.get_advanced_scores_ungridded(
            basic_score_table_xarray=BASIC_SCORE_TABLE_UNGRIDDED,
            training_event_frequency=TRAINING_EVENT_FREQUENCY
        )

        self.assertTrue(_compare_advanced_score_tables(
            this_score_table_xarray, ADVANCED_SCORE_TABLE_UNGRIDDED
        ))

    def test_get_advanced_scores_gridded(self):
        """Ensures correct output from get_advanced_scores_gridded."""

        this_score_table_xarray = evaluation.get_advanced_scores_gridded(
            basic_score_table_xarray=CONCAT_BASIC_TABLE_GRIDDED
        )

        self.assertTrue(_compare_advanced_score_tables(
            this_score_table_xarray, ADVANCED_SCORE_TABLE_GRIDDED
        ))

    def test_find_basic_score_file_ungridded(self):
        """Ensures correct output from find_basic_score_file.

        In this case, file contains ungridded scores.
        """

        this_file_name = evaluation.find_basic_score_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING,
            gridded=False, raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == BASIC_FILE_NAME_UNGRIDDED)

    def test_find_basic_score_file_gridded(self):
        """Ensures correct output from find_basic_score_file.

        In this case, file contains gridded scores.
        """

        this_file_name = evaluation.find_basic_score_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING,
            gridded=True, raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == BASIC_FILE_NAME_GRIDDED)

    def test_basic_file_name_to_date_ungridded(self):
        """Ensures correct output from basic_file_name_to_date.

        In this case, file contains ungridded scores.
        """

        self.assertTrue(
            evaluation.basic_file_name_to_date(BASIC_FILE_NAME_UNGRIDDED) ==
            VALID_DATE_STRING
        )

    def test_basic_file_name_to_date_gridded(self):
        """Ensures correct output from basic_file_name_to_date.

        In this case, file contains gridded scores.
        """

        self.assertTrue(
            evaluation.basic_file_name_to_date(BASIC_FILE_NAME_GRIDDED) ==
            VALID_DATE_STRING
        )

    def test_find_advanced_score_file_all_ungridded(self):
        """Ensures correct output from find_advanced_score_file.

        In this case, file contains ungridded scores for all hours/months.
        """

        this_file_name = evaluation.find_advanced_score_file(
            directory_name=TOP_DIRECTORY_NAME, gridded=False,
            month=None, hour=None, raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == ADVANCED_FILE_NAME_ALL_UNGRIDDED)

    def test_find_advanced_score_file_all_gridded(self):
        """Ensures correct output from find_advanced_score_file.

        In this case, file contains gridded scores for all hours/months.
        """

        this_file_name = evaluation.find_advanced_score_file(
            directory_name=TOP_DIRECTORY_NAME, gridded=True,
            month=None, hour=None, raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == ADVANCED_FILE_NAME_ALL_GRIDDED)

    def test_find_advanced_score_file_hour0_ungridded(self):
        """Ensures correct output from find_advanced_score_file.

        In this case, file contains ungridded scores for hour 0.
        """

        this_file_name = evaluation.find_advanced_score_file(
            directory_name=TOP_DIRECTORY_NAME, gridded=False,
            month=None, hour=0, raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == ADVANCED_FILE_NAME_HOUR0_UNGRIDDED)

    def test_find_advanced_score_file_hour0_gridded(self):
        """Ensures correct output from find_advanced_score_file.

        In this case, file contains gridded scores for hour 0.
        """

        this_file_name = evaluation.find_advanced_score_file(
            directory_name=TOP_DIRECTORY_NAME, gridded=True,
            month=None, hour=0, raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == ADVANCED_FILE_NAME_HOUR0_GRIDDED)

    def test_find_advanced_score_file_month1_ungridded(self):
        """Ensures correct output from find_advanced_score_file.

        In this case, file contains ungridded scores for month 1.
        """

        this_file_name = evaluation.find_advanced_score_file(
            directory_name=TOP_DIRECTORY_NAME, gridded=False,
            month=1, hour=None, raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == ADVANCED_FILE_NAME_MONTH1_UNGRIDDED)

    def test_find_advanced_score_file_month1_gridded(self):
        """Ensures correct output from find_advanced_score_file.

        In this case, file contains gridded scores for month 1.
        """

        this_file_name = evaluation.find_advanced_score_file(
            directory_name=TOP_DIRECTORY_NAME, gridded=True,
            month=1, hour=None, raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == ADVANCED_FILE_NAME_MONTH1_GRIDDED)


if __name__ == '__main__':
    unittest.main()
