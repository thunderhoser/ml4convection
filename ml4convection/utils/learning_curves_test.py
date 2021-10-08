"""Unit tests for learning_curves.py."""

import unittest
import numpy
import xarray
from ml4convection.utils import learning_curves

TOLERANCE = 1e-6

# The following constants are used to test basic scores.
NEIGH_DISTANCE_PX = 2.

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

MEAN_FILTERED_TARGET_MATRIX = 0.04 * numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0],
    [0, 0, 2, 4, 6, 8, 8, 6, 4, 2, 0, 0],
    [0, 0, 3, 6, 9, 12, 12, 9, 6, 3, 0, 0],
    [0, 0, 4, 8, 12, 16, 16, 12, 8, 4, 0, 0],
    [0, 0, 4, 8, 12, 16, 16, 12, 8, 4, 0, 0],
    [0, 0, 3, 6, 9, 12, 12, 9, 6, 3, 0, 0],
    [0, 0, 2, 4, 6, 8, 8, 6, 4, 2, 0, 0],
    [0, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)

# MAX_FILTERED_TARGET_MATRIX = numpy.array([
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ], dtype=int)

PROBABILITY_MATRIX = numpy.array([
    [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1],
    [0.5, 0.6, 0.7, 0.8, 0.0, 0.0, 0.0, 0.0, 0.8, 0.7, 0.6, 0.5],
    [0.2, 0.4, 0.6, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.6, 0.4, 0.2],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.2, 0.4, 0.6, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.6, 0.4, 0.2],
    [0.5, 0.6, 0.7, 0.8, 0.0, 0.0, 0.0, 0.0, 0.8, 0.7, 0.6, 0.5],
    [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1]
])

MEAN_FILTERED_PROB_MATRIX = 0.04 * numpy.array([
    [3.6, 5.8, 5.8, 5.0, 3.8, 2.2, 2.2, 3.8, -1, -1, -1, -1],
    [3.6, 5.8, 5.8, 5.0, 3.8, 2.2, 2.2, 3.8, -1, -1, -1, -1],
    [3.6, 5.8, 5.8, 5.0, 3.8, 2.2, 2.2, 3.8, -1, -1, -1, -1],
    [-1, -1, -1, -1, 3.1, 1.8, 1.8, 3.1, 4.1, 4.8, 4.8, 3.0],
    [-1, -1, -1, -1, 1.6, 1.0, 1.0, 1.6, 2.0, 2.2, 2.2, 1.2],
    [-1, -1, -1, -1, 1.6, 1.0, 1.0, 1.6, 2.0, 2.2, 2.2, 1.2],
    [-1, -1, -1, -1, 3.1, 1.8, 1.8, 3.1, 4.1, 4.8, 4.8, 3.0],
    [3.6, 5.8, 5.8, 5.0, 3.8, 2.2, 2.2, 3.8, 5.0, 5.8, 5.8, 3.6],
    [3.6, 5.8, 5.8, 5.0, 3.8, 2.2, 2.2, 3.8, 5.0, 5.8, 5.8, 3.6],
    [3.6, 5.8, 5.8, 5.0, 3.8, 2.2, 2.2, 3.8, 5.0, 5.8, 5.8, 3.6]
])

MEAN_FILTERED_PROB_MATRIX[MEAN_FILTERED_PROB_MATRIX < 0] = numpy.nan

# MAX_FILTERED_PROB_MATRIX = numpy.array([
#     [0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7],
#     [0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7],
#     [0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7],
#     [0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7],
#     [0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6],
#     [0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6],
#     [0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7],
#     [0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7],
#     [0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7],
#     [0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7]
# ])

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

NEIGH_BRIER_SSE = 26.8
NEIGH_BRIER_NUM_VALUES = 92.
FOURIER_BRIER_SSE = 44.2
FOURIER_BRIER_NUM_VALUES = 92.

NEIGH_FSS_ACTUAL_SSE = numpy.nansum(
    (MEAN_FILTERED_TARGET_MATRIX - MEAN_FILTERED_PROB_MATRIX) ** 2
)
NEIGH_FSS_REFERENCE_SSE = numpy.nansum(
    MEAN_FILTERED_TARGET_MATRIX ** 2 + MEAN_FILTERED_PROB_MATRIX ** 2
)
FOURIER_FSS_ACTUAL_SSE = 26.8
FOURIER_FSS_REFERENCE_SSE = 26.8

NEIGH_IOU_INTERSECTION = 9.3
NEIGH_IOU_UNION = 60.1
FOURIER_IOU_INTERSECTION = 0.
FOURIER_IOU_UNION = 33.4

NEIGH_DICE_INTERSECTION = 41.2
NEIGH_DICE_NUM_PIXELS = 92.
FOURIER_DICE_INTERSECTION = 58.6
FOURIER_DICE_NUM_PIXELS = 92.

FOURIER_NUM_TRUE_POSITIVES = 0.
FOURIER_NUM_FALSE_POSITIVES = 17.4
FOURIER_NUM_FALSE_NEGATIVES = 16.
FOURIER_NUM_TRUE_NEGATIVES = 58.6
NEIGH_NUM_OBS_ORIENTED_TP = 16.
NEIGH_NUM_FALSE_NEGATIVES = 0.
NEIGH_NUM_PRED_ORIENTED_TP = 9.3
NEIGH_NUM_FALSE_POSITIVES = 8.1

# The following constants are used to test advanced scores.
# METADATA_DICT = {
#     learning_curves.NEIGH_DISTANCE_DIM: numpy.array([0, 1, numpy.sqrt(2), 2]),
#     learning_curves.MIN_RESOLUTION_DIM: numpy.array([0, 0.1, 0.2, 0.4, 0.8]),
#     learning_curves.MAX_RESOLUTION_DIM:
#         numpy.array([0.1, 0.2, 0.4, 0.8, numpy.inf])
# }

METADATA_DICT = {
    learning_curves.NEIGH_DISTANCE_DIM: numpy.array([0, 1, numpy.sqrt(2), 2, 4])
}

NUM_NEIGH_DISTANCES = len(METADATA_DICT[learning_curves.NEIGH_DISTANCE_DIM])
# NUM_FOURIER_BANDS = len(METADATA_DICT[learning_curves.MIN_RESOLUTION_DIM])
NUM_TIMES = 2

THESE_DIM = (learning_curves.TIME_DIM, learning_curves.NEIGH_DISTANCE_DIM)
THIS_ARRAY = numpy.full((NUM_TIMES, NUM_NEIGH_DISTANCES), numpy.nan)
MAIN_DATA_DICT = {
    learning_curves.NEIGH_BRIER_SSE_KEY: (THESE_DIM, THIS_ARRAY + 0),
    learning_curves.NEIGH_BRIER_NUM_VALS_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    learning_curves.NEIGH_FSS_ACTUAL_SSE_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    learning_curves.NEIGH_FSS_REFERENCE_SSE_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    learning_curves.NEIGH_IOU_POS_ISCTN_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    learning_curves.NEIGH_IOU_POS_UNION_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    learning_curves.NEIGH_IOU_NEG_ISCTN_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    learning_curves.NEIGH_IOU_NEG_UNION_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    learning_curves.NEIGH_DICE_INTERSECTION_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    learning_curves.NEIGH_DICE_NUM_PIX_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    learning_curves.NEIGH_PRED_ORIENTED_TP_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    learning_curves.NEIGH_OBS_ORIENTED_TP_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    learning_curves.NEIGH_FALSE_POSITIVES_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    learning_curves.NEIGH_FALSE_NEGATIVES_KEY: (THESE_DIM, THIS_ARRAY + 0.)
}

# THESE_DIM = (learning_curves.TIME_DIM, learning_curves.MIN_RESOLUTION_DIM)
# THIS_ARRAY = numpy.full((NUM_TIMES, NUM_FOURIER_BANDS), numpy.nan)
# NEW_DICT = {
#     learning_curves.FREQ_SSE_REAL_KEY: (THESE_DIM, THIS_ARRAY + 0),
#     learning_curves.FREQ_SSE_IMAGINARY_KEY: (THESE_DIM, THIS_ARRAY + 0),
#     learning_curves.FREQ_SSE_TOTAL_KEY: (THESE_DIM, THIS_ARRAY + 0),
#     learning_curves.FREQ_SSE_NUM_WEIGHTS_KEY: (THESE_DIM, THIS_ARRAY + 0),
#     learning_curves.FOURIER_BRIER_SSE_KEY: (THESE_DIM, THIS_ARRAY + 0),
#     learning_curves.FOURIER_BRIER_NUM_VALS_KEY: (THESE_DIM, THIS_ARRAY + 0.),
#     learning_curves.FOURIER_FSS_ACTUAL_SSE_KEY: (THESE_DIM, THIS_ARRAY + 0.),
#     learning_curves.FOURIER_FSS_REFERENCE_SSE_KEY: (THESE_DIM, THIS_ARRAY + 0.),
#     learning_curves.FOURIER_IOU_INTERSECTION_KEY: (THESE_DIM, THIS_ARRAY + 0.),
#     learning_curves.FOURIER_IOU_UNION_KEY: (THESE_DIM, THIS_ARRAY + 0.),
#     learning_curves.FOURIER_DICE_INTERSECTION_KEY: (THESE_DIM, THIS_ARRAY + 0.),
#     learning_curves.FOURIER_DICE_NUM_PIX_KEY: (THESE_DIM, THIS_ARRAY + 0.),
#     learning_curves.FOURIER_NUM_TRUE_POSITIVES_KEY: (THESE_DIM, THIS_ARRAY + 0.),
#     learning_curves.FOURIER_CSI_DENOMINATOR_KEY: (THESE_DIM, THIS_ARRAY + 0.)
# }
# MAIN_DATA_DICT.update(NEW_DICT)

BASIC_SCORE_TABLE_XARRAY = xarray.Dataset(
    data_vars=MAIN_DATA_DICT, coords=METADATA_DICT
)
B = BASIC_SCORE_TABLE_XARRAY

B[learning_curves.NEIGH_BRIER_SSE_KEY].values = numpy.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
], dtype=float)

B[learning_curves.NEIGH_BRIER_NUM_VALS_KEY].values = numpy.array([
    [100, 100, 100, 100, 100],
    [200, 200, 200, 200, 200]
], dtype=float)

B[learning_curves.NEIGH_FSS_ACTUAL_SSE_KEY].values = numpy.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
], dtype=float)

B[learning_curves.NEIGH_FSS_REFERENCE_SSE_KEY].values = numpy.array([
    [100, 100, 100, 100, 100],
    [200, 200, 200, 200, 200]
], dtype=float)

B[learning_curves.NEIGH_IOU_POS_ISCTN_KEY].values = numpy.array([
    [10, 20, 30, 40, 50],
    [6, 7, 8, 9, 10]
], dtype=float)

B[learning_curves.NEIGH_IOU_POS_UNION_KEY].values = numpy.array([
    [10, 20, 30, 40, 50],
    [60, 70, 80, 90, 100]
], dtype=float)

B[learning_curves.NEIGH_IOU_NEG_ISCTN_KEY].values = numpy.array([
    [10, 9, 8, 7, 6],
    [50, 40, 30, 20, 10]
], dtype=float)

B[learning_curves.NEIGH_IOU_NEG_UNION_KEY].values = numpy.array([
    [100, 90, 80, 70, 60],
    [50, 40, 30, 20, 10]
], dtype=float)

B[learning_curves.NEIGH_DICE_INTERSECTION_KEY].values = numpy.array([
    [5, 10, 15, 20, 25],
    [25, 20, 15, 10, 5]
], dtype=float)

B[learning_curves.NEIGH_DICE_NUM_PIX_KEY].values = numpy.array([
    [50, 100, 50, 100, 50],
    [100, 50, 100, 50, 100]
], dtype=float)

B[learning_curves.NEIGH_PRED_ORIENTED_TP_KEY].values = numpy.array([
    [0, 1, 2, 3, 4],
    [0, 2, 4, 6, 8]
], dtype=float)

B[learning_curves.NEIGH_OBS_ORIENTED_TP_KEY].values = numpy.array([
    [4, 4, 4, 4, 4],
    [8, 8, 8, 8, 8]
], dtype=float)

B[learning_curves.NEIGH_FALSE_POSITIVES_KEY].values = numpy.array([
    [20, 30, 40, 30, 20],
    [10, 10, 10, 10, 10]
], dtype=float)

B[learning_curves.NEIGH_FALSE_NEGATIVES_KEY].values = numpy.array([
    [15, 10, 15, 10, 20],
    [25, 20, 25, 10, 10]
], dtype=float)

BASIC_SCORE_TABLE_XARRAY = B

THESE_DIM = (learning_curves.NEIGH_DISTANCE_DIM,)
THIS_ARRAY = numpy.full(NUM_NEIGH_DISTANCES, numpy.nan)
MAIN_DATA_DICT = {
    learning_curves.NEIGH_BRIER_SCORE_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    learning_curves.NEIGH_FSS_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    learning_curves.NEIGH_IOU_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    learning_curves.NEIGH_ALL_CLASS_IOU_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    learning_curves.NEIGH_DICE_COEFF_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    learning_curves.NEIGH_CSI_KEY: (THESE_DIM, THIS_ARRAY + 0.)
}

# THESE_DIM = (learning_curves.MIN_RESOLUTION_DIM,)
# THIS_ARRAY = numpy.full(NUM_FOURIER_BANDS, numpy.nan)
# NEW_DICT = {
#     learning_curves.FREQ_MSE_REAL_KEY: (THESE_DIM, THIS_ARRAY + 0.),
#     learning_curves.FREQ_MSE_IMAGINARY_KEY: (THESE_DIM, THIS_ARRAY + 0.),
#     learning_curves.FREQ_MSE_TOTAL_KEY: (THESE_DIM, THIS_ARRAY + 0.),
#     learning_curves.FOURIER_BRIER_SCORE_KEY: (THESE_DIM, THIS_ARRAY + 0.),
#     learning_curves.FOURIER_FSS_KEY: (THESE_DIM, THIS_ARRAY + 0.),
#     learning_curves.FOURIER_IOU_KEY: (THESE_DIM, THIS_ARRAY + 0.),
#     learning_curves.FOURIER_DICE_COEFF_KEY: (THESE_DIM, THIS_ARRAY + 0.),
#     learning_curves.FOURIER_CSI_KEY: (THESE_DIM, THIS_ARRAY + 0.)
# }
# MAIN_DATA_DICT.update(NEW_DICT)

ADVANCED_SCORE_TABLE_XARRAY = xarray.Dataset(
    data_vars=MAIN_DATA_DICT, coords=METADATA_DICT
)
A = ADVANCED_SCORE_TABLE_XARRAY

A[learning_curves.NEIGH_BRIER_SCORE_KEY].values = (1. / 300) * numpy.array(
    [7, 9, 11, 13, 15], dtype=float
)
A[learning_curves.NEIGH_FSS_KEY].values = 1. - (1. / 300) * numpy.array(
    [7, 9, 11, 13, 15], dtype=float
)
A[learning_curves.NEIGH_IOU_KEY].values = numpy.array([
    16. / 70, 27. / 90, 38. / 110, 49. / 130, 60. / 150
])
A[learning_curves.NEIGH_DICE_COEFF_KEY].values = numpy.array([
    0.2, 0.2, 0.2, 0.2, 0.2
])

NEGATIVE_IOU_VALUES = numpy.array([
    60. / 150, 49. / 130, 38. / 110, 27. / 90, 16. / 70
])
A[learning_curves.NEIGH_ALL_CLASS_IOU_KEY].values = 0.5 * (
    A[learning_curves.NEIGH_IOU_KEY].values + NEGATIVE_IOU_VALUES
)

POD_VALUES = numpy.array([35, 30, 40, 20, 30], dtype=float)
POD_VALUES = 12. / (12 + POD_VALUES)
SUCCESS_RATIOS = numpy.array(
    [0, 3. / 43, 6. / 56, 9. / 49, 12. / 42], dtype=float
)
A[learning_curves.NEIGH_CSI_KEY].values = (
    (POD_VALUES ** -1 + SUCCESS_RATIOS ** -1 - 1) ** -1
)

ADVANCED_SCORE_TABLE_XARRAY = A


def _compare_advanced_score_tables(first_table, second_table):
    """Compares two xarray tables with advanced scores.

    :param first_table: First table.
    :param second_table: Second table.
    :return: are_tables_equal: Boolean flag.
    """

    keys_to_compare = [
        learning_curves.NEIGH_BRIER_SCORE_KEY, learning_curves.NEIGH_FSS_KEY,
        learning_curves.NEIGH_IOU_KEY, learning_curves.NEIGH_ALL_CLASS_IOU_KEY,
        learning_curves.NEIGH_DICE_COEFF_KEY, learning_curves.NEIGH_CSI_KEY
    ]

    for this_key in keys_to_compare:
        if not numpy.allclose(
                first_table[this_key].values, second_table[this_key].values,
                atol=TOLERANCE, equal_nan=True
        ):
            return False

    return True


class LearningCurvesTests(unittest.TestCase):
    """Each method is a unit test for learning_curves.py."""

    def test_get_brier_components_one_time_neigh(self):
        """Ensures correct output from _get_brier_components_one_time.

        In this case, doing neighbourhood-based evaluation.
        """

        this_sse, this_num_values = (
            learning_curves._get_brier_components_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                probability_matrix=PROBABILITY_MATRIX,
                eval_mask_matrix=MASK_MATRIX,
                matching_distance_px=NEIGH_DISTANCE_PX
            )
        )

        self.assertTrue(numpy.isclose(
            this_sse, FOURIER_BRIER_SSE, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_num_values, FOURIER_BRIER_NUM_VALUES, atol=TOLERANCE
        ))

    def test_get_brier_components_one_time_fourier(self):
        """Ensures correct output from _get_brier_components_one_time.

        In this case, doing Fourier-based evaluation.
        """

        this_sse, this_num_values = (
            learning_curves._get_brier_components_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                probability_matrix=PROBABILITY_MATRIX,
                eval_mask_matrix=MASK_MATRIX, matching_distance_px=None
            )
        )

        self.assertTrue(numpy.isclose(
            this_sse, NEIGH_BRIER_SSE, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_num_values, NEIGH_BRIER_NUM_VALUES, atol=TOLERANCE
        ))

    def test_get_fss_components_one_time_neigh(self):
        """Ensures correct output from _get_fss_components_one_time.

        In this case, doing neighbourhood-based evaluation.
        """

        this_actual_sse, this_reference_sse = (
            learning_curves._get_fss_components_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                probability_matrix=PROBABILITY_MATRIX,
                eval_mask_matrix=MASK_MATRIX,
                matching_distance_px=NEIGH_DISTANCE_PX
            )
        )

        self.assertTrue(numpy.isclose(
            this_actual_sse, NEIGH_FSS_ACTUAL_SSE, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_reference_sse, NEIGH_FSS_REFERENCE_SSE, atol=TOLERANCE
        ))

    def test_get_fss_components_one_time_fourier(self):
        """Ensures correct output from _get_fss_components_one_time.

        In this case, doing Fourier-based evaluation.
        """

        this_actual_sse, this_reference_sse = (
            learning_curves._get_fss_components_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                probability_matrix=PROBABILITY_MATRIX,
                eval_mask_matrix=MASK_MATRIX, matching_distance_px=None
            )
        )

        self.assertTrue(numpy.isclose(
            this_actual_sse, FOURIER_FSS_ACTUAL_SSE, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_reference_sse, FOURIER_FSS_REFERENCE_SSE, atol=TOLERANCE
        ))

    def test_get_iou_components_one_time_neigh(self):
        """Ensures correct output from _get_iou_components_one_time.

        In this case, doing neighbourhood-based evaluation.
        """

        this_intersection, this_union = (
            learning_curves._get_iou_components_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                probability_matrix=PROBABILITY_MATRIX,
                eval_mask_matrix=MASK_MATRIX,
                matching_distance_px=NEIGH_DISTANCE_PX
            )[:2]
        )

        self.assertTrue(numpy.isclose(
            this_intersection, NEIGH_IOU_INTERSECTION, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_union, NEIGH_IOU_UNION, atol=TOLERANCE
        ))

    def test_get_iou_components_one_time_fourier(self):
        """Ensures correct output from _get_iou_components_one_time.

        In this case, doing Fourier-based evaluation.
        """

        this_intersection, this_union = (
            learning_curves._get_iou_components_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                probability_matrix=PROBABILITY_MATRIX,
                eval_mask_matrix=MASK_MATRIX,
                matching_distance_px=None
            )[:2]
        )

        self.assertTrue(numpy.isclose(
            this_intersection, FOURIER_IOU_INTERSECTION, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_union, FOURIER_IOU_UNION, atol=TOLERANCE
        ))

    def test_get_dice_components_one_time_neigh(self):
        """Ensures correct output from _get_dice_components_one_time.

        In this case, doing neighbourhood-based evaluation.
        """

        this_intersection, this_num_pixels = (
            learning_curves._get_dice_components_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                probability_matrix=PROBABILITY_MATRIX,
                eval_mask_matrix=MASK_MATRIX,
                matching_distance_px=NEIGH_DISTANCE_PX
            )
        )

        self.assertTrue(numpy.isclose(
            this_intersection, NEIGH_DICE_INTERSECTION, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_num_pixels, NEIGH_DICE_NUM_PIXELS, atol=TOLERANCE
        ))

    def test_get_dice_components_one_time_fourier(self):
        """Ensures correct output from _get_dice_components_one_time.

        In this case, doing Fourier-based evaluation.
        """

        this_intersection, this_num_pixels = (
            learning_curves._get_dice_components_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                probability_matrix=PROBABILITY_MATRIX,
                eval_mask_matrix=MASK_MATRIX,
                matching_distance_px=None
            )
        )

        self.assertTrue(numpy.isclose(
            this_intersection, FOURIER_DICE_INTERSECTION, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_num_pixels, FOURIER_DICE_NUM_PIXELS, atol=TOLERANCE
        ))

    def test_get_band_pass_contingency_one_time(self):
        """Ensures correctness of _get_band_pass_contingency_one_time."""

        (
            this_num_true_positives, this_num_false_positives,
            this_num_false_negatives, this_num_true_negatives
        ) = learning_curves._get_band_pass_contingency_one_time(
            actual_target_matrix=ACTUAL_TARGET_MATRIX,
            probability_matrix=PROBABILITY_MATRIX,
            eval_mask_matrix=MASK_MATRIX
        )

        self.assertTrue(numpy.isclose(
            this_num_true_positives, FOURIER_NUM_TRUE_POSITIVES, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_num_false_positives, FOURIER_NUM_FALSE_POSITIVES,
            atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_num_false_negatives, FOURIER_NUM_FALSE_NEGATIVES,
            atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_num_true_negatives, FOURIER_NUM_TRUE_NEGATIVES, atol=TOLERANCE
        ))

    def test_get_neigh_csi_components_one_time(self):
        """Ensures correct output from _get_neigh_csi_components_one_time."""

        (
            this_num_pred_oriented_tp, this_num_obs_oriented_tp,
            this_num_false_positives, this_num_false_negatives
        ) = learning_curves._get_neigh_csi_components_one_time(
            actual_target_matrix=ACTUAL_TARGET_MATRIX,
            probability_matrix=PROBABILITY_MATRIX,
            eval_mask_matrix=MASK_MATRIX,
            matching_distance_px=NEIGH_DISTANCE_PX
        )

        self.assertTrue(numpy.isclose(
            this_num_pred_oriented_tp, NEIGH_NUM_PRED_ORIENTED_TP,
            atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_num_obs_oriented_tp, NEIGH_NUM_OBS_ORIENTED_TP, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_num_false_positives, NEIGH_NUM_FALSE_POSITIVES, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_num_false_negatives, NEIGH_NUM_FALSE_NEGATIVES, atol=TOLERANCE
        ))

    def test_get_advanced_scores(self):
        """Ensures correct output from get_advanced_scores."""

        this_score_table_xarray = learning_curves.get_advanced_scores(
            BASIC_SCORE_TABLE_XARRAY
        )

        self.assertTrue(_compare_advanced_score_tables(
            this_score_table_xarray, ADVANCED_SCORE_TABLE_XARRAY
        ))


if __name__ == '__main__':
    unittest.main()
