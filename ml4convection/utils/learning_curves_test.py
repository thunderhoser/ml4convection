"""Unit tests for learning_curves.py."""

import unittest
import numpy
from ml4convection.utils import learning_curves

TOLERANCE = 1e-6
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

FOURIER_CSI_NUMERATOR = 0.
FOURIER_CSI_DENOMINATOR = 33.4
NEIGH_NUM_OBS_ORIENTED_TP = 16.
NEIGH_NUM_FALSE_NEGATIVES = 0.
NEIGH_NUM_PRED_ORIENTED_TP = 9.3
NEIGH_NUM_FALSE_POSITIVES = 8.1


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
            )
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
            )
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

    def test_get_fourier_csi_components_one_time(self):
        """Ensures correct output from _get_fourier_csi_components_one_time."""

        this_numerator, this_denominator = (
            learning_curves._get_fourier_csi_components_one_time(
                actual_target_matrix=ACTUAL_TARGET_MATRIX,
                probability_matrix=PROBABILITY_MATRIX,
                eval_mask_matrix=MASK_MATRIX
            )
        )

        self.assertTrue(numpy.isclose(
            this_numerator, FOURIER_CSI_NUMERATOR, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_denominator, FOURIER_CSI_DENOMINATOR, atol=TOLERANCE
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


if __name__ == '__main__':
    unittest.main()
