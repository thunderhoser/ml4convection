"""Unit tests for evaluation.py."""

import unittest
import numpy
from ml4convection.utils import evaluation
from ml4convection.utils import spatial_evaluation_test as spatial_eval_test

# TODO(thunderhoser): Still need to test last 3 methods.

TOLERANCE = 1e-6

# The following constants are used to test _dilate_binary_matrix and
# _erode_binary_matrix.
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

# The following constants are used to test _get_pod_get_pod, _get_success_ratio,
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


class EvaluationTests(unittest.TestCase):
    """Each method is a unit test for evaluation.py."""

    def test_dilate_binary_matrix_first(self):
        """Ensures correct output from _dilate_binary_matrix.

        In this case, using first dilation distance.
        """

        this_matrix = evaluation._dilate_binary_matrix(
            binary_matrix=ORIG_MASK_MATRIX,
            buffer_distance_px=FIRST_DILATION_DISTANCE_PX
        )

        self.assertTrue(numpy.array_equal(
            this_matrix, FIRST_DILATED_MASK_MATRIX
        ))

    def test_dilate_binary_matrix_second(self):
        """Ensures correct output from _dilate_binary_matrix.

        In this case, using second dilation distance.
        """

        this_matrix = evaluation._dilate_binary_matrix(
            binary_matrix=ORIG_MASK_MATRIX,
            buffer_distance_px=SECOND_DILATION_DISTANCE_PX
        )

        self.assertTrue(numpy.array_equal(
            this_matrix, SECOND_DILATED_MASK_MATRIX
        ))

    def test_dilate_binary_matrix_third(self):
        """Ensures correct output from _dilate_binary_matrix.

        In this case, using third dilation distance.
        """

        this_matrix = evaluation._dilate_binary_matrix(
            binary_matrix=ORIG_MASK_MATRIX,
            buffer_distance_px=THIRD_DILATION_DISTANCE_PX
        )

        self.assertTrue(numpy.array_equal(
            this_matrix, THIRD_DILATED_MASK_MATRIX
        ))

    def test_dilate_binary_matrix_fourth(self):
        """Ensures correct output from _dilate_binary_matrix.

        In this case, using fourth dilation distance.
        """

        this_matrix = evaluation._dilate_binary_matrix(
            binary_matrix=ORIG_MASK_MATRIX,
            buffer_distance_px=FOURTH_DILATION_DISTANCE_PX
        )

        self.assertTrue(numpy.array_equal(
            this_matrix, FOURTH_DILATED_MASK_MATRIX
        ))

    def test_erode_binary_matrix_first(self):
        """Ensures correct output from _erode_binary_matrix.

        In this case, using first erosion distance.
        """

        this_matrix = evaluation._erode_binary_matrix(
            binary_matrix=ORIG_MASK_MATRIX,
            buffer_distance_px=FIRST_DILATION_DISTANCE_PX
        )

        self.assertTrue(numpy.array_equal(
            this_matrix, FIRST_ERODED_MASK_MATRIX
        ))

    def test_erode_binary_matrix_second(self):
        """Ensures correct output from _erode_binary_matrix.

        In this case, using second erosion distance.
        """

        this_matrix = evaluation._erode_binary_matrix(
            binary_matrix=ORIG_MASK_MATRIX,
            buffer_distance_px=SECOND_DILATION_DISTANCE_PX
        )

        self.assertTrue(numpy.array_equal(
            this_matrix, SECOND_ERODED_MASK_MATRIX
        ))

    def test_erode_binary_matrix_third(self):
        """Ensures correct output from _erode_binary_matrix.

        In this case, using third erosion distance.
        """

        this_matrix = evaluation._erode_binary_matrix(
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

        this_mask_matrix = evaluation._erode_binary_matrix(
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

        this_mask_matrix = evaluation._erode_binary_matrix(
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


if __name__ == '__main__':
    unittest.main()
