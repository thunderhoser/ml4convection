"""Unit tests for custom_metrics.py."""

import unittest
import numpy
import tensorflow
from keras import backend as K
from ml4convection.machine_learning import custom_metrics

TOLERANCE = 1e-6
SMALL_NUMBER = K.eval(K.epsilon())

TARGET_MATRIX_EXAMPLE1 = numpy.array([
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

TARGET_MATRIX_EXAMPLE2 = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=int)

TARGET_MATRIX = numpy.stack(
    (TARGET_MATRIX_EXAMPLE1, TARGET_MATRIX_EXAMPLE2), axis=0
)

MASK_MATRIX = numpy.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
], dtype=bool)

PREDICTION_MATRIX_EXAMPLE1 = numpy.array([
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

PREDICTION_MATRIX_EXAMPLE2 = numpy.array([
    [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1],
    [0.5, 0.6, 0.7, 0.8, 0.0, 0.0, 0.0, 0.0, 0.8, 0.7, 0.6, 0.5],
    [0.2, 0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.4, 0.2],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.2, 0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.4, 0.2],
    [0.5, 0.6, 0.7, 0.8, 0.0, 0.0, 0.0, 0.0, 0.8, 0.7, 0.6, 0.5],
    [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2, 0.1]
])

PREDICTION_MATRIX = numpy.stack(
    (PREDICTION_MATRIX_EXAMPLE1, PREDICTION_MATRIX_EXAMPLE2), axis=0
)

TARGET_TENSOR = tensorflow.constant(TARGET_MATRIX, dtype=tensorflow.float32)
TARGET_TENSOR = tensorflow.expand_dims(TARGET_TENSOR, -1)

PREDICTION_TENSOR = tensorflow.constant(
    PREDICTION_MATRIX, dtype=tensorflow.float32
)
PREDICTION_TENSOR = tensorflow.expand_dims(PREDICTION_TENSOR, -1)

THIS_FIRST_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=int)

THIS_SECOND_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=int)

FILTERED_TARGET_MATRIX_NEIGH1 = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0
)

THIS_FIRST_MATRIX = numpy.array([
    [0.6, 0.7, 0.8, 0.8, 0.8, 0.0, 0.0, 0.8, 0.8, 0.8, 0.7, 0.6],
    [0.6, 0.7, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.7, 0.6],
    [0.6, 0.7, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.7, 0.6],
    [0.4, 0.6, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.6, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.4, 0.6, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.6, 0.4],
    [0.6, 0.7, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.7, 0.6],
    [0.6, 0.7, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.7, 0.6],
    [0.6, 0.7, 0.8, 0.8, 0.8, 0.0, 0.0, 0.8, 0.8, 0.8, 0.7, 0.6]
])

THIS_SECOND_MATRIX = numpy.array([
    [0.6, 0.7, 0.8, 0.8, 0.8, 0.0, 0.0, 0.8, 0.8, 0.8, 0.7, 0.6],
    [0.6, 0.7, 0.8, 0.8, 0.8, 0.0, 0.0, 0.8, 0.8, 0.8, 0.7, 0.6],
    [0.6, 0.7, 0.8, 0.8, 0.8, 0.0, 0.0, 0.8, 0.8, 0.8, 0.7, 0.6],
    [0.4, 0.6, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.6, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.4, 0.6, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.6, 0.4],
    [0.6, 0.7, 0.8, 0.8, 0.8, 0.0, 0.0, 0.8, 0.8, 0.8, 0.7, 0.6],
    [0.6, 0.7, 0.8, 0.8, 0.8, 0.0, 0.0, 0.8, 0.8, 0.8, 0.7, 0.6],
    [0.6, 0.7, 0.8, 0.8, 0.8, 0.0, 0.0, 0.8, 0.8, 0.8, 0.7, 0.6]
])

FILTERED_PREDICTION_MATRIX_NEIGH1 = numpy.stack(
    (THIS_FIRST_MATRIX, THIS_SECOND_MATRIX), axis=0
)

POD_NEIGH0 = 0.
SUCCESS_RATIO_NEIGH0 = 0.
FREQUENCY_BIAS_NEIGH0 = 0.
POSITIVE_IOU_NEIGH0 = 0.
BRIER_SCORE_NEIGH0 = (30.36 + 34.36) / 232

NEGATIVE_IOU_NEIGH0 = 0.5 * (77.2 + 73.2) / (116 + SMALL_NUMBER)
ALL_CLASS_IOU_NEIGH0 = 0.5 * (NEGATIVE_IOU_NEIGH0 + POSITIVE_IOU_NEIGH0)

THIS_POD_TERM = (SMALL_NUMBER + POD_NEIGH0) ** -1
THIS_SUCCESS_RATIO_TERM = (SMALL_NUMBER + SUCCESS_RATIO_NEIGH0) ** -1
CSI_NEIGH0 = (THIS_POD_TERM + THIS_SUCCESS_RATIO_TERM - 1) ** -1

DICE_COEFF_NEIGH0_EXAMPLE1 = 77.2 / 116
DICE_COEFF_NEIGH0_EXAMPLE2 = 73.2 / 116
DICE_COEFF_NEIGH0 = numpy.mean([
    DICE_COEFF_NEIGH0_EXAMPLE1, DICE_COEFF_NEIGH0_EXAMPLE2
])

POD_NEIGH1 = (4 + 2.4) / (16 + 24 + SMALL_NUMBER)
SUCCESS_RATIO_NEIGH1 = (4 + 2.4) / (17.6 + 13.6 + SMALL_NUMBER)
FREQUENCY_BIAS_NEIGH1 = POD_NEIGH1 / (SUCCESS_RATIO_NEIGH1 + SMALL_NUMBER)

THIS_POD_TERM = (SMALL_NUMBER + POD_NEIGH1) ** -1
THIS_SUCCESS_RATIO_TERM = (SMALL_NUMBER + SUCCESS_RATIO_NEIGH1) ** -1
CSI_NEIGH1 = (THIS_POD_TERM + THIS_SUCCESS_RATIO_TERM - 1) ** -1

POSITIVE_IOU_NEIGH1_EXAMPLE1 = 4. / (49.6 + SMALL_NUMBER)
POSITIVE_IOU_NEIGH1_EXAMPLE2 = 2.4 / (59.2 + SMALL_NUMBER)
POSITIVE_IOU_NEIGH1 = numpy.mean([
    POSITIVE_IOU_NEIGH1_EXAMPLE1, POSITIVE_IOU_NEIGH1_EXAMPLE2
])

NEGATIVE_IOU_NEIGH1_EXAMPLE1 = 54.4 / (100 + SMALL_NUMBER)
NEGATIVE_IOU_NEIGH1_EXAMPLE2 = 44.8 / (101.6 + SMALL_NUMBER)
NEGATIVE_IOU_NEIGH1 = numpy.mean([
    NEGATIVE_IOU_NEIGH1_EXAMPLE1, NEGATIVE_IOU_NEIGH1_EXAMPLE2
])

ALL_CLASS_IOU_NEIGH1 = 0.5 * (NEGATIVE_IOU_NEIGH1 + POSITIVE_IOU_NEIGH1)

DICE_COEFF_NEIGH1_EXAMPLE1 = 58.4 / 104
DICE_COEFF_NEIGH1_EXAMPLE2 = (2.4 + 44.8) / 104
DICE_COEFF_NEIGH1 = numpy.mean([
    DICE_COEFF_NEIGH1_EXAMPLE1, DICE_COEFF_NEIGH1_EXAMPLE2
])

BRIER_SCORE_NEIGH1 = (39.76 + 50.96) / 208


class CustomMetricsTests(unittest.TestCase):
    """Each method is a unit test for custom_metrics.py."""

    def test_apply_max_filter_targets_neigh0(self):
        """Ensures correct output from _apply_max_filter.

        In this case, applying filter to target matrix with half-window size =
        0 pixels.
        """

        this_tensor = custom_metrics._apply_max_filter(
            input_tensor=TARGET_TENSOR, half_window_size_px=0
        )
        this_matrix = K.eval(this_tensor)[..., 0]

        self.assertTrue(numpy.allclose(
            this_matrix, TARGET_MATRIX, atol=TOLERANCE
        ))

    def test_apply_max_filter_targets_neigh1(self):
        """Ensures correct output from _apply_max_filter.

        In this case, applying filter to target matrix with half-window size =
        1 pixel.
        """

        this_tensor = custom_metrics._apply_max_filter(
            input_tensor=TARGET_TENSOR, half_window_size_px=1
        )
        this_matrix = K.eval(this_tensor)[..., 0]

        self.assertTrue(numpy.allclose(
            this_matrix, FILTERED_TARGET_MATRIX_NEIGH1, atol=TOLERANCE
        ))

    def test_apply_max_filter_predictions_neigh0(self):
        """Ensures correct output from _apply_max_filter on predictions.

        In this case, applying filter to prediction matrix with half-window
        size = 0 pixels.
        """

        this_tensor = custom_metrics._apply_max_filter(
            input_tensor=PREDICTION_TENSOR, half_window_size_px=0
        )
        this_matrix = K.eval(this_tensor)[..., 0]

        self.assertTrue(numpy.allclose(
            this_matrix, PREDICTION_MATRIX, atol=TOLERANCE
        ))

    def test_apply_max_filter_predictions_neigh1(self):
        """Ensures correct output from _apply_max_filter on predictions.

        In this case, applying filter to prediction matrix with half-window
        size = 1 pixel.
        """

        this_tensor = custom_metrics._apply_max_filter(
            input_tensor=PREDICTION_TENSOR, half_window_size_px=1
        )
        this_matrix = K.eval(this_tensor)[..., 0]

        self.assertTrue(numpy.allclose(
            this_matrix, FILTERED_PREDICTION_MATRIX_NEIGH1, atol=TOLERANCE
        ))

    def test_pod_neigh0(self):
        """Ensures correct output from pod().

        In this case, half-window size = 0 pixels.
        """

        this_function = custom_metrics.pod(
            half_window_size_px=0, mask_matrix=MASK_MATRIX, test_mode=True
        )
        this_pod = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(numpy.isclose(this_pod, POD_NEIGH0, atol=TOLERANCE))

    def test_pod_neigh1(self):
        """Ensures correct output from pod().

        In this case, half-window size = 1 pixel.
        """

        this_function = custom_metrics.pod(
            half_window_size_px=1, mask_matrix=MASK_MATRIX, test_mode=True
        )
        this_pod = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(numpy.isclose(this_pod, POD_NEIGH1, atol=TOLERANCE))

    def test_success_ratio_neigh0(self):
        """Ensures correct output from success_ratio().

        In this case, half-window size = 0 pixels.
        """

        this_function = custom_metrics.success_ratio(
            half_window_size_px=0, mask_matrix=MASK_MATRIX, test_mode=True
        )
        this_success_ratio = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_success_ratio, SUCCESS_RATIO_NEIGH0, atol=TOLERANCE
        ))

    def test_success_ratio_neigh1(self):
        """Ensures correct output from success_ratio().

        In this case, half-window size = 1 pixel.
        """

        this_function = custom_metrics.success_ratio(
            half_window_size_px=1, mask_matrix=MASK_MATRIX, test_mode=True
        )
        this_success_ratio = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_success_ratio, SUCCESS_RATIO_NEIGH1, atol=TOLERANCE
        ))

    def test_frequency_bias_neigh0(self):
        """Ensures correct output from frequency_bias().

        In this case, half-window size = 0 pixels.
        """

        this_function = custom_metrics.frequency_bias(
            half_window_size_px=0, mask_matrix=MASK_MATRIX, test_mode=True
        )
        this_frequency_bias = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_frequency_bias, FREQUENCY_BIAS_NEIGH0, atol=TOLERANCE
        ))

    def test_frequency_bias_neigh1(self):
        """Ensures correct output from frequency_bias().

        In this case, half-window size = 1 pixel.
        """

        this_function = custom_metrics.frequency_bias(
            half_window_size_px=1, mask_matrix=MASK_MATRIX, test_mode=True
        )
        this_frequency_bias = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_frequency_bias, FREQUENCY_BIAS_NEIGH1, atol=TOLERANCE
        ))

    def test_csi_neigh0(self):
        """Ensures correct output from csi().

        In this case, half-window size = 0 pixels.
        """

        this_function = custom_metrics.csi(
            half_window_size_px=0, mask_matrix=MASK_MATRIX, test_mode=True
        )
        this_csi = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(numpy.isclose(this_csi, CSI_NEIGH0, atol=TOLERANCE))

    def test_csi_neigh1(self):
        """Ensures correct output from csi().

        In this case, half-window size = 1 pixel.
        """

        this_function = custom_metrics.csi(
            half_window_size_px=1, mask_matrix=MASK_MATRIX, test_mode=True
        )
        this_csi = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(numpy.isclose(this_csi, CSI_NEIGH1, atol=TOLERANCE))

    def test_dice_coeff_neigh0(self):
        """Ensures correct output from dice_coeff().

        In this case, half-window size = 0 pixels.
        """

        this_function = custom_metrics.dice_coeff(
            half_window_size_px=0, mask_matrix=MASK_MATRIX, test_mode=True
        )
        this_dice_coeff = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_dice_coeff, DICE_COEFF_NEIGH0, atol=TOLERANCE
        ))

    def test_dice_coeff_neigh1(self):
        """Ensures correct output from dice_coeff().

        In this case, half-window size = 1 pixel.
        """

        this_function = custom_metrics.dice_coeff(
            half_window_size_px=1, mask_matrix=MASK_MATRIX, test_mode=True
        )
        this_dice_coeff = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_dice_coeff, DICE_COEFF_NEIGH1, atol=TOLERANCE
        ))

    def test_iou_neigh0(self):
        """Ensures correct output from iou().

        In this case, half-window size = 0 pixels.
        """

        this_function = custom_metrics.iou(
            half_window_size_px=0, mask_matrix=MASK_MATRIX, test_mode=True
        )
        this_iou = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(numpy.isclose(
            this_iou, POSITIVE_IOU_NEIGH0, atol=TOLERANCE
        ))

    def test_iou_neigh1(self):
        """Ensures correct output from iou().

        In this case, half-window size = 1 pixel.
        """

        this_function = custom_metrics.iou(
            half_window_size_px=1, mask_matrix=MASK_MATRIX, test_mode=True
        )
        this_iou = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(numpy.isclose(
            this_iou, POSITIVE_IOU_NEIGH1, atol=TOLERANCE
        ))

    def test_all_class_iou_neigh0(self):
        """Ensures correct output from all_class_iou().

        In this case, half-window size = 0 pixels.
        """

        this_function = custom_metrics.all_class_iou(
            half_window_size_px=0, mask_matrix=MASK_MATRIX, test_mode=True
        )
        this_all_class_iou = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_all_class_iou, ALL_CLASS_IOU_NEIGH0, atol=TOLERANCE
        ))

    def test_all_class_iou_neigh1(self):
        """Ensures correct output from all_class_iou().

        In this case, half-window size = 1 pixel.
        """

        this_function = custom_metrics.all_class_iou(
            half_window_size_px=1, mask_matrix=MASK_MATRIX, test_mode=True
        )
        this_all_class_iou = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_all_class_iou, ALL_CLASS_IOU_NEIGH1, atol=TOLERANCE
        ))

    def test_brier_score_neigh0(self):
        """Ensures correct output from brier_score().

        In this case, half-window size = 0 pixels.
        """

        this_function = custom_metrics.brier_score(
            half_window_size_px=0, mask_matrix=MASK_MATRIX, test_mode=True
        )
        this_brier_score = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_brier_score, BRIER_SCORE_NEIGH0, atol=TOLERANCE
        ))

    def test_brier_score_neigh1(self):
        """Ensures correct output from brier_score().

        In this case, half-window size = 1 pixel.
        """

        this_function = custom_metrics.brier_score(
            half_window_size_px=1, mask_matrix=MASK_MATRIX, test_mode=True
        )
        this_brier_score = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_brier_score, BRIER_SCORE_NEIGH1, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
