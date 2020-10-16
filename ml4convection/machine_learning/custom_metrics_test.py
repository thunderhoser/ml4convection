"""Unit tests for custom_metrics.py."""

import unittest
import numpy
import tensorflow
from keras import backend as K
from ml4convection.machine_learning import custom_metrics

# TODO(thunderhoser): Still need to check exact values for focal_loss.

TOLERANCE = 1e-6
SMALL_NUMBER = K.eval(K.epsilon())

FALSE_POSITIVE_WEIGHT = 1.
FALSE_NEGATIVE_WEIGHT = 1.

TARGET_MATRIX = numpy.array([
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

TARGET_MATRIX = numpy.stack((TARGET_MATRIX, TARGET_MATRIX), axis=0)

PREDICTION_MATRIX = numpy.array([
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

PREDICTION_MATRIX = numpy.stack((PREDICTION_MATRIX, PREDICTION_MATRIX), axis=0)

TARGET_TENSOR = tensorflow.constant(TARGET_MATRIX, dtype=tensorflow.float32)
TARGET_TENSOR = tensorflow.expand_dims(TARGET_TENSOR, -1)

PREDICTION_TENSOR = tensorflow.constant(
    PREDICTION_MATRIX, dtype=tensorflow.float32
)
PREDICTION_TENSOR = tensorflow.expand_dims(PREDICTION_TENSOR, -1)

FILTERED_TARGET_MATRIX_HW1 = numpy.array([
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

FILTERED_TARGET_MATRIX_HW1 = numpy.stack(
    (FILTERED_TARGET_MATRIX_HW1, FILTERED_TARGET_MATRIX_HW1), axis=0
)

FILTERED_PREDICTION_MATRIX_HW1 = numpy.array([
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

FILTERED_PREDICTION_MATRIX_HW1 = numpy.stack(
    (FILTERED_PREDICTION_MATRIX_HW1, FILTERED_PREDICTION_MATRIX_HW1), axis=0
)

POD_HW0 = 0.
SUCCESS_RATIO_HW0 = 0.
FREQUENCY_BIAS_HW0 = 0.

THIS_POD_TERM = (SMALL_NUMBER + POD_HW0) ** -1
THIS_SUCCESS_RATIO_TERM = (SMALL_NUMBER + SUCCESS_RATIO_HW0) ** -1
CSI_HW0 = (THIS_POD_TERM + THIS_SUCCESS_RATIO_TERM - 1) ** -1

DICE_COEFF_HW0 = 0.
IOU_HW0 = 0.
TVERSKY_COEFF_HW0 = 0.

POD_HW1 = 4. / (16 + SMALL_NUMBER)
SUCCESS_RATIO_HW1 = 4. / (23.2 + SMALL_NUMBER)
FREQUENCY_BIAS_HW1 = POD_HW1 / (SUCCESS_RATIO_HW1 + SMALL_NUMBER)

THIS_POD_TERM = (SMALL_NUMBER + POD_HW1) ** -1
THIS_SUCCESS_RATIO_TERM = (SMALL_NUMBER + SUCCESS_RATIO_HW1) ** -1
CSI_HW1 = (THIS_POD_TERM + THIS_SUCCESS_RATIO_TERM - 1) ** -1

DICE_COEFF_HW1 = 32. / 120
IOU_HW1 = 16. / (36 + 65.2 - 16 + SMALL_NUMBER)
TVERSKY_COEFF_HW1 = 16. / (16 + 49.2 + 20 + SMALL_NUMBER)


class CustomMetricsTests(unittest.TestCase):
    """Each method is a unit test for custom_metrics.py."""

    def test_apply_max_filter_targets_hw0(self):
        """Ensures correct output from _apply_max_filter.

        In this case, applying filter to target matrix with half-window size =
        0 pixels.
        """

        this_tensor = custom_metrics._apply_max_filter(
            input_tensor=TARGET_TENSOR, half_window_size_px=0, test_mode=True
        )
        this_matrix = K.eval(this_tensor)[0, ..., 0]

        self.assertTrue(numpy.allclose(
            this_matrix, TARGET_MATRIX, atol=TOLERANCE
        ))

    def test_apply_max_filter_targets_hw1(self):
        """Ensures correct output from _apply_max_filter.

        In this case, applying filter to target matrix with half-window size =
        1 pixel.
        """

        this_tensor = custom_metrics._apply_max_filter(
            input_tensor=TARGET_TENSOR, half_window_size_px=1, test_mode=True
        )
        this_matrix = K.eval(this_tensor)[0, ..., 0]

        self.assertTrue(numpy.allclose(
            this_matrix, FILTERED_TARGET_MATRIX_HW1, atol=TOLERANCE
        ))

    def test_apply_max_filter_predictions_hw0(self):
        """Ensures correct output from _apply_max_filter on predictions.

        In this case, applying filter to prediction matrix with half-window
        size = 0 pixels.
        """

        this_tensor = custom_metrics._apply_max_filter(
            input_tensor=PREDICTION_TENSOR,
            half_window_size_px=0, test_mode=True
        )
        this_matrix = K.eval(this_tensor)[0, ..., 0]

        self.assertTrue(numpy.allclose(
            this_matrix, PREDICTION_MATRIX, atol=TOLERANCE
        ))

    def test_apply_max_filter_predictions_hw1(self):
        """Ensures correct output from _apply_max_filter on predictions.

        In this case, applying filter to prediction matrix with half-window
        size = 1 pixel.
        """

        this_tensor = custom_metrics._apply_max_filter(
            input_tensor=PREDICTION_TENSOR,
            half_window_size_px=1, test_mode=True
        )
        this_matrix = K.eval(this_tensor)[0, ..., 0]

        self.assertTrue(numpy.allclose(
            this_matrix, FILTERED_PREDICTION_MATRIX_HW1, atol=TOLERANCE
        ))

    def test_pod_hw0(self):
        """Ensures correct output from pod().

        In this case, half-window size = 0 pixels.
        """

        this_function = custom_metrics.pod(
            half_window_size_px=0, test_mode=True
        )
        this_pod = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(numpy.isclose(this_pod, POD_HW0, atol=TOLERANCE))

    def test_pod_hw1(self):
        """Ensures correct output from pod().

        In this case, half-window size = 1 pixel.
        """

        this_function = custom_metrics.pod(
            half_window_size_px=1, test_mode=True
        )
        this_pod = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(numpy.isclose(this_pod, POD_HW1, atol=TOLERANCE))

    def test_success_ratio_hw0(self):
        """Ensures correct output from success_ratio().

        In this case, half-window size = 0 pixels.
        """

        this_function = custom_metrics.success_ratio(
            half_window_size_px=0, test_mode=True
        )
        this_success_ratio = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_success_ratio, SUCCESS_RATIO_HW0, atol=TOLERANCE
        ))

    def test_success_ratio_hw1(self):
        """Ensures correct output from success_ratio().

        In this case, half-window size = 1 pixel.
        """

        this_function = custom_metrics.success_ratio(
            half_window_size_px=1, test_mode=True
        )
        this_success_ratio = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_success_ratio, SUCCESS_RATIO_HW1, atol=TOLERANCE
        ))

    def test_frequency_bias_hw0(self):
        """Ensures correct output from frequency_bias().

        In this case, half-window size = 0 pixels.
        """

        this_function = custom_metrics.frequency_bias(
            half_window_size_px=0, test_mode=True
        )
        this_frequency_bias = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_frequency_bias, FREQUENCY_BIAS_HW0, atol=TOLERANCE
        ))

    def test_frequency_bias_hw1(self):
        """Ensures correct output from frequency_bias().

        In this case, half-window size = 1 pixel.
        """

        this_function = custom_metrics.frequency_bias(
            half_window_size_px=1, test_mode=True
        )
        this_frequency_bias = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_frequency_bias, FREQUENCY_BIAS_HW1, atol=TOLERANCE
        ))

    def test_csi_hw0(self):
        """Ensures correct output from csi().

        In this case, half-window size = 0 pixels.
        """

        this_function = custom_metrics.csi(
            half_window_size_px=0, test_mode=True
        )
        this_csi = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(numpy.isclose(this_csi, CSI_HW0, atol=TOLERANCE))

    def test_csi_hw1(self):
        """Ensures correct output from csi().

        In this case, half-window size = 1 pixel.
        """

        this_function = custom_metrics.csi(
            half_window_size_px=1, test_mode=True
        )
        this_csi = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(numpy.isclose(this_csi, CSI_HW1, atol=TOLERANCE))

    def test_dice_coeff_hw0(self):
        """Ensures correct output from dice_coeff().

        In this case, half-window size = 0 pixels.
        """

        this_function = custom_metrics.dice_coeff(
            half_window_size_px=0, test_mode=True
        )
        this_dice_coeff = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_dice_coeff, DICE_COEFF_HW0, atol=TOLERANCE
        ))

    def test_dice_coeff_hw1(self):
        """Ensures correct output from dice_coeff().

        In this case, half-window size = 1 pixel.
        """

        this_function = custom_metrics.dice_coeff(
            half_window_size_px=1, test_mode=True
        )
        this_dice_coeff = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_dice_coeff, DICE_COEFF_HW1, atol=TOLERANCE
        ))

    def test_iou_hw0(self):
        """Ensures correct output from iou().

        In this case, half-window size = 0 pixels.
        """

        this_function = custom_metrics.iou(
            half_window_size_px=0, test_mode=True
        )
        this_iou = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(numpy.isclose(this_iou, IOU_HW0, atol=TOLERANCE))

    def test_iou_hw1(self):
        """Ensures correct output from iou().

        In this case, half-window size = 1 pixel.
        """

        this_function = custom_metrics.iou(
            half_window_size_px=1, test_mode=True
        )
        this_iou = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(numpy.isclose(this_iou, IOU_HW1, atol=TOLERANCE))

    def test_tversky_coeff_hw0(self):
        """Ensures correct output from tversky_coeff().

        In this case, half-window size = 0 pixels.
        """

        this_function = custom_metrics.tversky_coeff(
            half_window_size_px=0,
            false_positive_weight=FALSE_POSITIVE_WEIGHT,
            false_negative_weight=FALSE_NEGATIVE_WEIGHT,
            test_mode=True
        )
        this_tversky_coeff = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_tversky_coeff, TVERSKY_COEFF_HW0, atol=TOLERANCE
        ))

    def test_tversky_coeff_hw1(self):
        """Ensures correct output from tversky_coeff().

        In this case, half-window size = 1 pixel.
        """

        this_function = custom_metrics.tversky_coeff(
            half_window_size_px=1,
            false_positive_weight=FALSE_POSITIVE_WEIGHT,
            false_negative_weight=FALSE_NEGATIVE_WEIGHT,
            test_mode=True
        )
        this_tversky_coeff = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_tversky_coeff, TVERSKY_COEFF_HW1, atol=TOLERANCE
        ))

    def test_focal_loss(self):
        """Ensures correct output from focal_loss().

        In this case, half-window size = 0 pixels.
        """

        this_function = custom_metrics.focal_loss(
            half_window_size_px=0,
            training_event_freq=0.5, focusing_factor=1., test_mode=True
        )
        this_focal_loss = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(this_focal_loss >= 0.)


if __name__ == '__main__':
    unittest.main()
