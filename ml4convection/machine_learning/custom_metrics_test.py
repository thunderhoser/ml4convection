"""Unit tests for custom_metrics.py."""

import unittest
import numpy
import tensorflow
from keras import backend as K
from ml4convection.machine_learning import custom_metrics

# TODO(thunderhoser): Need unit tests that check exact values.

TOLERANCE = 1e-6
DEFAULT_HALF_WINDOW_SIZE_PX = 1

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

TARGET_TENSOR = tensorflow.constant(TARGET_MATRIX, dtype=tensorflow.float32)
TARGET_TENSOR = tensorflow.expand_dims(TARGET_TENSOR, 0)
TARGET_TENSOR = tensorflow.expand_dims(TARGET_TENSOR, -1)

PREDICTION_TENSOR = tensorflow.constant(
    PREDICTION_MATRIX, dtype=tensorflow.float32
)
PREDICTION_TENSOR = tensorflow.expand_dims(PREDICTION_TENSOR, 0)
PREDICTION_TENSOR = tensorflow.expand_dims(PREDICTION_TENSOR, -1)

# The following constants are used to test _apply_max_filter.
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

    def test_pod(self):
        """Ensures that pod() does not crash."""

        this_function = custom_metrics.pod(
            half_window_size_px=DEFAULT_HALF_WINDOW_SIZE_PX, test_mode=True
        )
        this_pod = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(0 <= this_pod <= 1)

    def test_success_ratio(self):
        """Ensures that success_ratio() does not crash."""

        this_function = custom_metrics.success_ratio(
            half_window_size_px=DEFAULT_HALF_WINDOW_SIZE_PX, test_mode=True
        )
        this_success_ratio = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(0 <= this_success_ratio <= 1)

    def test_frequency_bias(self):
        """Ensures that frequency_bias() does not crash."""

        this_function = custom_metrics.frequency_bias(
            half_window_size_px=DEFAULT_HALF_WINDOW_SIZE_PX, test_mode=True
        )
        this_frequency_bias = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(this_frequency_bias >= 0)
        self.assertTrue(numpy.isfinite(this_frequency_bias))

    def test_csi(self):
        """Ensures that csi() does not crash."""

        this_function = custom_metrics.csi(
            half_window_size_px=DEFAULT_HALF_WINDOW_SIZE_PX, test_mode=True
        )
        this_csi = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(0 <= this_csi <= 1)

    def test_dice_coeff(self):
        """Ensures that dice_coeff() does not crash."""

        this_function = custom_metrics.dice_coeff(
            half_window_size_px=DEFAULT_HALF_WINDOW_SIZE_PX, test_mode=True
        )
        this_dice_coeff = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(0 <= this_dice_coeff <= 1)

    def test_iou(self):
        """Ensures that iou() does not crash."""

        this_function = custom_metrics.iou(
            half_window_size_px=DEFAULT_HALF_WINDOW_SIZE_PX, test_mode=True
        )
        this_iou = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(0 <= this_iou <= 1)

    def test_tversky_coeff(self):
        """Ensures that tversky_coeff() does not crash."""

        this_function = custom_metrics.tversky_coeff(
            half_window_size_px=DEFAULT_HALF_WINDOW_SIZE_PX,
            false_positive_weight=1., false_negative_weight=1., test_mode=True
        )
        this_tversky_coeff = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(0 <= this_tversky_coeff <= 1)

    def test_focal_loss(self):
        """Ensures that focal_loss() does not crash."""

        this_function = custom_metrics.focal_loss(
            half_window_size_px=DEFAULT_HALF_WINDOW_SIZE_PX,
            training_event_freq=0.5, focusing_factor=1., test_mode=True
        )
        this_focal_loss = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(this_focal_loss >= 0.)


if __name__ == '__main__':
    unittest.main()
