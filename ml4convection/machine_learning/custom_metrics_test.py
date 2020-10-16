"""Unit tests for custom_metrics.py."""

import unittest
import numpy
import tensorflow
from keras import backend as K
from ml4convection.machine_learning import custom_metrics

TOLERANCE = 1e-6

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

TARGET_TENSOR = tensorflow.constant(TARGET_MATRIX, dtype=tensorflow.float32)
TARGET_TENSOR = tensorflow.expand_dims(TARGET_TENSOR, 0)
TARGET_TENSOR = tensorflow.expand_dims(TARGET_TENSOR, -1)

PREDICTION_TENSOR = tensorflow.constant(
    PROBABILITY_MATRIX, dtype=tensorflow.float32
)
PREDICTION_TENSOR = tensorflow.expand_dims(PREDICTION_TENSOR, 0)
PREDICTION_TENSOR = tensorflow.expand_dims(PREDICTION_TENSOR, -1)

# The following constants are used to test _apply_max_filter.
HALF_WINDOW_SIZE_PX = 1

FILTERED_TARGET_MATRIX = numpy.array([
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

FILTERED_PREDICTION_MATRIX = numpy.array([
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

    def test_apply_max_filter_targets(self):
        """Ensures correct output from _apply_max_filter on target matrix."""

        this_tensor = custom_metrics._apply_max_filter(
            input_tensor=TARGET_TENSOR, half_window_size_px=HALF_WINDOW_SIZE_PX,
            test_mode=True
        )
        this_matrix = K.eval(this_tensor)[0, ..., 0]

        self.assertTrue(numpy.allclose(
            this_matrix, FILTERED_TARGET_MATRIX, atol=TOLERANCE
        ))

    def test_apply_max_filter_predictions(self):
        """Ensures correct output from _apply_max_filter on predictions."""

        this_tensor = custom_metrics._apply_max_filter(
            input_tensor=PREDICTION_TENSOR,
            half_window_size_px=HALF_WINDOW_SIZE_PX, test_mode=True
        )
        this_matrix = K.eval(this_tensor)[0, ..., 0]

        self.assertTrue(numpy.allclose(
            this_matrix, FILTERED_PREDICTION_MATRIX, atol=TOLERANCE
        ))

    def test_pod(self):
        """Ensures that pod() does not crash."""

        this_function = custom_metrics.pod(
            half_window_size_px=HALF_WINDOW_SIZE_PX, test_mode=True
        )
        this_pod = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(0 <= this_pod <= 1)

    def test_success_ratio(self):
        """Ensures that success_ratio() does not crash."""

        this_function = custom_metrics.success_ratio(
            half_window_size_px=HALF_WINDOW_SIZE_PX, test_mode=True
        )
        this_success_ratio = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(0 <= this_success_ratio <= 1)

    def test_frequency_bias(self):
        """Ensures that frequency_bias() does not crash."""

        this_function = custom_metrics.frequency_bias(
            half_window_size_px=HALF_WINDOW_SIZE_PX, test_mode=True
        )
        this_frequency_bias = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(this_frequency_bias >= 0)
        self.assertTrue(numpy.isfinite(this_frequency_bias))

    def test_csi(self):
        """Ensures that csi() does not crash."""

        this_function = custom_metrics.csi(
            half_window_size_px=HALF_WINDOW_SIZE_PX, test_mode=True
        )
        this_csi = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(0 <= this_csi <= 1)


if __name__ == '__main__':
    unittest.main()
