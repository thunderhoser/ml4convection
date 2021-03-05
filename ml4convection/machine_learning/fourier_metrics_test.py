"""Unit tests for fourier_metrics.py."""

import unittest
import numpy
import tensorflow
from keras import backend as K
from ml4convection.machine_learning import fourier_metrics

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

# MASK_MATRIX = numpy.array([
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
# ], dtype=bool)

MASK_MATRIX = numpy.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
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

print(numpy.mean((PREDICTION_MATRIX - TARGET_MATRIX) ** 2))

TARGET_TENSOR = tensorflow.constant(TARGET_MATRIX, dtype=tensorflow.float32)
TARGET_TENSOR = tensorflow.expand_dims(TARGET_TENSOR, -1)

PREDICTION_TENSOR = tensorflow.constant(
    PREDICTION_MATRIX, dtype=tensorflow.float32
)
PREDICTION_TENSOR = tensorflow.expand_dims(PREDICTION_TENSOR, -1)


class FourierMetricsTests(unittest.TestCase):
    """Each method is a unit test for fourier_metrics.py."""

    def test_mean_squared_error(self):
        """Ensures correct output from mean_squared_error()."""

        this_function = fourier_metrics.mean_squared_error(
            spatial_coeff_matrix=numpy.full((30, 36), 1.),
            frequency_coeff_matrix=numpy.full((30, 36), 1.),
            mask_matrix=MASK_MATRIX
        )
        this_mse = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))
        print(this_mse)


if __name__ == '__main__':
    unittest.main()
