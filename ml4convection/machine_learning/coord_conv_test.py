"""Unit tests for coord_conv.py."""

import unittest
import numpy
from keras import backend as K
from ml4convection.machine_learning import coord_conv

TOLERANCE = 1e-6

NUM_EXAMPLES = 32
NUM_GRID_ROWS = 5
NUM_GRID_COLUMNS = 7
NUM_INPUT_CHANNELS_SANS_TIME = 14

THESE_DIM = (
    NUM_EXAMPLES, NUM_GRID_ROWS, NUM_GRID_COLUMNS, NUM_INPUT_CHANNELS_SANS_TIME
)
INPUT_MATRIX_SANS_TIME = numpy.random.uniform(low=5., high=6., size=THESE_DIM)
INPUT_TENSOR_SANS_TIME = K.constant(INPUT_MATRIX_SANS_TIME, dtype=float)

Y_COORD_MATRIX_SANS_TIME = numpy.array([
    [-1, -1, -1, -1, -1, -1, -1],
    [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
    [0, 0, 0, 0, 0, 0, 0],
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    [1, 1, 1, 1, 1, 1, 1]
])

X_COORD_MATRIX_SANS_TIME = numpy.array([
    [-1, -2. / 3, -1. / 3, 0, 1. / 3, 2. / 3, 1],
    [-1, -2. / 3, -1. / 3, 0, 1. / 3, 2. / 3, 1],
    [-1, -2. / 3, -1. / 3, 0, 1. / 3, 2. / 3, 1],
    [-1, -2. / 3, -1. / 3, 0, 1. / 3, 2. / 3, 1],
    [-1, -2. / 3, -1. / 3, 0, 1. / 3, 2. / 3, 1]
])

Y_COORD_MATRIX_SANS_TIME = numpy.expand_dims(Y_COORD_MATRIX_SANS_TIME, axis=0)
Y_COORD_MATRIX_SANS_TIME = numpy.repeat(
    Y_COORD_MATRIX_SANS_TIME, axis=0, repeats=NUM_EXAMPLES
)
Y_COORD_MATRIX_SANS_TIME = numpy.expand_dims(Y_COORD_MATRIX_SANS_TIME, axis=-1)

X_COORD_MATRIX_SANS_TIME = numpy.expand_dims(X_COORD_MATRIX_SANS_TIME, axis=0)
X_COORD_MATRIX_SANS_TIME = numpy.repeat(
    X_COORD_MATRIX_SANS_TIME, axis=0, repeats=NUM_EXAMPLES
)
X_COORD_MATRIX_SANS_TIME = numpy.expand_dims(X_COORD_MATRIX_SANS_TIME, axis=-1)

OUTPUT_MATRIX_SANS_TIME = numpy.concatenate((
    INPUT_MATRIX_SANS_TIME, Y_COORD_MATRIX_SANS_TIME, X_COORD_MATRIX_SANS_TIME
), axis=-1)

NUM_TIMES = 2
NUM_INPUT_CHANNELS_WITH_TIME = 7

THESE_DIM = (
    NUM_EXAMPLES, NUM_GRID_ROWS, NUM_GRID_COLUMNS, NUM_TIMES,
    NUM_INPUT_CHANNELS_WITH_TIME
)
INPUT_MATRIX_WITH_TIME = numpy.random.uniform(low=5., high=6., size=THESE_DIM)
INPUT_TENSOR_WITH_TIME = K.constant(INPUT_MATRIX_WITH_TIME, dtype=float)

Y_COORD_MATRIX_WITH_TIME = numpy.expand_dims(Y_COORD_MATRIX_SANS_TIME, axis=-2)
Y_COORD_MATRIX_WITH_TIME = numpy.repeat(
    Y_COORD_MATRIX_WITH_TIME, axis=-2, repeats=NUM_TIMES
)

X_COORD_MATRIX_WITH_TIME = numpy.expand_dims(X_COORD_MATRIX_SANS_TIME, axis=-2)
X_COORD_MATRIX_WITH_TIME = numpy.repeat(
    X_COORD_MATRIX_WITH_TIME, axis=-2, repeats=NUM_TIMES
)

OUTPUT_MATRIX_WITH_TIME = numpy.concatenate((
    INPUT_MATRIX_WITH_TIME, Y_COORD_MATRIX_WITH_TIME, X_COORD_MATRIX_WITH_TIME
), axis=-1)


class CoordConvTests(unittest.TestCase):
    """Each method is a unit test for coord_conv.py."""

    def test_add_spatial_coords_2d_sans_time(self):
        """Ensures correct output from add_spatial_coords_2d.

        In this case, data contain no time dimension.
        """

        output_tensor = coord_conv.add_spatial_coords_2d(INPUT_TENSOR_SANS_TIME)

        self.assertTrue(numpy.allclose(
            OUTPUT_MATRIX_SANS_TIME, K.eval(output_tensor), atol=TOLERANCE
        ))

    def test_add_spatial_coords_2d_with_time(self):
        """Ensures correct output from add_spatial_coords_2d.

        In this case, data contain the time dimension.
        """

        output_tensor = coord_conv.add_spatial_coords_2d(INPUT_TENSOR_WITH_TIME)

        self.assertTrue(numpy.allclose(
            OUTPUT_MATRIX_WITH_TIME, K.eval(output_tensor), atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
