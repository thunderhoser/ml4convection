"""Unit tests for standalone_utils.py."""

import unittest
import numpy
from ml4convection.machine_learning import standalone_utils
from ml4convection.machine_learning import custom_losses_test

TOLERANCE = 1e-6

# The following constants are used to test do_2d_convolution.
TARGET_MATRIX = numpy.expand_dims(custom_losses_test.TARGET_MATRIX, axis=0)
TARGET_MATRIX = numpy.expand_dims(TARGET_MATRIX, axis=-1)
TARGET_MATRIX = numpy.repeat(TARGET_MATRIX, repeats=3, axis=0)

PREDICTION_MATRIX = numpy.expand_dims(
    custom_losses_test.PREDICTION_MATRIX, axis=0
)
PREDICTION_MATRIX = numpy.expand_dims(PREDICTION_MATRIX, axis=-1)
PREDICTION_MATRIX = numpy.repeat(PREDICTION_MATRIX, repeats=3, axis=0)

TARGET_MATRIX_SMOOTHED1 = numpy.expand_dims(
    custom_losses_test.TARGET_MATRIX_SMOOTHED1, axis=0
)
TARGET_MATRIX_SMOOTHED1 = numpy.expand_dims(TARGET_MATRIX_SMOOTHED1, axis=-1)
TARGET_MATRIX_SMOOTHED1 = numpy.repeat(
    TARGET_MATRIX_SMOOTHED1, repeats=3, axis=0
)

PREDICTION_MATRIX_SMOOTHED1 = numpy.expand_dims(
    custom_losses_test.PREDICTION_MATRIX_SMOOTHED1, axis=0
)
PREDICTION_MATRIX_SMOOTHED1 = numpy.expand_dims(
    PREDICTION_MATRIX_SMOOTHED1, axis=-1
)
PREDICTION_MATRIX_SMOOTHED1 = numpy.repeat(
    PREDICTION_MATRIX_SMOOTHED1, repeats=3, axis=0
)

TARGET_MATRIX_SMOOTHED2 = numpy.expand_dims(
    custom_losses_test.TARGET_MATRIX_SMOOTHED2, axis=0
)
TARGET_MATRIX_SMOOTHED2 = numpy.expand_dims(TARGET_MATRIX_SMOOTHED2, axis=-1)
TARGET_MATRIX_SMOOTHED2 = numpy.repeat(
    TARGET_MATRIX_SMOOTHED2, repeats=3, axis=0
)

PREDICTION_MATRIX_SMOOTHED2 = numpy.expand_dims(
    custom_losses_test.PREDICTION_MATRIX_SMOOTHED2, axis=0
)
PREDICTION_MATRIX_SMOOTHED2 = numpy.expand_dims(
    PREDICTION_MATRIX_SMOOTHED2, axis=-1
)
PREDICTION_MATRIX_SMOOTHED2 = numpy.repeat(
    PREDICTION_MATRIX_SMOOTHED2, repeats=3, axis=0
)

WEIGHT_MATRIX_SIZE0 = numpy.full((1, 1, 1, 1), 1.)
WEIGHT_MATRIX_SIZE1 = numpy.full((3, 3, 1, 1), 1. / 9)
WEIGHT_MATRIX_SIZE2 = numpy.full((5, 5, 1, 1), 1. / 25)


class StandaloneUtilsTests(unittest.TestCase):
    """Each method is a unit test for standalone_utils.py."""

    def test_do_2d_convolution_size0(self):
        """Ensures correct output from do_2d_convolution.

        In this case, half-window size is 0 pixels.
        """

        this_target_matrix = standalone_utils.do_2d_convolution(
            feature_matrix=TARGET_MATRIX, kernel_matrix=WEIGHT_MATRIX_SIZE0,
            pad_edges=True, stride_length_px=1
        )
        self.assertTrue(numpy.allclose(
            this_target_matrix, TARGET_MATRIX, atol=TOLERANCE
        ))

        this_prediction_matrix = standalone_utils.do_2d_convolution(
            feature_matrix=PREDICTION_MATRIX, kernel_matrix=WEIGHT_MATRIX_SIZE0,
            pad_edges=True, stride_length_px=1
        )
        self.assertTrue(numpy.allclose(
            this_prediction_matrix, PREDICTION_MATRIX, atol=TOLERANCE
        ))

    def test_do_2d_convolution_size1(self):
        """Ensures correct output from do_2d_convolution.

        In this case, half-window size is 1 pixel.
        """

        this_target_matrix = standalone_utils.do_2d_convolution(
            feature_matrix=TARGET_MATRIX, kernel_matrix=WEIGHT_MATRIX_SIZE1,
            pad_edges=True, stride_length_px=1
        )
        self.assertTrue(numpy.allclose(
            this_target_matrix, TARGET_MATRIX_SMOOTHED1, atol=TOLERANCE
        ))

        this_prediction_matrix = standalone_utils.do_2d_convolution(
            feature_matrix=PREDICTION_MATRIX, kernel_matrix=WEIGHT_MATRIX_SIZE1,
            pad_edges=True, stride_length_px=1
        )
        self.assertTrue(numpy.allclose(
            this_prediction_matrix, PREDICTION_MATRIX_SMOOTHED1, atol=TOLERANCE
        ))

    def test_do_2d_convolution_size2(self):
        """Ensures correct output from do_2d_convolution.

        In this case, half-window size is 2 pixels.
        """

        this_target_matrix = standalone_utils.do_2d_convolution(
            feature_matrix=TARGET_MATRIX, kernel_matrix=WEIGHT_MATRIX_SIZE2,
            pad_edges=True, stride_length_px=1
        )
        self.assertTrue(numpy.allclose(
            this_target_matrix, TARGET_MATRIX_SMOOTHED2, atol=TOLERANCE
        ))

        this_prediction_matrix = standalone_utils.do_2d_convolution(
            feature_matrix=PREDICTION_MATRIX, kernel_matrix=WEIGHT_MATRIX_SIZE2,
            pad_edges=True, stride_length_px=1
        )
        self.assertTrue(numpy.allclose(
            this_prediction_matrix, PREDICTION_MATRIX_SMOOTHED2, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
