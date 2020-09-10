"""Unit tests for custom_losses.py."""

import unittest
import numpy
import tensorflow
from keras import backend as K
from ml4convection.machine_learning import custom_losses

TOLERANCE = 1e-6

# The following constants are used to test _create_mean_filter.
FIRST_HALF_NUM_ROWS = 0
FIRST_HALF_NUM_COLUMNS = 1
FIRST_NUM_CHANNELS = 3

FIRST_WEIGHT_MATRIX = numpy.full((1, 3), 1. / 3)
FIRST_WEIGHT_MATRIX = numpy.expand_dims(FIRST_WEIGHT_MATRIX, axis=-1)
FIRST_WEIGHT_MATRIX = numpy.expand_dims(FIRST_WEIGHT_MATRIX, axis=-1)
FIRST_WEIGHT_MATRIX = numpy.repeat(FIRST_WEIGHT_MATRIX, repeats=3, axis=-2)
FIRST_WEIGHT_MATRIX = numpy.repeat(FIRST_WEIGHT_MATRIX, repeats=3, axis=-1)

SECOND_HALF_NUM_ROWS = 2
SECOND_HALF_NUM_COLUMNS = 2
SECOND_NUM_CHANNELS = 1

SECOND_WEIGHT_MATRIX = numpy.full((5, 5), 1. / 25)
SECOND_WEIGHT_MATRIX = numpy.expand_dims(SECOND_WEIGHT_MATRIX, axis=-1)
SECOND_WEIGHT_MATRIX = numpy.expand_dims(SECOND_WEIGHT_MATRIX, axis=-1)

# The following constants are used to test fractions_skill_score.
TARGET_MATRIX = numpy.array([
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

PREDICTION_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

TARGET_TENSOR = tensorflow.constant(TARGET_MATRIX, dtype=tensorflow.float32)
TARGET_TENSOR = tensorflow.expand_dims(TARGET_TENSOR, 0)
TARGET_TENSOR = tensorflow.expand_dims(TARGET_TENSOR, -1)

PREDICTION_TENSOR = tensorflow.constant(
    PREDICTION_MATRIX, dtype=tensorflow.float32
)
PREDICTION_TENSOR = tensorflow.expand_dims(PREDICTION_TENSOR, 0)
PREDICTION_TENSOR = tensorflow.expand_dims(PREDICTION_TENSOR, -1)

FRACTIONS_SKILL_SCORE_SIZE0 = 0.
FRACTIONS_SKILL_SCORE_SIZE16 = 1.

TARGET_MATRIX_SMOOTHED1 = numpy.array([
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float) / 9

PREDICTION_MATRIX_SMOOTHED1 = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
], dtype=float) / 9

THIS_ACTUAL_MSE = numpy.mean(
    (TARGET_MATRIX_SMOOTHED1 - PREDICTION_MATRIX_SMOOTHED1) ** 2
)
THIS_REFERENCE_MSE = numpy.mean(
    TARGET_MATRIX_SMOOTHED1 ** 2 + PREDICTION_MATRIX_SMOOTHED1 ** 2
)
FRACTIONS_SKILL_SCORE_SIZE1 = 1. - THIS_ACTUAL_MSE / THIS_REFERENCE_MSE

TARGET_MATRIX_SMOOTHED2 = numpy.array([
    [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 0],
    [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 0],
    [2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 0],
    [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 0],
    [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 0],
    [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 0],
    [2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 0],
    [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 0],
    [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 0],
    [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 0],
    [2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 2, 0],
    [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 0],
    [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 0],
    [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 0],
    [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float) / 25

PREDICTION_MATRIX_SMOOTHED2 = numpy.array([
    [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
    [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
    [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
    [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
    [2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2],
    [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
    [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
    [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
    [2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2],
    [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
    [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
    [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
    [2, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2, 4, 2, 2, 2],
    [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
    [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
    [1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]
], dtype=float) / 25

THIS_ACTUAL_MSE = numpy.mean(
    (TARGET_MATRIX_SMOOTHED2 - PREDICTION_MATRIX_SMOOTHED2) ** 2
)
THIS_REFERENCE_MSE = numpy.mean(
    TARGET_MATRIX_SMOOTHED2 ** 2 + PREDICTION_MATRIX_SMOOTHED2 ** 2
)
FRACTIONS_SKILL_SCORE_SIZE2 = 1. - THIS_ACTUAL_MSE / THIS_REFERENCE_MSE

# The following constants are used to test weighted_xentropy.
TARGET_MATRIX_FOR_XENTROPY = numpy.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 1])
PREDICTION_MATRIX_FOR_XENTROPY = numpy.array([
    0, 1, 0.5, 1, 1, 0.5, 0, 1, 0.5, 0.5
])

TARGET_TENSOR_FOR_XENTROPY = tensorflow.constant(
    TARGET_MATRIX_FOR_XENTROPY, dtype=tensorflow.float32
)
TARGET_TENSOR_FOR_XENTROPY = tensorflow.expand_dims(
    TARGET_TENSOR_FOR_XENTROPY, 0
)
TARGET_TENSOR_FOR_XENTROPY = tensorflow.expand_dims(
    TARGET_TENSOR_FOR_XENTROPY, -1
)

PREDICTION_TENSOR_FOR_XENTROPY = tensorflow.constant(
    PREDICTION_MATRIX_FOR_XENTROPY, dtype=tensorflow.float32
)
PREDICTION_TENSOR_FOR_XENTROPY = tensorflow.expand_dims(
    PREDICTION_TENSOR_FOR_XENTROPY, 0
)
PREDICTION_TENSOR_FOR_XENTROPY = tensorflow.expand_dims(
    PREDICTION_TENSOR_FOR_XENTROPY, -1
)

FIRST_CLASS_WEIGHTS = numpy.array([1, 1], dtype=float)
WORST_SINGLE_XENTROPY = -numpy.log2(1e-6)
FIRST_XENTROPY = (3 * WORST_SINGLE_XENTROPY + 4) / 10

SECOND_CLASS_WEIGHTS = numpy.array([1, 100], dtype=float)
SECOND_XENTROPY = (3 * WORST_SINGLE_XENTROPY + 301) / 10


class CustomLossesTests(unittest.TestCase):
    """Each method is a unit test for custom_losses.py."""

    def test_create_mean_filter_first(self):
        """Ensures correct output from _create_mean_filter.

        In this case, using first set of inputs.
        """

        this_weight_matrix = custom_losses._create_mean_filter(
            half_num_rows=FIRST_HALF_NUM_ROWS,
            half_num_columns=FIRST_HALF_NUM_COLUMNS,
            num_channels=FIRST_NUM_CHANNELS
        )

        self.assertTrue(numpy.allclose(
            this_weight_matrix, FIRST_WEIGHT_MATRIX, atol=TOLERANCE
        ))

    def test_create_mean_filter_second(self):
        """Ensures correct output from _create_mean_filter.

        In this case, using second set of inputs.
        """

        this_weight_matrix = custom_losses._create_mean_filter(
            half_num_rows=SECOND_HALF_NUM_ROWS,
            half_num_columns=SECOND_HALF_NUM_COLUMNS,
            num_channels=SECOND_NUM_CHANNELS
        )

        self.assertTrue(numpy.allclose(
            this_weight_matrix, SECOND_WEIGHT_MATRIX, atol=TOLERANCE
        ))

    def test_fractions_skill_score_size0(self):
        """Ensures correct output from fractions_skill_score.

        In this case, half-window size is 0 grid cells (window is 1 x 1).
        """

        this_loss_function = custom_losses.fractions_skill_score(
            half_window_size_px=0, use_as_loss_function=False, test_mode=True
        )
        this_fss = K.eval(
            this_loss_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_fss, FRACTIONS_SKILL_SCORE_SIZE0, atol=TOLERANCE
        ))

    def test_fractions_skill_score_size1(self):
        """Ensures correct output from fractions_skill_score.

        In this case, half-window size is 1 grid cell (window is 3 x 3).
        """

        this_loss_function = custom_losses.fractions_skill_score(
            half_window_size_px=1, use_as_loss_function=False, test_mode=True
        )
        this_fss = K.eval(
            this_loss_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_fss, FRACTIONS_SKILL_SCORE_SIZE1, atol=TOLERANCE
        ))

    def test_fractions_skill_score_size2(self):
        """Ensures correct output from fractions_skill_score.

        In this case, half-window size is 2 grid cells (window is 5 x 5).
        """

        this_loss_function = custom_losses.fractions_skill_score(
            half_window_size_px=2, use_as_loss_function=False, test_mode=True
        )
        this_fss = K.eval(
            this_loss_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_fss, FRACTIONS_SKILL_SCORE_SIZE2, atol=TOLERANCE
        ))

    def test_fractions_skill_score_size16(self):
        """Ensures correct output from fractions_skill_score.

        In this case, half-window size is 16 grid cells (window is 33 x 33).
        """

        this_loss_function = custom_losses.fractions_skill_score(
            half_window_size_px=16, use_as_loss_function=False, test_mode=True
        )
        this_fss = K.eval(
            this_loss_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_fss, FRACTIONS_SKILL_SCORE_SIZE16, atol=TOLERANCE
        ))

    def test_weighted_xentropy_first(self):
        """Ensures correct output from weighted_xentropy.

        In this case, using first set of inputs.
        """

        this_loss_function = custom_losses.weighted_xentropy(
            FIRST_CLASS_WEIGHTS
        )
        this_xentropy = K.eval(this_loss_function(
            TARGET_TENSOR_FOR_XENTROPY, PREDICTION_TENSOR_FOR_XENTROPY
        ))

        self.assertTrue(numpy.isclose(
            this_xentropy, FIRST_XENTROPY, atol=TOLERANCE
        ))

    def test_weighted_xentropy_second(self):
        """Ensures correct output from weighted_xentropy.

        In this case, using second set of inputs.
        """

        this_loss_function = custom_losses.weighted_xentropy(
            SECOND_CLASS_WEIGHTS
        )
        this_xentropy = K.eval(this_loss_function(
            TARGET_TENSOR_FOR_XENTROPY, PREDICTION_TENSOR_FOR_XENTROPY
        ))

        self.assertTrue(numpy.isclose(
            this_xentropy, SECOND_XENTROPY, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
