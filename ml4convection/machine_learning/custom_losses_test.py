"""Unit tests for custom_losses.py."""

import unittest
import numpy
import tensorflow
from keras import backend as K
from ml4convection.machine_learning import custom_losses

TOLERANCE = 1e-6

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

MASK_MATRIX = numpy.full(TARGET_MATRIX.shape, True, dtype=bool)

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

MASK_MATRIX_FOR_ENTROPY = numpy.full(
    TARGET_MATRIX_FOR_XENTROPY.shape, True, dtype=bool
)
MASK_MATRIX_FOR_ENTROPY = numpy.expand_dims(MASK_MATRIX_FOR_ENTROPY, axis=0)

TARGET_TENSOR_FOR_XENTROPY = tensorflow.constant(
    TARGET_MATRIX_FOR_XENTROPY, dtype=tensorflow.float32
)
TARGET_TENSOR_FOR_XENTROPY = tensorflow.expand_dims(
    TARGET_TENSOR_FOR_XENTROPY, 0
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

    def test_fractions_skill_score_size0(self):
        """Ensures correct output from fractions_skill_score.

        In this case, half-window size is 0 grid cells (window is 1 x 1).
        """

        this_loss_function = custom_losses.fractions_skill_score(
            half_window_size_px=0, mask_matrix=MASK_MATRIX,
            use_as_loss_function=False, test_mode=True
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
            half_window_size_px=1, mask_matrix=MASK_MATRIX,
            use_as_loss_function=False, test_mode=True
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
            half_window_size_px=2, mask_matrix=MASK_MATRIX,
            use_as_loss_function=False, test_mode=True
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
            half_window_size_px=16, mask_matrix=MASK_MATRIX,
            use_as_loss_function=False, test_mode=True
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

    def test_cross_entropy_size0(self):
        """Ensures correct output from cross_entropy.

        In this case, half-window size is 0 grid cells (window is 1 x 1).
        """

        this_loss_function = custom_losses.cross_entropy(
            mask_matrix=MASK_MATRIX_FOR_ENTROPY
        )
        this_xentropy = K.eval(this_loss_function(
            TARGET_TENSOR_FOR_XENTROPY, PREDICTION_TENSOR_FOR_XENTROPY
        ))

        self.assertTrue(numpy.isclose(
            this_xentropy, FIRST_XENTROPY, atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
