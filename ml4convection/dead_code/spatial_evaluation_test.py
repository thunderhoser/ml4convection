"""Unit tests for spatial_evaluation.py."""

import unittest
import numpy
from ml4convection.utils import spatial_evaluation
from ml4convection.machine_learning import custom_losses_test

TOLERANCE = 1e-6

# The following constants are used to test _get_fss_components.
TARGET_MATRIX = numpy.expand_dims(
    custom_losses_test.TARGET_MATRIX, axis=0
)
TARGET_MATRIX = numpy.repeat(TARGET_MATRIX, repeats=3, axis=0)

PREDICTION_MATRIX = numpy.expand_dims(
    custom_losses_test.PREDICTION_MATRIX, axis=0
)
PREDICTION_MATRIX = numpy.repeat(
    PREDICTION_MATRIX, repeats=3, axis=0
).astype(float)

ACTUAL_SSE_SIZE0 = 96.
REFERENCE_SSE_SIZE0 = 96.
NUM_VALUES = 768

TARGET_MATRIX_SIZE1 = numpy.expand_dims(
    custom_losses_test.TARGET_MATRIX_SMOOTHED1, axis=0
)
TARGET_MATRIX_SIZE1 = numpy.repeat(TARGET_MATRIX_SIZE1, repeats=3, axis=0)

PREDICTION_MATRIX_SIZE1 = numpy.expand_dims(
    custom_losses_test.PREDICTION_MATRIX_SMOOTHED1, axis=0
)
PREDICTION_MATRIX_SIZE1 = numpy.repeat(
    PREDICTION_MATRIX_SIZE1, repeats=3, axis=0
)

ACTUAL_SSE_SIZE1 = numpy.sum(
    (TARGET_MATRIX_SIZE1 - PREDICTION_MATRIX_SIZE1) ** 2
)
REFERENCE_SSE_SIZE1 = numpy.sum(
    TARGET_MATRIX_SIZE1 ** 2 + PREDICTION_MATRIX_SIZE1 ** 2
)

TARGET_MATRIX_SIZE2 = numpy.expand_dims(
    custom_losses_test.TARGET_MATRIX_SMOOTHED2, axis=0
)
TARGET_MATRIX_SIZE2 = numpy.repeat(TARGET_MATRIX_SIZE2, repeats=3, axis=0)

PREDICTION_MATRIX_SIZE2 = numpy.expand_dims(
    custom_losses_test.PREDICTION_MATRIX_SMOOTHED2, axis=0
)
PREDICTION_MATRIX_SIZE2 = numpy.repeat(
    PREDICTION_MATRIX_SIZE2, repeats=3, axis=0
)

ACTUAL_SSE_SIZE2 = numpy.sum(
    (TARGET_MATRIX_SIZE2 - PREDICTION_MATRIX_SIZE2) ** 2
)
REFERENCE_SSE_SIZE2 = numpy.sum(
    TARGET_MATRIX_SIZE2 ** 2 + PREDICTION_MATRIX_SIZE2 ** 2
)

# The following constants are used to test _get_fss_from_components.
FRACTIONS_SKILL_SCORE_SIZE0 = custom_losses_test.FRACTIONS_SKILL_SCORE_SIZE0
FRACTIONS_SKILL_SCORE_SIZE1 = custom_losses_test.FRACTIONS_SKILL_SCORE_SIZE1
FRACTIONS_SKILL_SCORE_SIZE2 = custom_losses_test.FRACTIONS_SKILL_SCORE_SIZE2

FSS_DICT_SIZE0 = {
    spatial_evaluation.ACTUAL_SSE_KEY: ACTUAL_SSE_SIZE0,
    spatial_evaluation.REFERENCE_SSE_KEY: REFERENCE_SSE_SIZE0,
    spatial_evaluation.NUM_VALUES_KEY: NUM_VALUES
}
FSS_DICT_SIZE1 = {
    spatial_evaluation.ACTUAL_SSE_KEY: ACTUAL_SSE_SIZE1,
    spatial_evaluation.REFERENCE_SSE_KEY: REFERENCE_SSE_SIZE1,
    spatial_evaluation.NUM_VALUES_KEY: NUM_VALUES
}
FSS_DICT_SIZE2 = {
    spatial_evaluation.ACTUAL_SSE_KEY: ACTUAL_SSE_SIZE2,
    spatial_evaluation.REFERENCE_SSE_KEY: REFERENCE_SSE_SIZE2,
    spatial_evaluation.NUM_VALUES_KEY: NUM_VALUES
}


class SpatialEvaluationTests(unittest.TestCase):
    """Each method is a unit test for spatial_evaluation.py."""

    def test_get_fss_components_size0(self):
        """Ensures correct output from _get_fss_components.

        In this case, half-window size is 0 pixels.
        """

        this_actual_sse, this_reference_sse, this_num_values = (
            spatial_evaluation._get_fss_components(
                target_matrix=TARGET_MATRIX,
                forecast_probability_matrix=PREDICTION_MATRIX,
                half_window_size_px=0, test_mode=True
            )
        )

        self.assertTrue(numpy.isclose(
            this_actual_sse, ACTUAL_SSE_SIZE0, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_reference_sse, REFERENCE_SSE_SIZE0, atol=TOLERANCE
        ))
        self.assertTrue(this_num_values == NUM_VALUES)

    def test_get_fss_components_size1(self):
        """Ensures correct output from _get_fss_components.

        In this case, half-window size is 1 pixel.
        """

        this_actual_sse, this_reference_sse, this_num_values = (
            spatial_evaluation._get_fss_components(
                target_matrix=TARGET_MATRIX,
                forecast_probability_matrix=PREDICTION_MATRIX,
                half_window_size_px=1, test_mode=True
            )
        )

        self.assertTrue(numpy.isclose(
            this_actual_sse, ACTUAL_SSE_SIZE1, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_reference_sse, REFERENCE_SSE_SIZE1, atol=TOLERANCE
        ))
        self.assertTrue(this_num_values == NUM_VALUES)

    def test_get_fss_components_size2(self):
        """Ensures correct output from _get_fss_components.

        In this case, half-window size is 2 pixels.
        """

        this_actual_sse, this_reference_sse, this_num_values = (
            spatial_evaluation._get_fss_components(
                target_matrix=TARGET_MATRIX,
                forecast_probability_matrix=PREDICTION_MATRIX,
                half_window_size_px=2, test_mode=True
            )
        )

        self.assertTrue(numpy.isclose(
            this_actual_sse, ACTUAL_SSE_SIZE2, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            this_reference_sse, REFERENCE_SSE_SIZE2, atol=TOLERANCE
        ))
        self.assertTrue(this_num_values == NUM_VALUES)

    def test_get_fss_from_components_size0(self):
        """Ensures correct output from _get_fss_from_components.

        In this case, half-window size is 0 pixels.
        """

        this_fractions_skill_score = (
            spatial_evaluation._get_fss_from_components(FSS_DICT_SIZE0)
        )
        self.assertTrue(numpy.isclose(
            this_fractions_skill_score, FRACTIONS_SKILL_SCORE_SIZE0,
            atol=TOLERANCE
        ))

    def test_get_fss_from_components_size1(self):
        """Ensures correct output from _get_fss_from_components.

        In this case, half-window size is 1 pixel.
        """

        this_fractions_skill_score = (
            spatial_evaluation._get_fss_from_components(FSS_DICT_SIZE1)
        )
        self.assertTrue(numpy.isclose(
            this_fractions_skill_score, FRACTIONS_SKILL_SCORE_SIZE1,
            atol=TOLERANCE
        ))

    def test_get_fss_from_components_size2(self):
        """Ensures correct output from _get_fss_from_components.

        In this case, half-window size is 2 pixels.
        """

        this_fractions_skill_score = (
            spatial_evaluation._get_fss_from_components(FSS_DICT_SIZE2)
        )
        self.assertTrue(numpy.isclose(
            this_fractions_skill_score, FRACTIONS_SKILL_SCORE_SIZE2,
            atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
