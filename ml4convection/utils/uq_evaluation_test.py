"""Unit tests for uq_evaluation.py."""

import unittest
import numpy
from ml4convection.io import prediction_io
from ml4convection.utils import uq_evaluation

TOLERANCE = 1e-6

FORECAST_PROB_MATRIX = numpy.array([
    [0.50, 0.50, 0.50, 0.50, 0.50],
    [0.00, 1.00, 0.00, 1.00, 0.00],
    [0.10, 0.20, 0.30, 0.40, 0.50],
    [0.20, 0.40, 0.60, 0.80, 1.00],
    [0.00, 0.50, 0.00, 0.50, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00],
    [0.50, 0.52, 0.54, 0.56, 0.58],
    [0.01, 0.99, 0.01, 0.99, 0.01],
    [0.10, 0.22, 0.34, 0.46, 0.58],
    [0.00, 0.25, 0.50, 0.75, 1.00],
    [0.05, 0.60, 0.05, 0.60, 0.05],
    [0.00, 0.01, 0.02, 0.03, 0.04]
])

TARGET_MATRIX = numpy.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0], dtype=int)

PREDICTION_DICT = {
    prediction_io.PROBABILITY_MATRIX_KEY:
        numpy.reshape(FORECAST_PROB_MATRIX, (2, 2, 3, 5)),
    prediction_io.TARGET_MATRIX_KEY: numpy.reshape(TARGET_MATRIX, (2, 2, 3))
}

PREDICTION_STDEVS = numpy.array([
    0.00000000, 0.54772256, 0.15811388, 0.31622777, 0.27386128, 0.00000000,
    0.03162278, 0.53676811, 0.18973666, 0.39528471, 0.30124741, 0.01581139
])

ABSOLUTE_ERRORS = numpy.array([
    0.500, 0.400, 0.700, 0.400, 0.200, 0.000,
    0.460, 0.598, 0.660, 0.500, 0.270, 0.020
])

BIN_EDGE_PREDICTION_STDEVS = numpy.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

MEAN_PREDICTION_STDEVS = numpy.array([
    numpy.mean(PREDICTION_STDEVS[[0, 5, 6, 11]]),
    numpy.mean(PREDICTION_STDEVS[[2, 8]]),
    PREDICTION_STDEVS[4],
    numpy.mean(PREDICTION_STDEVS[[3, 9, 10]]),
    numpy.nan,
    numpy.mean(PREDICTION_STDEVS[[1, 7]]),
    numpy.nan
])

PREDICTION_STANDARD_ERRORS = numpy.array([
    numpy.mean(ABSOLUTE_ERRORS[[0, 5, 6, 11]]),
    numpy.mean(ABSOLUTE_ERRORS[[2, 8]]),
    ABSOLUTE_ERRORS[4],
    numpy.mean(ABSOLUTE_ERRORS[[3, 9, 10]]),
    numpy.nan,
    numpy.mean(ABSOLUTE_ERRORS[[1, 7]]),
    numpy.nan
])


class UqEvaluationTests(unittest.TestCase):
    """Each method is a unit test for uq_evaluation.py."""

    def test_get_spread_vs_skill(self):
        """Ensures correct output from get_spread_vs_skill."""

        these_mean_prediction_stdevs, these_prediction_standard_errors = (
            uq_evaluation.get_spread_vs_skill(
                prediction_dict=PREDICTION_DICT,
                bin_edge_prediction_stdevs=BIN_EDGE_PREDICTION_STDEVS,
                use_mean_to_compute_error=True
            )
        )

        self.assertTrue(numpy.allclose(
            these_mean_prediction_stdevs, MEAN_PREDICTION_STDEVS,
            atol=TOLERANCE, equal_nan=True
        ))
        self.assertTrue(numpy.allclose(
            these_prediction_standard_errors, PREDICTION_STANDARD_ERRORS,
            atol=TOLERANCE, equal_nan=True
        ))


if __name__ == '__main__':
    unittest.main()
