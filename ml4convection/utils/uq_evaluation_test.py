"""Unit tests for uq_evaluation.py."""

import unittest
import numpy
from scipy.integrate import simps
from ml4convection.io import prediction_io
from ml4convection.utils import uq_evaluation

TOLERANCE = 1e-6

# The following constants are used to test _get_crps_monte_carlo.
THIS_PROB_MATRIX = numpy.full((8, 205, 205, 100), numpy.nan)
THIS_PROB_MATRIX[0, ...] = 0.
THIS_PROB_MATRIX[1, ...] = 1.
THIS_PROB_MATRIX[2, ...] = 0.5
THIS_PROB_MATRIX[3, ...] = numpy.linspace(0, 1, num=100, dtype=float)
THIS_PROB_MATRIX[4, ...] = 0.
THIS_PROB_MATRIX[5, ...] = 1.
THIS_PROB_MATRIX[6, ...] = 0.5
THIS_PROB_MATRIX[7, ...] = numpy.linspace(0, 1, num=100, dtype=float)

THIS_TARGET_MATRIX = numpy.full((8, 205, 205), 1, dtype=int)
THIS_TARGET_MATRIX[4:8, ...] = 0

PREDICTION_DICT_MONTE_CARLO = {
    prediction_io.PROBABILITY_MATRIX_KEY: THIS_PROB_MATRIX + 0.,
    prediction_io.TARGET_MATRIX_KEY: THIS_TARGET_MATRIX + 0
}

PROB_LEVELS_TO_INTEG_NOT_QR = uq_evaluation.PROB_LEVELS_TO_INTEG_NOT_QR

FIRST_CDF_VALUES = numpy.full(101, 1.)
FIRST_CRPS_ADDEND = simps(
    y=FIRST_CDF_VALUES ** 2, x=PROB_LEVELS_TO_INTEG_NOT_QR, axis=-1
)

SECOND_CDF_VALUES = numpy.full(101, 0.)
SECOND_CDF_VALUES[-1] = 1.
SECOND_CRPS_ADDEND = simps(
    y=SECOND_CDF_VALUES ** 2, x=PROB_LEVELS_TO_INTEG_NOT_QR, axis=-1
)

THIRD_CDF_VALUES = numpy.full(101, 0.)
THIRD_CDF_VALUES[50:] = 1.
THIRD_CRPS_ADDEND = simps(
    y=THIRD_CDF_VALUES ** 2, x=PROB_LEVELS_TO_INTEG_NOT_QR, axis=-1
)

FOURTH_CDF_VALUES = numpy.concatenate((
    numpy.full(1, 0.01), numpy.linspace(0.01, 1, num=100, dtype=float)
))
FOURTH_CRPS_ADDEND = simps(
    y=FOURTH_CDF_VALUES ** 2, x=PROB_LEVELS_TO_INTEG_NOT_QR, axis=-1
)

FIFTH_CDF_VALUES = FIRST_CDF_VALUES
FIFTH_CRPS_ADDEND = simps(
    y=(FIFTH_CDF_VALUES - 1) ** 2, x=PROB_LEVELS_TO_INTEG_NOT_QR, axis=-1
)

SIXTH_CDF_VALUES = SECOND_CDF_VALUES
SIXTH_CRPS_ADDEND = simps(
    y=(SIXTH_CDF_VALUES - 1) ** 2, x=PROB_LEVELS_TO_INTEG_NOT_QR, axis=-1
)

SEVENTH_CDF_VALUES = THIRD_CDF_VALUES
SEVENTH_CRPS_ADDEND = simps(
    y=(SEVENTH_CDF_VALUES - 1) ** 2, x=PROB_LEVELS_TO_INTEG_NOT_QR, axis=-1
)

EIGHTH_CDF_VALUES = FOURTH_CDF_VALUES
EIGHTH_CRPS_ADDEND = simps(
    y=(EIGHTH_CDF_VALUES - 1) ** 2, x=PROB_LEVELS_TO_INTEG_NOT_QR, axis=-1
)

CRPS_MONTE_CARLO = numpy.mean(numpy.array([
    FIRST_CRPS_ADDEND, SECOND_CRPS_ADDEND, THIRD_CRPS_ADDEND,
    FOURTH_CRPS_ADDEND, FIFTH_CRPS_ADDEND, SIXTH_CRPS_ADDEND,
    SEVENTH_CRPS_ADDEND, EIGHTH_CRPS_ADDEND,
]))

# The following constants are used to test _get_crps_quantile_regression.
THIS_PROB_MATRIX = numpy.full((8, 205, 205, 101), numpy.nan)
THIS_PROB_MATRIX[0, ...] = numpy.linspace(0, 1e-12, num=101, dtype=float)
THIS_PROB_MATRIX[1, ...] = numpy.linspace(1. - 1e-12, 1, num=101, dtype=float)
THIS_PROB_MATRIX[2, ...] = numpy.linspace(
    0.5, 0.5 + 1e-12, num=101, dtype=float
)
THIS_PROB_MATRIX[3, ...] = numpy.linspace(0, 1, num=101, dtype=float)
THIS_PROB_MATRIX[4, ...] = numpy.linspace(0, 1e-12, num=101, dtype=float)
THIS_PROB_MATRIX[5, ...] = numpy.linspace(1. - 1e-12, 1, num=101, dtype=float)
THIS_PROB_MATRIX[6, ...] = numpy.linspace(
    0.5, 0.5 + 1e-12, num=101, dtype=float
)
THIS_PROB_MATRIX[7, ...] = numpy.linspace(0, 1, num=101, dtype=float)

NEW_PROB_MATRIX = numpy.full((8, 205, 205, 1), numpy.nan)
THIS_PROB_MATRIX = numpy.concatenate(
    (NEW_PROB_MATRIX, THIS_PROB_MATRIX), axis=-1
)

THIS_TARGET_MATRIX = numpy.full((8, 205, 205), 1, dtype=int)
THIS_TARGET_MATRIX[4:8, ...] = 0

PREDICTION_DICT_QUANTILE_REGRESSION = {
    prediction_io.PROBABILITY_MATRIX_KEY: THIS_PROB_MATRIX + 0.,
    prediction_io.TARGET_MATRIX_KEY: THIS_TARGET_MATRIX + 0,
    prediction_io.QUANTILE_LEVELS_KEY:
        numpy.linspace(0, 1, num=101, dtype=float)
}

PROB_LEVELS_TO_INTEG_FOR_QR = uq_evaluation.PROB_LEVELS_TO_INTEG_FOR_QR

FIRST_CDF_VALUES = numpy.full(41, 1.)
FIRST_CDF_VALUES[0] = 0.
FIRST_CRPS_ADDEND = simps(
    y=FIRST_CDF_VALUES ** 2, x=PROB_LEVELS_TO_INTEG_FOR_QR, axis=-1
)

SECOND_CDF_VALUES = numpy.full(41, 0.)
SECOND_CDF_VALUES[-1] = 1.
SECOND_CRPS_ADDEND = simps(
    y=SECOND_CDF_VALUES ** 2, x=PROB_LEVELS_TO_INTEG_FOR_QR, axis=-1
)

THIRD_CDF_VALUES = numpy.full(41, 0.)
THIRD_CDF_VALUES[21:] = 1.
THIRD_CRPS_ADDEND = simps(
    y=THIRD_CDF_VALUES ** 2, x=PROB_LEVELS_TO_INTEG_FOR_QR, axis=-1
)

FOURTH_CDF_VALUES = numpy.linspace(0, 1, num=41, dtype=float)
FOURTH_CRPS_ADDEND = simps(
    y=FOURTH_CDF_VALUES ** 2, x=PROB_LEVELS_TO_INTEG_FOR_QR, axis=-1
)

FIFTH_CDF_VALUES = FIRST_CDF_VALUES
FIFTH_CRPS_ADDEND = simps(
    y=(FIFTH_CDF_VALUES - 1) ** 2, x=PROB_LEVELS_TO_INTEG_FOR_QR, axis=-1
)

SIXTH_CDF_VALUES = SECOND_CDF_VALUES
SIXTH_CRPS_ADDEND = simps(
    y=(SIXTH_CDF_VALUES - 1) ** 2, x=PROB_LEVELS_TO_INTEG_FOR_QR, axis=-1
)

SEVENTH_CDF_VALUES = THIRD_CDF_VALUES
SEVENTH_CRPS_ADDEND = simps(
    y=(SEVENTH_CDF_VALUES - 1) ** 2, x=PROB_LEVELS_TO_INTEG_FOR_QR, axis=-1
)

EIGHTH_CDF_VALUES = FOURTH_CDF_VALUES
EIGHTH_CRPS_ADDEND = simps(
    y=(EIGHTH_CDF_VALUES - 1) ** 2, x=PROB_LEVELS_TO_INTEG_FOR_QR, axis=-1
)

CRPS_QUANTILE_REGRESSION = numpy.mean(numpy.array([
    FIRST_CRPS_ADDEND, SECOND_CRPS_ADDEND, THIRD_CRPS_ADDEND,
    FOURTH_CRPS_ADDEND, FIFTH_CRPS_ADDEND, SIXTH_CRPS_ADDEND,
    SEVENTH_CRPS_ADDEND, EIGHTH_CRPS_ADDEND,
]))

# The following constants are used to test many different methods.
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

SQUARED_ERRORS = numpy.array([
    0.500, 0.400, 0.700, 0.400, 0.200, 0.000,
    0.460, 0.598, 0.660, 0.500, 0.270, 0.020
])

SQUARED_ERRORS = SQUARED_ERRORS ** 2

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

RMSE_VALUES = numpy.array([
    numpy.mean(SQUARED_ERRORS[[0, 5, 6, 11]]),
    numpy.mean(SQUARED_ERRORS[[2, 8]]),
    SQUARED_ERRORS[4],
    numpy.mean(SQUARED_ERRORS[[3, 9, 10]]),
    numpy.nan,
    numpy.mean(SQUARED_ERRORS[[1, 7]]),
    numpy.nan
])

RMSE_VALUES = numpy.sqrt(RMSE_VALUES)

# The following constants are used to test get_stdev_uncertainty_function.
PREDICTION_STDEV_MATRIX = numpy.reshape(PREDICTION_STDEVS, (2, 2, 3))

# The following constants are used to test get_fss_error_function.
MEAN_PROB_MATRIX_EXAMPLE1 = numpy.array([
    [0.5, 0.4, 0.3],
    [0.6, 0.2, 0.0]
])
MEAN_PROB_MATRIX_EXAMPLE2 = numpy.array([
    [0.540, 0.402, 0.340],
    [0.500, 0.270, 0.020]
])
MEAN_PROB_MATRIX_SPATIAL = numpy.stack(
    (MEAN_PROB_MATRIX_EXAMPLE1, MEAN_PROB_MATRIX_EXAMPLE2), axis=0
)

SMOOTHED_PROB_MATRIX_EXAMPLE1 = (1. / 9) * numpy.array([
    [1.7, 2.0, 0.9],
    [1.7, 2.0, 0.9]
])
SMOOTHED_PROB_MATRIX_EXAMPLE2 = numpy.array([
    [0.19022222, 0.23022222, 0.11466667],
    [0.19022222, 0.23022222, 0.11466667]
])
SMOOTHED_MEAN_PROB_MATRIX_SPATIAL = numpy.stack(
    (SMOOTHED_PROB_MATRIX_EXAMPLE1, SMOOTHED_PROB_MATRIX_EXAMPLE2), axis=0
)

TARGET_MATRIX_EXAMPLE1 = numpy.array([
    [0, 0, 1],
    [1, 0, 0]
], dtype=int)

TARGET_MATRIX_EXAMPLE2 = numpy.array([
    [1, 1, 1],
    [1, 0, 0]
], dtype=int)

TARGET_MATRIX_SPATIAL = numpy.stack(
    (TARGET_MATRIX_EXAMPLE1, TARGET_MATRIX_EXAMPLE2), axis=0
)

SMOOTHED_TARGET_MATRIX_EXAMPLE1 = (1. / 9) * numpy.array([
    [1, 2, 1],
    [1, 2, 1]
], dtype=float)

SMOOTHED_TARGET_MATRIX_EXAMPLE2 = (1. / 9) * numpy.array([
    [3, 4, 2],
    [3, 4, 2]
], dtype=float)

SMOOTHED_TARGET_MATRIX_SPATIAL = numpy.stack(
    (SMOOTHED_TARGET_MATRIX_EXAMPLE1, SMOOTHED_TARGET_MATRIX_EXAMPLE2), axis=0
)

THIS_ACTUAL_SSE = numpy.sum(
    (TARGET_MATRIX_SPATIAL - MEAN_PROB_MATRIX_SPATIAL) ** 2
)
THIS_REFERENCE_SSE = numpy.sum(
    TARGET_MATRIX_SPATIAL ** 2 + MEAN_PROB_MATRIX_SPATIAL ** 2
)
FSS_VALUE_HW0 = 1. - THIS_ACTUAL_SSE / THIS_REFERENCE_SSE

THIS_ACTUAL_SSE = numpy.sum(
    (SMOOTHED_TARGET_MATRIX_SPATIAL - SMOOTHED_MEAN_PROB_MATRIX_SPATIAL) ** 2
)
THIS_REFERENCE_SSE = numpy.sum(
    SMOOTHED_TARGET_MATRIX_SPATIAL ** 2 + SMOOTHED_MEAN_PROB_MATRIX_SPATIAL ** 2
)
FSS_VALUE_HW1 = 1. - THIS_ACTUAL_SSE / THIS_REFERENCE_SSE

# The following constants are used to test run_discard_test.
DISCARD_FRACTIONS_SANS_ZERO = numpy.linspace(1. / 6, 1, num=6, dtype=float)[:-1]
DISCARD_FRACTIONS_WITH_ZERO = numpy.linspace(0, 1, num=7, dtype=float)[:-1]

THIS_MEAN_PROB_MATRIX_SPATIAL = MEAN_PROB_MATRIX_SPATIAL + 0.
THIS_MEAN_PROB_MATRIX_SPATIAL[0, 0, 1] = numpy.nan
THIS_MEAN_PROB_MATRIX_SPATIAL[1, 0, 1] = numpy.nan
THIS_ACTUAL_SSE = numpy.nansum(
    (TARGET_MATRIX_SPATIAL - THIS_MEAN_PROB_MATRIX_SPATIAL) ** 2
)
THIS_REFERENCE_SSE = numpy.nansum(
    TARGET_MATRIX_SPATIAL ** 2 + THIS_MEAN_PROB_MATRIX_SPATIAL ** 2
)
FSS_VALUE_HW0_DISCARD2 = 1. - THIS_ACTUAL_SSE / THIS_REFERENCE_SSE

THIS_MEAN_PROB_MATRIX_SPATIAL[0, 1, 0] = numpy.nan
THIS_MEAN_PROB_MATRIX_SPATIAL[1, 1, 0] = numpy.nan
THIS_ACTUAL_SSE = numpy.nansum(
    (TARGET_MATRIX_SPATIAL - THIS_MEAN_PROB_MATRIX_SPATIAL) ** 2
)
THIS_REFERENCE_SSE = numpy.nansum(
    TARGET_MATRIX_SPATIAL ** 2 + THIS_MEAN_PROB_MATRIX_SPATIAL ** 2
)
FSS_VALUE_HW0_DISCARD4 = 1. - THIS_ACTUAL_SSE / THIS_REFERENCE_SSE

THIS_MEAN_PROB_MATRIX_SPATIAL[0, 1, 1] = numpy.nan
THIS_MEAN_PROB_MATRIX_SPATIAL[1, 1, 1] = numpy.nan
THIS_ACTUAL_SSE = numpy.nansum(
    (TARGET_MATRIX_SPATIAL - THIS_MEAN_PROB_MATRIX_SPATIAL) ** 2
)
THIS_REFERENCE_SSE = numpy.nansum(
    TARGET_MATRIX_SPATIAL ** 2 + THIS_MEAN_PROB_MATRIX_SPATIAL ** 2
)
FSS_VALUE_HW0_DISCARD6 = 1. - THIS_ACTUAL_SSE / THIS_REFERENCE_SSE

THIS_MEAN_PROB_MATRIX_SPATIAL[0, 0, 2] = numpy.nan
THIS_MEAN_PROB_MATRIX_SPATIAL[1, 0, 2] = numpy.nan
THIS_ACTUAL_SSE = numpy.nansum(
    (TARGET_MATRIX_SPATIAL - THIS_MEAN_PROB_MATRIX_SPATIAL) ** 2
)
THIS_REFERENCE_SSE = numpy.nansum(
    TARGET_MATRIX_SPATIAL ** 2 + THIS_MEAN_PROB_MATRIX_SPATIAL ** 2
)
FSS_VALUE_HW0_DISCARD8 = 1. - THIS_ACTUAL_SSE / THIS_REFERENCE_SSE

THIS_MEAN_PROB_MATRIX_SPATIAL[1, 0, 0] = numpy.nan
THIS_MEAN_PROB_MATRIX_SPATIAL[1, 1, 2] = numpy.nan
THIS_ACTUAL_SSE = numpy.nansum(
    (TARGET_MATRIX_SPATIAL - THIS_MEAN_PROB_MATRIX_SPATIAL) ** 2
)
THIS_REFERENCE_SSE = numpy.nansum(
    TARGET_MATRIX_SPATIAL ** 2 + THIS_MEAN_PROB_MATRIX_SPATIAL ** 2
)
FSS_VALUE_HW0_DISCARD10 = 1. - THIS_ACTUAL_SSE / THIS_REFERENCE_SSE

ERROR_BY_DISCARD_FRACTION = numpy.array([
    FSS_VALUE_HW0, FSS_VALUE_HW0_DISCARD2, FSS_VALUE_HW0_DISCARD4,
    FSS_VALUE_HW0_DISCARD6, FSS_VALUE_HW0_DISCARD8, FSS_VALUE_HW0_DISCARD10
])

# The following constants are used to test _get_squared_errors.
SQERR_MATRIX_SPATIAL = (MEAN_PROB_MATRIX_SPATIAL - TARGET_MATRIX_SPATIAL) ** 2
SMOOTHED_SQERR_MATRIX_SPATIAL = (
    (SMOOTHED_MEAN_PROB_MATRIX_SPATIAL - SMOOTHED_TARGET_MATRIX_SPATIAL) ** 2
)


class UqEvaluationTests(unittest.TestCase):
    """Each method is a unit test for uq_evaluation.py."""

    def test_get_crps_monte_carlo(self):
        """Ensures correct output from _get_crps_monte_carlo."""

        this_crps_value = uq_evaluation._get_crps_monte_carlo(
            PREDICTION_DICT_MONTE_CARLO
        )

        self.assertTrue(numpy.isclose(
            this_crps_value, CRPS_MONTE_CARLO, atol=TOLERANCE
        ))

    def test_get_crps_quantile_regression(self):
        """Ensures correct output from _get_crps_quantile_regression."""

        this_crps_value = uq_evaluation._get_crps_quantile_regression(
            PREDICTION_DICT_QUANTILE_REGRESSION
        )

        self.assertTrue(numpy.isclose(
            this_crps_value, CRPS_QUANTILE_REGRESSION, atol=TOLERANCE
        ))

    def test_get_stdev_uncertainty_function(self):
        """Ensures correct output from get_stdev_uncertainty_function."""

        this_function = uq_evaluation.get_stdev_uncertainty_function()
        this_stdev_matrix = this_function(PREDICTION_DICT)

        self.assertTrue(numpy.allclose(
            this_stdev_matrix, PREDICTION_STDEV_MATRIX, atol=TOLERANCE
        ))

    def test_get_fss_error_function_hw0(self):
        """Ensures correct output from get_fss_error_function.

        In this case, neighbourhood half-width is 0 pixels.
        """

        this_function = uq_evaluation.get_fss_error_function(
            half_window_size_px=0, use_median=False, use_quantiles=False
        )
        eroded_eval_mask_matrix = numpy.full((2, 2, 3), 1, dtype=bool)
        this_fss_value = this_function(PREDICTION_DICT, eroded_eval_mask_matrix)

        self.assertTrue(numpy.isclose(
            this_fss_value, FSS_VALUE_HW0, atol=TOLERANCE
        ))

    def test_get_fss_error_function_hw1(self):
        """Ensures correct output from get_fss_error_function.

        In this case, neighbourhood half-width is 1 pixel.
        """

        this_function = uq_evaluation.get_fss_error_function(
            half_window_size_px=1, use_median=False, use_quantiles=False
        )
        eroded_eval_mask_matrix = numpy.full((2, 2, 3), 1, dtype=bool)
        this_fss_value = this_function(PREDICTION_DICT, eroded_eval_mask_matrix)

        self.assertTrue(numpy.isclose(
            this_fss_value, FSS_VALUE_HW1, atol=TOLERANCE
        ))

    def test_run_discard_test_hw0(self):
        """Ensures correct output from run_discard_test.

        In this case, neighbourhood half-width is 0 pixels.
        """

        eroded_eval_mask_matrix = numpy.full((2, 2, 3), 1, dtype=bool)
        error_function = uq_evaluation.get_fss_error_function(
            half_window_size_px=0, use_median=False, use_quantiles=False
        )
        uncertainty_function = uq_evaluation.get_stdev_uncertainty_function()

        result_dict = uq_evaluation.run_discard_test(
            prediction_dict=PREDICTION_DICT,
            discard_fractions=DISCARD_FRACTIONS_SANS_ZERO,
            eroded_eval_mask_matrix=eroded_eval_mask_matrix,
            error_function=error_function,
            uncertainty_function=uncertainty_function, use_median=False,
            is_error_pos_oriented=True
        )

        self.assertTrue(numpy.allclose(
            result_dict[uq_evaluation.DISCARD_FRACTIONS_KEY],
            DISCARD_FRACTIONS_WITH_ZERO, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            result_dict[uq_evaluation.ERROR_VALUES_KEY],
            ERROR_BY_DISCARD_FRACTION, atol=TOLERANCE
        ))

    def test_get_squared_errors_hw0(self):
        """Ensures correct output from _get_squared_errors.

        In this case, neighbourhood half-width is 0 pixels.
        """

        this_sqerr_matrix = uq_evaluation._get_squared_errors(
            prediction_dict=PREDICTION_DICT, half_window_size_px=0,
            use_median=False, use_quantiles=False
        )
        self.assertTrue(numpy.allclose(
            this_sqerr_matrix, SQERR_MATRIX_SPATIAL, atol=TOLERANCE
        ))

    def test_get_squared_errors_hw1(self):
        """Ensures correct output from _get_squared_errors.

        In this case, neighbourhood half-width is 1 pixel.
        """

        this_sqerr_matrix = uq_evaluation._get_squared_errors(
            prediction_dict=PREDICTION_DICT, half_window_size_px=1,
            use_median=False, use_quantiles=False
        )
        self.assertTrue(numpy.allclose(
            this_sqerr_matrix, SMOOTHED_SQERR_MATRIX_SPATIAL, atol=TOLERANCE
        ))

    def test_get_spread_vs_skill_hw0(self):
        """Ensures correct output from get_spread_vs_skill.

        In this case, neighbourhood half-width is 0 pixels.
        """

        eval_mask_matrix = numpy.full((2, 3), 1, dtype=bool)

        result_dict = uq_evaluation.get_spread_vs_skill(
            prediction_dict=PREDICTION_DICT,
            bin_edge_prediction_stdevs=BIN_EDGE_PREDICTION_STDEVS,
            half_window_size_px=0, eval_mask_matrix=eval_mask_matrix,
            use_median=False
        )

        self.assertTrue(numpy.allclose(
            result_dict[uq_evaluation.MEAN_PREDICTION_STDEVS_KEY],
            MEAN_PREDICTION_STDEVS, atol=TOLERANCE, equal_nan=True
        ))
        self.assertTrue(numpy.allclose(
            result_dict[uq_evaluation.RMSE_VALUES_KEY], RMSE_VALUES,
            atol=TOLERANCE, equal_nan=True
        ))


if __name__ == '__main__':
    unittest.main()
