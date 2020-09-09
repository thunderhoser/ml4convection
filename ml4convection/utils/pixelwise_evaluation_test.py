"""Unit tests for pixelwise_evaluation.py."""

import copy
import unittest
import numpy
import xarray
from ml4convection.io import prediction_io
from ml4convection.utils import pixelwise_evaluation as pixelwise_eval

TOLERANCE = 1e-6

# The following constants are used to test _update_basic_scores.
NUM_PROBABILITY_THRESHOLDS = 11
NUM_BINS_FOR_RELIABILITY = 10
NUM_LATITUDES = 6
NUM_LONGITUDES = 12
NUM_GRID_CELLS = NUM_LATITUDES * NUM_LONGITUDES

FORECAST_PROB_MATRIX = numpy.array([
    0.08, 0.05, 0.18, 0.11, 0.04, 0.80, 0.29, 0.27, 0.95, 0.95
])
FORECAST_PROB_MATRIX = numpy.expand_dims(FORECAST_PROB_MATRIX, axis=(1, 2))
FORECAST_PROB_MATRIX = numpy.repeat(
    FORECAST_PROB_MATRIX, repeats=NUM_LATITUDES, axis=1
)
FORECAST_PROB_MATRIX = numpy.repeat(
    FORECAST_PROB_MATRIX, repeats=NUM_LONGITUDES, axis=2
)

TARGET_MATRIX = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
TARGET_MATRIX = numpy.expand_dims(TARGET_MATRIX, axis=(1, 2))
TARGET_MATRIX = numpy.repeat(TARGET_MATRIX, repeats=NUM_LATITUDES, axis=1)
TARGET_MATRIX = numpy.repeat(TARGET_MATRIX, repeats=NUM_LONGITUDES, axis=2)

PREDICTION_DICT = {
    prediction_io.TARGET_MATRIX_KEY: TARGET_MATRIX,
    prediction_io.PROBABILITY_MATRIX_KEY: FORECAST_PROB_MATRIX
}

BASIC_SCORE_TABLE_ORIG = pixelwise_eval.get_basic_scores(
    prediction_file_names=[], event_frequency_in_training=None,
    num_prob_thresholds=NUM_PROBABILITY_THRESHOLDS,
    num_bins_for_reliability=NUM_BINS_FOR_RELIABILITY, test_mode=True
)

BASIC_SCORE_TABLE_NEW = copy.deepcopy(BASIC_SCORE_TABLE_ORIG)
BASIC_SCORE_TABLE_NEW[pixelwise_eval.NUM_TRUE_POSITIVES_KEY].values = (
    NUM_GRID_CELLS *
    numpy.array([5, 5, 5, 3, 3, 3, 3, 3, 3, 2, 0, 0], dtype=int)
)
BASIC_SCORE_TABLE_NEW[pixelwise_eval.NUM_FALSE_POSITIVES_KEY].values = (
    NUM_GRID_CELLS *
    numpy.array([5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int)
)
BASIC_SCORE_TABLE_NEW[pixelwise_eval.NUM_FALSE_NEGATIVES_KEY].values = (
    NUM_GRID_CELLS *
    numpy.array([0, 0, 0, 2, 2, 2, 2, 2, 2, 3, 5, 5], dtype=int)
)
BASIC_SCORE_TABLE_NEW[pixelwise_eval.NUM_TRUE_NEGATIVES_KEY].values = (
    NUM_GRID_CELLS *
    numpy.array([0, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=int)
)
BASIC_SCORE_TABLE_NEW[pixelwise_eval.NUM_EXAMPLES_KEY].values = (
    NUM_GRID_CELLS * numpy.array([3, 2, 2, 0, 0, 0, 0, 0, 1, 2], dtype=int)
)
BASIC_SCORE_TABLE_NEW[pixelwise_eval.EVENT_FREQUENCY_KEY].values = (
    numpy.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 1], dtype=float)
)
BASIC_SCORE_TABLE_NEW[pixelwise_eval.MEAN_FORECAST_PROB_KEY].values = (
    numpy.array([0.17 / 3, 0.145, 0.28, 0, 0, 0, 0, 0, 0.8, 0.95])
)
BASIC_SCORE_TABLE_NEW.attrs[pixelwise_eval.CLIMO_EVENT_FREQ_KEY] = 0.5

# The following constants are used to test get_advanced_scores.
PROBABILITY_THRESHOLDS = (
    BASIC_SCORE_TABLE_ORIG.coords[
        pixelwise_eval.PROBABILITY_THRESHOLD_DIM
    ].values
)

BIN_INDICES = numpy.linspace(
    0, NUM_BINS_FOR_RELIABILITY - 1, num=NUM_BINS_FOR_RELIABILITY, dtype=int
)

THIS_METADATA_DICT = {
    pixelwise_eval.PROBABILITY_THRESHOLD_DIM: PROBABILITY_THRESHOLDS,
    pixelwise_eval.RELIABILITY_BIN_DIM: BIN_INDICES
}

THESE_DIM = (pixelwise_eval.PROBABILITY_THRESHOLD_DIM,)
THIS_ARRAY = numpy.full(len(PROBABILITY_THRESHOLDS), numpy.nan)
THIS_DATA_DICT = {
    pixelwise_eval.POD_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    pixelwise_eval.POFD_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    pixelwise_eval.SUCCESS_RATIO_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    pixelwise_eval.FREQUENCY_BIAS_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    pixelwise_eval.CSI_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    pixelwise_eval.ACCURACY_KEY: (THESE_DIM, THIS_ARRAY + 0.),
    pixelwise_eval.HEIDKE_SCORE_KEY: (THESE_DIM, THIS_ARRAY + 0.)
}

ADVANCED_SCORE_TABLE = xarray.Dataset(
    data_vars=THIS_DATA_DICT, coords=THIS_METADATA_DICT
)

ADVANCED_SCORE_TABLE[pixelwise_eval.POD_KEY].values = numpy.array(
    [5, 5, 5, 3, 3, 3, 3, 3, 3, 2, 0, 0], dtype=float
) / 5

ADVANCED_SCORE_TABLE[pixelwise_eval.POFD_KEY].values = numpy.array(
    [5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float
) / 5

ADVANCED_SCORE_TABLE[pixelwise_eval.SUCCESS_RATIO_KEY].values = (
    numpy.array([0.5, 5. / 7, 1, 1, 1, 1, 1, 1, 1, 1, numpy.nan, numpy.nan])
)
ADVANCED_SCORE_TABLE[pixelwise_eval.FREQUENCY_BIAS_KEY].values = (
    numpy.array([10, 7, 5, 3, 3, 3, 3, 3, 3, 2, 0, 0], dtype=float) / 5
)
ADVANCED_SCORE_TABLE[pixelwise_eval.CSI_KEY].values = numpy.array([
    0.5, 5. / 7, 1, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0, 0
])
ADVANCED_SCORE_TABLE[pixelwise_eval.ACCURACY_KEY].values = numpy.array([
    0.5, 0.8, 1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.7, 0.5, 0.5
])


def _compare_basic_score_tables(first_table, second_table):
    """Compares two xarray tables with basic scores.

    :param first_table: First table.
    :param second_table: Second table.
    :return: are_tables_equal: Boolean flag.
    """

    float_keys = [
        pixelwise_eval.EVENT_FREQUENCY_KEY,
        pixelwise_eval.MEAN_FORECAST_PROB_KEY
    ]
    integer_keys = [
        pixelwise_eval.NUM_TRUE_POSITIVES_KEY,
        pixelwise_eval.NUM_FALSE_POSITIVES_KEY,
        pixelwise_eval.NUM_FALSE_NEGATIVES_KEY,
        pixelwise_eval.NUM_TRUE_NEGATIVES_KEY,
        pixelwise_eval.NUM_EXAMPLES_KEY
    ]

    for this_key in float_keys:
        if not numpy.allclose(
                first_table[this_key].values, second_table[this_key].values,
                atol=TOLERANCE
        ):
            return False

    for this_key in integer_keys:
        if not numpy.array_equal(
                first_table[this_key].values, second_table[this_key].values
        ):
            return False

    return True


def _compare_advanced_score_tables(first_table, second_table):
    """Compares two xarray tables with advanced scores.

    :param first_table: First table.
    :param second_table: Second table.
    :return: are_tables_equal: Boolean flag.
    """

    keys = [
        pixelwise_eval.POD_KEY, pixelwise_eval.POFD_KEY,
        pixelwise_eval.SUCCESS_RATIO_KEY, pixelwise_eval.FREQUENCY_BIAS_KEY,
        pixelwise_eval.CSI_KEY, pixelwise_eval.ACCURACY_KEY
    ]

    for this_key in keys:
        if not numpy.allclose(
                first_table[this_key].values, second_table[this_key].values,
                atol=TOLERANCE, equal_nan=True
        ):
            return False

    return True


class PixelwiseEvaluationTests(unittest.TestCase):
    """Each method is a unit test for pixelwise_evaluation.py."""

    def test_update_basic_scores(self):
        """Ensures correct output from _update_basic_scores."""

        this_basic_score_table = pixelwise_eval._update_basic_scores(
            basic_score_table_xarray=copy.deepcopy(BASIC_SCORE_TABLE_ORIG),
            prediction_file_name=None, prediction_dict=PREDICTION_DICT
        )[0]

        self.assertTrue(_compare_basic_score_tables(
            this_basic_score_table, BASIC_SCORE_TABLE_NEW
        ))

    def test_get_advanced_scores(self):
        """Ensures correct output from get_advanced_scores."""

        this_advanced_score_table = pixelwise_eval.get_advanced_scores(
            BASIC_SCORE_TABLE_NEW
        )

        self.assertTrue(_compare_advanced_score_tables(
            this_advanced_score_table, ADVANCED_SCORE_TABLE
        ))


if __name__ == '__main__':
    unittest.main()
