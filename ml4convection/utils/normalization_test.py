"""Unit tests for normalization.py."""

import copy
import unittest
import numpy
import scipy.stats
from ml4convection.io import satellite_io
from ml4convection.utils import normalization

TOLERANCE = 1e-6

# The following constants are used to test _update_normalization_params.
ORIGINAL_PARAM_DICT = {
    normalization.NUM_VALUES_KEY: 20,
    normalization.MEAN_VALUE_KEY: 5.,
    normalization.MEAN_OF_SQUARES_KEY: 10.
}

NEW_DATA_MATRIX = numpy.array([
    [0, 1, 2, 3, 4],
    [1, 2, 4, 2, 1]
], dtype=float)

NEW_PARAM_DICT = {
    normalization.NUM_VALUES_KEY: 30,
    normalization.MEAN_VALUE_KEY: 4.,
    normalization.MEAN_OF_SQUARES_KEY: 8.533333
}

# The following constants are used to test _get_standard_deviation.
INPUT_MATRIX_FOR_STDEV = numpy.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
], dtype=float)

STANDARD_DEVIATION = numpy.std(INPUT_MATRIX_FOR_STDEV, ddof=1)

# The following constants are used to test normalize_data and denormalize_data.
THESE_TEMPS_BAND9_KELVINS = numpy.linspace(200, 250, num=11, dtype=float)
THESE_TEMPS_BAND8_KELVINS = numpy.linspace(260, 310, num=11, dtype=float)
SAMPLED_TEMP_MATRIX_KELVINS = numpy.transpose(numpy.vstack((
    THESE_TEMPS_BAND9_KELVINS, THESE_TEMPS_BAND8_KELVINS
)))

THESE_BAND_NUMBERS = numpy.array([9, 8], dtype=int)

NORM_DICT_FOR_TEMPERATURE = {
    normalization.BAND_NUMBERS_KEY: THESE_BAND_NUMBERS,
    normalization.SAMPLED_VALUES_KEY: SAMPLED_TEMP_MATRIX_KELVINS,
    normalization.MEAN_VALUES_KEY: numpy.array([230, 280], dtype=float),
    normalization.STANDARD_DEVIATIONS_KEY: numpy.array([15, 10], dtype=float)
}

# THESE_COUNTS_BAND9 = numpy.linspace(0, 500, num=11, dtype=float)
# THESE_COUNTS_BAND8 = numpy.linspace(600, 1100, num=11, dtype=float)
# SAMPLED_COUNT_MATRIX = numpy.transpose(numpy.vstack((
#     THESE_COUNTS_BAND9, THESE_COUNTS_BAND8
# )))
#
# NORM_DICT_FOR_COUNT = {
#     normalization.BAND_NUMBERS_KEY: THESE_BAND_NUMBERS + 0,
#     normalization.SAMPLED_VALUES_KEY: SAMPLED_COUNT_MATRIX,
#     normalization.MEAN_VALUES_KEY: numpy.array([300, 800], dtype=float),
#     normalization.STANDARD_DEVIATIONS_KEY: numpy.array([75, 50], dtype=float)
# }

LATITUDES_DEG_N = numpy.linspace(53, 54, num=11, dtype=float)
LONGITUDES_DEG_E = numpy.linspace(246, 247, num=6, dtype=float)
VALID_TIMES_UNIX_SEC = numpy.linspace(0, 7, num=8, dtype=int)
THESE_BAND_NUMBERS = numpy.array([8, 9], dtype=int)

THESE_TEMPS_BAND9_KELVINS = numpy.array(
    [231, 233, 218, 240, 203, 231, 233, 230], dtype=float
)
THESE_TEMPS_BAND8_KELVINS = numpy.array(
    [311, 257, 314, 313, 289, 269, 280, 302], dtype=float
)
ACTUAL_TEMP_MATRIX_KELVINS = numpy.transpose(numpy.vstack((
    THESE_TEMPS_BAND8_KELVINS, THESE_TEMPS_BAND9_KELVINS
)))
ACTUAL_TEMP_MATRIX_KELVINS = numpy.expand_dims(
    ACTUAL_TEMP_MATRIX_KELVINS, axis=(1, 2)
)
ACTUAL_TEMP_MATRIX_KELVINS = numpy.repeat(
    ACTUAL_TEMP_MATRIX_KELVINS, repeats=len(LATITUDES_DEG_N), axis=1
)
ACTUAL_TEMP_MATRIX_KELVINS = numpy.repeat(
    ACTUAL_TEMP_MATRIX_KELVINS, repeats=len(LONGITUDES_DEG_E), axis=2
)

SATELLITE_DICT_ACTUAL = {
    satellite_io.BRIGHTNESS_TEMP_KEY: ACTUAL_TEMP_MATRIX_KELVINS + 0.,
    satellite_io.BRIGHTNESS_COUNT_KEY: None,
    satellite_io.LATITUDES_KEY: LATITUDES_DEG_N,
    satellite_io.LONGITUDES_KEY: LONGITUDES_DEG_E,
    satellite_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC,
    satellite_io.BAND_NUMBERS_KEY: THESE_BAND_NUMBERS + 0
}

THESE_TEMPS_BAND9_KELVINS = numpy.array(
    [7, 7, 4, 8, 1, 7, 7, 6], dtype=float
) / 10
THESE_TEMPS_BAND8_KELVINS = numpy.array(
    [10, 0, 10, 10, 6, 2, 4, 9], dtype=float
) / 10
UNIFORM_TEMP_MATRIX_KELVINS = numpy.transpose(numpy.vstack((
    THESE_TEMPS_BAND8_KELVINS, THESE_TEMPS_BAND9_KELVINS
)))
UNIFORM_TEMP_MATRIX_KELVINS = numpy.expand_dims(
    UNIFORM_TEMP_MATRIX_KELVINS, axis=(1, 2)
)
UNIFORM_TEMP_MATRIX_KELVINS = numpy.repeat(
    UNIFORM_TEMP_MATRIX_KELVINS, repeats=len(LATITUDES_DEG_N), axis=1
)
UNIFORM_TEMP_MATRIX_KELVINS = numpy.repeat(
    UNIFORM_TEMP_MATRIX_KELVINS, repeats=len(LONGITUDES_DEG_E), axis=2
)
UNIFORM_TEMP_MATRIX_KELVINS = numpy.maximum(
    UNIFORM_TEMP_MATRIX_KELVINS, normalization.MIN_CUMULATIVE_DENSITY
)
UNIFORM_TEMP_MATRIX_KELVINS = numpy.minimum(
    UNIFORM_TEMP_MATRIX_KELVINS, normalization.MAX_CUMULATIVE_DENSITY
)
UNIFORM_NORM_TEMP_MATRIX_KELVINS = scipy.stats.norm.ppf(
    UNIFORM_TEMP_MATRIX_KELVINS, loc=0., scale=1.
)

SATELLITE_DICT_UNIFORM_NORM = copy.deepcopy(SATELLITE_DICT_ACTUAL)
SATELLITE_DICT_UNIFORM_NORM[satellite_io.BRIGHTNESS_TEMP_KEY] = (
    UNIFORM_NORM_TEMP_MATRIX_KELVINS + 0.
)

THESE_TEMPS_BAND9_KELVINS = numpy.array(
    [231, 233, 218, 240, 203, 231, 233, 230], dtype=float
)
THESE_TEMPS_BAND8_KELVINS = numpy.array(
    [311, 257, 314, 313, 289, 269, 280, 302], dtype=float
)
THESE_TEMPS_BAND9_KELVINS = (THESE_TEMPS_BAND9_KELVINS - 230) / 15
THESE_TEMPS_BAND8_KELVINS = (THESE_TEMPS_BAND8_KELVINS - 280) / 10
SIMPLE_NORM_TEMP_MATRIX_KELVINS = numpy.transpose(numpy.vstack((
    THESE_TEMPS_BAND8_KELVINS, THESE_TEMPS_BAND9_KELVINS
)))
SIMPLE_NORM_TEMP_MATRIX_KELVINS = numpy.expand_dims(
    SIMPLE_NORM_TEMP_MATRIX_KELVINS, axis=(1, 2)
)
SIMPLE_NORM_TEMP_MATRIX_KELVINS = numpy.repeat(
    SIMPLE_NORM_TEMP_MATRIX_KELVINS, repeats=len(LATITUDES_DEG_N), axis=1
)
SIMPLE_NORM_TEMP_MATRIX_KELVINS = numpy.repeat(
    SIMPLE_NORM_TEMP_MATRIX_KELVINS, repeats=len(LONGITUDES_DEG_E), axis=2
)

SATELLITE_DICT_SIMPLE_NORM = copy.deepcopy(SATELLITE_DICT_ACTUAL)
SATELLITE_DICT_SIMPLE_NORM[satellite_io.BRIGHTNESS_TEMP_KEY] = (
    SIMPLE_NORM_TEMP_MATRIX_KELVINS + 0.
)

# THESE_TEMPS_BAND9_KELVINS = numpy.array([
#     235, 235, 220, 240, 207.5, 235, 235, 230
# ])
# THESE_TEMPS_BAND8_KELVINS = numpy.array([
#     307.5, 262.5, 307.5, 307.5, 290, 267.5, 280, 305
# ])

THESE_TEMPS_BAND9_KELVINS = numpy.array(
    [235, 235, 220, 240, 205, 235, 235, 230], dtype=float
)
THESE_TEMPS_BAND8_KELVINS = numpy.array([
    309.99995, 260.00005, 309.99995, 309.99995, 290, 270, 280, 305
])
DENORM_TEMP_MATRIX_KELVINS = numpy.transpose(numpy.vstack((
    THESE_TEMPS_BAND8_KELVINS, THESE_TEMPS_BAND9_KELVINS
)))
DENORM_TEMP_MATRIX_KELVINS = numpy.expand_dims(
    DENORM_TEMP_MATRIX_KELVINS, axis=(1, 2)
)
DENORM_TEMP_MATRIX_KELVINS = numpy.repeat(
    DENORM_TEMP_MATRIX_KELVINS, repeats=len(LATITUDES_DEG_N), axis=1
)
DENORM_TEMP_MATRIX_KELVINS = numpy.repeat(
    DENORM_TEMP_MATRIX_KELVINS, repeats=len(LONGITUDES_DEG_E), axis=2
)

SATELLITE_DICT_UNIFORM_DENORM = copy.deepcopy(SATELLITE_DICT_ACTUAL)
SATELLITE_DICT_UNIFORM_DENORM[satellite_io.BRIGHTNESS_TEMP_KEY] = (
    DENORM_TEMP_MATRIX_KELVINS + 0.
)


def _compare_param_dicts(first_param_dict, second_param_dict):
    """Compares two dictionaries with normalization parameters.

    :param first_param_dict: First dictionary.
    :param second_param_dict: Second dictionary.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = list(first_param_dict.keys())
    second_keys = list(second_param_dict.keys())
    if set(first_keys) != set(second_keys):
        return False

    for this_key in first_keys:
        if not numpy.isclose(
                first_param_dict[this_key], second_param_dict[this_key],
                atol=TOLERANCE
        ):
            return False

    return True


def _compare_satellite_dicts(first_satellite_dict, second_satellite_dict):
    """Compares two dictionaries with satellite data.

    :param first_satellite_dict: First dictionary.
    :param second_satellite_dict: Second dictionary.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = list(first_satellite_dict.keys())
    second_keys = list(second_satellite_dict.keys())
    if set(first_keys) != set(second_keys):
        return False

    possibly_none_keys = [
        satellite_io.BRIGHTNESS_TEMP_KEY, satellite_io.BRIGHTNESS_COUNT_KEY
    ]
    integer_keys = [satellite_io.VALID_TIMES_KEY, satellite_io.BAND_NUMBERS_KEY]

    for this_key in first_keys:
        if (
                this_key in possibly_none_keys
                and first_satellite_dict[this_key] is None
                and second_satellite_dict[this_key] is None
        ):
            continue

        if this_key in integer_keys:
            if numpy.array_equal(
                    first_satellite_dict[this_key],
                    second_satellite_dict[this_key]
            ):
                continue

            return False

        if numpy.allclose(
                first_satellite_dict[this_key], second_satellite_dict[this_key],
                atol=TOLERANCE
        ):
            continue

        return False

    return True


class NormalizationTests(unittest.TestCase):
    """Each method is a unit test for normalization.py."""

    def test_update_normalization_params(self):
        """Ensures correct output from _update_normalization_params."""

        this_new_param_dict = normalization._update_normalization_params(
            normalization_param_dict=copy.deepcopy(ORIGINAL_PARAM_DICT),
            new_data_matrix=NEW_DATA_MATRIX
        )

        self.assertTrue(_compare_param_dicts(
            this_new_param_dict, NEW_PARAM_DICT
        ))

    def test_get_standard_deviation(self):
        """Ensures correct output from _get_standard_deviation."""

        this_normalization_param_dict = {
            normalization.NUM_VALUES_KEY: INPUT_MATRIX_FOR_STDEV.size,
            normalization.MEAN_VALUE_KEY: numpy.mean(INPUT_MATRIX_FOR_STDEV),
            normalization.MEAN_OF_SQUARES_KEY:
                numpy.mean(INPUT_MATRIX_FOR_STDEV ** 2)
        }

        this_standard_deviation = (
            normalization._get_standard_deviation(this_normalization_param_dict)
        )

        self.assertTrue(numpy.isclose(
            this_standard_deviation, STANDARD_DEVIATION, atol=TOLERANCE
        ))

    def test_normalize_data_uniform(self):
        """Ensures correct output from normalize_data.

        In this case, converts to uniform distribution before normal
        distribution.
        """

        this_satellite_dict = normalization.normalize_data(
            satellite_dict=copy.deepcopy(SATELLITE_DICT_ACTUAL),
            uniformize=True,
            norm_dict_for_temperature=NORM_DICT_FOR_TEMPERATURE
        )

        self.assertTrue(_compare_satellite_dicts(
            this_satellite_dict, SATELLITE_DICT_UNIFORM_NORM
        ))

    def test_normalize_data_simple(self):
        """Ensures correct output from normalize_data.

        In this case, does not convert to uniform distribution before normal
        distribution.
        """

        this_satellite_dict = normalization.normalize_data(
            satellite_dict=copy.deepcopy(SATELLITE_DICT_ACTUAL),
            uniformize=False,
            norm_dict_for_temperature=NORM_DICT_FOR_TEMPERATURE
        )

        self.assertTrue(_compare_satellite_dicts(
            this_satellite_dict, SATELLITE_DICT_SIMPLE_NORM
        ))

    def test_denormalize_data_uniform(self):
        """Ensures correct output from denormalize_data.

        In this case, converts from uniform distribution after normal
        distribution.
        """

        this_satellite_dict = normalization.denormalize_data(
            satellite_dict=copy.deepcopy(SATELLITE_DICT_UNIFORM_NORM),
            uniformize=True,
            norm_dict_for_temperature=NORM_DICT_FOR_TEMPERATURE
        )

        self.assertTrue(_compare_satellite_dicts(
            this_satellite_dict, SATELLITE_DICT_UNIFORM_DENORM
        ))

    def test_denormalize_data_simple(self):
        """Ensures correct output from denormalize_data.

        In this case, does not convert from uniform distribution after normal
        distribution.
        """

        this_satellite_dict = normalization.denormalize_data(
            satellite_dict=copy.deepcopy(SATELLITE_DICT_SIMPLE_NORM),
            uniformize=False,
            norm_dict_for_temperature=NORM_DICT_FOR_TEMPERATURE
        )

        self.assertTrue(_compare_satellite_dicts(
            this_satellite_dict, SATELLITE_DICT_ACTUAL
        ))


if __name__ == '__main__':
    unittest.main()
