"""Unit tests for example_io.py."""

import copy
import unittest
import numpy
from ml4convection.io import radar_io
from ml4convection.io import satellite_io
from ml4convection.io import satellite_io_test
from ml4convection.io import example_io

TOLERANCE = 1e-6

# The following constants are used to test downsample_data_in_space.
TEMP_MATRIX_BAND8_EXAMPLE1_KELVINS = numpy.array([
    [200, 210, 220, 230],
    [210, 220, 230, 240],
    [220, 230, 240, 250],
    [230, 240, 250, 260],
    [250, 260, 270, 280],
    [290, 300, 310, 320]
], dtype=float)

TEMP_MATRIX_BAND9_EXAMPLE1_KELVINS = TEMP_MATRIX_BAND8_EXAMPLE1_KELVINS - 10.
TEMP_MATRIX_EXAMPLE1_KELVINS = numpy.stack(
    (TEMP_MATRIX_BAND8_EXAMPLE1_KELVINS, TEMP_MATRIX_BAND9_EXAMPLE1_KELVINS),
    axis=-1
)
TEMPERATURE_MATRIX_KELVINS = numpy.stack((
    TEMP_MATRIX_EXAMPLE1_KELVINS, TEMP_MATRIX_EXAMPLE1_KELVINS + 2.5,
    TEMP_MATRIX_EXAMPLE1_KELVINS + 5, TEMP_MATRIX_EXAMPLE1_KELVINS + 7.5
))

LATITUDES_DEG_N = numpy.array([53, 53.2, 53.4, 53.6, 53.8, 54])
LONGITUDES_DEG_E = numpy.array([246, 246.5, 247, 247.5])
VALID_TIMES_UNIX_SEC = numpy.linspace(0, 3, num=4, dtype=int)
BAND_NUMBERS = numpy.array([8, 9], dtype=int)

SATELLITE_DICT = {
    satellite_io.BRIGHTNESS_TEMP_KEY: TEMPERATURE_MATRIX_KELVINS + 0.,
    satellite_io.BRIGHTNESS_COUNT_KEY: None,
    satellite_io.LATITUDES_KEY: LATITUDES_DEG_N + 0.,
    satellite_io.LONGITUDES_KEY: LONGITUDES_DEG_E + 0.,
    satellite_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC + 0,
    satellite_io.BAND_NUMBERS_KEY: BAND_NUMBERS + 0
}

CONV_FLAG_MATRIX_EXAMPLE1 = numpy.array([
    [0, 0, 0, 1],
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [1, 1, 1, 1],
    [0, 1, 1, 1],
    [0, 0, 1, 1]
], dtype=bool)

CONV_FLAG_MATRIX_EXAMPLE2 = numpy.invert(CONV_FLAG_MATRIX_EXAMPLE1)
CONV_FLAG_MATRIX_EXAMPLE3 = numpy.full(
    CONV_FLAG_MATRIX_EXAMPLE1.shape, False, dtype=bool
)
CONV_FLAG_MATRIX_EXAMPLE4 = numpy.full(
    CONV_FLAG_MATRIX_EXAMPLE1.shape, True, dtype=bool
)

CONVECTIVE_FLAG_MATRIX = numpy.stack((
    CONV_FLAG_MATRIX_EXAMPLE1, CONV_FLAG_MATRIX_EXAMPLE2,
    CONV_FLAG_MATRIX_EXAMPLE3, CONV_FLAG_MATRIX_EXAMPLE4,
), axis=0)

ECHO_CLASSIFN_DICT = {
    radar_io.CONVECTIVE_FLAGS_KEY: copy.deepcopy(CONVECTIVE_FLAG_MATRIX),
    radar_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC + 0,
    radar_io.LATITUDES_KEY: LATITUDES_DEG_N + 0.,
    radar_io.LONGITUDES_KEY: LONGITUDES_DEG_E + 0.
}

TEMP_MATRIX_BAND8_EXAMPLE1_KELVINS = numpy.array([
    [210, 230],
    [230, 250],
    [275, 295]
], dtype=float)

TEMP_MATRIX_BAND9_EXAMPLE1_KELVINS = TEMP_MATRIX_BAND8_EXAMPLE1_KELVINS - 10.
TEMP_MATRIX_EXAMPLE1_KELVINS = numpy.stack(
    (TEMP_MATRIX_BAND8_EXAMPLE1_KELVINS, TEMP_MATRIX_BAND9_EXAMPLE1_KELVINS),
    axis=-1
)
TEMPERATURE_MATRIX_KELVINS = numpy.stack((
    TEMP_MATRIX_EXAMPLE1_KELVINS, TEMP_MATRIX_EXAMPLE1_KELVINS + 2.5,
    TEMP_MATRIX_EXAMPLE1_KELVINS + 5, TEMP_MATRIX_EXAMPLE1_KELVINS + 7.5
))

LATITUDES_DEG_N = numpy.array([53.1, 53.5, 53.9])
LONGITUDES_DEG_E = numpy.array([246.25, 247.25])

SATELLITE_DICT_DOWNSAMPLED2 = {
    satellite_io.BRIGHTNESS_TEMP_KEY: TEMPERATURE_MATRIX_KELVINS + 0.,
    satellite_io.BRIGHTNESS_COUNT_KEY: None,
    satellite_io.LATITUDES_KEY: LATITUDES_DEG_N + 0.,
    satellite_io.LONGITUDES_KEY: LONGITUDES_DEG_E + 0.,
    satellite_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC + 0,
    satellite_io.BAND_NUMBERS_KEY: BAND_NUMBERS + 0
}

CONV_FLAG_MATRIX_EXAMPLE1 = numpy.array([
    [0, 1],
    [1, 1],
    [1, 1]
], dtype=bool)

CONV_FLAG_MATRIX_EXAMPLE2 = numpy.array([
    [1, 1],
    [1, 0],
    [1, 0]
], dtype=bool)

CONV_FLAG_MATRIX_EXAMPLE3 = numpy.full(
    CONV_FLAG_MATRIX_EXAMPLE1.shape, False, dtype=bool
)
CONV_FLAG_MATRIX_EXAMPLE4 = numpy.full(
    CONV_FLAG_MATRIX_EXAMPLE1.shape, True, dtype=bool
)

CONVECTIVE_FLAG_MATRIX = numpy.stack((
    CONV_FLAG_MATRIX_EXAMPLE1, CONV_FLAG_MATRIX_EXAMPLE2,
    CONV_FLAG_MATRIX_EXAMPLE3, CONV_FLAG_MATRIX_EXAMPLE4,
), axis=0)

ECHO_CLASSIFN_DICT_DOWNSAMPLED2 = {
    radar_io.CONVECTIVE_FLAGS_KEY: copy.deepcopy(CONVECTIVE_FLAG_MATRIX),
    radar_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC + 0,
    radar_io.LATITUDES_KEY: LATITUDES_DEG_N + 0.,
    radar_io.LONGITUDES_KEY: LONGITUDES_DEG_E + 0.
}

TEMP_MATRIX_BAND8_EXAMPLE1_KELVINS = numpy.array([[230]], dtype=float)
TEMP_MATRIX_BAND9_EXAMPLE1_KELVINS = TEMP_MATRIX_BAND8_EXAMPLE1_KELVINS - 10.
TEMP_MATRIX_EXAMPLE1_KELVINS = numpy.stack(
    (TEMP_MATRIX_BAND8_EXAMPLE1_KELVINS, TEMP_MATRIX_BAND9_EXAMPLE1_KELVINS),
    axis=-1
)
TEMPERATURE_MATRIX_KELVINS = numpy.stack((
    TEMP_MATRIX_EXAMPLE1_KELVINS, TEMP_MATRIX_EXAMPLE1_KELVINS + 2.5,
    TEMP_MATRIX_EXAMPLE1_KELVINS + 5, TEMP_MATRIX_EXAMPLE1_KELVINS + 7.5
))

LATITUDES_DEG_N = numpy.array([53.3])
LONGITUDES_DEG_E = numpy.array([246.75])

SATELLITE_DICT_DOWNSAMPLED4 = {
    satellite_io.BRIGHTNESS_TEMP_KEY: TEMPERATURE_MATRIX_KELVINS + 0.,
    satellite_io.BRIGHTNESS_COUNT_KEY: None,
    satellite_io.LATITUDES_KEY: LATITUDES_DEG_N + 0.,
    satellite_io.LONGITUDES_KEY: LONGITUDES_DEG_E + 0.,
    satellite_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC + 0,
    satellite_io.BAND_NUMBERS_KEY: BAND_NUMBERS + 0
}

CONV_FLAG_MATRIX_EXAMPLE1 = numpy.array([
    [1]
], dtype=bool)

CONV_FLAG_MATRIX_EXAMPLE2 = numpy.array([
    [1]
], dtype=bool)

CONV_FLAG_MATRIX_EXAMPLE3 = numpy.full(
    CONV_FLAG_MATRIX_EXAMPLE1.shape, False, dtype=bool
)
CONV_FLAG_MATRIX_EXAMPLE4 = numpy.full(
    CONV_FLAG_MATRIX_EXAMPLE1.shape, True, dtype=bool
)

CONVECTIVE_FLAG_MATRIX = numpy.stack((
    CONV_FLAG_MATRIX_EXAMPLE1, CONV_FLAG_MATRIX_EXAMPLE2,
    CONV_FLAG_MATRIX_EXAMPLE3, CONV_FLAG_MATRIX_EXAMPLE4,
), axis=0)

ECHO_CLASSIFN_DICT_DOWNSAMPLED4 = {
    radar_io.CONVECTIVE_FLAGS_KEY: copy.deepcopy(CONVECTIVE_FLAG_MATRIX),
    radar_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC + 0,
    radar_io.LATITUDES_KEY: LATITUDES_DEG_N + 0.,
    radar_io.LONGITUDES_KEY: LONGITUDES_DEG_E + 0.
}

# The following constants are used to test find_predictor_file,
# predictor_file_name_to_date, find_target_file, and target_file_name_to_date.
TOP_PREDICTOR_DIR_NAME = 'predictors'
PREDICTOR_DATE_STRING = '20200820'
PREDICTOR_FILE_NAME = 'predictors/2020/predictors_20200820.nc'

TOP_TARGET_DIR_NAME = 'targets'
TARGET_DATE_STRING = '40550405'
TARGET_FILE_NAME = 'targets/4055/targets_40550405.nc'

# The following constants are used to test find_many_predictor_files and
# find_many_target_files.
FIRST_DATE_STRING = '20200818'
LAST_DATE_STRING = '20200821'

PREDICTOR_FILE_NAMES = [
    'predictors/2020/predictors_20200818.nc',
    'predictors/2020/predictors_20200819.nc',
    'predictors/2020/predictors_20200820.nc',
    'predictors/2020/predictors_20200821.nc'
]
TARGET_FILE_NAMES = [
    'targets/2020/targets_20200818.nc',
    'targets/2020/targets_20200819.nc',
    'targets/2020/targets_20200820.nc',
    'targets/2020/targets_20200821.nc'
]

# The following constants are used to test subset* and concat*.
LATITUDES_DEG_N = numpy.linspace(53, 54, num=11, dtype=float)
LONGITUDES_DEG_E = numpy.linspace(246, 247, num=6, dtype=float)
VALID_TIMES_UNIX_SEC = numpy.linspace(0, 7, num=8, dtype=int)
BAND_NUMBERS = numpy.array([8, 9, 10], dtype=int)
NORMALIZATION_FILE_NAME = 'foo'

BAND8_TEMPS_KELVINS = numpy.array(
    [311, 257, 314, 313, 289, 269, 280, 302], dtype=float
)
BAND9_TEMPS_KELVINS = numpy.array(
    [231, 233, 218, 240, 203, 231, 233, 230], dtype=float
)
BAND10_TEMPS_KELVINS = numpy.array(
    [254, 264, 295, 260, 245, 281, 273, 258], dtype=float
)

TEMPERATURE_MATRIX_KELVINS = numpy.transpose(numpy.vstack((
    BAND8_TEMPS_KELVINS, BAND9_TEMPS_KELVINS, BAND10_TEMPS_KELVINS
)))
TEMPERATURE_MATRIX_KELVINS = numpy.expand_dims(
    TEMPERATURE_MATRIX_KELVINS, axis=(1, 2)
)
TEMPERATURE_MATRIX_KELVINS = numpy.repeat(
    TEMPERATURE_MATRIX_KELVINS, repeats=len(LATITUDES_DEG_N), axis=1
)
TEMPERATURE_MATRIX_KELVINS = numpy.repeat(
    TEMPERATURE_MATRIX_KELVINS, repeats=len(LONGITUDES_DEG_E), axis=2
)

COUNT_MATRIX = numpy.round(TEMPERATURE_MATRIX_KELVINS - 100)

PREDICTOR_DICT_ALL_EXAMPLES = {
    example_io.PREDICTOR_MATRIX_UNNORM_KEY: TEMPERATURE_MATRIX_KELVINS + 0.,
    example_io.PREDICTOR_MATRIX_NORM_KEY: COUNT_MATRIX + 0.,
    example_io.PREDICTOR_MATRIX_UNIF_NORM_KEY: None,
    example_io.LATITUDES_KEY: LATITUDES_DEG_N,
    example_io.LONGITUDES_KEY: LONGITUDES_DEG_E,
    example_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC + 0,
    example_io.BAND_NUMBERS_KEY: BAND_NUMBERS + 0,
    example_io.NORMALIZATION_FILE_KEY: NORMALIZATION_FILE_NAME
}

DESIRED_BAND_NUMBERS = numpy.array([10, 8], dtype=int)

PREDICTOR_DICT_SUBSET_BY_BAND = {
    example_io.PREDICTOR_MATRIX_UNNORM_KEY:
        TEMPERATURE_MATRIX_KELVINS[..., [2, 0]],
    example_io.PREDICTOR_MATRIX_NORM_KEY: COUNT_MATRIX[..., [2, 0]],
    example_io.PREDICTOR_MATRIX_UNIF_NORM_KEY: None,
    example_io.LATITUDES_KEY: LATITUDES_DEG_N,
    example_io.LONGITUDES_KEY: LONGITUDES_DEG_E,
    example_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC + 0,
    example_io.BAND_NUMBERS_KEY: BAND_NUMBERS[..., [2, 0]],
    example_io.NORMALIZATION_FILE_KEY: NORMALIZATION_FILE_NAME
}

REFLECTIVITIES_DBZ = numpy.array([10, 20, 33, 44, 35, 66, 77, 66.6])
COMPOSITE_REFL_MATRIX_DBZ = numpy.expand_dims(
    REFLECTIVITIES_DBZ, axis=(1, 2)
)
COMPOSITE_REFL_MATRIX_DBZ = numpy.repeat(
    COMPOSITE_REFL_MATRIX_DBZ, repeats=len(LATITUDES_DEG_N), axis=1
)
COMPOSITE_REFL_MATRIX_DBZ = numpy.repeat(
    COMPOSITE_REFL_MATRIX_DBZ, repeats=len(LONGITUDES_DEG_E), axis=2
)

COMPOSITE_REFL_THRESHOLD_DBZ = 35.
TARGET_MATRIX = (
    COMPOSITE_REFL_MATRIX_DBZ >= COMPOSITE_REFL_THRESHOLD_DBZ
).astype(int)

TARGET_DICT_ALL_EXAMPLES = {
    example_io.TARGET_MATRIX_KEY: TARGET_MATRIX + 0,
    example_io.LATITUDES_KEY: LATITUDES_DEG_N,
    example_io.LONGITUDES_KEY: LONGITUDES_DEG_E,
    example_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC + 0
}

DESIRED_INDICES = numpy.array([1, 7, 5, 3], dtype=int)

PREDICTOR_DICT_SUBSET_BY_INDEX = {
    example_io.PREDICTOR_MATRIX_UNNORM_KEY:
        TEMPERATURE_MATRIX_KELVINS[[1, 7, 5, 3], ...],
    example_io.PREDICTOR_MATRIX_NORM_KEY: COUNT_MATRIX[[1, 7, 5, 3], ...],
    example_io.PREDICTOR_MATRIX_UNIF_NORM_KEY: None,
    example_io.LATITUDES_KEY: LATITUDES_DEG_N,
    example_io.LONGITUDES_KEY: LONGITUDES_DEG_E,
    example_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC[[1, 7, 5, 3]],
    example_io.BAND_NUMBERS_KEY: BAND_NUMBERS + 0,
    example_io.NORMALIZATION_FILE_KEY: NORMALIZATION_FILE_NAME
}

TARGET_DICT_SUBSET_BY_INDEX = {
    example_io.TARGET_MATRIX_KEY: TARGET_MATRIX[[1, 7, 5, 3], ...],
    example_io.LATITUDES_KEY: LATITUDES_DEG_N,
    example_io.LONGITUDES_KEY: LONGITUDES_DEG_E,
    example_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC[[1, 7, 5, 3]]
}

DESIRED_TIMES_UNIX_SEC = numpy.array([2, 6, 0, 4], dtype=int)

PREDICTOR_DICT_SUBSET_BY_TIME = {
    example_io.PREDICTOR_MATRIX_UNNORM_KEY:
        TEMPERATURE_MATRIX_KELVINS[[2, 6, 0, 4], ...],
    example_io.PREDICTOR_MATRIX_NORM_KEY: COUNT_MATRIX[[2, 6, 0, 4], ...],
    example_io.PREDICTOR_MATRIX_UNIF_NORM_KEY: None,
    example_io.LATITUDES_KEY: LATITUDES_DEG_N,
    example_io.LONGITUDES_KEY: LONGITUDES_DEG_E,
    example_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC[[2, 6, 0, 4]],
    example_io.BAND_NUMBERS_KEY: BAND_NUMBERS + 0,
    example_io.NORMALIZATION_FILE_KEY: NORMALIZATION_FILE_NAME
}

TARGET_DICT_SUBSET_BY_TIME = {
    example_io.TARGET_MATRIX_KEY: TARGET_MATRIX[[2, 6, 0, 4], ...],
    example_io.LATITUDES_KEY: LATITUDES_DEG_N,
    example_io.LONGITUDES_KEY: LONGITUDES_DEG_E,
    example_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC[[2, 6, 0, 4]]
}

PREDICTOR_DICT_CONCAT = {
    example_io.PREDICTOR_MATRIX_UNNORM_KEY:
        TEMPERATURE_MATRIX_KELVINS[[1, 7, 5, 3, 2, 6, 0, 4], ...],
    example_io.PREDICTOR_MATRIX_NORM_KEY:
        COUNT_MATRIX[[1, 7, 5, 3, 2, 6, 0, 4], ...],
    example_io.PREDICTOR_MATRIX_UNIF_NORM_KEY: None,
    example_io.LATITUDES_KEY: LATITUDES_DEG_N,
    example_io.LONGITUDES_KEY: LONGITUDES_DEG_E,
    example_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC[[1, 7, 5, 3, 2, 6, 0, 4]],
    example_io.BAND_NUMBERS_KEY: BAND_NUMBERS + 0,
    example_io.NORMALIZATION_FILE_KEY: NORMALIZATION_FILE_NAME
}


def _compare_predictor_dicts(first_predictor_dict, second_predictor_dict):
    """Compares two dictionaries with predictor data.

    :param first_predictor_dict: See doc for `example_io.read_predictor_file`.
    :param second_predictor_dict: Same.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = list(first_predictor_dict.keys())
    second_keys = list(second_predictor_dict.keys())
    if set(first_keys) != set(second_keys):
        return False

    maybe_none_keys = [
        example_io.PREDICTOR_MATRIX_UNNORM_KEY,
        example_io.PREDICTOR_MATRIX_NORM_KEY,
        example_io.PREDICTOR_MATRIX_UNIF_NORM_KEY
    ]
    float_keys = maybe_none_keys + [
        example_io.LATITUDES_KEY, example_io.LONGITUDES_KEY
    ]
    integer_keys = [example_io.VALID_TIMES_KEY, example_io.BAND_NUMBERS_KEY]
    string_keys = [example_io.NORMALIZATION_FILE_KEY]

    for this_key in float_keys:
        if (
                this_key in maybe_none_keys
                and first_predictor_dict[this_key] is None
                and second_predictor_dict[this_key] is None
        ):
            continue

        if not numpy.allclose(
                first_predictor_dict[this_key], second_predictor_dict[this_key],
                atol=TOLERANCE
        ):
            return False

    for this_key in integer_keys:
        if not numpy.array_equal(
                first_predictor_dict[this_key], second_predictor_dict[this_key]
        ):
            return False

    for this_key in string_keys:
        if first_predictor_dict[this_key] != second_predictor_dict[this_key]:
            return False

    return True


def _compare_echo_classifn_dicts(first_echo_classifn_dict,
                                 second_echo_classifn_dict):
    """Compares two dictionaries with echo-classification data.

    :param first_echo_classifn_dict: See doc for
        `radar_io.read_echo_classifn_file`.
    :param second_echo_classifn_dict: Same.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = list(first_echo_classifn_dict.keys())
    second_keys = list(second_echo_classifn_dict.keys())
    if set(first_keys) != set(second_keys):
        return False

    float_keys = [radar_io.LATITUDES_KEY, radar_io.LONGITUDES_KEY]
    exact_keys = [radar_io.CONVECTIVE_FLAGS_KEY, radar_io.VALID_TIMES_KEY]

    for this_key in float_keys:
        if not numpy.allclose(
                first_echo_classifn_dict[this_key],
                second_echo_classifn_dict[this_key],
                atol=TOLERANCE
        ):
            return False

    for this_key in exact_keys:
        if not numpy.array_equal(
                first_echo_classifn_dict[this_key],
                second_echo_classifn_dict[this_key]
        ):
            return False

    return True


def _compare_target_dicts(first_target_dict, second_target_dict):
    """Compares two dictionaries with target data.

    :param first_target_dict: See doc for `example_io.read_target_file`.
    :param second_target_dict: Same.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = list(first_target_dict.keys())
    second_keys = list(second_target_dict.keys())
    if set(first_keys) != set(second_keys):
        return False

    float_keys = [example_io.LATITUDES_KEY, example_io.LONGITUDES_KEY]
    integer_keys = [example_io.TARGET_MATRIX_KEY, example_io.VALID_TIMES_KEY]

    for this_key in float_keys:
        if not numpy.allclose(
                first_target_dict[this_key], second_target_dict[this_key],
                atol=TOLERANCE
        ):
            return False

    for this_key in integer_keys:
        if not numpy.array_equal(
                first_target_dict[this_key], second_target_dict[this_key]
        ):
            return False

    return True


class ExampleIoTests(unittest.TestCase):
    """Each method is a unit test for example_io.py."""

    def test_downsample_data_in_space_factor2(self):
        """Ensures correct output from downsample_data_in_space.

        In this case, downsampling factor is 2.
        """

        this_satellite_dict, this_echo_classifn_dict = (
            example_io.downsample_data_in_space(
                satellite_dict=copy.deepcopy(SATELLITE_DICT),
                echo_classifn_dict=copy.deepcopy(ECHO_CLASSIFN_DICT),
                downsampling_factor=2, change_coordinates=True
            )
        )

        self.assertTrue(satellite_io_test.compare_satellite_dicts(
            this_satellite_dict, SATELLITE_DICT_DOWNSAMPLED2
        ))
        self.assertTrue(_compare_echo_classifn_dicts(
            this_echo_classifn_dict, ECHO_CLASSIFN_DICT_DOWNSAMPLED2
        ))

    def test_downsample_data_in_space_factor4(self):
        """Ensures correct output from downsample_data_in_space.

        In this case, downsampling factor is 4.
        """

        this_satellite_dict, this_echo_classifn_dict = (
            example_io.downsample_data_in_space(
                satellite_dict=copy.deepcopy(SATELLITE_DICT),
                echo_classifn_dict=copy.deepcopy(ECHO_CLASSIFN_DICT),
                downsampling_factor=4, change_coordinates=True
            )
        )

        self.assertTrue(satellite_io_test.compare_satellite_dicts(
            this_satellite_dict, SATELLITE_DICT_DOWNSAMPLED4
        ))
        self.assertTrue(_compare_echo_classifn_dicts(
            this_echo_classifn_dict, ECHO_CLASSIFN_DICT_DOWNSAMPLED4
        ))

    def test_find_predictor_file(self):
        """Ensures correct output from find_predictor_file."""

        this_file_name = example_io.find_predictor_file(
            top_directory_name=TOP_PREDICTOR_DIR_NAME,
            date_string=PREDICTOR_DATE_STRING, raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == PREDICTOR_FILE_NAME)

    def test_predictor_file_name_to_date(self):
        """Ensures correct output from predictor_file_name_to_date."""

        self.assertTrue(
            example_io.predictor_file_name_to_date(PREDICTOR_FILE_NAME) ==
            PREDICTOR_DATE_STRING
        )

    def test_find_many_predictor_files(self):
        """Ensures correct output from find_many_predictor_files."""

        these_file_names = example_io.find_many_predictor_files(
            top_directory_name=TOP_PREDICTOR_DIR_NAME,
            first_date_string=FIRST_DATE_STRING,
            last_date_string=LAST_DATE_STRING, test_mode=True
        )

        self.assertTrue(these_file_names == PREDICTOR_FILE_NAMES)

    def test_find_target_file(self):
        """Ensures correct output from find_target_file."""

        this_file_name = example_io.find_target_file(
            top_directory_name=TOP_TARGET_DIR_NAME,
            date_string=TARGET_DATE_STRING, raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == TARGET_FILE_NAME)

    def test_target_file_name_to_date(self):
        """Ensures correct output from target_file_name_to_date."""

        self.assertTrue(
            example_io.target_file_name_to_date(TARGET_FILE_NAME) ==
            TARGET_DATE_STRING
        )

    def test_find_many_target_files(self):
        """Ensures correct output from find_many_target_files."""

        these_file_names = example_io.find_many_target_files(
            top_directory_name=TOP_TARGET_DIR_NAME,
            first_date_string=FIRST_DATE_STRING,
            last_date_string=LAST_DATE_STRING, test_mode=True
        )

        self.assertTrue(these_file_names == TARGET_FILE_NAMES)

    def test_subset_predictors_by_band(self):
        """Ensures correct output from subset_predictors_by_band."""

        this_predictor_dict = example_io.subset_predictors_by_band(
            predictor_dict=copy.deepcopy(PREDICTOR_DICT_ALL_EXAMPLES),
            band_numbers=DESIRED_BAND_NUMBERS
        )

        self.assertTrue(_compare_predictor_dicts(
            this_predictor_dict, PREDICTOR_DICT_SUBSET_BY_BAND
        ))

    def test_subset_by_time_predictors_only(self):
        """Ensures correct output from subset_by_time.

        In this case, using predictors only.
        """

        this_predictor_dict = example_io.subset_by_time(
            predictor_dict=copy.deepcopy(PREDICTOR_DICT_ALL_EXAMPLES),
            target_dict=None, desired_times_unix_sec=DESIRED_TIMES_UNIX_SEC
        )[0]

        self.assertTrue(_compare_predictor_dicts(
            this_predictor_dict, PREDICTOR_DICT_SUBSET_BY_TIME
        ))

    def test_subset_by_time_targets_only(self):
        """Ensures correct output from subset_by_time.

        In this case, using targets only.
        """

        this_target_dict = example_io.subset_by_time(
            predictor_dict=None,
            target_dict=copy.deepcopy(TARGET_DICT_ALL_EXAMPLES),
            desired_times_unix_sec=DESIRED_TIMES_UNIX_SEC
        )[1]

        self.assertTrue(_compare_target_dicts(
            this_target_dict, TARGET_DICT_SUBSET_BY_TIME
        ))

    def test_subset_by_time_both(self):
        """Ensures correct output from subset_by_time.

        In this case, using both predictors and targets.
        """

        this_predictor_dict, this_target_dict = example_io.subset_by_time(
            predictor_dict=copy.deepcopy(PREDICTOR_DICT_ALL_EXAMPLES),
            target_dict=copy.deepcopy(TARGET_DICT_ALL_EXAMPLES),
            desired_times_unix_sec=DESIRED_TIMES_UNIX_SEC
        )

        self.assertTrue(_compare_predictor_dicts(
            this_predictor_dict, PREDICTOR_DICT_SUBSET_BY_TIME
        ))
        self.assertTrue(_compare_target_dicts(
            this_target_dict, TARGET_DICT_SUBSET_BY_TIME
        ))

    def test_subset_by_index_predictors_only(self):
        """Ensures correct output from subset_by_index.

        In this case, using predictors only.
        """

        this_predictor_dict = example_io.subset_by_index(
            predictor_dict=copy.deepcopy(PREDICTOR_DICT_ALL_EXAMPLES),
            target_dict=None, desired_indices=DESIRED_INDICES
        )[0]

        self.assertTrue(_compare_predictor_dicts(
            this_predictor_dict, PREDICTOR_DICT_SUBSET_BY_INDEX
        ))

    def test_subset_by_index_targets_only(self):
        """Ensures correct output from subset_by_index.

        In this case, using targets only.
        """

        this_target_dict = example_io.subset_by_index(
            predictor_dict=None,
            target_dict=copy.deepcopy(TARGET_DICT_ALL_EXAMPLES),
            desired_indices=DESIRED_INDICES
        )[1]

        self.assertTrue(_compare_target_dicts(
            this_target_dict, TARGET_DICT_SUBSET_BY_INDEX
        ))

    def test_subset_by_index_both(self):
        """Ensures correct output from subset_by_index.

        In this case, using both predictors and targets.
        """

        this_predictor_dict, this_target_dict = example_io.subset_by_index(
            predictor_dict=copy.deepcopy(PREDICTOR_DICT_ALL_EXAMPLES),
            target_dict=copy.deepcopy(TARGET_DICT_ALL_EXAMPLES),
            desired_indices=DESIRED_INDICES
        )

        self.assertTrue(_compare_predictor_dicts(
            this_predictor_dict, PREDICTOR_DICT_SUBSET_BY_INDEX
        ))
        self.assertTrue(_compare_target_dicts(
            this_target_dict, TARGET_DICT_SUBSET_BY_INDEX
        ))

    def test_concat_predictor_data(self):
        """Ensures correct output from concat_predictor_data."""

        this_predictor_dict = example_io.concat_predictor_data([
            PREDICTOR_DICT_SUBSET_BY_INDEX, PREDICTOR_DICT_SUBSET_BY_TIME
        ])

        self.assertTrue(_compare_predictor_dicts(
            this_predictor_dict, PREDICTOR_DICT_CONCAT
        ))


if __name__ == '__main__':
    unittest.main()
