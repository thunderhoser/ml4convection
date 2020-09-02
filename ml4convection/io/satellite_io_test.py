"""Unit tests for satellite_io.py."""

import copy
import unittest
import numpy
from ml4convection.io import satellite_io

TOLERANCE = 1e-6

# The following constants are used to test find_file and file_name_to_date.
TOP_DIRECTORY_NAME = 'stuff'
VALID_DATE_STRING = '20200820'
FILE_NAME = 'stuff/2020/satellite_20200820.nc'

# The following constants are used to test find_many_files.
FIRST_DATE_STRING = '20200818'
LAST_DATE_STRING = '20200821'

FILE_NAMES = [
    'stuff/2020/satellite_20200818.nc',
    'stuff/2020/satellite_20200819.nc',
    'stuff/2020/satellite_20200820.nc',
    'stuff/2020/satellite_20200821.nc'
]

# The following constants are used to test subset_by_band, subset_by_index,
# subset_by_time, and concat_data.
LATITUDES_DEG_N = numpy.linspace(53, 54, num=11, dtype=float)
LONGITUDES_DEG_E = numpy.linspace(246, 247, num=6, dtype=float)
VALID_TIMES_UNIX_SEC = numpy.linspace(0, 7, num=8, dtype=int)
BAND_NUMBERS = numpy.array([8, 9, 10], dtype=int)

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

SATELLITE_DICT_ALL_EXAMPLES = {
    satellite_io.BRIGHTNESS_TEMP_KEY: TEMPERATURE_MATRIX_KELVINS + 0.,
    satellite_io.BRIGHTNESS_COUNT_KEY: COUNT_MATRIX + 0.,
    satellite_io.LATITUDES_KEY: LATITUDES_DEG_N,
    satellite_io.LONGITUDES_KEY: LONGITUDES_DEG_E,
    satellite_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC + 0,
    satellite_io.BAND_NUMBERS_KEY: BAND_NUMBERS + 0
}

DESIRED_BAND_NUMBERS = numpy.array([10, 8], dtype=int)

SATELLITE_DICT_SUBSET_BY_BAND = {
    satellite_io.BRIGHTNESS_TEMP_KEY: TEMPERATURE_MATRIX_KELVINS[..., [2, 0]],
    satellite_io.BRIGHTNESS_COUNT_KEY: COUNT_MATRIX[..., [2, 0]],
    satellite_io.LATITUDES_KEY: LATITUDES_DEG_N,
    satellite_io.LONGITUDES_KEY: LONGITUDES_DEG_E,
    satellite_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC + 0,
    satellite_io.BAND_NUMBERS_KEY: BAND_NUMBERS[[2, 0]]
}

DESIRED_INDICES = numpy.array([3, 7, 1, 4], dtype=int)

SATELLITE_DICT_SUBSET_BY_INDEX = {
    satellite_io.BRIGHTNESS_TEMP_KEY:
        TEMPERATURE_MATRIX_KELVINS[[3, 7, 1, 4], ...],
    satellite_io.BRIGHTNESS_COUNT_KEY: COUNT_MATRIX[[3, 7, 1, 4], ...],
    satellite_io.LATITUDES_KEY: LATITUDES_DEG_N,
    satellite_io.LONGITUDES_KEY: LONGITUDES_DEG_E,
    satellite_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC[[3, 7, 1, 4]],
    satellite_io.BAND_NUMBERS_KEY: BAND_NUMBERS
}

DESIRED_TIMES_UNIX_SEC = numpy.array([2, 6, 0, 3], dtype=int)

SATELLITE_DICT_SUBSET_BY_TIME = {
    satellite_io.BRIGHTNESS_TEMP_KEY:
        TEMPERATURE_MATRIX_KELVINS[[2, 6, 0, 3], ...],
    satellite_io.BRIGHTNESS_COUNT_KEY: COUNT_MATRIX[[2, 6, 0, 3], ...],
    satellite_io.LATITUDES_KEY: LATITUDES_DEG_N,
    satellite_io.LONGITUDES_KEY: LONGITUDES_DEG_E,
    satellite_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC[[2, 6, 0, 3]],
    satellite_io.BAND_NUMBERS_KEY: BAND_NUMBERS
}

THESE_INDICES = numpy.array([3, 7, 1, 4, 2, 6, 0, 3], dtype=int)

SATELLITE_DICT_CONCAT = {
    satellite_io.BRIGHTNESS_TEMP_KEY:
        TEMPERATURE_MATRIX_KELVINS[THESE_INDICES, ...],
    satellite_io.BRIGHTNESS_COUNT_KEY: COUNT_MATRIX[THESE_INDICES, ...],
    satellite_io.LATITUDES_KEY: LATITUDES_DEG_N,
    satellite_io.LONGITUDES_KEY: LONGITUDES_DEG_E,
    satellite_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC[THESE_INDICES],
    satellite_io.BAND_NUMBERS_KEY: BAND_NUMBERS
}


def _compare_satellite_dicts(first_satellite_dict, second_satellite_dict):
    """Compares two dictionaries with radar data.

    :param first_satellite_dict: See doc for `satellite_io.read_file`.
    :param second_satellite_dict: Same.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = list(first_satellite_dict.keys())
    second_keys = list(second_satellite_dict.keys())
    if set(first_keys) != set(second_keys):
        return False

    float_keys = [
        satellite_io.BRIGHTNESS_TEMP_KEY, satellite_io.BRIGHTNESS_COUNT_KEY,
        satellite_io.LATITUDES_KEY, satellite_io.LONGITUDES_KEY
    ]
    integer_keys = [satellite_io.VALID_TIMES_KEY, satellite_io.BAND_NUMBERS_KEY]

    for this_key in float_keys:
        if (
                this_key == satellite_io.BRIGHTNESS_COUNT_KEY
                and first_satellite_dict[this_key] is None
                and second_satellite_dict[this_key] is None
        ):
            continue

        if not numpy.allclose(
                first_satellite_dict[this_key], second_satellite_dict[this_key],
                atol=TOLERANCE
        ):
            return False

    for this_key in integer_keys:
        if not numpy.array_equal(
                first_satellite_dict[this_key], second_satellite_dict[this_key]
        ):
            return False

    return True


class SatelliteIoTests(unittest.TestCase):
    """Each method is a unit test for satellite_io.py."""

    def test_find_file(self):
        """Ensures correct output from find_file."""

        this_file_name = satellite_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FILE_NAME)

    def test_find_many_files(self):
        """Ensures correct output from find_many_files."""

        these_file_names = satellite_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_date_string=FIRST_DATE_STRING,
            last_date_string=LAST_DATE_STRING, test_mode=True
        )

        self.assertTrue(these_file_names == FILE_NAMES)

    def test_file_name_to_date(self):
        """Ensures correct output from file_name_to_date."""

        self.assertTrue(
            satellite_io.file_name_to_date(FILE_NAME) == VALID_DATE_STRING
        )

    def test_subset_by_band(self):
        """Ensures correct output from subset_by_band."""

        this_satellite_dict = satellite_io.subset_by_band(
            satellite_dict=copy.deepcopy(SATELLITE_DICT_ALL_EXAMPLES),
            band_numbers=DESIRED_BAND_NUMBERS
        )

        self.assertTrue(_compare_satellite_dicts(
            this_satellite_dict, SATELLITE_DICT_SUBSET_BY_BAND
        ))

    def test_subset_by_index(self):
        """Ensures correct output from subset_by_index."""

        this_satellite_dict = satellite_io.subset_by_index(
            satellite_dict=copy.deepcopy(SATELLITE_DICT_ALL_EXAMPLES),
            desired_indices=DESIRED_INDICES
        )

        self.assertTrue(_compare_satellite_dicts(
            this_satellite_dict, SATELLITE_DICT_SUBSET_BY_INDEX
        ))

    def test_subset_by_time(self):
        """Ensures correct output from subset_by_time."""

        this_satellite_dict = satellite_io.subset_by_time(
            satellite_dict=copy.deepcopy(SATELLITE_DICT_ALL_EXAMPLES),
            desired_times_unix_sec=DESIRED_TIMES_UNIX_SEC
        )[0]

        self.assertTrue(_compare_satellite_dicts(
            this_satellite_dict, SATELLITE_DICT_SUBSET_BY_TIME
        ))

    def test_concat_data(self):
        """Ensures correct output from concat_data."""

        this_satellite_dict = satellite_io.concat_data(
            satellite_dicts=[
                SATELLITE_DICT_SUBSET_BY_INDEX, SATELLITE_DICT_SUBSET_BY_TIME
            ]
        )

        self.assertTrue(_compare_satellite_dicts(
            this_satellite_dict, SATELLITE_DICT_CONCAT
        ))


if __name__ == '__main__':
    unittest.main()
