"""Unit tests for radar_io.py."""

import copy
import unittest
import numpy
from ml4convection.io import radar_io

TOLERANCE = 1e-6

# The following constants are used to test find_file and file_name_to_date.
TOP_DIRECTORY_NAME = 'stuff'
VALID_DATE_STRING = '20200820'

FILE_NAME_UNZIPPED = 'stuff/2020/radar_20200820.nc'
FILE_NAME_ZIPPED = 'stuff/2020/radar_20200820.nc.gz'

# The following constants are used to test find_many_files.
FIRST_DATE_STRING = '20200818'
LAST_DATE_STRING = '20200821'

FILE_NAMES_UNZIPPED = [
    'stuff/2020/radar_20200818.nc',
    'stuff/2020/radar_20200819.nc',
    'stuff/2020/radar_20200820.nc',
    'stuff/2020/radar_20200821.nc'
]
FILE_NAMES_ZIPPED = [
    'stuff/2020/radar_20200818.nc.gz',
    'stuff/2020/radar_20200819.nc.gz',
    'stuff/2020/radar_20200820.nc.gz',
    'stuff/2020/radar_20200821.nc.gz'
]

# The following constants are used to test subset_by_index and subset_by_time.
LATITUDES_DEG_N = numpy.array([53, 53.2, 53.4, 53.6, 53.8, 54])
LONGITUDES_DEG_E = numpy.array([246, 246.5, 247])
HEIGHTS_M_ASL = numpy.array([1000, 5000, 3000], dtype=float)
VALID_TIMES_UNIX_SEC = numpy.array([600, 1200, 1800, 2400], dtype=int)

REFL_MATRIX_EXAMPLE1_DBZ = numpy.array([
    [10, 20, 30],
    [20, 30, 40],
    [30, 40, 50],
    [40, 50, 60],
    [50, 60, 70],
    [0, 0, 75]
], dtype=float)

REFL_MATRIX_HEIGHT1_DBZ = numpy.stack((
    REFL_MATRIX_EXAMPLE1_DBZ, REFL_MATRIX_EXAMPLE1_DBZ + 1,
    REFL_MATRIX_EXAMPLE1_DBZ + 2, REFL_MATRIX_EXAMPLE1_DBZ + 3
), axis=0)

REFLECTIVITY_MATRIX_DBZ = numpy.stack((
    REFL_MATRIX_HEIGHT1_DBZ, REFL_MATRIX_HEIGHT1_DBZ - 20,
    REFL_MATRIX_HEIGHT1_DBZ - 10
), axis=-1)

RADAR_DICT_ALL_EXAMPLES = {
    radar_io.REFLECTIVITY_KEY: REFLECTIVITY_MATRIX_DBZ + 0.,
    radar_io.LATITUDES_KEY: LATITUDES_DEG_N,
    radar_io.LONGITUDES_KEY: LONGITUDES_DEG_E,
    radar_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC + 0,
    radar_io.HEIGHTS_KEY: HEIGHTS_M_ASL
}

DESIRED_TIMES_UNIX_SEC = numpy.array([1800, 600], dtype=int)

RADAR_DICT_SUBSET_BY_TIME = {
    radar_io.REFLECTIVITY_KEY: REFLECTIVITY_MATRIX_DBZ[[2, 0], ...],
    radar_io.LATITUDES_KEY: LATITUDES_DEG_N,
    radar_io.LONGITUDES_KEY: LONGITUDES_DEG_E,
    radar_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC[[2, 0]],
    radar_io.HEIGHTS_KEY: HEIGHTS_M_ASL
}

DESIRED_INDICES = numpy.array([3, 1], dtype=int)

RADAR_DICT_SUBSET_BY_INDEX = {
    radar_io.REFLECTIVITY_KEY: REFLECTIVITY_MATRIX_DBZ[[3, 1], ...],
    radar_io.LATITUDES_KEY: LATITUDES_DEG_N,
    radar_io.LONGITUDES_KEY: LONGITUDES_DEG_E,
    radar_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC[[3, 1]],
    radar_io.HEIGHTS_KEY: HEIGHTS_M_ASL
}


def compare_radar_dicts(first_radar_dict, second_radar_dict):
    """Compares two dictionaries with radar data.

    :param first_radar_dict: See doc for `radar_io.read_file`.
    :param second_radar_dict: Same.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = list(first_radar_dict.keys())
    second_keys = list(second_radar_dict.keys())
    if set(first_keys) != set(second_keys):
        return False

    float_keys = [
        radar_io.REFLECTIVITY_KEY, radar_io.LATITUDES_KEY,
        radar_io.LONGITUDES_KEY, radar_io.HEIGHTS_KEY
    ]
    integer_keys = [radar_io.VALID_TIMES_KEY]

    for this_key in float_keys:
        if not numpy.allclose(
                first_radar_dict[this_key], second_radar_dict[this_key],
                atol=TOLERANCE
        ):
            return False

    for this_key in integer_keys:
        if not numpy.array_equal(
                first_radar_dict[this_key], second_radar_dict[this_key]
        ):
            return False

    return True


class RadarIoTests(unittest.TestCase):
    """Each method is a unit test for radar_io.py."""

    def test_find_file_zipped_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for zipped file but will allow unzipped file.
        """

        this_file_name = radar_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING,
            prefer_zipped=True, allow_other_format=True,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FILE_NAME_UNZIPPED)

    def test_find_file_zipped_no_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for zipped file and will *not* allow unzipped
        file.
        """

        this_file_name = radar_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING,
            prefer_zipped=True, allow_other_format=False,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FILE_NAME_ZIPPED)

    def test_find_file_unzipped_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for unzipped file but will allow zipped file.
        """

        this_file_name = radar_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FILE_NAME_ZIPPED)

    def test_find_file_unzipped_no_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for unzipped file and will *not* allow zipped
        file.
        """

        this_file_name = radar_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING,
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FILE_NAME_UNZIPPED)

    def test_find_many_files_zipped_allow(self):
        """Ensures correct output from find_many_files.

        In this case, looking for zipped files but will allow unzipped files.
        """

        these_file_names = radar_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_date_string=FIRST_DATE_STRING,
            last_date_string=LAST_DATE_STRING,
            prefer_zipped=True, allow_other_format=True, test_mode=True
        )

        self.assertTrue(these_file_names == FILE_NAMES_UNZIPPED)

    def test_find_many_files_zipped_no_allow(self):
        """Ensures correct output from find_many_files.

        In this case, looking for zipped files and will *not* allow unzipped
        files.
        """

        these_file_names = radar_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_date_string=FIRST_DATE_STRING,
            last_date_string=LAST_DATE_STRING,
            prefer_zipped=True, allow_other_format=False, test_mode=True
        )

        self.assertTrue(these_file_names == FILE_NAMES_ZIPPED)

    def test_find_many_files_unzipped_allow(self):
        """Ensures correct output from find_many_files.

        In this case, looking for unzipped files but will allow zipped files.
        """

        these_file_names = radar_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_date_string=FIRST_DATE_STRING,
            last_date_string=LAST_DATE_STRING,
            prefer_zipped=False, allow_other_format=True, test_mode=True
        )

        self.assertTrue(these_file_names == FILE_NAMES_ZIPPED)

    def test_find_many_files_unzipped_no_allow(self):
        """Ensures correct output from find_many_files.

        In this case, looking for unzipped files and will *not* allow zipped
        files.
        """

        these_file_names = radar_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_date_string=FIRST_DATE_STRING,
            last_date_string=LAST_DATE_STRING,
            prefer_zipped=False, allow_other_format=False, test_mode=True
        )

        self.assertTrue(these_file_names == FILE_NAMES_UNZIPPED)

    def test_file_name_to_date_zipped(self):
        """Ensures correct output from file_name_to_date.

        In this case, file is zipped.
        """

        self.assertTrue(
            radar_io.file_name_to_date(FILE_NAME_ZIPPED) == VALID_DATE_STRING
        )

    def test_file_name_to_date_unzipped(self):
        """Ensures correct output from file_name_to_date.

        In this case, file is unzipped.
        """

        self.assertTrue(
            radar_io.file_name_to_date(FILE_NAME_UNZIPPED) == VALID_DATE_STRING
        )

    def test_subset_by_index(self):
        """Ensures correct output from subset_by_index."""

        this_radar_dict = radar_io.subset_by_index(
            radar_dict=copy.deepcopy(RADAR_DICT_ALL_EXAMPLES),
            desired_indices=DESIRED_INDICES
        )

        self.assertTrue(compare_radar_dicts(
            this_radar_dict, RADAR_DICT_SUBSET_BY_INDEX
        ))

    def test_subset_by_time(self):
        """Ensures correct output from subset_by_time."""

        this_radar_dict = radar_io.subset_by_time(
            radar_dict=copy.deepcopy(RADAR_DICT_ALL_EXAMPLES),
            desired_times_unix_sec=DESIRED_TIMES_UNIX_SEC
        )[0]

        self.assertTrue(compare_radar_dicts(
            this_radar_dict, RADAR_DICT_SUBSET_BY_TIME
        ))


if __name__ == '__main__':
    unittest.main()
