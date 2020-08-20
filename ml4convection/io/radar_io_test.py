"""Unit tests for radar_io.py."""

import unittest
from ml4convection.io import radar_io

# The following constants are used to test find_file and file_name_to_date.
TOP_DIRECTORY_NAME = 'stuff'
VALID_DATE_STRING = '20200820'
FILE_NAME_3D = 'stuff/2020/3d-radar_20200820.nc'
FILE_NAME_2D = 'stuff/2020/2d-radar_20200820.nc'

# The following constants are used to test find_many_files.
FIRST_DATE_STRING = '20200818'
LAST_DATE_STRING = '20200821'

FILE_NAMES_3D = [
    'stuff/2020/3d-radar_20200818.nc',
    'stuff/2020/3d-radar_20200819.nc',
    'stuff/2020/3d-radar_20200820.nc',
    'stuff/2020/3d-radar_20200821.nc'
]
FILE_NAMES_2D = [
    'stuff/2020/2d-radar_20200818.nc',
    'stuff/2020/2d-radar_20200819.nc',
    'stuff/2020/2d-radar_20200820.nc',
    'stuff/2020/2d-radar_20200821.nc'
]


class RadarIoTests(unittest.TestCase):
    """Each method is a unit test for radar_io.py."""

    def test_find_file_3d(self):
        """Ensures correct output from find_file (looking for 3-D file here)."""

        this_file_name = radar_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING,
            with_3d=True, raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FILE_NAME_3D)

    def test_find_file_2d(self):
        """Ensures correct output from find_file (looking for 2-D file here)."""

        this_file_name = radar_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING,
            with_3d=False, raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FILE_NAME_2D)

    def test_find_many_files_3d(self):
        """Ensures correct output from find_many_files.

        In this case, looking for 3-D files.
        """

        these_file_names = radar_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_date_string=FIRST_DATE_STRING,
            last_date_string=LAST_DATE_STRING,
            with_3d=True, test_mode=True
        )

        self.assertTrue(these_file_names == FILE_NAMES_3D)

    def test_find_many_files_2d(self):
        """Ensures correct output from find_many_files.

        In this case, looking for 2-D files.
        """

        these_file_names = radar_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_date_string=FIRST_DATE_STRING,
            last_date_string=LAST_DATE_STRING,
            with_3d=False, test_mode=True
        )

        self.assertTrue(these_file_names == FILE_NAMES_2D)

    def test_file_name_to_date_3d(self):
        """Ensures correct output from file_name_to_date.

        In this case, file contains 3-D data.
        """

        self.assertTrue(
            radar_io.file_name_to_date(FILE_NAME_3D) == VALID_DATE_STRING
        )

    def test_file_name_to_date_2d(self):
        """Ensures correct output from file_name_to_date.

        In this case, file contains 2-D data.
        """

        self.assertTrue(
            radar_io.file_name_to_date(FILE_NAME_2D) == VALID_DATE_STRING
        )


if __name__ == '__main__':
    unittest.main()
