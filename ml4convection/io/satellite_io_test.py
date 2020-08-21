"""Unit tests for satellite_io.py."""

import unittest
from ml4convection.io import satellite_io

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


if __name__ == '__main__':
    unittest.main()
