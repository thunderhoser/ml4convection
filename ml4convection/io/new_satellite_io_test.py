"""Unit tests for new_satellite_io.py."""

import unittest
from ml4convection.io import new_satellite_io

TOLERANCE = 1e-6

# The following constants are used to test get_cyclone_id, parse_cyclone_id,
# find_file, and file_name_to_cyclone_id.
YEAR = 1998
BASIN_ID_STRING = 'AL'
CYCLONE_NUMBER = 5
CYCLONE_ID_STRING = '1998AL05'

TOP_DIRECTORY_NAME = 'foo'
VALID_TIME_UNIX_SEC = 907411500
SATELLITE_FILE_NAME = 'foo/1998/1998AL05/AL0598_19982761045M.nc'


class NewSatelliteIoTests(unittest.TestCase):
    """Each method is a unit test for new_satellite_io.py."""

    def test_get_cyclone_id(self):
        """Ensures correct output from get_cyclone_id."""

        this_id_string = new_satellite_io.get_cyclone_id(
            year=YEAR, basin_id_string=BASIN_ID_STRING,
            cyclone_number=CYCLONE_NUMBER
        )

        self.assertTrue(this_id_string == CYCLONE_ID_STRING)

    def test_parse_cyclone_id(self):
        """Ensures correct output from parse_cyclone_id."""

        this_year, this_basin_id_string, this_cyclone_number = (
            new_satellite_io.parse_cyclone_id(CYCLONE_ID_STRING)
        )

        self.assertTrue(this_year == YEAR)
        self.assertTrue(this_basin_id_string == BASIN_ID_STRING)
        self.assertTrue(this_cyclone_number == CYCLONE_NUMBER)

    def test_find_file(self):
        """Ensures correct output from find_file."""

        this_file_name = new_satellite_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            cyclone_id_string=CYCLONE_ID_STRING,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == SATELLITE_FILE_NAME)

    def test_file_name_to_cyclone_id(self):
        """Ensures correct output from file_name_to_cyclone_id."""

        this_id_string = new_satellite_io.file_name_to_cyclone_id(
            SATELLITE_FILE_NAME
        )

        self.assertTrue(this_id_string == CYCLONE_ID_STRING)


if __name__ == '__main__':
    unittest.main()
