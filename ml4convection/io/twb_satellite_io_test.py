"""Unit tests for twb_satellite_io.py."""

import unittest
from gewittergefahr.gg_utils import time_conversion
from ml4convection.io import twb_satellite_io

# The following constants are used to test find_file, file_name_to_time, and
# file_name_to_band_number.
TOP_DIRECTORY_NAME = 'stuff'
VALID_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2020-08-20-22', '%Y-%m-%d-%H'
)
BAND_NUMBER = 8

FILE_NAME = 'stuff/2020-08/2020-08-20_2200.B08.GSD.Cnt'

# The following constants are used to test find_many_files.
FIRST_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2020-08-20-235959', '%Y-%m-%d-%H%M%S'
)
LAST_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2020-08-21-001712', '%Y-%m-%d-%H%M%S'
)

FILE_NAMES = [
    'stuff/2020-08/2020-08-20_2350.B08.GSD.Cnt',
    'stuff/2020-08/2020-08-21_0000.B08.GSD.Cnt',
    'stuff/2020-08/2020-08-21_0010.B08.GSD.Cnt',
    'stuff/2020-08/2020-08-21_0020.B08.GSD.Cnt'
]


class TwbSatelliteIoTests(unittest.TestCase):
    """Each method is a unit test for twb_satellite_io.py."""

    def test_find_file(self):
        """Ensures correct output from find_file."""

        this_file_name = twb_satellite_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC, band_number=BAND_NUMBER,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FILE_NAME)

    def test_find_many_files(self):
        """Ensures correct output from find_many_files."""

        these_file_names = twb_satellite_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_TIME_UNIX_SEC,
            band_number=BAND_NUMBER, test_mode=True
        )

        self.assertTrue(these_file_names == FILE_NAMES)

    def test_file_name_to_time(self):
        """Ensures correct output from file_name_to_time."""

        self.assertTrue(
            twb_satellite_io.file_name_to_time(FILE_NAME) ==
            VALID_TIME_UNIX_SEC
        )

    def test_file_name_to_band_number(self):
        """Ensures correct output from file_name_to_band_number."""

        self.assertTrue(
            twb_satellite_io.file_name_to_band_number(FILE_NAME) ==
            BAND_NUMBER
        )


if __name__ == '__main__':
    unittest.main()
