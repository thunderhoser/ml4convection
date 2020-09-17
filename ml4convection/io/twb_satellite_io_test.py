"""Unit tests for twb_satellite_io.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml4convection.io import twb_satellite_io

TOLERANCE = 1e-6

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

# The following constants are used to test count_to_temperature.
MISSING_COUNT = twb_satellite_io.MISSING_COUNT

BRIGHTNESS_COUNT_MATRIX = numpy.array([
    [0, 5, 10, 15, 20, 25],
    [100, 105, 110, 115, 120, 125],
    [200, 205, 210, 215, 220, 225],
    [MISSING_COUNT, 8, 4, 6, 2, MISSING_COUNT]
], dtype=int)

BRIGHTNESS_TEMP_MATRIX_BAND8_KELVINS = numpy.array([
    [300, 284.28, 279.78, 278.08, 276.37, 274.67],
    [248.84, 247.09, 245.34, 243.59, 241.84, 240.14],
    [214.19, 212.44, 210.69, 208.94, 207.19, 205.44],
    [numpy.nan, 280.46, 287.43, 281.14, 293.71, numpy.nan]
])

BRIGHTNESS_TEMP_MATRIX_BAND13_KELVINS = numpy.array([
    [330, 328.26, 326.17, 324.07, 321.98, 319.88],
    [283.15, 279.58, 276, 272.54, 269.52, 266.5],
    [217.91, 214.52, 211.14, 207.69, 203.94, 200.74],
    [numpy.nan, 327.01, 328.68, 327.84, 329.52, numpy.nan]
])


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

    def test_count_to_temperature_band8(self):
        """Ensures correct output from count_to_temperature.

        In this case, band number is 8.
        """

        this_temp_matrix_kelvins = twb_satellite_io.count_to_temperature(
            brightness_counts=BRIGHTNESS_COUNT_MATRIX, band_number=8
        )

        self.assertTrue(numpy.allclose(
            this_temp_matrix_kelvins, BRIGHTNESS_TEMP_MATRIX_BAND8_KELVINS,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_count_to_temperature_band13(self):
        """Ensures correct output from count_to_temperature.

        In this case, band number is 13.
        """

        this_temp_matrix_kelvins = twb_satellite_io.count_to_temperature(
            brightness_counts=BRIGHTNESS_COUNT_MATRIX, band_number=13
        )

        self.assertTrue(numpy.allclose(
            this_temp_matrix_kelvins, BRIGHTNESS_TEMP_MATRIX_BAND13_KELVINS,
            atol=TOLERANCE, equal_nan=True
        ))


if __name__ == '__main__':
    unittest.main()
