"""Unit tests for twb_radar_io.py."""

import unittest
from gewittergefahr.gg_utils import time_conversion
from ml4convection.io import twb_radar_io

# The following constants are used to test find_file and file_name_to_time.
TOP_DIRECTORY_NAME = 'stuff'
VALID_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2020-08-20-22', '%Y-%m-%d-%H'
)

FILE_NAME_3D = 'stuff/20200820/MREF3D21L.20200820.2200.gz'
FILE_NAME_2D = 'stuff/20200820/compref_mosaic/COMPREF.20200820.2200.gz'

# The following constants are used to test find_many_files.
FIRST_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2020-08-20-235959', '%Y-%m-%d-%H%M%S'
)
LAST_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2020-08-21-001712', '%Y-%m-%d-%H%M%S'
)

FILE_NAMES_3D = [
    'stuff/20200820/MREF3D21L.20200820.2350.gz',
    'stuff/20200821/MREF3D21L.20200821.0000.gz',
    'stuff/20200821/MREF3D21L.20200821.0010.gz',
    'stuff/20200821/MREF3D21L.20200821.0020.gz'
]
FILE_NAMES_2D = [
    'stuff/20200820/compref_mosaic/COMPREF.20200820.2350.gz',
    'stuff/20200821/compref_mosaic/COMPREF.20200821.0000.gz',
    'stuff/20200821/compref_mosaic/COMPREF.20200821.0010.gz',
    'stuff/20200821/compref_mosaic/COMPREF.20200821.0020.gz'
]


class TwbRadarIoTests(unittest.TestCase):
    """Each method is a unit test for twb_radar_io.py."""

    def test_find_file_3d(self):
        """Ensures correct output from find_file.

        In this case, looking for file with 3-D data.
        """

        this_file_name = twb_radar_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            with_3d=True, raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FILE_NAME_3D)

    def test_find_file_2d(self):
        """Ensures correct output from find_file.

        In this case, looking for file with 2-D data.
        """

        this_file_name = twb_radar_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC,
            with_3d=False, raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FILE_NAME_2D)

    def test_find_many_files_3d(self):
        """Ensures correct output from find_many_files.

        In this case, looking for files with 3-D data.
        """

        these_file_names = twb_radar_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_TIME_UNIX_SEC,
            with_3d=True, test_mode=True
        )

        self.assertTrue(these_file_names == FILE_NAMES_3D)

    def test_find_many_files_2d(self):
        """Ensures correct output from find_many_files.

        In this case, looking for files with 2-D data.
        """

        these_file_names = twb_radar_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_TIME_UNIX_SEC,
            with_3d=False, test_mode=True
        )

        self.assertTrue(these_file_names == FILE_NAMES_2D)

    def test_file_name_to_time_3d(self):
        """Ensures correct output from file_name_to_time.

        In this case, file contains 3-D data.
        """

        self.assertTrue(
            twb_radar_io.file_name_to_time(FILE_NAME_3D) ==
            VALID_TIME_UNIX_SEC
        )

    def test_file_name_to_time_2d(self):
        """Ensures correct output from file_name_to_time.

        In this case, file contains 2-D data.
        """

        self.assertTrue(
            twb_radar_io.file_name_to_time(FILE_NAME_2D) ==
            VALID_TIME_UNIX_SEC
        )


if __name__ == '__main__':
    unittest.main()
