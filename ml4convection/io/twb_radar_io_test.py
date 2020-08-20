"""Unit tests for twb_radar_io.py."""

import unittest
from gewittergefahr.gg_utils import time_conversion
from ml4convection.io import twb_radar_io

TOLERANCE = 1e-6

# The following constants are used to test find_file and file_name_to_time.
TOP_DIRECTORY_NAME = 'stuff'
VALID_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2020-08-20-22', '%Y-%m-%d-%H'
)

FILE_NAME_ZIPPED_3D = 'stuff/20200820/MREF3D21L.20200820.2200.gz'
FILE_NAME_UNZIPPED_3D = 'stuff/20200820/MREF3D21L.20200820.2200'
FILE_NAME_ZIPPED_2D = 'stuff/20200820/compref_mosaic/COMPREF.20200820.2200.gz'
FILE_NAME_UNZIPPED_2D = 'stuff/20200820/compref_mosaic/COMPREF.20200820.2200'

# The following constants are used to test find_many_files.
FIRST_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2020-08-20-235959', '%Y-%m-%d-%H%M%S'
)
LAST_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2020-08-21-000712', '%Y-%m-%d-%H%M%S'
)

FILE_NAMES_ZIPPED_3D = [
    'stuff/20200820/MREF3D21L.20200820.2355.gz',
    'stuff/20200821/MREF3D21L.20200821.0000.gz',
    'stuff/20200821/MREF3D21L.20200821.0005.gz',
    'stuff/20200821/MREF3D21L.20200821.0010.gz'
]
FILE_NAMES_UNZIPPED_3D = [
    'stuff/20200820/MREF3D21L.20200820.2355',
    'stuff/20200821/MREF3D21L.20200821.0000',
    'stuff/20200821/MREF3D21L.20200821.0005',
    'stuff/20200821/MREF3D21L.20200821.0010'
]
FILE_NAMES_ZIPPED_2D = [
    'stuff/20200820/compref_mosaic/COMPREF.20200820.2355.gz',
    'stuff/20200821/compref_mosaic/COMPREF.20200821.0000.gz',
    'stuff/20200821/compref_mosaic/COMPREF.20200821.0005.gz',
    'stuff/20200821/compref_mosaic/COMPREF.20200821.0010.gz'
]
FILE_NAMES_UNZIPPED_2D = [
    'stuff/20200820/compref_mosaic/COMPREF.20200820.2355',
    'stuff/20200821/compref_mosaic/COMPREF.20200821.0000',
    'stuff/20200821/compref_mosaic/COMPREF.20200821.0005',
    'stuff/20200821/compref_mosaic/COMPREF.20200821.0010'
]


class TwbRadarIoTests(unittest.TestCase):
    """Each method is a unit test for twb_radar_io.py."""

    def test_find_file_zipped_3d_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for zipped file with 3-D data but will allow
        unzipped file.
        """

        this_file_name = twb_radar_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC, with_3d=True,
            prefer_zipped=True, allow_other_format=True,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FILE_NAME_UNZIPPED_3D)

    def test_find_file_zipped_3d_no_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for zipped file with 3-D data and will *not* allow
        unzipped file.
        """

        this_file_name = twb_radar_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC, with_3d=True,
            prefer_zipped=True, allow_other_format=False,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FILE_NAME_ZIPPED_3D)

    def test_find_file_unzipped_3d_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for unzipped file with 3-D data but will allow
        zipped file.
        """

        this_file_name = twb_radar_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC, with_3d=True,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FILE_NAME_ZIPPED_3D)

    def test_find_file_unzipped_3d_no_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for unzipped file with 3-D data and will *not*
        allow zipped file.
        """

        this_file_name = twb_radar_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC, with_3d=True,
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FILE_NAME_UNZIPPED_3D)

    def test_find_file_zipped_2d_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for zipped file with 2-D data but will allow
        unzipped file.
        """

        this_file_name = twb_radar_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC, with_3d=False,
            prefer_zipped=True, allow_other_format=True,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FILE_NAME_UNZIPPED_2D)

    def test_find_file_zipped_2d_no_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for zipped file with 2-D data and will *not* allow
        unzipped file.
        """

        this_file_name = twb_radar_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC, with_3d=False,
            prefer_zipped=True, allow_other_format=False,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FILE_NAME_ZIPPED_2D)

    def test_find_file_unzipped_2d_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for unzipped file with 2-D data but will allow
        zipped file.
        """

        this_file_name = twb_radar_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC, with_3d=False,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FILE_NAME_ZIPPED_2D)

    def test_find_file_unzipped_2d_no_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for unzipped file with 2-D data and will *not*
        allow zipped file.
        """

        this_file_name = twb_radar_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_time_unix_sec=VALID_TIME_UNIX_SEC, with_3d=False,
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FILE_NAME_UNZIPPED_2D)

    def test_find_many_files_zipped_3d_allow(self):
        """Ensures correct output from find_many_files.

        In this case, looking for zipped files with 3-D data but will allow
        unzipped files.
        """

        these_file_names = twb_radar_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_TIME_UNIX_SEC,
            with_3d=True, prefer_zipped=True, allow_other_format=True,
            test_mode=True
        )

        self.assertTrue(these_file_names == FILE_NAMES_UNZIPPED_3D)

    def test_find_many_files_zipped_3d_no_allow(self):
        """Ensures correct output from find_many_files.

        In this case, looking for zipped files with 3-D data and will *not*
        allow unzipped files.
        """

        these_file_names = twb_radar_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_TIME_UNIX_SEC,
            with_3d=True, prefer_zipped=True, allow_other_format=False,
            test_mode=True
        )

        self.assertTrue(these_file_names == FILE_NAMES_ZIPPED_3D)

    def test_find_many_files_unzipped_3d_allow(self):
        """Ensures correct output from find_many_files.

        In this case, looking for unzipped files with 3-D data but will allow
        zipped files.
        """

        these_file_names = twb_radar_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_TIME_UNIX_SEC,
            with_3d=True, prefer_zipped=False, allow_other_format=True,
            test_mode=True
        )

        self.assertTrue(these_file_names == FILE_NAMES_ZIPPED_3D)

    def test_find_many_files_unzipped_3d_no_allow(self):
        """Ensures correct output from find_many_files.

        In this case, looking for unzipped files with 3-D data and will *not*
        allow zipped files.
        """

        these_file_names = twb_radar_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_TIME_UNIX_SEC,
            with_3d=True, prefer_zipped=False, allow_other_format=False,
            test_mode=True
        )

        self.assertTrue(these_file_names == FILE_NAMES_UNZIPPED_3D)

    def test_find_many_files_zipped_2d_allow(self):
        """Ensures correct output from find_many_files.

        In this case, looking for zipped files with 2-D data but will allow
        unzipped files.
        """

        these_file_names = twb_radar_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_TIME_UNIX_SEC,
            with_3d=False, prefer_zipped=True, allow_other_format=True,
            test_mode=True
        )

        self.assertTrue(these_file_names == FILE_NAMES_UNZIPPED_2D)

    def test_find_many_files_zipped_2d_no_allow(self):
        """Ensures correct output from find_many_files.

        In this case, looking for zipped files with 2-D data and will *not*
        allow unzipped files.
        """

        these_file_names = twb_radar_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_TIME_UNIX_SEC,
            with_3d=False, prefer_zipped=True, allow_other_format=False,
            test_mode=True
        )

        self.assertTrue(these_file_names == FILE_NAMES_ZIPPED_2D)

    def test_find_many_files_unzipped_2d_allow(self):
        """Ensures correct output from find_many_files.

        In this case, looking for unzipped files with 2-D data but will allow
        zipped files.
        """

        these_file_names = twb_radar_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_TIME_UNIX_SEC,
            with_3d=False, prefer_zipped=False, allow_other_format=True,
            test_mode=True
        )

        self.assertTrue(these_file_names == FILE_NAMES_ZIPPED_2D)

    def test_find_many_files_unzipped_2d_no_allow(self):
        """Ensures correct output from find_many_files.

        In this case, looking for unzipped files with 2-D data and will *not*
        allow zipped files.
        """

        these_file_names = twb_radar_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_TIME_UNIX_SEC,
            with_3d=False, prefer_zipped=False, allow_other_format=False,
            test_mode=True
        )

        self.assertTrue(these_file_names == FILE_NAMES_UNZIPPED_2D)

    def test_file_name_to_time_zipped_3d(self):
        """Ensures correct output from file_name_to_time.

        In this case, file is zipped and contains 3-D data.
        """

        self.assertTrue(
            twb_radar_io.file_name_to_time(FILE_NAME_ZIPPED_3D) ==
            VALID_TIME_UNIX_SEC
        )

    def test_file_name_to_time_unzipped_3d(self):
        """Ensures correct output from file_name_to_time.

        In this case, file is unzipped and contains 3-D data.
        """

        self.assertTrue(
            twb_radar_io.file_name_to_time(FILE_NAME_UNZIPPED_3D) ==
            VALID_TIME_UNIX_SEC
        )

    def test_file_name_to_time_zipped_2d(self):
        """Ensures correct output from file_name_to_time.

        In this case, file is zipped and contains 2-D data.
        """

        self.assertTrue(
            twb_radar_io.file_name_to_time(FILE_NAME_ZIPPED_2D) ==
            VALID_TIME_UNIX_SEC
        )

    def test_file_name_to_time_unzipped_2d(self):
        """Ensures correct output from file_name_to_time.

        In this case, file is unzipped and contains 2-D data.
        """

        self.assertTrue(
            twb_radar_io.file_name_to_time(FILE_NAME_UNZIPPED_2D) ==
            VALID_TIME_UNIX_SEC
        )


if __name__ == '__main__':
    unittest.main()
