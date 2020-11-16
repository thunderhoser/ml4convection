"""Unit tests for prediction_io.py."""

import unittest
from ml4convection.io import prediction_io

# The following constants are used to test find_file, file_name_to_date, and
# file_name_to_radar_number.
TOP_DIRECTORY_NAME = 'foo'
VALID_DATE_STRING = '19670502'
RADAR_NUMBER = 1

PREDICTION_FILE_NAME_FULL_UNZIPPED = 'foo/1967/predictions_19670502.nc'
PREDICTION_FILE_NAME_FULL_ZIPPED = 'foo/1967/predictions_19670502.nc.gz'
PREDICTION_FILE_NAME_PARTIAL_UNZIPPED = (
    'foo/1967/predictions_19670502_radar1.nc'
)
PREDICTION_FILE_NAME_PARTIAL_ZIPPED = (
    'foo/1967/predictions_19670502_radar1.nc.gz'
)

# The following constants are used to test find_many_files.
FIRST_DATE_STRING = '19670501'
LAST_DATE_STRING = '19670504'

PREDICTION_FILE_NAMES_UNZIPPED = [
    'foo/1967/predictions_19670501.nc',
    'foo/1967/predictions_19670502.nc',
    'foo/1967/predictions_19670503.nc',
    'foo/1967/predictions_19670504.nc'
]

PREDICTION_FILE_NAMES_ZIPPED = [
    'foo/1967/predictions_19670501.nc.gz',
    'foo/1967/predictions_19670502.nc.gz',
    'foo/1967/predictions_19670503.nc.gz',
    'foo/1967/predictions_19670504.nc.gz'
]


class PredictionIoTests(unittest.TestCase):
    """Each method is a unit test for prediction_io.py."""

    def test_find_file_full_zipped_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for zipped full-grid file but will allow unzipped
        file.
        """

        this_file_name = prediction_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING, radar_number=None,
            prefer_zipped=True, allow_other_format=True,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == PREDICTION_FILE_NAME_FULL_UNZIPPED)

    def test_find_file_full_zipped_strict(self):
        """Ensures correct output from find_file.

        In this case, looking for zipped full-grid file and will *not* allow
        unzipped file.
        """

        this_file_name = prediction_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING, radar_number=None,
            prefer_zipped=True, allow_other_format=False,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == PREDICTION_FILE_NAME_FULL_ZIPPED)

    def test_find_file_full_unzipped_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for unzipped full-grid file but will allow zipped
        file.
        """

        this_file_name = prediction_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING, radar_number=None,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == PREDICTION_FILE_NAME_FULL_ZIPPED)

    def test_find_file_full_unzipped_strict(self):
        """Ensures correct output from find_file.

        In this case, looking for unzipped full-grid file and will *not* allow
        zipped file.
        """

        this_file_name = prediction_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING, radar_number=None,
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == PREDICTION_FILE_NAME_FULL_UNZIPPED)

    def test_find_file_partial_zipped_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for zipped partial-grid file but will allow
        unzipped file.
        """

        this_file_name = prediction_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING, radar_number=RADAR_NUMBER,
            prefer_zipped=True, allow_other_format=True,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == PREDICTION_FILE_NAME_PARTIAL_UNZIPPED)

    def test_find_file_partial_zipped_strict(self):
        """Ensures correct output from find_file.

        In this case, looking for zipped partial-grid file and will *not* allow
        unzipped file.
        """

        this_file_name = prediction_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING, radar_number=RADAR_NUMBER,
            prefer_zipped=True, allow_other_format=False,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == PREDICTION_FILE_NAME_PARTIAL_ZIPPED)

    def test_find_file_partial_unzipped_allow(self):
        """Ensures correct output from find_file.

        In this case, looking for unzipped partial-grid file but will allow
        zipped file.
        """

        this_file_name = prediction_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING, radar_number=RADAR_NUMBER,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == PREDICTION_FILE_NAME_PARTIAL_ZIPPED)

    def test_find_file_partial_unzipped_strict(self):
        """Ensures correct output from find_file.

        In this case, looking for unzipped partial-grid file and will *not*
        allow zipped file.
        """

        this_file_name = prediction_io.find_file(
            top_directory_name=TOP_DIRECTORY_NAME,
            valid_date_string=VALID_DATE_STRING, radar_number=RADAR_NUMBER,
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == PREDICTION_FILE_NAME_PARTIAL_UNZIPPED)

    def test_find_many_files_full_zipped_allow(self):
        """Ensures correct output from find_many_files.

        In this case, looking for zipped full-grid files but will allow unzipped
        files.
        """

        these_file_names = prediction_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_date_string=FIRST_DATE_STRING,
            last_date_string=LAST_DATE_STRING,
            radar_number=None,
            prefer_zipped=True, allow_other_format=True, test_mode=True
        )

        self.assertTrue(these_file_names == PREDICTION_FILE_NAMES_UNZIPPED)

    def test_find_many_files_full_zipped_strict(self):
        """Ensures correct output from find_many_files.

        In this case, looking for zipped full-grid files and will *not* allow
        unzipped files.
        """

        these_file_names = prediction_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_date_string=FIRST_DATE_STRING,
            last_date_string=LAST_DATE_STRING,
            radar_number=None,
            prefer_zipped=True, allow_other_format=False, test_mode=True
        )

        self.assertTrue(these_file_names == PREDICTION_FILE_NAMES_ZIPPED)

    def test_find_many_files_full_unzipped_allow(self):
        """Ensures correct output from find_many_files.

        In this case, looking for unzipped full-grid files but will allow zipped
        files.
        """

        these_file_names = prediction_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_date_string=FIRST_DATE_STRING,
            last_date_string=LAST_DATE_STRING,
            radar_number=None,
            prefer_zipped=False, allow_other_format=True, test_mode=True
        )

        self.assertTrue(these_file_names == PREDICTION_FILE_NAMES_ZIPPED)

    def test_find_many_files_full_unzipped_strict(self):
        """Ensures correct output from find_many_files.

        In this case, looking for unzipped full-grid files and will *not* allow
        zipped files.
        """

        these_file_names = prediction_io.find_many_files(
            top_directory_name=TOP_DIRECTORY_NAME,
            first_date_string=FIRST_DATE_STRING,
            last_date_string=LAST_DATE_STRING,
            radar_number=None,
            prefer_zipped=False, allow_other_format=False, test_mode=True
        )

        self.assertTrue(these_file_names == PREDICTION_FILE_NAMES_UNZIPPED)

    def test_file_name_to_date_full_zipped(self):
        """Ensures correct output from file_name_to_date.

        In this case, file contains full grid and is zipped.
        """

        self.assertTrue(
            prediction_io.file_name_to_date(PREDICTION_FILE_NAME_FULL_ZIPPED)
            == VALID_DATE_STRING
        )

    def test_file_name_to_date_full_unzipped(self):
        """Ensures correct output from file_name_to_date.

        In this case, file contains full grid and is unzipped.
        """

        self.assertTrue(
            prediction_io.file_name_to_date(PREDICTION_FILE_NAME_FULL_UNZIPPED)
            == VALID_DATE_STRING
        )

    def test_file_name_to_date_partial_zipped(self):
        """Ensures correct output from file_name_to_date.

        In this case, file contains partial grid and is zipped.
        """

        self.assertTrue(
            prediction_io.file_name_to_date(PREDICTION_FILE_NAME_PARTIAL_ZIPPED)
            == VALID_DATE_STRING
        )

    def test_file_name_to_date_partial_unzipped(self):
        """Ensures correct output from file_name_to_date.

        In this case, file contains partial grid and is unzipped.
        """

        self.assertTrue(
            prediction_io.file_name_to_date(
                PREDICTION_FILE_NAME_PARTIAL_UNZIPPED
            )
            == VALID_DATE_STRING
        )

    def test_file_name_to_radar_number_full_zipped(self):
        """Ensures correct output from file_name_to_radar_number.

        In this case, file contains full grid and is zipped.
        """

        self.assertTrue(
            prediction_io.file_name_to_radar_number(
                PREDICTION_FILE_NAME_FULL_ZIPPED
            ) is None
        )

    def test_file_name_to_radar_number_full_unzipped(self):
        """Ensures correct output from file_name_to_radar_number.

        In this case, file contains full grid and is unzipped.
        """

        self.assertTrue(
            prediction_io.file_name_to_radar_number(
                PREDICTION_FILE_NAME_FULL_UNZIPPED
            ) is None
        )

    def test_file_name_to_radar_number_partial_zipped(self):
        """Ensures correct output from file_name_to_radar_number.

        In this case, file contains partial grid and is zipped.
        """

        self.assertTrue(
            prediction_io.file_name_to_radar_number(
                PREDICTION_FILE_NAME_PARTIAL_ZIPPED
            )
            == RADAR_NUMBER
        )

    def test_file_name_to_radar_number_partial_unzipped(self):
        """Ensures correct output from file_name_to_radar_number.

        In this case, file contains partial grid and is unzipped.
        """

        self.assertTrue(
            prediction_io.file_name_to_radar_number(
                PREDICTION_FILE_NAME_PARTIAL_UNZIPPED
            )
            == RADAR_NUMBER
        )


if __name__ == '__main__':
    unittest.main()
