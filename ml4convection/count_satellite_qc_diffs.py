"""Counts changes made by satellite QC at each grid cell."""

import os
import sys
import argparse
import numpy
import netCDF4

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import satellite_io

TOLERANCE = 1e-6
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

ROW_DIMENSION_KEY = 'row'
COLUMN_DIMENSION_KEY = 'column'
BAND_DIMENSION_KEY = 'band'

LATITUDES_KEY = 'latitudes_deg_n'
LONGITUDES_KEY = 'longitudes_deg_e'
BAND_NUMBERS_KEY = 'band_numbers'
DIFFERENCE_COUNT_KEY = 'count_matrix'

BEFORE_QC_DIR_ARG_NAME = 'input_before_qc_dir_name'
AFTER_QC_DIR_ARG_NAME = 'input_after_qc_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

BEFORE_QC_DIR_HELP_STRING = (
    'Name of directory with satellite data before QC.  Files therein will be '
    'found by `satellite_io.find_file` and read by `satellite_io.read_file`.'
)
AFTER_QC_DIR_HELP_STRING = 'Same as `{0:s}` but with data after QC.'.format(
    BEFORE_QC_DIR_ARG_NAME
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will count differences for the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file (NetCDF).  Results will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + BEFORE_QC_DIR_ARG_NAME, type=str, required=True,
    help=BEFORE_QC_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + AFTER_QC_DIR_ARG_NAME, type=str, required=True,
    help=AFTER_QC_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _increment_counts_one_day(
        before_qc_file_name, after_qc_file_name, count_matrix,
        expected_latitudes_deg_n, expected_longitudes_deg_e,
        expected_band_numbers):
    """Increments difference count at each grid cell, based on one day of data.

    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (spectral bands)

    :param before_qc_file_name: Path to file with satellite data before QC (will
        be read by `satellite_io.read_file`).
    :param after_qc_file_name: Same but with satellite data after QC.
    :param count_matrix: M-by-N-by-C numpy array of difference counts.  If
        None, a new one will be created.
    :param expected_latitudes_deg_n: length-M numpy array of expected latitudes
        (deg N).  If None, latitudes will not be verified.
    :param expected_longitudes_deg_e: length-N numpy array of expected
        longitudes (deg E).  May be None.
    :param expected_band_numbers: length-C numpy array of expected band numbers
        (integers).  May be None.
    :return: count_matrix: Updated version of input array.
    :return: latitudes_deg_n: Same as input but read from new file.
    :return: longitudes_deg_e: Same as input but read from new file.
    :return: band_numbers: Same as input but read from new file.
    """

    print('Reading data from: "{0:s}"...'.format(before_qc_file_name))
    before_qc_dict = satellite_io.read_file(
        netcdf_file_name=before_qc_file_name, fill_nans=False
    )

    print('Reading data from: "{0:s}"...'.format(after_qc_file_name))
    after_qc_dict = satellite_io.read_file(
        netcdf_file_name=after_qc_file_name, fill_nans=False
    )

    assert numpy.allclose(
        before_qc_dict[satellite_io.LATITUDES_KEY],
        after_qc_dict[satellite_io.LATITUDES_KEY],
        atol=TOLERANCE
    )
    assert numpy.allclose(
        before_qc_dict[satellite_io.LONGITUDES_KEY],
        after_qc_dict[satellite_io.LONGITUDES_KEY],
        atol=TOLERANCE
    )
    assert numpy.array_equal(
        before_qc_dict[satellite_io.BAND_NUMBERS_KEY],
        after_qc_dict[satellite_io.BAND_NUMBERS_KEY]
    )

    if expected_latitudes_deg_n is not None:
        assert numpy.allclose(
            expected_latitudes_deg_n,
            before_qc_dict[satellite_io.LATITUDES_KEY],
            atol=TOLERANCE
        )

    if expected_longitudes_deg_e is not None:
        assert numpy.allclose(
            expected_longitudes_deg_e,
            before_qc_dict[satellite_io.LONGITUDES_KEY],
            atol=TOLERANCE
        )

    if expected_band_numbers is not None:
        assert numpy.array_equal(
            expected_band_numbers, before_qc_dict[satellite_io.BAND_NUMBERS_KEY]
        )

    difference_matrix_kelvins = numpy.absolute(
        after_qc_dict[satellite_io.BRIGHTNESS_TEMP_KEY] -
        before_qc_dict[satellite_io.BRIGHTNESS_TEMP_KEY]
    )
    new_count_matrix = numpy.sum(difference_matrix_kelvins > TOLERANCE, axis=0)

    if count_matrix is None:
        count_matrix = new_count_matrix + 0
    else:
        count_matrix = count_matrix + new_count_matrix

    return (
        count_matrix,
        before_qc_dict[satellite_io.LATITUDES_KEY],
        before_qc_dict[satellite_io.LONGITUDES_KEY],
        before_qc_dict[satellite_io.BAND_NUMBERS_KEY]
    )


def _write_count_file(netcdf_file_name, count_matrix, latitudes_deg_n,
                      longitudes_deg_e, band_numbers):
    """Writes gridded difference counts to NetCDF file.

    :param netcdf_file_name: Path to output file.
    :param count_matrix: See doc for `_increment_counts_one_day`.
    :param latitudes_deg_n: Same.
    :param longitudes_deg_e: Same.
    :param band_numbers: Same.
    """

    # Do actual stuff.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.createDimension(ROW_DIMENSION_KEY, len(latitudes_deg_n))
    dataset_object.createDimension(COLUMN_DIMENSION_KEY, len(longitudes_deg_e))
    dataset_object.createDimension(BAND_DIMENSION_KEY, len(band_numbers))

    dataset_object.createVariable(
        LATITUDES_KEY, datatype=numpy.float32, dimensions=ROW_DIMENSION_KEY
    )
    dataset_object.variables[LATITUDES_KEY][:] = latitudes_deg_n

    dataset_object.createVariable(
        LONGITUDES_KEY, datatype=numpy.float32, dimensions=COLUMN_DIMENSION_KEY
    )
    dataset_object.variables[LONGITUDES_KEY][:] = longitudes_deg_e

    dataset_object.createVariable(
        BAND_NUMBERS_KEY, datatype=numpy.int32, dimensions=BAND_DIMENSION_KEY
    )
    dataset_object.variables[BAND_NUMBERS_KEY][:] = band_numbers

    these_dim = (ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY, BAND_DIMENSION_KEY)
    dataset_object.createVariable(
        DIFFERENCE_COUNT_KEY, datatype=numpy.int32, dimensions=these_dim
    )
    dataset_object.variables[DIFFERENCE_COUNT_KEY][:] = count_matrix

    dataset_object.close()


def read_count_file(netcdf_file_name):
    """Reads gridded difference counts from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: count_dict: Dictionary with the following keys.
    count_dict['count_matrix']: See doc for `_write_count_file`.
    count_dict['latitudes_deg_n']: Same.
    count_dict['longitudes_deg_e']: Same.
    count_dict['band_numbers']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    count_dict = {
        DIFFERENCE_COUNT_KEY:
            dataset_object.variables[DIFFERENCE_COUNT_KEY][:],
        LATITUDES_KEY: dataset_object.variables[LATITUDES_KEY][:],
        LONGITUDES_KEY: dataset_object.variables[LONGITUDES_KEY][:],
        BAND_NUMBERS_KEY: dataset_object.variables[BAND_NUMBERS_KEY][:]
    }

    dataset_object.close()
    return count_dict


def _run(before_qc_dir_name, after_qc_dir_name, first_date_string,
         last_date_string, output_file_name):
    """Counts changes made by satellite QC at each grid cell.

    This is effectively the main method.

    :param before_qc_dir_name: See documentation at top of file.
    :param after_qc_dir_name: Same.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param output_file_name: Same.
    """

    before_qc_file_names = satellite_io.find_many_files(
        top_directory_name=before_qc_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        prefer_zipped=True, allow_other_format=True,
        raise_error_if_any_missing=False
    )

    date_strings = [
        satellite_io.file_name_to_date(f) for f in before_qc_file_names
    ]
    num_days = len(date_strings)

    after_qc_file_names = [
        satellite_io.find_file(
            top_directory_name=after_qc_dir_name, valid_date_string=d,
            prefer_zipped=True, allow_other_format=True,
            raise_error_if_missing=True
        ) for d in date_strings
    ]

    count_matrix = None
    latitudes_deg_n = None
    longitudes_deg_e = None
    band_numbers = None

    for i in range(num_days):
        count_matrix, latitudes_deg_n, longitudes_deg_e, band_numbers = (
            _increment_counts_one_day(
                before_qc_file_name=before_qc_file_names[i],
                after_qc_file_name=after_qc_file_names[i],
                count_matrix=count_matrix,
                expected_latitudes_deg_n=latitudes_deg_n,
                expected_longitudes_deg_e=longitudes_deg_e,
                expected_band_numbers=band_numbers
            )
        )

    print(SEPARATOR_STRING)
    num_bands = len(band_numbers)

    for k in range(num_bands):
        print((
            'Number of grid cells with > 0 differences for band {0:d} = '
            '{1:d} of {2:d}'
        ).format(
            band_numbers[k],
            numpy.sum(count_matrix[..., k] > 0),
            count_matrix[..., k].size
        ))

    print(SEPARATOR_STRING)
    print('Writing results to file: "{0:s}"...'.format(output_file_name))

    _write_count_file(
        netcdf_file_name=output_file_name, count_matrix=count_matrix,
        latitudes_deg_n=latitudes_deg_n, longitudes_deg_e=longitudes_deg_e,
        band_numbers=band_numbers
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        before_qc_dir_name=getattr(INPUT_ARG_OBJECT, BEFORE_QC_DIR_ARG_NAME),
        after_qc_dir_name=getattr(INPUT_ARG_OBJECT, AFTER_QC_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
