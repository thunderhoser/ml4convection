"""IO methods for processed radar data."""

import os
import sys
import numpy
import netCDF4

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import longitude_conversion as lng_conversion
import file_system_utils
import error_checking

TOLERANCE = 1e-6

DATE_FORMAT = '%Y%m%d'
DAYS_TO_SECONDS = 86400

TIME_DIMENSION_KEY = 'time'
ROW_DIMENSION_KEY = 'row'
COLUMN_DIMENSION_KEY = 'column'

VALID_TIMES_KEY = 'valid_times_unix_sec'
LATITUDES_KEY = 'latitudes_deg_n'
LONGITUDES_KEY = 'longitudes_deg_e'
COMPOSITE_REFL_KEY = 'composite_refl_matrix_dbz'


def find_file(
        top_directory_name, valid_date_string, with_3d=False,
        raise_error_if_missing=True):
    """Finds NetCDF file with radar data.

    :param top_directory_name: Name of top-level directory where file is
        expected.
    :param valid_date_string: Valid date (format "yyyymmdd").
    :param with_3d: Boolean flag.  If True, will look for file with 3-D data.
        If False, will look for file with 2-D data (composite reflectivity).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: radar_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(with_3d)
    error_checking.assert_is_boolean(raise_error_if_missing)
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    radar_file_name = '{0:s}/{1:s}/{2:d}d-radar_{3:s}.nc'.format(
        top_directory_name, valid_date_string[:4], 3 if with_3d else 2,
        valid_date_string
    )

    if os.path.isfile(radar_file_name) or not raise_error_if_missing:
        return radar_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        radar_file_name
    )
    raise ValueError(error_string)


def file_name_to_date(radar_file_name):
    """Parses valid date from file name.

    :param radar_file_name: Path to radar file (see `find_file` for naming
        convention).
    :return: valid_date_string: Valid date (format "yyyymmdd").
    """

    error_checking.assert_is_string(radar_file_name)
    pathless_file_name = os.path.split(radar_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    valid_date_string = extensionless_file_name.split('_')[-1]
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    return valid_date_string


def find_many_files(
        top_directory_name, first_date_string, last_date_string, with_3d=False,
        raise_error_if_all_missing=True, raise_error_if_any_missing=False,
        test_mode=False):
    """Finds many NetCDF files with radar data.

    :param top_directory_name: See doc for `find_file`.
    :param first_date_string: First valid date (format "yyyymmdd").
    :param last_date_string: Last valid date (format "yyyymmdd").
    :param with_3d: See doc for `find_file`.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :param test_mode: Leave this alone.
    :return: radar_file_names: 1-D list of paths to radar files.  This list
        does *not* contain expected paths to non-existent files.
    :raises: ValueError: if all files are missing and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)
    error_checking.assert_is_boolean(test_mode)

    valid_date_strings = time_conversion.get_spc_dates_in_range(
        first_date_string, last_date_string
    )

    radar_file_names = []

    for this_date_string in valid_date_strings:
        this_file_name = find_file(
            top_directory_name=top_directory_name,
            valid_date_string=this_date_string, with_3d=with_3d,
            raise_error_if_missing=raise_error_if_any_missing
        )

        if test_mode or os.path.isfile(this_file_name):
            radar_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(radar_file_names) == 0:
        error_string = (
            'Cannot find any file in directory "{0:s}" from dates {1:s} to '
            '{2:s}.'
        ).format(
            top_directory_name, first_date_string, last_date_string
        )
        raise ValueError(error_string)

    return radar_file_names


def write_2d_file(
        netcdf_file_name, composite_refl_matrix_dbz, latitudes_deg_n,
        longitudes_deg_e, valid_time_unix_sec, append):
    """Writes 2-D radar data (composite reflectivity) to NetCDF file.

    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to output file.
    :param composite_refl_matrix_dbz: M-by-N numpy array of composite
        (column-maximum) reflectivities.
    :param latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :param longitudes_deg_e: length-N numpy array of longitudes (deg E).
    :param valid_time_unix_sec: Valid time.
    :param append: Boolean flag.  If True, will append to file if file already
        exists.  If False, will create new file, overwriting if necessary.
    """

    # Check input args.
    valid_date_string = file_name_to_date(netcdf_file_name)
    date_start_time_unix_sec = (
        time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)
    )
    date_end_time_unix_sec = date_start_time_unix_sec + DAYS_TO_SECONDS - 1

    error_checking.assert_is_integer(valid_time_unix_sec)
    error_checking.assert_is_geq(valid_time_unix_sec, date_start_time_unix_sec)
    error_checking.assert_is_leq(valid_time_unix_sec, date_end_time_unix_sec)

    error_checking.assert_is_numpy_array(latitudes_deg_n, num_dimensions=1)
    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg_n)

    error_checking.assert_is_numpy_array(longitudes_deg_e, num_dimensions=1)
    longitudes_deg_e = (
        lng_conversion.convert_lng_positive_in_west(longitudes_deg_e)
    )

    num_grid_rows = len(latitudes_deg_n)
    num_grid_columns = len(longitudes_deg_e)
    expected_dim = numpy.array([num_grid_rows, num_grid_columns], dtype=int)
    error_checking.assert_is_numpy_array(
        composite_refl_matrix_dbz, exact_dimensions=expected_dim
    )

    error_checking.assert_is_boolean(append)
    append = append and os.path.isfile(netcdf_file_name)

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    if append:
        dataset_object = netCDF4.Dataset(
            netcdf_file_name, 'a', format='NETCDF3_64BIT_OFFSET'
        )

        # TODO(thunderhoser): Make sure time being added is new.

        assert numpy.allclose(
            numpy.array(dataset_object.variables[LATITUDES_KEY][:]),
            latitudes_deg_n, atol=TOLERANCE
        )

        assert numpy.allclose(
            numpy.array(dataset_object.variables[LONGITUDES_KEY][:]),
            longitudes_deg_e, atol=TOLERANCE
        )

        num_times_orig = len(numpy.array(
            dataset_object.variables[VALID_TIMES_KEY][:]
        ))
        dataset_object.variables[VALID_TIMES_KEY][num_times_orig, ...] = (
            valid_time_unix_sec
        )
        dataset_object.variables[COMPOSITE_REFL_KEY][num_times_orig, ...] = (
            composite_refl_matrix_dbz
        )

        dataset_object.close()
        return

    # Do actual stuff.
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.createDimension(TIME_DIMENSION_KEY, None)
    dataset_object.createDimension(ROW_DIMENSION_KEY, num_grid_rows)
    dataset_object.createDimension(COLUMN_DIMENSION_KEY, num_grid_columns)

    dataset_object.createVariable(
        VALID_TIMES_KEY, datatype=numpy.int32, dimensions=TIME_DIMENSION_KEY
    )
    dataset_object.variables[VALID_TIMES_KEY][:] = numpy.array(
        [valid_time_unix_sec], dtype=int
    )

    dataset_object.createVariable(
        LATITUDES_KEY, datatype=numpy.float32, dimensions=ROW_DIMENSION_KEY
    )
    dataset_object.variables[LATITUDES_KEY][:] = latitudes_deg_n

    dataset_object.createVariable(
        LONGITUDES_KEY, datatype=numpy.float32, dimensions=COLUMN_DIMENSION_KEY
    )
    dataset_object.variables[LONGITUDES_KEY][:] = longitudes_deg_e

    dataset_object.createVariable(
        COMPOSITE_REFL_KEY, datatype=numpy.float32,
        dimensions=(TIME_DIMENSION_KEY, ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    )
    dataset_object.variables[COMPOSITE_REFL_KEY][:] = (
        numpy.expand_dims(composite_refl_matrix_dbz, axis=0)
    )

    dataset_object.close()


def read_2d_file(netcdf_file_name):
    """Reads 2-D radar data (composite reflectivity) from NetCDF file.

    T = number of time steps
    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to input file.
    :return: radar_dict: Dictionary with the following keys.
    radar_dict['composite_refl_matrix_dbz']: T-by-M-by-N numpy array of
        composite (column-maximum) reflectivities.
    radar_dict['latitudes_deg_n']: length-M numpy array of latitudes (deg N).
    radar_dict['longitudes_deg_e']: length-N numpy array of longitudes (deg E).
    radar_dict['valid_times_unix_sec']: length-T numpy array of valid times.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    radar_dict = {
        COMPOSITE_REFL_KEY: dataset_object.variables[COMPOSITE_REFL_KEY][:],
        LATITUDES_KEY: dataset_object.variables[LATITUDES_KEY][:],
        LONGITUDES_KEY: dataset_object.variables[LONGITUDES_KEY][:],
        VALID_TIMES_KEY: dataset_object.variables[VALID_TIMES_KEY][:]
    }

    dataset_object.close()
    return radar_dict
