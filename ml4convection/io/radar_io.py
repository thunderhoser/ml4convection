"""IO methods for processed radar data."""

import os
import gzip
import shutil
import numpy
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6
GZIP_FILE_EXTENSION = '.gz'

DATE_FORMAT = '%Y%m%d'
DAYS_TO_SECONDS = 86400

TIME_DIMENSION_KEY = 'time'
ROW_DIMENSION_KEY = 'row'
COLUMN_DIMENSION_KEY = 'column'
HEIGHT_DIMENSION_KEY = 'height'

VALID_TIMES_KEY = 'valid_times_unix_sec'
LATITUDES_KEY = 'latitudes_deg_n'
LONGITUDES_KEY = 'longitudes_deg_e'
HEIGHTS_KEY = 'heights_m_asl'
REFLECTIVITY_KEY = 'reflectivity_matrix_dbz'

ONE_PER_EXAMPLE_KEYS = [VALID_TIMES_KEY, REFLECTIVITY_KEY]


def find_file(
        top_directory_name, valid_date_string, prefer_zipped=True,
        allow_other_format=True, raise_error_if_missing=True):
    """Finds NetCDF file with radar data.

    :param top_directory_name: Name of top-level directory where file is
        expected.
    :param valid_date_string: Valid date (format "yyyymmdd").
    :param prefer_zipped: Boolean flag.  If True, will look for zipped file
        first.  If False, will look for unzipped file first.
    :param allow_other_format: Boolean flag.  If True, will allow opposite of
        preferred file format (zipped or unzipped).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: radar_file_name: File path.
    :raises: ValueError: if file is missing and
        `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(prefer_zipped)
    error_checking.assert_is_boolean(allow_other_format)
    error_checking.assert_is_boolean(raise_error_if_missing)
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    radar_file_name = '{0:s}/{1:s}/radar_{2:s}.nc{3:s}'.format(
        top_directory_name, valid_date_string[:4], valid_date_string,
        GZIP_FILE_EXTENSION if prefer_zipped else ''
    )

    if os.path.isfile(radar_file_name):
        return radar_file_name

    if allow_other_format:
        if prefer_zipped:
            radar_file_name = radar_file_name[:-len(GZIP_FILE_EXTENSION)]
        else:
            radar_file_name += GZIP_FILE_EXTENSION

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

    valid_date_string = pathless_file_name.split('.')[0].split('_')[1]
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    return valid_date_string


def find_many_files(
        top_directory_name, first_date_string, last_date_string,
        prefer_zipped=True, allow_other_format=True,
        raise_error_if_all_missing=True, raise_error_if_any_missing=False,
        test_mode=False):
    """Finds many NetCDF files with radar data.

    :param top_directory_name: See doc for `find_file`.
    :param first_date_string: First valid date (format "yyyymmdd").
    :param last_date_string: Last valid date (format "yyyymmdd").
    :param prefer_zipped: See doc for `find_file`.
    :param allow_other_format: Same.
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
            valid_date_string=this_date_string,
            prefer_zipped=prefer_zipped, allow_other_format=allow_other_format,
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


def write_file(
        netcdf_file_name, reflectivity_matrix_dbz, latitudes_deg_n,
        longitudes_deg_e, heights_m_asl, valid_time_unix_sec, append):
    """Writes radar data to NetCDF file.

    M = number of rows in grid
    N = number of columns in grid
    H = number of heights in grid

    :param netcdf_file_name: Path to output file.
    :param reflectivity_matrix_dbz: M-by-N-by-H numpy array of reflectivities.
    :param latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :param longitudes_deg_e: length-N numpy array of longitudes (deg E).
    :param heights_m_asl: length-H numpy array of heights (metres above sea
        level).
    :param valid_time_unix_sec: Valid time.
    :param valid_time_unix_sec: Valid time.
    :param append: Boolean flag.  If True, will append to file if file already
        exists.  If False, will create new file, overwriting if necessary.
    :raises: ValueError: if output file is a gzip file.
    """

    if netcdf_file_name.endswith(GZIP_FILE_EXTENSION):
        raise ValueError('Output file must not be gzip file.')

    # Check valid time.
    valid_date_string = file_name_to_date(netcdf_file_name)
    date_start_time_unix_sec = (
        time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)
    )
    date_end_time_unix_sec = date_start_time_unix_sec + DAYS_TO_SECONDS - 1

    error_checking.assert_is_integer(valid_time_unix_sec)
    error_checking.assert_is_geq(valid_time_unix_sec, date_start_time_unix_sec)
    error_checking.assert_is_leq(valid_time_unix_sec, date_end_time_unix_sec)

    # Check coordinates.
    error_checking.assert_is_numpy_array(latitudes_deg_n, num_dimensions=1)
    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg_n)

    error_checking.assert_is_numpy_array(longitudes_deg_e, num_dimensions=1)
    longitudes_deg_e = (
        lng_conversion.convert_lng_positive_in_west(longitudes_deg_e)
    )

    error_checking.assert_is_numpy_array(heights_m_asl, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(heights_m_asl, 0.)

    # Check other input args.
    num_grid_rows = len(latitudes_deg_n)
    num_grid_columns = len(longitudes_deg_e)
    num_grid_heights = len(heights_m_asl)

    expected_dim = numpy.array(
        [num_grid_rows, num_grid_columns, num_grid_heights], dtype=int
    )
    error_checking.assert_is_numpy_array(
        reflectivity_matrix_dbz, exact_dimensions=expected_dim
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
        assert numpy.allclose(
            numpy.array(dataset_object.variables[HEIGHTS_KEY][:]),
            heights_m_asl, atol=TOLERANCE
        )

        num_times_orig = len(numpy.array(
            dataset_object.variables[VALID_TIMES_KEY][:]
        ))
        dataset_object.variables[VALID_TIMES_KEY][num_times_orig, ...] = (
            valid_time_unix_sec
        )
        dataset_object.variables[REFLECTIVITY_KEY][num_times_orig, ...] = (
            reflectivity_matrix_dbz
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
    dataset_object.createDimension(HEIGHT_DIMENSION_KEY, num_grid_heights)

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
        HEIGHTS_KEY, datatype=numpy.float32, dimensions=HEIGHT_DIMENSION_KEY
    )
    dataset_object.variables[HEIGHTS_KEY][:] = heights_m_asl

    these_dim = (
        TIME_DIMENSION_KEY, ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY,
        HEIGHT_DIMENSION_KEY
    )
    dataset_object.createVariable(
        REFLECTIVITY_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[REFLECTIVITY_KEY][:] = (
        numpy.expand_dims(reflectivity_matrix_dbz, axis=0)
    )

    dataset_object.close()


def read_file(netcdf_file_name, fill_nans=True):
    """Reads radar data from NetCDF file.

    T = number of time steps
    M = number of rows in grid
    N = number of columns in grid
    H = number of heights in grid

    :param netcdf_file_name: Path to input file.
    :param fill_nans: Boolean flag.  If True, will replace reflectivity values
        of NaN with zero.
    :return: radar_dict: Dictionary with the following keys.
    radar_dict['reflectivity_matrix_dbz']: T-by-M-by-N-by-H numpy array of
        reflectivities.
    radar_dict['latitudes_deg_n']: length-M numpy array of latitudes (deg N).
    radar_dict['longitudes_deg_e']: length-N numpy array of longitudes (deg E).
    radar_dict['heights_m_asl']: length-H numpy array of heights (metres above
        sea level).
    radar_dict['valid_times_unix_sec']: length-T numpy array of valid times.
    """

    error_checking.assert_is_boolean(fill_nans)
    error_checking.assert_is_string(netcdf_file_name)

    if netcdf_file_name.endswith(GZIP_FILE_EXTENSION):
        with gzip.open(netcdf_file_name) as gzip_handle:
            with netCDF4.Dataset(
                    'dummy', mode='r', memory=gzip_handle.read()
            ) as dataset_object:
                radar_dict = {
                    REFLECTIVITY_KEY:
                        dataset_object.variables[REFLECTIVITY_KEY][:],
                    LATITUDES_KEY: dataset_object.variables[LATITUDES_KEY][:],
                    LONGITUDES_KEY: dataset_object.variables[LONGITUDES_KEY][:],
                    HEIGHTS_KEY: dataset_object.variables[HEIGHTS_KEY][:],
                    VALID_TIMES_KEY:
                        dataset_object.variables[VALID_TIMES_KEY][:]
                }
    else:
        dataset_object = netCDF4.Dataset(netcdf_file_name)

        radar_dict = {
            REFLECTIVITY_KEY: dataset_object.variables[REFLECTIVITY_KEY][:],
            LATITUDES_KEY: dataset_object.variables[LATITUDES_KEY][:],
            LONGITUDES_KEY: dataset_object.variables[LONGITUDES_KEY][:],
            HEIGHTS_KEY: dataset_object.variables[HEIGHTS_KEY][:],
            VALID_TIMES_KEY: dataset_object.variables[VALID_TIMES_KEY][:]
        }

        dataset_object.close()

    if numpy.any(numpy.diff(radar_dict[LATITUDES_KEY]) < 0):
        radar_dict[LATITUDES_KEY] = radar_dict[LATITUDES_KEY][::-1]
        radar_dict[REFLECTIVITY_KEY] = numpy.flip(
            radar_dict[REFLECTIVITY_KEY], axis=1
        )

    if fill_nans:
        radar_dict[REFLECTIVITY_KEY][
            numpy.isnan(radar_dict[REFLECTIVITY_KEY])
        ] = 0.

    return radar_dict


def compress_file(netcdf_file_name):
    """Compresses NetCDF file with radar data (turns it into a gzip file).

    :param netcdf_file_name: Path to NetCDF file.
    :raises: ValueError: if file is already gzipped.
    """

    error_checking.assert_is_string(netcdf_file_name)
    if netcdf_file_name.endswith(GZIP_FILE_EXTENSION):
        raise ValueError('File must not already be gzipped.')

    gzip_file_name = '{0:s}{1:s}'.format(netcdf_file_name, GZIP_FILE_EXTENSION)

    with open(netcdf_file_name, 'rb') as netcdf_handle:
        with gzip.open(gzip_file_name, 'wb') as gzip_handle:
            shutil.copyfileobj(netcdf_handle, gzip_handle)


def subset_by_index(radar_dict, desired_indices):
    """Subsets examples (time steps) by index.

    :param radar_dict: See doc for `read_file`.
    :param desired_indices: 1-D numpy array of desired indices.
    :return: radar_dict: Same as input but with fewer examples.
    """

    error_checking.assert_is_numpy_array(desired_indices, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(desired_indices)
    error_checking.assert_is_geq_numpy_array(desired_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_indices, len(radar_dict[VALID_TIMES_KEY])
    )

    for this_key in ONE_PER_EXAMPLE_KEYS:
        if radar_dict[this_key] is None:
            continue

        radar_dict[this_key] = (
            radar_dict[this_key][desired_indices, ...]
        )

    return radar_dict


def subset_by_time(radar_dict, desired_times_unix_sec):
    """Subsets data by time.

    T = number of desired times

    :param radar_dict: See doc for `read_file`.
    :param desired_times_unix_sec: length-T numpy array of desired times.
    :return: radar_dict: Same as input but with fewer examples.
    :return: desired_indices: length-T numpy array of corresponding indices.
    """

    error_checking.assert_is_numpy_array(
        desired_times_unix_sec, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_times_unix_sec)

    desired_indices = numpy.array([
        numpy.where(radar_dict[VALID_TIMES_KEY] == t)[0][0]
        for t in desired_times_unix_sec
    ], dtype=int)

    radar_dict = subset_by_index(
        radar_dict=radar_dict, desired_indices=desired_indices
    )

    return radar_dict, desired_indices
