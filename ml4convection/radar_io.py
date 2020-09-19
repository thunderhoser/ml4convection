"""IO methods for processed radar data."""

import os
import sys
import gzip
import shutil
import numpy
import netCDF4

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import longitude_conversion as lng_conversion
import echo_classification as echo_classifn
import file_system_utils
import error_checking

TOLERANCE = 1e-6
GZIP_FILE_EXTENSION = '.gz'

DATE_FORMAT = '%Y%m%d'
DAYS_TO_SECONDS = 86400

REFL_TYPE_STRING = 'reflectivity'
ECHO_CLASSIFN_TYPE_STRING = 'echo_classification'
VALID_FILE_TYPE_STRINGS = [REFL_TYPE_STRING, ECHO_CLASSIFN_TYPE_STRING]

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

CONVECTIVE_FLAGS_KEY = 'convective_flag_matrix'
PEAKEDNESS_NEIGH_KEY = echo_classifn.PEAKEDNESS_NEIGH_KEY
MAX_PEAKEDNESS_HEIGHT_KEY = echo_classifn.MAX_PEAKEDNESS_HEIGHT_KEY
MIN_ECHO_TOP_KEY = echo_classifn.MIN_ECHO_TOP_KEY
ECHO_TOP_LEVEL_KEY = echo_classifn.ECHO_TOP_LEVEL_KEY
MIN_REFL_CRITERION1_KEY = echo_classifn.MIN_COMPOSITE_REFL_CRITERION1_KEY
MIN_REFL_CRITERION5_KEY = echo_classifn.MIN_COMPOSITE_REFL_CRITERION5_KEY
MIN_REFLECTIVITY_AML_KEY = echo_classifn.MIN_COMPOSITE_REFL_AML_KEY

MAIN_ECHO_CLASSIFN_KEYS = [
    CONVECTIVE_FLAGS_KEY, VALID_TIMES_KEY, LATITUDES_KEY, LONGITUDES_KEY
]
ECHO_CLASSIFN_METADATA_KEYS = [
    PEAKEDNESS_NEIGH_KEY, MAX_PEAKEDNESS_HEIGHT_KEY, MIN_ECHO_TOP_KEY,
    ECHO_TOP_LEVEL_KEY, MIN_REFL_CRITERION1_KEY, MIN_REFL_CRITERION5_KEY,
    MIN_REFLECTIVITY_AML_KEY
]


def _check_file_type(file_type_string):
    """Error-checks file type.

    :param file_type_string: File type.
    :raises: ValueError: if `file_type_string not in VALID_FILE_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(file_type_string)

    if file_type_string not in VALID_FILE_TYPE_STRINGS:
        error_string = (
            'Valid file types (listed below) do not include "{0:s}".\n{1:s}'
        ).format(
            file_type_string, str(VALID_FILE_TYPE_STRINGS)
        )
        raise ValueError(error_string)


def find_file(
        top_directory_name, valid_date_string, file_type_string,
        prefer_zipped=True, allow_other_format=True,
        raise_error_if_missing=True):
    """Finds NetCDF file with radar data.

    :param top_directory_name: Name of top-level directory where file is
        expected.
    :param valid_date_string: Valid date (format "yyyymmdd").
    :param file_type_string: File type (must be accepted by `_check_file_type`).
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
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)
    _check_file_type(file_type_string)
    error_checking.assert_is_boolean(prefer_zipped)
    error_checking.assert_is_boolean(allow_other_format)
    error_checking.assert_is_boolean(raise_error_if_missing)

    radar_file_name = '{0:s}/{1:s}/{2:s}_{3:s}.nc{4:s}'.format(
        top_directory_name, valid_date_string[:4], file_type_string,
        valid_date_string, GZIP_FILE_EXTENSION if prefer_zipped else ''
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

    valid_date_string = pathless_file_name.split('.')[0].split('_')[-1]
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    return valid_date_string


def find_many_files(
        top_directory_name, first_date_string, last_date_string,
        file_type_string, prefer_zipped=True, allow_other_format=True,
        raise_error_if_all_missing=True, raise_error_if_any_missing=False,
        test_mode=False):
    """Finds many NetCDF files with radar data.

    :param top_directory_name: See doc for `find_file`.
    :param first_date_string: First valid date (format "yyyymmdd").
    :param last_date_string: Last valid date (format "yyyymmdd").
    :param file_type_string: See doc for `find_file`.
    :param prefer_zipped: Same.
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
            file_type_string=file_type_string,
            prefer_zipped=prefer_zipped,
            allow_other_format=allow_other_format,
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


def write_reflectivity_file(
        netcdf_file_name, reflectivity_matrix_dbz, latitudes_deg_n,
        longitudes_deg_e, heights_m_asl, valid_time_unix_sec, append):
    """Writes reflectivity data to NetCDF file.

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

    # Do actual stuff.
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


def read_reflectivity_file(netcdf_file_name, metadata_only=False,
                           fill_nans=True):
    """Reads reflectivity data from NetCDF file.

    T = number of time steps
    M = number of rows in grid
    N = number of columns in grid
    H = number of heights in grid

    If `metadata_only == True`, the key `reflectivity_matrix_dbz` is not
    included in the output.

    :param netcdf_file_name: Path to input file.
    :param fill_nans: Boolean flag.  If True, will replace reflectivity values
        of NaN with zero.
    :param metadata_only: Boolean flag.  If True, will read metadata only.
    :return: reflectivity_dict: Dictionary with the following keys.
    reflectivity_dict['reflectivity_matrix_dbz']: T-by-M-by-N-by-H numpy array
        of reflectivities.
    reflectivity_dict['latitudes_deg_n']: length-M numpy array of latitudes
        (deg N).
    reflectivity_dict['longitudes_deg_e']: length-N numpy array of longitudes
        (deg E).
    reflectivity_dict['heights_m_asl']: length-H numpy array of heights (metres
        above sea level).
    reflectivity_dict['valid_times_unix_sec']: length-T numpy array of valid
        times.
    """

    error_checking.assert_is_boolean(metadata_only)
    error_checking.assert_is_boolean(fill_nans)
    fill_nans = fill_nans and not metadata_only
    error_checking.assert_is_string(netcdf_file_name)

    if netcdf_file_name.endswith(GZIP_FILE_EXTENSION):
        with gzip.open(netcdf_file_name) as gzip_handle:
            with netCDF4.Dataset(
                    'dummy', mode='r', memory=gzip_handle.read()
            ) as dataset_object:
                reflectivity_dict = {
                    LATITUDES_KEY: dataset_object.variables[LATITUDES_KEY][:],
                    LONGITUDES_KEY: dataset_object.variables[LONGITUDES_KEY][:],
                    HEIGHTS_KEY: dataset_object.variables[HEIGHTS_KEY][:],
                    VALID_TIMES_KEY:
                        dataset_object.variables[VALID_TIMES_KEY][:]
                }

                if not metadata_only:
                    reflectivity_dict[REFLECTIVITY_KEY] = (
                        dataset_object.variables[REFLECTIVITY_KEY][:]
                    )
    else:
        dataset_object = netCDF4.Dataset(netcdf_file_name)

        reflectivity_dict = {
            LATITUDES_KEY: dataset_object.variables[LATITUDES_KEY][:],
            LONGITUDES_KEY: dataset_object.variables[LONGITUDES_KEY][:],
            HEIGHTS_KEY: dataset_object.variables[HEIGHTS_KEY][:],
            VALID_TIMES_KEY: dataset_object.variables[VALID_TIMES_KEY][:]
        }

        if not metadata_only:
            reflectivity_dict[REFLECTIVITY_KEY] = (
                dataset_object.variables[REFLECTIVITY_KEY][:]
            )

        dataset_object.close()

    if numpy.any(numpy.diff(reflectivity_dict[LATITUDES_KEY]) < 0):
        reflectivity_dict[LATITUDES_KEY] = (
            reflectivity_dict[LATITUDES_KEY][::-1]
        )
        if not metadata_only:
            reflectivity_dict[REFLECTIVITY_KEY] = numpy.flip(
                reflectivity_dict[REFLECTIVITY_KEY], axis=1
            )

    if fill_nans:
        reflectivity_dict[REFLECTIVITY_KEY][
            numpy.isnan(reflectivity_dict[REFLECTIVITY_KEY])
        ] = 0.

    return reflectivity_dict


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


def write_echo_classifn_file(
        netcdf_file_name, convective_flag_matrix, latitudes_deg_n,
        longitudes_deg_e, valid_times_unix_sec, option_dict):
    """Writes echo classifications to NetCDF file.

    T = number of time steps
    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to output file.
    :param convective_flag_matrix: T-by-M-by-N numpy array of Boolean flags.
    :param latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :param longitudes_deg_e: length-N numpy array of longitudes (deg E).
    :param valid_times_unix_sec: length-T numpy array of valid times.
    :param option_dict: See doc for `echo_classification.find_convective_pixels`
        in GewitterGefahr.
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

    error_checking.assert_is_numpy_array(valid_times_unix_sec, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(valid_times_unix_sec)
    error_checking.assert_is_geq_numpy_array(
        valid_times_unix_sec, date_start_time_unix_sec
    )
    error_checking.assert_is_leq_numpy_array(
        valid_times_unix_sec, date_end_time_unix_sec
    )

    # Check coordinates.
    error_checking.assert_is_numpy_array(latitudes_deg_n, num_dimensions=1)
    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg_n)

    error_checking.assert_is_numpy_array(longitudes_deg_e, num_dimensions=1)
    longitudes_deg_e = (
        lng_conversion.convert_lng_positive_in_west(longitudes_deg_e)
    )

    # Check other input args.
    num_times = len(valid_times_unix_sec)
    num_grid_rows = len(latitudes_deg_n)
    num_grid_columns = len(longitudes_deg_e)
    expected_dim = numpy.array(
        [num_times, num_grid_rows, num_grid_columns], dtype=int
    )

    error_checking.assert_is_boolean_numpy_array(convective_flag_matrix)
    error_checking.assert_is_numpy_array(
        convective_flag_matrix, exact_dimensions=expected_dim
    )

    # Do actual stuff.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(
        PEAKEDNESS_NEIGH_KEY, option_dict[PEAKEDNESS_NEIGH_KEY]
    )
    dataset_object.setncattr(
        MAX_PEAKEDNESS_HEIGHT_KEY, option_dict[MAX_PEAKEDNESS_HEIGHT_KEY]
    )
    dataset_object.setncattr(MIN_ECHO_TOP_KEY, option_dict[MIN_ECHO_TOP_KEY])
    dataset_object.setncattr(
        ECHO_TOP_LEVEL_KEY, option_dict[ECHO_TOP_LEVEL_KEY]
    )
    dataset_object.setncattr(
        MIN_REFL_CRITERION1_KEY, option_dict[MIN_REFL_CRITERION1_KEY]
    )
    dataset_object.setncattr(
        MIN_REFL_CRITERION5_KEY, option_dict[MIN_REFL_CRITERION5_KEY]
    )
    dataset_object.setncattr(
        MIN_REFLECTIVITY_AML_KEY, option_dict[MIN_REFLECTIVITY_AML_KEY]
    )

    dataset_object.createDimension(TIME_DIMENSION_KEY, num_times)
    dataset_object.createDimension(ROW_DIMENSION_KEY, num_grid_rows)
    dataset_object.createDimension(COLUMN_DIMENSION_KEY, num_grid_columns)

    dataset_object.createVariable(
        VALID_TIMES_KEY, datatype=numpy.int32, dimensions=TIME_DIMENSION_KEY
    )
    dataset_object.variables[VALID_TIMES_KEY][:] = valid_times_unix_sec

    dataset_object.createVariable(
        LATITUDES_KEY, datatype=numpy.float32, dimensions=ROW_DIMENSION_KEY
    )
    dataset_object.variables[LATITUDES_KEY][:] = latitudes_deg_n

    dataset_object.createVariable(
        LONGITUDES_KEY, datatype=numpy.float32, dimensions=COLUMN_DIMENSION_KEY
    )
    dataset_object.variables[LONGITUDES_KEY][:] = longitudes_deg_e

    these_dim = (
        TIME_DIMENSION_KEY, ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY
    )
    dataset_object.createVariable(
        CONVECTIVE_FLAGS_KEY, datatype=numpy.int8, dimensions=these_dim
    )
    dataset_object.variables[CONVECTIVE_FLAGS_KEY][:] = (
        convective_flag_matrix.astype(int)
    )

    dataset_object.close()


def read_echo_classifn_file(netcdf_file_name):
    """Reads echo classifications from NetCDF file.

    T = number of time steps
    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to input file.
    :return: echo_classifn_dict: Dictionary with the following keys, plus keys
        in `option_dict` (input arg to `write_echo_classifn_file`).
    echo_classifn_dict['convective_flag_matrix']: T-by-M-by-N numpy array of
        Boolean flags.
    echo_classifn_dict['latitudes_deg_n']: length-M numpy array of latitudes
        (deg N).
    echo_classifn_dict['longitudes_deg_e']: length-N numpy array of longitudes
        (deg E).
    echo_classifn_dict['valid_times_unix_sec']: length-T numpy array of valid
        times.
    """

    error_checking.assert_is_string(netcdf_file_name)
    echo_classifn_dict = dict()

    if netcdf_file_name.endswith(GZIP_FILE_EXTENSION):
        with gzip.open(netcdf_file_name) as gzip_handle:
            with netCDF4.Dataset(
                    'dummy', mode='r', memory=gzip_handle.read()
            ) as dataset_object:
                for this_key in MAIN_ECHO_CLASSIFN_KEYS:
                    echo_classifn_dict[this_key] = (
                        dataset_object.variables[this_key][:]
                    )

                for this_key in ECHO_CLASSIFN_METADATA_KEYS:
                    echo_classifn_dict[this_key] = getattr(
                        dataset_object, this_key
                    )
    else:
        dataset_object = netCDF4.Dataset(netcdf_file_name)

        for this_key in MAIN_ECHO_CLASSIFN_KEYS:
            echo_classifn_dict[this_key] = (
                dataset_object.variables[this_key][:]
            )

        for this_key in ECHO_CLASSIFN_METADATA_KEYS:
            echo_classifn_dict[this_key] = getattr(
                dataset_object, this_key
            )

        dataset_object.close()

    echo_classifn_dict[CONVECTIVE_FLAGS_KEY] = (
        echo_classifn_dict[CONVECTIVE_FLAGS_KEY].astype(bool)
    )

    if numpy.any(numpy.diff(echo_classifn_dict[LATITUDES_KEY]) < 0):
        echo_classifn_dict[LATITUDES_KEY] = (
            echo_classifn_dict[LATITUDES_KEY][::-1]
        )
        echo_classifn_dict[CONVECTIVE_FLAGS_KEY] = numpy.flip(
            echo_classifn_dict[CONVECTIVE_FLAGS_KEY], axis=1
        )

    return echo_classifn_dict


def subset_by_index(reflectivity_dict, desired_indices):
    """Subsets examples (time steps) by index.

    :param reflectivity_dict: See doc for `read_reflectivity_file`.
    :param desired_indices: 1-D numpy array of desired indices.
    :return: reflectivity_dict: Same as input but with fewer examples.
    """

    # TODO(thunderhoser): Maybe make this work for echo classifications.

    error_checking.assert_is_numpy_array(desired_indices, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(desired_indices)
    error_checking.assert_is_geq_numpy_array(desired_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_indices, len(reflectivity_dict[VALID_TIMES_KEY])
    )

    for this_key in ONE_PER_EXAMPLE_KEYS:
        if reflectivity_dict[this_key] is None:
            continue

        reflectivity_dict[this_key] = (
            reflectivity_dict[this_key][desired_indices, ...]
        )

    return reflectivity_dict


def subset_by_time(reflectivity_dict, desired_times_unix_sec):
    """Subsets data by time.

    T = number of desired times

    :param reflectivity_dict: See doc for `read_reflectivity_file`.
    :param desired_times_unix_sec: length-T numpy array of desired times.
    :return: reflectivity_dict: Same as input but with fewer examples.
    :return: desired_indices: length-T numpy array of corresponding indices.
    """

    # TODO(thunderhoser): Maybe make this work for echo classifications.

    error_checking.assert_is_numpy_array(
        desired_times_unix_sec, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_times_unix_sec)

    desired_indices = numpy.array([
        numpy.where(reflectivity_dict[VALID_TIMES_KEY] == t)[0][0]
        for t in desired_times_unix_sec
    ], dtype=int)

    reflectivity_dict = subset_by_index(
        reflectivity_dict=reflectivity_dict, desired_indices=desired_indices
    )

    return reflectivity_dict, desired_indices
