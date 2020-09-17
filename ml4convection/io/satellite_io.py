"""IO methods for processed satellite data."""

import os
import copy
import numpy
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4convection.io import twb_satellite_io

TOLERANCE = 1e-6

DATE_FORMAT = '%Y%m%d'
DAYS_TO_SECONDS = 86400

BAND_NUMBERS = numpy.array([8, 9, 10, 11, 13, 14, 16], dtype=int)

MIN_BRIGHTNESS_COUNT = 0
MAX_BRIGHTNESS_COUNT = 255

TIME_DIMENSION_KEY = 'time'
ROW_DIMENSION_KEY = 'row'
COLUMN_DIMENSION_KEY = 'column'
BAND_DIMENSION_KEY = 'band'

VALID_TIMES_KEY = 'valid_times_unix_sec'
LATITUDES_KEY = 'latitudes_deg_n'
LONGITUDES_KEY = 'longitudes_deg_e'
BAND_NUMBERS_KEY = 'band_numbers'
BRIGHTNESS_TEMP_KEY = 'brightness_temp_matrix_kelvins'
BRIGHTNESS_COUNT_KEY = 'brightness_count_matrix'

ONE_PER_EXAMPLE_KEYS = [
    VALID_TIMES_KEY, BRIGHTNESS_TEMP_KEY, BRIGHTNESS_COUNT_KEY
]


def find_file(top_directory_name, valid_date_string,
              raise_error_if_missing=True):
    """Finds NetCDF file with satellite data.

    :param top_directory_name: Name of top-level directory where file is
        expected.
    :param valid_date_string: Valid date (format "yyyymmdd").
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: satellite_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    satellite_file_name = '{0:s}/{1:s}/satellite_{2:s}.nc'.format(
        top_directory_name, valid_date_string[:4], valid_date_string
    )

    if os.path.isfile(satellite_file_name) or not raise_error_if_missing:
        return satellite_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        satellite_file_name
    )
    raise ValueError(error_string)


def file_name_to_date(satellite_file_name):
    """Parses valid date from file name.

    :param satellite_file_name: Path to satellite file (see `find_file` for
        naming convention).
    :return: valid_date_string: Valid date (format "yyyymmdd").
    """

    error_checking.assert_is_string(satellite_file_name)
    pathless_file_name = os.path.split(satellite_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    valid_date_string = extensionless_file_name.split('_')[-1]
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    return valid_date_string


def find_many_files(
        top_directory_name, first_date_string, last_date_string,
        raise_error_if_all_missing=True, raise_error_if_any_missing=False,
        test_mode=False):
    """Finds many NetCDF files with satellite data.

    :param top_directory_name: See doc for `find_file`.
    :param first_date_string: First valid date (format "yyyymmdd").
    :param last_date_string: Last valid date (format "yyyymmdd").
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :param test_mode: Leave this alone.
    :return: satellite_file_names: 1-D list of paths to satellite files.  This
        list does *not* contain expected paths to non-existent files.
    :raises: ValueError: if all files are missing and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)
    error_checking.assert_is_boolean(test_mode)

    valid_date_strings = time_conversion.get_spc_dates_in_range(
        first_date_string, last_date_string
    )

    satellite_file_names = []

    for this_date_string in valid_date_strings:
        this_file_name = find_file(
            top_directory_name=top_directory_name,
            valid_date_string=this_date_string,
            raise_error_if_missing=raise_error_if_any_missing
        )

        if test_mode or os.path.isfile(this_file_name):
            satellite_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(satellite_file_names) == 0:
        error_string = (
            'Cannot find any file in directory "{0:s}" from dates {1:s} to '
            '{2:s}.'
        ).format(
            top_directory_name, first_date_string, last_date_string
        )
        raise ValueError(error_string)

    return satellite_file_names


def write_file(
        netcdf_file_name, latitudes_deg_n, longitudes_deg_e, band_numbers,
        valid_time_unix_sec, append, brightness_temp_matrix_kelvins=None,
        brightness_count_matrix=None):
    """Writes satellite data to NetCDF file.

    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (spectral bands)

    At least one of the last two arguments must be specified.

    :param netcdf_file_name: Path to output file.
    :param latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :param longitudes_deg_e: length-N numpy array of longitudes (deg E).
    :param band_numbers: length-C numpy array of band numbers (integers).
    :param valid_time_unix_sec: Valid time.
    :param append: Boolean flag.  If True, will append to file if file already
        exists.  If False, will create new file, overwriting if necessary.
    :param brightness_temp_matrix_kelvins: M-by-N-by-C numpy array of brightness
        temperatures.
    :param brightness_count_matrix: M-by-N-by-C numpy array of brightness
        counts.
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

    error_checking.assert_is_numpy_array(band_numbers, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(band_numbers)
    error_checking.assert_is_greater_numpy_array(band_numbers, 0)

    num_grid_rows = len(latitudes_deg_n)
    num_grid_columns = len(longitudes_deg_e)
    num_channels = len(band_numbers)
    expected_dim = numpy.array(
        [num_grid_rows, num_grid_columns, num_channels], dtype=int
    )

    if brightness_temp_matrix_kelvins is not None:
        error_checking.assert_is_numpy_array(
            brightness_temp_matrix_kelvins, exact_dimensions=expected_dim
        )
        error_checking.assert_is_greater_numpy_array(
            brightness_temp_matrix_kelvins, 0., allow_nan=True
        )

    if brightness_count_matrix is not None:
        error_checking.assert_is_numpy_array(
            brightness_count_matrix, exact_dimensions=expected_dim
        )
        error_checking.assert_is_geq_numpy_array(
            brightness_count_matrix, MIN_BRIGHTNESS_COUNT, allow_nan=True
        )
        error_checking.assert_is_leq_numpy_array(
            brightness_count_matrix, MAX_BRIGHTNESS_COUNT, allow_nan=True
        )

    assert not (
        brightness_temp_matrix_kelvins is None
        and brightness_count_matrix is None
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
        assert numpy.array_equal(
            numpy.array(dataset_object.variables[BAND_NUMBERS_KEY][:]),
            band_numbers
        )

        num_times_orig = len(numpy.array(
            dataset_object.variables[VALID_TIMES_KEY][:]
        ))
        dataset_object.variables[VALID_TIMES_KEY][num_times_orig, ...] = (
            valid_time_unix_sec
        )

        if brightness_temp_matrix_kelvins is not None:
            dataset_object.variables[BRIGHTNESS_TEMP_KEY][
                num_times_orig, ...
            ] = brightness_temp_matrix_kelvins

        if brightness_count_matrix is not None:
            dataset_object.variables[BRIGHTNESS_COUNT_KEY][
                num_times_orig, ...
            ] = brightness_count_matrix

        dataset_object.close()
        return

    # Do actual stuff.
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.createDimension(TIME_DIMENSION_KEY, None)
    dataset_object.createDimension(ROW_DIMENSION_KEY, num_grid_rows)
    dataset_object.createDimension(COLUMN_DIMENSION_KEY, num_grid_columns)
    dataset_object.createDimension(BAND_DIMENSION_KEY, num_channels)

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
        BAND_NUMBERS_KEY, datatype=numpy.int32, dimensions=BAND_DIMENSION_KEY
    )
    dataset_object.variables[BAND_NUMBERS_KEY][:] = band_numbers

    these_dim = (
        TIME_DIMENSION_KEY, ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY,
        BAND_DIMENSION_KEY
    )

    if brightness_temp_matrix_kelvins is not None:
        dataset_object.createVariable(
            BRIGHTNESS_TEMP_KEY, datatype=numpy.float32, dimensions=these_dim
        )
        dataset_object.variables[BRIGHTNESS_TEMP_KEY][:] = (
            numpy.expand_dims(brightness_temp_matrix_kelvins, axis=0)
        )

    if brightness_count_matrix is not None:
        dataset_object.createVariable(
            BRIGHTNESS_COUNT_KEY, datatype=numpy.float32, dimensions=these_dim
        )
        dataset_object.variables[BRIGHTNESS_COUNT_KEY][:] = (
            numpy.expand_dims(brightness_count_matrix, axis=0)
        )

    dataset_object.close()


def read_file(netcdf_file_name, read_temperatures, read_counts, fill_nans=True):
    """Reads satellite data from NetCDF file.

    T = number of time steps
    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (spectral bands)

    :param netcdf_file_name: Path to input file.
    :param read_temperatures: Boolean flag.  If True, will read brightness
        temperatures.
    :param read_counts: Boolean flag.  If True, will read brightness counts.
    :param fill_nans: Boolean flag.  If True, will use interpolation to fill NaN
        values.
    :return: satellite_dict: Dictionary with the following keys.
    satellite_dict['brightness_temp_matrix_kelvins']: T-by-M-by-N-by-C numpy
        array of brightness temperatures.  If `read_temperatures == False`,
        this will be None.
    satellite_dict['brightness_count_matrix']: T-by-M-by-N-by-C numpy
        array of brightness counts.  If `read_counts == False`,
        this will be None.
    satellite_dict['valid_times_unix_sec']: length-T numpy array of valid times.
    satellite_dict['latitudes_deg_n']: length-M numpy array of latitudes
        (deg N).
    satellite_dict['longitudes_deg_e']: length-N numpy array of longitudes
        (deg E).
    satellite_dict['band_numbers']: length-C numpy array of band numbers
        (integers).
    """

    error_checking.assert_is_boolean(read_temperatures)
    error_checking.assert_is_boolean(read_counts)
    error_checking.assert_is_boolean(fill_nans)

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    satellite_dict = {
        BRIGHTNESS_TEMP_KEY: None,
        BRIGHTNESS_COUNT_KEY: None,
        VALID_TIMES_KEY: dataset_object.variables[VALID_TIMES_KEY][:],
        LATITUDES_KEY: dataset_object.variables[LATITUDES_KEY][:],
        LONGITUDES_KEY: dataset_object.variables[LONGITUDES_KEY][:],
        BAND_NUMBERS_KEY: dataset_object.variables[BAND_NUMBERS_KEY][:]
    }

    if read_temperatures:
        satellite_dict[BRIGHTNESS_TEMP_KEY] = (
            dataset_object.variables[BRIGHTNESS_TEMP_KEY][:]
        )

        # TODO(thunderhoser): Find smarter way to deal with this.
        satellite_dict[BRIGHTNESS_TEMP_KEY][
            numpy.isnan(satellite_dict[BRIGHTNESS_TEMP_KEY])
        ] = 300.

    if read_counts:
        satellite_dict[BRIGHTNESS_COUNT_KEY] = (
            dataset_object.variables[BRIGHTNESS_COUNT_KEY][:]
        )

        if fill_nans:
            satellite_dict[BRIGHTNESS_COUNT_KEY][
                numpy.isnan(satellite_dict[BRIGHTNESS_COUNT_KEY])
            ] = 0.


    dataset_object.close()
    return satellite_dict


def subset_by_band(satellite_dict, band_numbers):
    """Subsets data by spectral band.

    :param satellite_dict: See doc for `read_file`.
    :param band_numbers: 1-D numpy array of desired band numbers (integers).
    :return: satellite_dict: Same as input but maybe with fewer bands.
    """

    error_checking.assert_is_integer_numpy_array(band_numbers)
    error_checking.assert_is_greater_numpy_array(band_numbers, 0)

    indices_to_keep = numpy.array([
        numpy.where(satellite_dict[BAND_NUMBERS_KEY] == n)[0][0]
        for n in band_numbers
    ], dtype=int)

    satellite_dict[BAND_NUMBERS_KEY] = (
        satellite_dict[BAND_NUMBERS_KEY][indices_to_keep]
    )

    if satellite_dict[BRIGHTNESS_TEMP_KEY] is not None:
        satellite_dict[BRIGHTNESS_TEMP_KEY] = (
            satellite_dict[BRIGHTNESS_TEMP_KEY][..., indices_to_keep]
        )

    if satellite_dict[BRIGHTNESS_COUNT_KEY] is not None:
        satellite_dict[BRIGHTNESS_COUNT_KEY] = (
            satellite_dict[BRIGHTNESS_COUNT_KEY][..., indices_to_keep]
        )

    return satellite_dict


def subset_by_index(satellite_dict, desired_indices):
    """Subsets examples (time steps) by index.

    :param satellite_dict: See doc for `read_file`.
    :param desired_indices: 1-D numpy array of desired indices.
    :return: satellite_dict: Same as input but with fewer examples.
    """

    error_checking.assert_is_numpy_array(desired_indices, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(desired_indices)
    error_checking.assert_is_geq_numpy_array(desired_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_indices, len(satellite_dict[VALID_TIMES_KEY])
    )

    for this_key in ONE_PER_EXAMPLE_KEYS:
        if satellite_dict[this_key] is None:
            continue

        satellite_dict[this_key] = (
            satellite_dict[this_key][desired_indices, ...]
        )

    return satellite_dict


def subset_by_time(satellite_dict, desired_times_unix_sec):
    """Subsets data by time.

    T = number of desired times

    :param satellite_dict: See doc for `read_file`.
    :param desired_times_unix_sec: length-T numpy array of desired times.
    :return: satellite_dict: Same as input but with fewer examples.
    :return: desired_indices: length-T numpy array of corresponding indices.
    """

    error_checking.assert_is_numpy_array(
        desired_times_unix_sec, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_times_unix_sec)

    desired_indices = numpy.array([
        numpy.where(satellite_dict[VALID_TIMES_KEY] == t)[0][0]
        for t in desired_times_unix_sec
    ], dtype=int)

    satellite_dict = subset_by_index(
        satellite_dict=satellite_dict, desired_indices=desired_indices
    )

    return satellite_dict, desired_indices


def concat_data(satellite_dicts):
    """Concatenates many dictionaries with satellite data into one.

    :param satellite_dicts: List of dictionaries, each in the format returned by
        `read_file`.
    :return: satellite_dict: Single dictionary, also in the format returned by
        `read_file`.
    :raises: ValueError: if any two dictionaries have different band numbers,
        latitudes, longitudes, or variables.
    """

    satellite_dict = copy.deepcopy(satellite_dicts[0])
    keys_to_match = [BAND_NUMBERS_KEY, LATITUDES_KEY, LONGITUDES_KEY]

    for i in range(1, len(satellite_dicts)):
        for this_key in keys_to_match:
            if this_key == BAND_NUMBERS_KEY:
                if numpy.array_equal(
                        satellite_dict[this_key], satellite_dicts[i][this_key]
                ):
                    continue
            else:
                if numpy.allclose(
                        satellite_dict[this_key], satellite_dicts[i][this_key],
                        atol=TOLERANCE
                ):
                    continue

            error_string = (
                '1st and {0:d}th dictionaries have different values for '
                '"{1:s}".  1st dictionary:\n{2:s}\n\n'
                '{0:d}th dictionary:\n{3:s}'
            ).format(
                i + 1, this_key,
                str(satellite_dict[this_key]),
                str(satellite_dicts[i][this_key])
            )

            raise ValueError(error_string)

    have_temperatures_by_dict = numpy.array([
        d[BRIGHTNESS_TEMP_KEY] is not None for d in satellite_dicts
    ], dtype=bool)

    have_counts_by_dict = numpy.array([
        d[BRIGHTNESS_COUNT_KEY] is not None for d in satellite_dicts
    ], dtype=bool)

    if len(numpy.unique(have_temperatures_by_dict)) > 1:
        error_string = (
            '{0:d} dictionaries have brightness temperatures, and {1:d} do not.'
            '  Either all or none should have brightness temperatures.'
        ).format(
            numpy.sum(have_temperatures_by_dict),
            numpy.sum(numpy.invert(have_temperatures_by_dict))
        )

        raise ValueError(error_string)

    if len(numpy.unique(have_counts_by_dict)) > 1:
        error_string = (
            '{0:d} dictionaries have brightness counts, and {1:d} do not.'
            '  Either all or none should have brightness counts.'
        ).format(
            numpy.sum(have_counts_by_dict),
            numpy.sum(numpy.invert(have_counts_by_dict))
        )

        raise ValueError(error_string)

    for i in range(1, len(satellite_dicts)):
        for this_key in ONE_PER_EXAMPLE_KEYS:
            if satellite_dict[this_key] is None:
                continue

            satellite_dict[this_key] = numpy.concatenate((
                satellite_dict[this_key], satellite_dicts[i][this_key]
            ), axis=0)

    return satellite_dict


def counts_to_temperatures(satellite_dict):
    """Converts raw counts to brightness temperatures.

    :param satellite_dict: List of dictionaries, each in the format returned by
        `read_file`.
    :return: satellite_dict: Same but with brightness temperatures, not just
        counts.
    """

    band_numbers = satellite_dict[BAND_NUMBERS_KEY]
    brightness_count_matrix = satellite_dict[BRIGHTNESS_COUNT_KEY]
    brightness_temp_matrix_kelvins = numpy.full(
        brightness_count_matrix.shape, numpy.nan
    )

    num_bands = len(band_numbers)

    for j in range(num_bands):
        brightness_temp_matrix_kelvins[..., j] = (
            twb_satellite_io.count_to_temperature(
                brightness_counts=brightness_count_matrix[..., j],
                band_number=band_numbers[j]
            )
        )

    satellite_dict[BRIGHTNESS_TEMP_KEY] = brightness_temp_matrix_kelvins
    return satellite_dict
