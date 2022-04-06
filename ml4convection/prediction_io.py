"""Input/output methods for model predictions."""

import os
import sys
import gzip
import numpy
import netCDF4

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import gg_general_utils
import time_conversion
import longitude_conversion as lng_conversion
import file_system_utils
import error_checking
import radar_io

DATE_FORMAT = '%Y%m%d'
GZIP_FILE_EXTENSION = '.gz'

EXAMPLE_DIMENSION_KEY = 'example'
GRID_ROW_DIMENSION_KEY = 'row'
GRID_COLUMN_DIMENSION_KEY = 'column'
PREDICTION_SET_DIMENSION_KEY = 'prediction_set'

PROBABILITY_MATRIX_KEY = 'forecast_probability_matrix'
TARGET_MATRIX_KEY = 'target_matrix'
VALID_TIMES_KEY = 'valid_times_unix_sec'
LATITUDES_KEY = 'latitude_deg_n'
LONGITUDES_KEY = 'longitude_deg_e'
MODEL_FILE_KEY = 'model_file_name'

ONE_PER_EXAMPLE_KEYS = [
    PROBABILITY_MATRIX_KEY, TARGET_MATRIX_KEY, VALID_TIMES_KEY
]


def find_file(
        top_directory_name, valid_date_string, radar_number=None,
        prefer_zipped=True, allow_other_format=True,
        raise_error_if_missing=True):
    """Finds NetCDF file with predictions.

    :param top_directory_name: Name of top-level directory where file is
        expected.
    :param valid_date_string: Valid date (format "yyyymmdd").
    :param radar_number: Radar number (non-negative integer).  If you are
        looking for data on the full grid, leave this alone.
    :param prefer_zipped: Boolean flag.  If True, will look for zipped file
        first.  If False, will look for unzipped file first.
    :param allow_other_format: Boolean flag.  If True, will allow opposite of
        preferred file format (zipped or unzipped).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: prediction_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)
    error_checking.assert_is_boolean(prefer_zipped)
    error_checking.assert_is_boolean(allow_other_format)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if radar_number is not None:
        error_checking.assert_is_integer(radar_number)
        error_checking.assert_is_geq(radar_number, 0)

    prediction_file_name = '{0:s}/{1:s}/predictions_{2:s}{3:s}.nc{4:s}'.format(
        top_directory_name, valid_date_string[:4], valid_date_string,
        '' if radar_number is None else '_radar{0:d}'.format(radar_number),
        GZIP_FILE_EXTENSION if prefer_zipped else ''
    )

    if os.path.isfile(prediction_file_name):
        return prediction_file_name

    if allow_other_format:
        if prefer_zipped:
            prediction_file_name = (
                prediction_file_name[:-len(GZIP_FILE_EXTENSION)]
            )
        else:
            prediction_file_name += GZIP_FILE_EXTENSION

    if os.path.isfile(prediction_file_name) or not raise_error_if_missing:
        return prediction_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        prediction_file_name
    )
    raise ValueError(error_string)


def file_name_to_date(prediction_file_name):
    """Parses date from name of prediction file.

    :param prediction_file_name: Path to prediction file (see `find_file` for
        naming convention).
    :return: valid_date_string: Valid date (format "yyyymmdd").
    """

    error_checking.assert_is_string(prediction_file_name)
    pathless_file_name = os.path.split(prediction_file_name)[-1]

    valid_date_string = pathless_file_name.split('.')[0].split('_')[1]
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    return valid_date_string


def file_name_to_radar_number(prediction_file_name):
    """Parses radar number from name of prediction file.

    :param prediction_file_name: Path to prediction file (see `find_file` for
        naming convention).
    :return: radar_number: Radar number (non-negative integer).  If file
        contains data on the full grid, this is None.
    """

    error_checking.assert_is_string(prediction_file_name)
    pathless_file_name = os.path.split(prediction_file_name)[-1]
    radar_word = pathless_file_name.split('.')[0].split('_')[-1]

    if 'radar' not in radar_word:
        return None

    return int(radar_word.replace('radar', ''))


def find_many_files(
        top_directory_name, first_date_string, last_date_string,
        prefer_zipped=True, allow_other_format=True, radar_number=None,
        raise_error_if_all_missing=True, raise_error_if_any_missing=False,
        test_mode=False):
    """Finds many NetCDF files with predictions.

    :param top_directory_name: See doc for `find_file`.
    :param first_date_string: First valid date (format "yyyymmdd").
    :param last_date_string: Last valid date (format "yyyymmdd").
    :param prefer_zipped: See doc for `find_file`.
    :param allow_other_format: Same.
    :param radar_number: Same.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :param test_mode: Leave this alone.
    :return: prediction_file_names: 1-D list of paths to prediction files.  This
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

    prediction_file_names = []

    for this_date_string in valid_date_strings:
        this_file_name = find_file(
            top_directory_name=top_directory_name,
            valid_date_string=this_date_string,
            prefer_zipped=prefer_zipped, allow_other_format=allow_other_format,
            radar_number=radar_number,
            raise_error_if_missing=raise_error_if_any_missing
        )

        if test_mode or os.path.isfile(this_file_name):
            prediction_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(prediction_file_names) == 0:
        error_string = (
            'Cannot find any file in directory "{0:s}" from dates {1:s} to '
            '{2:s}.'
        ).format(
            top_directory_name, first_date_string, last_date_string
        )
        raise ValueError(error_string)

    return prediction_file_names


def write_file(
        netcdf_file_name, target_matrix, forecast_probability_matrix,
        valid_times_unix_sec, latitudes_deg_n, longitudes_deg_e,
        model_file_name):
    """Writes predictions to NetCDF file.

    E = number of examples (times)
    M = number of rows in grid
    N = number of columns in grid
    S = number of prediction sets

    :param netcdf_file_name: Path to output file.
    :param target_matrix: E-by-M-by-N numpy array of true classes (integers in
        0...1).
    :param forecast_probability_matrix: E-by-M-by-N or E-by-M-by-N-by-S numpy
        array of forecast event probabilities (the "event" is when class = 1).
    :param valid_times_unix_sec: length-E numpy array of valid times.
    :param latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :param longitudes_deg_e: length-N numpy array of longitudes (deg E).
    :param model_file_name: Path to file with trained model (readable by
        `neural_net.read_model`).
    :raises: ValueError: if output file is a gzip file.
    """

    # Check input args.
    if netcdf_file_name.endswith(GZIP_FILE_EXTENSION):
        raise ValueError('Output file must not be gzip file.')

    error_checking.assert_is_integer_numpy_array(target_matrix)
    error_checking.assert_is_numpy_array(target_matrix, num_dimensions=3)
    error_checking.assert_is_geq_numpy_array(target_matrix, 0)
    error_checking.assert_is_leq_numpy_array(target_matrix, 1)

    error_checking.assert_is_numpy_array(forecast_probability_matrix)
    if len(forecast_probability_matrix.shape) == 3:
        forecast_probability_matrix = numpy.expand_dims(
            forecast_probability_matrix, axis=-1
        )

    error_checking.assert_is_numpy_array(
        forecast_probability_matrix[..., 0],
        exact_dimensions=numpy.array(target_matrix.shape, dtype=int)
    )
    error_checking.assert_is_geq_numpy_array(forecast_probability_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(forecast_probability_matrix, 1.)

    num_examples = target_matrix.shape[0]
    num_grid_rows = target_matrix.shape[1]
    num_grid_columns = target_matrix.shape[2]
    num_prediction_sets = forecast_probability_matrix.shape[3]

    error_checking.assert_is_integer_numpy_array(valid_times_unix_sec)
    error_checking.assert_is_numpy_array(
        valid_times_unix_sec,
        exact_dimensions=numpy.array([num_examples], dtype=int)
    )

    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg_n)
    error_checking.assert_is_numpy_array(
        latitudes_deg_n,
        exact_dimensions=numpy.array([num_grid_rows], dtype=int)
    )

    longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitudes_deg=longitudes_deg_e, allow_nan=False
    )
    error_checking.assert_is_numpy_array(
        longitudes_deg_e,
        exact_dimensions=numpy.array([num_grid_columns], dtype=int)
    )

    error_checking.assert_is_string(model_file_name)

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)

    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(GRID_ROW_DIMENSION_KEY, num_grid_rows)
    dataset_object.createDimension(GRID_COLUMN_DIMENSION_KEY, num_grid_columns)
    dataset_object.createDimension(
        PREDICTION_SET_DIMENSION_KEY, num_prediction_sets
    )

    these_dim = (
        EXAMPLE_DIMENSION_KEY, GRID_ROW_DIMENSION_KEY, GRID_COLUMN_DIMENSION_KEY
    )
    dataset_object.createVariable(
        TARGET_MATRIX_KEY, datatype=numpy.int32, dimensions=these_dim
    )
    dataset_object.variables[TARGET_MATRIX_KEY][:] = target_matrix

    these_dim = (
        EXAMPLE_DIMENSION_KEY, GRID_ROW_DIMENSION_KEY,
        GRID_COLUMN_DIMENSION_KEY, PREDICTION_SET_DIMENSION_KEY
    )
    dataset_object.createVariable(
        PROBABILITY_MATRIX_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[PROBABILITY_MATRIX_KEY][:] = (
        forecast_probability_matrix
    )

    dataset_object.createVariable(
        VALID_TIMES_KEY, datatype=numpy.int32, dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[VALID_TIMES_KEY][:] = valid_times_unix_sec

    dataset_object.createVariable(
        LATITUDES_KEY, datatype=numpy.float32, dimensions=GRID_ROW_DIMENSION_KEY
    )
    dataset_object.variables[LATITUDES_KEY][:] = latitudes_deg_n

    dataset_object.createVariable(
        LONGITUDES_KEY, datatype=numpy.float32,
        dimensions=GRID_COLUMN_DIMENSION_KEY
    )
    dataset_object.variables[LONGITUDES_KEY][:] = longitudes_deg_e

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads predictions from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: prediction_dict: Dictionary with the following keys.
    prediction_dict['target_matrix']: See doc for `write_file`.
    prediction_dict['forecast_probability_matrix']: Same.
    prediction_dict['valid_times_unix_sec']: Same.
    prediction_dict['latitudes_deg_n']: Same.
    prediction_dict['longitudes_deg_e']: Same.
    prediction_dict['model_file_name']: Same.
    """

    if netcdf_file_name.endswith(GZIP_FILE_EXTENSION):
        with gzip.open(netcdf_file_name) as gzip_handle:
            with netCDF4.Dataset(
                    'dummy', mode='r', memory=gzip_handle.read()
            ) as dataset_object:
                prediction_dict = {
                    TARGET_MATRIX_KEY:
                        dataset_object.variables[TARGET_MATRIX_KEY][:],
                    PROBABILITY_MATRIX_KEY:
                        dataset_object.variables[PROBABILITY_MATRIX_KEY][:],
                    VALID_TIMES_KEY:
                        dataset_object.variables[VALID_TIMES_KEY][:],
                    LATITUDES_KEY: dataset_object.variables[LATITUDES_KEY][:],
                    LONGITUDES_KEY: dataset_object.variables[LONGITUDES_KEY][:],
                    MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY))
                }

                if len(prediction_dict[PROBABILITY_MATRIX_KEY].shape) == 3:
                    prediction_dict[PROBABILITY_MATRIX_KEY] = numpy.expand_dims(
                        prediction_dict[PROBABILITY_MATRIX_KEY], axis=-1
                    )

                return prediction_dict

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    prediction_dict = {
        TARGET_MATRIX_KEY: dataset_object.variables[TARGET_MATRIX_KEY][:],
        PROBABILITY_MATRIX_KEY:
            dataset_object.variables[PROBABILITY_MATRIX_KEY][:],
        VALID_TIMES_KEY: dataset_object.variables[VALID_TIMES_KEY][:],
        LATITUDES_KEY: dataset_object.variables[LATITUDES_KEY][:],
        LONGITUDES_KEY: dataset_object.variables[LONGITUDES_KEY][:],
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY))
    }

    dataset_object.close()

    if len(prediction_dict[PROBABILITY_MATRIX_KEY].shape) == 3:
        prediction_dict[PROBABILITY_MATRIX_KEY] = numpy.expand_dims(
            prediction_dict[PROBABILITY_MATRIX_KEY], axis=-1
        )

    return prediction_dict


def compress_file(netcdf_file_name):
    """Compresses file with predictions (turns it into gzip file).

    :param netcdf_file_name: Path to NetCDF file with predictions.
    :raises: ValueError: if file is already gzipped.
    """

    radar_io.compress_file(netcdf_file_name)


def subset_by_index(prediction_dict, desired_indices):
    """Subsets examples (time steps) by index.

    :param prediction_dict: See doc for `read_file`.
    :param desired_indices: 1-D numpy array of desired indices.
    :return: prediction_dict: Same as input but with fewer examples.
    """

    error_checking.assert_is_numpy_array(desired_indices, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(desired_indices)
    error_checking.assert_is_geq_numpy_array(desired_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_indices, len(prediction_dict[VALID_TIMES_KEY])
    )

    for this_key in ONE_PER_EXAMPLE_KEYS:
        if prediction_dict[this_key] is None:
            continue

        prediction_dict[this_key] = (
            prediction_dict[this_key][desired_indices, ...]
        )

    return prediction_dict


def subset_by_time(prediction_dict, desired_times_unix_sec):
    """Subsets data by time.

    T = number of desired times

    :param prediction_dict: See doc for `read_file`.
    :param desired_times_unix_sec: length-T numpy array of desired times.
    :return: prediction_dict: Same as input but with fewer examples.
    :return: desired_indices: length-T numpy array of corresponding indices.
    """

    error_checking.assert_is_numpy_array(
        desired_times_unix_sec, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_times_unix_sec)

    desired_indices = numpy.array([
        numpy.where(prediction_dict[VALID_TIMES_KEY] == t)[0][0]
        for t in desired_times_unix_sec
    ], dtype=int)

    prediction_dict = subset_by_index(
        prediction_dict=prediction_dict, desired_indices=desired_indices
    )

    return prediction_dict, desired_indices


def smooth_probabilities(prediction_dict, smoothing_radius_px):
    """Smooths the map of predicted probabilities at each time step.

    :param prediction_dict: Dictionary returned by `read_file`.
    :param smoothing_radius_px: e-folding radius for Gaussian smoother (pixels).
    :return: prediction_dict: Same as input but with smoothed fields.
    """

    error_checking.assert_is_greater(smoothing_radius_px, 0.)

    probability_matrix = prediction_dict[PROBABILITY_MATRIX_KEY]
    num_times = probability_matrix.shape[0]
    num_prediction_sets = probability_matrix.shape[1]

    print((
        'Applying Gaussian smoother with e-folding radius = {0:f} pixels...'
    ).format(
        smoothing_radius_px
    ))

    for i in range(num_times):
        for j in range(num_prediction_sets):
            probability_matrix[i, ..., j] = (
                gg_general_utils.apply_gaussian_filter(
                    input_matrix=probability_matrix[i, ..., j],
                    e_folding_radius_grid_cells=smoothing_radius_px
                )
            )

    prediction_dict[PROBABILITY_MATRIX_KEY] = probability_matrix
    return prediction_dict
