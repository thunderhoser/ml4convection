"""Input/output methods for model predictions."""

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

DATE_FORMAT = '%Y%m%d'

EXAMPLE_DIMENSION_KEY = 'example'
GRID_ROW_DIMENSION_KEY = 'row'
GRID_COLUMN_DIMENSION_KEY = 'column'

PROBABILITY_MATRIX_KEY = 'forecast_probability_matrix'
TARGET_MATRIX_KEY = 'target_matrix'
VALID_TIMES_KEY = 'valid_times_unix_sec'
LATITUDES_KEY = 'latitude_deg_n'
LONGITUDES_KEY = 'longitude_deg_e'
MODEL_FILE_KEY = 'model_file_name'


def find_file(top_directory_name, valid_date_string,
              raise_error_if_missing=True):
    """Finds NetCDF file with predictions.

    :param top_directory_name: Name of top-level directory where file is
        expected.
    :param valid_date_string: Valid date (format "yyyymmdd").
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: prediction_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    radar_file_name = '{0:s}/{1:s}/predictions_{2:s}.nc'.format(
        top_directory_name, valid_date_string[:4], valid_date_string
    )

    if os.path.isfile(radar_file_name) or not raise_error_if_missing:
        return radar_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        radar_file_name
    )
    raise ValueError(error_string)


def write_file(
        netcdf_file_name, target_matrix, forecast_probability_matrix,
        valid_times_unix_sec, latitudes_deg_n, longitudes_deg_e,
        model_file_name):
    """Writes predictions to NetCDF file.

    E = number of examples (times)
    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to output file.
    :param target_matrix: E-by-M-by-N numpy array of true classes (integers in
        0...1).
    :param forecast_probability_matrix: E-by-M-by-N numpy array of forecast
        event probabilities (the "event" is when class = 1).
    :param valid_times_unix_sec: length-E numpy array of valid times.
    :param latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :param longitudes_deg_e: length-N numpy array of longitudes (deg E).
    :param model_file_name: Path to file with trained model (readable by
        `neural_net.read_model`).
    """

    # Check input args.
    error_checking.assert_is_integer_numpy_array(target_matrix)
    error_checking.assert_is_numpy_array(target_matrix, num_dimensions=3)
    error_checking.assert_is_geq_numpy_array(target_matrix, 0)
    error_checking.assert_is_leq_numpy_array(target_matrix, 1)

    error_checking.assert_is_numpy_array(
        forecast_probability_matrix,
        exact_dimensions=numpy.array(target_matrix.shape, dtype=int)
    )
    error_checking.assert_is_geq_numpy_array(forecast_probability_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(forecast_probability_matrix, 1.)

    num_examples = target_matrix.shape[0]
    num_grid_rows = target_matrix.shape[1]
    num_grid_columns = target_matrix.shape[2]

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

    these_dim = (
        EXAMPLE_DIMENSION_KEY, GRID_ROW_DIMENSION_KEY, GRID_COLUMN_DIMENSION_KEY
    )
    dataset_object.createVariable(
        TARGET_MATRIX_KEY, datatype=numpy.int32, dimensions=these_dim
    )
    dataset_object.variables[TARGET_MATRIX_KEY][:] = target_matrix

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
    return prediction_dict
