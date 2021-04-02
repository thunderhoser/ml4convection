"""Methods for computing, reading, and writing saliency maps."""

import os
import sys
import numpy
import netCDF4

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import error_checking
import gg_saliency_maps as saliency_utils
import radar_utils

DATE_FORMAT = '%Y%m%d'
NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)

EXAMPLE_DIMENSION_KEY = 'example'
ROW_DIMENSION_KEY = 'row'
COLUMN_DIMENSION_KEY = 'column'
LAG_TIME_DIMENSION_KEY = 'lag_time'
CHANNEL_DIMENSION_KEY = 'channel'

MODEL_FILE_KEY = 'model_file_name'
IS_LAYER_OUTPUT_KEY = 'is_layer_output'
LAYER_NAME_KEY = 'layer_name'
NEURON_INDICES_KEY = 'neuron_indices'
IDEAL_ACTIVATION_KEY = 'ideal_activation'
MULTIPLY_BY_INPUT_KEY = 'multiply_by_input'

VALID_TIMES_KEY = 'valid_times_unix_sec'
LATITUDES_KEY = 'latitudes_deg_n'
LONGITUDES_KEY = 'longitudes_deg_e'
SALIENCY_MATRIX_KEY = 'saliency_matrix'
MEAN_SALIENCY_KEY = 'mean_saliency_matrix'
MEAN_DENORM_PREDICTORS_KEY = 'mean_denorm_predictor_matrix'


def check_metadata(layer_name, neuron_indices, ideal_activation,
                   multiply_by_input):
    """Checks metadata for errors.

    The "relevant neuron" is that whose activation will be used in the numerator
    of the saliency equation.  In other words, if the relevant neuron is n,
    the saliency of each predictor x will be d(a_n) / dx, where a_n is the
    activation of n.

    :param layer_name: Name of layer with relevant neuron.
    :param neuron_indices: 1-D numpy array with indices of relevant neuron.
        Must have length D - 1, where D = number of dimensions in layer output.
        The first dimension is the batch dimension, which always has length
        `None` in Keras.
    :param ideal_activation: Ideal neuron activation, used to define loss
        function.  The loss function will be
        (neuron_activation - ideal_activation)**2.
    :param multiply_by_input: Boolean flag.  If True, saliency will be
        multiplied by input, so the interpretation method will actually be
        input * gradient.
    """

    error_checking.assert_is_string(layer_name)
    error_checking.assert_is_integer_numpy_array(neuron_indices)
    error_checking.assert_is_geq_numpy_array(neuron_indices, 0)
    error_checking.assert_is_numpy_array(neuron_indices, num_dimensions=1)
    error_checking.assert_is_not_nan(ideal_activation)
    error_checking.assert_is_boolean(multiply_by_input)


def get_saliency_one_neuron(
        model_object, predictor_matrix, layer_name, neuron_indices,
        ideal_activation, multiply_by_input):
    """Computes saliency maps with respect to activation of one neuron.

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix: numpy array of predictors.  Must be formatted in
        the same way as for training and inference.
    :param layer_name: See doc for `check_metadata`.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
    :param multiply_by_input: Same.
    :return: saliency_matrix: Matrix of saliency values, with same shape as
        `predictor_matrix`.
    """

    check_metadata(
        layer_name=layer_name, neuron_indices=neuron_indices,
        ideal_activation=ideal_activation, multiply_by_input=multiply_by_input
    )
    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)

    activation_tensor = None

    for k in neuron_indices[::-1]:
        if activation_tensor is None:
            activation_tensor = (
                model_object.get_layer(name=layer_name).output[..., k]
            )
        else:
            activation_tensor = activation_tensor[..., k]

    # if ideal_activation is None:
    #     loss_tensor = -K.sign(activation_tensor) * activation_tensor ** 2

    loss_tensor = (activation_tensor - ideal_activation) ** 2

    saliency_matrix = saliency_utils.do_saliency_calculations(
        model_object=model_object, loss_tensor=loss_tensor,
        list_of_input_matrices=[predictor_matrix]
    )[0]

    if multiply_by_input:
        saliency_matrix = saliency_matrix * predictor_matrix

    return saliency_matrix


def find_file(
        top_directory_name, valid_date_string, radar_number,
        raise_error_if_missing=True):
    """Finds NetCDF file with saliency values.

    :param top_directory_name: Name of top-level directory where file is
        expected.
    :param valid_date_string: Valid date (format "yyyymmdd").
    :param radar_number: Radar number (non-negative integer).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: saliency_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_integer(radar_number)
    error_checking.assert_is_geq(radar_number, 0)
    error_checking.assert_is_less_than(radar_number, NUM_RADARS)
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)
    error_checking.assert_is_boolean(raise_error_if_missing)

    saliency_file_name = '{0:s}/{1:s}/saliency_{2:s}_radar{3:d}.nc'.format(
        top_directory_name, valid_date_string[:4], valid_date_string,
        radar_number
    )

    if os.path.isfile(saliency_file_name) or not raise_error_if_missing:
        return saliency_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        saliency_file_name
    )
    raise ValueError(error_string)


def file_name_to_date(saliency_file_name):
    """Parses date from name of saliency file.

    :param saliency_file_name: File name created by `find_file`.
    :return: valid_date_string: Valid date (format "yyyymmdd").
    """

    error_checking.assert_is_string(saliency_file_name)

    pathless_file_name = os.path.split(saliency_file_name)[-1]
    valid_date_string = pathless_file_name.split('_')[1]
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    return valid_date_string


def file_name_to_radar_num(saliency_file_name):
    """Parses radar number from name of saliency file.

    :param saliency_file_name: File name created by `find_file`.
    :return: radar_number: Radar number (non-negative integer).
    """

    error_checking.assert_is_string(saliency_file_name)

    pathless_file_name = os.path.split(saliency_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    radar_word = extensionless_file_name.split('_')[-1]

    assert radar_word.startswith('radar')

    radar_number = int(radar_word.replace('radar', ''))
    error_checking.assert_is_geq(radar_number, 0)

    return radar_number


def write_file(
        netcdf_file_name, saliency_matrix, valid_times_unix_sec,
        latitudes_deg_n, longitudes_deg_e, model_file_name, is_layer_output,
        layer_name, neuron_indices, ideal_activation, multiply_by_input):
    """Writes saliency maps to NetCDF file.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    T = number of lag times
    C = number of channels

    :param netcdf_file_name: Path to output file.
    :param saliency_matrix: numpy array of saliency values.  Must be
        E x M x N x C or E x M x N x T x C.
    :param valid_times_unix_sec: length-E numpy array of valid times.
    :param latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :param longitudes_deg_e: length-N numpy array of longitudes (deg E).
    :param model_file_name: Path to file with neural net used to create saliency
        maps (readable by `neural_net.read_model`).
    :param is_layer_output: Boolean flag.  If True, target layer is output
        layer.
    :param layer_name: See doc for `check_metadata`.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
    :param multiply_by_input: Same.
    """

    # Check input args.
    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_boolean(is_layer_output)

    check_metadata(
        layer_name=layer_name, neuron_indices=neuron_indices,
        ideal_activation=ideal_activation, multiply_by_input=multiply_by_input
    )

    error_checking.assert_is_numpy_array(valid_times_unix_sec, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(valid_times_unix_sec)

    error_checking.assert_is_numpy_array(latitudes_deg_n, num_dimensions=1)
    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg_n)
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(latitudes_deg_n), 0.
    )

    error_checking.assert_is_numpy_array(longitudes_deg_e, num_dimensions=1)
    error_checking.assert_is_valid_lng_numpy_array(
        longitudes_deg_e, positive_in_west_flag=True
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(longitudes_deg_e), 0.
    )

    num_examples = len(valid_times_unix_sec)
    num_grid_rows = len(latitudes_deg_n)
    num_grid_columns = len(longitudes_deg_e)

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix)
    num_saliency_dim = len(saliency_matrix.shape)
    error_checking.assert_is_geq(num_saliency_dim, 4)
    error_checking.assert_is_leq(num_saliency_dim, 5)

    has_time_dimension = num_saliency_dim == 5

    expected_dim = (num_examples, num_grid_rows, num_grid_columns)
    if has_time_dimension:
        expected_dim += (saliency_matrix.shape[3],)

    expected_dim += (saliency_matrix.shape[-1],)
    expected_dim = numpy.array(expected_dim, dtype=int)
    error_checking.assert_is_numpy_array(
        saliency_matrix, exact_dimensions=expected_dim
    )

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(IS_LAYER_OUTPUT_KEY, int(is_layer_output))
    dataset_object.setncattr(LAYER_NAME_KEY, layer_name)
    dataset_object.setncattr(NEURON_INDICES_KEY, neuron_indices)
    dataset_object.setncattr(IDEAL_ACTIVATION_KEY, ideal_activation)
    dataset_object.setncattr(MULTIPLY_BY_INPUT_KEY, int(multiply_by_input))

    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(ROW_DIMENSION_KEY, saliency_matrix.shape[1])
    dataset_object.createDimension(
        COLUMN_DIMENSION_KEY, saliency_matrix.shape[2]
    )
    dataset_object.createDimension(
        CHANNEL_DIMENSION_KEY, saliency_matrix.shape[-1]
    )

    if has_time_dimension:
        dataset_object.createDimension(
            LAG_TIME_DIMENSION_KEY, saliency_matrix.shape[3]
        )

    dataset_object.createVariable(
        VALID_TIMES_KEY, datatype=numpy.int32, dimensions=EXAMPLE_DIMENSION_KEY
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
        EXAMPLE_DIMENSION_KEY, ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY
    )
    if has_time_dimension:
        these_dim += (LAG_TIME_DIMENSION_KEY,)
    these_dim += (CHANNEL_DIMENSION_KEY,)

    dataset_object.createVariable(
        SALIENCY_MATRIX_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[SALIENCY_MATRIX_KEY][:] = saliency_matrix

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads saliency maps from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: saliency_dict: Dictionary with the following keys.
    saliency_dict['saliency_matrix']: See doc for `write_file`.
    saliency_dict['valid_times_unix_sec']: Same.
    saliency_dict['latitudes_deg_n']: Same.
    saliency_dict['longitudes_deg_e']: Same.
    saliency_dict['model_file_name']: Same.
    saliency_dict['is_layer_output']: Same.
    saliency_dict['layer_name']: Same.
    saliency_dict['neuron_indices']: Same.
    saliency_dict['ideal_activation']: Same.
    saliency_dict['multiply_by_input']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    saliency_dict = {
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY)),
        IS_LAYER_OUTPUT_KEY: bool(getattr(dataset_object, IS_LAYER_OUTPUT_KEY)),
        LAYER_NAME_KEY: str(getattr(dataset_object, LAYER_NAME_KEY)),
        NEURON_INDICES_KEY: numpy.array(
            getattr(dataset_object, NEURON_INDICES_KEY), dtype=int
        ),
        IDEAL_ACTIVATION_KEY: getattr(dataset_object, IDEAL_ACTIVATION_KEY),
        MULTIPLY_BY_INPUT_KEY:
            bool(getattr(dataset_object, MULTIPLY_BY_INPUT_KEY)),
        VALID_TIMES_KEY: numpy.array(
            dataset_object.variables[VALID_TIMES_KEY][:], dtype=int
        ),
        LATITUDES_KEY: numpy.array(
            dataset_object.variables[LATITUDES_KEY][:], dtype=float
        ),
        LONGITUDES_KEY: numpy.array(
            dataset_object.variables[LONGITUDES_KEY][:], dtype=float
        ),
        SALIENCY_MATRIX_KEY: numpy.array(
            dataset_object.variables[SALIENCY_MATRIX_KEY][:], dtype=float
        )
    }

    dataset_object.close()
    return saliency_dict


def write_composite_file(
        netcdf_file_name, mean_saliency_matrix, mean_denorm_predictor_matrix,
        model_file_name, is_layer_output, layer_name, neuron_indices,
        ideal_activation, multiply_by_input):
    """Writes composite saliency map (average over many cases) to NetCDF file.

    M = number of rows in grid
    N = number of columns in grid
    T = number of lag times
    C = number of channels

    :param netcdf_file_name: Path to output file.
    :param mean_saliency_matrix: numpy array of saliency values.  Must be
        M x N x C or M x N x T x C.
    :param mean_denorm_predictor_matrix: numpy array of denormalized predictors
        (same shape as `mean_saliency_matrix`).
    :param model_file_name: See doc for `write_file`.
    :param is_layer_output: Same.
    :param layer_name: Same.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
    :param multiply_by_input: Same.
    """

    # Check input args.
    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_boolean(is_layer_output)

    check_metadata(
        layer_name=layer_name, neuron_indices=neuron_indices,
        ideal_activation=ideal_activation, multiply_by_input=multiply_by_input
    )

    error_checking.assert_is_numpy_array_without_nan(mean_saliency_matrix)
    num_saliency_dim = len(mean_saliency_matrix.shape)
    error_checking.assert_is_geq(num_saliency_dim, 3)
    error_checking.assert_is_leq(num_saliency_dim, 4)

    has_time_dimension = num_saliency_dim == 4

    error_checking.assert_is_numpy_array_without_nan(
        mean_denorm_predictor_matrix
    )
    error_checking.assert_is_numpy_array(
        mean_denorm_predictor_matrix,
        exact_dimensions=numpy.array(mean_saliency_matrix.shape, dtype=int)
    )

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(IS_LAYER_OUTPUT_KEY, int(is_layer_output))
    dataset_object.setncattr(LAYER_NAME_KEY, layer_name)
    dataset_object.setncattr(NEURON_INDICES_KEY, neuron_indices)
    dataset_object.setncattr(IDEAL_ACTIVATION_KEY, ideal_activation)
    dataset_object.setncattr(MULTIPLY_BY_INPUT_KEY, int(multiply_by_input))

    dataset_object.createDimension(
        ROW_DIMENSION_KEY, mean_saliency_matrix.shape[0]
    )
    dataset_object.createDimension(
        COLUMN_DIMENSION_KEY, mean_saliency_matrix.shape[1]
    )
    dataset_object.createDimension(
        CHANNEL_DIMENSION_KEY, mean_saliency_matrix.shape[-1]
    )

    if has_time_dimension:
        dataset_object.createDimension(
            LAG_TIME_DIMENSION_KEY, mean_saliency_matrix.shape[-2]
        )

    these_dim = (ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    if has_time_dimension:
        these_dim += (LAG_TIME_DIMENSION_KEY,)
    these_dim += (CHANNEL_DIMENSION_KEY,)

    dataset_object.createVariable(
        MEAN_SALIENCY_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[MEAN_SALIENCY_KEY][:] = mean_saliency_matrix

    dataset_object.createVariable(
        MEAN_DENORM_PREDICTORS_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[MEAN_DENORM_PREDICTORS_KEY][:] = (
        mean_denorm_predictor_matrix
    )

    dataset_object.close()


def read_composite_file(netcdf_file_name):
    """Reads composite saliency map from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: mean_saliency_dict: Dictionary with the following keys.
    mean_saliency_dict['mean_saliency_matrix']: See doc for
        `write_composite_file`.
    mean_saliency_dict['mean_denorm_predictor_matrix']: Same.
    mean_saliency_dict['model_file_name']: Same.
    mean_saliency_dict['is_layer_output']: Same.
    mean_saliency_dict['layer_name']: Same.
    mean_saliency_dict['neuron_indices']: Same.
    mean_saliency_dict['ideal_activation']: Same.
    mean_saliency_dict['multiply_by_input']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    mean_saliency_dict = {
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY)),
        IS_LAYER_OUTPUT_KEY: bool(getattr(dataset_object, IS_LAYER_OUTPUT_KEY)),
        LAYER_NAME_KEY: str(getattr(dataset_object, LAYER_NAME_KEY)),
        NEURON_INDICES_KEY: numpy.array(
            getattr(dataset_object, NEURON_INDICES_KEY), dtype=int
        ),
        IDEAL_ACTIVATION_KEY: getattr(dataset_object, IDEAL_ACTIVATION_KEY),
        MULTIPLY_BY_INPUT_KEY:
            bool(getattr(dataset_object, MULTIPLY_BY_INPUT_KEY)),
        MEAN_SALIENCY_KEY: numpy.array(
            dataset_object.variables[MEAN_SALIENCY_KEY][:], dtype=float
        ),
        MEAN_DENORM_PREDICTORS_KEY: numpy.array(
            dataset_object.variables[MEAN_DENORM_PREDICTORS_KEY][:], dtype=float
        )
    }

    dataset_object.close()
    return mean_saliency_dict
