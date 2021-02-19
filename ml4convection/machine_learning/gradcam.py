"""Methods for computing, reading, and writing class-activation maps."""

import os
import numpy
import netCDF4
from keras import backend as K
from scipy.interpolate import (
    UnivariateSpline, RectBivariateSpline, RegularGridInterpolator
)
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4convection.utils import radar_utils

DATE_FORMAT = '%Y%m%d'
NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)

EXAMPLE_DIMENSION_KEY = 'example'
ROW_DIMENSION_KEY = 'row'
COLUMN_DIMENSION_KEY = 'column'

MODEL_FILE_KEY = 'model_file_name'
ACTIVATION_LAYER_KEY = 'activation_layer_name'
OUTPUT_LAYER_KEY = 'output_layer_name'
OUTPUT_ROW_KEY = 'output_row'
OUTPUT_COLUMN_KEY = 'output_column'

VALID_TIMES_KEY = 'valid_times_unix_sec'
LATITUDES_KEY = 'latitudes_deg_n'
LONGITUDES_KEY = 'longitudes_deg_e'
CLASS_ACTIVATIONS_KEY = 'class_activation_matrix'


def _normalize_tensor(input_tensor):
    """Normalizes tensor by its L2 norm.

    :param input_tensor: Unnormalized tensor.
    :return: output_tensor: Normalized tensor.
    """

    rms_tensor = K.sqrt(K.mean(K.square(input_tensor)))
    return input_tensor / (rms_tensor + K.epsilon())


def _upsample_cam(class_activation_matrix, new_dimensions):
    """Upsamples class-activation matrix (CAM).

    CAM may be 1-D, 2-D, or 3-D.

    :param class_activation_matrix: numpy array containing 1-D, 2-D, or 3-D
        class-activation nao.
    :param new_dimensions: numpy array of new dimensions.  If matrix is
        {1D, 2D, 3D}, this must be a length-{1, 2, 3} array, respectively.
    :return: class_activation_matrix: Upsampled version of input.
    """

    num_rows_new = new_dimensions[0]
    row_indices_new = numpy.linspace(
        1, num_rows_new, num=num_rows_new, dtype=float
    )
    row_indices_orig = numpy.linspace(
        1, num_rows_new, num=class_activation_matrix.shape[0], dtype=float
    )

    if len(new_dimensions) == 1:
        # interp_object = UnivariateSpline(
        #     x=row_indices_orig, y=numpy.ravel(class_activation_matrix),
        #     k=1, s=0
        # )

        interp_object = UnivariateSpline(
            x=row_indices_orig, y=numpy.ravel(class_activation_matrix),
            k=3, s=0
        )

        return interp_object(row_indices_new)

    num_columns_new = new_dimensions[1]
    column_indices_new = numpy.linspace(
        1, num_columns_new, num=num_columns_new, dtype=float
    )
    column_indices_orig = numpy.linspace(
        1, num_columns_new, num=class_activation_matrix.shape[1], dtype=float
    )

    if len(new_dimensions) == 2:
        interp_object = RectBivariateSpline(
            x=row_indices_orig, y=column_indices_orig,
            z=class_activation_matrix, kx=3, ky=3, s=0
        )

        return interp_object(x=row_indices_new, y=column_indices_new, grid=True)

    num_heights_new = new_dimensions[2]
    height_indices_new = numpy.linspace(
        1, num_heights_new, num=num_heights_new, dtype=float
    )
    height_indices_orig = numpy.linspace(
        1, num_heights_new, num=class_activation_matrix.shape[2], dtype=float
    )

    interp_object = RegularGridInterpolator(
        points=(row_indices_orig, column_indices_orig, height_indices_orig),
        values=class_activation_matrix, method='linear'
    )

    column_index_matrix, row_index_matrix, height_index_matrix = (
        numpy.meshgrid(column_indices_new, row_indices_new, height_indices_new)
    )
    query_point_matrix = numpy.stack(
        (row_index_matrix, column_index_matrix, height_index_matrix), axis=-1
    )

    return interp_object(query_point_matrix)


def check_metadata(
        activation_layer_name, output_layer_name, output_row, output_column):
    """Checks metadata for errors.

    :param activation_layer_name: Name of activation layer.
    :param output_layer_name: Name of output layer.  This layer should output
        either probabilities (activation outputs) or pseudo-probabilities
        (activation inputs).
    :param output_row: Class activation will be computed with respect to output
        at this grid row (non-negative integer).
    :param output_column: Class activation will be computed with respect to
        output at this grid column (non-negative integer).
    """

    error_checking.assert_is_string(activation_layer_name)
    error_checking.assert_is_string(output_layer_name)
    error_checking.assert_is_integer(output_row)
    error_checking.assert_is_geq(output_row, 0)
    error_checking.assert_is_integer(output_column)
    error_checking.assert_is_geq(output_column, 0)


def run_gradcam(
        model_object, predictor_matrix, activation_layer_name,
        output_layer_name, output_row, output_column):
    """Runs the Grad-CAM algorithm.

    M = number of rows in grid
    N = number of columns in grid

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix: numpy array of predictors.  Must be formatted in
        the same way as for training and inference.  First two dimensions must
        be M x N.
    :param activation_layer_name: See doc for `check_metadata`.
    :param output_layer_name: Same.
    :param output_row: Same.
    :param output_column: Same.
    :return: class_activation_matrix: M-by-N numpy array of class activations.
    """

    # Check input args.
    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)
    error_checking.assert_is_numpy_array(predictor_matrix, num_dimensions=2)
    predictor_matrix = numpy.expand_dims(predictor_matrix, axis=0)

    check_metadata(
        activation_layer_name=activation_layer_name,
        output_layer_name=output_layer_name,
        output_row=output_row, output_column=output_column
    )

    # Set up loss function.
    output_tensor = model_object.get_layer(
        name=output_layer_name
    ).output[:, output_row, output_column]

    # TODO(thunderhoser): Is this right?
    # loss_tensor = (output_tensor - ideal_activation) ** 2
    loss_tensor = output_tensor

    # Set up gradient function.
    layer_activation_tensor = (
        model_object.get_layer(name=activation_layer_name).output
    )
    gradient_tensor = K.gradients(
        loss_tensor, [layer_activation_tensor]
    )[0]
    gradient_tensor = _normalize_tensor(gradient_tensor)
    gradient_function = K.function(
        [model_object.input],
        [layer_activation_tensor, gradient_tensor]
    )

    # Evaluate gradient function.
    layer_activation_matrix, gradient_matrix = gradient_function(
        [predictor_matrix]
    )
    layer_activation_matrix = layer_activation_matrix[0, ...]
    gradient_matrix = gradient_matrix[0, ...]

    # Compute class-activation map in activation layer's space.
    mean_weight_by_filter = numpy.mean(gradient_matrix, axis=(0, 1))
    class_activation_matrix = numpy.full(layer_activation_matrix.shape[:-1], 0.)
    num_filters = len(mean_weight_by_filter)

    for k in range(num_filters):
        class_activation_matrix += (
            mean_weight_by_filter[k] * layer_activation_matrix[:, k]
        )

    class_activation_matrix = _upsample_cam(
        class_activation_matrix=class_activation_matrix,
        new_dimensions=numpy.array(predictor_matrix.shape[1:], dtype=int)
    )

    return numpy.maximum(class_activation_matrix, 0.)


def find_file(
        top_directory_name, valid_date_string, radar_number,
        raise_error_if_missing=True):
    """Finds NetCDF file with class-activation maps.

    :param top_directory_name: Name of top-level directory where file is
        expected.
    :param valid_date_string: Valid date (format "yyyymmdd").
    :param radar_number: Radar number (non-negative integer).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: gradcam_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_integer(radar_number)
    error_checking.assert_is_geq(radar_number, 0)
    error_checking.assert_is_less_than(radar_number, NUM_RADARS)
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)
    error_checking.assert_is_boolean(raise_error_if_missing)

    gradcam_file_name = '{0:s}/{1:s}/gradcam_{2:s}_radar{3:d}.nc'.format(
        top_directory_name, valid_date_string[:4], valid_date_string,
        radar_number
    )

    if os.path.isfile(gradcam_file_name) or not raise_error_if_missing:
        return gradcam_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        gradcam_file_name
    )
    raise ValueError(error_string)


def file_name_to_date(gradcam_file_name):
    """Parses date from name of Grad-CAM file.

    :param gradcam_file_name: File name created by `find_file`.
    :return: valid_date_string: Valid date (format "yyyymmdd").
    """

    error_checking.assert_is_string(gradcam_file_name)

    pathless_file_name = os.path.split(gradcam_file_name)[-1]
    valid_date_string = pathless_file_name.split('_')[1]
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    return valid_date_string


def file_name_to_radar_num(gradcam_file_name):
    """Parses radar number from name of Grad-CAM file.

    :param gradcam_file_name: File name created by `find_file`.
    :return: radar_number: Radar number (non-negative integer).
    """

    error_checking.assert_is_string(gradcam_file_name)

    pathless_file_name = os.path.split(gradcam_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    radar_word = extensionless_file_name.split('_')[-1]

    assert radar_word.startswith('radar')

    radar_number = int(radar_word.replace('radar', ''))
    error_checking.assert_is_geq(radar_number, 0)

    return radar_number


def write_file(
        netcdf_file_name, class_activation_matrix, valid_times_unix_sec,
        latitudes_deg_n, longitudes_deg_e, model_file_name,
        activation_layer_name, output_layer_name, output_row, output_column):
    """Writes class-activation maps to NetCDF file.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to output file.
    :param class_activation_matrix: E-by-M-by-N numpy array of class
        activations.
    :param valid_times_unix_sec: length-E numpy array of valid times.
    :param latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :param longitudes_deg_e: length-N numpy array of longitudes (deg E).
    :param model_file_name: Path to file with neural net used to create class-
        activation maps (readable by `neural_net.read_model`).
    :param activation_layer_name: See doc for `check_metadata`.
    :param output_layer_name: Same.
    :param output_row: Same.
    :param output_column: Same.
    """

    # Check input args.
    error_checking.assert_is_string(model_file_name)

    check_metadata(
        activation_layer_name=activation_layer_name,
        output_layer_name=output_layer_name,
        output_row=output_row, output_column=output_column
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
    expected_dim = numpy.array(
        [num_examples, num_grid_rows, num_grid_columns], dtype=int
    )

    error_checking.assert_is_numpy_array_without_nan(class_activation_matrix)
    error_checking.assert_is_geq_numpy_array(class_activation_matrix, 0.)
    error_checking.assert_is_numpy_array(
        class_activation_matrix, exact_dimensions=expected_dim
    )

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(ACTIVATION_LAYER_KEY, activation_layer_name)
    dataset_object.setncattr(OUTPUT_LAYER_KEY, output_layer_name)
    dataset_object.setncattr(OUTPUT_ROW_KEY, output_row)
    dataset_object.setncattr(OUTPUT_COLUMN_KEY, output_column)

    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(ROW_DIMENSION_KEY, num_grid_rows)
    dataset_object.createDimension(COLUMN_DIMENSION_KEY, num_grid_columns)

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
    dataset_object.createVariable(
        CLASS_ACTIVATIONS_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[CLASS_ACTIVATIONS_KEY][:] = class_activation_matrix

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads class-activation maps from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: gradcam_dict: Dictionary with the following keys.
    gradcam_dict['class_activation_matrix']: See doc for `write_file`.
    gradcam_dict['valid_times_unix_sec']: Same.
    gradcam_dict['latitudes_deg_n']: Same.
    gradcam_dict['longitudes_deg_e']: Same.
    gradcam_dict['model_file_name']: Same.
    gradcam_dict['activation_layer_name']: Same.
    gradcam_dict['output_layer_name']: Same.
    gradcam_dict['output_row']: Same.
    gradcam_dict['output_column']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    gradcam_dict = {
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY)),
        ACTIVATION_LAYER_KEY:
            str(getattr(dataset_object, ACTIVATION_LAYER_KEY)),
        OUTPUT_LAYER_KEY: str(getattr(dataset_object, OUTPUT_LAYER_KEY)),
        OUTPUT_ROW_KEY: getattr(dataset_object, OUTPUT_ROW_KEY),
        OUTPUT_COLUMN_KEY: getattr(dataset_object, OUTPUT_COLUMN_KEY),
        VALID_TIMES_KEY: numpy.array(
            dataset_object.variables[VALID_TIMES_KEY][:], dtype=int
        ),
        LATITUDES_KEY: numpy.array(
            dataset_object.variables[LATITUDES_KEY][:], dtype=float
        ),
        LONGITUDES_KEY: numpy.array(
            dataset_object.variables[LONGITUDES_KEY][:], dtype=float
        ),
        CLASS_ACTIVATIONS_KEY: numpy.array(
            dataset_object.variables[CLASS_ACTIVATIONS_KEY][:], dtype=float
        )
    }

    dataset_object.close()
    return gradcam_dict
