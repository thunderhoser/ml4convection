"""Evaluation methods for uncertainty quantification (UQ)."""

import numpy
import netCDF4
from scipy.signal import convolve2d
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4convection.io import prediction_io
from ml4convection.utils import general_utils

ERROR_FUNCTION_KEY = 'error_function_name'
UNCERTAINTY_FUNCTION_KEY = 'uncertainty_function_name'
DISCARD_FRACTION_DIM_KEY = 'discard_fraction'
DISCARD_FRACTIONS_KEY = 'discard_fractions'
ERROR_VALUES_KEY = 'error_values'

HALF_WINDOW_SIZE_KEY = 'half_window_size_px'
USE_MEDIAN_KEY = 'use_median'
SPREAD_SKILL_BIN_DIM_KEY = 'bin'
MEAN_PREDICTION_STDEVS_KEY = 'mean_prediction_stdevs'
RMSE_VALUES_KEY = 'rmse_values'


def _get_squared_errors(prediction_dict, half_window_size_px, use_median):
    """Returns squared errors.

    E = number of examples (time steps)
    M = number of rows in grid
    N = number of columns in grid

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :param half_window_size_px: Half-width (pixels) for smoother.
    :param use_median: Boolean flag.  If True (False), will use median (mean) of
        each predictive distribution.
    :return: squared_error_matrix: E-by-M-by-N numpy array of squared errors.
    """

    structure_matrix = general_utils.get_structure_matrix(
        half_window_size_px
    )
    weight_matrix = numpy.full(
        structure_matrix.shape, 1. / structure_matrix.size
    )

    if use_median:
        forecast_prob_matrix = numpy.median(
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY], axis=-1
        )
    else:
        forecast_prob_matrix = numpy.mean(
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY], axis=-1
        )

    target_matrix = prediction_dict[prediction_io.TARGET_MATRIX_KEY]
    num_examples = target_matrix.shape[0]
    squared_error_matrix = numpy.full(forecast_prob_matrix.shape, numpy.nan)

    for i in range(num_examples):
        this_smoothed_target_matrix = convolve2d(
            target_matrix[i, ...].astype(float), weight_matrix, mode='same'
        )
        this_smoothed_prob_matrix = convolve2d(
            forecast_prob_matrix[i, ...], weight_matrix, mode='same'
        )
        squared_error_matrix[i, ...] = (
            (this_smoothed_target_matrix - this_smoothed_prob_matrix) ** 2
        )

    return squared_error_matrix


def get_fss_error_function(half_window_size_px, use_median):
    """Creates error function to compute fractions skill score (FSS).

    :param half_window_size_px: Neighbourhood half-width (pixels).
    :param use_median: Boolean flag.  If True (False), will use median (mean) of
        each predictive distribution.
    :return: error_function: Function handle.
    """

    def error_function(prediction_dict, eroded_eval_mask_matrix):
        """Computes FSS with a given neighbourhood half-width.

        E = number of examples (time steps)
        M = number of rows in grid
        N = number of columns in grid

        :param prediction_dict: Dictionary in format returned by
            `prediction_io.read_file`.
        :param eroded_eval_mask_matrix: E-by-M-by-N numpy array of Boolean
            values, indicating which grid points should be used for evaluation.
            This mask must already have been eroded with the relevant
            neighbourhood distance.
        :return: fss_value: FSS (scalar).
        """

        structure_matrix = general_utils.get_structure_matrix(
            half_window_size_px
        )
        weight_matrix = numpy.full(
            structure_matrix.shape, 1. / structure_matrix.size
        )

        if use_median:
            forecast_prob_matrix = numpy.median(
                prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY], axis=-1
            )
        else:
            forecast_prob_matrix = numpy.mean(
                prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY], axis=-1
            )

        target_matrix = prediction_dict[prediction_io.TARGET_MATRIX_KEY]
        num_examples = target_matrix.shape[0]

        actual_sse = 0.
        reference_sse = 0.

        for i in range(num_examples):
            this_smoothed_target_matrix = convolve2d(
                target_matrix[i, ...].astype(float), weight_matrix, mode='same'
            )
            this_smoothed_prob_matrix = convolve2d(
                forecast_prob_matrix[i, ...], weight_matrix, mode='same'
            )

            this_smoothed_prob_matrix[
                eroded_eval_mask_matrix[i, ...].astype(bool) == False
            ] = numpy.nan

            actual_sse += numpy.nansum(
                (this_smoothed_target_matrix - this_smoothed_prob_matrix) ** 2
            )
            reference_sse += numpy.nansum(
                this_smoothed_target_matrix ** 2 +
                this_smoothed_prob_matrix ** 2
            )

        return 1. - actual_sse / reference_sse

    return error_function


def get_stdev_uncertainty_function():
    """Creates function to compute stdev of predictive distribution.

    :return: uncertainty_function: Function handle.
    """

    def uncertainty_function(prediction_dict):
        """Computes stdev of predictive distribution for each point/time.

        E = number of examples (time steps)
        M = number of rows in grid
        N = number of columns in grid

        :param prediction_dict: Dictionary in format returned by
            `prediction_io.read_file`.
        :return: stdev_matrix: E-by-M-by-N matrix with stdev of predictive
            distribution for each point/time.
        """

        return numpy.std(
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY],
            ddof=1, axis=-1
        )

    return uncertainty_function


def run_discard_test(
        prediction_dict, discard_fractions, eroded_eval_mask_matrix,
        error_function, uncertainty_function):
    """Runs the discard test.

    F = number of discard fractions
    E = number of examples
    M = number of rows in grid
    N = number of columns in grid

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :param discard_fractions: length-(F - 1) numpy array of discard fractions,
        ranging from (0, 1).  This method will use 0 as the lowest discard
        fraction.
    :param eroded_eval_mask_matrix: E-by-M-by-N numpy array of Boolean values,
        indicating which grid points should be used for evaluation.  If
        `error_function` uses neighbourhood evaluation, this mask should already
        have been eroded with the relevant neighbourhood distance.

    :param error_function: Function with the following inputs and outputs...
    Input: prediction_dict: See above.
    Input: eroded_eval_mask_matrix: See above.
    Output: error_value: Scalar value of error metric.

    :param uncertainty_function: Function with the following inputs and
        outputs...
    Input: prediction_dict: See above.
    Output: uncertainty_matrix: E-by-M-by-N numpy array with values of
        uncertainty metric.  The metric must be oriented so that higher value =
        more uncertainty.

    :return: discard_fractions: length-F numpy array of discard fractions,
        sorted in increasing order.
    :return: error_values: length-F numpy array of corresponding error values.
    """

    # Check input args.
    expected_dim = numpy.array(
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY].shape[:3],
        dtype=int
    )
    error_checking.assert_is_boolean_numpy_array(eroded_eval_mask_matrix)
    error_checking.assert_is_numpy_array(
        eroded_eval_mask_matrix, exact_dimensions=expected_dim
    )

    error_checking.assert_is_numpy_array(discard_fractions, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(discard_fractions, 0.)
    error_checking.assert_is_less_than_numpy_array(discard_fractions, 1.)

    discard_fractions = numpy.concatenate((
        numpy.array([0.]),
        discard_fractions
    ))

    num_fractions = len(discard_fractions)
    assert num_fractions >= 2

    # Do actual stuff.
    uncertainty_matrix = uncertainty_function(prediction_dict)
    uncertainty_matrix[eroded_eval_mask_matrix == False] = numpy.nan

    discard_fractions = numpy.sort(discard_fractions)
    error_by_discard_fraction = numpy.full(num_fractions, numpy.nan)

    for k in range(num_fractions):
        this_percentile_level = 100 * (1 - discard_fractions[k])
        this_inverted_mask_matrix = (
            uncertainty_matrix >
            numpy.nanpercentile(uncertainty_matrix, this_percentile_level)
        )
        eroded_eval_mask_matrix[this_inverted_mask_matrix] = False

        error_by_discard_fraction[k] = error_function(
            prediction_dict, eroded_eval_mask_matrix
        )

    return discard_fractions, error_by_discard_fraction


def get_spread_vs_skill(prediction_dict, bin_edge_prediction_stdevs,
                        half_window_size_px, eval_mask_matrix, use_median):
    """Computes model spread vs. model skill.

    B = number of bins
    M = number of rows in grid
    N = number of columns in grid

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :param bin_edge_prediction_stdevs: length-(B - 1) numpy array of bin
        cutoffs.  Each is a standard deviation for the predictive distribution.
        Ultimately, there will be B + 1 edges; this method will use 0 as the
        lowest edge and 1 as the highest edge.
    :param half_window_size_px: Half-width (pixels) for neighbourhood
        evaluation.
    :param eval_mask_matrix: M-by-N numpy array of Boolean flags, indicating
        which pixels to use for evaluation.  This mask should *not* already be
        eroded with the given neighbourhood half-width.  This method will do the
        erosion for you.
    :param use_median: Boolean flag.  If True (False), will use median (mean) of
        each predictive distribution.
    :return: mean_prediction_stdevs: length-B numpy array, where the [i]th
        entry is the mean standard deviation of predictive distributions in the
        [i]th bin.
    :return: rmse_values: length-B numpy array, where the [i]th entry is the
        root mean squared error of central (mean or median) predictions in the
        [i]th bin.
    """

    # Check input args.
    error_checking.assert_is_boolean(use_median)

    expected_dim = numpy.array(
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY].shape[1:3],
        dtype=int
    )
    error_checking.assert_is_boolean_numpy_array(eval_mask_matrix)
    error_checking.assert_is_numpy_array(
        eval_mask_matrix, exact_dimensions=expected_dim
    )

    error_checking.assert_is_numpy_array(
        bin_edge_prediction_stdevs, num_dimensions=1
    )
    error_checking.assert_is_greater_numpy_array(bin_edge_prediction_stdevs, 0.)
    error_checking.assert_is_less_than_numpy_array(
        bin_edge_prediction_stdevs, 1.
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(bin_edge_prediction_stdevs), 0.
    )

    bin_edge_prediction_stdevs = numpy.concatenate((
        numpy.array([0.]),
        bin_edge_prediction_stdevs,
        numpy.array([numpy.inf])
    ))

    num_bins = len(bin_edge_prediction_stdevs) - 1
    assert num_bins >= 2

    forecast_prob_matrix = prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY]
    num_examples = forecast_prob_matrix.shape[0]
    num_monte_carlo_iters = forecast_prob_matrix.shape[3]
    assert num_monte_carlo_iters > 2

    # Do actual stuff.
    eroded_eval_mask_matrix = general_utils.erode_binary_matrix(
        binary_matrix=eval_mask_matrix, buffer_distance_px=half_window_size_px
    )
    eroded_eval_mask_matrix = numpy.repeat(
        a=numpy.expand_dims(eroded_eval_mask_matrix, axis=0),
        axis=0, repeats=num_examples
    )

    prediction_stdev_matrix = numpy.std(forecast_prob_matrix, axis=3, ddof=1)
    prediction_stdev_matrix[eroded_eval_mask_matrix == False] = numpy.nan
    squared_error_matrix = _get_squared_errors(
        prediction_dict=prediction_dict,
        half_window_size_px=half_window_size_px, use_median=use_median
    )

    mean_prediction_stdevs = numpy.full(num_bins, numpy.nan)
    rmse_values = numpy.full(num_bins, numpy.nan)

    for k in range(num_bins):
        these_indices = numpy.where(numpy.logical_and(
            prediction_stdev_matrix >= bin_edge_prediction_stdevs[k],
            prediction_stdev_matrix < bin_edge_prediction_stdevs[k + 1]
        ))

        mean_prediction_stdevs[k] = numpy.mean(
            prediction_stdev_matrix[these_indices]
        )
        rmse_values[k] = numpy.sqrt(numpy.mean(
            squared_error_matrix[these_indices]
        ))

    return mean_prediction_stdevs, rmse_values


def write_discard_results(
        netcdf_file_name, discard_fractions, error_values, error_function_name,
        uncertainty_function_name):
    """Writes results of discard test to NetCDF file.

    :param netcdf_file_name: Path to output file.
    :param discard_fractions: length-F numpy array of discard fractions, ranging
        from [0, 1).
    :param error_values: length-F numpy array of corresponding error values.
    :param error_function_name: Name of error function (string).  This will be
        used later for plotting.
    :param uncertainty_function_name: Name of uncertainty function (string).
        This will be used later for plotting.
    """

    # Check input args.
    error_checking.assert_is_numpy_array(discard_fractions, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(discard_fractions, 0.)
    error_checking.assert_is_less_than_numpy_array(discard_fractions, 1.)
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(discard_fractions), 0.
    )

    assert discard_fractions[0] <= 1e-6
    num_fractions = len(discard_fractions)
    assert num_fractions >= 2

    error_checking.assert_is_numpy_array(
        error_values,
        exact_dimensions=numpy.array([num_fractions], dtype=int)
    )

    error_checking.assert_is_string(error_function_name)
    error_checking.assert_is_string(uncertainty_function_name)

    # Write file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(ERROR_FUNCTION_KEY, error_function_name)
    dataset_object.setncattr(
        UNCERTAINTY_FUNCTION_KEY, uncertainty_function_name
    )

    dataset_object.createDimension(DISCARD_FRACTION_DIM_KEY, num_fractions)

    dataset_object.createVariable(
        DISCARD_FRACTIONS_KEY, datatype=numpy.float32,
        dimensions=DISCARD_FRACTION_DIM_KEY
    )
    dataset_object.variables[DISCARD_FRACTIONS_KEY][:] = discard_fractions

    dataset_object.createVariable(
        ERROR_VALUES_KEY, datatype=numpy.float32,
        dimensions=DISCARD_FRACTION_DIM_KEY
    )
    dataset_object.variables[ERROR_VALUES_KEY][:] = error_values

    dataset_object.close()


def read_discard_results(netcdf_file_name):
    """Reads results of discard test from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: discard_dict: Dictionary with the following keys.
    discard_dict['discard_fractions']: See doc for `write_discard_results`.
    discard_dict['error_values']: Same.
    discard_dict['error_function_name']: Same.
    discard_dict['uncertainty_function_name']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    discard_dict = {
        ERROR_FUNCTION_KEY: str(getattr(dataset_object, ERROR_FUNCTION_KEY)),
        UNCERTAINTY_FUNCTION_KEY: str(
            getattr(dataset_object, UNCERTAINTY_FUNCTION_KEY)
        ),
        DISCARD_FRACTIONS_KEY: numpy.array(
            dataset_object.variables[DISCARD_FRACTIONS_KEY][:], dtype=float
        ),
        ERROR_VALUES_KEY: numpy.array(
            dataset_object.variables[ERROR_VALUES_KEY][:], dtype=float
        )
    }

    dataset_object.close()
    return discard_dict


def write_spread_vs_skill(
        netcdf_file_name, mean_prediction_stdevs, rmse_values,
        half_window_size_px, use_median):
    """Writes spread vs. skill to NetCDF file.

    :param netcdf_file_name: Path to output file.
    :param mean_prediction_stdevs: See output doc for `get_spread_vs_skill`.
    :param rmse_values: Same.
    :param half_window_size_px: See input doc for `get_spread_vs_skill`.
    :param use_median: Same.
    """

    # Check input args.
    error_checking.assert_is_numpy_array(
        mean_prediction_stdevs, num_dimensions=1
    )
    error_checking.assert_is_geq_numpy_array(
        mean_prediction_stdevs, 0., allow_nan=True
    )
    error_checking.assert_is_leq_numpy_array(
        mean_prediction_stdevs, 1., allow_nan=True
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(mean_prediction_stdevs), 0., allow_nan=True
    )

    num_bins = len(mean_prediction_stdevs)
    assert num_bins >= 2

    error_checking.assert_is_geq_numpy_array(rmse_values, 0., allow_nan=True)
    error_checking.assert_is_leq_numpy_array(rmse_values, 1., allow_nan=True)
    error_checking.assert_is_numpy_array(
        rmse_values,
        exact_dimensions=numpy.array([num_bins], dtype=int)
    )

    error_checking.assert_is_geq(half_window_size_px, 0)
    error_checking.assert_is_boolean(use_median)

    # Write file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(HALF_WINDOW_SIZE_KEY, half_window_size_px)
    dataset_object.setncattr(USE_MEDIAN_KEY, int(use_median))

    dataset_object.createDimension(SPREAD_SKILL_BIN_DIM_KEY, num_bins)

    dataset_object.createVariable(
        MEAN_PREDICTION_STDEVS_KEY, datatype=numpy.float32,
        dimensions=SPREAD_SKILL_BIN_DIM_KEY
    )
    dataset_object.variables[MEAN_PREDICTION_STDEVS_KEY][:] = (
        mean_prediction_stdevs
    )

    dataset_object.createVariable(
        RMSE_VALUES_KEY, datatype=numpy.float32,
        dimensions=SPREAD_SKILL_BIN_DIM_KEY
    )
    dataset_object.variables[RMSE_VALUES_KEY][:] = rmse_values

    dataset_object.close()


def read_spread_vs_skill(netcdf_file_name):
    """Reads spread vs. skill from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: spread_skill_dict: Dictionary with the following keys.
    spread_skill_dict['mean_prediction_stdevs']: See doc for
        `write_spread_vs_skill`.
    spread_skill_dict['rmse_values']: Same.
    spread_skill_dict['half_window_size_px']: Same.
    spread_skill_dict['use_median']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    discard_dict = {
        HALF_WINDOW_SIZE_KEY: getattr(dataset_object, HALF_WINDOW_SIZE_KEY),
        USE_MEDIAN_KEY: bool(getattr(dataset_object, USE_MEDIAN_KEY)),
        MEAN_PREDICTION_STDEVS_KEY: numpy.array(
            dataset_object.variables[MEAN_PREDICTION_STDEVS_KEY][:], dtype=float
        ),
        RMSE_VALUES_KEY: numpy.array(
            dataset_object.variables[RMSE_VALUES_KEY][:], dtype=float
        )
    }

    dataset_object.close()
    return discard_dict
