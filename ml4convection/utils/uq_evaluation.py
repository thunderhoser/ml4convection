"""Evaluation methods for uncertainty quantification (UQ)."""

import numpy
from scipy.signal import convolve2d
from gewittergefahr.gg_utils import error_checking
from ml4convection.io import prediction_io
from ml4convection.utils import general_utils


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


def get_rmse_error_function(half_window_size_px, use_median):
    """Creates function to compute root mean squared error (RMSE).

    :param half_window_size_px: Half-width (pixels) for smoother.
    :param use_median: Boolean flag.  If True (False), will use median (mean) of
        each predictive distribution.
    :return: error_function: Function handle.
    """

    def error_function(prediction_dict, eroded_eval_mask_matrix):
        """Computes RMSE.

        E = number of examples (time steps)
        M = number of rows in grid
        N = number of columns in grid

        :param prediction_dict: Dictionary in format returned by
            `prediction_io.read_file`.
        :param eroded_eval_mask_matrix: E-by-M-by-N numpy array of Boolean
            values, indicating which grid points should be used for evaluation.
            This mask must already have been eroded with the relevant smoothing
            distance.
        :return: rmse_value: RMSE (scalar).
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

        sum_of_squared_errors = 0.

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

            sum_of_squared_errors += numpy.nansum(
                (this_smoothed_target_matrix - this_smoothed_prob_matrix) ** 2
            )

        return numpy.sqrt(
            sum_of_squared_errors / numpy.sum(eroded_eval_mask_matrix == True)
        )

    return error_function


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
        ranging from (0, 1].  This method will use 0 as the lowest discard
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
