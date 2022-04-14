"""Evaluation methods for uncertainty quantification (UQ)."""

import numpy
from gewittergefahr.gg_utils import error_checking
from ml4convection.io import prediction_io


def get_spread_vs_skill(prediction_dict, bin_edge_prediction_stdevs,
                        use_mean_to_compute_error):
    """Computes model spread vs. model skill.

    B = number of bins

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :param bin_edge_prediction_stdevs: length-(B - 1) numpy array of bin
        cutoffs.  Each is a standard deviation for the predictive distribution.
        Ultimately, there will be B + 1 edges; this method will use 0 as the
        lowest edge and 1 as the highest edge.
    :param use_mean_to_compute_error: Boolean flag.  If True (False), for each
        example this method will use the mean (median) to compute the error
        between predictions and observations.
    :return: mean_prediction_stdevs: length-B numpy array, where the [i]th
        entry is the mean standard deviation of predictive distributions in the
        [i]th bin.
    :return: prediction_standard_errors: length-B numpy array, where the [i]th
        entry is the standard deviation of mean absolute errors in the [i]th
        bin.
    """

    # Check input args.
    error_checking.assert_is_boolean(use_mean_to_compute_error)
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
    target_matrix = prediction_dict[prediction_io.TARGET_MATRIX_KEY]

    num_monte_carlo_iters = forecast_prob_matrix.shape[3]
    assert num_monte_carlo_iters > 2

    # Do actual stuff.
    absolute_errors = numpy.ravel(numpy.absolute(
        numpy.mean(forecast_prob_matrix, axis=3) -
        target_matrix
    ))
    prediction_stdevs = numpy.ravel(
        numpy.std(forecast_prob_matrix, axis=3, ddof=1)
    )

    mean_prediction_stdevs = numpy.full(num_bins, numpy.nan)
    prediction_standard_errors = numpy.full(num_bins, numpy.nan)

    for k in range(num_bins):
        these_indices = numpy.where(numpy.logical_and(
            prediction_stdevs >= bin_edge_prediction_stdevs[k],
            prediction_stdevs < bin_edge_prediction_stdevs[k + 1]
        ))[0]

        mean_prediction_stdevs[k] = numpy.mean(
            prediction_stdevs[these_indices]
        )
        prediction_standard_errors[k] = numpy.mean(
            absolute_errors[these_indices]
        )

    return mean_prediction_stdevs, prediction_standard_errors
