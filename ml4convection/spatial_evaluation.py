"""Spatially aware (not pixelwise) evaluation."""

import os
import sys
import pickle
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import prediction_io
import general_utils
import standalone_utils

ACTUAL_SSE_KEY = 'actual_sse'
REFERENCE_SSE_KEY = 'reference_sse'
NUM_VALUES_KEY = 'num_values'


def _get_fss_components(target_matrix, forecast_probability_matrix,
                        half_window_size_px, test_mode=False):
    """Returns components of fractions skill score (FSS).

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid

    :param target_matrix: E-by-M-by-N numpy array of true classes (integers in
        0...1).
    :param forecast_probability_matrix: E-by-M-by-N numpy array of event
        probabilities.
    :param half_window_size_px: Number of pixels (grid cells) in half of
        smoothing window (on either side of center).  If this argument is K, the
        window size will be (1 + 2 * K) by (1 + 2 * K).
    :param test_mode: Leave this alone.
    :return: actual_sse: Actual sum of squared errors (SSE).
    :return: reference_sse: Reference SSE.
    :return: num_values: Number of values used to compute each SSE.
    """

    weight_matrix = general_utils.create_mean_filter(
        half_num_rows=half_window_size_px,
        half_num_columns=half_window_size_px, num_channels=1
    )

    smoothed_target_matrix = standalone_utils.do_2d_convolution(
        feature_matrix=numpy.expand_dims(target_matrix, axis=-1).astype(float),
        kernel_matrix=weight_matrix, pad_edges=test_mode, stride_length_px=1
    )

    smoothed_prob_matrix = standalone_utils.do_2d_convolution(
        feature_matrix=numpy.expand_dims(forecast_probability_matrix, axis=-1),
        kernel_matrix=weight_matrix, pad_edges=test_mode, stride_length_px=1
    )

    actual_sse = numpy.sum(
        (smoothed_target_matrix - smoothed_prob_matrix) ** 2
    )
    reference_sse = numpy.sum(
        smoothed_target_matrix ** 2 + smoothed_prob_matrix ** 2
    )

    return actual_sse, reference_sse, smoothed_target_matrix.size


def _update_fss_components(half_window_sizes_px, fss_dict_by_scale,
                           prediction_file_name):
    """Updates FSS components, using one prediction file.

    S = number of scales

    :param half_window_sizes_px: See doc for `get_fractions_skill_scores`.
    :param fss_dict_by_scale: length-S list of dictionaries with the following
        keys.
    fss_dict_by_scale[k]['actual_sse']: Actual sum of squared errors (SSE) for
        [k]th scale.
    fss_dict_by_scale[k]['reference_sse']: Reference SSE for [k]th scale.
    fss_dict_by_scale[k]['num_values']: Number of values used so far to compute
        SSEs for [k]th scale.

    :param prediction_file_name: Path to input file (will be read by
        `prediction_io.read_file`).
    :return: fss_dict_by_scale: Same as input but with different values.
    """

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    num_scales = len(fss_dict_by_scale)

    for k in range(num_scales):
        this_actual_sse, this_reference_sse, this_num_values = (
            _get_fss_components(
                target_matrix=prediction_dict[prediction_io.TARGET_MATRIX_KEY],
                forecast_probability_matrix=
                prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY],
                half_window_size_px=half_window_sizes_px[k]
            )
        )

        fss_dict_by_scale[k][ACTUAL_SSE_KEY] += this_actual_sse
        fss_dict_by_scale[k][REFERENCE_SSE_KEY] += this_reference_sse
        fss_dict_by_scale[k][NUM_VALUES_KEY] += this_num_values

    return fss_dict_by_scale


def _get_fss_from_components(fss_dict):
    """Computes fractions skill score (FSS) from components.

    :param fss_dict: See doc for `_update_fss_components`.
    :return: fractions_skill_score: FSS.
    """

    actual_mse = fss_dict[ACTUAL_SSE_KEY] / fss_dict[NUM_VALUES_KEY]
    reference_mse = fss_dict[REFERENCE_SSE_KEY] / fss_dict[NUM_VALUES_KEY]
    return 1. - actual_mse / reference_mse


def get_fractions_skill_scores(prediction_file_names, half_window_sizes_px):
    """Computes fractions skill score (FSS) at each scale.

    S = number of scales

    :param prediction_file_names: 1-D list of paths to prediction files (will be
        read by `prediction_io.read_file`).
    :param half_window_sizes_px: length-S numpy array of half-window sizes for
        FSS.  For more details on the half-window size, see doc for
        `_get_fss_components`.
    :return: fractions_skill_scores: length-S numpy array of FSS values.
    """

    error_checking.assert_is_string_list(prediction_file_names)
    error_checking.assert_is_integer_numpy_array(half_window_sizes_px)
    error_checking.assert_is_numpy_array(half_window_sizes_px, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(half_window_sizes_px, 0)

    num_scales = len(half_window_sizes_px)
    fss_dict_by_scale = [dict()] * num_scales

    for k in range(num_scales):
        fss_dict_by_scale[k] = {
            ACTUAL_SSE_KEY: 0.,
            REFERENCE_SSE_KEY: 0.,
            NUM_VALUES_KEY: 0
        }

    for this_file_name in prediction_file_names:
        fss_dict_by_scale = _update_fss_components(
            half_window_sizes_px=half_window_sizes_px,
            fss_dict_by_scale=fss_dict_by_scale,
            prediction_file_name=this_file_name
        )

    return numpy.array([
        _get_fss_from_components(d) for d in fss_dict_by_scale
    ])


def write_file(fractions_skill_scores, half_window_sizes_px, pickle_file_name):
    """Writes results to Pickle file.

    S = number of scales

    :param fractions_skill_scores: length-S numpy array of scores.
    :param half_window_sizes_px: See doc for `get_fractions_skill_scores`.
    :param pickle_file_name: Path to output file.
    """

    error_checking.assert_is_integer_numpy_array(half_window_sizes_px)
    error_checking.assert_is_numpy_array(half_window_sizes_px, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(half_window_sizes_px, 0)

    num_scales = len(half_window_sizes_px)
    expected_dim = numpy.array([num_scales], dtype=int)

    error_checking.assert_is_numpy_array(
        fractions_skill_scores, exact_dimensions=expected_dim
    )
    error_checking.assert_is_geq_numpy_array(fractions_skill_scores, 0.)
    error_checking.assert_is_leq_numpy_array(fractions_skill_scores, 1.)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(fractions_skill_scores, pickle_file_handle)
    pickle.dump(half_window_sizes_px, pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads results from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: fractions_skill_scores: See doc for `write_file`.
    :return: half_window_sizes_px: Same.
    """

    error_checking.assert_file_exists(pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'rb')
    fractions_skill_scores = pickle.load(pickle_file_handle)
    half_window_sizes_px = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return fractions_skill_scores, half_window_sizes_px
