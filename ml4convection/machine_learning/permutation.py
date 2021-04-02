"""Permutation-based importance test."""

import copy
import numpy
import netCDF4
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4convection.machine_learning import neural_net
from ml4convection.utils import evaluation
from ml4convection.utils import general_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

DEFAULT_NUM_BOOTSTRAP_REPS = 1000
NUM_EXAMPLES_PER_BATCH = 32
MAX_EXAMPLES_PER_DAY = 144 * 4

ORIGINAL_COST_KEY = 'orig_cost_estimates'
BEST_PREDICTORS_KEY = 'best_predictor_names'
BEST_COSTS_KEY = 'best_cost_matrix'
STEP1_PREDICTORS_KEY = 'step1_predictor_names'
STEP1_COSTS_KEY = 'step1_cost_matrix'
BACKWARDS_FLAG_KEY = 'is_backwards_test'

PERMUTED_INDICES_KEY = 'permuted_index_matrix'
PERMUTED_COSTS_KEY = 'permuted_cost_matrix'
DEPERMUTED_COSTS_KEY = 'depermuted_cost_matrix'

PREDICTOR_DIM_KEY = 'predictor'
BOOTSTRAP_REPLICATE_DIM_KEY = 'bootstrap_replicate'
PREDICTOR_CHAR_DIM_KEY = 'predictor_name_char'


def _permute_values(predictor_matrix, data_option_dict, channel_index,
                    permuted_example_indices=None):
    """Permutes values of one channel over examples.

    E = number of examples
    C = number of channels

    :param predictor_matrix: numpy array of predictors, where first axis has
        length E and last axis has length C.
    :param data_option_dict: See input doc for
        `neural_net.create_data_partial_grids`.
    :param channel_index: Index of channel to permute.
    :param permuted_example_indices: length-E numpy array of permuted indices.
        If None, this method will permute randomly.
    :return: predictor_matrix: Same as input but after permutation.
    :return: permuted_example_indices: If input was None, this will be an array
        created on the fly.  Otherwise, this array will be the same as the
        input.
    """

    if permuted_example_indices is None:
        permuted_example_indices = numpy.random.permutation(
            predictor_matrix.shape[0]
        )

    if not data_option_dict[neural_net.INCLUDE_TIME_DIM_KEY]:
        num_lag_times = len(data_option_dict[neural_net.LAG_TIMES_KEY])

        predictor_matrix = neural_net.predictor_matrix_from_keras(
            predictor_matrix=predictor_matrix, num_lag_times=num_lag_times
        )
        predictor_matrix = neural_net.predictor_matrix_to_keras(
            predictor_matrix=predictor_matrix, num_lag_times=num_lag_times,
            add_time_dimension=True
        )

    predictor_matrix[..., channel_index] = (
        predictor_matrix[..., channel_index][permuted_example_indices, ...]
    )

    if not data_option_dict[neural_net.INCLUDE_TIME_DIM_KEY]:
        num_lag_times = len(data_option_dict[neural_net.LAG_TIMES_KEY])

        predictor_matrix = neural_net.predictor_matrix_from_keras(
            predictor_matrix=predictor_matrix, num_lag_times=num_lag_times
        )
        predictor_matrix = neural_net.predictor_matrix_to_keras(
            predictor_matrix=predictor_matrix, num_lag_times=num_lag_times,
            add_time_dimension=False
        )

    return predictor_matrix, permuted_example_indices


def _depermute_values(predictor_matrix, data_option_dict, channel_index,
                      permuted_example_indices):
    """Depermutes (cleans up) values of one channel over examples.

    :param predictor_matrix: See doc for `_permute_values`.
    :param data_option_dict: Same.
    :param channel_index: Same.
    :param permuted_example_indices: Same.
    :return: predictor_matrix: Same.
    """

    sort_indices = numpy.empty_like(permuted_example_indices)
    sort_indices[permuted_example_indices] = numpy.arange(
        len(permuted_example_indices)
    )

    if not data_option_dict[neural_net.INCLUDE_TIME_DIM_KEY]:
        num_lag_times = len(data_option_dict[neural_net.LAG_TIMES_KEY])

        predictor_matrix = neural_net.predictor_matrix_from_keras(
            predictor_matrix=predictor_matrix, num_lag_times=num_lag_times
        )
        predictor_matrix = neural_net.predictor_matrix_to_keras(
            predictor_matrix=predictor_matrix, num_lag_times=num_lag_times,
            add_time_dimension=True
        )

    predictor_matrix[..., channel_index] = (
        predictor_matrix[..., channel_index][sort_indices, ...]
    )

    if not data_option_dict[neural_net.INCLUDE_TIME_DIM_KEY]:
        num_lag_times = len(data_option_dict[neural_net.LAG_TIMES_KEY])

        predictor_matrix = neural_net.predictor_matrix_from_keras(
            predictor_matrix=predictor_matrix, num_lag_times=num_lag_times
        )
        predictor_matrix = neural_net.predictor_matrix_to_keras(
            predictor_matrix=predictor_matrix, num_lag_times=num_lag_times,
            add_time_dimension=False
        )

    return predictor_matrix


def _get_fss_components_one_batch(
        data_dict, data_option_dict, permuted_index_matrix, is_forward_test,
        is_start_of_test, model_object, matching_distance_px, square_filter,
        eroded_mask_matrix):
    """Returns FSS (fractions skill score) components for one batch of data.

    E = number of examples
    C = number of channels

    :param data_dict: Dictionary returned by
        `neural_net.create_data_partial_grids`.
    :param data_option_dict: See input doc for
        `neural_net.create_data_partial_grids`.
    :param permuted_index_matrix: See doc for `run_forward_test_one_step`.
    :param is_forward_test: Boolean flag.  If True (False), running forward
        (backwards) test.
    :param is_start_of_test: Boolean flag.  If True, this is the start of the
        test (step 0).
    :param model_object: See doc for `run_forward_test_one_step`.
    :param matching_distance_px: Same.
    :param square_filter: Same.
    :param eroded_mask_matrix: Same.
    :return: actual_sse_matrix: E-by-C numpy array of actual SSE (sum of squared
        errors) values.
    :return: reference_sse_matrix: E-by-C numpy array of reference SSE values.
    :return: new_permuted_index_matrix: Same as input `permuted_index_matrix`
        but only for channels that were permuted by this method.
    """

    predictor_matrix = data_dict[neural_net.PREDICTOR_MATRIX_KEY] + 0.
    target_matrix = data_dict[neural_net.TARGET_MATRIX_KEY][..., 0] + 0

    num_examples = predictor_matrix.shape[0]
    num_channels = len(data_option_dict[neural_net.BAND_NUMBERS_KEY])

    actual_sse_matrix = numpy.full((num_examples, num_channels), numpy.nan)
    reference_sse_matrix = numpy.full((num_examples, num_channels), numpy.nan)
    new_permuted_index_matrix = numpy.full(
        (num_examples, num_channels), -1, dtype=int
    )

    if is_start_of_test:
        if not is_forward_test:
            print(
                'Permuting all predictors in preparation for backwards test...'
            )

            for j in range(num_channels):
                predictor_matrix, new_permuted_index_matrix[:, j] = (
                    _permute_values(
                        predictor_matrix=predictor_matrix,
                        data_option_dict=data_option_dict, channel_index=j,
                        permuted_example_indices=None
                    )
                )

        print('Applying model to fully {0:s} data...\n'.format(
            'clean' if is_forward_test else 'dirty'
        ))

        prediction_matrix = neural_net.apply_model_full_grid(
            model_object=model_object, predictor_matrix=predictor_matrix,
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH, verbose=False
        )

        for i in range(num_examples):
            spatial_actual_sse_matrix, spatial_reference_sse_matrix = (
                evaluation._get_fss_components_one_time(
                    actual_target_matrix=target_matrix[i, ...],
                    probability_matrix=prediction_matrix[i, ...],
                    matching_distance_px=matching_distance_px,
                    eroded_eval_mask_matrix=eroded_mask_matrix,
                    square_filter=square_filter
                )
            )

            actual_sse_matrix[i, 0] = numpy.nansum(spatial_actual_sse_matrix)
            reference_sse_matrix[i, 0] = numpy.nansum(
                spatial_reference_sse_matrix
            )

        return (
            actual_sse_matrix, reference_sse_matrix, new_permuted_index_matrix
        )

    permuted_channel_flags = numpy.any(permuted_index_matrix >= 0, axis=0)

    if is_forward_test:
        skip_channel_flags = numpy.any(permuted_index_matrix >= 0, axis=0)
    else:
        skip_channel_flags = numpy.any(permuted_index_matrix < 0, axis=0)

    for j in range(num_channels):
        if not permuted_channel_flags[j]:
            continue

        predictor_matrix = _permute_values(
            predictor_matrix=predictor_matrix,
            data_option_dict=data_option_dict, channel_index=j,
            permuted_example_indices=permuted_index_matrix[:, j]
        )[0]

    for j in range(num_channels):
        if skip_channel_flags[j]:
            continue

        if is_forward_test:
            print('Permuting channel {0:d} of {1:d}...'.format(
                j + 1, num_channels
            ))

            this_predictor_matrix, new_permuted_index_matrix[:, j] = (
                _permute_values(
                    predictor_matrix=predictor_matrix + 0.,
                    data_option_dict=data_option_dict, channel_index=j,
                    permuted_example_indices=None
                )
            )

            print('Applying model to permuted data...\n')
        else:
            print('Cleaning channel {0:d} of {1:d}...'.format(
                j + 1, num_channels
            ))

            this_predictor_matrix = _depermute_values(
                predictor_matrix=predictor_matrix + 0.,
                data_option_dict=data_option_dict, channel_index=j,
                permuted_example_indices=permuted_index_matrix[:, j]
            )

            print('Applying model to cleaned data...\n')

        this_prediction_matrix = neural_net.apply_model_full_grid(
            model_object=model_object, predictor_matrix=this_predictor_matrix,
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH, verbose=False
        )

        for i in range(num_examples):

            # TODO(thunderhoser): Make this method public.
            spatial_actual_sse_matrix, spatial_reference_sse_matrix = (
                evaluation._get_fss_components_one_time(
                    actual_target_matrix=target_matrix[i, ...],
                    probability_matrix=this_prediction_matrix[i, ...],
                    matching_distance_px=matching_distance_px,
                    eroded_eval_mask_matrix=eroded_mask_matrix,
                    square_filter=square_filter
                )
            )

            actual_sse_matrix[i, j] = numpy.nansum(spatial_actual_sse_matrix)
            reference_sse_matrix[i, j] = numpy.nansum(
                spatial_reference_sse_matrix
            )

    return actual_sse_matrix, reference_sse_matrix, new_permuted_index_matrix


def _bootstrap_fss_cost(actual_sse_matrix, reference_sse_matrix,
                        num_bootstrap_reps):
    """Bootstraps FSS (fractions skill score) cost function.

    E = number of examples
    C = number of channels
    B = number of replicates for bootstrapping

    :param actual_sse_matrix: E-by-C numpy array of actual SSE (sum of squared
        errors) values.
    :param reference_sse_matrix: E-by-C numpy array of reference SSE values.
    :param num_bootstrap_reps: Number of replicates for bootstrapping.
    :return: cost_matrix: C-by-B numpy array of negative-FSS values.  If the
        [k]th channel is already permanently permuted, cost_matrix[:, k] will be
        all NaN.
    """

    num_channels = actual_sse_matrix.shape[1]
    cost_matrix = numpy.full((num_channels, num_bootstrap_reps), numpy.nan)

    num_examples = actual_sse_matrix.shape[0]
    example_indices = numpy.linspace(
        0, num_examples - 1, num=num_examples, dtype=int
    )

    skip_channel_flags = numpy.any(numpy.isnan(actual_sse_matrix), axis=0)

    for i in range(num_bootstrap_reps):
        if num_bootstrap_reps == 1:
            these_example_indices = example_indices + 0
        else:
            these_example_indices = numpy.random.choice(
                example_indices, size=num_examples, replace=True
            )

        for j in range(num_channels):
            if skip_channel_flags[j]:
                continue

            this_actual_sse = numpy.sum(
                actual_sse_matrix[these_example_indices, j]
            )
            this_reference_sse = numpy.sum(
                reference_sse_matrix[these_example_indices, j]
            )

            cost_matrix[j, i] = this_actual_sse / this_reference_sse

    return cost_matrix


def _run_forward_test_one_step(
        model_object, data_option_dict, valid_date_strings, cost_function,
        permuted_index_matrix, num_bootstrap_reps):
    """Runs one step of the forward permutation test.

    E = number of examples
    C = number of channels
    B = number of replicates for bootstrapping

    :param model_object: See doc for `run_forward_test`.
    :param data_option_dict: Same.
    :param valid_date_strings: Same.
    :param cost_function: Same.
    :param permuted_index_matrix: E-by-C numpy array of integers.  If the
        [k]th channel is currently permuted, permuted_index_matrix[:, k]
        contains indices used for permutation.  Otherwise,
        permuted_index_matrix[:, k] contains all negative numbers.
    :param num_bootstrap_reps: Number of replicates for bootstrapping.
    :return: result_dict: Dictionary with the following keys.
    result_dict['permuted_index_matrix']: Same as input but with
        different values in the matrices.
    result_dict['permuted_cost_matrix']: C-by-B numpy array of costs after
        permutation in this step.
    """

    permuted_forever_flags = numpy.any(permuted_index_matrix >= 0, axis=0)
    if numpy.all(permuted_forever_flags):
        return None

    permuted_cost_matrix, new_permuted_index_matrix = cost_function(
        model_object=model_object, data_option_dict=data_option_dict,
        valid_date_strings=valid_date_strings,
        is_forward_test=True, is_start_of_test=False,
        permuted_index_matrix=permuted_index_matrix,
        num_bootstrap_reps=num_bootstrap_reps
    )

    mean_costs = numpy.mean(permuted_cost_matrix, axis=1)
    best_cost = numpy.nanmax(mean_costs)
    best_channel_index = numpy.nanargmax(mean_costs)

    print('Best cost = {0:.4f} ... best channel = {1:d}'.format(
        best_cost, best_channel_index + 1
    ))

    permuted_index_matrix[:, best_channel_index] = (
        new_permuted_index_matrix[:, best_channel_index]
    )

    return {
        PERMUTED_INDICES_KEY: permuted_index_matrix,
        PERMUTED_COSTS_KEY: permuted_cost_matrix
    }


def _run_backwards_test_one_step(
        model_object, data_option_dict, valid_date_strings, cost_function,
        permuted_index_matrix, num_bootstrap_reps):
    """Runs one step of the backwards permutation test.

    C = number of channels
    B = number of replicates for bootstrapping

    :param model_object: See doc for `_run_forward_test_one_step`.
    :param data_option_dict: Same.
    :param valid_date_strings: Same.
    :param cost_function: Same.
    :param permuted_index_matrix: Same.
    :param num_bootstrap_reps: Same.
    :return: result_dict: Dictionary with the following keys.
    result_dict['permuted_index_matrix']: Same as input but with
        different values in the matrices.
    result_dict['depermuted_cost_matrix']: C-by-B numpy array of costs after
        depermutation in this step.
    """

    depermuted_forever_flags = numpy.any(permuted_index_matrix < 0, axis=0)
    if numpy.all(depermuted_forever_flags):
        return None

    depermuted_cost_matrix = cost_function(
        model_object=model_object, data_option_dict=data_option_dict,
        valid_date_strings=valid_date_strings,
        is_forward_test=False, is_start_of_test=False,
        permuted_index_matrix=permuted_index_matrix,
        num_bootstrap_reps=num_bootstrap_reps
    )[0]

    mean_costs = numpy.mean(depermuted_cost_matrix, axis=1)
    best_cost = numpy.nanmin(mean_costs)
    best_channel_index = numpy.nanargmin(mean_costs)

    print('Best cost = {0:.4f} ... best channel = {1:d}'.format(
        best_cost, best_channel_index + 1
    ))

    permuted_index_matrix[:, best_channel_index] = -1

    return {
        PERMUTED_INDICES_KEY: permuted_index_matrix,
        DEPERMUTED_COSTS_KEY: depermuted_cost_matrix
    }


def make_fss_cost_function(
        matching_distance_px, square_filter, model_metadata_dict):
    """Creates FSS (fractions skill score) cost function.

    :param matching_distance_px: Matching distance (number of pixels).
    :param square_filter: Boolean flag.  If True, will square FSS filter.  If
        False, will use circular filter.
    :param model_metadata_dict: Dictionary returned by
        `neural_net.read_metafile`.
    :return: cost_function: Function (see below).
    """

    error_checking.assert_is_geq(matching_distance_px, 0.)
    error_checking.assert_is_boolean(square_filter)

    mask_matrix = model_metadata_dict[neural_net.MASK_MATRIX_KEY]
    eroded_mask_matrix = general_utils.erode_binary_matrix(
        binary_matrix=mask_matrix, buffer_distance_px=matching_distance_px
    )

    def cost_function(
            model_object, data_option_dict, valid_date_strings, is_forward_test,
            is_start_of_test, permuted_index_matrix, num_bootstrap_reps):
        """FSS cost function.

        E = number of examples
        C = number of channels
        B = number of replicates for bootstrapping cost
        D = number of valid dates

        :param model_object: Model (trained instance of `keras.models.Model` or
            `keras.models.Sequential`).
        :param data_option_dict: See input doc for
            `neural_net.create_data_partial_grids`.
        :param valid_date_strings: length-D list of valid dates (format
            "yyyymmdd").
        :param is_forward_test: Boolean flag.  If True (False), running forward
            (backwards) test.
        :param is_start_of_test: Boolean flag.  If True, this is the start of
            the test (step 0).
        :param permuted_index_matrix: E-by-C numpy array of integers.  If the
            [k]th channel is currently permuted, permuted_index_matrix[:, k]
            contains indices used for permutation.  Otherwise,
            permuted_index_matrix[:, k] contains all negative numbers.
        :param num_bootstrap_reps: Number of replicates for bootstrapping cost.
        :return: cost_matrix: C-by-B numpy array of negative-FSS values.  If
            the [k]th channel is already permanently permuted, cost_matrix[:, k]
            will be all NaN.
        :return: new_permuted_index_matrix: Same as input
            `permuted_index_matrix`, but indices pertain to permutations done by
            this method only.
        """

        # Check input args.
        error_checking.assert_is_numpy_array(
            numpy.array(valid_date_strings), num_dimensions=1
        )
        error_checking.assert_is_boolean(is_forward_test)
        error_checking.assert_is_boolean(is_start_of_test)

        num_channels = len(data_option_dict[neural_net.BAND_NUMBERS_KEY])

        if permuted_index_matrix is None:
            num_examples = MAX_EXAMPLES_PER_DAY * len(valid_date_strings)
        else:
            error_checking.assert_is_numpy_array(
                permuted_index_matrix, num_dimensions=2
            )
            error_checking.assert_is_integer_numpy_array(permuted_index_matrix)

            num_examples = permuted_index_matrix.shape[0]
            expected_dim = numpy.array([num_examples, num_channels], dtype=int)

            error_checking.assert_is_numpy_array(
                permuted_index_matrix, exact_dimensions=expected_dim
            )

        error_checking.assert_is_integer(num_bootstrap_reps)
        error_checking.assert_is_geq(num_bootstrap_reps, 1)

        # Housekeeping.
        actual_sse_matrix = numpy.full((num_examples, num_channels), numpy.nan)
        reference_sse_matrix = numpy.full(
            (num_examples, num_channels), numpy.nan
        )
        new_permuted_index_matrix = numpy.full(
            (num_examples, num_channels), -1, dtype=int
        )

        num_examples_read = 0

        # Do actual stuff.
        for this_date_string in valid_date_strings:
            data_option_dict[neural_net.VALID_DATE_KEY] = this_date_string
            data_dict_by_radar = neural_net.create_data_partial_grids(
                option_dict=data_option_dict, return_coords=False
            )
            print(MINOR_SEPARATOR_STRING)

            if data_dict_by_radar is None:
                continue

            for this_data_dict in data_dict_by_radar:
                if len(list(this_data_dict.keys())) == 0:
                    continue

                this_num_examples = (
                    this_data_dict[neural_net.TARGET_MATRIX_KEY].shape[0]
                )

                first_index = num_examples_read + 0
                last_index = first_index + this_num_examples
                num_examples_read = last_index + 0

                if permuted_index_matrix is None:
                    this_permuted_index_matrix = numpy.full(
                        (this_num_examples, num_channels), -1, dtype=int
                    )
                else:
                    this_permuted_index_matrix = (
                        permuted_index_matrix[first_index:last_index, :]
                    )

                (
                    actual_sse_matrix[first_index:last_index, :],
                    reference_sse_matrix[first_index:last_index, :],
                    new_permuted_index_matrix[first_index:last_index, :]
                ) = _get_fss_components_one_batch(
                    data_dict=this_data_dict, data_option_dict=data_option_dict,
                    permuted_index_matrix=this_permuted_index_matrix,
                    is_forward_test=is_forward_test,
                    is_start_of_test=is_start_of_test,
                    model_object=model_object,
                    matching_distance_px=matching_distance_px,
                    square_filter=square_filter,
                    eroded_mask_matrix=eroded_mask_matrix
                )

        actual_sse_matrix = actual_sse_matrix[:num_examples_read, :]
        reference_sse_matrix = reference_sse_matrix[:num_examples_read, :]
        new_permuted_index_matrix = (
            new_permuted_index_matrix[:num_examples_read, :]
        )

        cost_matrix = _bootstrap_fss_cost(
            actual_sse_matrix=actual_sse_matrix,
            reference_sse_matrix=reference_sse_matrix,
            num_bootstrap_reps=num_bootstrap_reps
        )

        return cost_matrix, new_permuted_index_matrix

    return cost_function


def run_forward_test(
        model_object, data_option_dict, valid_date_strings, cost_function,
        num_bootstrap_reps=DEFAULT_NUM_BOOTSTRAP_REPS):
    """Runs forward versions (single- and multi-pass) of permutation test.

    C = number of channels
    B = number of replicates for bootstrapping

    :param model_object: Trained model (trained instance of `keras.models.Model`
        or `keras.models.Sequential`).
    :param data_option_dict: See input doc for
        `neural_net.create_data_partial_grids`.
    :param valid_date_strings: 1-D list of valid dates (format "yyyymmdd").
    :param cost_function: Cost function.  Must be negatively oriented (i.e.,
        lower is better), with the following inputs and outputs.

        Input: model_object: See documentation under `make_fss_cost_function`.
        Input: data_option_dict: Same.
        Input: valid_date_strings: Same.
        Input: permuted_index_matrix_by_date: Same.
        Input: num_bootstrap_reps: Same.
        Output: cost_matrix: Same.
        Output: new_permuted_index_matrix_by_date: Same.

    :param num_bootstrap_reps: Number of bootstrap replicates (i.e., number of
        times to estimate cost).

    :return: result_dict: Dictionary with the following keys.
    result_dict['orig_cost_estimates']: length-B numpy array with estimates of
        original cost (before permutation).
    result_dict['best_band_number']: length-C list with best band number at each
        step.
    result_dict['best_cost_matrix']: C-by-B numpy array of costs after
        permutation at each step.
    result_dict['step1_band_numbers']: length-C list with predictors in order
        that they were permuted in step 1.
    result_dict['step1_cost_matrix']: C-by-B numpy array of costs after
        permutation in step 1.
    result_dict['is_backwards_test']: Boolean flag (always False for this
        method).
    """

    band_numbers = data_option_dict[neural_net.BAND_NUMBERS_KEY]
    predictor_names = ['Band {0:d}'.format(b) for b in band_numbers]
    num_channels = len(band_numbers)

    # Find original cost (before permutation).
    orig_cost_estimates, permuted_index_matrix = cost_function(
        model_object=model_object, data_option_dict=data_option_dict,
        valid_date_strings=valid_date_strings,
        is_forward_test=True, is_start_of_test=True,
        permuted_index_matrix=None, num_bootstrap_reps=num_bootstrap_reps
    )

    orig_cost_estimates = orig_cost_estimates[0, :]
    permuted_index_matrix = numpy.full(
        permuted_index_matrix.shape, -1, dtype=int
    )

    print('Original cost (before permutation) = {0:.4f}'.format(
        numpy.mean(orig_cost_estimates)
    ))
    print(SEPARATOR_STRING)

    step1_predictor_names = None
    step1_cost_matrix = None
    best_predictor_names = [''] * num_channels
    best_cost_matrix = numpy.full((num_channels, num_bootstrap_reps), numpy.nan)

    for k in range(num_channels):
        this_result_dict = _run_forward_test_one_step(
            model_object=model_object, data_option_dict=data_option_dict,
            valid_date_strings=valid_date_strings, cost_function=cost_function,
            permuted_index_matrix=permuted_index_matrix,
            num_bootstrap_reps=num_bootstrap_reps
        )

        permuted_index_matrix = this_result_dict[PERMUTED_INDICES_KEY]
        permuted_cost_matrix = this_result_dict[PERMUTED_COSTS_KEY]

        this_best_index = numpy.nanargmax(
            numpy.nanmean(permuted_cost_matrix, axis=1)
        )
        best_predictor_names[k] = predictor_names[this_best_index]
        best_cost_matrix[k, :] = permuted_cost_matrix[this_best_index, :]

        print('Best predictor at step {0:d} = {1:s} (cost = {2:.4f})'.format(
            k + 1, best_predictor_names[k], numpy.mean(best_cost_matrix[k, :])
        ))
        print(SEPARATOR_STRING)

        if k > 0:
            continue

        step1_predictor_names = copy.deepcopy(predictor_names)
        step1_cost_matrix = permuted_cost_matrix + 0.

    return {
        ORIGINAL_COST_KEY: orig_cost_estimates,
        BEST_PREDICTORS_KEY: best_predictor_names,
        BEST_COSTS_KEY: best_cost_matrix,
        STEP1_PREDICTORS_KEY: step1_predictor_names,
        STEP1_COSTS_KEY: step1_cost_matrix,
        BACKWARDS_FLAG_KEY: False
    }


def run_backwards_test(
        model_object, data_option_dict, valid_date_strings, cost_function,
        num_bootstrap_reps=DEFAULT_NUM_BOOTSTRAP_REPS):
    """Runs backwards versions (single- and multi-pass) of permutation test.

    C = number of channels
    B = number of replicates for bootstrapping

    :param model_object: See doc for `run_forward_test`.
    :param data_option_dict: Same.
    :param valid_date_strings: Same.
    :param cost_function: Same.
    :param num_bootstrap_reps: Same.

    :return: result_dict: Dictionary with the following keys.
    result_dict['orig_cost_estimates']: length-B numpy array with estimates of
        original cost (before *de*permutation).
    result_dict['best_band_number']: length-C list with best band number at each
        step.
    result_dict['best_cost_matrix']: C-by-B numpy array of costs after
        *de*permutation at each step.
    result_dict['step1_band_numbers']: length-C list with predictors in order
        that they were *de*permuted in step 1.
    result_dict['step1_cost_matrix']: C-by-B numpy array of costs after
        *de*permutation in step 1.
    result_dict['is_backwards_test']: Boolean flag (always True for this
        method).
    """

    band_numbers = data_option_dict[neural_net.BAND_NUMBERS_KEY]
    predictor_names = ['Band {0:d}'.format(b) for b in band_numbers]
    num_channels = len(band_numbers)

    # Find original cost (before *de*permutation).
    orig_cost_estimates, permuted_index_matrix = cost_function(
        model_object=model_object, data_option_dict=data_option_dict,
        valid_date_strings=valid_date_strings,
        is_forward_test=False, is_start_of_test=True,
        permuted_index_matrix=None, num_bootstrap_reps=num_bootstrap_reps
    )
    orig_cost_estimates = orig_cost_estimates[0, :]

    print('Original cost (before *de*permutation) = {0:.4f}'.format(
        numpy.mean(orig_cost_estimates)
    ))
    print(SEPARATOR_STRING)

    step1_predictor_names = None
    step1_cost_matrix = None
    best_predictor_names = [''] * num_channels
    best_cost_matrix = numpy.full((num_channels, num_bootstrap_reps), numpy.nan)

    for k in range(num_channels):
        this_result_dict = _run_backwards_test_one_step(
            model_object=model_object, data_option_dict=data_option_dict,
            valid_date_strings=valid_date_strings, cost_function=cost_function,
            permuted_index_matrix=permuted_index_matrix,
            num_bootstrap_reps=num_bootstrap_reps
        )

        permuted_index_matrix = this_result_dict[PERMUTED_INDICES_KEY]
        depermuted_cost_matrix = this_result_dict[DEPERMUTED_COSTS_KEY]

        this_best_index = numpy.nanargmin(
            numpy.nanmean(depermuted_cost_matrix, axis=1)
        )
        best_predictor_names[k] = predictor_names[this_best_index]
        best_cost_matrix[k, :] = depermuted_cost_matrix[this_best_index, :]

        print('Best predictor at step {0:d} = {1:s} (cost = {2:.4f})'.format(
            k + 1, best_predictor_names[k], numpy.mean(best_cost_matrix[k, :])
        ))
        print(SEPARATOR_STRING)

        if k > 0:
            continue

        step1_predictor_names = copy.deepcopy(predictor_names)
        step1_cost_matrix = depermuted_cost_matrix + 0.

    return {
        ORIGINAL_COST_KEY: orig_cost_estimates,
        BEST_PREDICTORS_KEY: best_predictor_names,
        BEST_COSTS_KEY: best_cost_matrix,
        STEP1_PREDICTORS_KEY: step1_predictor_names,
        STEP1_COSTS_KEY: step1_cost_matrix,
        BACKWARDS_FLAG_KEY: True
    }


def write_file(result_dict, netcdf_file_name):
    """Writes results of permutation test to NetCDF file.

    :param result_dict: Dictionary created by `run_forward_test` or
        `run_backwards_test`.
    :param netcdf_file_name: Path to output file.
    """

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(
        BACKWARDS_FLAG_KEY, int(result_dict[BACKWARDS_FLAG_KEY])
    )

    num_predictors = result_dict[BEST_COSTS_KEY].shape[0]
    num_bootstrap_reps = result_dict[BEST_COSTS_KEY].shape[1]

    dataset_object.createDimension(PREDICTOR_DIM_KEY, num_predictors)
    dataset_object.createDimension(
        BOOTSTRAP_REPLICATE_DIM_KEY, num_bootstrap_reps
    )

    best_predictor_names = result_dict[BEST_PREDICTORS_KEY]
    step1_predictor_names = result_dict[STEP1_PREDICTORS_KEY]
    num_predictor_chars = numpy.max(numpy.array([
        len(n) for n in best_predictor_names + step1_predictor_names
    ]))

    dataset_object.createDimension(PREDICTOR_CHAR_DIM_KEY, num_predictor_chars)

    this_string_format = 'S{0:d}'.format(num_predictor_chars)
    best_predictor_names_char_array = netCDF4.stringtochar(numpy.array(
        best_predictor_names, dtype=this_string_format
    ))

    dataset_object.createVariable(
        BEST_PREDICTORS_KEY, datatype='S1',
        dimensions=(PREDICTOR_DIM_KEY, PREDICTOR_CHAR_DIM_KEY)
    )
    dataset_object.variables[BEST_PREDICTORS_KEY][:] = numpy.array(
        best_predictor_names_char_array
    )

    dataset_object.createVariable(
        BEST_COSTS_KEY, datatype=numpy.float32,
        dimensions=(PREDICTOR_DIM_KEY, BOOTSTRAP_REPLICATE_DIM_KEY)
    )
    dataset_object.variables[BEST_COSTS_KEY][:] = result_dict[BEST_COSTS_KEY]

    this_string_format = 'S{0:d}'.format(num_predictor_chars)
    step1_predictor_names_char_array = netCDF4.stringtochar(numpy.array(
        step1_predictor_names, dtype=this_string_format
    ))

    dataset_object.createVariable(
        STEP1_PREDICTORS_KEY, datatype='S1',
        dimensions=(PREDICTOR_DIM_KEY, PREDICTOR_CHAR_DIM_KEY)
    )
    dataset_object.variables[STEP1_PREDICTORS_KEY][:] = numpy.array(
        step1_predictor_names_char_array
    )

    dataset_object.createVariable(
        STEP1_COSTS_KEY, datatype=numpy.float32,
        dimensions=(PREDICTOR_DIM_KEY, BOOTSTRAP_REPLICATE_DIM_KEY)
    )
    dataset_object.variables[STEP1_COSTS_KEY][:] = result_dict[STEP1_COSTS_KEY]

    dataset_object.createVariable(
        ORIGINAL_COST_KEY, datatype=numpy.float32,
        dimensions=BOOTSTRAP_REPLICATE_DIM_KEY
    )
    dataset_object.variables[ORIGINAL_COST_KEY][:] = (
        result_dict[ORIGINAL_COST_KEY]
    )

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads results of permutation test from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_dict: See doc for `run_forward_test` or
        `run_backwards_test`.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    result_dict = {
        ORIGINAL_COST_KEY: dataset_object.variables[ORIGINAL_COST_KEY][:],
        BEST_PREDICTORS_KEY: [
            str(n) for n in netCDF4.chartostring(
                dataset_object.variables[BEST_PREDICTORS_KEY][:]
            )
        ],
        BEST_COSTS_KEY: dataset_object.variables[BEST_COSTS_KEY][:],
        STEP1_PREDICTORS_KEY: [
            str(n) for n in netCDF4.chartostring(
                dataset_object.variables[STEP1_PREDICTORS_KEY][:]
            )
        ],
        STEP1_COSTS_KEY: dataset_object.variables[STEP1_COSTS_KEY][:],
        BACKWARDS_FLAG_KEY: bool(getattr(dataset_object, BACKWARDS_FLAG_KEY))
    }

    dataset_object.close()

    return result_dict
