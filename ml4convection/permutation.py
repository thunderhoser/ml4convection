"""Permutation-based importance test."""

import os
import sys
import copy
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import neural_net
import evaluation
import general_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DEFAULT_NUM_BOOTSTRAP_REPS = 1000
NUM_EXAMPLES_PER_BATCH = 32
MAX_EXAMPLES_PER_DAY = 144 * 4

NO_TEST_ENUM = 0
FORWARD_TEST_ENUM = 1
BACKWARDS_TEST_ENUM = 2
TEST_TYPE_ENUMS = numpy.array(
    [NO_TEST_ENUM, FORWARD_TEST_ENUM, BACKWARDS_TEST_ENUM], dtype=int
)

ORIGINAL_COST_KEY = 'orig_cost_estimates'
BEST_PREDICTORS_KEY = 'best_predictor_names'
BEST_COSTS_KEY = 'best_cost_matrix'
STEP1_PREDICTORS_KEY = 'step1_predictor_names'
STEP1_COSTS_KEY = 'step1_cost_matrix'
BACKWARDS_FLAG_KEY = 'is_backwards_test'


def _check_test_type(test_type_enum):
    """Ensures that test type is valid.

    :param test_type_enum: Integer.
    :raises: ValueError: if `test_type_enum not in TEST_TYPE_ENUMS`.
    """

    error_checking.assert_is_integer(test_type_enum)
    if test_type_enum in TEST_TYPE_ENUMS:
        return

    error_string = (
        'Test type ({0:d}) is not in list of valid types (below):\n{1:s}'
    ).format(test_type_enum, str(TEST_TYPE_ENUMS))

    raise ValueError(error_string)


def _permute_values(predictor_matrix, channel_index,
                    permuted_example_indices=None):
    """Permutes values of one channel over examples.

    E = number of examples
    C = number of channels

    :param predictor_matrix: numpy array of predictors, where first axis has
        length E and last axis has length C.
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

    predictor_matrix[..., channel_index] = (
        predictor_matrix[..., channel_index][permuted_example_indices, ...]
    )

    return predictor_matrix, permuted_example_indices


def _get_fss_components_one_batch(
        data_dict, permuted_index_matrix, test_type_enum, model_object,
        matching_distance_px, square_filter, eroded_mask_matrix):
    """Returns FSS (fractions skill score) components for one batch of data.

    E = number of examples
    C = number of channels

    :param data_dict: Dictionary returned by
        `neural_net.create_data_partial_grids`.
    :param permuted_index_matrix: See doc for `run_forward_test_one_step`.
    :param test_type_enum: Test type (must be accepted by `_check_test_type`).
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
    target_matrix = data_dict[neural_net.TARGET_MATRIX_KEY] + 0

    num_examples = predictor_matrix.shape[0]
    num_channels = predictor_matrix.shape[-1]

    actual_sse_matrix = numpy.full((num_examples, num_channels), numpy.nan)
    reference_sse_matrix = numpy.full((num_examples, num_channels), numpy.nan)
    new_permuted_index_matrix = numpy.full(
        (num_examples, num_channels), -1, dtype=int
    )

    if test_type_enum == NO_TEST_ENUM:
        prediction_matrix = neural_net.apply_model_full_grid(
            model_object=model_object, predictor_matrix=predictor_matrix,
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH, verbose=True
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

    if test_type_enum == FORWARD_TEST_ENUM:
        skip_channel_flags = numpy.any(permuted_index_matrix >= 0, axis=0)
    else:
        raise ValueError('Fuck')

    # if is_forward_step:
    #     skip_channel_flags = numpy.any(permuted_index_matrix >= 0, axis=0)
    # else:
    #     skip_channel_flags = numpy.any(permuted_index_matrix < 0, axis=0)

    for j in range(num_channels):
        if skip_channel_flags[j]:
            continue

        this_predictor_matrix, new_permuted_index_matrix[:, j] = (
            _permute_values(
                predictor_matrix=predictor_matrix + 0.,
                channel_index=j, permuted_example_indices=None
            )
        )

        this_prediction_matrix = neural_net.apply_model_full_grid(
            model_object=model_object, predictor_matrix=this_predictor_matrix,
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH, verbose=True
        )
        print(SEPARATOR_STRING)

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

    forever_permuted_channel_flags = numpy.any(
        numpy.isnan(actual_sse_matrix), axis=0
    )

    for i in range(num_bootstrap_reps):
        if num_bootstrap_reps == 1:
            these_example_indices = example_indices + 0
        else:
            these_example_indices = numpy.random.choice(
                example_indices, size=num_examples, replace=True
            )

        for j in range(num_channels):
            if forever_permuted_channel_flags[j]:
                continue

            this_actual_sse = numpy.sum(
                actual_sse_matrix[these_example_indices, j]
            )
            this_reference_sse = numpy.sum(
                reference_sse_matrix[these_example_indices, j]
            )

            cost_matrix[j, i] = this_actual_sse / this_reference_sse

    return cost_matrix


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
            model_object, data_option_dict, valid_date_strings,
            test_type_enum, permuted_index_matrix, num_bootstrap_reps):
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
        :param test_type_enum: Test type (must be accepted by
            `_check_test_type`).
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
        _check_test_type(test_type_enum)

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
                    data_dict=this_data_dict,
                    permuted_index_matrix=this_permuted_index_matrix,
                    test_type_enum=test_type_enum,
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

        if test_type_enum == NO_TEST_ENUM:
            actual_sse_matrix = actual_sse_matrix[:, [0]]
            reference_sse_matrix = reference_sse_matrix[:, [0]]
            new_permuted_index_matrix = new_permuted_index_matrix[:, [0]]

        cost_matrix = _bootstrap_fss_cost(
            actual_sse_matrix=actual_sse_matrix,
            reference_sse_matrix=reference_sse_matrix,
            num_bootstrap_reps=num_bootstrap_reps
        )

        return cost_matrix, new_permuted_index_matrix

    return cost_function


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

    if permuted_index_matrix is None:
        num_channels = len(data_option_dict[neural_net.BAND_NUMBERS_KEY])
        permuted_forever_flags = numpy.full(num_channels, 0, dtype=bool)
    else:
        permuted_forever_flags = numpy.any(permuted_index_matrix >= 0, axis=0)

    if numpy.all(permuted_forever_flags):
        return None

    # Housekeeping.
    permuted_cost_matrix, new_permuted_index_matrix = cost_function(
        model_object=model_object, data_option_dict=data_option_dict,
        valid_date_strings=valid_date_strings,
        test_type_enum=FORWARD_TEST_ENUM,
        permuted_index_matrix=permuted_index_matrix,
        num_bootstrap_reps=num_bootstrap_reps
    )

    mean_costs = numpy.mean(permuted_cost_matrix, axis=1)
    best_cost = numpy.max(mean_costs)
    best_channel_index = numpy.argmax(mean_costs)

    best_band_number = (
        data_option_dict[neural_net.BAND_NUMBERS_KEY][best_channel_index]
    )
    print('Best cost = {0:.4f} ... best band number = {1:d}'.format(
        best_cost, best_band_number
    ))

    permuted_index_matrix[:, best_channel_index] = (
        new_permuted_index_matrix[:, best_channel_index]
    )

    if permuted_index_matrix is None:
        permuted_index_matrix = numpy.full(
            new_permuted_index_matrix.shape, -1, dtype=int
        )

    return {
        'permuted_index_matrix': permuted_index_matrix,
        'permuted_cost_matrix': permuted_cost_matrix
    }


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
    orig_cost_estimates = cost_function(
        model_object=model_object, data_option_dict=data_option_dict,
        valid_date_strings=valid_date_strings,
        test_type_enum=NO_TEST_ENUM, permuted_index_matrix=None,
        num_bootstrap_reps=num_bootstrap_reps
    )[0][:, 0]

    print('Original cost = {0:.4f}'.format(
        numpy.mean(orig_cost_estimates)
    ))

    step1_predictor_names = None
    step1_cost_matrix = None
    best_predictor_names = [''] * num_channels
    best_cost_matrix = numpy.full((num_channels, num_bootstrap_reps), numpy.nan)

    permuted_index_matrix = None

    for k in range(num_channels):
        this_result_dict = _run_forward_test_one_step(
            model_object=model_object, data_option_dict=data_option_dict,
            valid_date_strings=valid_date_strings, cost_function=cost_function,
            permuted_index_matrix=permuted_index_matrix,
            num_bootstrap_reps=num_bootstrap_reps
        )

        permuted_index_matrix = this_result_dict['permuted_index_matrix']
        permuted_cost_matrix = this_result_dict['permuted_cost_matrix']

        this_best_index = numpy.argmax(
            numpy.mean(permuted_cost_matrix, axis=1)
        )
        best_predictor_names[k] = predictor_names[this_best_index]
        best_cost_matrix[k, :] = permuted_cost_matrix[this_best_index, :]

        print('Best predictor at step {0:d} = {1:s} (cost = {2:.4f})'.format(
            k + 1, best_predictor_names[k], numpy.mean(best_cost_matrix[k, :])
        ))

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
