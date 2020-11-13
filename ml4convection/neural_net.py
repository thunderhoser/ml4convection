"""Methods for training and applying neural nets."""

import os
import sys
import copy
import random
import pickle
import numpy
import keras
import tensorflow.keras as tf_keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import satellite_io
import example_io
import general_utils
import radar_utils
import custom_losses
import custom_metrics

TOLERANCE = 1e-6

DAYS_TO_SECONDS = 86400
DATE_FORMAT = '%Y%m%d'
NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)

PLATEAU_PATIENCE_EPOCHS = 10
DEFAULT_LEARNING_RATE_MULTIPLIER = 0.5
PLATEAU_COOLDOWN_EPOCHS = 0
EARLY_STOPPING_PATIENCE_EPOCHS = 30
LOSS_PATIENCE = 0.

BATCH_SIZE_KEY = 'num_examples_per_batch'
MAX_DAILY_EXAMPLES_KEY = 'max_examples_per_day_in_batch'
BAND_NUMBERS_KEY = 'band_numbers'
LEAD_TIME_KEY = 'lead_time_seconds'
LAG_TIMES_KEY = 'lag_times_seconds'
FIRST_VALID_DATE_KEY = 'first_valid_date_string'
LAST_VALID_DATE_KEY = 'last_valid_date_string'
NORMALIZE_FLAG_KEY = 'normalize'
UNIFORMIZE_FLAG_KEY = 'uniformize'
PREDICTOR_DIRECTORY_KEY = 'top_predictor_dir_name'
TARGET_DIRECTORY_KEY = 'top_target_dir_name'

DEFAULT_GENERATOR_OPTION_DICT = {
    BATCH_SIZE_KEY: 256,
    MAX_DAILY_EXAMPLES_KEY: 64,
    BAND_NUMBERS_KEY: satellite_io.BAND_NUMBERS,
    LAG_TIMES_KEY: numpy.array([0], dtype=int),
    NORMALIZE_FLAG_KEY: True,
    UNIFORMIZE_FLAG_KEY: True
}

VALID_DATE_KEY = 'valid_date_string'

USE_PARTIAL_GRIDS_KEY = 'use_partial_grids'
NUM_EPOCHS_KEY = 'num_epochs'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
TRAINING_OPTIONS_KEY = 'training_option_dict'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
VALIDATION_OPTIONS_KEY = 'validation_option_dict'
EARLY_STOPPING_KEY = 'do_early_stopping'
PLATEAU_LR_MUTIPLIER_KEY = 'plateau_lr_multiplier'
CLASS_WEIGHTS_KEY = 'class_weights'
FSS_HALF_WINDOW_SIZE_KEY = 'fss_half_window_size_px'
MASK_MATRIX_KEY = 'mask_matrix'
FULL_MASK_MATRIX_KEY = 'full_mask_matrix'

METADATA_KEYS = [
    USE_PARTIAL_GRIDS_KEY, NUM_EPOCHS_KEY, NUM_TRAINING_BATCHES_KEY,
    TRAINING_OPTIONS_KEY, NUM_VALIDATION_BATCHES_KEY, VALIDATION_OPTIONS_KEY,
    EARLY_STOPPING_KEY, PLATEAU_LR_MUTIPLIER_KEY, CLASS_WEIGHTS_KEY,
    FSS_HALF_WINDOW_SIZE_KEY, MASK_MATRIX_KEY, FULL_MASK_MATRIX_KEY
]

PREDICTOR_MATRIX_KEY = 'predictor_matrix'
TARGET_MATRIX_KEY = 'target_matrix'
VALID_TIMES_KEY = 'valid_times_unix_sec'
LATITUDES_KEY = 'latitudes_deg_n'
LONGITUDES_KEY = 'longitudes_deg_e'


def _check_generator_args(option_dict):
    """Error-checks input arguments for generator.

    :param option_dict: See doc for `generator_full_grid`.
    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_GENERATOR_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_numpy_array(
        option_dict[BAND_NUMBERS_KEY], num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(option_dict[BAND_NUMBERS_KEY])

    error_checking.assert_is_integer(option_dict[BATCH_SIZE_KEY])
    error_checking.assert_is_geq(option_dict[BATCH_SIZE_KEY], 2)
    error_checking.assert_is_integer(option_dict[MAX_DAILY_EXAMPLES_KEY])
    error_checking.assert_is_geq(option_dict[MAX_DAILY_EXAMPLES_KEY], 2)
    error_checking.assert_is_integer(option_dict[LEAD_TIME_KEY])
    error_checking.assert_is_geq(option_dict[LEAD_TIME_KEY], 0)
    error_checking.assert_is_integer_numpy_array(option_dict[LAG_TIMES_KEY])
    error_checking.assert_is_geq_numpy_array(option_dict[LAG_TIMES_KEY], 0)

    max_time_diff_seconds = numpy.max(
        option_dict[LAG_TIMES_KEY] + option_dict[LEAD_TIME_KEY]
    )
    error_checking.assert_is_less_than(max_time_diff_seconds, DAYS_TO_SECONDS)

    return option_dict


def _check_inference_args(predictor_matrix, num_examples_per_batch, verbose):
    """Error-checks input arguments for inference.

    :param predictor_matrix: See doc for `apply_model_full_grid`.
    :param num_examples_per_batch: Batch size.
    :param verbose: Boolean flag.  If True, will print progress messages during
        inference.
    :return: num_examples_per_batch: Batch size (may be different than input).
    """

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)
    num_examples = predictor_matrix.shape[0]

    if num_examples_per_batch is None:
        num_examples_per_batch = num_examples + 0
    else:
        error_checking.assert_is_integer(num_examples_per_batch)
        # error_checking.assert_is_geq(num_examples_per_batch, 100)
        error_checking.assert_is_geq(num_examples_per_batch, 1)

    num_examples_per_batch = min([num_examples_per_batch, num_examples])
    error_checking.assert_is_boolean(verbose)

    return num_examples_per_batch


def _reshape_predictor_matrix(predictor_matrix, num_lag_times):
    """Reshapes predictor matrix to account for lag times.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    B = number of spectral bands
    L = number of lag times

    :param predictor_matrix: numpy array (EL x M x N x B) of predictors.
    :param num_lag_times: Number of lag times.
    :return: predictor_matrix: numpy array (E x M x N x BL) of predictors.
    :raises: ValueError: if length of first axis of predictor matrix is not an
        integer multiple of L.
    """

    # Check for errors.
    first_axis_length = predictor_matrix.shape[0]
    num_examples = float(first_axis_length) / num_lag_times
    this_diff = numpy.absolute(numpy.round(num_examples) - num_examples)

    if this_diff > TOLERANCE:
        error_string = (
            'Length of first axis of predictor matrix ({0:d}) must be an '
            'integer multiple of the number of lag times ({1:d}).'
        ).format(first_axis_length, num_lag_times)

        raise ValueError(error_string)

    # Do actual stuff.
    num_bands = predictor_matrix.shape[-1]

    predictor_matrix_by_lag = [
        predictor_matrix[j::num_lag_times, ...] for j in range(num_lag_times)
    ]
    predictor_matrix = numpy.stack(predictor_matrix_by_lag, axis=-1)

    num_channels = num_bands * num_lag_times
    these_dim = predictor_matrix.shape[:-2] + (num_channels,)
    return numpy.reshape(predictor_matrix, these_dim)


def _read_inputs_one_day(
        valid_date_string, predictor_file_names, band_numbers,
        normalize, uniformize, target_file_names, lead_time_seconds,
        lag_times_seconds, num_examples_to_read, return_coords):
    """Reads inputs (predictor and target files) for one day.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid

    :param valid_date_string: Valid date (format "yyyymmdd").
    :param predictor_file_names: 1-D list of paths to predictor files (readable
        by `example_io.read_predictor_file`).
    :param band_numbers: See doc for `generator_full_grid`.
    :param normalize: Same.
    :param uniformize: Same.
    :param target_file_names: 1-D list of paths to target files (readable by
        `example_io.read_target_file`).
    :param lead_time_seconds: See doc for `generator_full_grid`.
    :param lag_times_seconds: Same.
    :param num_examples_to_read: Number of examples to read.
    :param return_coords: Boolean flag.  If True, will return latitudes and
        longitudes for grid points.

    :return: data_dict: Dictionary with the following keys.
    data_dict['predictor_matrix']: See doc for
        `generator_full_grid`.
    data_dict['target_matrix']: Same.
    data_dict['valid_times_unix_sec']: length-E numpy array of valid times.
    data_dict['latitudes_deg_n']: length-M numpy array of latitudes (deg N).
        If `return_coords == False`, this is None.
    data_dict['longitudes_deg_e']: length-N numpy array of longitudes (deg E).
        If `return_coords == False`, this is None.
    """

    uniformize = uniformize and normalize

    target_date_strings = [
        example_io.file_name_to_date(f) for f in target_file_names
    ]
    index = target_date_strings.index(valid_date_string)
    desired_target_file_name = target_file_names[index]

    predictor_date_strings = [
        example_io.file_name_to_date(f) for f in predictor_file_names
    ]
    index = predictor_date_strings.index(valid_date_string)
    desired_predictor_file_names = [predictor_file_names[index]]

    if lead_time_seconds > 0 or numpy.any(lag_times_seconds > 0):
        desired_predictor_file_names.insert(0, predictor_file_names[index - 1])

    print('Reading data from: "{0:s}"...'.format(desired_target_file_name))
    target_dict = example_io.read_target_file(
        netcdf_file_name=desired_target_file_name
    )

    predictor_dicts = []

    for this_file_name in desired_predictor_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_predictor_dict = example_io.read_predictor_file(
            netcdf_file_name=this_file_name,
            read_unnormalized=not normalize,
            read_normalized=normalize and not uniformize,
            read_unif_normalized=normalize and uniformize
        )
        this_predictor_dict = example_io.subset_predictors_by_band(
            predictor_dict=this_predictor_dict, band_numbers=band_numbers
        )
        predictor_dicts.append(this_predictor_dict)

    predictor_dict = example_io.concat_predictor_data(predictor_dicts)

    assert numpy.allclose(
        target_dict[example_io.LATITUDES_KEY],
        predictor_dict[example_io.LATITUDES_KEY],
        atol=TOLERANCE
    )

    assert numpy.allclose(
        target_dict[example_io.LONGITUDES_KEY],
        predictor_dict[example_io.LONGITUDES_KEY],
        atol=TOLERANCE
    )

    valid_times_unix_sec = target_dict[example_io.VALID_TIMES_KEY]

    num_valid_times = len(valid_times_unix_sec)
    num_lag_times = len(lag_times_seconds)
    init_time_matrix_unix_sec = numpy.full(
        (num_valid_times, num_lag_times), -1, dtype=int
    )

    for i in range(num_valid_times):
        these_init_times_unix_sec = (
            valid_times_unix_sec[i] - lead_time_seconds - lag_times_seconds
        )

        if not all([
                t in predictor_dict[example_io.VALID_TIMES_KEY]
                for t in these_init_times_unix_sec
        ]):
            continue

        init_time_matrix_unix_sec[i, :] = these_init_times_unix_sec

    good_indices = numpy.where(
        numpy.all(init_time_matrix_unix_sec >= 0, axis=1)
    )[0]

    if len(good_indices) == 0:
        return None

    valid_times_unix_sec = valid_times_unix_sec[good_indices]
    init_time_matrix_unix_sec = init_time_matrix_unix_sec[good_indices, :]
    num_examples = len(valid_times_unix_sec)

    if num_examples >= num_examples_to_read:
        desired_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        )
        desired_indices = numpy.random.choice(
            desired_indices, size=num_examples_to_read, replace=False
        )

        valid_times_unix_sec = valid_times_unix_sec[desired_indices]
        init_time_matrix_unix_sec = (
            init_time_matrix_unix_sec[desired_indices, :]
        )

    predictor_dict = example_io.subset_by_time(
        predictor_or_target_dict=predictor_dict,
        desired_times_unix_sec=numpy.ravel(init_time_matrix_unix_sec)
    )[0]
    target_dict = example_io.subset_by_time(
        predictor_or_target_dict=target_dict,
        desired_times_unix_sec=valid_times_unix_sec
    )[0]

    if normalize:
        if uniformize:
            predictor_matrix = (
                predictor_dict[example_io.PREDICTOR_MATRIX_UNIF_NORM_KEY]
            )
        else:
            predictor_matrix = (
                predictor_dict[example_io.PREDICTOR_MATRIX_NORM_KEY]
            )
    else:
        predictor_matrix = (
            predictor_dict[example_io.PREDICTOR_MATRIX_UNNORM_KEY]
        )

    predictor_matrix = _reshape_predictor_matrix(
        predictor_matrix=predictor_matrix, num_lag_times=num_lag_times
    )
    target_matrix = target_dict[example_io.TARGET_MATRIX_KEY]

    print('Number of target values in batch = {0:d} ... mean = {1:.3g}'.format(
        target_matrix.size, numpy.mean(target_matrix)
    ))

    data_dict = {
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        TARGET_MATRIX_KEY: numpy.expand_dims(target_matrix, axis=-1),
        VALID_TIMES_KEY: valid_times_unix_sec,
        LATITUDES_KEY: None,
        LONGITUDES_KEY: None
    }

    if return_coords:
        data_dict[LATITUDES_KEY] = predictor_dict[example_io.LATITUDES_KEY]
        data_dict[LONGITUDES_KEY] = predictor_dict[example_io.LONGITUDES_KEY]

    return data_dict


def _write_metafile(
        dill_file_name, use_partial_grids, num_epochs,
        num_training_batches_per_epoch,
        training_option_dict, num_validation_batches_per_epoch,
        validation_option_dict, do_early_stopping, plateau_lr_multiplier,
        class_weights, fss_half_window_size_px, mask_matrix, full_mask_matrix):
    """Writes metadata to Dill file.

    M = number of rows in prediction grid
    N = number of columns in prediction grid

    :param dill_file_name: Path to output file.
    :param use_partial_grids: See doc for `train_model`.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param training_option_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_option_dict: Same.
    :param do_early_stopping: Same.
    :param plateau_lr_multiplier: Same.
    :param class_weights: Same.
    :param fss_half_window_size_px: Same.
    :param mask_matrix: Same.
    :param full_mask_matrix: Same.
    """

    metadata_dict = {
        USE_PARTIAL_GRIDS_KEY: use_partial_grids,
        NUM_EPOCHS_KEY: num_epochs,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        TRAINING_OPTIONS_KEY: training_option_dict,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        VALIDATION_OPTIONS_KEY: validation_option_dict,
        EARLY_STOPPING_KEY: do_early_stopping,
        PLATEAU_LR_MUTIPLIER_KEY: plateau_lr_multiplier,
        CLASS_WEIGHTS_KEY: class_weights,
        FSS_HALF_WINDOW_SIZE_KEY: fss_half_window_size_px,
        MASK_MATRIX_KEY: mask_matrix,
        FULL_MASK_MATRIX_KEY: full_mask_matrix
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=dill_file_name)

    dill_file_handle = open(dill_file_name, 'wb')
    pickle.dump(metadata_dict, dill_file_handle)
    dill_file_handle.close()


def _find_days_with_both_inputs(
        predictor_file_names, target_file_names, lead_time_seconds,
        lag_times_seconds):
    """Finds days with both inputs (predictor and target file) available.

    :param predictor_file_names: See doc for
        `_read_inputs_one_day`.
    :param target_file_names: Same.
    :param lead_time_seconds: Same.
    :param lag_times_seconds: Same.
    :return: valid_date_strings: List of valid dates (target dates) for which
        both predictors and targets exist, in format "yyyymmdd".
    """

    max_time_diff_seconds = numpy.max(lag_times_seconds) + lead_time_seconds

    predictor_date_strings = [
        example_io.file_name_to_date(f) for f in predictor_file_names
    ]
    target_date_strings = [
        example_io.file_name_to_date(f) for f in target_file_names
    ]
    valid_date_strings = []

    for this_target_date_string in target_date_strings:
        if this_target_date_string not in predictor_date_strings:
            continue

        if max_time_diff_seconds > 0:
            if (
                    general_utils.get_previous_date(this_target_date_string)
                    not in predictor_date_strings
            ):
                continue

        valid_date_strings.append(this_target_date_string)

    return valid_date_strings


def get_metrics(mask_matrix):
    """Returns metrics used for on-the-fly monitoring.

    M = number of rows in grid
    N = number of columns in grid

    :param mask_matrix: M-by-N numpy array of Boolean flags.  Only pixels marked
        "True" are considered in each metric.
    :return: metric_function_list: 1-D list of functions.
    :return: metric_function_dict: Dictionary, where each key is a function name
        and each value is a function itself.
    """

    error_checking.assert_is_boolean_numpy_array(mask_matrix)
    error_checking.assert_is_numpy_array(mask_matrix, num_dimensions=2)

    fss_function_size1 = custom_losses.fractions_skill_score(
        half_window_size_px=1, mask_matrix=mask_matrix,
        use_as_loss_function=False, function_name='fss_3by3'
    )
    fss_function_size2 = custom_losses.fractions_skill_score(
        half_window_size_px=2, mask_matrix=mask_matrix,
        use_as_loss_function=False, function_name='fss_5by5'
    )
    fss_function_size3 = custom_losses.fractions_skill_score(
        half_window_size_px=3, mask_matrix=mask_matrix,
        use_as_loss_function=False, function_name='fss_7by7'
    )
    fss_function_size4 = custom_losses.fractions_skill_score(
        half_window_size_px=4, mask_matrix=mask_matrix,
        use_as_loss_function=False, function_name='fss_9by9'
    )
    fss_function_size5 = custom_losses.fractions_skill_score(
        half_window_size_px=5, mask_matrix=mask_matrix,
        use_as_loss_function=False, function_name='fss_11by11'
    )
    fss_function_size6 = custom_losses.fractions_skill_score(
        half_window_size_px=6, mask_matrix=mask_matrix,
        use_as_loss_function=False, function_name='fss_13by13'
    )
    fss_function_size7 = custom_losses.fractions_skill_score(
        half_window_size_px=7, mask_matrix=mask_matrix,
        use_as_loss_function=False, function_name='fss_15by15'
    )

    bias_function_size1 = custom_metrics.frequency_bias(
        half_window_size_px=1, mask_matrix=mask_matrix,
        function_name='bias_3by3'
    )
    bias_function_size2 = custom_metrics.frequency_bias(
        half_window_size_px=2, mask_matrix=mask_matrix,
        function_name='bias_5by5'
    )
    bias_function_size3 = custom_metrics.frequency_bias(
        half_window_size_px=3, mask_matrix=mask_matrix,
        function_name='bias_7by7'
    )

    csi_function_size1 = custom_metrics.csi(
        half_window_size_px=1, mask_matrix=mask_matrix, function_name='csi_3by3'
    )
    csi_function_size2 = custom_metrics.csi(
        half_window_size_px=2, mask_matrix=mask_matrix, function_name='csi_5by5'
    )
    csi_function_size3 = custom_metrics.csi(
        half_window_size_px=3, mask_matrix=mask_matrix, function_name='csi_7by7'
    )

    dice_function_size1 = custom_metrics.dice_coeff(
        half_window_size_px=1, mask_matrix=mask_matrix,
        function_name='dice_coeff_3by3'
    )
    dice_function_size2 = custom_metrics.dice_coeff(
        half_window_size_px=2, mask_matrix=mask_matrix,
        function_name='dice_coeff_5by5'
    )
    dice_function_size3 = custom_metrics.dice_coeff(
        half_window_size_px=3, mask_matrix=mask_matrix,
        function_name='dice_coeff_7by7'
    )

    iou_function_size1 = custom_metrics.iou(
        half_window_size_px=1, mask_matrix=mask_matrix, function_name='iou_3by3'
    )
    iou_function_size2 = custom_metrics.iou(
        half_window_size_px=2, mask_matrix=mask_matrix, function_name='iou_5by5'
    )
    iou_function_size3 = custom_metrics.iou(
        half_window_size_px=3, mask_matrix=mask_matrix, function_name='iou_7by7'
    )

    metric_function_list = [
        fss_function_size1, fss_function_size2, fss_function_size3,
        csi_function_size1, csi_function_size2, csi_function_size3,
        bias_function_size1, bias_function_size2, bias_function_size3,
        dice_function_size1, dice_function_size2, dice_function_size3,
        iou_function_size1, iou_function_size2, iou_function_size3
    ]

    metric_function_dict = {
        'fss_3by3': fss_function_size1,
        'fss_5by5': fss_function_size2,
        'fss_7by7': fss_function_size3,
        'fss_9by9': fss_function_size4,
        'fss_11by11': fss_function_size5,
        'fss_13by13': fss_function_size6,
        'fss_15by15': fss_function_size7,
        'csi_3by3': csi_function_size1,
        'csi_5by5': csi_function_size2,
        'csi_7by7': csi_function_size3,
        'bias_3by3': bias_function_size1,
        'bias_5by5': bias_function_size2,
        'bias_7by7': bias_function_size3,
        'dice_coeff_3by3': dice_function_size1,
        'dice_coeff_5by5': dice_function_size2,
        'dice_coeff_7by7': dice_function_size3,
        'iou_3by3': iou_function_size1,
        'iou_5by5': iou_function_size2,
        'iou_7by7': iou_function_size3
    }

    return metric_function_list, metric_function_dict


def check_class_weights(class_weights):
    """Error-checks class weights.

    :param class_weights: length-2 numpy with class weights for loss function.
        Elements will be interpreted as
        (negative_class_weight, positive_class_weight).
    """

    error_checking.assert_is_numpy_array(
        class_weights, exact_dimensions=numpy.array([2], dtype=int)
    )
    error_checking.assert_is_greater_numpy_array(class_weights, 0.)


def create_data_full_grid(option_dict, return_coords=False):
    """Creates input data on full satellite grid.

    This method is the same as `generator_full_grid`, except that
    it returns all the data at once, rather than generating batches on the fly.

    :param option_dict: Dictionary with the following keys.
    option_dict['top_predictor_dir_name']: See doc for
        `generator_full_grid`.
    option_dict['top_target_dir_name']: Same.
    option_dict['band_numbers']: Same.
    option_dict['lead_time_seconds']: Same.
    option_dict['lag_times_seconds']: Same.
    option_dict['valid_date_string']: Valid date (format "yyyymmdd").  Will
        create examples with targets valid on this day.
    option_dict['normalize']: See doc for `generator_full_grid`.
    option_dict['uniformize']: Same.

    :param return_coords: See doc for `_read_inputs_one_day`.
    :return: data_dict: Same.
    """

    option_dict = _check_generator_args(option_dict)
    error_checking.assert_is_boolean(return_coords)

    top_predictor_dir_name = option_dict[PREDICTOR_DIRECTORY_KEY]
    top_target_dir_name = option_dict[TARGET_DIRECTORY_KEY]
    band_numbers = option_dict[BAND_NUMBERS_KEY]
    lead_time_seconds = option_dict[LEAD_TIME_KEY]
    lag_times_seconds = option_dict[LAG_TIMES_KEY]
    valid_date_string = option_dict[VALID_DATE_KEY]
    normalize = option_dict[NORMALIZE_FLAG_KEY]
    uniformize = option_dict[UNIFORMIZE_FLAG_KEY]

    if lead_time_seconds > 0 or numpy.any(lag_times_seconds > 0):
        first_init_date_string = general_utils.get_previous_date(
            valid_date_string
        )
    else:
        first_init_date_string = copy.deepcopy(valid_date_string)

    predictor_file_names = example_io.find_many_predictor_files(
        top_directory_name=top_predictor_dir_name,
        first_date_string=first_init_date_string,
        last_date_string=valid_date_string,
        prefer_zipped=True, allow_other_format=True,
        raise_error_if_all_missing=False,
        raise_error_if_any_missing=False
    )

    target_file_names = example_io.find_many_target_files(
        top_directory_name=top_target_dir_name,
        first_date_string=valid_date_string,
        last_date_string=valid_date_string,
        prefer_zipped=True, allow_other_format=True,
        raise_error_if_all_missing=False,
        raise_error_if_any_missing=False
    )

    valid_date_strings = _find_days_with_both_inputs(
        predictor_file_names=predictor_file_names,
        target_file_names=target_file_names,
        lead_time_seconds=lead_time_seconds,
        lag_times_seconds=lag_times_seconds
    )

    if len(valid_date_strings) == 0:
        return None

    return _read_inputs_one_day(
        valid_date_string=valid_date_string,
        predictor_file_names=predictor_file_names,
        band_numbers=band_numbers, normalize=normalize, uniformize=uniformize,
        target_file_names=target_file_names,
        lead_time_seconds=lead_time_seconds,
        lag_times_seconds=lag_times_seconds,
        num_examples_to_read=int(1e6), return_coords=return_coords
    )


def create_data_partial_grids(option_dict, return_coords=False):
    """Creates input data on partial, radar-centered grids.

    This method is the same as `generator_partial_grids`, except that
    it returns all the data at once, rather than generating batches on the fly.

    R = number of radar sites

    :param option_dict: Dictionary with the following keys.
    option_dict['top_predictor_dir_name']: See doc for `generator_full_grid`.
    option_dict['top_target_dir_name']: Same.
    option_dict['band_numbers']: Same.
    option_dict['lead_time_seconds']: Same.
    option_dict['lag_times_seconds']: Same.
    option_dict['valid_date_string']: Valid date (format "yyyymmdd").  Will
        create examples with targets valid on this day.
    option_dict['normalize']: See doc for `generator_full_grid`.
    option_dict['uniformize']: Same.

    :param return_coords: See doc for `_read_inputs_one_day`.
    :return: data_dicts: length-R list of dictionaries, each in format returned
        by `_read_inputs_one_day`.
    """

    option_dict = _check_generator_args(option_dict)
    error_checking.assert_is_boolean(return_coords)

    top_predictor_dir_name = option_dict[PREDICTOR_DIRECTORY_KEY]
    top_target_dir_name = option_dict[TARGET_DIRECTORY_KEY]
    band_numbers = option_dict[BAND_NUMBERS_KEY]
    lead_time_seconds = option_dict[LEAD_TIME_KEY]
    lag_times_seconds = option_dict[LAG_TIMES_KEY]
    valid_date_string = option_dict[VALID_DATE_KEY]
    normalize = option_dict[NORMALIZE_FLAG_KEY]
    uniformize = option_dict[UNIFORMIZE_FLAG_KEY]

    if lead_time_seconds > 0 or numpy.any(lag_times_seconds > 0):
        first_init_date_string = general_utils.get_previous_date(
            valid_date_string
        )
    else:
        first_init_date_string = copy.deepcopy(valid_date_string)

    valid_date_strings = None
    data_dicts = [dict()] * NUM_RADARS

    for k in range(NUM_RADARS):
        these_predictor_file_names = example_io.find_many_predictor_files(
            top_directory_name=top_predictor_dir_name,
            first_date_string=first_init_date_string,
            last_date_string=valid_date_string, radar_number=k,
            prefer_zipped=True, allow_other_format=True,
            raise_error_if_all_missing=False,
            raise_error_if_any_missing=False
        )

        these_target_file_names = example_io.find_many_target_files(
            top_directory_name=top_target_dir_name,
            first_date_string=valid_date_string,
            last_date_string=valid_date_string, radar_number=k,
            prefer_zipped=True, allow_other_format=True,
            raise_error_if_all_missing=False,
            raise_error_if_any_missing=False
        )

        these_valid_date_strings = _find_days_with_both_inputs(
            predictor_file_names=these_predictor_file_names,
            target_file_names=these_target_file_names,
            lead_time_seconds=lead_time_seconds,
            lag_times_seconds=lag_times_seconds
        )

        if valid_date_strings is None:
            valid_date_strings = copy.deepcopy(these_valid_date_strings)

        assert valid_date_strings == these_valid_date_strings

        if len(valid_date_strings) == 0:
            return None

        data_dicts[k] = _read_inputs_one_day(
            valid_date_string=valid_date_string,
            predictor_file_names=these_predictor_file_names,
            band_numbers=band_numbers, normalize=normalize,
            uniformize=uniformize,
            target_file_names=these_target_file_names,
            lead_time_seconds=lead_time_seconds,
            lag_times_seconds=lag_times_seconds,
            num_examples_to_read=int(1e6), return_coords=return_coords
        )

    return data_dicts


def generator_full_grid(option_dict):
    """Generates training data on full satellite grid.

    E = number of examples per batch
    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (num spectral bands * num lag times)

    :param option_dict: Dictionary with the following keys.
    option_dict['top_predictor_dir_name']: Name of top-level directory with
        predictors.  Files therein will be found by
        `example_io.find_predictor_file` and read by
        `example_io.read_predictor_file`.
    option_dict['top_target_dir_name']: Name of top-level directory with
        targets.  Files therein will be found by `example_io.find_target_file`
        and read by `example_io.read_target_file`.
    option_dict['num_examples_per_batch']: Batch size.
    option_dict['max_examples_per_day_in_batch']: Max number of examples from
        the same day in one batch.
    option_dict['band_numbers']: 1-D numpy array of band numbers (integers) for
        satellite data.  Will use only these spectral bands as predictors.
    option_dict['lead_time_seconds']: Lead time (valid time minus forecast
        time).
    option_dict['lag_times_seconds']: 1-D numpy array of lag times.  Each lag
        time is forecast time minus predictor time, so must be >= 0.
    option_dict['first_valid_date_string']: First valid date (format
        "yyyymmdd").  Will not generate examples with earlier valid times.
    option_dict['last_valid_date_string']: Last valid date (format
        "yyyymmdd").  Will not generate examples with later valid times.
    option_dict['normalize']: Boolean flag.  If True (False), will use
        normalized (unnormalized) predictors.
    option_dict['uniformize']: [used only if `normalize == True`]
        Boolean flag.  If True, will use uniformized and normalized predictors.
        If False, will use only normalized predictors.

    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values,
        based on satellite data.
    :return: target_matrix: E-by-M-by-N-by-1 numpy array of target values
        (integers in 0...1, indicating whether or not convection occurs at
        the given lead time).
    :raises: ValueError: if no valid date can be found for which predictors and
        targets are available.
    """

    option_dict = _check_generator_args(option_dict)

    top_predictor_dir_name = option_dict[PREDICTOR_DIRECTORY_KEY]
    top_target_dir_name = option_dict[TARGET_DIRECTORY_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    max_examples_per_day_in_batch = option_dict[MAX_DAILY_EXAMPLES_KEY]
    band_numbers = option_dict[BAND_NUMBERS_KEY]
    lead_time_seconds = option_dict[LEAD_TIME_KEY]
    lag_times_seconds = option_dict[LAG_TIMES_KEY]
    first_valid_date_string = option_dict[FIRST_VALID_DATE_KEY]
    last_valid_date_string = option_dict[LAST_VALID_DATE_KEY]
    normalize = option_dict[NORMALIZE_FLAG_KEY]
    uniformize = option_dict[UNIFORMIZE_FLAG_KEY]

    if lead_time_seconds > 0 or numpy.any(lag_times_seconds > 0):
        first_init_date_string = general_utils.get_previous_date(
            first_valid_date_string
        )
    else:
        first_init_date_string = copy.deepcopy(first_valid_date_string)

    predictor_file_names = example_io.find_many_predictor_files(
        top_directory_name=top_predictor_dir_name,
        first_date_string=first_init_date_string,
        last_date_string=last_valid_date_string,
        prefer_zipped=True, allow_other_format=True,
        raise_error_if_any_missing=False
    )

    target_file_names = example_io.find_many_target_files(
        top_directory_name=top_target_dir_name,
        first_date_string=first_init_date_string,
        last_date_string=last_valid_date_string,
        prefer_zipped=True, allow_other_format=True,
        raise_error_if_any_missing=False
    )

    valid_date_strings = _find_days_with_both_inputs(
        predictor_file_names=predictor_file_names,
        target_file_names=target_file_names,
        lead_time_seconds=lead_time_seconds,
        lag_times_seconds=lag_times_seconds
    )

    if len(valid_date_strings) == 0:
        raise ValueError(
            'Cannot find any valid date for which both predictors and targets '
            ' are available.'
        )

    random.shuffle(valid_date_strings)
    date_index = 0

    while True:
        predictor_matrix = None
        target_matrix = None
        num_examples_in_memory = 0

        while num_examples_in_memory < num_examples_per_batch:
            if date_index == len(valid_date_strings):
                date_index = 0

            num_examples_to_read = min([
                max_examples_per_day_in_batch,
                num_examples_per_batch - num_examples_in_memory
            ])

            this_data_dict = _read_inputs_one_day(
                valid_date_string=valid_date_strings[date_index],
                predictor_file_names=predictor_file_names,
                band_numbers=band_numbers,
                normalize=normalize, uniformize=uniformize,
                target_file_names=target_file_names,
                lead_time_seconds=lead_time_seconds,
                lag_times_seconds=lag_times_seconds,
                num_examples_to_read=num_examples_to_read, return_coords=False
            )

            date_index += 1
            if this_data_dict is None:
                continue

            this_predictor_matrix = this_data_dict[PREDICTOR_MATRIX_KEY]
            this_target_matrix = this_data_dict[TARGET_MATRIX_KEY]

            if predictor_matrix is None:
                predictor_matrix = this_predictor_matrix + 0.
                target_matrix = this_target_matrix + 0
            else:
                predictor_matrix = numpy.concatenate(
                    (predictor_matrix, this_predictor_matrix), axis=0
                )
                target_matrix = numpy.concatenate(
                    (target_matrix, this_target_matrix), axis=0
                )

            num_examples_in_memory = predictor_matrix.shape[0]

        predictor_matrix = predictor_matrix.astype('float32')
        target_matrix = target_matrix.astype('float32')
        yield predictor_matrix, target_matrix


def generator_partial_grids(option_dict):
    """Generates training data on partial, radar-centered grids.

    :param option_dict: See doc for `generator_full_grid`.
    :return: predictor_matrix: Same.
    :return: target_matrix: Same.
    :raises: ValueError: if no valid date can be found for which predictors and
        targets are available.
    """

    option_dict = _check_generator_args(option_dict)

    top_predictor_dir_name = option_dict[PREDICTOR_DIRECTORY_KEY]
    top_target_dir_name = option_dict[TARGET_DIRECTORY_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    max_examples_per_day_in_batch = option_dict[MAX_DAILY_EXAMPLES_KEY]
    band_numbers = option_dict[BAND_NUMBERS_KEY]
    lead_time_seconds = option_dict[LEAD_TIME_KEY]
    lag_times_seconds = option_dict[LAG_TIMES_KEY]
    first_valid_date_string = option_dict[FIRST_VALID_DATE_KEY]
    last_valid_date_string = option_dict[LAST_VALID_DATE_KEY]
    normalize = option_dict[NORMALIZE_FLAG_KEY]
    uniformize = option_dict[UNIFORMIZE_FLAG_KEY]

    if lead_time_seconds > 0 or numpy.any(lag_times_seconds > 0):
        first_init_date_string = general_utils.get_previous_date(
            first_valid_date_string
        )
    else:
        first_init_date_string = copy.deepcopy(first_valid_date_string)

    these_predictor_file_names = example_io.find_many_predictor_files(
        top_directory_name=top_predictor_dir_name,
        first_date_string=first_init_date_string,
        last_date_string=last_valid_date_string, radar_number=0,
        prefer_zipped=True, allow_other_format=True,
        raise_error_if_any_missing=False
    )
    these_target_file_names = example_io.find_many_target_files(
        top_directory_name=top_target_dir_name,
        first_date_string=first_init_date_string,
        last_date_string=last_valid_date_string, radar_number=0,
        prefer_zipped=True, allow_other_format=True,
        raise_error_if_any_missing=False
    )

    predictor_date_strings = [
        example_io.file_name_to_date(f) for f in these_predictor_file_names
    ]
    target_date_strings = [
        example_io.file_name_to_date(f) for f in these_target_file_names
    ]

    predictor_file_name_matrix = numpy.full(
        (len(these_predictor_file_names), NUM_RADARS), '', dtype=object
    )
    target_file_name_matrix = numpy.full(
        (len(these_target_file_names), NUM_RADARS), '', dtype=object
    )

    predictor_file_name_matrix[:, 0] = numpy.array(these_predictor_file_names)
    target_file_name_matrix[:, 0] = numpy.array(these_target_file_names)

    valid_date_strings = _find_days_with_both_inputs(
        predictor_file_names=predictor_file_name_matrix[:, 0],
        target_file_names=target_file_name_matrix[:, 0],
        lead_time_seconds=lead_time_seconds,
        lag_times_seconds=lag_times_seconds
    )

    if len(valid_date_strings) == 0:
        raise ValueError(
            'Cannot find any valid date for which both predictors and targets '
            ' are available.'
        )

    for k in range(1, NUM_RADARS):
        these_predictor_file_names = [
            example_io.find_predictor_file(
                top_directory_name=top_predictor_dir_name, date_string=d,
                radar_number=k, prefer_zipped=True, allow_other_format=True,
                raise_error_if_missing=True
            ) for d in predictor_date_strings
        ]

        these_target_file_names = [
            example_io.find_target_file(
                top_directory_name=top_target_dir_name, date_string=d,
                radar_number=k, prefer_zipped=True, allow_other_format=True,
                raise_error_if_missing=True
            ) for d in target_date_strings
        ]

        predictor_file_name_matrix[:, k] = numpy.array(
            these_predictor_file_names
        )
        target_file_name_matrix[:, k] = numpy.array(these_target_file_names)

    radar_indices = numpy.linspace(0, NUM_RADARS - 1, num=NUM_RADARS, dtype=int)
    date_indices = numpy.linspace(
        0, len(valid_date_strings) - 1, num=len(valid_date_strings), dtype=int
    )

    date_index_matrix, radar_index_matrix = numpy.meshgrid(
        date_indices, radar_indices
    )
    date_indices_1d = numpy.ravel(date_index_matrix)
    radar_indices_1d = numpy.ravel(radar_index_matrix)

    random_indices = numpy.linspace(
        0, len(radar_indices_1d) - 1, num=len(radar_indices_1d), dtype=int
    )
    numpy.random.shuffle(random_indices)
    date_indices_1d = date_indices_1d[random_indices]
    radar_indices_1d = radar_indices_1d[random_indices]

    current_index = 0

    while True:
        predictor_matrix = None
        target_matrix = None
        num_examples_in_memory = 0

        while num_examples_in_memory < num_examples_per_batch:
            if current_index == len(radar_indices_1d):
                current_index = 0

            num_examples_to_read = min([
                max_examples_per_day_in_batch,
                num_examples_per_batch - num_examples_in_memory
            ])

            current_date_index = date_indices_1d[current_index]
            current_radar_index = radar_indices_1d[current_index]

            this_data_dict = _read_inputs_one_day(
                valid_date_string=valid_date_strings[current_date_index],
                predictor_file_names=
                predictor_file_name_matrix[:, current_radar_index],
                band_numbers=band_numbers,
                normalize=normalize, uniformize=uniformize,
                target_file_names=
                target_file_name_matrix[:, current_radar_index],
                lead_time_seconds=lead_time_seconds,
                lag_times_seconds=lag_times_seconds,
                num_examples_to_read=num_examples_to_read, return_coords=False
            )

            current_index += 1
            if this_data_dict is None:
                continue

            this_predictor_matrix = this_data_dict[PREDICTOR_MATRIX_KEY]
            this_target_matrix = this_data_dict[TARGET_MATRIX_KEY]

            if predictor_matrix is None:
                predictor_matrix = this_predictor_matrix + 0.
                target_matrix = this_target_matrix + 0
            else:
                predictor_matrix = numpy.concatenate(
                    (predictor_matrix, this_predictor_matrix), axis=0
                )
                target_matrix = numpy.concatenate(
                    (target_matrix, this_target_matrix), axis=0
                )

            num_examples_in_memory = predictor_matrix.shape[0]

        predictor_matrix = predictor_matrix.astype('float32')
        target_matrix = target_matrix.astype('float32')
        yield predictor_matrix, target_matrix


def train_model(
        model_object, output_dir_name, num_epochs, use_partial_grids,
        num_training_batches_per_epoch, training_option_dict,
        num_validation_batches_per_epoch, validation_option_dict,
        mask_matrix, full_mask_matrix, do_early_stopping=True,
        plateau_lr_multiplier=DEFAULT_LEARNING_RATE_MULTIPLIER,
        class_weights=None, fss_half_window_size_px=None):
    """Trains neural net on either full grid or partial grids.

    M = number of rows in full grid
    N = number of columns in full grid
    m = number of rows in prediction grid
    n = number of columns in prediction grid

    :param model_object: Untrained neural net (instance of `keras.models.Model`
        or `keras.models.Sequential`).
    :param output_dir_name: Path to output directory (model and training history
        will be saved here).
    :param num_epochs: Number of training epochs.
    :param use_partial_grids: Boolean flag.  If True (False), neural net will be
        trained on full (partial) grids.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param training_option_dict: See doc for
        `generator_full_grid`.  This dictionary will be used to
        generate training data.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param validation_option_dict: See doc for
        `generator_full_grid`.  For validation only, the following
        values will replace corresponding values in `training_option_dict`:
    validation_option_dict['top_predictor_dir_name']
    validation_option_dict['top_target_dir_name']
    validation_option_dict['first_valid_date_string']
    validation_option_dict['last_valid_date_string']

    :param mask_matrix: m-by-n numpy array of Boolean flags.  Grid cells labeled
        True (False) are (not) used for model evaluation.
    :param full_mask_matrix: Same but with dimensions of M x N.
    :param do_early_stopping: Boolean flag.  If True, will stop training early
        if validation loss has not improved over last several epochs (see
        constants at top of file for what exactly this means).
    :param plateau_lr_multiplier: Multiplier for learning rate.  Learning
        rate will be multiplied by this factor upon plateau in validation
        performance.
    :param class_weights: See doc for `check_class_weights`.  If weighted cross-
        entropy is not the loss function, leave this alone.
    :param fss_half_window_size_px: Number of pixels (grid cells) in half of
        smoothing window for fractions skill score (FSS).  If FSS is not the
        loss function, leave this alone.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 2)
    error_checking.assert_is_boolean(use_partial_grids)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 10)
    error_checking.assert_is_integer(num_validation_batches_per_epoch)
    error_checking.assert_is_geq(num_validation_batches_per_epoch, 10)
    error_checking.assert_is_boolean(do_early_stopping)

    error_checking.assert_is_numpy_array(mask_matrix, num_dimensions=2)
    error_checking.assert_is_numpy_array(full_mask_matrix, num_dimensions=2)

    try:
        error_checking.assert_is_integer_numpy_array(mask_matrix)
        error_checking.assert_is_geq_numpy_array(mask_matrix, 0)
        error_checking.assert_is_leq_numpy_array(mask_matrix, 1)
    except TypeError:
        error_checking.assert_is_boolean_numpy_array(mask_matrix)

    try:
        error_checking.assert_is_integer_numpy_array(full_mask_matrix)
        error_checking.assert_is_geq_numpy_array(full_mask_matrix, 0)
        error_checking.assert_is_leq_numpy_array(full_mask_matrix, 1)
    except TypeError:
        error_checking.assert_is_boolean_numpy_array(full_mask_matrix)

    if do_early_stopping:
        error_checking.assert_is_greater(plateau_lr_multiplier, 0.)
        error_checking.assert_is_less_than(plateau_lr_multiplier, 1.)

    if class_weights is not None:
        check_class_weights(class_weights)

    if fss_half_window_size_px is not None:
        error_checking.assert_is_integer(fss_half_window_size_px)
        error_checking.assert_is_geq(fss_half_window_size_px, 0)

    training_option_dict = _check_generator_args(training_option_dict)

    validation_keys_to_keep = [
        PREDICTOR_DIRECTORY_KEY, TARGET_DIRECTORY_KEY,
        FIRST_VALID_DATE_KEY, LAST_VALID_DATE_KEY
    ]

    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

    validation_option_dict = _check_generator_args(validation_option_dict)
    model_file_name = '{0:s}/model.h5'.format(output_dir_name)

    history_object = keras.callbacks.CSVLogger(
        filename='{0:s}/history.csv'.format(output_dir_name),
        separator=',', append=False
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=do_early_stopping, save_weights_only=False, mode='min',
        period=1
    )
    list_of_callback_objects = [history_object, checkpoint_object]

    if do_early_stopping:
        early_stopping_object = keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=LOSS_PATIENCE,
            patience=EARLY_STOPPING_PATIENCE_EPOCHS, verbose=1, mode='min'
        )
        list_of_callback_objects.append(early_stopping_object)

        plateau_object = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=plateau_lr_multiplier,
            patience=PLATEAU_PATIENCE_EPOCHS, verbose=1, mode='min',
            min_delta=LOSS_PATIENCE, cooldown=PLATEAU_COOLDOWN_EPOCHS
        )
        list_of_callback_objects.append(plateau_object)

    metafile_name = find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=False
    )
    print('Writing metadata to: "{0:s}"...'.format(metafile_name))

    _write_metafile(
        dill_file_name=metafile_name,
        use_partial_grids=use_partial_grids, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        do_early_stopping=do_early_stopping,
        plateau_lr_multiplier=plateau_lr_multiplier,
        class_weights=class_weights,
        fss_half_window_size_px=fss_half_window_size_px,
        mask_matrix=mask_matrix, full_mask_matrix=full_mask_matrix
    )

    if use_partial_grids:
        training_generator = generator_partial_grids(training_option_dict)
        validation_generator = generator_partial_grids(
            validation_option_dict
        )
    else:
        training_generator = generator_full_grid(training_option_dict)
        validation_generator = generator_full_grid(
            validation_option_dict
        )

    model_object.fit_generator(
        generator=training_generator,
        steps_per_epoch=num_training_batches_per_epoch,
        epochs=num_epochs, verbose=1, callbacks=list_of_callback_objects,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch
    )


def read_model(hdf5_file_name):
    """Reads model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model`.
    """

    error_checking.assert_file_exists(hdf5_file_name)

    metafile_name = find_metafile(
        model_file_name=hdf5_file_name, raise_error_if_missing=True
    )
    metadata_dict = read_metafile(metafile_name)
    mask_matrix = metadata_dict[MASK_MATRIX_KEY]
    custom_object_dict = get_metrics(mask_matrix)[1]

    try:
        return tf_keras.models.load_model(
            hdf5_file_name, custom_objects=custom_object_dict
        )
    except ValueError:
        pass

    metafile_name = find_metafile(
        model_file_name=hdf5_file_name, raise_error_if_missing=True
    )

    metadata_dict = read_metafile(metafile_name)
    class_weights = metadata_dict[CLASS_WEIGHTS_KEY]
    fss_half_window_size_px = metadata_dict[FSS_HALF_WINDOW_SIZE_KEY]

    if fss_half_window_size_px is not None:
        custom_object_dict['loss'] = custom_losses.fractions_skill_score(
            half_window_size_px=fss_half_window_size_px,
            mask_matrix=mask_matrix, use_as_loss_function=True
        )
    elif class_weights is not None:
        custom_object_dict['loss'] = custom_losses.weighted_xentropy(
            class_weights
        )

    return tf_keras.models.load_model(
        hdf5_file_name, custom_objects=custom_object_dict
    )


def find_metafile(model_file_name, raise_error_if_missing=True):
    """Finds metafile for neural net.

    :param model_file_name: Path to trained model.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: metafile_name: Path to metafile.
    """

    error_checking.assert_is_string(model_file_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    metafile_name = '{0:s}/model_metadata.dill'.format(
        os.path.split(model_file_name)[0]
    )

    if raise_error_if_missing and not os.path.isfile(metafile_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            metafile_name
        )
        raise ValueError(error_string)

    return metafile_name


def read_metafile(dill_file_name):
    """Reads metadata for neural net from Dill file.

    :param dill_file_name: Path to input file.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['use_partial_grids']: See doc for `train_model`.
    metadata_dict['num_epochs']: Same.
    metadata_dict['num_training_batches_per_epoch']: Same.
    metadata_dict['training_option_dict']: Same.
    metadata_dict['num_validation_batches_per_epoch']: Same.
    metadata_dict['validation_option_dict']: Same.
    metadata_dict['do_early_stopping']: Same.
    metadata_dict['plateau_lr_multiplier']: Same.
    metadata_dict['class_weights']: Same.
    metadata_dict['fss_half_window_size_px']: Same.
    metadata_dict['mask_matrix']: Same.
    metadata_dict['full_mask_matrix']: Same.

    :raises: ValueError: if any expected key is not found in dictionary.
    """

    error_checking.assert_file_exists(dill_file_name)

    dill_file_handle = open(dill_file_name, 'rb')
    metadata_dict = pickle.load(dill_file_handle)
    dill_file_handle.close()

    if CLASS_WEIGHTS_KEY not in metadata_dict:
        metadata_dict[CLASS_WEIGHTS_KEY] = None

    if FSS_HALF_WINDOW_SIZE_KEY not in metadata_dict:
        metadata_dict[FSS_HALF_WINDOW_SIZE_KEY] = None

    if USE_PARTIAL_GRIDS_KEY not in metadata_dict:
        metadata_dict[USE_PARTIAL_GRIDS_KEY] = False

    if FULL_MASK_MATRIX_KEY not in metadata_dict:
        metadata_dict[FULL_MASK_MATRIX_KEY] = metadata_dict[MASK_MATRIX_KEY]

    training_option_dict = metadata_dict[TRAINING_OPTIONS_KEY]
    validation_option_dict = metadata_dict[VALIDATION_OPTIONS_KEY]

    if LAG_TIMES_KEY not in training_option_dict:
        training_option_dict[LAG_TIMES_KEY] = numpy.array([0], dtype=int)
        validation_option_dict[LAG_TIMES_KEY] = numpy.array([0], dtype=int)

    metadata_dict[TRAINING_OPTIONS_KEY] = training_option_dict
    metadata_dict[VALIDATION_OPTIONS_KEY] = validation_option_dict

    missing_keys = list(set(METADATA_KEYS) - set(metadata_dict.keys()))
    if len(missing_keys) == 0:
        return metadata_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), dill_file_name)

    raise ValueError(error_string)


def apply_model_full_grid(
        model_object, predictor_matrix, num_examples_per_batch, verbose=False):
    """Applies trained neural net to full grid.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix: See output doc for
        `generator_full_grid`.
    :param num_examples_per_batch: Batch size.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: forecast_prob_matrix: E-by-M-by-N numpy array of forecast event
        probabilities.
    """

    num_examples_per_batch = _check_inference_args(
        predictor_matrix=predictor_matrix,
        num_examples_per_batch=num_examples_per_batch, verbose=verbose
    )

    forecast_prob_matrix = None
    num_examples = predictor_matrix.shape[0]

    for i in range(0, num_examples, num_examples_per_batch):
        this_first_index = i
        this_last_index = min(
            [i + num_examples_per_batch - 1, num_examples - 1]
        )

        these_indices = numpy.linspace(
            this_first_index, this_last_index,
            num=this_last_index - this_first_index + 1, dtype=int
        )

        if verbose:
            print((
                'Applying model to examples {0:d}-{1:d} of {2:d}...'
            ).format(
                this_first_index + 1, this_last_index + 1, num_examples
            ))

        this_prob_matrix = model_object.predict(
            predictor_matrix[these_indices, ...], batch_size=len(these_indices)
        )

        if forecast_prob_matrix is None:
            dimensions = (num_examples,) + this_prob_matrix.shape[1:3]
            forecast_prob_matrix = numpy.full(dimensions, numpy.nan)

        forecast_prob_matrix[these_indices, ...] = this_prob_matrix[..., 0]

    if verbose:
        print('Have applied model to all {0:d} examples!'.format(num_examples))

    return forecast_prob_matrix


def apply_model_partial_grids(
        model_object, predictor_matrix, num_examples_per_batch, overlap_size_px,
        verbose=False):
    """Applies trained NN to full grid, predicting one partial grid at a time.

    :param model_object: See doc for `apply_model_full_grid`.
    :param predictor_matrix: Same.
    :param num_examples_per_batch: Same.
    :param overlap_size_px: Overlap between adjacent partial grids (number of
        pixels).
    :param verbose: See doc for `apply_model_full_grid`.
    :return: forecast_prob_matrix: Same.
    """

    # Check input args.
    num_examples_per_batch = _check_inference_args(
        predictor_matrix=predictor_matrix,
        num_examples_per_batch=num_examples_per_batch, verbose=verbose
    )

    these_dim = model_object.layers[-1].output.get_shape().as_list()
    num_partial_grid_rows = these_dim[1]
    num_partial_grid_columns = these_dim[2]

    error_checking.assert_is_integer(overlap_size_px)
    error_checking.assert_is_geq(overlap_size_px, 0)
    error_checking.assert_is_less_than(
        2 * overlap_size_px,
        min([num_partial_grid_rows, num_partial_grid_columns])
    )

    num_examples = predictor_matrix.shape[0]
    num_full_grid_rows = predictor_matrix.shape[1]
    num_full_grid_columns = predictor_matrix.shape[2]

    error_checking.assert_is_less_than(
        num_partial_grid_rows, num_full_grid_rows
    )
    error_checking.assert_is_less_than(
        num_partial_grid_columns, num_full_grid_columns
    )

    # Do actual stuff.
    forecast_prob_matrix = numpy.full(
        (num_examples, num_full_grid_rows, num_full_grid_columns), 0.
    )
    first_input_row = -1
    first_input_column = -1
    last_input_row = -1

    while last_input_row < num_full_grid_rows - 1:
        if first_input_row < 0:
            first_input_row = 0
            first_input_column = 0
        else:
            first_input_row += num_partial_grid_rows - 2 * overlap_size_px
            first_input_column += num_partial_grid_columns - 2 * overlap_size_px

        last_input_row = min([
            first_input_row + num_partial_grid_rows - 1,
            num_full_grid_rows - 1
        ])
        last_input_column = min([
            first_input_column + num_partial_grid_columns - 1,
            num_full_grid_columns - 1
        ])

        first_input_row = last_input_row - num_partial_grid_rows + 1
        first_input_column = last_input_column - num_partial_grid_columns + 1

        first_output_row = first_input_row + overlap_size_px
        last_output_row = last_input_row - overlap_size_px
        first_output_column = first_input_column + overlap_size_px
        last_output_column = last_input_column - overlap_size_px

        for i in range(0, num_examples, num_examples_per_batch):
            first_example_index = i
            last_example_index = min([
                i + num_examples_per_batch - 1, num_examples - 1
            ])

            if verbose:
                print((
                    'Applying model to rows {0:d}-{1:d} of {2:d}, '
                    'columns {3:d}-{4:d} of {5:d}, '
                    'examples {6:d}-{7:d} of {8:d}...'
                ).format(
                    first_input_row + 1, last_input_row + 1, num_full_grid_rows,
                    first_input_column + 1, last_input_column + 1,
                    num_full_grid_columns,
                    first_example_index + 1, last_example_index + 1,
                    num_examples
                ))

            this_predictor_matrix = predictor_matrix[
                first_example_index:(last_example_index + 1),
                first_input_row:(last_input_row + 1),
                first_input_column:(last_input_column + 1),
                :
            ]

            this_prob_matrix = model_object.predict(
                this_predictor_matrix, batch_size=this_predictor_matrix.shape[0]
            )[..., 0]

            if verbose:
                print((
                    'Taking predictions from rows {0:d}-{1:d} of {2:d}, '
                    'columns {3:d}-{4:d} of {5:d}...'
                ).format(
                    first_output_row + 1, last_output_row + 1,
                    num_full_grid_rows,
                    first_output_column + 1, last_output_column + 1,
                    num_full_grid_columns
                ))

            this_prob_matrix = this_prob_matrix[
                :,
                overlap_size_px:-overlap_size_px,
                overlap_size_px:-overlap_size_px
            ]

            forecast_prob_matrix[
                first_example_index:(last_example_index + 1),
                first_output_row:(last_output_row + 1),
                first_output_column:(last_output_column + 1)
            ] = this_prob_matrix

    if verbose:
        print((
            'Have applied model to all grid points and all {0:d} examples!'
        ).format(
            num_examples
        ))

    return forecast_prob_matrix
