"""Methods for training and applying neural nets."""

import copy
import os.path
import dill
import numpy
# numpy.random.seed(6695)
import keras
import tensorflow
# tensorflow.random.set_seed(6695)
import tensorflow.keras as tf_keras
from tensorflow.keras import backend as K
from tensorflow.python.keras import backend as python_K
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4convection.io import radar_io
from ml4convection.io import satellite_io
from ml4convection.io import twb_radar_io
from ml4convection.io import twb_satellite_io
from ml4convection.io import example_io
from ml4convection.utils import general_utils
from ml4convection.utils import radar_utils
from ml4convection.utils import fourier_utils
from ml4convection.utils import wavelet_utils
from ml4convection.machine_learning import custom_losses
from ml4convection.machine_learning import custom_metrics
from ml4convection.machine_learning import fourier_metrics
from ml4convection.machine_learning import wavelet_metrics
from wavetf import WaveTFFactory

TOLERANCE = 1e-6

GRID_SPACING_DEG = 0.0125
MAX_TARGET_RESOLUTION_DEG = 6.4

FSS_NAME = fourier_metrics.FSS_NAME
BRIER_SCORE_NAME = fourier_metrics.BRIER_SCORE_NAME
CROSS_ENTROPY_NAME = fourier_metrics.CROSS_ENTROPY_NAME
CRPS_NAME = 'crps'
FSS_PLUS_CRPS_NAME = 'fss-plus-crps'
CSI_NAME = fourier_metrics.CSI_NAME
FREQUENCY_BIAS_NAME = fourier_metrics.FREQUENCY_BIAS_NAME
IOU_NAME = fourier_metrics.IOU_NAME
ALL_CLASS_IOU_NAME = fourier_metrics.ALL_CLASS_IOU_NAME
DICE_COEFF_NAME = fourier_metrics.DICE_COEFF_NAME
HEIDKE_SCORE_NAME = fourier_metrics.HEIDKE_SCORE_NAME
PEIRCE_SCORE_NAME = fourier_metrics.PEIRCE_SCORE_NAME
GERRITY_SCORE_NAME = fourier_metrics.GERRITY_SCORE_NAME
REAL_FREQ_MSE_NAME = fourier_metrics.REAL_FREQ_MSE_NAME
IMAGINARY_FREQ_MSE_NAME = fourier_metrics.IMAGINARY_FREQ_MSE_NAME
FREQ_MSE_NAME = fourier_metrics.FREQ_MSE_NAME

VALID_SCORE_NAMES_NEIGH = [
    FSS_NAME, BRIER_SCORE_NAME, CROSS_ENTROPY_NAME, CRPS_NAME,
    FSS_PLUS_CRPS_NAME, CSI_NAME, FREQUENCY_BIAS_NAME,
    IOU_NAME, ALL_CLASS_IOU_NAME, DICE_COEFF_NAME,
    HEIDKE_SCORE_NAME, PEIRCE_SCORE_NAME, GERRITY_SCORE_NAME
]
VALID_SCORE_NAMES_WAVELET = VALID_SCORE_NAMES_NEIGH + []
VALID_SCORE_NAMES_FOURIER = VALID_SCORE_NAMES_WAVELET + [
    REAL_FREQ_MSE_NAME, IMAGINARY_FREQ_MSE_NAME, FREQ_MSE_NAME
]

SCORE_NAME_KEY = 'score_name'
WEIGHT_KEY = 'weight'
CRPS_WEIGHT_KEY = 'crps_weight'
HALF_WINDOW_SIZE_KEY = 'half_window_size_px'
MIN_RESOLUTION_KEY = 'min_resolution_deg'
MAX_RESOLUTION_KEY = 'max_resolution_deg'
USE_WAVELETS_KEY = 'use_wavelets'

DAYS_TO_SECONDS = 86400
DATE_FORMAT = '%Y%m%d'
NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)

MIN_RESOLUTION_DEG = 0.01
MAX_RESOLUTION_DEG = 5.

MIN_NORMALIZED_COORD = -3.
MAX_NORMALIZED_COORD = 3.

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
INCLUDE_TIME_DIM_KEY = 'include_time_dimension'
OMIT_NORTH_RADAR_KEY = 'omit_north_radar'
FIRST_VALID_DATE_KEY = 'first_valid_date_string'
LAST_VALID_DATE_KEY = 'last_valid_date_string'
NORMALIZE_FLAG_KEY = 'normalize'
UNIFORMIZE_FLAG_KEY = 'uniformize'
ADD_COORDS_KEY = 'add_coords'
FOURIER_TRANSFORM_KEY = 'fourier_transform_targets'
WAVELET_TRANSFORM_KEY = 'wavelet_transform_targets'
MIN_TARGET_RESOLUTION_KEY = 'min_target_resolution_deg'
MAX_TARGET_RESOLUTION_KEY = 'max_target_resolution_deg'
PREDICTOR_DIRECTORY_KEY = 'top_predictor_dir_name'
TARGET_DIRECTORY_KEY = 'top_target_dir_name'

DEFAULT_GENERATOR_OPTION_DICT = {
    BATCH_SIZE_KEY: 256,
    MAX_DAILY_EXAMPLES_KEY: 64,
    BAND_NUMBERS_KEY: satellite_io.BAND_NUMBERS,
    LAG_TIMES_KEY: numpy.array([0], dtype=int),
    OMIT_NORTH_RADAR_KEY: False,
    NORMALIZE_FLAG_KEY: True,
    UNIFORMIZE_FLAG_KEY: True,
    ADD_COORDS_KEY: False,
    FOURIER_TRANSFORM_KEY: False,
    WAVELET_TRANSFORM_KEY: False
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
LOSS_FUNCTION_KEY = 'loss_function_name'
QUANTILE_LEVELS_KEY = 'quantile_levels'
QFSS_HALF_WINDOW_SIZE_KEY = 'qfss_half_window_size_px'
METRIC_NAMES_KEY = 'metric_names'
MASK_MATRIX_KEY = 'mask_matrix'
FULL_MASK_MATRIX_KEY = 'full_mask_matrix'

METADATA_KEYS = [
    USE_PARTIAL_GRIDS_KEY, NUM_EPOCHS_KEY,
    NUM_TRAINING_BATCHES_KEY, TRAINING_OPTIONS_KEY,
    NUM_VALIDATION_BATCHES_KEY, VALIDATION_OPTIONS_KEY,
    EARLY_STOPPING_KEY, PLATEAU_LR_MUTIPLIER_KEY, LOSS_FUNCTION_KEY,
    QUANTILE_LEVELS_KEY, QFSS_HALF_WINDOW_SIZE_KEY, METRIC_NAMES_KEY,
    MASK_MATRIX_KEY, FULL_MASK_MATRIX_KEY
]

PREDICTOR_MATRIX_KEY = 'predictor_matrix'
TARGET_MATRIX_KEY = 'target_matrix'
VALID_TIMES_KEY = 'valid_times_unix_sec'
LATITUDES_KEY = 'latitudes_deg_n'
LONGITUDES_KEY = 'longitudes_deg_e'

NUM_FULL_ROWS_KEY = 'num_full_grid_rows'
NUM_FULL_COLUMNS_KEY = 'num_full_grid_columns'
NUM_PARTIAL_ROWS_KEY = 'num_partial_grid_rows'
NUM_PARTIAL_COLUMNS_KEY = 'num_partial_grid_columns'
OVERLAP_SIZE_KEY = 'overlap_size_px'
FIRST_INPUT_ROW_KEY = 'first_input_row'
LAST_INPUT_ROW_KEY = 'last_input_row'
FIRST_INPUT_COLUMN_KEY = 'first_input_column'
LAST_INPUT_COLUMN_KEY = 'last_input_column'


def _check_score_name(score_name, neigh_based=False, fourier_based=False,
                      wavelet_based=False):
    """Error-checks name of evaluation score.

    :param score_name: Name of evaluation score.
    :param neigh_based: Boolean flag.  If True, will ensure that score is valid
        for neighbourhood-based evaluation.
    :param fourier_based: Boolean flag.  If True, will ensure that score is
        valid for Fourier-transform-based evaluation.
    :param wavelet_based: Boolean flag.  If True, will ensure that score is
        valid for wavelet-transform-based evaluation.
    :raises: ValueError: if `score_name not in VALID_SCORE_NAMES`.
    """

    error_checking.assert_is_string(score_name)

    if neigh_based:
        valid_score_names = VALID_SCORE_NAMES_NEIGH
    elif fourier_based:
        valid_score_names = VALID_SCORE_NAMES_FOURIER
    elif wavelet_based:
        valid_score_names = VALID_SCORE_NAMES_WAVELET
    else:
        valid_score_names = ['']

    if score_name in valid_score_names:
        return

    error_string = (
        'Valid scores (listed below) do not include "{0:s}":\n{1:s}'
    ).format(
        score_name, str(valid_score_names)
    )

    raise ValueError(error_string)


def _check_generator_args(option_dict):
    """Error-checks input arguments for generator.

    :param option_dict: See doc for `generator_full_grid` or
        `generator_partial_grids`.
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

    error_checking.assert_is_boolean(option_dict[INCLUDE_TIME_DIM_KEY])
    error_checking.assert_is_boolean(option_dict[OMIT_NORTH_RADAR_KEY])

    error_checking.assert_is_boolean(option_dict[FOURIER_TRANSFORM_KEY])
    error_checking.assert_is_boolean(option_dict[WAVELET_TRANSFORM_KEY])
    if option_dict[FOURIER_TRANSFORM_KEY]:
        option_dict[WAVELET_TRANSFORM_KEY] = False

    if option_dict[FOURIER_TRANSFORM_KEY] or option_dict[WAVELET_TRANSFORM_KEY]:
        error_checking.assert_is_geq(option_dict[MIN_TARGET_RESOLUTION_KEY], 0.)
        error_checking.assert_is_greater(
            option_dict[MAX_TARGET_RESOLUTION_KEY],
            option_dict[MIN_TARGET_RESOLUTION_KEY]
        )

        if option_dict[MAX_TARGET_RESOLUTION_KEY] > MAX_TARGET_RESOLUTION_DEG:
            option_dict[MAX_TARGET_RESOLUTION_KEY] = numpy.inf
    else:
        option_dict[MIN_TARGET_RESOLUTION_KEY] = numpy.nan
        option_dict[MAX_TARGET_RESOLUTION_KEY] = numpy.nan

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


def _add_coords_to_predictors(predictor_matrix, predictor_dict, normalize):
    """Adds coordinates to predictors.

    :param predictor_matrix: See doc for `generator_full_grid`.
    :param predictor_dict: Dictionary returned by
        `example_io.read_predictor_file`.
    :param normalize: Boolean flag.  If True, will normalize coords to z-scores.
    :return: predictor_matrix: Same as input but with more channels.
    """

    num_examples = predictor_matrix.shape[0]
    num_grid_rows = predictor_matrix.shape[1]
    num_grid_columns = predictor_matrix.shape[2]
    has_time_dim = len(predictor_matrix.shape) == 5

    y_coords = predictor_dict[example_io.LATITUDES_KEY]

    if normalize:
        min_latitude_deg_n = numpy.min(twb_satellite_io.GRID_LATITUDES_DEG_N)
        max_latitude_deg_n = numpy.max(twb_satellite_io.GRID_LATITUDES_DEG_N)

        y_coords = (
            (y_coords - min_latitude_deg_n) /
            (max_latitude_deg_n - min_latitude_deg_n)
        )
        y_coords = MIN_NORMALIZED_COORD + y_coords * (
            MAX_NORMALIZED_COORD - MIN_NORMALIZED_COORD
        )

    y_coord_matrix = numpy.expand_dims(y_coords, axis=-1)
    y_coord_matrix = numpy.repeat(
        y_coord_matrix, repeats=num_grid_columns, axis=-1
    )

    y_coord_matrix = numpy.expand_dims(y_coord_matrix, axis=0)
    y_coord_matrix = numpy.repeat(y_coord_matrix, repeats=num_examples, axis=0)

    if has_time_dim:
        num_times = predictor_matrix.shape[-2]

        y_coord_matrix = numpy.expand_dims(y_coord_matrix, axis=-1)
        y_coord_matrix = numpy.repeat(
            y_coord_matrix, repeats=num_times, axis=-1
        )

    y_coord_matrix = numpy.expand_dims(y_coord_matrix, axis=-1)

    x_coords = predictor_dict[example_io.LONGITUDES_KEY]

    if normalize:
        min_longitude_deg_e = numpy.min(twb_satellite_io.GRID_LONGITUDES_DEG_E)
        max_longitude_deg_e = numpy.max(twb_satellite_io.GRID_LONGITUDES_DEG_E)

        x_coords = (
            (x_coords - min_longitude_deg_e) /
            (max_longitude_deg_e - min_longitude_deg_e)
        )
        x_coords = MIN_NORMALIZED_COORD + x_coords * (
            MAX_NORMALIZED_COORD - MIN_NORMALIZED_COORD
        )

    x_coord_matrix = numpy.expand_dims(x_coords, axis=0)
    x_coord_matrix = numpy.repeat(x_coord_matrix, repeats=num_grid_rows, axis=0)

    x_coord_matrix = numpy.expand_dims(x_coord_matrix, axis=0)
    x_coord_matrix = numpy.repeat(x_coord_matrix, repeats=num_examples, axis=0)

    if has_time_dim:
        num_times = predictor_matrix.shape[-2]

        x_coord_matrix = numpy.expand_dims(x_coord_matrix, axis=-1)
        x_coord_matrix = numpy.repeat(
            x_coord_matrix, repeats=num_times, axis=-1
        )

    x_coord_matrix = numpy.expand_dims(x_coord_matrix, axis=-1)

    return numpy.concatenate(
        (predictor_matrix, x_coord_matrix, y_coord_matrix), axis=-1
    )


def _read_inputs_one_day(
        valid_date_string, predictor_file_names, band_numbers,
        normalize, uniformize, add_coords, target_file_names, lead_time_seconds,
        lag_times_seconds, include_time_dimension, num_examples_to_read,
        return_coords, fourier_transform_targets, wavelet_transform_targets,
        min_target_resolution_deg, max_target_resolution_deg):
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
    :param add_coords: Same.
    :param target_file_names: 1-D list of paths to target files (readable by
        `example_io.read_target_file`).
    :param lead_time_seconds: See doc for `generator_full_grid`.
    :param lag_times_seconds: Same.
    :param include_time_dimension: Same.
    :param num_examples_to_read: Number of examples to read.
    :param return_coords: Boolean flag.  If True, will return latitudes and
        longitudes for grid points.
    :param fourier_transform_targets: See doc for `generator_full_grid`.
    :param wavelet_transform_targets: Same.
    :param min_target_resolution_deg: Same.
    :param max_target_resolution_deg: Same.
    :return: data_dict: Dictionary with the following keys.
    data_dict['predictor_matrix']: See doc for `generator_full_grid`.
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

    predictor_matrix = predictor_matrix_to_keras(
        predictor_matrix=predictor_matrix, num_lag_times=num_lag_times,
        add_time_dimension=include_time_dimension
    )

    if add_coords:
        predictor_matrix = _add_coords_to_predictors(
            predictor_matrix=predictor_matrix, predictor_dict=predictor_dict,
            normalize=normalize
        )

    target_matrix = target_dict[example_io.TARGET_MATRIX_KEY]
    print('Number of target values in batch = {0:d} ... mean = {1:.3g}'.format(
        target_matrix.size, numpy.mean(target_matrix)
    ))

    if fourier_transform_targets:
        target_matrix = target_matrix.astype(float)
        num_examples = target_matrix.shape[0]

        target_matrix = numpy.stack([
            fourier_utils.taper_spatial_data(target_matrix[i, ...])
            for i in range(num_examples)
        ], axis=0)

        blackman_matrix = fourier_utils.apply_blackman_window(
            numpy.ones(target_matrix.shape[1:])
        )
        target_matrix = numpy.stack([
            target_matrix[i, ...] * blackman_matrix for i in range(num_examples)
        ], axis=0)

        target_tensor = tensorflow.constant(
            target_matrix, dtype=tensorflow.complex128
        )
        target_weight_tensor = tensorflow.signal.fft2d(target_tensor)
        target_weight_matrix = K.eval(target_weight_tensor)

        butterworth_matrix = fourier_utils.apply_butterworth_filter(
            coefficient_matrix=numpy.ones(target_matrix.shape[1:]),
            filter_order=2, grid_spacing_metres=GRID_SPACING_DEG,
            min_resolution_metres=min_target_resolution_deg,
            max_resolution_metres=max_target_resolution_deg
        )

        target_weight_matrix = numpy.stack([
            target_weight_matrix[i, ...] * butterworth_matrix
            for i in range(num_examples)
        ], axis=0)

        target_weight_tensor = tensorflow.constant(
            target_weight_matrix, dtype=tensorflow.complex128
        )
        target_tensor = tensorflow.signal.ifft2d(target_weight_tensor)
        target_tensor = tensorflow.math.real(target_tensor)
        target_matrix = K.eval(target_tensor)

        target_matrix = numpy.stack([
            fourier_utils.untaper_spatial_data(target_matrix[i, ...])
            for i in range(num_examples)
        ], axis=0)

        target_matrix = numpy.maximum(target_matrix, 0.)
        target_matrix = numpy.minimum(target_matrix, 1.)

        print((
            'Number of target values and mean after Fourier transform = '
            '{0:d}, {1:.3g}'
        ).format(
            target_matrix.size, numpy.mean(target_matrix)
        ))

    if wavelet_transform_targets:
        target_matrix = target_matrix.astype(float)
        target_matrix, padding_arg = wavelet_utils.taper_spatial_data(
            target_matrix
        )

        coeff_tensor_by_level = wavelet_utils.do_forward_transform(
            target_matrix
        )
        coeff_tensor_by_level = wavelet_utils.filter_coefficients(
            coeff_tensor_by_level=coeff_tensor_by_level,
            grid_spacing_metres=GRID_SPACING_DEG,
            min_resolution_metres=min_target_resolution_deg,
            max_resolution_metres=max_target_resolution_deg, verbose=True
        )

        inverse_dwt_object = WaveTFFactory().build('haar', dim=2, inverse=True)
        target_tensor = inverse_dwt_object.call(coeff_tensor_by_level[0])
        target_matrix = K.eval(target_tensor)[..., 0]

        target_matrix = wavelet_utils.untaper_spatial_data(
            spatial_data_matrix=target_matrix, numpy_pad_width=padding_arg
        )
        target_matrix = numpy.maximum(target_matrix, 0.)
        target_matrix = numpy.minimum(target_matrix, 1.)

        print((
            'Number of target values and mean after wavelet transform = '
            '{0:d}, {1:.3g}'
        ).format(
            target_matrix.size, numpy.mean(target_matrix)
        ))

    data_dict = {
        PREDICTOR_MATRIX_KEY: predictor_matrix.astype('float16'),
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
        loss_function_name, quantile_levels, qfss_half_window_size_px,
        metric_names, mask_matrix, full_mask_matrix):
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
    :param loss_function_name: Same.
    :param quantile_levels: Same.
    :param qfss_half_window_size_px: Same.
    :param metric_names: Same.
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
        LOSS_FUNCTION_KEY: loss_function_name,
        QUANTILE_LEVELS_KEY: quantile_levels,
        QFSS_HALF_WINDOW_SIZE_KEY: qfss_half_window_size_px,
        METRIC_NAMES_KEY: metric_names,
        MASK_MATRIX_KEY: mask_matrix,
        FULL_MASK_MATRIX_KEY: full_mask_matrix
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=dill_file_name)

    dill_file_handle = open(dill_file_name, 'wb')
    dill.dump(metadata_dict, dill_file_handle)
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


def _get_input_px_for_partial_grid(partial_grid_dict):
    """Returns input pixels for partial grid.

    "Input pixels" = indices in full grid from which predictors will be taken

    :param partial_grid_dict: Dictionary with the following keys.
    partial_grid_dict['num_full_grid_rows']: Number of rows in full grid.
    partial_grid_dict['num_full_grid_columns']: Number of columns in full grid.
    partial_grid_dict['num_partial_grid_rows']: Number of rows in each partial
        grid.
    partial_grid_dict['num_partial_grid_columns']: Number of columns in each
        partial grid.
    partial_grid_dict['overlap_size_px']: Overlap between adjacent partial grids
        (number of pixels).
    partial_grid_dict['last_input_row']: Last input row from previous partial
        grid.  If last_input_row = i, this means the last row in the previous
        partial grid is the [i]th row in the full grid.  If there was no
        previous partial grid, make this -1.
    partial_grid_dict['last_input_column']: Same but for last column.
    partial_grid_dict['first_input_row']: Same but for first row.
    partial_grid_dict['first_input_column']: Same but for first column.

    :return: partial_grid_dict: Same but with different values for the last 4
        keys.
    """

    num_full_grid_rows = partial_grid_dict[NUM_FULL_ROWS_KEY]
    num_full_grid_columns = partial_grid_dict[NUM_FULL_COLUMNS_KEY]
    num_partial_grid_rows = partial_grid_dict[NUM_PARTIAL_ROWS_KEY]
    num_partial_grid_columns = partial_grid_dict[NUM_PARTIAL_COLUMNS_KEY]
    overlap_size_px = partial_grid_dict[OVERLAP_SIZE_KEY]
    last_input_row = partial_grid_dict[LAST_INPUT_ROW_KEY]
    last_input_column = partial_grid_dict[LAST_INPUT_COLUMN_KEY]

    if last_input_row < 0:
        last_input_row = num_partial_grid_rows - 1
        last_input_column = num_partial_grid_columns - 1
    elif last_input_column == num_full_grid_columns - 1:
        if last_input_row == num_full_grid_rows - 1:
            last_input_row = -1
            last_input_column = -1
        else:
            last_input_row += num_partial_grid_rows - 2 * overlap_size_px
            last_input_column = num_partial_grid_columns - 1
    else:
        last_input_column += num_partial_grid_columns - 2 * overlap_size_px

    last_input_row = min([
        last_input_row, num_full_grid_rows - 1
    ])
    last_input_column = min([
        last_input_column, num_full_grid_columns - 1
    ])

    first_input_row = last_input_row - num_partial_grid_rows + 1
    first_input_column = last_input_column - num_partial_grid_columns + 1

    partial_grid_dict[FIRST_INPUT_ROW_KEY] = first_input_row
    partial_grid_dict[FIRST_INPUT_COLUMN_KEY] = first_input_column
    partial_grid_dict[LAST_INPUT_ROW_KEY] = last_input_row
    partial_grid_dict[LAST_INPUT_COLUMN_KEY] = last_input_column

    return partial_grid_dict


def predictor_matrix_to_keras(predictor_matrix, num_lag_times,
                              add_time_dimension):
    """Reshapes predictor matrix into format required by Keras.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    B = number of spectral bands
    L = number of lag times

    :param predictor_matrix: numpy array (EL x M x N x B) of predictors.
    :param num_lag_times: Number of lag times.
    :param add_time_dimension: Boolean flag.  If True, will add time dimension,
        so output will be E x M x N x L x B.  If True, will just reorder data,
        so output will be E x M x N x BL.
    :return: predictor_matrix: numpy array of predictors.
    :raises: ValueError: if length of first axis of predictor matrix is not an
        integer multiple of L.
    """

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)
    error_checking.assert_is_numpy_array(predictor_matrix, num_dimensions=4)
    error_checking.assert_is_integer(num_lag_times)
    error_checking.assert_is_greater(num_lag_times, 0)
    error_checking.assert_is_boolean(add_time_dimension)

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
    predictor_matrix_by_lag = [
        predictor_matrix[j::num_lag_times, ...] for j in range(num_lag_times)
    ]

    if add_time_dimension:
        return numpy.stack(predictor_matrix_by_lag, axis=-2)

    num_bands = predictor_matrix.shape[-1]
    predictor_matrix = numpy.stack(predictor_matrix_by_lag, axis=-1)

    num_channels = num_bands * num_lag_times
    these_dim = predictor_matrix.shape[:-2] + (num_channels,)
    return numpy.reshape(predictor_matrix, these_dim)


def predictor_matrix_from_keras(predictor_matrix, num_lag_times):
    """Inverse of `predictor_matrix_to_keras`.

    :param predictor_matrix: See output doc for `predictor_matrix_to_keras`.
    :param num_lag_times: Same.
    :return: predictor_matrix: Same.
    """

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)
    num_dimensions = len(predictor_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, 4)
    error_checking.assert_is_leq(num_dimensions, 5)
    error_checking.assert_is_integer(num_lag_times)
    error_checking.assert_is_greater(num_lag_times, 0)

    if num_dimensions == 5:
        num_examples = predictor_matrix.shape[0]
        predictor_matrix_by_example = [
            predictor_matrix[i, ...] for i in range(num_examples)
        ]

        predictor_matrix_by_example = [
            numpy.swapaxes(a, 0, 2) for a in predictor_matrix_by_example
        ]
        predictor_matrix_by_example = [
            numpy.swapaxes(a, 1, 2) for a in predictor_matrix_by_example
        ]

        return numpy.concatenate(predictor_matrix_by_example, axis=0)

    last_axis_length = predictor_matrix.shape[-1]
    num_bands = float(last_axis_length) / num_lag_times
    this_diff = numpy.absolute(numpy.round(num_bands) - num_bands)

    if this_diff > TOLERANCE:
        error_string = (
            'Length of last axis of predictor matrix ({0:d}) must be an '
            'integer multiple of the number of lag times ({1:d}).'
        ).format(last_axis_length, num_lag_times)

        raise ValueError(error_string)

    num_bands = int(numpy.round(num_bands))
    these_dim = predictor_matrix.shape[:-1] + (num_bands, num_lag_times)
    predictor_matrix = numpy.reshape(predictor_matrix, these_dim)

    num_examples = predictor_matrix.shape[0]
    predictor_matrix_by_example = [
        predictor_matrix[i, ...] for i in range(num_examples)
    ]
    predictor_matrix_by_example = [
        numpy.swapaxes(a, 3, 2) for a in predictor_matrix_by_example
    ]
    predictor_matrix_by_example = [
        numpy.swapaxes(a, 2, 1) for a in predictor_matrix_by_example
    ]
    predictor_matrix_by_example = [
        numpy.swapaxes(a, 1, 0) for a in predictor_matrix_by_example
    ]

    return numpy.concatenate(predictor_matrix_by_example, axis=0)


def metric_params_to_name(
        score_name, weight=1., crps_weight=None, half_window_size_px=None,
        min_resolution_deg=None, max_resolution_deg=None, use_wavelets=False):
    """Converts parameters for evaluation metric to name.

    If `half_window_size_px is not None`, will assume neighbourhood-based
    metric.

    If `half_window_size_px is None`, will assume scale-separation-based metric
    with Fourier decomposition.

    :param score_name: Name of score (must be accepted by `_check_score_name`).
    :param weight: Real number by which metric is multiplied.
    :param crps_weight: Real number by which CRPS part of metric is multiplied.
        Used only if metric is FSS + CRPS.
    :param half_window_size_px: Half-window size (pixels) for neighbourhood.
    :param min_resolution_deg: Minimum resolution (degrees) allowed through
        band-pass filter.
    :param max_resolution_deg: Max resolution (degrees) allowed through
        band-pass filter.
    :param use_wavelets: Boolean flag.  If True (False), will use wavelet
        (Fourier) decomposition for band-pass filter.
    :return: metric_name: Metric name (string).
    """

    error_checking.assert_is_greater(weight, 0.)

    if score_name == FSS_PLUS_CRPS_NAME:
        error_checking.assert_is_geq(crps_weight, 1.)
    else:
        crps_weight = None

    if half_window_size_px is not None:
        error_checking.assert_is_not_nan(half_window_size_px)
        half_window_size_px = int(numpy.round(half_window_size_px))
        error_checking.assert_is_geq(half_window_size_px, 0)

        _check_score_name(score_name=score_name, neigh_based=True)

        return '{0:s}_neigh{1:d}_weight{2:.10f}_crps-weight{3:.10f}'.format(
            score_name, half_window_size_px, weight,
            0. if crps_weight is None else crps_weight
        )

    error_checking.assert_is_boolean(use_wavelets)
    _check_score_name(
        score_name=score_name, fourier_based=not use_wavelets,
        wavelet_based=use_wavelets
    )

    error_checking.assert_is_geq(min_resolution_deg, 0.)
    error_checking.assert_is_greater(max_resolution_deg, min_resolution_deg)

    if min_resolution_deg <= MIN_RESOLUTION_DEG:
        min_resolution_deg = 0.
    if max_resolution_deg >= MAX_RESOLUTION_DEG:
        max_resolution_deg = numpy.inf

    return (
        '{0:s}_{1:.4f}d_{2:.4f}d_wavelets{3:d}_weight{4:.10f}_'
        'crps-weight{5:.10f}'
    ).format(
        score_name, min_resolution_deg, max_resolution_deg, int(use_wavelets),
        weight,
        0. if crps_weight is None else crps_weight
    )


def metric_name_to_params(metric_name):
    """Converts name of evaluation metric to parameters.

    This method is the inverse of `metric_params_to_name`.

    :param metric_name: Metric name (string).
    :return: param_dict: Dictionary with the following keys.
    param_dict['score_name']: See doc for `metric_params_to_name`.
    param_dict['weight']: Same.
    param_dict['crps_weight']: Same.
    param_dict['half_window_size_px']: Same.
    param_dict['min_resolution_deg']: Same.
    param_dict['max_resolution_deg']: Same.
    param_dict['use_wavelets']: Same.
    """

    error_checking.assert_is_string(metric_name)
    metric_name_parts = metric_name.split('_')

    if 'crps-weight' in metric_name_parts[-1]:
        crps_weight = float(
            metric_name_parts[-1].replace('crps-weight', '')
        )
        metric_name_parts = metric_name_parts[:-1]
    else:
        crps_weight = None

    if 'weight' in metric_name_parts[-1]:
        weight = float(
            metric_name_parts[-1].replace('weight', '')
        )
        metric_name_parts = metric_name_parts[:-1]
    else:
        weight = 1.

    score_name = metric_name_parts[0]
    if score_name != FSS_PLUS_CRPS_NAME:
        crps_weight = None

    if len(metric_name_parts) == 2:
        assert metric_name_parts[1].startswith('neigh')
        half_window_size_px = int(
            metric_name_parts[1].replace('neigh', '')
        )

        _check_score_name(score_name=score_name, neigh_based=True)

        return {
            SCORE_NAME_KEY: score_name,
            WEIGHT_KEY: weight,
            CRPS_WEIGHT_KEY: crps_weight,
            HALF_WINDOW_SIZE_KEY: half_window_size_px,
            MIN_RESOLUTION_KEY: None,
            MAX_RESOLUTION_KEY: None,
            USE_WAVELETS_KEY: False
        }

    assert len(metric_name_parts) == 3 or len(metric_name_parts) == 4

    assert metric_name_parts[1].endswith('d')
    min_resolution_deg = float(
        metric_name_parts[1].replace('d', '')
    )

    assert metric_name_parts[2].endswith('d')
    max_resolution_deg = float(
        metric_name_parts[2].replace('d', '')
    )

    error_checking.assert_is_geq(min_resolution_deg, 0.)
    error_checking.assert_is_greater(max_resolution_deg, min_resolution_deg)

    if len(metric_name_parts) == 4:
        use_wavelets = bool(int(metric_name_parts[-1][-1]))
    else:
        use_wavelets = False

    _check_score_name(
        score_name=score_name, fourier_based=not use_wavelets,
        wavelet_based=use_wavelets
    )

    return {
        SCORE_NAME_KEY: score_name,
        WEIGHT_KEY: weight,
        CRPS_WEIGHT_KEY: crps_weight,
        HALF_WINDOW_SIZE_KEY: None,
        MIN_RESOLUTION_KEY: min_resolution_deg,
        MAX_RESOLUTION_KEY: max_resolution_deg,
        USE_WAVELETS_KEY: use_wavelets
    }


def get_metrics_legacy(mask_matrix):
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


def _create_multiply_function(real_number):
    """Creates function that multiplies input by real number.

    :param real_number: Multiplier.
    :return: multiply_function: Function handle.
    """

    def multiply_function(input_array):
        """Multiplies input array by real number.

        :param input_array: numpy array.
        :return: output_array: numpy array.
        """

        return input_array * real_number

    return multiply_function


def _multiply_a_function(orig_function_handle, real_number):
    """Multiplies function by a real number.

    :param orig_function_handle: Handle for function to be multiplied.
    :param real_number: Real number.
    :return: new_function_handle: Handle for new function, which is the original
        function multiplied by the given number.
    """

    this_function_handle = _create_multiply_function(real_number)
    return lambda x, y: this_function_handle(orig_function_handle(x, y))


def get_metrics(metric_names, mask_matrix, use_as_loss_function):
    """Returns metrics used for on-the-fly monitoring.

    K = number of metrics
    M = number of rows in grid
    N = number of columns in grid

    :param metric_names: length-K list of metric names (each must be accepted by
        `metric_name_to_params`).
    :param mask_matrix: M-by-N numpy array of Boolean flags.  Only pixels marked
        "True" are considered in each metric.
    :param use_as_loss_function: Boolean flag.  If True (False), will return
        each metric to be used as a loss function (metric).
    :return: metric_function_list: length-K list of functions.
    :return: metric_function_dict: Dictionary, where each key is a function name
        and each value is a function itself.
    """

    error_checking.assert_is_string_list(metric_names)
    error_checking.assert_is_boolean_numpy_array(mask_matrix)
    error_checking.assert_is_numpy_array(mask_matrix, num_dimensions=2)

    if len(metric_names) > 1:
        use_as_loss_function = False

    error_checking.assert_is_boolean(use_as_loss_function)

    fourier_dimensions = 3 * numpy.array(mask_matrix.shape, dtype=int)
    metric_function_list = []
    metric_function_dict = dict()

    for this_metric_name in metric_names:
        this_param_dict = metric_name_to_params(this_metric_name)

        if this_param_dict[USE_WAVELETS_KEY]:
            if this_param_dict[SCORE_NAME_KEY] == BRIER_SCORE_NAME:
                this_function = wavelet_metrics.brier_score(
                    min_resolution_deg=this_param_dict[MIN_RESOLUTION_KEY],
                    max_resolution_deg=this_param_dict[MAX_RESOLUTION_KEY],
                    mask_matrix=mask_matrix,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == CROSS_ENTROPY_NAME:
                this_function = wavelet_metrics.cross_entropy(
                    min_resolution_deg=this_param_dict[MIN_RESOLUTION_KEY],
                    max_resolution_deg=this_param_dict[MAX_RESOLUTION_KEY],
                    mask_matrix=mask_matrix,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == CSI_NAME:
                this_function = wavelet_metrics.csi(
                    min_resolution_deg=this_param_dict[MIN_RESOLUTION_KEY],
                    max_resolution_deg=this_param_dict[MAX_RESOLUTION_KEY],
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == HEIDKE_SCORE_NAME:
                this_function = wavelet_metrics.heidke_score(
                    min_resolution_deg=this_param_dict[MIN_RESOLUTION_KEY],
                    max_resolution_deg=this_param_dict[MAX_RESOLUTION_KEY],
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == PEIRCE_SCORE_NAME:
                this_function = wavelet_metrics.peirce_score(
                    min_resolution_deg=this_param_dict[MIN_RESOLUTION_KEY],
                    max_resolution_deg=this_param_dict[MAX_RESOLUTION_KEY],
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == GERRITY_SCORE_NAME:
                this_function = wavelet_metrics.gerrity_score(
                    min_resolution_deg=this_param_dict[MIN_RESOLUTION_KEY],
                    max_resolution_deg=this_param_dict[MAX_RESOLUTION_KEY],
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == FREQUENCY_BIAS_NAME:
                this_function = wavelet_metrics.frequency_bias(
                    min_resolution_deg=this_param_dict[MIN_RESOLUTION_KEY],
                    max_resolution_deg=this_param_dict[MAX_RESOLUTION_KEY],
                    mask_matrix=mask_matrix,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == FSS_NAME:
                this_function = wavelet_metrics.pixelwise_fss(
                    min_resolution_deg=this_param_dict[MIN_RESOLUTION_KEY],
                    max_resolution_deg=this_param_dict[MAX_RESOLUTION_KEY],
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == IOU_NAME:
                this_function = wavelet_metrics.iou(
                    min_resolution_deg=this_param_dict[MIN_RESOLUTION_KEY],
                    max_resolution_deg=this_param_dict[MAX_RESOLUTION_KEY],
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == ALL_CLASS_IOU_NAME:
                this_function = wavelet_metrics.all_class_iou(
                    min_resolution_deg=this_param_dict[MIN_RESOLUTION_KEY],
                    max_resolution_deg=this_param_dict[MAX_RESOLUTION_KEY],
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            else:
                this_function = wavelet_metrics.dice_coeff(
                    min_resolution_deg=this_param_dict[MIN_RESOLUTION_KEY],
                    max_resolution_deg=this_param_dict[MAX_RESOLUTION_KEY],
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )

        elif this_param_dict[HALF_WINDOW_SIZE_KEY] is None:
            this_spatial_coeff_matrix = fourier_utils.apply_blackman_window(
                numpy.full(fourier_dimensions, 1.)
            )
            this_frequency_coeff_matrix = (
                fourier_utils.apply_butterworth_filter(
                    coefficient_matrix=numpy.full(fourier_dimensions, 1.),
                    filter_order=2., grid_spacing_metres=0.0125,
                    min_resolution_metres=this_param_dict[MIN_RESOLUTION_KEY],
                    max_resolution_metres=this_param_dict[MAX_RESOLUTION_KEY]
                )
            )

            if this_param_dict[SCORE_NAME_KEY] == FSS_NAME:
                this_function = fourier_metrics.pixelwise_fss(
                    spatial_coeff_matrix=this_spatial_coeff_matrix,
                    frequency_coeff_matrix=this_frequency_coeff_matrix,
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == BRIER_SCORE_NAME:
                this_function = fourier_metrics.brier_score(
                    spatial_coeff_matrix=this_spatial_coeff_matrix,
                    frequency_coeff_matrix=this_frequency_coeff_matrix,
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == CROSS_ENTROPY_NAME:
                this_function = fourier_metrics.cross_entropy(
                    spatial_coeff_matrix=this_spatial_coeff_matrix,
                    frequency_coeff_matrix=this_frequency_coeff_matrix,
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == CSI_NAME:
                this_function = fourier_metrics.csi(
                    spatial_coeff_matrix=this_spatial_coeff_matrix,
                    frequency_coeff_matrix=this_frequency_coeff_matrix,
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == PEIRCE_SCORE_NAME:
                this_function = fourier_metrics.peirce_score(
                    spatial_coeff_matrix=this_spatial_coeff_matrix,
                    frequency_coeff_matrix=this_frequency_coeff_matrix,
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == HEIDKE_SCORE_NAME:
                this_function = fourier_metrics.heidke_score(
                    spatial_coeff_matrix=this_spatial_coeff_matrix,
                    frequency_coeff_matrix=this_frequency_coeff_matrix,
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == GERRITY_SCORE_NAME:
                this_function = fourier_metrics.gerrity_score(
                    spatial_coeff_matrix=this_spatial_coeff_matrix,
                    frequency_coeff_matrix=this_frequency_coeff_matrix,
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == FREQUENCY_BIAS_NAME:
                this_function = fourier_metrics.frequency_bias(
                    spatial_coeff_matrix=this_spatial_coeff_matrix,
                    frequency_coeff_matrix=this_frequency_coeff_matrix,
                    mask_matrix=mask_matrix,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == IOU_NAME:
                this_function = fourier_metrics.iou(
                    spatial_coeff_matrix=this_spatial_coeff_matrix,
                    frequency_coeff_matrix=this_frequency_coeff_matrix,
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == ALL_CLASS_IOU_NAME:
                this_function = fourier_metrics.all_class_iou(
                    spatial_coeff_matrix=this_spatial_coeff_matrix,
                    frequency_coeff_matrix=this_frequency_coeff_matrix,
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == DICE_COEFF_NAME:
                this_function = fourier_metrics.dice_coeff(
                    spatial_coeff_matrix=this_spatial_coeff_matrix,
                    frequency_coeff_matrix=this_frequency_coeff_matrix,
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == REAL_FREQ_MSE_NAME:
                this_function = fourier_metrics.frequency_domain_mse_real(
                    spatial_coeff_matrix=this_spatial_coeff_matrix,
                    frequency_coeff_matrix=this_frequency_coeff_matrix,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == IMAGINARY_FREQ_MSE_NAME:
                this_function = fourier_metrics.frequency_domain_mse_imag(
                    spatial_coeff_matrix=this_spatial_coeff_matrix,
                    frequency_coeff_matrix=this_frequency_coeff_matrix,
                    function_name=this_metric_name
                )
            else:
                this_function = fourier_metrics.frequency_domain_mse(
                    spatial_coeff_matrix=this_spatial_coeff_matrix,
                    frequency_coeff_matrix=this_frequency_coeff_matrix,
                    function_name=this_metric_name
                )
        else:
            # TODO(thunderhoser): Having some loss functions in
            # custom_losses.py, and some in custom_metrics.py, is a HACK.
            # Eventually, I need to put the bulk of the code into
            # custom_metrics.py and have the other modules (neigh_metrics.py,
            # wavelet_metrics.py, fourier_metrics.py) contain mostly wrapper
            # methods.

            if this_param_dict[SCORE_NAME_KEY] == FSS_NAME:
                this_function = custom_losses.fractions_skill_score(
                    half_window_size_px=this_param_dict[HALF_WINDOW_SIZE_KEY],
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == HEIDKE_SCORE_NAME:
                this_function = custom_losses.heidke_score(
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == PEIRCE_SCORE_NAME:
                this_function = custom_losses.peirce_score(
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == GERRITY_SCORE_NAME:
                this_function = custom_losses.gerrity_score(
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == BRIER_SCORE_NAME:
                this_function = custom_metrics.brier_score(
                    half_window_size_px=this_param_dict[HALF_WINDOW_SIZE_KEY],
                    mask_matrix=mask_matrix,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == CROSS_ENTROPY_NAME:
                this_function = custom_metrics.cross_entropy(
                    half_window_size_px=this_param_dict[HALF_WINDOW_SIZE_KEY],
                    mask_matrix=mask_matrix,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == CRPS_NAME:
                this_function = custom_metrics.crps(
                    half_window_size_px=this_param_dict[HALF_WINDOW_SIZE_KEY],
                    mask_matrix=mask_matrix,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == FSS_PLUS_CRPS_NAME:
                this_function = custom_metrics.fss_plus_crps(
                    half_window_size_px=this_param_dict[HALF_WINDOW_SIZE_KEY],
                    mask_matrix=mask_matrix,
                    crps_weight=this_param_dict[CRPS_WEIGHT_KEY],
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == CSI_NAME:
                this_function = custom_metrics.csi(
                    half_window_size_px=this_param_dict[HALF_WINDOW_SIZE_KEY],
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == FREQUENCY_BIAS_NAME:
                this_function = custom_metrics.frequency_bias(
                    half_window_size_px=this_param_dict[HALF_WINDOW_SIZE_KEY],
                    mask_matrix=mask_matrix,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == IOU_NAME:
                this_function = custom_metrics.iou(
                    half_window_size_px=this_param_dict[HALF_WINDOW_SIZE_KEY],
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            elif this_param_dict[SCORE_NAME_KEY] == ALL_CLASS_IOU_NAME:
                this_function = custom_metrics.all_class_iou(
                    half_window_size_px=this_param_dict[HALF_WINDOW_SIZE_KEY],
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )
            else:
                this_function = custom_metrics.dice_coeff(
                    half_window_size_px=this_param_dict[HALF_WINDOW_SIZE_KEY],
                    mask_matrix=mask_matrix,
                    use_as_loss_function=use_as_loss_function,
                    function_name=this_metric_name
                )

        this_function = _multiply_a_function(
            orig_function_handle=this_function,
            real_number=this_param_dict[WEIGHT_KEY]
        )
        metric_function_list.append(this_function)
        metric_function_dict[this_metric_name] = this_function

    return metric_function_list, metric_function_dict


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
    option_dict['include_time_dimension']: Same.
    option_dict['valid_date_string']: Valid date (format "yyyymmdd").  Will
        create examples with targets valid on this day.
    option_dict['normalize']: See doc for `generator_full_grid`.
    option_dict['uniformize']: Same.
    option_dict['add_coords']: Same.
    option_dict['fourier_transform_targets']: Same.
    option_dict['wavelet_transform_targets']: Same.
    option_dict['min_target_resolution_deg']: Same.
    option_dict['max_target_resolution_deg']: Same.

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
    include_time_dimension = option_dict[INCLUDE_TIME_DIM_KEY]
    valid_date_string = option_dict[VALID_DATE_KEY]
    normalize = option_dict[NORMALIZE_FLAG_KEY]
    uniformize = option_dict[UNIFORMIZE_FLAG_KEY]
    add_coords = option_dict[ADD_COORDS_KEY]
    fourier_transform_targets = option_dict[FOURIER_TRANSFORM_KEY]
    wavelet_transform_targets = option_dict[WAVELET_TRANSFORM_KEY]
    min_target_resolution_deg = option_dict[MIN_TARGET_RESOLUTION_KEY]
    max_target_resolution_deg = option_dict[MAX_TARGET_RESOLUTION_KEY]

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
        prefer_zipped=False, allow_other_format=True,
        raise_error_if_all_missing=False,
        raise_error_if_any_missing=False
    )

    target_file_names = example_io.find_many_target_files(
        top_directory_name=top_target_dir_name,
        first_date_string=valid_date_string,
        last_date_string=valid_date_string,
        prefer_zipped=False, allow_other_format=True,
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
        band_numbers=band_numbers, normalize=normalize,
        uniformize=uniformize, add_coords=add_coords,
        target_file_names=target_file_names,
        lead_time_seconds=lead_time_seconds,
        lag_times_seconds=lag_times_seconds,
        include_time_dimension=include_time_dimension,
        num_examples_to_read=int(1e6), return_coords=return_coords,
        fourier_transform_targets=fourier_transform_targets,
        wavelet_transform_targets=wavelet_transform_targets,
        min_target_resolution_deg=min_target_resolution_deg,
        max_target_resolution_deg=max_target_resolution_deg
    )


def create_data_partial_grids(option_dict, return_coords=False,
                              radar_number=None):
    """Creates input data on partial, radar-centered grids.

    This method is the same as `generator_partial_grids`, except that
    it returns all the data at once, rather than generating batches on the fly.

    R = number of radar sites

    :param option_dict: Dictionary with the following keys.
    option_dict['top_predictor_dir_name']: See doc for
        `generator_partial_grids`.
    option_dict['top_target_dir_name']: Same.
    option_dict['band_numbers']: Same.
    option_dict['lead_time_seconds']: Same.
    option_dict['lag_times_seconds']: Same.
    option_dict['include_time_dimension']: Same.
    option_dict['omit_north_radar']: Same.
    option_dict['valid_date_string']: Valid date (format "yyyymmdd").  Will
        create examples with targets valid on this day.
    option_dict['normalize']: See doc for `generator_partial_grids`.
    option_dict['uniformize']: Same.
    option_dict['add_coords']: Same.
    option_dict['fourier_transform_targets']: Same.
    option_dict['wavelet_transform_targets']: Same.
    option_dict['min_target_resolution_deg']: Same.
    option_dict['max_target_resolution_deg']: Same.

    :param return_coords: See doc for `_read_inputs_one_day`.
    :param radar_number: Will return data only for this radar (non-negative
        integer).  If None, will return data for all radars.
    :return: data_dicts: length-R list of dictionaries, each in format returned
        by `_read_inputs_one_day`.
    """

    option_dict = _check_generator_args(option_dict)
    error_checking.assert_is_boolean(return_coords)

    if radar_number is not None:
        error_checking.assert_is_integer(radar_number)
        error_checking.assert_is_geq(radar_number, 0)

    top_predictor_dir_name = option_dict[PREDICTOR_DIRECTORY_KEY]
    top_target_dir_name = option_dict[TARGET_DIRECTORY_KEY]
    band_numbers = option_dict[BAND_NUMBERS_KEY]
    lead_time_seconds = option_dict[LEAD_TIME_KEY]
    lag_times_seconds = option_dict[LAG_TIMES_KEY]
    include_time_dimension = option_dict[INCLUDE_TIME_DIM_KEY]
    omit_north_radar = (
        option_dict[OMIT_NORTH_RADAR_KEY] and radar_number is None
    )
    valid_date_string = option_dict[VALID_DATE_KEY]
    normalize = option_dict[NORMALIZE_FLAG_KEY]
    uniformize = option_dict[UNIFORMIZE_FLAG_KEY]
    add_coords = option_dict[ADD_COORDS_KEY]
    fourier_transform_targets = option_dict[FOURIER_TRANSFORM_KEY]
    wavelet_transform_targets = option_dict[WAVELET_TRANSFORM_KEY]
    min_target_resolution_deg = option_dict[MIN_TARGET_RESOLUTION_KEY]
    max_target_resolution_deg = option_dict[MAX_TARGET_RESOLUTION_KEY]

    if lead_time_seconds > 0 or numpy.any(lag_times_seconds > 0):
        first_init_date_string = general_utils.get_previous_date(
            valid_date_string
        )
    else:
        first_init_date_string = copy.deepcopy(valid_date_string)

    valid_date_strings = None
    data_dicts = [dict()] * NUM_RADARS

    for k in range(NUM_RADARS):
        if omit_north_radar and k == 0:
            continue

        if radar_number is not None and radar_number != k:
            continue

        these_predictor_file_names = example_io.find_many_predictor_files(
            top_directory_name=top_predictor_dir_name,
            first_date_string=first_init_date_string,
            last_date_string=valid_date_string, radar_number=k,
            prefer_zipped=False, allow_other_format=True,
            raise_error_if_all_missing=False,
            raise_error_if_any_missing=False
        )

        these_target_file_names = example_io.find_many_target_files(
            top_directory_name=top_target_dir_name,
            first_date_string=valid_date_string,
            last_date_string=valid_date_string, radar_number=k,
            prefer_zipped=False, allow_other_format=True,
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
            uniformize=uniformize, add_coords=add_coords,
            target_file_names=these_target_file_names,
            lead_time_seconds=lead_time_seconds,
            lag_times_seconds=lag_times_seconds,
            include_time_dimension=include_time_dimension,
            num_examples_to_read=int(1e6), return_coords=return_coords,
            fourier_transform_targets=fourier_transform_targets,
            wavelet_transform_targets=wavelet_transform_targets,
            min_target_resolution_deg=min_target_resolution_deg,
            max_target_resolution_deg=max_target_resolution_deg
        )

    return data_dicts


def generator_full_grid(option_dict):
    """Generates training data on full satellite grid.

    E = number of examples per batch
    M = number of rows in grid
    N = number of columns in grid
    L = number of lag times
    B = number of spectral bands

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
    option_dict['include_time_dimension']: Boolean flag.  If True, predictor
        matrix will include a time dimension.  If False, times and spectral
        bands will be combined into the last axis.
    option_dict['first_valid_date_string']: First valid date (format
        "yyyymmdd").  Will not generate examples with earlier valid times.
    option_dict['last_valid_date_string']: Last valid date (format
        "yyyymmdd").  Will not generate examples with later valid times.
    option_dict['normalize']: Boolean flag.  If True (False), will use
        normalized (unnormalized) predictors.
    option_dict['uniformize']: [used only if `normalize == True`]
        Boolean flag.  If True, will use uniformized and normalized predictors.
        If False, will use only normalized predictors.
    option_dict['add_coords']: Boolean flag.  If True (False), will use
        coordinates (latitude and longitude) as predictors.
    option_dict['fourier_transform_targets']: Boolean flag.  If True, will use
        Fourier transform to apply band-pass filter to targets.
    option_dict['wavelet_transform_targets']: Boolean flag.  If True, will use
        wavelet transform to apply band-pass filter to targets.
    option_dict['min_target_resolution_deg']: Minimum resolution (degrees) to
        allow through band-pass filter.
    option_dict['max_target_resolution_deg']: Max resolution (degrees) to allow
        through band-pass filter.

    :return: predictor_matrix: numpy array (E x M x N x LB or E x M x N x L x B)
        of predictor values, based on satellite data.
    :return: target_matrix: E-by-M-by-N-by-1 numpy array of target values
        (floats in 0...1, indicating whether or not convection occurs at
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
    include_time_dimension = option_dict[INCLUDE_TIME_DIM_KEY]
    first_valid_date_string = option_dict[FIRST_VALID_DATE_KEY]
    last_valid_date_string = option_dict[LAST_VALID_DATE_KEY]
    normalize = option_dict[NORMALIZE_FLAG_KEY]
    uniformize = option_dict[UNIFORMIZE_FLAG_KEY]
    add_coords = option_dict[ADD_COORDS_KEY]
    fourier_transform_targets = option_dict[FOURIER_TRANSFORM_KEY]
    wavelet_transform_targets = option_dict[WAVELET_TRANSFORM_KEY]
    min_target_resolution_deg = option_dict[MIN_TARGET_RESOLUTION_KEY]
    max_target_resolution_deg = option_dict[MAX_TARGET_RESOLUTION_KEY]

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
        prefer_zipped=False, allow_other_format=True,
        raise_error_if_any_missing=False
    )

    target_file_names = example_io.find_many_target_files(
        top_directory_name=top_target_dir_name,
        first_date_string=first_init_date_string,
        last_date_string=last_valid_date_string,
        prefer_zipped=False, allow_other_format=True,
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

    valid_date_strings = numpy.array(valid_date_strings)
    numpy.random.shuffle(valid_date_strings)
    valid_date_strings = valid_date_strings.tolist()
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
                band_numbers=band_numbers, normalize=normalize,
                uniformize=uniformize, add_coords=add_coords,
                target_file_names=target_file_names,
                lead_time_seconds=lead_time_seconds,
                lag_times_seconds=lag_times_seconds,
                include_time_dimension=include_time_dimension,
                num_examples_to_read=num_examples_to_read, return_coords=False,
                fourier_transform_targets=fourier_transform_targets,
                wavelet_transform_targets=wavelet_transform_targets,
                min_target_resolution_deg=min_target_resolution_deg,
                max_target_resolution_deg=max_target_resolution_deg
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

        predictor_matrix = predictor_matrix.astype('float16')
        target_matrix = target_matrix.astype('float16')
        yield predictor_matrix, target_matrix


def generator_partial_grids(option_dict):
    """Generates training data on partial, radar-centered grids.

    :param option_dict: Same as input to `generator_full_grid`, except with one
        extra key.
    option_dict['omit_north_radar']: Boolean flag.  If True, will not generate
        partial grids centered on northernmost radar.

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
    include_time_dimension = option_dict[INCLUDE_TIME_DIM_KEY]
    omit_north_radar = option_dict[OMIT_NORTH_RADAR_KEY]
    first_valid_date_string = option_dict[FIRST_VALID_DATE_KEY]
    last_valid_date_string = option_dict[LAST_VALID_DATE_KEY]
    normalize = option_dict[NORMALIZE_FLAG_KEY]
    uniformize = option_dict[UNIFORMIZE_FLAG_KEY]
    add_coords = option_dict[ADD_COORDS_KEY]
    fourier_transform_targets = option_dict[FOURIER_TRANSFORM_KEY]
    wavelet_transform_targets = option_dict[WAVELET_TRANSFORM_KEY]
    min_target_resolution_deg = option_dict[MIN_TARGET_RESOLUTION_KEY]
    max_target_resolution_deg = option_dict[MAX_TARGET_RESOLUTION_KEY]

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
        prefer_zipped=False, allow_other_format=True,
        raise_error_if_any_missing=False
    )
    these_target_file_names = example_io.find_many_target_files(
        top_directory_name=top_target_dir_name,
        first_date_string=first_init_date_string,
        last_date_string=last_valid_date_string, radar_number=0,
        prefer_zipped=False, allow_other_format=True,
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
            'are available.'
        )

    for k in range(1, NUM_RADARS):
        these_predictor_file_names = [
            example_io.find_predictor_file(
                top_directory_name=top_predictor_dir_name, date_string=d,
                radar_number=k, prefer_zipped=False, allow_other_format=True,
                raise_error_if_missing=True
            ) for d in predictor_date_strings
        ]

        these_target_file_names = [
            example_io.find_target_file(
                top_directory_name=top_target_dir_name, date_string=d,
                radar_number=k, prefer_zipped=False, allow_other_format=True,
                raise_error_if_missing=True
            ) for d in target_date_strings
        ]

        predictor_file_name_matrix[:, k] = numpy.array(
            these_predictor_file_names
        )
        target_file_name_matrix[:, k] = numpy.array(these_target_file_names)

    if omit_north_radar:
        predictor_file_name_matrix = predictor_file_name_matrix[:, 1:]
        target_file_name_matrix = target_file_name_matrix[:, 1:]

    num_radars = predictor_file_name_matrix.shape[1]
    radar_indices = numpy.linspace(0, num_radars - 1, num=num_radars, dtype=int)
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
                band_numbers=band_numbers, normalize=normalize,
                uniformize=uniformize, add_coords=add_coords,
                target_file_names=
                target_file_name_matrix[:, current_radar_index],
                lead_time_seconds=lead_time_seconds,
                lag_times_seconds=lag_times_seconds,
                include_time_dimension=include_time_dimension,
                num_examples_to_read=num_examples_to_read, return_coords=False,
                fourier_transform_targets=fourier_transform_targets,
                wavelet_transform_targets=wavelet_transform_targets,
                min_target_resolution_deg=min_target_resolution_deg,
                max_target_resolution_deg=max_target_resolution_deg
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

        predictor_matrix = predictor_matrix.astype('float16')
        target_matrix = target_matrix.astype('float16')
        yield predictor_matrix, target_matrix


def train_model(
        model_object, output_dir_name, num_epochs, use_partial_grids,
        num_training_batches_per_epoch, training_option_dict,
        num_validation_batches_per_epoch, validation_option_dict,
        mask_matrix, full_mask_matrix, loss_function_name, metric_names,
        quantile_levels=None, qfss_half_window_size_px=None,
        do_early_stopping=True,
        plateau_lr_multiplier=DEFAULT_LEARNING_RATE_MULTIPLIER,
        save_every_epoch=False):
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
    :param loss_function_name: Name of loss function.  Must be accepted by
        `metric_name_to_params`.
    :param metric_names: 1-D list of metric names.  Each name must be accepted
        by `metric_name_to_params`.
    :param quantile_levels: 1-D numpy array of quantile levels for quantile
        regression.  Levels must range from (0, 1).  If the model is not doing
        quantile regression, make this None.
    :param qfss_half_window_size_px:
        [used only if `quantile_levels is not None`]
        Half-neighbourhood size (pixels) for quantile-based FSS.  If pixelwise
        quantile loss is being used instead, make this None.
    :param do_early_stopping: Boolean flag.  If True, will stop training early
        if validation loss has not improved over last several epochs (see
        constants at top of file for what exactly this means).
    :param plateau_lr_multiplier: Multiplier for learning rate.  Learning
        rate will be multiplied by this factor upon plateau in validation
        performance.
    :param save_every_epoch: Boolean flag.  If True, will save new model after
        every epoch.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 2)
    error_checking.assert_is_boolean(use_partial_grids)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 2)
    error_checking.assert_is_integer(num_validation_batches_per_epoch)
    error_checking.assert_is_geq(num_validation_batches_per_epoch, 2)
    error_checking.assert_is_boolean(do_early_stopping)
    error_checking.assert_is_boolean(save_every_epoch)

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

    _ = metric_name_to_params(loss_function_name)

    if metric_names is not None:
        error_checking.assert_is_string_list(metric_names)
        for this_metric_name in metric_names:
            _ = metric_name_to_params(this_metric_name)

    if quantile_levels is None:
        qfss_half_window_size_px = None
    else:
        error_checking.assert_is_numpy_array(quantile_levels, num_dimensions=1)
        error_checking.assert_is_greater_numpy_array(quantile_levels, 0.)
        error_checking.assert_is_less_than_numpy_array(quantile_levels, 1.)
        error_checking.assert_is_greater_numpy_array(
            numpy.diff(quantile_levels), 0.
        )

    if qfss_half_window_size_px is not None:
        error_checking.assert_is_geq(qfss_half_window_size_px, 0.)

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

    if save_every_epoch:
        model_file_name = (
            output_dir_name +
            '/model_epoch={epoch:03d}_val-loss={val_loss:.6f}.h5'
        )
    else:
        model_file_name = '{0:s}/model.h5'.format(output_dir_name)

    history_object = keras.callbacks.CSVLogger(
        filename='{0:s}/history.csv'.format(output_dir_name),
        separator=',', append=False
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=not save_every_epoch, save_weights_only=False,
        mode='min', period=1
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
        loss_function_name=loss_function_name, quantile_levels=quantile_levels,
        qfss_half_window_size_px=qfss_half_window_size_px,
        metric_names=metric_names, mask_matrix=mask_matrix,
        full_mask_matrix=full_mask_matrix
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


def read_model(hdf5_file_name, for_mirrored_training=False):
    """Reads model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :param for_mirrored_training: Boolean flag.  If True, will read model for
        mirrored training (where each GPU reads one batch).  If False, will read
        model for any other purpose (inference or normal training).
    :return: model_object: Instance of `keras.models.Model`.
    """

    error_checking.assert_file_exists(hdf5_file_name)
    error_checking.assert_is_boolean(for_mirrored_training)

    metafile_name = find_metafile(
        model_file_name=hdf5_file_name, raise_error_if_missing=True
    )

    metadata_dict = read_metafile(metafile_name)
    mask_matrix = metadata_dict[MASK_MATRIX_KEY]
    loss_function_name = metadata_dict[LOSS_FUNCTION_KEY]
    quantile_levels = metadata_dict[QUANTILE_LEVELS_KEY]
    qfss_half_window_size_px = metadata_dict[QFSS_HALF_WINDOW_SIZE_KEY]
    metric_names = metadata_dict[METRIC_NAMES_KEY]

    if metric_names is None:
        metric_list, custom_object_dict = get_metrics_legacy(mask_matrix)
    else:
        metric_list, custom_object_dict = get_metrics(
            metric_names=metric_names, mask_matrix=mask_matrix,
            use_as_loss_function=False
        )

    loss_function = get_metrics(
        metric_names=[loss_function_name], mask_matrix=mask_matrix,
        use_as_loss_function=True
    )[0][0]

    if quantile_levels is None:
        custom_object_dict['loss'] = loss_function
    else:
        custom_object_dict = {'central_output_loss': loss_function}
        loss_dict = {'central_output': loss_function}
        metric_list = []

        for k in range(len(quantile_levels)):
            if qfss_half_window_size_px is None:
                this_loss_function = custom_losses.quantile_loss(
                    quantile_level=quantile_levels[k], mask_matrix=mask_matrix
                )
            else:
                this_loss_function = custom_losses.quantile_based_fss(
                    quantile_level=quantile_levels[k],
                    half_window_size_px=qfss_half_window_size_px,
                    use_as_loss_function=True,
                    mask_matrix=mask_matrix.astype(bool)
                )

            loss_dict['quantile_output{0:03d}'.format(k + 1)] = (
                this_loss_function
            )
            custom_object_dict['quantile_output{0:03d}_loss'.format(k + 1)] = (
                this_loss_function
            )

        custom_object_dict['loss'] = loss_dict

    model_object = tf_keras.models.load_model(
        hdf5_file_name, custom_objects=custom_object_dict, compile=False
    )

    if for_mirrored_training:
        strategy_object = tensorflow.distribute.MirroredStrategy()

        with strategy_object.scope():
            model_object = keras.models.Model.from_config(
                model_object.get_config()
            )

            model_object.compile(
                loss=custom_object_dict['loss'],
                optimizer=keras.optimizers.Adam(),
                metrics=metric_list
            )
    else:
        model_object.compile(
            loss=custom_object_dict['loss'], optimizer=keras.optimizers.Adam(),
            metrics=metric_list
        )

    return model_object


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
    metadata_dict['loss_function_name']: Same.
    metadata_dict['quantile_levels']: Same.
    metadata_dict['qfss_half_window_size_px']: Same.
    metadata_dict['metric_names']: Same.
    metadata_dict['mask_matrix']: Same.
    metadata_dict['full_mask_matrix']: Same.

    :raises: ValueError: if any expected key is not found in dictionary.
    """

    error_checking.assert_file_exists(dill_file_name)

    dill_file_handle = open(dill_file_name, 'rb')
    metadata_dict = dill.load(dill_file_handle)
    dill_file_handle.close()

    if LOSS_FUNCTION_KEY not in metadata_dict:
        metadata_dict[LOSS_FUNCTION_KEY] = metric_params_to_name(
            score_name=FSS_NAME,
            half_window_size_px=metadata_dict['fss_half_window_size_px']
        )

    if METRIC_NAMES_KEY not in metadata_dict:
        metadata_dict[METRIC_NAMES_KEY] = None
    if QUANTILE_LEVELS_KEY not in metadata_dict:
        metadata_dict[QUANTILE_LEVELS_KEY] = None
    if QFSS_HALF_WINDOW_SIZE_KEY not in metadata_dict:
        metadata_dict[QFSS_HALF_WINDOW_SIZE_KEY] = None

    num_grid_points = (
        len(twb_satellite_io.GRID_LATITUDES_DEG_N) *
        len(twb_satellite_io.GRID_LONGITUDES_DEG_E)
    )

    if num_grid_points != metadata_dict[FULL_MASK_MATRIX_KEY].size:
        full_mask_dict = {
            radar_io.MASK_MATRIX_KEY: metadata_dict[FULL_MASK_MATRIX_KEY],
            radar_io.LATITUDES_KEY: twb_radar_io.GRID_LATITUDES_DEG_N,
            radar_io.LONGITUDES_KEY: twb_radar_io.GRID_LONGITUDES_DEG_E
        }

        full_mask_dict = radar_io.expand_to_satellite_grid(
            any_radar_dict=full_mask_dict
        )

        metadata_dict[FULL_MASK_MATRIX_KEY] = (
            full_mask_dict[radar_io.MASK_MATRIX_KEY]
        )

    training_option_dict = metadata_dict[TRAINING_OPTIONS_KEY]
    validation_option_dict = metadata_dict[VALIDATION_OPTIONS_KEY]

    if LAG_TIMES_KEY not in training_option_dict:
        training_option_dict[LAG_TIMES_KEY] = numpy.array([0], dtype=int)
        validation_option_dict[LAG_TIMES_KEY] = numpy.array([0], dtype=int)

    if INCLUDE_TIME_DIM_KEY not in training_option_dict:
        training_option_dict[INCLUDE_TIME_DIM_KEY] = False
        validation_option_dict[INCLUDE_TIME_DIM_KEY] = False

    if OMIT_NORTH_RADAR_KEY not in training_option_dict:
        training_option_dict[OMIT_NORTH_RADAR_KEY] = False
        validation_option_dict[OMIT_NORTH_RADAR_KEY] = False

    if FOURIER_TRANSFORM_KEY not in training_option_dict:
        training_option_dict[FOURIER_TRANSFORM_KEY] = False
        validation_option_dict[FOURIER_TRANSFORM_KEY] = False

        training_option_dict[WAVELET_TRANSFORM_KEY] = False
        validation_option_dict[WAVELET_TRANSFORM_KEY] = False

        training_option_dict[MIN_TARGET_RESOLUTION_KEY] = numpy.nan
        validation_option_dict[MIN_TARGET_RESOLUTION_KEY] = numpy.nan

        training_option_dict[MAX_TARGET_RESOLUTION_KEY] = numpy.nan
        validation_option_dict[MAX_TARGET_RESOLUTION_KEY] = numpy.nan

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
        model_object, predictor_matrix, num_examples_per_batch,
        use_dropout=False, verbose=False):
    """Applies trained neural net to full grid.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    Q = number of quantile levels

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix: See output doc for
        `generator_full_grid`.
    :param num_examples_per_batch: Batch size.
    :param use_dropout: Boolean flag.  If True, will keep dropout in all layers
        turned on.  Using dropout at inference time is called "Monte Carlo
        dropout".
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: forecast_prob_matrix: numpy array of forecast event probabilities.
        If the model does quantile regression, this array will be E x M x N x Q.
        If not, E x M x N.
    """

    num_examples_per_batch = _check_inference_args(
        predictor_matrix=predictor_matrix,
        num_examples_per_batch=num_examples_per_batch, verbose=verbose
    )

    error_checking.assert_is_boolean(use_dropout)
    if use_dropout:
        for layer_object in model_object.layers:
            if 'batch' in layer_object.name.lower():
                print('Layer "{0:s}" set to NON-TRAINABLE!'.format(
                    layer_object.name
                ))
                layer_object.trainable = False

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

        if use_dropout:
            # this_prob_matrix = predict_function(
            #     [predictor_matrix[these_indices, ...], True]
            # )[0]

            these_predictions = model_object(
                predictor_matrix[these_indices, ...], training=True
            ).numpy()
        else:
            these_predictions = model_object.predict_on_batch(
                predictor_matrix[these_indices, ...]
            )

        if isinstance(these_predictions, list):
            this_prob_matrix = numpy.stack(these_predictions, axis=-1)

            if len(this_prob_matrix.shape) == 5:
                this_prob_matrix = this_prob_matrix[..., 0, :]
        else:
            this_prob_matrix = these_predictions + 0.

            if (
                    len(this_prob_matrix.shape) == 4 and
                    this_prob_matrix.shape[-1] <= 2
            ):
                this_prob_matrix = this_prob_matrix[..., 0]

        if forecast_prob_matrix is None:
            dimensions = (num_examples,) + this_prob_matrix.shape[1:]
            forecast_prob_matrix = numpy.full(dimensions, numpy.nan)

        forecast_prob_matrix[these_indices, ...] = this_prob_matrix

    if verbose:
        print('Have applied model to all {0:d} examples!'.format(num_examples))

    forecast_prob_matrix = numpy.maximum(forecast_prob_matrix, 0.)
    forecast_prob_matrix = numpy.minimum(forecast_prob_matrix, 1.)
    return forecast_prob_matrix


def apply_model_partial_grids(
        model_object, predictor_matrix, num_examples_per_batch, overlap_size_px,
        use_dropout=False, verbose=False):
    """Applies trained NN to full grid, predicting one partial grid at a time.

    :param model_object: See doc for `apply_model_full_grid`.
    :param predictor_matrix: Same.
    :param num_examples_per_batch: Same.
    :param overlap_size_px: Overlap between adjacent partial grids (number of
        pixels).
    :param use_dropout: See doc for `apply_model_full_grid`.
    :param verbose: Same.
    :return: forecast_prob_matrix: Same.
    """

    # Check input args.
    num_examples_per_batch = _check_inference_args(
        predictor_matrix=predictor_matrix,
        num_examples_per_batch=num_examples_per_batch, verbose=verbose
    )

    error_checking.assert_is_boolean(use_dropout)
    if use_dropout:
        for layer_object in model_object.layers:
            if 'batch' in layer_object.name.lower():
                print('Layer "{0:s}" set to NON-TRAINABLE!'.format(
                    layer_object.name
                ))
                layer_object.trainable = False

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
    summed_prob_matrix = None
    num_forecasts_matrix = None

    # summed_prob_matrix = numpy.full(
    #     (num_examples, num_full_grid_rows, num_full_grid_columns), 0.
    # )
    # num_forecasts_matrix = numpy.full(
    #     (num_examples, num_full_grid_rows, num_full_grid_columns), 0, dtype=int
    # )

    partial_grid_dict = {
        NUM_FULL_ROWS_KEY: num_full_grid_rows,
        NUM_FULL_COLUMNS_KEY: num_full_grid_columns,
        NUM_PARTIAL_ROWS_KEY: num_partial_grid_rows,
        NUM_PARTIAL_COLUMNS_KEY: num_partial_grid_columns,
        OVERLAP_SIZE_KEY: overlap_size_px,
        FIRST_INPUT_ROW_KEY: -1,
        LAST_INPUT_ROW_KEY: -1,
        FIRST_INPUT_COLUMN_KEY: -1,
        LAST_INPUT_COLUMN_KEY: -1
    }

    while True:
        partial_grid_dict = _get_input_px_for_partial_grid(partial_grid_dict)

        first_input_row = partial_grid_dict[FIRST_INPUT_ROW_KEY]
        if first_input_row < 0:
            break

        last_input_row = partial_grid_dict[LAST_INPUT_ROW_KEY]
        first_input_column = partial_grid_dict[FIRST_INPUT_COLUMN_KEY]
        last_input_column = partial_grid_dict[LAST_INPUT_COLUMN_KEY]

        # TODO(thunderhoser): The "50" here is a HACK.
        first_output_row = first_input_row + 50
        last_output_row = last_input_row - 50
        first_output_column = first_input_column + 50
        last_output_column = last_input_column - 50

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

            if use_dropout:
                # this_prob_matrix = predict_function(
                #     [this_predictor_matrix, True]
                # )[0]

                these_predictions = model_object(
                    this_predictor_matrix, training=True
                ).numpy()
            else:
                these_predictions = model_object.predict_on_batch(
                    this_predictor_matrix
                )

            if isinstance(these_predictions, list):
                this_prob_matrix = numpy.stack(these_predictions, axis=-1)

                if len(this_prob_matrix.shape) == 5:
                    this_prob_matrix = this_prob_matrix[..., 0, :]
            else:
                this_prob_matrix = these_predictions + 0.

                if len(this_prob_matrix.shape) == 4:
                    this_prob_matrix = this_prob_matrix[..., 0]

            this_prob_matrix = numpy.maximum(this_prob_matrix, 0.)
            this_prob_matrix = numpy.minimum(this_prob_matrix, 1.)

            if summed_prob_matrix is None:
                dimensions = (
                    (num_examples, num_full_grid_rows, num_full_grid_columns) +
                    this_prob_matrix.shape[3:]
                )
                summed_prob_matrix = numpy.full(dimensions, 0.)
                num_forecasts_matrix = numpy.full(dimensions, 0, dtype=int)

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

            # TODO(thunderhoser): The "50" here is a HACK.
            this_prob_matrix = this_prob_matrix[:, 50:-50, 50:-50, ...]

            summed_prob_matrix[
                first_example_index:(last_example_index + 1),
                first_output_row:(last_output_row + 1),
                first_output_column:(last_output_column + 1), ...
            ] += this_prob_matrix

            num_forecasts_matrix[
                first_example_index:(last_example_index + 1),
                first_output_row:(last_output_row + 1),
                first_output_column:(last_output_column + 1), ...
            ] += 1

    if verbose:
        print((
            'Have applied model to all grid points and all {0:d} examples!'
        ).format(
            num_examples
        ))

    num_forecasts_matrix = num_forecasts_matrix.astype(float)
    num_forecasts_matrix[num_forecasts_matrix < 0.01] = numpy.nan
    forecast_prob_matrix = summed_prob_matrix / num_forecasts_matrix
    forecast_prob_matrix[numpy.isnan(forecast_prob_matrix)] = 0.

    return forecast_prob_matrix
