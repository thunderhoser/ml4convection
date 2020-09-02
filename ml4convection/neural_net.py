"""Methods for training and applying neural nets."""

import os
import sys
import copy
import dill
import numpy
import keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import error_checking
import custom_metrics
import satellite_io
import radar_io
import normalization

TOLERANCE = 1e-6

DAYS_TO_SECONDS = 86400
DATE_FORMAT = '%Y%m%d'

PLATEAU_PATIENCE_EPOCHS = 10
DEFAULT_LEARNING_RATE_MULTIPLIER = 0.5
PLATEAU_COOLDOWN_EPOCHS = 0
EARLY_STOPPING_PATIENCE_EPOCHS = 30
LOSS_PATIENCE = 0.

METRIC_FUNCTION_LIST = [
    custom_metrics.accuracy, custom_metrics.binary_accuracy,
    custom_metrics.binary_csi, custom_metrics.binary_frequency_bias,
    custom_metrics.binary_pod, custom_metrics.binary_pofd,
    custom_metrics.binary_peirce_score, custom_metrics.binary_success_ratio,
    custom_metrics.binary_focn
]

METRIC_FUNCTION_DICT = {
    'accuracy': custom_metrics.accuracy,
    'binary_accuracy': custom_metrics.binary_accuracy,
    'binary_csi': custom_metrics.binary_csi,
    'binary_frequency_bias': custom_metrics.binary_frequency_bias,
    'binary_pod': custom_metrics.binary_pod,
    'binary_pofd': custom_metrics.binary_pofd,
    'binary_peirce_score': custom_metrics.binary_peirce_score,
    'binary_success_ratio': custom_metrics.binary_success_ratio,
    'binary_focn': custom_metrics.binary_focn
}

SATELLITE_DIRECTORY_KEY = 'top_satellite_dir_name'
RADAR_DIRECTORY_KEY = 'top_radar_dir_name'
BATCH_SIZE_KEY = 'num_examples_per_batch'
MAX_DAILY_EXAMPLES_KEY = 'max_examples_per_day_in_batch'
BAND_NUMBERS_KEY = 'band_numbers'
LEAD_TIME_KEY = 'lead_time_seconds'
REFL_THRESHOLD_KEY = 'reflectivity_threshold_dbz'
FIRST_VALID_DATE_KEY = 'first_valid_date_string'
LAST_VALID_DATE_KEY = 'last_valid_date_string'
NORMALIZATION_FILE_KEY = 'normalization_file_name'
UNIFORMIZE_FLAG_KEY = 'uniformize'

VALID_DATE_KEY = 'valid_date_string'
NORMALIZATION_DICT_KEY = 'norm_dict_for_count'

DEFAULT_GENERATOR_OPTION_DICT = {
    BATCH_SIZE_KEY: 256,
    MAX_DAILY_EXAMPLES_KEY: 64,
    BAND_NUMBERS_KEY: satellite_io.BAND_NUMBERS,
    REFL_THRESHOLD_KEY: 35.,
    UNIFORMIZE_FLAG_KEY: False
}

NUM_EPOCHS_KEY = 'num_epochs'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
TRAINING_OPTIONS_KEY = 'training_option_dict'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
VALIDATION_OPTIONS_KEY = 'validation_option_dict'
EARLY_STOPPING_KEY = 'do_early_stopping'
PLATEAU_LR_MUTIPLIER_KEY = 'plateau_lr_multiplier'

METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_TRAINING_BATCHES_KEY, TRAINING_OPTIONS_KEY,
    NUM_VALIDATION_BATCHES_KEY, VALIDATION_OPTIONS_KEY,
    EARLY_STOPPING_KEY, PLATEAU_LR_MUTIPLIER_KEY
]


def _check_generator_args(option_dict):
    """Error-checks input arguments for generator.

    :param option_dict: See doc for `data_generator`.
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
    error_checking.assert_is_less_than(
        option_dict[LEAD_TIME_KEY], DAYS_TO_SECONDS
    )
    error_checking.assert_is_greater(option_dict[REFL_THRESHOLD_KEY], 0.)

    return option_dict


def _check_inference_args(predictor_matrix, num_examples_per_batch, verbose):
    """Error-checks input arguments for inference.

    :param predictor_matrix: See doc for `apply_model`.
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


def _read_inputs_one_day(
        valid_date_string, satellite_file_names, band_numbers,
        norm_dict_for_count, uniformize, radar_file_names, lead_time_seconds,
        reflectivity_threshold_dbz, num_examples_to_read):
    """Reads inputs (satellite and radar data) for one day.

    :param valid_date_string: Valid date (format "yyyymmdd").
    :param satellite_file_names: 1-D list of paths to satellite files (readable
        by `satellite_io.read_file`).
    :param band_numbers: See doc for `data_generator`.
    :param norm_dict_for_count: Dictionary returned by
        `normalization.read_file`.  Will use this to normalize satellite data.
        If None, will not normalize.
    :param uniformize: See doc for `data_generator`.
    :param radar_file_names: 1-D list of paths to radar files (readable by
        `radar_io.read_2d_file`).
    :param lead_time_seconds: See doc for `data_generator`.
    :param reflectivity_threshold_dbz: Same.
    :param num_examples_to_read: Number of examples to read.
    :return: predictor_matrix: See doc for `data_generator`.
    :return: target_matrix: Same.
    """

    radar_date_strings = [
        radar_io.file_name_to_date(f) for f in radar_file_names
    ]
    index = radar_date_strings.index(valid_date_string)
    desired_radar_file_name = radar_file_names[index]

    satellite_date_strings = [
        satellite_io.file_name_to_date(f) for f in satellite_file_names
    ]
    index = satellite_date_strings.index(valid_date_string)
    desired_satellite_file_names = [satellite_file_names[index]]

    if lead_time_seconds > 0:
        desired_satellite_file_names.insert(0, satellite_file_names[index - 1])

    print('Reading data from: "{0:s}"...'.format(desired_radar_file_name))
    radar_dict = radar_io.read_2d_file(desired_radar_file_name)

    satellite_dicts = []

    for this_file_name in desired_satellite_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_satellite_dict = satellite_io.read_file(
            netcdf_file_name=this_file_name, read_temperatures=False,
            read_counts=True
        )
        this_satellite_dict = satellite_io.subset_by_band(
            satellite_dict=this_satellite_dict, band_numbers=band_numbers
        )
        satellite_dicts.append(this_satellite_dict)

    satellite_dict = satellite_io.concat_data(satellite_dicts)

    assert numpy.allclose(
        radar_dict[radar_io.LATITUDES_KEY],
        satellite_dict[satellite_io.LATITUDES_KEY],
        atol=TOLERANCE
    )

    assert numpy.allclose(
        radar_dict[radar_io.LONGITUDES_KEY],
        satellite_dict[satellite_io.LONGITUDES_KEY],
        atol=TOLERANCE
    )

    valid_times_unix_sec = radar_dict[radar_io.VALID_TIMES_KEY]
    init_times_unix_sec = valid_times_unix_sec - lead_time_seconds

    good_flags = numpy.array([
        t in satellite_dict[satellite_io.VALID_TIMES_KEY]
        for t in init_times_unix_sec
    ], dtype=bool)

    if not numpy.any(good_flags):
        return None, None

    good_indices = numpy.where(good_flags)[0]
    valid_times_unix_sec = valid_times_unix_sec[good_indices]
    init_times_unix_sec = init_times_unix_sec[good_indices]

    radar_dict = radar_io.subset_by_time(
        radar_dict=radar_dict, desired_times_unix_sec=valid_times_unix_sec
    )[0]
    satellite_dict = satellite_io.subset_by_time(
        satellite_dict=satellite_dict,
        desired_times_unix_sec=init_times_unix_sec
    )[0]
    num_examples = len(good_indices)

    if num_examples >= num_examples_to_read:
        desired_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        )
        desired_indices = numpy.random.choice(
            desired_indices, size=num_examples_to_read, replace=False
        )

        radar_dict = radar_io.subset_by_index(
            radar_dict=radar_dict, desired_indices=desired_indices
        )
        satellite_dict = satellite_io.subset_by_index(
            satellite_dict=satellite_dict, desired_indices=desired_indices
        )

    if norm_dict_for_count is not None:
        satellite_dict = normalization.normalize_data(
            satellite_dict=satellite_dict, uniformize=uniformize,
            norm_dict_for_count=norm_dict_for_count
        )

    predictor_matrix = satellite_dict[satellite_io.BRIGHTNESS_COUNT_KEY]
    target_matrix = (
        radar_dict[radar_io.COMPOSITE_REFL_KEY] >= reflectivity_threshold_dbz
    ).astype(int)

    return predictor_matrix, numpy.expand_dims(target_matrix, axis=-1)


def _write_metafile(
        dill_file_name, num_epochs, num_training_batches_per_epoch,
        training_option_dict, num_validation_batches_per_epoch,
        validation_option_dict, do_early_stopping, plateau_lr_multiplier):
    """Writes metadata to Dill file.

    :param dill_file_name: Path to output file.
    :param num_epochs: See doc for `train_model_with_generator`.
    :param num_training_batches_per_epoch: Same.
    :param training_option_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_option_dict: Same.
    :param do_early_stopping: Same.
    :param plateau_lr_multiplier: Same.
    """

    metadata_dict = {
        NUM_EPOCHS_KEY: num_epochs,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        TRAINING_OPTIONS_KEY: training_option_dict,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        VALIDATION_OPTIONS_KEY: validation_option_dict,
        EARLY_STOPPING_KEY: do_early_stopping,
        PLATEAU_LR_MUTIPLIER_KEY: plateau_lr_multiplier
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=dill_file_name)

    dill_file_handle = open(dill_file_name, 'wb')
    dill.dump(metadata_dict, dill_file_handle)
    dill_file_handle.close()


def create_data(option_dict):
    """Creates data for neural net.

    This method is the same as `data_generator`, except that it returns all the
    data at once, rather than generating batches on the fly.

    :param option_dict: Dictionary with the following keys.
    option_dict['top_satellite_dir_name']: See doc for `data_generator`.
    option_dict['top_radar_dir_name']: Same.
    option_dict['band_numbers']: Same.
    option_dict['lead_time_seconds']: Same.
    option_dict['reflectivity_threshold_dbz']: Same.
    option_dict['valid_date_string']: Valid date (format "yyyymmdd").  Will
        create examples with radar data valid on this day.
    option_dict['norm_dict_for_count']: See doc for `_read_inputs_one_day`.
    option_dict['uniformize']: See doc for `data_generator`.

    :return: predictor_matrix: See doc for `data_generator`.
    :return: target_matrix: Same.
    """

    option_dict = _check_generator_args(option_dict)

    top_satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    top_radar_dir_name = option_dict[RADAR_DIRECTORY_KEY]
    band_numbers = option_dict[BAND_NUMBERS_KEY]
    lead_time_seconds = option_dict[LEAD_TIME_KEY]
    reflectivity_threshold_dbz = option_dict[REFL_THRESHOLD_KEY]
    valid_date_string = option_dict[VALID_DATE_KEY]
    norm_dict_for_count = option_dict[NORMALIZATION_DICT_KEY]
    uniformize = option_dict[UNIFORMIZE_FLAG_KEY]

    if lead_time_seconds == 0:
        first_init_date_string = copy.deepcopy(valid_date_string)
    else:
        valid_date_unix_sec = time_conversion.string_to_unix_sec(
            valid_date_string, DATE_FORMAT
        )
        first_init_date_string = time_conversion.unix_sec_to_string(
            valid_date_unix_sec - DAYS_TO_SECONDS, DATE_FORMAT
        )

    satellite_file_names = satellite_io.find_many_files(
        top_directory_name=top_satellite_dir_name,
        first_date_string=first_init_date_string,
        last_date_string=valid_date_string,
        raise_error_if_any_missing=True
    )

    radar_file_names = radar_io.find_many_files(
        top_directory_name=top_radar_dir_name,
        first_date_string=valid_date_string,
        last_date_string=valid_date_string,
        with_3d=False, raise_error_if_any_missing=True
    )

    predictor_matrix, target_matrix = _read_inputs_one_day(
        valid_date_string=valid_date_string,
        satellite_file_names=satellite_file_names,
        band_numbers=band_numbers,
        norm_dict_for_count=norm_dict_for_count, uniformize=uniformize,
        radar_file_names=radar_file_names,
        lead_time_seconds=lead_time_seconds,
        reflectivity_threshold_dbz=reflectivity_threshold_dbz,
        num_examples_to_read=int(1e6)
    )

    predictor_matrix = predictor_matrix.astype('float32')
    target_matrix = target_matrix.astype('float32')
    return predictor_matrix, target_matrix


def data_generator(option_dict):
    """Generates training data for neural net.

    E = number of examples per batch
    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (spectral bands)

    :param option_dict: Dictionary with the following keys.
    option_dict['top_satellite_dir_name']: Name of top-level directory with
        satellite data (predictors).  Files therein will be found by
        `satellite_io.find_file` and read by `satellite_io.read_file`.
    option_dict['top_radar_dir_name']: Name of top-level directory with radar
        data (targets).  Files therein will be found by `radar_io.find_file` and
        read by `radar_io.read_2d_file`.
    option_dict['num_examples_per_batch']: Batch size.
    option_dict['max_examples_per_day_in_batch']: Max number of examples from
        the same day in one batch.
    option_dict['band_numbers']: 1-D numpy array of band numbers (integers) for
        satellite data.  Will use only these spectral bands as predictors.
    option_dict['lead_time_seconds']: Lead time for prediction.
    option_dict['reflectivity_threshold_dbz']: Reflectivity threshold for
       convection.  Only grid cells with composite (column-max) reflectivity >=
       threshold will be called convective.
    option_dict['first_valid_date_string']: First valid date (format
        "yyyymmdd").  Will not generate examples with radar data before this
        day.
    option_dict['last_valid_date_string']: Last valid date (format "yyyymmdd").
        Will not generate examples with radar data after this day.
    option_dict['normalization_file_name']: File with normalization parameters
        (will be read by `normalization.read_file`).  If you do not want to
        normalize, make this None.
    option_dict['uniformize']: Boolean flag.  If True, will convert satellite
        values to uniform distribution before normal distribution.  If False,
        will go directly to normal distribution.

    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values,
        based on satellite data.
    :return: target_matrix: E-by-M-by-N-by-1 numpy array of target values
        (integers in 0...1, indicating whether or not convection occurs at
        the given lead time).
    """

    # TODO(thunderhoser): Allow downsampling?

    # TODO(thunderhoser): Allow generator to read brightness temperatures
    # instead of counts.

    option_dict = _check_generator_args(option_dict)

    top_satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    top_radar_dir_name = option_dict[RADAR_DIRECTORY_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    max_examples_per_day_in_batch = option_dict[MAX_DAILY_EXAMPLES_KEY]
    band_numbers = option_dict[BAND_NUMBERS_KEY]
    lead_time_seconds = option_dict[LEAD_TIME_KEY]
    reflectivity_threshold_dbz = option_dict[REFL_THRESHOLD_KEY]
    first_valid_date_string = option_dict[FIRST_VALID_DATE_KEY]
    last_valid_date_string = option_dict[LAST_VALID_DATE_KEY]
    normalization_file_name = option_dict[NORMALIZATION_FILE_KEY]
    uniformize = option_dict[UNIFORMIZE_FLAG_KEY]

    if lead_time_seconds == 0:
        first_init_date_string = copy.deepcopy(first_valid_date_string)
    else:
        first_valid_time_unix_sec = time_conversion.string_to_unix_sec(
            first_valid_date_string, DATE_FORMAT
        )
        first_init_date_string = time_conversion.unix_sec_to_string(
            first_valid_time_unix_sec - DAYS_TO_SECONDS, DATE_FORMAT
        )

    if normalization_file_name is None:
        norm_dict_for_count = None
    else:
        print('Reading normalization parameters from: "{0:s}"...'.format(
            normalization_file_name
        ))
        norm_dict_for_count = (
            normalization.read_file(normalization_file_name)[1]
        )

    satellite_file_names = satellite_io.find_many_files(
        top_directory_name=top_satellite_dir_name,
        first_date_string=first_init_date_string,
        last_date_string=last_valid_date_string,
        raise_error_if_any_missing=True
    )

    radar_file_names = radar_io.find_many_files(
        top_directory_name=top_radar_dir_name,
        first_date_string=first_valid_date_string,
        last_date_string=last_valid_date_string,
        with_3d=False, raise_error_if_any_missing=True
    )

    valid_date_strings = time_conversion.get_spc_dates_in_range(
        first_valid_date_string, last_valid_date_string
    )

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

            this_predictor_matrix, this_target_matrix = _read_inputs_one_day(
                valid_date_string=valid_date_strings[date_index],
                satellite_file_names=satellite_file_names,
                band_numbers=band_numbers,
                norm_dict_for_count=norm_dict_for_count, uniformize=uniformize,
                radar_file_names=radar_file_names,
                lead_time_seconds=lead_time_seconds,
                reflectivity_threshold_dbz=reflectivity_threshold_dbz,
                num_examples_to_read=num_examples_to_read
            )

            date_index += 1
            if this_predictor_matrix is None:
                continue

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


def train_model_with_generator(
        model_object, output_dir_name, num_epochs,
        num_training_batches_per_epoch, training_option_dict,
        num_validation_batches_per_epoch, validation_option_dict,
        do_early_stopping=True,
        plateau_lr_multiplier=DEFAULT_LEARNING_RATE_MULTIPLIER):
    """Trains neural net with generator.

    :param model_object: Untrained neural net (instance of `keras.models.Model`
        or `keras.models.Sequential`).
    :param output_dir_name: Path to output directory (model and training history
        will be saved here).
    :param num_epochs: Number of training epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param training_option_dict: See doc for `data_generator`.  This dictionary
        will be used to generate training data.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param validation_option_dict: See doc for `data_generator`.  For validation
        only, the following values will replace corresponding values in
        `training_option_dict`:
    validation_option_dict['top_satellite_dir_name']
    validation_option_dict['top_radar_dir_name']
    validation_option_dict['first_valid_date_string']
    validation_option_dict['last_valid_date_string']

    :param do_early_stopping: Boolean flag.  If True, will stop training early
        if validation loss has not improved over last several epochs (see
        constants at top of file for what exactly this means).
    :param plateau_lr_multiplier: Multiplier for learning rate.  Learning
        rate will be multiplied by this factor upon plateau in validation
        performance.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 2)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 10)
    error_checking.assert_is_integer(num_validation_batches_per_epoch)
    error_checking.assert_is_geq(num_validation_batches_per_epoch, 10)
    error_checking.assert_is_boolean(do_early_stopping)

    if do_early_stopping:
        error_checking.assert_is_greater(plateau_lr_multiplier, 0.)
        error_checking.assert_is_less_than(plateau_lr_multiplier, 1.)

    training_option_dict = _check_generator_args(training_option_dict)

    validation_keys_to_keep = [
        SATELLITE_DIRECTORY_KEY, RADAR_DIRECTORY_KEY,
        FIRST_VALID_DATE_KEY, LAST_VALID_DATE_KEY
    ]

    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

    validation_option_dict = _check_generator_args(validation_option_dict)

    model_file_name = (
        output_dir_name + '/model_epoch={epoch:03d}_val-loss={val_loss:.6f}.h5'
    )

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
        dill_file_name=metafile_name, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        do_early_stopping=do_early_stopping,
        plateau_lr_multiplier=plateau_lr_multiplier
    )

    training_generator = data_generator(training_option_dict)
    validation_generator = data_generator(validation_option_dict)

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
    return keras.models.load_model(
        hdf5_file_name, custom_objects=METRIC_FUNCTION_DICT
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
    metadata_dict['num_epochs']: See doc for `train_model`.
    metadata_dict['num_training_batches_per_epoch']: Same.
    metadata_dict['training_option_dict']: Same.
    metadata_dict['num_validation_batches_per_epoch']: Same.
    metadata_dict['validation_option_dict']: Same.
    metadata_dict['do_early_stopping']: Same.
    metadata_dict['plateau_lr_multiplier']: Same.

    :raises: ValueError: if any expected key is not found in dictionary.
    """

    error_checking.assert_file_exists(dill_file_name)

    dill_file_handle = open(dill_file_name, 'rb')
    metadata_dict = dill.load(dill_file_handle)
    dill_file_handle.close()

    missing_keys = list(set(METADATA_KEYS) - set(metadata_dict.keys()))
    if len(missing_keys) == 0:
        return metadata_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), dill_file_name)

    raise ValueError(error_string)


def apply_model(
        model_object, predictor_matrix, num_examples_per_batch, verbose=False):
    """Applies trained neural net to new data.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix: See output doc for `data_generator`.
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
