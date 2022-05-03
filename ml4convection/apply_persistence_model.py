"""Applies persistence model."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import gg_general_utils
import error_checking
import example_io
import prediction_io
import satellite_io
import neural_net

DUMMY_LOSS_FUNCTION_NAME = neural_net.metric_params_to_name(
    score_name=neural_net.FSS_NAME, half_window_size_px=1
)

TARGET_DIR_ARG_NAME = 'input_target_dir_name'
LEAD_TIME_ARG_NAME = 'lead_time_seconds'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_px'
FIRST_DATE_ARG_NAME = 'first_valid_date_string'
LAST_DATE_ARG_NAME = 'last_valid_date_string'
DUMMY_MODEL_FILE_ARG_NAME = 'output_dummy_model_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

TARGET_DIR_HELP_STRING = (
    'Name of top-level directory with targets.  Files therein will be found '
    'by `example_io.find_target_file` and read by '
    '`example_io.read_target_file`.'
)
LEAD_TIME_HELP_STRING = 'Lead time for predictions.'
SMOOTHING_RADIUS_HELP_STRING = (
    'Radius for Gaussian smoother (number of pixels).  Will be used to turn '
    'labels at forecast (initialization) time into probabilities.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  The model will be applied to valid times '
    '(target times) from the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

DUMMY_MODEL_FILE_HELP_STRING = (
    'Path to dummy file for persistence model.  This file will not actually be '
    'created, but an accompanying metafile will be created.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Predictions will be written by '
    '`prediction_io.write_file`, to exact locations therein determined by '
    '`prediction_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=True,
    help=TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LEAD_TIME_ARG_NAME, type=int, required=True,
    help=LEAD_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=float, required=True,
    help=SMOOTHING_RADIUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DUMMY_MODEL_FILE_ARG_NAME, type=str, required=True,
    help=DUMMY_MODEL_FILE_ARG_NAME
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _write_metafile(target_file_names, lead_time_seconds,
                    dummy_model_file_name):
    """Writes metafile (readable by `neural_net.read_metafile`).

    :param target_file_names: 1-D list of paths to target files.  Will be read
        by `example_io.read_target_file`.
    :param lead_time_seconds: See documentation at top of file.
    :param dummy_model_file_name: Same.
    """

    first_target_dict = example_io.read_target_file(target_file_names[0])
    mask_matrix = first_target_dict[example_io.FULL_MASK_MATRIX_KEY]

    metafile_name = neural_net.find_metafile(
        model_file_name=dummy_model_file_name, raise_error_if_missing=False
    )
    print('Writing metafile to: "{0:s}"...'.format(metafile_name))

    training_option_dict = {
        neural_net.BAND_NUMBERS_KEY: satellite_io.BAND_NUMBERS,
        neural_net.LEAD_TIME_KEY: lead_time_seconds,
        neural_net.LAG_TIMES_KEY: numpy.array([0], dtype=int)
    }

    neural_net._write_metafile(
        dill_file_name=metafile_name,
        use_partial_grids=False, num_epochs=100,
        num_training_batches_per_epoch=100,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=100,
        validation_option_dict=training_option_dict,
        do_early_stopping=True, plateau_lr_multiplier=0.6,
        loss_function_name=DUMMY_LOSS_FUNCTION_NAME,
        metric_names=[DUMMY_LOSS_FUNCTION_NAME],
        mask_matrix=mask_matrix, full_mask_matrix=mask_matrix,
        quantile_levels=None
    )


def _make_predictions_one_day(
        target_file_names, valid_date_string, lead_time_seconds,
        smoothing_radius_px):
    """Makes predictions for one day.

    :param target_file_names: 1-D list of paths to target files.  Will be read
        by `example_io.read_target_file`.
    :param valid_date_string: Valid date (format "yyyymmdd").
    :param lead_time_seconds: See documentation at top of file.
    :param smoothing_radius_px: Same.
    :return: prediction_dict: Dictionary with the following keys.
    prediction_dict['target_matrix']: See doc for `prediction_io.write_file`.
    prediction_dict['forecast_probability_matrix']: Same.
    prediction_dict['valid_times_unix_sec']: Same.
    prediction_dict['latitudes_deg_n']: Same.
    prediction_dict['longitudes_deg_e']: Same.
    """

    target_date_strings = [
        example_io.file_name_to_date(f) for f in target_file_names
    ]
    valid_date_index = target_date_strings.index(valid_date_string)
    init_date_indices = numpy.array([valid_date_index], dtype=int)

    if lead_time_seconds > 0 and valid_date_index != 0:
        init_date_indices = numpy.concatenate((
            init_date_indices - 1, init_date_indices
        ))

    print('Reading data from: "{0:s}"...'.format(
        target_file_names[valid_date_index]
    ))
    valid_target_dict = example_io.read_target_file(
        target_file_names[valid_date_index]
    )

    init_target_dicts = []

    for this_index in init_date_indices:
        print('Reading data from: "{0:s}"...'.format(
            target_file_names[this_index]
        ))
        this_dict = example_io.read_target_file(target_file_names[this_index])
        init_target_dicts.append(this_dict)

    init_target_dict = example_io.concat_target_data(init_target_dicts)

    desired_valid_times_unix_sec = valid_target_dict[example_io.VALID_TIMES_KEY]
    desired_init_times_unix_sec = (
        desired_valid_times_unix_sec - lead_time_seconds
    )
    good_flags = numpy.array([
        t in init_target_dict[example_io.VALID_TIMES_KEY]
        for t in desired_init_times_unix_sec
    ], dtype=bool)

    good_indices = numpy.where(good_flags)[0]
    valid_times_unix_sec = desired_valid_times_unix_sec[good_indices]
    init_times_unix_sec = desired_init_times_unix_sec[good_indices]

    valid_target_dict = example_io.subset_by_time(
        predictor_or_target_dict=valid_target_dict,
        desired_times_unix_sec=valid_times_unix_sec
    )[0]
    init_target_dict = example_io.subset_by_time(
        predictor_or_target_dict=init_target_dict,
        desired_times_unix_sec=init_times_unix_sec
    )[0]

    forecast_prob_matrix = (
        init_target_dict[example_io.TARGET_MATRIX_KEY].astype(float)
    )
    num_examples = forecast_prob_matrix.shape[0]

    if smoothing_radius_px > 0:
        print((
            'Applying Gaussian smoother with e-folding radius = {0:f} pixels...'
        ).format(
            smoothing_radius_px
        ))

        for i in range(num_examples):
            forecast_prob_matrix[i, ...] = (
                gg_general_utils.apply_gaussian_filter(
                    input_matrix=forecast_prob_matrix[i, ...],
                    e_folding_radius_grid_cells=smoothing_radius_px
                )
            )

    forecast_prob_matrix[forecast_prob_matrix < 0.] = 0.
    forecast_prob_matrix[forecast_prob_matrix > 1.] = 1.

    return {
        prediction_io.TARGET_MATRIX_KEY:
            valid_target_dict[example_io.TARGET_MATRIX_KEY],
        prediction_io.PROBABILITY_MATRIX_KEY: forecast_prob_matrix,
        prediction_io.VALID_TIMES_KEY:
            valid_target_dict[example_io.VALID_TIMES_KEY],
        prediction_io.LATITUDES_KEY:
            valid_target_dict[example_io.LATITUDES_KEY],
        prediction_io.LONGITUDES_KEY:
            valid_target_dict[example_io.LONGITUDES_KEY],
    }


def _run(top_target_dir_name, lead_time_seconds, smoothing_radius_px,
         first_valid_date_string, last_valid_date_string, dummy_model_file_name,
         top_output_dir_name):
    """Applies persistence model.

    This is effectively the main method.

    :param top_target_dir_name: See documentation at top of file.
    :param lead_time_seconds: Same.
    :param smoothing_radius_px: Same.
    :param first_valid_date_string: Same.
    :param last_valid_date_string: Same.
    :param dummy_model_file_name: Same.
    :param top_output_dir_name: Same.
    """

    error_checking.assert_is_geq(lead_time_seconds, 0)
    error_checking.assert_is_geq(smoothing_radius_px, 0.)

    target_file_names = example_io.find_many_target_files(
        top_directory_name=top_target_dir_name,
        first_date_string=first_valid_date_string,
        last_date_string=last_valid_date_string,
        radar_number=None, prefer_zipped=False, allow_other_format=True,
        raise_error_if_all_missing=True,
        raise_error_if_any_missing=False
    )

    _write_metafile(
        target_file_names=target_file_names,
        lead_time_seconds=lead_time_seconds,
        dummy_model_file_name=dummy_model_file_name
    )
    print('\n')

    valid_date_strings = [
        example_io.file_name_to_date(f) for f in target_file_names
    ]

    for this_valid_date_string in valid_date_strings:
        this_prediction_dict = _make_predictions_one_day(
            target_file_names=target_file_names,
            valid_date_string=this_valid_date_string,
            lead_time_seconds=lead_time_seconds,
            smoothing_radius_px=smoothing_radius_px
        )

        this_output_file_name = prediction_io.find_file(
            top_directory_name=top_output_dir_name,
            valid_date_string=this_valid_date_string,
            radar_number=None, prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        print('Writing predictions to: "{0:s}"...\n'.format(
            this_output_file_name
        ))
        prediction_io.write_file(
            netcdf_file_name=this_output_file_name,
            target_matrix=
            this_prediction_dict[prediction_io.TARGET_MATRIX_KEY],
            forecast_probability_matrix=
            this_prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY],
            valid_times_unix_sec=
            this_prediction_dict[prediction_io.VALID_TIMES_KEY],
            latitudes_deg_n=this_prediction_dict[prediction_io.LATITUDES_KEY],
            longitudes_deg_e=this_prediction_dict[prediction_io.LONGITUDES_KEY],
            model_file_name=dummy_model_file_name, quantile_levels=None
        )

        prediction_io.compress_file(this_output_file_name)
        os.remove(this_output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        lead_time_seconds=getattr(INPUT_ARG_OBJECT, LEAD_TIME_ARG_NAME),
        smoothing_radius_px=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME
        ),
        first_valid_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_valid_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        dummy_model_file_name=getattr(
            INPUT_ARG_OBJECT, DUMMY_MODEL_FILE_ARG_NAME
        ),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
