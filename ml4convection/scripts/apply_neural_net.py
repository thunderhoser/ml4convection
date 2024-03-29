"""Applies trained neural net in inference mode."""

import os
import glob
import copy
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml4convection.io import prediction_io
from ml4convection.machine_learning import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_EXAMPLES_PER_BATCH = 32

MODEL_FILE_ARG_NAME = 'input_model_file_name'
PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
FULL_GRIDS_ARG_NAME = 'apply_to_full_grids'
OVERLAP_SIZE_ARG_NAME = 'overlap_size_px'
FIRST_DATE_ARG_NAME = 'first_valid_date_string'
LAST_DATE_ARG_NAME = 'last_valid_date_string'
NUM_DROPOUT_ITERS_ARG_NAME = 'num_dropout_iterations'
USE_QUANTILES_ARG_NAME = 'use_quantiles'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
PREDICTOR_DIR_HELP_STRING = (
    'Name of top-level directory with predictors.  Files therein will be found '
    'by `example_io.find_predictor_file` and read by '
    '`example_io.read_predictor_file`.'
)
TARGET_DIR_HELP_STRING = (
    'Name of top-level directory with targets.  Files therein will be found '
    'by `example_io.find_target_file` and read by '
    '`example_io.read_target_file`.'
)
FULL_GRIDS_HELP_STRING = (
    '[used only if neural net was trained on partial grids] Boolean flag.  If '
    '1 (0), will create predictions on full (partial) grids.'
)
OVERLAP_SIZE_HELP_STRING = (
    '[used only if neural net was trained on partial grids and `{0:s} == 0`] '
    'Amount of overlap (in pixels) between adjacent partial grids.'
).format(FULL_GRIDS_ARG_NAME)

DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  The model will be applied to valid times (radar'
    ' times) from the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

NUM_DROPOUT_ITERS_HELP_STRING = (
    'Number of iterations for Monte Carlo dropout.  If you do not want to use '
    'MC dropout, make this argument <= 0.'
)
USE_QUANTILES_HELP_STRING = (
    '[used only if NN does quantile regression] Boolean flag.  If 1, will save '
    'predictions for every quantile.  If 0, will save only mean predictions.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Predictions will be written by '
    '`prediction_io.write_file`, to exact locations therein determined by '
    '`prediction_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTOR_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=True,
    help=TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FULL_GRIDS_ARG_NAME, type=int, required=False, default=1,
    help=FULL_GRIDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OVERLAP_SIZE_ARG_NAME, type=int, required=False, default=-1,
    help=OVERLAP_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_DROPOUT_ITERS_ARG_NAME, type=int, required=False, default=0,
    help=NUM_DROPOUT_ITERS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_QUANTILES_ARG_NAME, type=int, required=False, default=0,
    help=USE_QUANTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _apply_to_full_grid_one_day(
        model_object, base_option_dict, trained_on_partial_grids,
        quantile_levels, use_quantiles, overlap_size_px, valid_date_string,
        model_file_name, top_output_dir_name):
    """Applies trained neural net to full grid for one day.

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param base_option_dict: Dictionary with data-processing options.  See doc
        for `neural_net.create_data`.
    :param trained_on_partial_grids: Boolean flag.  If True (False), model was
        trained on partial (full) grids.
    :param quantile_levels: 1-D numpy array of quantile levels used to train
        model for quantile regression.  If model was not trained for quantile
        regression, this is None.
    :param use_quantiles: See documentation at top of file.
    :param overlap_size_px: Same.
    :param valid_date_string: Valid date (radar date), in format "yyyymmdd").
    :param model_file_name: See documentation at top of file.
    :param top_output_dir_name: Same.
    """

    option_dict = copy.deepcopy(base_option_dict)
    option_dict[neural_net.VALID_DATE_KEY] = valid_date_string

    data_dict = neural_net.create_data_full_grid(
        option_dict=option_dict, return_coords=True
    )
    if data_dict is None:
        return

    if trained_on_partial_grids:
        forecast_probability_matrix = neural_net.apply_model_partial_grids(
            model_object=model_object,
            predictor_matrix=data_dict[neural_net.PREDICTOR_MATRIX_KEY],
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            overlap_size_px=overlap_size_px,
            use_dropout=False, verbose=True
        )
    else:
        forecast_probability_matrix = neural_net.apply_model_full_grid(
            model_object=model_object,
            predictor_matrix=data_dict[neural_net.PREDICTOR_MATRIX_KEY],
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            use_dropout=False, verbose=True
        )

    if quantile_levels is not None and not use_quantiles:
        prediction_dict = {
            prediction_io.PROBABILITY_MATRIX_KEY: forecast_probability_matrix,
            prediction_io.QUANTILE_LEVELS_KEY: quantile_levels
        }
        forecast_probability_matrix = prediction_io.get_mean_predictions(
            prediction_dict
        )
        quantile_levels_to_write = None
    else:
        quantile_levels_to_write = quantile_levels

    these_percentiles = numpy.array(
        [50, 75, 90, 95, 96, 97, 98, 99, 100], dtype=float
    )
    print(numpy.percentile(forecast_probability_matrix, these_percentiles))

    output_file_name = prediction_io.find_file(
        top_directory_name=top_output_dir_name,
        valid_date_string=valid_date_string,
        radar_number=None, prefer_zipped=False, allow_other_format=False,
        raise_error_if_missing=False
    )

    print('Writing predictions to: "{0:s}"...'.format(output_file_name))
    prediction_io.write_file(
        netcdf_file_name=output_file_name,
        target_matrix=data_dict[neural_net.TARGET_MATRIX_KEY][..., 0],
        forecast_probability_matrix=forecast_probability_matrix,
        valid_times_unix_sec=data_dict[neural_net.VALID_TIMES_KEY],
        latitudes_deg_n=data_dict[neural_net.LATITUDES_KEY],
        longitudes_deg_e=data_dict[neural_net.LONGITUDES_KEY],
        model_file_name=model_file_name,
        quantile_levels=quantile_levels_to_write
    )

    prediction_io.compress_file(output_file_name)
    os.remove(output_file_name)


def _apply_to_partial_grids_one_day(
        model_object, base_option_dict, quantile_levels, use_quantiles,
        valid_date_string, model_file_name, num_dropout_iterations,
        top_output_dir_name):
    """Applies trained neural net to partial grids for one day.

    :param model_object: See doc for `_apply_to_full_grid_one_day`.
    :param base_option_dict: Same.
    :param quantile_levels: Same.
    :param use_quantiles: Same.
    :param valid_date_string: Same.
    :param model_file_name: Same.
    :param num_dropout_iterations: See documentation at top of file.
    :param top_output_dir_name: Same.
    """

    option_dict = copy.deepcopy(base_option_dict)
    option_dict[neural_net.VALID_DATE_KEY] = valid_date_string

    data_dicts = neural_net.create_data_partial_grids(
        option_dict=option_dict, return_coords=True
    )
    if data_dicts is None:
        return

    if quantile_levels is not None:
        num_dropout_iterations = 0

    num_radars = len(data_dicts)

    if num_dropout_iterations > 1:
        for k in range(num_radars):
            if len(list(data_dicts[k].keys())) == 0:
                continue

            forecast_probability_matrix = None

            for i in range(num_dropout_iterations):
                this_prob_matrix = neural_net.apply_model_full_grid(
                    model_object=model_object,
                    predictor_matrix=
                    data_dicts[k][neural_net.PREDICTOR_MATRIX_KEY],
                    num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
                    use_dropout=True, verbose=True
                )

                if forecast_probability_matrix is None:
                    forecast_probability_matrix = numpy.full(
                        this_prob_matrix.shape + (num_dropout_iterations,),
                        numpy.nan
                    )

                forecast_probability_matrix[..., i] = this_prob_matrix

            these_percentiles = numpy.array(
                [0, 50, 75, 90, 95, 96, 97, 98, 99, 100], dtype=float
            )
            print(numpy.percentile(
                forecast_probability_matrix, these_percentiles
            ))

            output_file_name = prediction_io.find_file(
                top_directory_name=top_output_dir_name,
                valid_date_string=valid_date_string,
                radar_number=k, prefer_zipped=False, allow_other_format=False,
                raise_error_if_missing=False
            )

            print('Writing predictions to: "{0:s}"...'.format(output_file_name))
            prediction_io.write_file(
                netcdf_file_name=output_file_name,
                target_matrix=
                data_dicts[k][neural_net.TARGET_MATRIX_KEY][..., 0],
                forecast_probability_matrix=forecast_probability_matrix,
                valid_times_unix_sec=data_dicts[k][neural_net.VALID_TIMES_KEY],
                latitudes_deg_n=data_dicts[k][neural_net.LATITUDES_KEY],
                longitudes_deg_e=data_dicts[k][neural_net.LONGITUDES_KEY],
                model_file_name=model_file_name, quantile_levels=quantile_levels
            )

            prediction_io.compress_file(output_file_name)
            os.remove(output_file_name)

        return

    for k in range(num_radars):
        if len(list(data_dicts[k].keys())) == 0:
            continue

        forecast_probability_matrix = neural_net.apply_model_full_grid(
            model_object=model_object,
            predictor_matrix=data_dicts[k][neural_net.PREDICTOR_MATRIX_KEY],
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            use_dropout=num_dropout_iterations > 0, verbose=True
        )

        if quantile_levels is not None and not use_quantiles:
            prediction_dict = {
                prediction_io.PROBABILITY_MATRIX_KEY:
                    forecast_probability_matrix,
                prediction_io.QUANTILE_LEVELS_KEY: quantile_levels
            }
            forecast_probability_matrix = prediction_io.get_mean_predictions(
                prediction_dict
            )
            quantile_levels_to_write = None
        else:
            quantile_levels_to_write = quantile_levels

        these_percentiles = numpy.array(
            [0, 50, 75, 90, 95, 96, 97, 98, 99, 100], dtype=float
        )
        print(numpy.percentile(
            forecast_probability_matrix, these_percentiles
        ))

        output_file_name = prediction_io.find_file(
            top_directory_name=top_output_dir_name,
            valid_date_string=valid_date_string,
            radar_number=k, prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        print('Writing predictions to: "{0:s}"...'.format(output_file_name))
        prediction_io.write_file(
            netcdf_file_name=output_file_name,
            target_matrix=
            data_dicts[k][neural_net.TARGET_MATRIX_KEY][..., 0],
            forecast_probability_matrix=forecast_probability_matrix,
            valid_times_unix_sec=data_dicts[k][neural_net.VALID_TIMES_KEY],
            latitudes_deg_n=data_dicts[k][neural_net.LATITUDES_KEY],
            longitudes_deg_e=data_dicts[k][neural_net.LONGITUDES_KEY],
            model_file_name=model_file_name,
            quantile_levels=quantile_levels_to_write
        )

        prediction_io.compress_file(output_file_name)
        os.remove(output_file_name)


def _run(model_file_name, top_predictor_dir_name, top_target_dir_name,
         apply_to_full_grids, overlap_size_px, first_valid_date_string,
         last_valid_date_string, num_dropout_iterations, use_quantiles,
         top_output_dir_name):
    """Applies trained neural net in inference mode.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_predictor_dir_name: Same.
    :param top_target_dir_name: Same.
    :param apply_to_full_grids: Same.
    :param overlap_size_px: Same.
    :param first_valid_date_string: Same.
    :param last_valid_date_string: Same.
    :param num_dropout_iterations: Same.
    :param use_quantiles: Same.
    :param top_output_dir_name: Same.
    """

    if not os.path.isfile(model_file_name):
        model_file_names = glob.glob(model_file_name)
        pathless_model_file_names = [
            os.path.split(f)[1] for f in model_file_names
        ]
        extensionless_model_file_names = [
            os.path.splitext(f)[0] for f in pathless_model_file_names
        ]
        validation_loss_strings = [
            f.split('_')[-1] for f in extensionless_model_file_names
        ]

        for this_string in validation_loss_strings:
            assert this_string.startswith('val-loss=')

        validation_losses = numpy.array([
            float(s.replace('val-loss=', '')) for s in validation_loss_strings
        ])
        min_index = numpy.nanargmin(validation_losses)
        model_file_name = model_file_names[min_index]

        extensionless_model_file_name = os.path.splitext(model_file_name)[0]
        top_output_dir_name = '{0:s}/{1:s}'.format(
            extensionless_model_file_name, top_output_dir_name
        )

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)
    metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(metafile_name))
    metadata_dict = neural_net.read_metafile(metafile_name)
    trained_on_partial_grids = metadata_dict[neural_net.USE_PARTIAL_GRIDS_KEY]
    apply_to_full_grids = apply_to_full_grids or not trained_on_partial_grids

    training_option_dict = metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    base_option_dict = {
        neural_net.PREDICTOR_DIRECTORY_KEY: top_predictor_dir_name,
        neural_net.TARGET_DIRECTORY_KEY: top_target_dir_name,
        neural_net.BAND_NUMBERS_KEY:
            training_option_dict[neural_net.BAND_NUMBERS_KEY],
        neural_net.LEAD_TIME_KEY:
            training_option_dict[neural_net.LEAD_TIME_KEY],
        neural_net.LAG_TIMES_KEY:
            training_option_dict[neural_net.LAG_TIMES_KEY],
        neural_net.INCLUDE_TIME_DIM_KEY:
            training_option_dict[neural_net.INCLUDE_TIME_DIM_KEY],
        neural_net.OMIT_NORTH_RADAR_KEY:
            training_option_dict[neural_net.OMIT_NORTH_RADAR_KEY],
        neural_net.NORMALIZE_FLAG_KEY:
            training_option_dict[neural_net.NORMALIZE_FLAG_KEY],
        neural_net.UNIFORMIZE_FLAG_KEY:
            training_option_dict[neural_net.UNIFORMIZE_FLAG_KEY],
        neural_net.ADD_COORDS_KEY:
            training_option_dict[neural_net.ADD_COORDS_KEY]
    }

    valid_date_strings = time_conversion.get_spc_dates_in_range(
        first_valid_date_string, last_valid_date_string
    )
    print(SEPARATOR_STRING)

    for i in range(len(valid_date_strings)):
        if apply_to_full_grids:
            _apply_to_full_grid_one_day(
                model_object=model_object, base_option_dict=base_option_dict,
                quantile_levels=metadata_dict[neural_net.QUANTILE_LEVELS_KEY],
                use_quantiles=use_quantiles,
                trained_on_partial_grids=trained_on_partial_grids,
                overlap_size_px=overlap_size_px,
                valid_date_string=valid_date_strings[i],
                model_file_name=model_file_name,
                top_output_dir_name=top_output_dir_name
            )
        else:
            _apply_to_partial_grids_one_day(
                model_object=model_object, base_option_dict=base_option_dict,
                quantile_levels=metadata_dict[neural_net.QUANTILE_LEVELS_KEY],
                use_quantiles=use_quantiles,
                valid_date_string=valid_date_strings[i],
                model_file_name=model_file_name,
                num_dropout_iterations=num_dropout_iterations,
                top_output_dir_name=top_output_dir_name
            )

        if i != len(valid_date_strings) - 1:
            print('\n')


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        top_predictor_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_DIR_ARG_NAME
        ),
        top_target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        apply_to_full_grids=bool(
            getattr(INPUT_ARG_OBJECT, FULL_GRIDS_ARG_NAME)
        ),
        overlap_size_px=getattr(INPUT_ARG_OBJECT, OVERLAP_SIZE_ARG_NAME),
        first_valid_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_valid_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        num_dropout_iterations=getattr(
            INPUT_ARG_OBJECT, NUM_DROPOUT_ITERS_ARG_NAME
        ),
        use_quantiles=bool(getattr(INPUT_ARG_OBJECT, USE_QUANTILES_ARG_NAME)),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
