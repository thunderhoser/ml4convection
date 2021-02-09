"""Trains neural net."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import neural_net
import training_args

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
NONE_STRINGS = ['', 'none', 'None']

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = training_args.add_input_args(parser_object=INPUT_ARG_PARSER)


def _run(training_predictor_dir_name, training_target_dir_name,
         validn_predictor_dir_name, validn_target_dir_name,
         input_model_file_name, output_model_dir_name, band_numbers,
         lead_time_seconds, lag_times_seconds, include_time_dimension,
         first_training_date_string, last_training_date_string,
         first_validn_date_string, last_validn_date_string,
         normalize, uniformize, add_coords,
         num_examples_per_batch, max_examples_per_day_in_batch,
         use_partial_grids, omit_north_radar, num_epochs,
         num_training_batches_per_epoch, num_validn_batches_per_epoch,
         plateau_lr_multiplier):
    """Trains neural net.

    This is effectively the main method.

    :param training_predictor_dir_name: See documentation at top of
        training_args.py.
    :param training_target_dir_name: Same.
    :param validn_predictor_dir_name: Same.
    :param validn_target_dir_name: Same.
    :param input_model_file_name: Same.
    :param output_model_dir_name: Same.
    :param band_numbers: Same.
    :param lead_time_seconds: Same.
    :param lag_times_seconds: Same.
    :param include_time_dimension: Same.
    :param first_training_date_string: Same.
    :param last_training_date_string: Same.
    :param first_validn_date_string: Same.
    :param last_validn_date_string: Same.
    :param normalize: Same.
    :param add_coords: Same.
    :param uniformize: Same.
    :param num_examples_per_batch: Same.
    :param max_examples_per_day_in_batch: Same.
    :param use_partial_grids: Same.
    :param omit_north_radar: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validn_batches_per_epoch: Same.
    :param plateau_lr_multiplier: Same.
    """

    training_option_dict = {
        neural_net.PREDICTOR_DIRECTORY_KEY: training_predictor_dir_name,
        neural_net.TARGET_DIRECTORY_KEY: training_target_dir_name,
        neural_net.BATCH_SIZE_KEY: num_examples_per_batch,
        neural_net.MAX_DAILY_EXAMPLES_KEY: max_examples_per_day_in_batch,
        neural_net.BAND_NUMBERS_KEY: band_numbers,
        neural_net.LEAD_TIME_KEY: lead_time_seconds,
        neural_net.LAG_TIMES_KEY: lag_times_seconds,
        neural_net.INCLUDE_TIME_DIM_KEY: include_time_dimension,
        neural_net.OMIT_NORTH_RADAR_KEY: omit_north_radar,
        neural_net.FIRST_VALID_DATE_KEY: first_training_date_string,
        neural_net.LAST_VALID_DATE_KEY: last_training_date_string,
        neural_net.NORMALIZE_FLAG_KEY: normalize,
        neural_net.UNIFORMIZE_FLAG_KEY: uniformize,
        neural_net.ADD_COORDS_KEY: add_coords
    }

    validation_option_dict = {
        neural_net.PREDICTOR_DIRECTORY_KEY: validn_predictor_dir_name,
        neural_net.TARGET_DIRECTORY_KEY: validn_target_dir_name,
        neural_net.FIRST_VALID_DATE_KEY: first_validn_date_string,
        neural_net.LAST_VALID_DATE_KEY: last_validn_date_string
    }

    print('Reading untrained model from: "{0:s}"...'.format(
        input_model_file_name
    ))
    model_object = neural_net.read_model(
        hdf5_file_name=input_model_file_name, for_mirrored_training=True
    )
    input_metafile_name = neural_net.find_metafile(
        model_file_name=input_model_file_name
    )

    print('Reading metadata from: "{0:s}"...'.format(input_metafile_name))
    metadata_dict = neural_net.read_metafile(input_metafile_name)

    print(SEPARATOR_STRING)

    neural_net.train_model(
        model_object=model_object, output_dir_name=output_model_dir_name,
        use_partial_grids=use_partial_grids, num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validn_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        mask_matrix=metadata_dict[neural_net.MASK_MATRIX_KEY],
        full_mask_matrix=metadata_dict[neural_net.FULL_MASK_MATRIX_KEY],
        do_early_stopping=True, plateau_lr_multiplier=plateau_lr_multiplier,
        class_weights=metadata_dict[neural_net.CLASS_WEIGHTS_KEY],
        fss_half_window_size_px=
        metadata_dict[neural_net.FSS_HALF_WINDOW_SIZE_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        training_predictor_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.TRAINING_PREDICTOR_DIR_ARG_NAME
        ),
        training_target_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.TRAINING_TARGET_DIR_ARG_NAME
        ),
        validn_predictor_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.VALIDN_PREDICTOR_DIR_ARG_NAME
        ),
        validn_target_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.VALIDN_TARGET_DIR_ARG_NAME
        ),
        input_model_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.INPUT_MODEL_FILE_ARG_NAME
        ),
        output_model_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.OUTPUT_MODEL_DIR_ARG_NAME
        ),
        band_numbers=numpy.array(
            getattr(INPUT_ARG_OBJECT, training_args.BAND_NUMBERS_ARG_NAME),
            dtype=int
        ),
        lead_time_seconds=getattr(
            INPUT_ARG_OBJECT, training_args.LEAD_TIME_ARG_NAME
        ),
        lag_times_seconds=numpy.array(
            getattr(INPUT_ARG_OBJECT, training_args.LAG_TIMES_ARG_NAME),
            dtype=int
        ),
        include_time_dimension=bool(getattr(
            INPUT_ARG_OBJECT, training_args.INCLUDE_TIME_DIM_ARG_NAME
        )),
        first_training_date_string=getattr(
            INPUT_ARG_OBJECT, training_args.FIRST_TRAIN_DATE_ARG_NAME
        ),
        last_training_date_string=getattr(
            INPUT_ARG_OBJECT, training_args.LAST_TRAIN_DATE_ARG_NAME
        ),
        first_validn_date_string=getattr(
            INPUT_ARG_OBJECT, training_args.FIRST_VALIDN_DATE_ARG_NAME
        ),
        last_validn_date_string=getattr(
            INPUT_ARG_OBJECT, training_args.LAST_VALIDN_DATE_ARG_NAME
        ),
        normalize=bool(getattr(
            INPUT_ARG_OBJECT, training_args.NORMALIZE_ARG_NAME
        )),
        uniformize=bool(getattr(
            INPUT_ARG_OBJECT, training_args.UNIFORMIZE_ARG_NAME
        )),
        add_coords=bool(getattr(
            INPUT_ARG_OBJECT, training_args.ADD_COORDS_ARG_NAME
        )),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, training_args.BATCH_SIZE_ARG_NAME
        ),
        max_examples_per_day_in_batch=getattr(
            INPUT_ARG_OBJECT, training_args.MAX_DAILY_EXAMPLES_ARG_NAME
        ),
        use_partial_grids=bool(getattr(
            INPUT_ARG_OBJECT, training_args.USE_PARTIAL_GRIDS_ARG_NAME
        )),
        omit_north_radar=bool(getattr(
            INPUT_ARG_OBJECT, training_args.OMIT_NORTH_RADAR_ARG_NAME
        )),
        num_epochs=getattr(INPUT_ARG_OBJECT, training_args.NUM_EPOCHS_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_TRAINING_BATCHES_ARG_NAME
        ),
        num_validn_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_VALIDN_BATCHES_ARG_NAME
        ),
        plateau_lr_multiplier=getattr(
            INPUT_ARG_OBJECT, training_args.PLATEAU_LR_MULTIPLIER_ARG_NAME
        )
    )
