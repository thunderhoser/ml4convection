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


def _run(training_satellite_dir_name, training_radar_dir_name,
         validn_satellite_dir_name, validn_radar_dir_name,
         input_model_file_name, output_model_dir_name, band_numbers,
         reflectivity_threshold_dbz, lead_time_seconds,
         first_training_date_string, last_training_date_string,
         first_validn_date_string, last_validn_date_string,
         normalization_file_name, uniformize, num_examples_per_batch,
         max_examples_per_day_in_batch, num_epochs,
         num_training_batches_per_epoch, num_validn_batches_per_epoch,
         plateau_lr_multiplier):
    """Trains neural net.

    This is effectively the main method.

    :param training_satellite_dir_name: See documentation at top of
        training_args.py.
    :param training_radar_dir_name: Same.
    :param validn_satellite_dir_name: Same.
    :param validn_radar_dir_name: Same.
    :param input_model_file_name: Same.
    :param output_model_dir_name: Same.
    :param band_numbers: Same.
    :param reflectivity_threshold_dbz: Same.
    :param lead_time_seconds: Same.
    :param first_training_date_string: Same.
    :param last_training_date_string: Same.
    :param first_validn_date_string: Same.
    :param last_validn_date_string: Same.
    :param normalization_file_name: Same.
    :param uniformize: Same.
    :param num_examples_per_batch: Same.
    :param max_examples_per_day_in_batch: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validn_batches_per_epoch: Same.
    :param plateau_lr_multiplier: Same.
    """

    if normalization_file_name in NONE_STRINGS:
        normalization_file_name = None

    training_option_dict = {
        neural_net.SATELLITE_DIRECTORY_KEY: training_satellite_dir_name,
        neural_net.RADAR_DIRECTORY_KEY: training_radar_dir_name,
        neural_net.BATCH_SIZE_KEY: num_examples_per_batch,
        neural_net.MAX_DAILY_EXAMPLES_KEY: max_examples_per_day_in_batch,
        neural_net.BAND_NUMBERS_KEY: band_numbers,
        neural_net.LEAD_TIME_KEY: lead_time_seconds,
        neural_net.REFL_THRESHOLD_KEY: reflectivity_threshold_dbz,
        neural_net.FIRST_VALID_DATE_KEY: first_training_date_string,
        neural_net.LAST_VALID_DATE_KEY: last_training_date_string,
        neural_net.NORMALIZATION_FILE_KEY: normalization_file_name,
        neural_net.UNIFORMIZE_FLAG_KEY: uniformize
    }

    validation_option_dict = {
        neural_net.SATELLITE_DIRECTORY_KEY: validn_satellite_dir_name,
        neural_net.RADAR_DIRECTORY_KEY: validn_radar_dir_name,
        neural_net.FIRST_VALID_DATE_KEY: first_validn_date_string,
        neural_net.LAST_VALID_DATE_KEY: last_validn_date_string
    }

    print('Reading untrained model from: "{0:s}"...'.format(
        input_model_file_name
    ))
    model_object = neural_net.read_model(input_model_file_name)
    print(SEPARATOR_STRING)

    neural_net.train_model_with_generator(
        model_object=model_object, output_dir_name=output_model_dir_name,
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validn_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        do_early_stopping=True, plateau_lr_multiplier=plateau_lr_multiplier
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        training_satellite_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.TRAINING_SATELLITE_DIR_ARG_NAME
        ),
        training_radar_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.TRAINING_RADAR_DIR_ARG_NAME
        ),
        validn_satellite_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.VALIDN_SATELLITE_DIR_ARG_NAME
        ),
        validn_radar_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.VALIDN_RADAR_DIR_ARG_NAME
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
        reflectivity_threshold_dbz=getattr(
            INPUT_ARG_OBJECT, training_args.REFL_THRESHOLD_ARG_NAME
        ),
        lead_time_seconds=getattr(
            INPUT_ARG_OBJECT, training_args.LEAD_TIME_ARG_NAME
        ),
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
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.NORMALIZATION_FILE_ARG_NAME
        ),
        uniformize=bool(getattr(
            INPUT_ARG_OBJECT, training_args.UNIFORMIZE_ARG_NAME
        )),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, training_args.BATCH_SIZE_ARG_NAME
        ),
        max_examples_per_day_in_batch=getattr(
            INPUT_ARG_OBJECT, training_args.MAX_DAILY_EXAMPLES_ARG_NAME
        ),
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
