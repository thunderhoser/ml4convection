"""Makes Chiu-net template for prelim test."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import example_io
import chiu_architecture
import neural_net
import custom_losses

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = '{0:s}/ml4convection_models/chiu_net_test/template'.format(
    HOME_DIR_NAME
)
TOP_TARGET_DIR_NAME = (
    '{0:s}/ml4convection_project/targets/new_echo_classification/no_tracking/'
    'partial_grids'
).format(HOME_DIR_NAME)

OPTION_DICT = {
    chiu_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([205, 205, 4, 7], dtype=int),
    chiu_architecture.NUM_FC_CONV_LAYERS_KEY: 1,
    chiu_architecture.FC_MODULE_DROPOUT_RATES_KEY: numpy.full(1, 0.),
    chiu_architecture.NUM_LEVELS_KEY: 5,
    chiu_architecture.CONV_LAYER_COUNTS_KEY: numpy.full(6, 2, dtype=int),
    chiu_architecture.CHANNEL_COUNTS_KEY:
        numpy.array([16, 24, 32, 48, 64, 96], dtype=int),
    chiu_architecture.ENCODER_DROPOUT_RATES_KEY: numpy.full(6, 0.),
    chiu_architecture.DECODER_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    chiu_architecture.SKIP_DROPOUT_RATES_KEY: numpy.full(5, 0.)
}


def _run():
    """Makes Chiu-net template for prelim test.

    This is effectively the main method.
    """

    target_file_name = example_io.find_target_file(
        top_directory_name=TOP_TARGET_DIR_NAME, date_string='20160101',
        radar_number=0, prefer_zipped=True, allow_other_format=True,
        raise_error_if_missing=False
    )

    print('Reading mask from: "{0:s}"...'.format(target_file_name))
    mask_matrix = example_io.read_target_file(target_file_name)[
        example_io.MASK_MATRIX_KEY
    ]
    full_mask_matrix = example_io.read_target_file(target_file_name)[
        example_io.FULL_MASK_MATRIX_KEY
    ]

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME
    )

    loss_function = custom_losses.fractions_skill_score(
        half_window_size_px=3, mask_matrix=mask_matrix,
        use_as_loss_function=True
    )
    model_object = chiu_architecture.create_model(
        option_dict=OPTION_DICT, loss_function=loss_function,
        mask_matrix=mask_matrix
    )

    model_file_name = '{0:s}/model_template.h5'.format(OUTPUT_DIR_NAME)
    print('Writing model to: "{0:s}"...'.format(model_file_name))
    model_object.save(
        filepath=model_file_name, overwrite=True, include_optimizer=True
    )

    metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=False
    )
    print('Writing metadata to: "{0:s}"...'.format(metafile_name))

    dummy_option_dict = neural_net.DEFAULT_GENERATOR_OPTION_DICT

    neural_net._write_metafile(
        dill_file_name=metafile_name, use_partial_grids=True,
        num_epochs=100, num_training_batches_per_epoch=100,
        training_option_dict=dummy_option_dict,
        num_validation_batches_per_epoch=100,
        validation_option_dict=dummy_option_dict,
        do_early_stopping=True, plateau_lr_multiplier=0.6,
        class_weights=None, fss_half_window_size_px=3,
        mask_matrix=mask_matrix, full_mask_matrix=full_mask_matrix
    )


if __name__ == '__main__':
    _run()
