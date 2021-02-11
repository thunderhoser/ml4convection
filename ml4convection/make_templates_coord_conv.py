"""Makes templates for little coord-conv experiment."""

import sys
import copy
import os.path
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import radar_io
import chiu_architecture
import neural_net
import custom_losses

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = (
    '{0:s}/ml4convection_models/coord_conv_experiment/templates'
).format(HOME_DIR_NAME)

FULL_MASK_FILE_NAME = (
    '{0:s}/ml4convection_project/radar_data/radar_mask_100km_omit-north.nc'
).format(HOME_DIR_NAME)

PARTIAL_MASK_FILE_NAME = (
    '{0:s}/ml4convection_project/radar_data/'
    'radar_mask_100km_omit-north_partial.nc'
).format(HOME_DIR_NAME)

FSS_HALF_WINDOW_SIZE_PX = 4
NUM_LAG_TIMES = 7

COORD_CONV_FLAGS = numpy.array([0, 1, 1], dtype=bool)
INCLUDE_LATLONG_FLAGS = numpy.array([1, 0, 1], dtype=bool)

DEFAULT_OPTION_DICT = {
    chiu_architecture.NUM_FC_CONV_LAYERS_KEY: 1,
    chiu_architecture.FC_MODULE_DROPOUT_RATES_KEY: numpy.full(1, 0.),
    chiu_architecture.NUM_LEVELS_KEY: 5,
    chiu_architecture.CONV_LAYER_COUNTS_KEY: numpy.full(6, 2, dtype=int),
    chiu_architecture.CHANNEL_COUNTS_KEY:
        numpy.array([16, 24, 32, 48, 64, 96], dtype=int),
    chiu_architecture.ENCODER_DROPOUT_RATES_KEY: numpy.full(6, 0.),
    chiu_architecture.DECODER_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    chiu_architecture.SKIP_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    chiu_architecture.L2_WEIGHT_KEY: 1e-7
}


def _run():
    """Makes templates for little coord-conv experiment.

    This is effectively the main method.
    """

    print('Reading full mask from: "{0:s}"...'.format(FULL_MASK_FILE_NAME))
    full_mask_matrix = radar_io.read_mask_file(FULL_MASK_FILE_NAME)[
        radar_io.MASK_MATRIX_KEY
    ]

    print('Reading partial mask from: "{0:s}"...'.format(
        PARTIAL_MASK_FILE_NAME
    ))
    partial_mask_matrix = radar_io.read_mask_file(PARTIAL_MASK_FILE_NAME)[
        radar_io.MASK_MATRIX_KEY
    ]

    loss_function = custom_losses.fractions_skill_score(
        half_window_size_px=FSS_HALF_WINDOW_SIZE_PX,
        mask_matrix=partial_mask_matrix, use_as_loss_function=True
    )

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME
    )

    num_trials = len(COORD_CONV_FLAGS)

    for i in range(num_trials):
        this_option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
        this_num_channels = 7 + int(INCLUDE_LATLONG_FLAGS[i]) * 2

        this_option_dict[chiu_architecture.USE_COORD_CONV_KEY] = (
            COORD_CONV_FLAGS[i]
        )
        this_option_dict[chiu_architecture.INPUT_DIMENSIONS_KEY] = numpy.array(
            [205, 205, NUM_LAG_TIMES, this_num_channels], dtype=int
        )

        print(this_option_dict)
        print(SEPARATOR_STRING)

        this_model_object = chiu_architecture.create_model(
            option_dict=this_option_dict, loss_function=loss_function,
            mask_matrix=partial_mask_matrix
        )

        this_model_file_name = (
            '{0:s}/model_coord-conv={1:d}_include-lat-long={2:d}.h5'
        ).format(
            OUTPUT_DIR_NAME, int(COORD_CONV_FLAGS[i]),
            int(INCLUDE_LATLONG_FLAGS[i])
        )

        print('Writing model to: "{0:s}"...'.format(this_model_file_name))
        this_model_object.save(
            filepath=this_model_file_name, overwrite=True,
            include_optimizer=True
        )

        this_metafile_name = neural_net.find_metafile(
            model_file_name=this_model_file_name,
            raise_error_if_missing=False
        )
        dummy_option_dict = neural_net.DEFAULT_GENERATOR_OPTION_DICT

        print('Writing metadata to: "{0:s}"...'.format(
            this_metafile_name
        ))
        neural_net._write_metafile(
            dill_file_name=this_metafile_name, use_partial_grids=True,
            num_epochs=100, num_training_batches_per_epoch=100,
            training_option_dict=dummy_option_dict,
            num_validation_batches_per_epoch=100,
            validation_option_dict=dummy_option_dict,
            do_early_stopping=True, plateau_lr_multiplier=0.6,
            class_weights=None,
            fss_half_window_size_px=FSS_HALF_WINDOW_SIZE_PX,
            mask_matrix=partial_mask_matrix,
            full_mask_matrix=full_mask_matrix
        )


if __name__ == '__main__':
    _run()
