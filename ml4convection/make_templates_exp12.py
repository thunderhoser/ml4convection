"""Makes U-net templates for Experiment 12."""

import sys
import copy
import os.path
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
OUTPUT_DIR_NAME = '{0:s}/ml4convection_models/experiment11/templates'.format(
    HOME_DIR_NAME
)
TOP_TARGET_DIR_NAME = (
    '{0:s}/ml4convection_project/targets/new_echo_classification/no_tracking/'
    'partial_grids'
).format(HOME_DIR_NAME)

FSS_HALF_WINDOW_SIZE_PX = 4

L2_WEIGHTS = numpy.logspace(-7, -3, num=7)
LAG_TIME_COUNTS = numpy.array([2, 3, 4, 5], dtype=int)

DEFAULT_OPTION_DICT = {
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
    """Makes U-net templates for Experiment 12.

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

    loss_function = custom_losses.fractions_skill_score(
        half_window_size_px=FSS_HALF_WINDOW_SIZE_PX, mask_matrix=mask_matrix,
        use_as_loss_function=True
    )

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME
    )

    num_l2_weights = len(L2_WEIGHTS)
    num_lag_time_counts = len(LAG_TIME_COUNTS)

    for i in range(num_l2_weights):
        for j in range(num_lag_time_counts):
            this_option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
            this_option_dict[chiu_architecture.L2_WEIGHT_KEY] = L2_WEIGHTS[i]
            this_option_dict[chiu_architecture.INPUT_DIMENSIONS_KEY] = (
                numpy.array([205, 205, LAG_TIME_COUNTS[j], 7], dtype=int)
            )

            print(this_option_dict)
            print(SEPARATOR_STRING)

            this_model_object = chiu_architecture.create_model(
                option_dict=this_option_dict, loss_function=loss_function,
                mask_matrix=mask_matrix
            )

            this_model_file_name = (
                '{0:s}/model_l2-weight={1:.10f}_num-lag-times={2:d}.h5'
            ).format(
                OUTPUT_DIR_NAME, L2_WEIGHTS[i], LAG_TIME_COUNTS[j]
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
                mask_matrix=mask_matrix, full_mask_matrix=full_mask_matrix
            )


if __name__ == '__main__':
    _run()
