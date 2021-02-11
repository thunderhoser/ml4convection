"""Makes U-net templates for Experiment 10."""

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
import u_net_architecture
import neural_net
import custom_losses

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = '{0:s}/ml4convection_models/experiment10/templates'.format(
    HOME_DIR_NAME
)
TOP_TARGET_DIR_NAME = (
    '{0:s}/ml4convection_project/targets/new_echo_classification/no_tracking/'
    'partial_grids'
).format(HOME_DIR_NAME)

FSS_HALF_WINDOW_SIZES_PX = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
L2_WEIGHTS = numpy.logspace(-7, -3, num=17)

DEFAULT_OPTION_DICT = {
    u_net_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([205, 205, 7], dtype=int),
    u_net_architecture.NUM_LEVELS_KEY: 5,
    u_net_architecture.NUM_CONV_LAYERS_KEY: 2,
    u_net_architecture.CONV_LAYER_CHANNEL_COUNTS_KEY:
        numpy.array([16, 24, 32, 48, 64, 96], dtype=int),
    u_net_architecture.UPCONV_LAYER_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    u_net_architecture.SKIP_LAYER_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    u_net_architecture.CONV_LAYER_DROPOUT_RATES_KEY: numpy.full(6, 0.)
}


def _run():
    """Makes U-net templates for Experiment 10.

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

    num_window_sizes = len(FSS_HALF_WINDOW_SIZES_PX)
    num_l2_weights = len(L2_WEIGHTS)

    for i in range(num_window_sizes):
        this_loss_function = custom_losses.fractions_skill_score(
            half_window_size_px=FSS_HALF_WINDOW_SIZES_PX[i],
            mask_matrix=mask_matrix, use_as_loss_function=True
        )

        for j in range(num_l2_weights):
            this_option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
            this_option_dict[u_net_architecture.L2_WEIGHT_KEY] = (
                L2_WEIGHTS[j]
            )

            print(this_option_dict)
            print(SEPARATOR_STRING)

            this_model_object = u_net_architecture.create_model(
                option_dict=this_option_dict,
                loss_function=this_loss_function, mask_matrix=mask_matrix
            )

            this_model_file_name = (
                '{0:s}/model_fss-half-window-size-px={1:d}_'
                'l2-weight={2:.10f}.h5'
            ).format(
                OUTPUT_DIR_NAME, FSS_HALF_WINDOW_SIZES_PX[i], L2_WEIGHTS[j]
            )

            print('Writing model to: "{0:s}"...'.format(
                this_model_file_name
            ))
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
                fss_half_window_size_px=FSS_HALF_WINDOW_SIZES_PX[i],
                mask_matrix=mask_matrix, full_mask_matrix=full_mask_matrix
            )


if __name__ == '__main__':
    _run()