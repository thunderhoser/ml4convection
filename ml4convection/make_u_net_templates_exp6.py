"""Makes U-net templates for Experiment 6."""

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
import u_net_architecture
import neural_net
import custom_losses

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = '{0:s}/ml4convection_models/experiment06/templates'.format(
    HOME_DIR_NAME
)
MASK_FILE_NAME = '{0:s}/ml4convection_project/radar_data/radar_mask.nc'.format(
    HOME_DIR_NAME
)

LAG_TIME_COUNTS = numpy.array([1, 3, 4, 5, 7, 10, 13], dtype=int)
L2_WEIGHTS = numpy.logspace(-7, -3, num=17)

DEFAULT_OPTION_DICT = {
    # u_net_architecture.INPUT_DIMENSIONS_KEY:
    #     numpy.array([220, 230, 7], dtype=int),
    u_net_architecture.NUM_LEVELS_KEY: 5,
    u_net_architecture.NUM_CONV_LAYERS_KEY: 2,
    u_net_architecture.CONV_LAYER_CHANNEL_COUNTS_KEY:
        numpy.array([16, 24, 32, 48, 64, 96], dtype=int),
    u_net_architecture.CONV_LAYER_DROPOUT_RATES_KEY: numpy.full(6, 0.),
    u_net_architecture.UPCONV_LAYER_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    u_net_architecture.SKIP_LAYER_DROPOUT_RATES_KEY: numpy.full(5, 0.)
}

LOSS_FUNCTION = custom_losses.fractions_skill_score(
    half_window_size_px=1, use_as_loss_function=True
)


def _run():
    """Makes U-net templates for Experiment 5.

    This is effectively the main method.
    """

    print('Reading mask from: "{0:s}"...'.format(MASK_FILE_NAME))
    mask_dict = radar_io.read_mask_file(MASK_FILE_NAME)
    mask_dict = radar_io.expand_to_satellite_grid(any_radar_dict=mask_dict)
    mask_dict = radar_io.downsample_in_space(
        any_radar_dict=mask_dict, downsampling_factor=4
    )
    mask_matrix = mask_dict[radar_io.MASK_MATRIX_KEY]

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME
    )

    num_lag_time_counts = len(LAG_TIME_COUNTS)
    num_l2_weights = len(L2_WEIGHTS)

    for i in range(num_lag_time_counts):
        for j in range(num_l2_weights):
            this_option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
            this_option_dict[u_net_architecture.L2_WEIGHT_KEY] = L2_WEIGHTS[j]

            this_option_dict[u_net_architecture.INPUT_DIMENSIONS_KEY] = (
                numpy.array([220, 230, 7 * LAG_TIME_COUNTS[i]], dtype=int)
            )

            print(this_option_dict)
            print(SEPARATOR_STRING)

            this_model_object = u_net_architecture.create_model(
                option_dict=this_option_dict, loss_function=LOSS_FUNCTION,
                mask_matrix=mask_matrix
            )

            this_model_file_name = (
                '{0:s}/model_num-lag-times={1:02d}_l2-weight={2:.10f}.h5'
            ).format(
                OUTPUT_DIR_NAME, LAG_TIME_COUNTS[i], L2_WEIGHTS[j]
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
                dill_file_name=this_metafile_name, num_epochs=100,
                num_training_batches_per_epoch=100,
                training_option_dict=dummy_option_dict,
                num_validation_batches_per_epoch=100,
                validation_option_dict=dummy_option_dict,
                do_early_stopping=True, plateau_lr_multiplier=0.6,
                class_weights=None, fss_half_window_size_px=1,
                mask_matrix=mask_matrix
            )


if __name__ == '__main__':
    _run()
