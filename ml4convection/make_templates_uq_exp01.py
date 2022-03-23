"""Makes U-net templates for Experiment 1 with uncertainty quantification."""

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

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = '{0:s}/ml4convection_models/uq_experiment01/templates'.format(
    HOME_DIR_NAME
)

FULL_MASK_FILE_NAME = (
    '{0:s}/ml4convection_project/radar_data/radar_mask_100km_omit-north.nc'
).format(HOME_DIR_NAME)

PARTIAL_MASK_FILE_NAME = (
    '{0:s}/ml4convection_project/radar_data/'
    'radar_mask_100km_omit-north_partial.nc'
).format(HOME_DIR_NAME)

LOSS_FUNCTION_NAME = 'fss_neigh0'
METRIC_NAMES = [
    'fss_neigh0', 'fss_neigh1', 'fss_neigh2', 'fss_neigh3',
    'csi_neigh0', 'csi_neigh1', 'csi_neigh2', 'csi_neigh3',
    'bias_neigh0', 'bias_neigh1', 'bias_neigh2', 'bias_neigh3'
]

TOP_LEVEL_SKIP_DROPOUT_RATES = numpy.linspace(0, 0.5, num=5)
PENULTIMATE_LAYER_DROPOUT_RATES = numpy.linspace(0, 0.5, num=5)
OUTPUT_LAYER_DROPOUT_RATES = numpy.linspace(0, 0.5, num=5)

DEFAULT_OPTION_DICT = {
    u_net_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([205, 205, 21], dtype=int),
    u_net_architecture.NUM_LEVELS_KEY: 5,
    u_net_architecture.CONV_LAYER_COUNTS_KEY: numpy.full(6, 2, dtype=int),
    u_net_architecture.OUTPUT_CHANNEL_COUNTS_KEY:
        numpy.array([16, 24, 32, 48, 64, 96], dtype=int),
    u_net_architecture.CONV_DROPOUT_RATES_KEY: [numpy.full(2, 0.)] * 6,
    u_net_architecture.UPCONV_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    u_net_architecture.SKIP_DROPOUT_RATES_KEY: [numpy.full(2, 0.)] * 5,
    u_net_architecture.SKIP_DROPOUT_MC_FLAGS_KEY:
        [numpy.full(2, 1, dtype=bool)] * 5,
    u_net_architecture.INCLUDE_PENULTIMATE_KEY: True,
    u_net_architecture.PENULTIMATE_DROPOUT_MC_FLAG_KEY: True,
    u_net_architecture.OUTPUT_DROPOUT_MC_FLAG_KEY: True,
    u_net_architecture.L2_WEIGHT_KEY: 10 ** -5.5
}


def _run():
    """Makes U-net templates for Experiment 1 with uncertainty quantification.

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

    loss_function = neural_net.get_metrics(
        metric_names=[LOSS_FUNCTION_NAME],
        mask_matrix=partial_mask_matrix,
        use_as_loss_function=True
    )[0][0]

    for i in range(len(TOP_LEVEL_SKIP_DROPOUT_RATES)):
        for j in range(len(PENULTIMATE_LAYER_DROPOUT_RATES)):
            for k in range(len(OUTPUT_LAYER_DROPOUT_RATES)):
                print(SEPARATOR_STRING)

                this_option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
                this_option_dict[
                    u_net_architecture.SKIP_DROPOUT_RATES_KEY
                ][0] = numpy.full(2, TOP_LEVEL_SKIP_DROPOUT_RATES[i])

                this_option_dict[
                    u_net_architecture.PENULTIMATE_DROPOUT_RATE_KEY
                ] = PENULTIMATE_LAYER_DROPOUT_RATES[j]

                this_option_dict[
                    u_net_architecture.OUTPUT_DROPOUT_RATE_KEY
                ] = OUTPUT_LAYER_DROPOUT_RATES[k]

                this_model_object = u_net_architecture.create_model(
                    option_dict=this_option_dict, loss_function=loss_function,
                    mask_matrix=partial_mask_matrix, metric_names=METRIC_NAMES
                )

                this_model_file_name = (
                    '{0:s}/top-level-skip-dropout={1:.3f}_'
                    'penultimate-layer-dropout={2:.3f}_'
                    'output-layer-dropout={3:.3f}'
                ).format(
                    OUTPUT_DIR_NAME,
                    TOP_LEVEL_SKIP_DROPOUT_RATES[i],
                    PENULTIMATE_LAYER_DROPOUT_RATES[j],
                    OUTPUT_LAYER_DROPOUT_RATES[k]
                )

                file_system_utils.mkdir_recursive_if_necessary(
                    file_name=this_model_file_name
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
                    loss_function_name=LOSS_FUNCTION_NAME,
                    metric_names=METRIC_NAMES,
                    mask_matrix=partial_mask_matrix,
                    full_mask_matrix=full_mask_matrix
                )


if __name__ == '__main__':
    _run()
