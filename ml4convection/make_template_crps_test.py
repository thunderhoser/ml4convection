"""Makes U-net template for CRPS test."""

import os
import sys
import copy
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import radar_io
import u_net_architecture
import neural_net

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = '{0:s}/ml4convection_models/crps_test/template'.format(
    HOME_DIR_NAME
)

FULL_MASK_FILE_NAME = (
    '{0:s}/ml4convection_project/radar_data/radar_mask_100km_omit-north.nc'
).format(HOME_DIR_NAME)

PARTIAL_MASK_FILE_NAME = (
    '{0:s}/ml4convection_project/radar_data/'
    'radar_mask_100km_omit-north_partial.nc'
).format(HOME_DIR_NAME)

LOSS_FUNCTION_NAME = 'crps_neigh0'
TOP_LEVEL_SKIP_DROPOUT_RATE = 0.5

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
        [numpy.full(2, 0, dtype=bool)] * 5,
    u_net_architecture.INCLUDE_PENULTIMATE_KEY: False,
    u_net_architecture.PENULTIMATE_DROPOUT_RATE_KEY: 0.5,
    u_net_architecture.PENULTIMATE_DROPOUT_MC_FLAG_KEY: False,
    u_net_architecture.OUTPUT_DROPOUT_RATE_KEY: 0.,
    u_net_architecture.OUTPUT_DROPOUT_MC_FLAG_KEY: False,
    u_net_architecture.L2_WEIGHT_KEY: 10 ** -5.5
}


def _run():
    """Makes U-net template for CRPS test.

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

    option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
    option_dict[u_net_architecture.SKIP_DROPOUT_RATES_KEY][0] = numpy.full(
        2, TOP_LEVEL_SKIP_DROPOUT_RATE
    )

    loss_function = neural_net.get_metrics(
        metric_names=[LOSS_FUNCTION_NAME],
        mask_matrix=partial_mask_matrix,
        use_as_loss_function=True
    )[0][0]

    model_object = u_net_architecture.create_crps_model(
        option_dict=option_dict, crps_loss_function=loss_function,
        mask_matrix=partial_mask_matrix, num_estimates=100
    )

    model_file_name = '{0:s}/model.h5'.format(OUTPUT_DIR_NAME)
    file_system_utils.mkdir_recursive_if_necessary(file_name=model_file_name)

    print('Writing model to: "{0:s}"...'.format(model_file_name))
    model_object.save(
        filepath=model_file_name, overwrite=True, include_optimizer=True
    )

    metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=False
    )
    dummy_option_dict = neural_net.DEFAULT_GENERATOR_OPTION_DICT

    print('Writing metadata to: "{0:s}"...'.format(metafile_name))
    neural_net._write_metafile(
        dill_file_name=metafile_name, use_partial_grids=True,
        num_epochs=100, num_training_batches_per_epoch=100,
        training_option_dict=dummy_option_dict,
        num_validation_batches_per_epoch=100,
        validation_option_dict=dummy_option_dict,
        do_early_stopping=True, plateau_lr_multiplier=0.6,
        loss_function_name=LOSS_FUNCTION_NAME, metric_names=[],
        mask_matrix=partial_mask_matrix,
        full_mask_matrix=full_mask_matrix,
        quantile_levels=None, qfss_half_window_size_px=None
    )


if __name__ == '__main__':
    _run()
