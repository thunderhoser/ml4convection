"""Makes template for Fourier test."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import radar_io
import u_net_architecture
import neural_net
import fourier_utils
import fourier_metrics

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = '{0:s}/ml4convection_models/fourier/templates'.format(
    HOME_DIR_NAME
)

FULL_MASK_FILE_NAME = (
    '{0:s}/ml4convection_project/radar_data/radar_mask_100km_omit-north.nc'
).format(HOME_DIR_NAME)

PARTIAL_MASK_FILE_NAME = (
    '{0:s}/ml4convection_project/radar_data/'
    'radar_mask_100km_omit-north_partial.nc'
).format(HOME_DIR_NAME)

DEFAULT_OPTION_DICT = {
    u_net_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([205, 205, 7], dtype=int),
    u_net_architecture.NUM_LEVELS_KEY: 5,
    u_net_architecture.NUM_CONV_LAYERS_KEY: 2,
    u_net_architecture.CONV_LAYER_CHANNEL_COUNTS_KEY:
        numpy.array([16, 24, 32, 48, 64, 96], dtype=int),
    u_net_architecture.CONV_LAYER_DROPOUT_RATES_KEY: numpy.full(6, 0.),
    u_net_architecture.UPCONV_LAYER_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    u_net_architecture.SKIP_LAYER_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    u_net_architecture.L2_WEIGHT_KEY: 1e-7,
}


def _run():
    """Makes U-net template for Fourier test.

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

    spatial_coeff_matrix = fourier_utils.apply_blackman_window(
        numpy.full((615, 615), 1.)
    )
    frequency_coeff_matrix = fourier_utils.apply_butterworth_filter(
        coefficient_matrix=numpy.full((615, 615), 1.),
        filter_order=2., grid_spacing_metres=1250.,
        min_resolution_metres=5000., max_resolution_metres=numpy.inf
    )
    loss_function = fourier_metrics.mean_squared_error(
        spatial_coeff_matrix=spatial_coeff_matrix,
        frequency_coeff_matrix=frequency_coeff_matrix,
        mask_matrix=partial_mask_matrix,
    )

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME
    )

    this_model_object = u_net_architecture.create_model(
        option_dict=DEFAULT_OPTION_DICT, loss_function=loss_function,
        mask_matrix=partial_mask_matrix
    )

    this_model_file_name = '{0:s}/model.h5'.format(OUTPUT_DIR_NAME)
    print('Writing model to: "{0:s}"...'.format(this_model_file_name))
    this_model_object.save(
        filepath=this_model_file_name, overwrite=True, include_optimizer=True
    )

    this_metafile_name = neural_net.find_metafile(
        model_file_name=this_model_file_name, raise_error_if_missing=False
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
        class_weights=None, fss_half_window_size_px=None,
        fourier_spatial_coeff_matrix=spatial_coeff_matrix,
        fourier_freq_coeff_matrix=frequency_coeff_matrix,
        mask_matrix=partial_mask_matrix, full_mask_matrix=full_mask_matrix
    )


if __name__ == '__main__':
    _run()
