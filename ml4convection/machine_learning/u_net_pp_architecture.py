"""Methods for building U-net++.

Based on: https://github.com/longuyen97/UnetPlusPlus/blob/master/unetpp.py
"""

import numpy
import keras
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from ml4convection.machine_learning import neural_net
from ml4convection.machine_learning import coord_conv

INPUT_DIMENSIONS_KEY = 'input_dimensions'
NUM_LEVELS_KEY = 'num_levels'
CONV_LAYER_COUNTS_KEY = 'num_conv_layers_by_level'
CONV_LAYER_CHANNEL_COUNTS_KEY = 'conv_layer_channel_counts'
ENCODER_DROPOUT_RATES_KEY = 'encoder_dropout_rate_by_level'
UPCONV_DROPOUT_RATES_KEY = 'upconv_dropout_rate_by_level'
SKIP_DROPOUT_RATES_KEY = 'skip_dropout_rate_by_level'
INCLUDE_PENULTIMATE_KEY = 'include_penultimate_conv'
PENULTIMATE_DROPOUT_RATE_KEY = 'penultimate_conv_dropout_rate'
INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'
OUTPUT_ACTIV_FUNCTION_KEY = 'output_activ_function_name'
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = 'output_activ_function_alpha'
L1_WEIGHT_KEY = 'l1_weight'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'
USE_COORD_CONV_KEY = 'use_coord_conv'

DEFAULT_ARCHITECTURE_OPTION_DICT = {
    NUM_LEVELS_KEY: 7,
    CONV_LAYER_COUNTS_KEY: numpy.full(8, 2, dtype=int),
    CONV_LAYER_CHANNEL_COUNTS_KEY:
        numpy.array([16, 24, 32, 48, 64, 96, 128, 192], dtype=int),
    ENCODER_DROPOUT_RATES_KEY: numpy.full(8, 0.),
    UPCONV_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    SKIP_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    INCLUDE_PENULTIMATE_KEY: True,
    PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    OUTPUT_ACTIV_FUNCTION_KEY: architecture_utils.SIGMOID_FUNCTION_STRING,
    OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    L1_WEIGHT_KEY: 0.,
    L2_WEIGHT_KEY: 0.001,
    USE_BATCH_NORM_KEY: True,
    USE_COORD_CONV_KEY: False
}


def _check_architecture_args(option_dict):
    """Error-checks input arguments for architecture.

    L = number of levels = number of pooling operations
                         = number of upsampling operations

    :param option_dict: Dictionary with the following keys.
    option_dict['input_dimensions']: See doc for
        `u_net_architecture._check_architecture_args`.
    option_dict['num_levels']: Same.
    option_dict['num_conv_layers_by_level']: Same.
    option_dict['conv_layer_channel_counts']: Same.
    option_dict['encoder_dropout_rate_by_level']: length-(L + 1) numpy array
        with dropout rate for conv layers in encoder at each level.
    option_dict['upconv_dropout_rate_by_level']: length-L numpy array
        with dropout rate for upconv layers at each level.
    option_dict['skip_dropout_rate_by_level']: length-L numpy array with dropout
        rate for conv layer after skip connection at each level.
    option_dict['include_penultimate_conv']: Boolean flag.  If True, will put in
        extra conv layer (with 3 x 3 filter) before final pixelwise conv.
    option_dict['penultimate_conv_dropout_rate']: Dropout rate for penultimate
        conv layer.
    option_dict['inner_activ_function_name']: See doc for
        `u_net_architecture._check_architecture_args`.
    option_dict['inner_activ_function_alpha']: Same.
    option_dict['output_activ_function_name']: Same.
    option_dict['output_activ_function_alpha']: Same.
    option_dict['l1_weight']: Same.
    option_dict['l2_weight']: Same.
    option_dict['use_batch_normalization']: Same.
    option_dict['use_coord_conv']: Same.

    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_ARCHITECTURE_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    input_dimensions = option_dict[INPUT_DIMENSIONS_KEY]
    error_checking.assert_is_numpy_array(
        input_dimensions, exact_dimensions=numpy.array([3], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(input_dimensions)
    error_checking.assert_is_greater_numpy_array(input_dimensions, 0)

    error_checking.assert_is_integer(option_dict[NUM_LEVELS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_LEVELS_KEY], 2)

    num_levels = option_dict[NUM_LEVELS_KEY]
    expected_dim = numpy.array([num_levels + 1], dtype=int)

    num_conv_layers_by_level = option_dict[CONV_LAYER_COUNTS_KEY]
    error_checking.assert_is_numpy_array(
        num_conv_layers_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(num_conv_layers_by_level)
    error_checking.assert_is_greater_numpy_array(num_conv_layers_by_level, 0)

    error_checking.assert_is_numpy_array(
        option_dict[CONV_LAYER_CHANNEL_COUNTS_KEY],
        exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[CONV_LAYER_CHANNEL_COUNTS_KEY]
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[CONV_LAYER_CHANNEL_COUNTS_KEY], 2
    )

    encoder_dropout_rate_by_level = option_dict[ENCODER_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        encoder_dropout_rate_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        encoder_dropout_rate_by_level, 1., allow_nan=True
    )

    expected_dim = numpy.array([num_levels], dtype=int)

    upconv_dropout_rate_by_level = option_dict[UPCONV_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        upconv_dropout_rate_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        upconv_dropout_rate_by_level, 1., allow_nan=True
    )

    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        skip_dropout_rate_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        skip_dropout_rate_by_level, 1., allow_nan=True
    )

    error_checking.assert_is_boolean(option_dict[INCLUDE_PENULTIMATE_KEY])
    error_checking.assert_is_leq(
        option_dict[PENULTIMATE_DROPOUT_RATE_KEY], 1., allow_nan=True
    )

    error_checking.assert_is_geq(option_dict[L1_WEIGHT_KEY], 0.)
    error_checking.assert_is_geq(option_dict[L2_WEIGHT_KEY], 0.)
    error_checking.assert_is_boolean(option_dict[USE_BATCH_NORM_KEY])
    error_checking.assert_is_boolean(option_dict[USE_COORD_CONV_KEY])

    return option_dict


def create_model(option_dict, loss_function, mask_matrix, metric_names):
    """Creates U-net++.

    This method sets up the architecture, loss function, and optimizer -- and
    compiles the model -- but does not train it.

    M = number of rows in grid
    N = number of columns in grid

    :param option_dict: See doc for `_check_architecture_args`.
    :param loss_function: Loss function.
    :param mask_matrix: M-by-N numpy array of Boolean flags.  Only pixels marked
        "True" are considered in the loss function and metrics.
    :param metric_names: See doc for `neural_net.get_metrics`.
    :return: model_object: Instance of `keras.models.Model`, with the
        aforementioned architecture.
    """

    metric_function_list = neural_net.get_metrics(
        metric_names=metric_names, mask_matrix=mask_matrix,
        use_as_loss_function=False
    )[0]
    option_dict = _check_architecture_args(option_dict)

    input_dimensions = option_dict[INPUT_DIMENSIONS_KEY]
    num_levels = option_dict[NUM_LEVELS_KEY]
    num_conv_layers_by_level = option_dict[CONV_LAYER_COUNTS_KEY]
    conv_layer_channel_counts = option_dict[CONV_LAYER_CHANNEL_COUNTS_KEY]
    encoder_dropout_rate_by_level = option_dict[ENCODER_DROPOUT_RATES_KEY]
    upconv_dropout_rate_by_level = option_dict[UPCONV_DROPOUT_RATES_KEY]
    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    include_penultimate_conv = option_dict[INCLUDE_PENULTIMATE_KEY]
    penultimate_conv_dropout_rate = option_dict[PENULTIMATE_DROPOUT_RATE_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    output_activ_function_name = option_dict[OUTPUT_ACTIV_FUNCTION_KEY]
    output_activ_function_alpha = option_dict[OUTPUT_ACTIV_FUNCTION_ALPHA_KEY]
    l1_weight = option_dict[L1_WEIGHT_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]
    use_coord_conv = option_dict[USE_COORD_CONV_KEY]

    input_layer_object = keras.layers.Input(
        shape=tuple(input_dimensions.tolist())
    )
    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    last_conv_layer_matrix = numpy.full(
        (num_levels + 1, num_levels + 1), '', dtype=object
    )
    pooling_layer_by_level = [None] * num_levels

    for i in range(num_levels + 1):
        for k in range(num_conv_layers_by_level[i]):
            if k == 0:
                if i == 0:
                    this_input_layer_object = input_layer_object
                else:
                    this_input_layer_object = pooling_layer_by_level[i - 1]
            else:
                this_input_layer_object = last_conv_layer_matrix[i, 0]

            if use_coord_conv:
                this_input_layer_object = coord_conv.add_spatial_coords_2d(
                    this_input_layer_object
                )

            this_name = 'block{0:d}-{1:d}_conv{2:d}'.format(i, 0, k)

            last_conv_layer_matrix[i, 0] = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=conv_layer_channel_counts[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object, layer_name=this_name
            )(this_input_layer_object)

            this_name = 'block{0:d}-{1:d}_conv{2:d}_activation'.format(i, 0, k)

            last_conv_layer_matrix[i, 0] = (
                architecture_utils.get_activation_layer(
                    activation_function_string=inner_activ_function_name,
                    alpha_for_relu=inner_activ_function_alpha,
                    alpha_for_elu=inner_activ_function_alpha,
                    layer_name=this_name
                )(last_conv_layer_matrix[i, 0])
            )

            if encoder_dropout_rate_by_level[i] > 0:
                this_name = 'block{0:d}-{1:d}_conv{2:d}_dropout'.format(i, 0, k)

                last_conv_layer_matrix[i, 0] = (
                    architecture_utils.get_dropout_layer(
                        dropout_fraction=encoder_dropout_rate_by_level[i],
                        layer_name=this_name
                    )(last_conv_layer_matrix[i, 0])
                )

            if use_batch_normalization:
                this_name = 'block{0:d}-{1:d}_conv{2:d}_bn'.format(i, 0, k)

                last_conv_layer_matrix[i, 0] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(last_conv_layer_matrix[i, 0])
                )

        if i != num_levels:
            this_name = 'block{0:d}-{1:d}_pooling'.format(i, 0)

            pooling_layer_by_level[i] = architecture_utils.get_2d_pooling_layer(
                num_rows_in_window=2, num_columns_in_window=2,
                num_rows_per_stride=2, num_columns_per_stride=2,
                pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                layer_name=this_name
            )(last_conv_layer_matrix[i, 0])

        i_new = i + 0
        j = 0

        while i_new > 0:
            i_new -= 1
            j += 1

            this_name = 'block{0:d}-{1:d}_upsampling'.format(i_new, j)

            this_layer_object = keras.layers.UpSampling2D(
                size=(2, 2), name=this_name
            )(last_conv_layer_matrix[i_new + 1, j - 1])

            if use_coord_conv:
                this_layer_object = coord_conv.add_spatial_coords_2d(
                    this_layer_object
                )

            this_name = 'block{0:d}-{1:d}_upconv'.format(i_new, j)

            this_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=conv_layer_channel_counts[i_new],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object, layer_name=this_name
            )(this_layer_object)

            this_name = 'block{0:d}-{1:d}_upconv_activation'.format(i_new, j)

            this_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(this_layer_object)

            if upconv_dropout_rate_by_level[i_new] > 0:
                this_name = 'block{0:d}-{1:d}_upconv_dropout'.format(i_new, j)

                this_layer_object = architecture_utils.get_dropout_layer(
                    dropout_fraction=upconv_dropout_rate_by_level[i_new],
                    layer_name=this_name
                )(this_layer_object)

            num_upconv_rows = this_layer_object.get_shape()[1]
            num_desired_rows = last_conv_layer_matrix[i_new, 0].get_shape()[1]
            num_padding_rows = num_desired_rows - num_upconv_rows

            num_upconv_columns = this_layer_object.get_shape()[2]
            num_desired_columns = (
                last_conv_layer_matrix[i_new, 0].get_shape()[2]
            )
            num_padding_columns = num_desired_columns - num_upconv_columns

            if num_padding_rows + num_padding_columns > 0:
                padding_arg = ((0, num_padding_rows), (0, num_padding_columns))

                this_layer_object = keras.layers.ZeroPadding2D(
                    padding=padding_arg
                )(this_layer_object)

            last_conv_layer_matrix[i_new, j] = this_layer_object

            this_name = 'block{0:d}-{1:d}_skip'.format(i_new, j)

            last_conv_layer_matrix[i_new, j] = keras.layers.Concatenate(
                axis=-1, name=this_name
            )(last_conv_layer_matrix[i_new, :(j + 1)].tolist())

            for k in range(num_conv_layers_by_level[i_new]):
                if use_coord_conv:
                    last_conv_layer_matrix[i_new, j] = (
                        coord_conv.add_spatial_coords_2d(
                            last_conv_layer_matrix[i_new, j]
                        )
                    )

                this_name = 'block{0:d}-{1:d}_skipconv{2:d}'.format(i_new, j, k)

                last_conv_layer_matrix[i_new, j] = (
                    architecture_utils.get_2d_conv_layer(
                        num_kernel_rows=3, num_kernel_columns=3,
                        num_rows_per_stride=1, num_columns_per_stride=1,
                        num_filters=conv_layer_channel_counts[i_new],
                        padding_type_string=
                        architecture_utils.YES_PADDING_STRING,
                        weight_regularizer=regularizer_object,
                        layer_name=this_name
                    )(last_conv_layer_matrix[i_new, j])
                )

                this_name = 'block{0:d}-{1:d}_skipconv{2:d}_activation'.format(
                    i_new, j, k
                )

                last_conv_layer_matrix[i_new, j] = (
                    architecture_utils.get_activation_layer(
                        activation_function_string=inner_activ_function_name,
                        alpha_for_relu=inner_activ_function_alpha,
                        alpha_for_elu=inner_activ_function_alpha,
                        layer_name=this_name
                    )(last_conv_layer_matrix[i_new, j])
                )

                if skip_dropout_rate_by_level[i_new] > 0:
                    this_name = 'block{0:d}-{1:d}_skipconv{2:d}_dropout'.format(
                        i_new, j, k
                    )

                    last_conv_layer_matrix[i_new, j] = (
                        architecture_utils.get_dropout_layer(
                            dropout_fraction=skip_dropout_rate_by_level[i_new],
                            layer_name=this_name
                        )(last_conv_layer_matrix[i_new, j])
                    )

                if use_batch_normalization:
                    this_name = 'block{0:d}-{1:d}_skipconv{2:d}_bn'.format(
                        i_new, j, k
                    )

                    last_conv_layer_matrix[i_new, j] = (
                        architecture_utils.get_batch_norm_layer(
                            layer_name=this_name
                        )(last_conv_layer_matrix[i_new, j])
                    )

    if include_penultimate_conv:
        if use_coord_conv:
            last_conv_layer_matrix[0, -1] = coord_conv.add_spatial_coords_2d(
                last_conv_layer_matrix[0, -1]
            )

        last_conv_layer_matrix[0, -1] = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=3, num_kernel_columns=3,
            num_rows_per_stride=1, num_columns_per_stride=1, num_filters=2,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object, layer_name='penultimate_conv'
        )(last_conv_layer_matrix[0, -1])

        last_conv_layer_matrix[0, -1] = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha,
            layer_name='penultimate_conv_activation'
        )(last_conv_layer_matrix[0, -1])

        if penultimate_conv_dropout_rate > 0:
            last_conv_layer_matrix[0, -1] = (
                architecture_utils.get_dropout_layer(
                    dropout_fraction=penultimate_conv_dropout_rate,
                    layer_name='penultimate_conv_dropout'
                )(last_conv_layer_matrix[0, -1])
            )

        if use_batch_normalization:
            last_conv_layer_matrix[0, -1] = (
                architecture_utils.get_batch_norm_layer(
                    layer_name='penultimate_conv_bn'
                )(last_conv_layer_matrix[0, -1])
            )

    if use_coord_conv:
        last_conv_layer_matrix[0, -1] = coord_conv.add_spatial_coords_2d(
            last_conv_layer_matrix[0, -1]
        )

    output_layer_object = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=1, num_kernel_columns=1,
        num_rows_per_stride=1, num_columns_per_stride=1, num_filters=1,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object, layer_name='last_conv'
    )(last_conv_layer_matrix[0, -1])

    output_layer_object = architecture_utils.get_activation_layer(
        activation_function_string=output_activ_function_name,
        alpha_for_relu=output_activ_function_alpha,
        alpha_for_elu=output_activ_function_alpha,
        layer_name='last_conv_activation'
    )(output_layer_object)

    if mask_matrix is not None:
        this_matrix = numpy.expand_dims(
            mask_matrix.astype(float), axis=(0, -1)
        )
        output_layer_object = keras.layers.Multiply()([
            this_matrix, output_layer_object
        ])

    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=output_layer_object
    )

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=metric_function_list
    )

    model_object.summary()
    return model_object
