"""Methods for building U-nets."""

import os
import sys
import numpy
import keras

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import architecture_utils
import neural_net

INPUT_DIMENSIONS_KEY = 'input_dimensions'
NUM_LEVELS_KEY = 'num_levels'
NUM_CONV_LAYERS_KEY = 'num_conv_layers_per_level'
CONV_LAYER_CHANNEL_COUNTS_KEY = 'conv_layer_channel_counts'
CONV_LAYER_DROPOUT_RATES_KEY = 'conv_layer_dropout_rates'
UPCONV_LAYER_DROPOUT_RATES_KEY = 'upconv_layer_dropout_rates'
SKIP_LAYER_DROPOUT_RATES_KEY = 'skip_layer_dropout_rates'
INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'
OUTPUT_ACTIV_FUNCTION_KEY = 'output_activ_function_name'
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = 'output_activ_function_alpha'
L1_WEIGHT_KEY = 'l1_weight'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'

DEFAULT_ARCHITECTURE_OPTION_DICT = {
    NUM_LEVELS_KEY: 7,
    NUM_CONV_LAYERS_KEY: 2,
    CONV_LAYER_CHANNEL_COUNTS_KEY:
        numpy.array([16, 24, 32, 48, 64, 96, 128, 192], dtype=int),
        # numpy.array([16, 32, 64, 128, 256, 512, 1024, 2048], dtype=int),
    CONV_LAYER_DROPOUT_RATES_KEY: numpy.full(8, 0.5),
    UPCONV_LAYER_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    SKIP_LAYER_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    OUTPUT_ACTIV_FUNCTION_KEY: architecture_utils.SIGMOID_FUNCTION_STRING,
    OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    L1_WEIGHT_KEY: 0.,
    L2_WEIGHT_KEY: 0.001,
    USE_BATCH_NORM_KEY: True
}


def _check_architecture_args(option_dict):
    """Error-checks input arguments for architecture.

    L = number of levels = number of pooling operations
                         = number of upsampling operations

    :param option_dict: Dictionary with the following keys.
    option_dict['input_dimensions']: length-3 numpy array with input dimensions
        (num_rows, num_columns, num_channels).
    option_dict['num_levels']: L in the above discussion.
    option_dict['num_conv_layers_per_level']: Number of conv layers per level.
    option_dict['conv_layer_channel_counts']:length-(L + 1) numpy array with
        number of channels (filters) produced by each conv layer.
    option_dict['conv_layer_dropout_rates']: length-(L + 1) numpy array with
        dropout rate for each conv layer.  To omit dropout for a particular
        layer, use NaN or a number <= 0.
    option_dict['upconv_layer_dropout_rates']: Same as above, except for upconv
        layers and array has length L.
    option_dict['skip_layer_dropout_rates']: Same as above, except for skip
        layers and array has length L.
    option_dict['inner_activ_function_name']: Name of activation function for
        all inner (non-output) layers.  Must be accepted by
        `architecture_utils.check_activation_function`.
    option_dict['inner_activ_function_alpha']: Alpha (slope parameter) for
        activation function for all inner layers.  Applies only to ReLU and eLU.
    option_dict['output_activ_function_name']: Same as
        `inner_activ_function_name` but for output layers (profiles and
        scalars).
    option_dict['output_activ_function_alpha']: Same as
        `inner_activ_function_alpha` but for output layers (profiles and
        scalars).
    option_dict['l1_weight']: Weight for L_1 regularization.
    option_dict['l2_weight']: Weight for L_2 regularization.
    option_dict['use_batch_normalization']: Boolean flag.  If True, will use
        batch normalization after each inner (non-output) layer.

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
    error_checking.assert_is_integer(option_dict[NUM_CONV_LAYERS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_CONV_LAYERS_KEY], 1)

    num_levels = option_dict[NUM_LEVELS_KEY]
    expected_dim = numpy.array([num_levels + 1], dtype=int)

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

    error_checking.assert_is_numpy_array(
        option_dict[CONV_LAYER_DROPOUT_RATES_KEY], exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        option_dict[CONV_LAYER_DROPOUT_RATES_KEY], 1., allow_nan=True
    )

    expected_dim = numpy.array([num_levels], dtype=int)

    error_checking.assert_is_numpy_array(
        option_dict[UPCONV_LAYER_DROPOUT_RATES_KEY],
        exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        option_dict[UPCONV_LAYER_DROPOUT_RATES_KEY], 1., allow_nan=True
    )

    error_checking.assert_is_numpy_array(
        option_dict[SKIP_LAYER_DROPOUT_RATES_KEY], exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        option_dict[SKIP_LAYER_DROPOUT_RATES_KEY], 1., allow_nan=True
    )

    error_checking.assert_is_geq(option_dict[L1_WEIGHT_KEY], 0.)
    error_checking.assert_is_geq(option_dict[L2_WEIGHT_KEY], 0.)
    error_checking.assert_is_boolean(option_dict[USE_BATCH_NORM_KEY])

    return option_dict


def create_model(option_dict, loss_function, mask_matrix,
                 num_batches_per_update=None):
    """Creates U-net.

    This method sets up the architecture, loss function, and optimizer -- and
    compiles the model -- but does not train it.

    Architecture taken from:
    https://github.com/zhixuhao/unet/blob/master/model.py

    M = number of rows in grid
    N = number of columns in grid

    :param option_dict: See doc for `_check_architecture_args`.
    :param loss_function: Loss function.
    :param mask_matrix: M-by-N numpy array of Boolean flags.  Only pixels marked
        "True" are considered in the loss function and metrics.
    :param num_batches_per_update: Number of batches per weight update.  If you
        want to update weights after each batch, like a normal person, leave
        this argument alone.
    :return: model_object: Instance of `keras.models.Model`, with the
        aforementioned architecture.
    """

    if num_batches_per_update is not None:
        error_checking.assert_is_integer(num_batches_per_update)
        error_checking.assert_is_greater(num_batches_per_update, 1)

    metric_function_list = neural_net.get_metrics(mask_matrix)[0]
    option_dict = _check_architecture_args(option_dict)

    input_dimensions = option_dict[INPUT_DIMENSIONS_KEY]
    num_levels = option_dict[NUM_LEVELS_KEY]
    num_conv_layers_per_level = option_dict[NUM_CONV_LAYERS_KEY]
    conv_layer_channel_counts = option_dict[CONV_LAYER_CHANNEL_COUNTS_KEY]
    conv_layer_dropout_rates = option_dict[CONV_LAYER_DROPOUT_RATES_KEY]
    upconv_layer_dropout_rates = option_dict[UPCONV_LAYER_DROPOUT_RATES_KEY]
    skip_layer_dropout_rates = option_dict[SKIP_LAYER_DROPOUT_RATES_KEY]
    inner_activ_function_name = option_dict[INNER_ACTIV_FUNCTION_KEY]
    inner_activ_function_alpha = option_dict[INNER_ACTIV_FUNCTION_ALPHA_KEY]
    output_activ_function_name = option_dict[OUTPUT_ACTIV_FUNCTION_KEY]
    output_activ_function_alpha = option_dict[OUTPUT_ACTIV_FUNCTION_ALPHA_KEY]
    l1_weight = option_dict[L1_WEIGHT_KEY]
    l2_weight = option_dict[L2_WEIGHT_KEY]
    use_batch_normalization = option_dict[USE_BATCH_NORM_KEY]

    input_layer_object = keras.layers.Input(
        shape=tuple(input_dimensions.tolist())
    )
    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    conv_layer_by_level = [None] * (num_levels + 1)
    pooling_layer_by_level = [None] * num_levels

    for i in range(num_levels + 1):
        for j in range(num_conv_layers_per_level):
            if j == 0:
                if i == 0:
                    this_input_layer_object = input_layer_object
                else:
                    this_input_layer_object = pooling_layer_by_level[i - 1]
            else:
                this_input_layer_object = conv_layer_by_level[i]

            conv_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=conv_layer_channel_counts[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )(this_input_layer_object)

            conv_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(conv_layer_by_level[i])

            if conv_layer_dropout_rates[i] > 0:
                conv_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=conv_layer_dropout_rates[i]
                )(conv_layer_by_level[i])

            if use_batch_normalization:
                conv_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer()(
                        conv_layer_by_level[i]
                    )
                )

        if i == num_levels:
            break

        pooling_layer_by_level[i] = architecture_utils.get_2d_pooling_layer(
            num_rows_in_window=2, num_columns_in_window=2,
            num_rows_per_stride=2, num_columns_per_stride=2,
            pooling_type_string=architecture_utils.MAX_POOLING_STRING
        )(conv_layer_by_level[i])

    upconv_layer_by_level = [None] * num_levels
    skip_layer_by_level = [None] * num_levels
    merged_layer_by_level = [None] * num_levels

    try:
        this_layer_object = keras.layers.UpSampling2D(
            size=(2, 2), interpolation='bilinear'
        )(conv_layer_by_level[num_levels])
    except:
        this_layer_object = keras.layers.UpSampling2D(
            size=(2, 2)
        )(conv_layer_by_level[num_levels])

    i = num_levels - 1
    upconv_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=2, num_kernel_columns=2,
        num_rows_per_stride=1, num_columns_per_stride=1,
        num_filters=conv_layer_channel_counts[i],
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )(this_layer_object)

    if upconv_layer_dropout_rates[i] > 0:
        upconv_layer_by_level[i] = architecture_utils.get_dropout_layer(
            dropout_fraction=upconv_layer_dropout_rates[i]
        )(upconv_layer_by_level[i])

    num_upconv_rows = upconv_layer_by_level[i].get_shape()[1]
    num_desired_rows = conv_layer_by_level[i].get_shape()[1]
    num_padding_rows = num_desired_rows - num_upconv_rows

    num_upconv_columns = upconv_layer_by_level[i].get_shape()[2]
    num_desired_columns = conv_layer_by_level[i].get_shape()[2]
    num_padding_columns = num_desired_columns - num_upconv_columns

    if num_padding_rows + num_padding_columns > 0:
        padding_arg = ((0, num_padding_rows), (0, num_padding_columns))

        upconv_layer_by_level[i] = keras.layers.ZeroPadding2D(
            padding=padding_arg
        )(upconv_layer_by_level[i])

    merged_layer_by_level[i] = keras.layers.Concatenate(axis=-1)(
        [conv_layer_by_level[i], upconv_layer_by_level[i]]
    )

    level_indices = numpy.linspace(
        0, num_levels - 1, num=num_levels, dtype=int
    )[::-1]

    for i in level_indices:
        for j in range(num_conv_layers_per_level):
            if j == 0:
                this_input_layer_object = merged_layer_by_level[i]
            else:
                this_input_layer_object = skip_layer_by_level[i]

            skip_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=conv_layer_channel_counts[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )(this_input_layer_object)

            skip_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(skip_layer_by_level[i])

            if skip_layer_dropout_rates[i] > 0:
                skip_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=skip_layer_dropout_rates[i]
                )(skip_layer_by_level[i])

            if use_batch_normalization:
                skip_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer()(
                        skip_layer_by_level[i]
                    )
                )

        if i == 0:
            skip_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1, num_filters=2,
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )(skip_layer_by_level[i])

            skip_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(skip_layer_by_level[i])

            if use_batch_normalization:
                skip_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer()(
                        skip_layer_by_level[i]
                    )
                )

            break

        try:
            this_layer_object = keras.layers.UpSampling2D(
                size=(2, 2), interpolation='bilinear'
            )(skip_layer_by_level[i])
        except:
            this_layer_object = keras.layers.UpSampling2D(
                size=(2, 2)
            )(skip_layer_by_level[i])

        upconv_layer_by_level[i - 1] = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=2, num_kernel_columns=2,
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_filters=conv_layer_channel_counts[i - 1],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_layer_object)

        if upconv_layer_dropout_rates[i - 1] > 0:
            upconv_layer_by_level[i - 1] = architecture_utils.get_dropout_layer(
                dropout_fraction=upconv_layer_dropout_rates[i - 1]
            )(upconv_layer_by_level[i - 1])

        num_upconv_rows = upconv_layer_by_level[i - 1].get_shape()[1]
        num_desired_rows = conv_layer_by_level[i - 1].get_shape()[1]
        num_padding_rows = num_desired_rows - num_upconv_rows

        num_upconv_columns = upconv_layer_by_level[i - 1].get_shape()[2]
        num_desired_columns = conv_layer_by_level[i - 1].get_shape()[2]
        num_padding_columns = num_desired_columns - num_upconv_columns

        if num_padding_rows + num_padding_columns > 0:
            padding_arg = ((0, num_padding_rows), (0, num_padding_columns))

            upconv_layer_by_level[i - 1] = keras.layers.ZeroPadding2D(
                padding=padding_arg
            )(upconv_layer_by_level[i - 1])

        merged_layer_by_level[i - 1] = keras.layers.Concatenate(axis=-1)(
            [conv_layer_by_level[i - 1], upconv_layer_by_level[i - 1]]
        )

    skip_layer_by_level[0] = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=1, num_kernel_columns=1,
        num_rows_per_stride=1, num_columns_per_stride=1, num_filters=1,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )(skip_layer_by_level[0])

    skip_layer_by_level[0] = architecture_utils.get_activation_layer(
        activation_function_string=output_activ_function_name,
        alpha_for_relu=output_activ_function_alpha,
        alpha_for_elu=output_activ_function_alpha
    )(skip_layer_by_level[0])

    if mask_matrix is not None:
        this_matrix = numpy.expand_dims(
            mask_matrix.astype(float), axis=(0, -1)
        )
        skip_layer_by_level[0] = keras.layers.Multiply()([
            this_matrix, skip_layer_by_level[0]
        ])

    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=skip_layer_by_level[0]
    )

    if num_batches_per_update is None:
        optimizer_object = keras.optimizers.Adam()
    else:
        optimizer_object = neural_net.AccumOptimizer(
            optimizer_object=keras.optimizers.Adam(),
            num_batches_per_update=num_batches_per_update
        )

    model_object.compile(
        loss=loss_function, optimizer=optimizer_object,
        metrics=metric_function_list
    )

    model_object.summary()
    return model_object
