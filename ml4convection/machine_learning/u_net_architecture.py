"""Methods for building U-nets."""

import numpy
import keras
from keras.layers import Add
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from ml4convection.machine_learning import neural_net
from ml4convection.machine_learning import coord_conv
from ml4convection.machine_learning import custom_losses

INPUT_DIMENSIONS_KEY = 'input_dimensions'
NUM_LEVELS_KEY = 'num_levels'
CONV_LAYER_COUNTS_KEY = 'num_conv_layers_by_level'
OUTPUT_CHANNEL_COUNTS_KEY = 'num_output_channels_by_level'
CONV_DROPOUT_RATES_KEY = 'conv_dropout_rates_by_level'
UPCONV_DROPOUT_RATES_KEY = 'upconv_dropout_rate_by_level'
SKIP_DROPOUT_RATES_KEY = 'skip_dropout_rates_by_level'
SKIP_DROPOUT_MC_FLAGS_KEY = 'skip_dropout_mc_flags_by_level'
INCLUDE_PENULTIMATE_KEY = 'include_penultimate_conv'
PENULTIMATE_DROPOUT_RATE_KEY = 'penultimate_conv_dropout_rate'
PENULTIMATE_DROPOUT_MC_FLAG_KEY = 'penultimate_conv_dropout_mc_flag'
OUTPUT_DROPOUT_RATE_KEY = 'output_layer_dropout_rate'
OUTPUT_DROPOUT_MC_FLAG_KEY = 'output_layer_dropout_mc_flag'
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
    OUTPUT_CHANNEL_COUNTS_KEY:
        numpy.array([16, 24, 32, 48, 64, 96, 128, 192], dtype=int),
    CONV_DROPOUT_RATES_KEY: [numpy.full(2, 0.)] * 8,
    UPCONV_DROPOUT_RATES_KEY: numpy.full(7, 0.),
    SKIP_DROPOUT_RATES_KEY: [numpy.full(2, 0.)] * 7,
    SKIP_DROPOUT_MC_FLAGS_KEY: [numpy.full(2, 0, dtype=bool)] * 7,
    INCLUDE_PENULTIMATE_KEY: True,
    PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    PENULTIMATE_DROPOUT_MC_FLAG_KEY: False,
    OUTPUT_DROPOUT_RATE_KEY: 0.,
    OUTPUT_DROPOUT_MC_FLAG_KEY: False,
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
    option_dict['input_dimensions']: length-3 numpy array with input dimensions
        (num_rows, num_columns, num_channels).
    option_dict['num_levels']: L in the above discussion.
    option_dict['num_conv_layers_by_level']: length-(L + 1) numpy array with
        number of conv layers at each level.
    option_dict['num_output_channels_by_level']: length-(L + 1) numpy array with
        number of output channels at each level.
    option_dict['conv_dropout_rates_by_level']: length-(L + 1) list, where each
        list item is a 1-D numpy array of dropout rates.  The [k]th list item
        should be an array with length = number of conv layers at the [k]th
        level.  Use values <= 0 to omit dropout.
    option_dict['upconv_dropout_rate_by_level']: length-L numpy array of dropout
        rates for upconv layers.
    option_dict['skip_dropout_rates_by_level']: length-L list, where each list
        item is a 1-D numpy array of dropout rates.  The [k]th list item
        should be an array with length = number of conv layers at the [k]th
        level.  Use values <= 0 to omit dropout.
    option_dict['skip_dropout_mc_flags_by_level']: Same as above, but each list
        item is a numpy array of Boolean flags, not dropout rates.  The Boolean
        flags tell the model whether or not to use Monte Carlo dropout, i.e.,
        dropout at inference time.
    option_dict['include_penultimate_conv']: Boolean flag.  If True, will put in
        extra conv layer (with 3 x 3 filter) before final pixelwise conv.
    option_dict['penultimate_conv_dropout_rate']: Dropout rate for penultimate
        conv layer.
    option_dict['penultimate_conv_dropout_mc_flag']: Boolean flag, telling the
        model whether or not to use Monte Carlo dropout in the penultimate conv
        layer.
    option_dict['output_layer_dropout_rate']: Dropout rate for output layer.
    option_dict['output_layer_dropout_mc_flag']: Boolean flag, telling the
        model whether or not to use Monte Carlo dropout in the output layer.
    option_dict['inner_activ_function_name']: Name of activation function for
        all inner (non-output) layers.  Must be accepted by
        `architecture_utils.check_activation_function`.
    option_dict['inner_activ_function_alpha']: Alpha (slope parameter) for
        activation function for all inner layers.  Applies only to ReLU and eLU.
    option_dict['output_activ_function_name']: Same as
        `inner_activ_function_name` but for output layer.
    option_dict['output_activ_function_alpha']: Same as
        `inner_activ_function_alpha` but for output layer.
    option_dict['l1_weight']: Weight for L_1 regularization.
    option_dict['l2_weight']: Weight for L_2 regularization.
    option_dict['use_batch_normalization']: Boolean flag.  If True, will use
        batch normalization after each inner (non-output) layer.
    option_dict['use_coord_conv']: Boolean flag.  If True, will use coord-conv
        in each convolutional layer.

    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_ARCHITECTURE_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_numpy_array(
        option_dict[INPUT_DIMENSIONS_KEY],
        exact_dimensions=numpy.array([3], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[INPUT_DIMENSIONS_KEY]
    )
    error_checking.assert_is_greater_numpy_array(
        option_dict[INPUT_DIMENSIONS_KEY], 0
    )

    error_checking.assert_is_integer(option_dict[NUM_LEVELS_KEY])
    error_checking.assert_is_geq(option_dict[NUM_LEVELS_KEY], 2)
    expected_dim = numpy.array([option_dict[NUM_LEVELS_KEY] + 1], dtype=int)

    error_checking.assert_is_numpy_array(
        option_dict[CONV_LAYER_COUNTS_KEY], exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[CONV_LAYER_COUNTS_KEY]
    )
    error_checking.assert_is_greater_numpy_array(
        option_dict[CONV_LAYER_COUNTS_KEY], 0
    )

    error_checking.assert_is_numpy_array(
        option_dict[OUTPUT_CHANNEL_COUNTS_KEY], exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[OUTPUT_CHANNEL_COUNTS_KEY]
    )
    error_checking.assert_is_geq_numpy_array(
        option_dict[OUTPUT_CHANNEL_COUNTS_KEY], 2
    )

    assert (
        len(option_dict[CONV_DROPOUT_RATES_KEY]) ==
        option_dict[NUM_LEVELS_KEY] + 1
    )

    for k in range(option_dict[NUM_LEVELS_KEY] + 1):
        these_dim = numpy.array(
            [option_dict[CONV_LAYER_COUNTS_KEY][k]], dtype=int
        )
        error_checking.assert_is_numpy_array(
            option_dict[CONV_DROPOUT_RATES_KEY][k], exact_dimensions=these_dim
        )
        error_checking.assert_is_leq_numpy_array(
            option_dict[CONV_DROPOUT_RATES_KEY][k], 1., allow_nan=True
        )

    expected_dim = numpy.array([option_dict[NUM_LEVELS_KEY]], dtype=int)

    error_checking.assert_is_numpy_array(
        option_dict[UPCONV_DROPOUT_RATES_KEY], exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        option_dict[UPCONV_DROPOUT_RATES_KEY], 1., allow_nan=True
    )

    assert (
        len(option_dict[SKIP_DROPOUT_RATES_KEY]) ==
        option_dict[NUM_LEVELS_KEY]
    )
    assert (
        len(option_dict[SKIP_DROPOUT_MC_FLAGS_KEY]) ==
        option_dict[NUM_LEVELS_KEY]
    )

    for k in range(option_dict[NUM_LEVELS_KEY]):
        these_dim = numpy.array(
            [option_dict[CONV_LAYER_COUNTS_KEY][k]], dtype=int
        )
        error_checking.assert_is_numpy_array(
            option_dict[SKIP_DROPOUT_RATES_KEY][k], exact_dimensions=these_dim
        )
        error_checking.assert_is_leq_numpy_array(
            option_dict[SKIP_DROPOUT_RATES_KEY][k], 1., allow_nan=True
        )

        error_checking.assert_is_numpy_array(
            option_dict[SKIP_DROPOUT_MC_FLAGS_KEY][k],
            exact_dimensions=these_dim
        )
        error_checking.assert_is_boolean_numpy_array(
            option_dict[SKIP_DROPOUT_MC_FLAGS_KEY][k]
        )

    error_checking.assert_is_boolean(option_dict[INCLUDE_PENULTIMATE_KEY])
    error_checking.assert_is_leq(
        option_dict[PENULTIMATE_DROPOUT_RATE_KEY], 1., allow_nan=True
    )
    error_checking.assert_is_boolean(
        option_dict[PENULTIMATE_DROPOUT_MC_FLAG_KEY]
    )
    error_checking.assert_is_leq(
        option_dict[OUTPUT_DROPOUT_RATE_KEY], 1., allow_nan=True
    )
    error_checking.assert_is_boolean(option_dict[OUTPUT_DROPOUT_MC_FLAG_KEY])

    error_checking.assert_is_geq(option_dict[L1_WEIGHT_KEY], 0.)
    error_checking.assert_is_geq(option_dict[L2_WEIGHT_KEY], 0.)
    error_checking.assert_is_boolean(option_dict[USE_BATCH_NORM_KEY])
    error_checking.assert_is_boolean(option_dict[USE_COORD_CONV_KEY])

    return option_dict


def create_quantile_regression_model(
        option_dict, central_loss_function, mask_matrix, quantile_levels):
    """Creates U-net for quantile regression.

    M = number of rows in grid
    N = number of columns in grid

    :param option_dict: See doc for `create_model`.
    :param central_loss_function: Loss function for central prediction.
    :param mask_matrix: See doc for `create_model`.
    :param quantile_levels: 1-D numpy array of quantile levels, ranging from
        (0, 1).
    :return: model_object: Instance of `keras.models.Model`.
    """

    option_dict = _check_architecture_args(option_dict)

    error_checking.assert_is_numpy_array(quantile_levels, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(quantile_levels, 0.)
    error_checking.assert_is_less_than_numpy_array(quantile_levels, 1.)
    quantile_levels = numpy.sort(quantile_levels)

    input_dimensions = option_dict[INPUT_DIMENSIONS_KEY]
    num_levels = option_dict[NUM_LEVELS_KEY]
    num_conv_layers_by_level = option_dict[CONV_LAYER_COUNTS_KEY]
    num_output_channels_by_level = option_dict[OUTPUT_CHANNEL_COUNTS_KEY]
    conv_dropout_rates_by_level = option_dict[CONV_DROPOUT_RATES_KEY]
    upconv_dropout_rate_by_level = option_dict[UPCONV_DROPOUT_RATES_KEY]
    skip_dropout_rates_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    include_penultimate_conv = option_dict[INCLUDE_PENULTIMATE_KEY]
    penultimate_conv_dropout_rate = option_dict[PENULTIMATE_DROPOUT_RATE_KEY]
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
        for j in range(num_conv_layers_by_level[i]):
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
                num_filters=num_output_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )(this_input_layer_object)

            conv_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(conv_layer_by_level[i])

            if conv_dropout_rates_by_level[i][j] > 0:
                conv_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=conv_dropout_rates_by_level[i][j]
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
        num_filters=num_output_channels_by_level[i],
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )(this_layer_object)

    if upconv_dropout_rate_by_level[i] > 0:
        upconv_layer_by_level[i] = architecture_utils.get_dropout_layer(
            dropout_fraction=upconv_dropout_rate_by_level[i]
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

    num_output_channels = len(quantile_levels) + 1
    penultimate_conv_layers = [None] * num_output_channels

    for i in level_indices:
        for j in range(num_conv_layers_by_level[i]):
            if j == 0:
                this_input_layer_object = merged_layer_by_level[i]
            else:
                this_input_layer_object = skip_layer_by_level[i]

            skip_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_output_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )(this_input_layer_object)

            skip_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(skip_layer_by_level[i])

            this_dropout_rate = skip_dropout_rates_by_level[i][j]

            if this_dropout_rate > 0:
                skip_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=this_dropout_rate
                )(skip_layer_by_level[i])

            if use_batch_normalization:
                skip_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer()(
                        skip_layer_by_level[i]
                    )
                )

        if i == 0 and include_penultimate_conv:
            for k in range(num_output_channels):
                penultimate_conv_layers[k] = (
                    architecture_utils.get_2d_conv_layer(
                        num_kernel_rows=3, num_kernel_columns=3,
                        num_rows_per_stride=1, num_columns_per_stride=1,
                        num_filters=2,
                        padding_type_string=
                        architecture_utils.YES_PADDING_STRING,
                        weight_regularizer=regularizer_object
                    )(skip_layer_by_level[i])
                )

                penultimate_conv_layers[k] = (
                    architecture_utils.get_activation_layer(
                        activation_function_string=inner_activ_function_name,
                        alpha_for_relu=inner_activ_function_alpha,
                        alpha_for_elu=inner_activ_function_alpha
                    )(penultimate_conv_layers[k])
                )

                if penultimate_conv_dropout_rate > 0:
                    penultimate_conv_layers[k] = (
                        architecture_utils.get_dropout_layer(
                            dropout_fraction=penultimate_conv_dropout_rate
                        )(penultimate_conv_layers[k])
                    )

                if use_batch_normalization:
                    penultimate_conv_layers[k] = (
                        architecture_utils.get_batch_norm_layer()(
                            penultimate_conv_layers[k]
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
            num_filters=num_output_channels_by_level[i - 1],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_layer_object)

        if upconv_dropout_rate_by_level[i - 1] > 0:
            upconv_layer_by_level[i - 1] = architecture_utils.get_dropout_layer(
                dropout_fraction=upconv_dropout_rate_by_level[i - 1]
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

    output_layers = [None] * num_output_channels
    output_layer_names = [
        'quantile_output{0:03d}'.format(k) for k in range(num_output_channels)
    ]
    output_layer_names[0] = 'central_output'

    loss_dict = {}

    for k in range(num_output_channels):
        output_layers[k] = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=1, num_kernel_columns=1,
            num_rows_per_stride=1, num_columns_per_stride=1, num_filters=1,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(penultimate_conv_layers[k])

        output_layers[k] = architecture_utils.get_activation_layer(
            activation_function_string=output_activ_function_name,
            alpha_for_relu=output_activ_function_alpha,
            alpha_for_elu=output_activ_function_alpha,
            layer_name=output_layer_names[k] if mask_matrix is None else None
        )(output_layers[k])

        if mask_matrix is not None:
            this_matrix = numpy.expand_dims(
                mask_matrix.astype(float), axis=(0, -1)
            )
            output_layers[k] = keras.layers.Multiply(
                name=output_layer_names[k]
            )([this_matrix, output_layers[k]])

            # TODO(thunderhoser): This is a HACK, because for some reason Keras
            # doesn't let you name Multiply layers.
            output_layers[k] = keras.layers.Activation(
                None, name=output_layer_names[k]
            )(output_layers[k])

        if k == 0:
            loss_dict[output_layer_names[k]] = central_loss_function
        else:
            loss_dict[output_layer_names[k]] = custom_losses.quantile_loss(
                quantile_level=quantile_levels[k - 1],
                mask_matrix=mask_matrix.astype(int)
            )

    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=output_layers
    )

    model_object.compile(
        loss=loss_dict, optimizer=keras.optimizers.Adam()
    )

    model_object.summary()
    return model_object


def create_qr_model_fancy(
        option_dict, central_loss_function, mask_matrix, quantile_levels):
    """Creates 'fancy' U-net for quantile regression.

    The 'fancy' U-net completely prevents quantile-crossing.

    M = number of rows in grid
    N = number of columns in grid

    :param option_dict: See doc for `create_model`.
    :param central_loss_function: Loss function for central prediction.
    :param mask_matrix: See doc for `create_model`.
    :param quantile_levels: 1-D numpy array of quantile levels, ranging from
        (0, 1).
    :return: model_object: Instance of `keras.models.Model`.
    """

    option_dict = _check_architecture_args(option_dict)

    error_checking.assert_is_numpy_array(quantile_levels, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(quantile_levels, 0.)
    error_checking.assert_is_less_than_numpy_array(quantile_levels, 1.)
    quantile_levels = numpy.sort(quantile_levels)

    input_dimensions = option_dict[INPUT_DIMENSIONS_KEY]
    num_levels = option_dict[NUM_LEVELS_KEY]
    num_conv_layers_by_level = option_dict[CONV_LAYER_COUNTS_KEY]
    num_output_channels_by_level = option_dict[OUTPUT_CHANNEL_COUNTS_KEY]
    conv_dropout_rates_by_level = option_dict[CONV_DROPOUT_RATES_KEY]
    upconv_dropout_rate_by_level = option_dict[UPCONV_DROPOUT_RATES_KEY]
    skip_dropout_rates_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    include_penultimate_conv = option_dict[INCLUDE_PENULTIMATE_KEY]
    penultimate_conv_dropout_rate = option_dict[PENULTIMATE_DROPOUT_RATE_KEY]
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
        for j in range(num_conv_layers_by_level[i]):
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
                num_filters=num_output_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )(this_input_layer_object)

            conv_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(conv_layer_by_level[i])

            if conv_dropout_rates_by_level[i][j] > 0:
                conv_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=conv_dropout_rates_by_level[i][j]
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
        num_filters=num_output_channels_by_level[i],
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )(this_layer_object)

    if upconv_dropout_rate_by_level[i] > 0:
        upconv_layer_by_level[i] = architecture_utils.get_dropout_layer(
            dropout_fraction=upconv_dropout_rate_by_level[i]
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

    num_output_channels = len(quantile_levels) + 1
    penultimate_conv_layers = [None] * num_output_channels

    for i in level_indices:
        for j in range(num_conv_layers_by_level[i]):
            if j == 0:
                this_input_layer_object = merged_layer_by_level[i]
            else:
                this_input_layer_object = skip_layer_by_level[i]

            skip_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_output_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )(this_input_layer_object)

            skip_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(skip_layer_by_level[i])

            this_dropout_rate = skip_dropout_rates_by_level[i][j]

            if this_dropout_rate > 0:
                skip_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=this_dropout_rate
                )(skip_layer_by_level[i])

            if use_batch_normalization:
                skip_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer()(
                        skip_layer_by_level[i]
                    )
                )

        if i == 0 and include_penultimate_conv:
            for k in range(num_output_channels):
                penultimate_conv_layers[k] = (
                    architecture_utils.get_2d_conv_layer(
                        num_kernel_rows=3, num_kernel_columns=3,
                        num_rows_per_stride=1, num_columns_per_stride=1,
                        num_filters=2,
                        padding_type_string=
                        architecture_utils.YES_PADDING_STRING,
                        weight_regularizer=regularizer_object
                    )(skip_layer_by_level[i])
                )

                penultimate_conv_layers[k] = (
                    architecture_utils.get_activation_layer(
                        activation_function_string=inner_activ_function_name,
                        alpha_for_relu=inner_activ_function_alpha,
                        alpha_for_elu=inner_activ_function_alpha
                    )(penultimate_conv_layers[k])
                )

                if penultimate_conv_dropout_rate > 0:
                    penultimate_conv_layers[k] = (
                        architecture_utils.get_dropout_layer(
                            dropout_fraction=penultimate_conv_dropout_rate
                        )(penultimate_conv_layers[k])
                    )

                if use_batch_normalization:
                    penultimate_conv_layers[k] = (
                        architecture_utils.get_batch_norm_layer()(
                            penultimate_conv_layers[k]
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
            num_filters=num_output_channels_by_level[i - 1],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_layer_object)

        if upconv_dropout_rate_by_level[i - 1] > 0:
            upconv_layer_by_level[i - 1] = architecture_utils.get_dropout_layer(
                dropout_fraction=upconv_dropout_rate_by_level[i - 1]
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

    pre_activn_out_layers = [None] * num_output_channels
    output_layers = [None] * num_output_channels
    output_layer_names = [
        'quantile_output{0:03d}'.format(k) for k in range(num_output_channels)
    ]
    output_layer_names[0] = 'central_output'

    loss_dict = {}

    for k in range(num_output_channels):
        pre_activn_out_layers[k] = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=1, num_kernel_columns=1,
            num_rows_per_stride=1, num_columns_per_stride=1, num_filters=1,
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(penultimate_conv_layers[k])

        if k > 1:
            pre_activn_out_layers[k] = architecture_utils.get_activation_layer(
                activation_function_string=
                architecture_utils.RELU_FUNCTION_STRING,
                alpha_for_relu=0., alpha_for_elu=0.
            )(pre_activn_out_layers[k])

            pre_activn_out_layers[k] = Add()(
                [pre_activn_out_layers[k - 1], pre_activn_out_layers[k]]
            )

        output_layers[k] = architecture_utils.get_activation_layer(
            activation_function_string=output_activ_function_name,
            alpha_for_relu=output_activ_function_alpha,
            alpha_for_elu=output_activ_function_alpha,
            layer_name=
            output_layer_names[k] if mask_matrix is None else None
        )(pre_activn_out_layers[k])

        if mask_matrix is not None:
            this_matrix = numpy.expand_dims(
                mask_matrix.astype(float), axis=(0, -1)
            )
            output_layers[k] = keras.layers.Multiply(
                name=output_layer_names[k]
            )([this_matrix, output_layers[k]])

            # TODO(thunderhoser): This is a HACK, because for some reason Keras
            # doesn't let you name Multiply layers.
            output_layers[k] = keras.layers.Activation(
                None, name=output_layer_names[k]
            )(output_layers[k])

        if k == 0:
            loss_dict[output_layer_names[k]] = central_loss_function
        else:
            loss_dict[output_layer_names[k]] = custom_losses.quantile_loss(
                quantile_level=quantile_levels[k - 1],
                mask_matrix=mask_matrix.astype(int)
            )

    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=output_layers
    )

    model_object.compile(
        loss=loss_dict, optimizer=keras.optimizers.Adam()
    )

    model_object.summary()
    return model_object


def create_model(option_dict, loss_function, mask_matrix, metric_names):
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
    num_output_channels_by_level = option_dict[OUTPUT_CHANNEL_COUNTS_KEY]
    conv_dropout_rates_by_level = option_dict[CONV_DROPOUT_RATES_KEY]
    upconv_dropout_rate_by_level = option_dict[UPCONV_DROPOUT_RATES_KEY]
    skip_dropout_rates_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    skip_dropout_mc_flags_by_level = option_dict[SKIP_DROPOUT_MC_FLAGS_KEY]
    include_penultimate_conv = option_dict[INCLUDE_PENULTIMATE_KEY]
    penultimate_conv_dropout_rate = option_dict[PENULTIMATE_DROPOUT_RATE_KEY]
    penultimate_conv_dropout_mc_flag = (
        option_dict[PENULTIMATE_DROPOUT_MC_FLAG_KEY]
    )
    output_layer_dropout_rate = option_dict[OUTPUT_DROPOUT_RATE_KEY]
    output_layer_dropout_mc_flag = option_dict[OUTPUT_DROPOUT_MC_FLAG_KEY]
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

    conv_layer_by_level = [None] * (num_levels + 1)
    pooling_layer_by_level = [None] * num_levels

    for i in range(num_levels + 1):
        for j in range(num_conv_layers_by_level[i]):
            if j == 0:
                if i == 0:
                    this_input_layer_object = input_layer_object
                else:
                    this_input_layer_object = pooling_layer_by_level[i - 1]
            else:
                this_input_layer_object = conv_layer_by_level[i]

            if use_coord_conv:
                this_input_layer_object = coord_conv.add_spatial_coords_2d(
                    this_input_layer_object
                )

            conv_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_output_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )(this_input_layer_object)

            conv_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(conv_layer_by_level[i])

            if conv_dropout_rates_by_level[i][j] > 0:
                conv_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=conv_dropout_rates_by_level[i][j]
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

    if use_coord_conv:
        this_layer_object = coord_conv.add_spatial_coords_2d(this_layer_object)

    i = num_levels - 1
    upconv_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=2, num_kernel_columns=2,
        num_rows_per_stride=1, num_columns_per_stride=1,
        num_filters=num_output_channels_by_level[i],
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )(this_layer_object)

    if upconv_dropout_rate_by_level[i] > 0:
        upconv_layer_by_level[i] = architecture_utils.get_dropout_layer(
            dropout_fraction=upconv_dropout_rate_by_level[i]
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
        for j in range(num_conv_layers_by_level[i]):
            if j == 0:
                this_input_layer_object = merged_layer_by_level[i]
            else:
                this_input_layer_object = skip_layer_by_level[i]

            if use_coord_conv:
                this_input_layer_object = coord_conv.add_spatial_coords_2d(
                    this_input_layer_object
                )

            skip_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_output_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object
            )(this_input_layer_object)

            skip_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(skip_layer_by_level[i])

            this_dropout_rate = skip_dropout_rates_by_level[i][j]

            if this_dropout_rate > 0:
                this_mc_flag = bool(skip_dropout_mc_flags_by_level[i][j])

                skip_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=this_dropout_rate
                )(skip_layer_by_level[i], training=this_mc_flag)

            if use_batch_normalization:
                skip_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer()(
                        skip_layer_by_level[i]
                    )
                )

        if i == 0 and include_penultimate_conv:
            if use_coord_conv:
                skip_layer_by_level[i] = coord_conv.add_spatial_coords_2d(
                    skip_layer_by_level[i]
                )

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

            if penultimate_conv_dropout_rate > 0:
                skip_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=penultimate_conv_dropout_rate
                )(
                    skip_layer_by_level[i],
                    training=penultimate_conv_dropout_mc_flag
                )

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

        if use_coord_conv:
            this_layer_object = coord_conv.add_spatial_coords_2d(
                this_layer_object
            )

        upconv_layer_by_level[i - 1] = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=2, num_kernel_columns=2,
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_filters=num_output_channels_by_level[i - 1],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_layer_object)

        if upconv_dropout_rate_by_level[i - 1] > 0:
            upconv_layer_by_level[i - 1] = architecture_utils.get_dropout_layer(
                dropout_fraction=upconv_dropout_rate_by_level[i - 1]
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

    if use_coord_conv:
        skip_layer_by_level[0] = coord_conv.add_spatial_coords_2d(
            skip_layer_by_level[0]
        )

    skip_layer_by_level[0] = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=1, num_kernel_columns=1,
        num_rows_per_stride=1, num_columns_per_stride=1, num_filters=1,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object
    )(skip_layer_by_level[0])

    if output_layer_dropout_rate > 0:
        skip_layer_by_level[0] = architecture_utils.get_dropout_layer(
            dropout_fraction=output_layer_dropout_rate
        )(
            skip_layer_by_level[0],
            training=output_layer_dropout_mc_flag
        )

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

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=metric_function_list
    )

    model_object.summary()
    return model_object
