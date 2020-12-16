"""Methods for building Chiu nets.

https://doi.org/10.1109/LRA.2020.2992184
"""

import numpy
import keras
from keras import backend as K
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from ml4convection.machine_learning import neural_net

INPUT_DIMENSIONS_KEY = 'input_dimensions'
NUM_FC_CONV_LAYERS_KEY = 'num_conv_layers_in_fc_module'
FC_MODULE_DROPOUT_RATES_KEY = 'fc_module_dropout_rates'
USE_3D_CONV_IN_FC_KEY = 'use_3d_conv_in_fc_module'
NUM_LEVELS_KEY = 'num_levels'
CONV_LAYER_COUNTS_KEY = 'num_conv_layers_by_level'
CHANNEL_COUNTS_KEY = 'num_channels_by_level'
ENCODER_DROPOUT_RATES_KEY = 'encoder_dropout_rate_by_level'
DECODER_DROPOUT_RATES_KEY = 'decoder_dropout_rate_by_level'
SKIP_DROPOUT_RATES_KEY = 'skip_dropout_rate_by_level'
INNER_ACTIV_FUNCTION_KEY = 'inner_activ_function_name'
INNER_ACTIV_FUNCTION_ALPHA_KEY = 'inner_activ_function_alpha'
OUTPUT_ACTIV_FUNCTION_KEY = 'output_activ_function_name'
OUTPUT_ACTIV_FUNCTION_ALPHA_KEY = 'output_activ_function_alpha'
L1_WEIGHT_KEY = 'l1_weight'
L2_WEIGHT_KEY = 'l2_weight'
USE_BATCH_NORM_KEY = 'use_batch_normalization'

DEFAULT_ARCHITECTURE_OPTION_DICT = {
    NUM_FC_CONV_LAYERS_KEY: 1,
    FC_MODULE_DROPOUT_RATES_KEY: numpy.full(1, 0.),
    USE_3D_CONV_IN_FC_KEY: True,
    NUM_LEVELS_KEY: 4,
    CONV_LAYER_COUNTS_KEY: numpy.array([2, 2, 2, 4, 4], dtype=int),
    CHANNEL_COUNTS_KEY: numpy.array([16, 24, 32, 64, 128], dtype=int),
    ENCODER_DROPOUT_RATES_KEY: numpy.full(5, 0.),
    DECODER_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    SKIP_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    INNER_ACTIV_FUNCTION_KEY: architecture_utils.RELU_FUNCTION_STRING,
    INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    OUTPUT_ACTIV_FUNCTION_KEY: architecture_utils.SIGMOID_FUNCTION_STRING,
    OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    L1_WEIGHT_KEY: 0.,
    L2_WEIGHT_KEY: 0.001,
    USE_BATCH_NORM_KEY: True
}


def _check_args(option_dict):
    """Error-checks input arguments.

    L = number of levels in encoder = number of levels in decoder

    :param option_dict: Dictionary with the following keys.
    option_dict['input_dimensions']: numpy array with input dimensions
        (num_rows, num_columns, num_times, num_channels).
    option_dict['num_conv_layers_in_fc_module']: Number of conv layers in
        forecasting module.
    option_dict['fc_module_dropout_rates']: length-N numpy array of dropout rates
        in forecasting module, where N = 'num_conv_layers_in_fc_module'.
    option_dict['num_levels']: L in the above discussion.
    option_dict['num_conv_layers_by_level']: length-(L + 1) numpy array with
        number of conv layers at each level.
    option_dict['num_channels_by_level']: length-(L + 1) numpy array with number
        of channels at each level.
    option_dict['encoder_dropout_rate_by_level']: length-(L + 1) numpy array
        with dropout rate for conv layers in encoder at each level.
    option_dict['decoder_dropout_rate_by_level']: length-L numpy array
        with dropout rate for conv layers in decoder at each level.
    option_dict['skip_dropout_rate_by_level']: length-L numpy array with dropout
        rate for conv layer after skip connection at each level.
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
        batch normalization after each inner (non-output) conv layer.

    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_ARCHITECTURE_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    input_dimensions = option_dict[INPUT_DIMENSIONS_KEY]
    error_checking.assert_is_numpy_array(
        input_dimensions, exact_dimensions=numpy.array([4], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(input_dimensions)
    error_checking.assert_is_greater_numpy_array(input_dimensions, 0)

    num_conv_layers_in_fc_module = option_dict[NUM_FC_CONV_LAYERS_KEY]
    error_checking.assert_is_integer(num_conv_layers_in_fc_module)
    error_checking.assert_is_greater(num_conv_layers_in_fc_module, 0)

    expected_dim = numpy.array([num_conv_layers_in_fc_module], dtype=int)

    fc_module_dropout_rates = option_dict[FC_MODULE_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        fc_module_dropout_rates, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        fc_module_dropout_rates, 1., allow_nan=True
    )

    error_checking.assert_is_boolean(option_dict[USE_3D_CONV_IN_FC_KEY])

    num_levels = option_dict[NUM_LEVELS_KEY]
    error_checking.assert_is_integer(num_levels)
    error_checking.assert_is_geq(num_levels, 2)

    expected_dim = numpy.array([num_levels + 1], dtype=int)

    num_conv_layers_by_level = option_dict[CONV_LAYER_COUNTS_KEY]
    error_checking.assert_is_numpy_array(
        num_conv_layers_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(num_conv_layers_by_level)
    error_checking.assert_is_greater_numpy_array(num_conv_layers_by_level, 0)

    num_channels_by_level = option_dict[CHANNEL_COUNTS_KEY]
    error_checking.assert_is_numpy_array(
        num_channels_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(num_channels_by_level)
    error_checking.assert_is_greater_numpy_array(num_channels_by_level, 0)

    encoder_dropout_rate_by_level = option_dict[ENCODER_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        encoder_dropout_rate_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        encoder_dropout_rate_by_level, 1., allow_nan=True
    )

    expected_dim = numpy.array([num_levels], dtype=int)

    decoder_dropout_rate_by_level = option_dict[DECODER_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        decoder_dropout_rate_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        decoder_dropout_rate_by_level, 1., allow_nan=True
    )

    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
    error_checking.assert_is_numpy_array(
        skip_dropout_rate_by_level, exact_dimensions=expected_dim
    )
    error_checking.assert_is_leq_numpy_array(
        skip_dropout_rate_by_level, 1., allow_nan=True
    )

    error_checking.assert_is_geq(option_dict[L1_WEIGHT_KEY], 0.)
    error_checking.assert_is_geq(option_dict[L2_WEIGHT_KEY], 0.)
    error_checking.assert_is_boolean(option_dict[USE_BATCH_NORM_KEY])

    return option_dict


def _get_time_slicing_function(time_index):
    """Returns function that takes one time step from input tensor.

    :param time_index: Will take the [k]th time step, where k = `time_index`.
    :return: time_slicing_function: Function handle (see below).
    """

    def time_slicing_function(input_tensor_3d):
        """Takes one time step from the input tensor.

        :param input_tensor_3d: Input tensor with 3 spatiotemporal dimensions.
        :return: input_tensor_2d: Input tensor with 2 spatial dimensions.
        """

        return input_tensor_3d[..., time_index, :]

    return time_slicing_function


def create_model(option_dict, loss_function, mask_matrix):
    """Creates Chiu net.

    This method sets up the architecture, loss function, and optimizer -- and
    compiles the model -- but does not train it.

    Architecture based on: https://doi.org/10.1109/LRA.2020.2992184

    M = number of rows in grid
    N = number of columns in grid

    :param option_dict: See doc for `_check_args`.
    :param loss_function: Loss function.
    :param mask_matrix: M-by-N numpy array of Boolean flags.  Only pixels marked
        "True" are considered in the loss function and metrics.
    :return: model_object: Instance of `keras.models.Model`, with the
        aforementioned architecture.
    """

    metric_function_list = neural_net.get_metrics(mask_matrix)[0]
    option_dict = _check_args(option_dict)

    input_dimensions = option_dict[INPUT_DIMENSIONS_KEY]
    num_conv_layers_in_fc_module = option_dict[NUM_FC_CONV_LAYERS_KEY]
    fc_module_dropout_rates = option_dict[FC_MODULE_DROPOUT_RATES_KEY]
    use_3d_conv_in_fc_module = option_dict[USE_3D_CONV_IN_FC_KEY]
    num_levels = option_dict[NUM_LEVELS_KEY]
    num_conv_layers_by_level = option_dict[CONV_LAYER_COUNTS_KEY]
    num_channels_by_level = option_dict[CHANNEL_COUNTS_KEY]
    encoder_dropout_rate_by_level = option_dict[ENCODER_DROPOUT_RATES_KEY]
    decoder_dropout_rate_by_level = option_dict[DECODER_DROPOUT_RATES_KEY]
    skip_dropout_rate_by_level = option_dict[SKIP_DROPOUT_RATES_KEY]
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

    num_input_times = input_dimensions[2]
    last_conv_layer_matrix = numpy.full(
        (num_input_times, num_levels + 1), '', dtype=object
    )
    pooling_layer_matrix = numpy.full(
        (num_input_times, num_levels), '', dtype=object
    )

    for k in range(num_input_times):
        for i in range(num_levels + 1):
            for j in range(num_conv_layers_by_level[i]):
                if j == 0:
                    if i == 0:
                        this_function = _get_time_slicing_function(k)
                        this_name = 'time_slice_{0:d}'.format(k)

                        this_input_layer_object = keras.layers.Lambda(
                            this_function, name=this_name
                        )(input_layer_object)
                    else:
                        this_input_layer_object = pooling_layer_matrix[k, i - 1]
                else:
                    this_input_layer_object = last_conv_layer_matrix[k, i]

                this_name = 'time{0:d}_level{1:d}_conv{2:d}'.format(k, i, j)

                last_conv_layer_matrix[k, i] = (
                    architecture_utils.get_2d_conv_layer(
                        num_kernel_rows=3, num_kernel_columns=3,
                        num_rows_per_stride=1, num_columns_per_stride=1,
                        num_filters=num_channels_by_level[i],
                        padding_type_string=
                        architecture_utils.YES_PADDING_STRING,
                        weight_regularizer=regularizer_object,
                        layer_name=this_name
                    )(this_input_layer_object)
                )

                this_name = 'time{0:d}_level{1:d}_conv{2:d}_activation'.format(
                    k, i, j
                )

                last_conv_layer_matrix[k, i] = (
                    architecture_utils.get_activation_layer(
                        activation_function_string=inner_activ_function_name,
                        alpha_for_relu=inner_activ_function_alpha,
                        alpha_for_elu=inner_activ_function_alpha,
                        layer_name=this_name
                    )(last_conv_layer_matrix[k, i])
                )

                if encoder_dropout_rate_by_level[i] > 0:
                    this_name = 'time{0:d}_level{1:d}_conv{2:d}_dropout'.format(
                        k, i, j
                    )

                    last_conv_layer_matrix[k, i] = (
                        architecture_utils.get_dropout_layer(
                            dropout_fraction=encoder_dropout_rate_by_level[i],
                            layer_name=this_name
                        )(last_conv_layer_matrix[k, i])
                    )

                if use_batch_normalization:
                    this_name = 'time{0:d}_level{1:d}_conv{2:d}_bn'.format(
                        k, i, j
                    )

                    last_conv_layer_matrix[k, i] = (
                        architecture_utils.get_batch_norm_layer(
                            layer_name=this_name
                        )(last_conv_layer_matrix[k, i])
                    )

            if i == num_levels:
                break

            this_name = 'time{0:d}_level{1:d}_pooling'.format(k, i)

            pooling_layer_matrix[k, i] = (
                architecture_utils.get_2d_pooling_layer(
                    num_rows_in_window=2, num_columns_in_window=2,
                    num_rows_per_stride=2, num_columns_per_stride=2,
                    pooling_type_string=architecture_utils.MAX_POOLING_STRING,
                    layer_name=this_name
                )(last_conv_layer_matrix[k, i])
            )

        this_name = 'time{0:d}_add-time-dim'.format(k)

        last_conv_layer_matrix[k, -1] = keras.layers.Lambda(
            lambda x: K.expand_dims(x, axis=-2), name=this_name
        )(last_conv_layer_matrix[k, -1])

    fc_module_layer_object = keras.layers.Concatenate(
        axis=-2, name='concat_times'
    )(last_conv_layer_matrix[:, -1].tolist())

    if not use_3d_conv_in_fc_module:
        orig_shape = fc_module_layer_object.get_shape()
        shape_sans_time = orig_shape[1:-2] + [orig_shape[-2] * orig_shape[-1]]

        fc_module_layer_object = keras.layers.Reshape(
            target_shape=shape_sans_time, name='fc_module_remove-time-dim'
        )(fc_module_layer_object)

    for j in range(num_conv_layers_in_fc_module):
        this_name = 'fc_module_conv{0:d}'.format(j)

        if use_3d_conv_in_fc_module:
            if j == 0:
                fc_module_layer_object = architecture_utils.get_3d_conv_layer(
                    num_kernel_rows=1, num_kernel_columns=1,
                    num_kernel_heights=num_input_times,
                    num_rows_per_stride=1, num_columns_per_stride=1,
                    num_heights_per_stride=1,
                    num_filters=num_channels_by_level[-1],
                    padding_type_string=architecture_utils.NO_PADDING_STRING,
                    weight_regularizer=regularizer_object,
                    layer_name=this_name
                )(fc_module_layer_object)

                shape_sans_time = (
                    fc_module_layer_object.shape[1:3] +
                    [fc_module_layer_object.shape[-1]]
                )
                fc_module_layer_object = keras.layers.Reshape(
                    target_shape=shape_sans_time,
                    name='fc_module_remove-time-dim'
                )(fc_module_layer_object)
            else:
                fc_module_layer_object = architecture_utils.get_2d_conv_layer(
                    num_kernel_rows=3, num_kernel_columns=3,
                    num_rows_per_stride=1, num_columns_per_stride=1,
                    num_filters=num_channels_by_level[-1],
                    padding_type_string=architecture_utils.YES_PADDING_STRING,
                    weight_regularizer=regularizer_object,
                    layer_name=this_name
                )(fc_module_layer_object)
        else:
            fc_module_layer_object = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_level[-1],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=this_name
            )(fc_module_layer_object)

        this_name = 'fc_module_conv{0:d}_activation'.format(j)

        fc_module_layer_object = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha,
            layer_name=this_name
        )(fc_module_layer_object)

        if fc_module_dropout_rates[j] > 0:
            this_name = 'fc_module_conv{0:d}_dropout'.format(j)

            fc_module_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=fc_module_dropout_rates[j],
                layer_name=this_name
            )(fc_module_layer_object)

        if use_batch_normalization:
            this_name = 'fc_module_conv{0:d}_bn'.format(j)

            fc_module_layer_object = architecture_utils.get_batch_norm_layer(
                layer_name=this_name
            )(fc_module_layer_object)

    upconv_layer_by_level = [None] * num_levels
    skip_layer_by_level = [None] * num_levels
    merged_layer_by_level = [None] * num_levels

    this_name = 'upsampling_level{0:d}'.format(num_levels - 1)

    try:
        this_layer_object = keras.layers.UpSampling2D(
            size=(2, 2), interpolation='bilinear', name=this_name
        )(fc_module_layer_object)
    except:
        this_layer_object = keras.layers.UpSampling2D(
            size=(2, 2), name=this_name
        )(fc_module_layer_object)

    this_name = 'upsampling_level{0:d}_conv'.format(num_levels - 1)

    i = num_levels - 1
    upconv_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=2, num_kernel_columns=2,
        num_rows_per_stride=1, num_columns_per_stride=1,
        num_filters=num_channels_by_level[i],
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object,
        layer_name=this_name
    )(this_layer_object)

    this_name = 'upsampling_level{0:d}_activation'.format(num_levels - 1)

    upconv_layer_by_level[i] = architecture_utils.get_activation_layer(
        activation_function_string=inner_activ_function_name,
        alpha_for_relu=inner_activ_function_alpha,
        alpha_for_elu=inner_activ_function_alpha,
        layer_name=this_name
    )(upconv_layer_by_level[i])

    if decoder_dropout_rate_by_level[i] > 0:
        this_name = 'upsampling_level{0:d}_dropout'.format(i)

        upconv_layer_by_level[i] = architecture_utils.get_dropout_layer(
            dropout_fraction=decoder_dropout_rate_by_level[i],
            layer_name=this_name
        )(upconv_layer_by_level[i])

    num_upconv_rows = upconv_layer_by_level[i].get_shape()[1]
    num_desired_rows = last_conv_layer_matrix[-1, i].get_shape()[1]
    num_padding_rows = num_desired_rows - num_upconv_rows

    num_upconv_columns = upconv_layer_by_level[i].get_shape()[2]
    num_desired_columns = last_conv_layer_matrix[-1, i].get_shape()[2]
    num_padding_columns = num_desired_columns - num_upconv_columns

    if num_padding_rows + num_padding_columns > 0:
        padding_arg = ((0, num_padding_rows), (0, num_padding_columns))
        this_name = 'padding_level{0:d}'.format(i)

        upconv_layer_by_level[i] = keras.layers.ZeroPadding2D(
            padding=padding_arg, name=this_name
        )(upconv_layer_by_level[i])

    this_name = 'skip_level{0:d}'.format(i)

    merged_layer_by_level[i] = keras.layers.Concatenate(
        axis=-1, name=this_name
    )(
        [last_conv_layer_matrix[-1, i], upconv_layer_by_level[i]]
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

            this_name = 'skip_level{0:d}_conv{1:d}'.format(i, j)

            skip_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1,
                num_filters=num_channels_by_level[i],
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name=this_name
            )(this_input_layer_object)

            this_name = 'skip_level{0:d}_conv{1:d}_activation'.format(i, j)

            skip_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name=this_name
            )(skip_layer_by_level[i])

            if skip_dropout_rate_by_level[i] > 0:
                this_name = 'skip_level{0:d}_conv{1:d}_dropout'.format(i, j)

                skip_layer_by_level[i] = architecture_utils.get_dropout_layer(
                    dropout_fraction=skip_dropout_rate_by_level[i],
                    layer_name=this_name
                )(skip_layer_by_level[i])

            if use_batch_normalization:
                this_name = 'skip_level{0:d}_conv{1:d}_bn'.format(i, j)

                skip_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name=this_name
                    )(skip_layer_by_level[i])
                )

        if i == 0:
            skip_layer_by_level[i] = architecture_utils.get_2d_conv_layer(
                num_kernel_rows=3, num_kernel_columns=3,
                num_rows_per_stride=1, num_columns_per_stride=1, num_filters=2,
                padding_type_string=architecture_utils.YES_PADDING_STRING,
                weight_regularizer=regularizer_object,
                layer_name='penultimate_conv'
            )(skip_layer_by_level[i])

            skip_layer_by_level[i] = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha,
                layer_name='penultimate_conv_activation'
            )(skip_layer_by_level[i])

            if use_batch_normalization:
                skip_layer_by_level[i] = (
                    architecture_utils.get_batch_norm_layer(
                        layer_name='penultimate_conv_bn'
                    )(skip_layer_by_level[i])
                )

            break

        this_name = 'upsampling_level{0:d}'.format(i - 1)

        try:
            this_layer_object = keras.layers.UpSampling2D(
                size=(2, 2), interpolation='bilinear', name=this_name
            )(skip_layer_by_level[i])
        except:
            this_layer_object = keras.layers.UpSampling2D(
                size=(2, 2), name=this_name
            )(skip_layer_by_level[i])

        this_name = 'upsampling_level{0:d}_conv'.format(i - 1)

        upconv_layer_by_level[i - 1] = architecture_utils.get_2d_conv_layer(
            num_kernel_rows=2, num_kernel_columns=2,
            num_rows_per_stride=1, num_columns_per_stride=1,
            num_filters=num_channels_by_level[i - 1],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object,
            layer_name=this_name
        )(this_layer_object)

        this_name = 'upsampling_level{0:d}_activation'.format(i - 1)

        upconv_layer_by_level[i - 1] = architecture_utils.get_activation_layer(
            activation_function_string=inner_activ_function_name,
            alpha_for_relu=inner_activ_function_alpha,
            alpha_for_elu=inner_activ_function_alpha,
            layer_name=this_name
        )(upconv_layer_by_level[i - 1])

        if decoder_dropout_rate_by_level[i - 1] > 0:
            this_name = 'upsampling_level{0:d}_dropout'.format(i - 1)

            upconv_layer_by_level[i - 1] = architecture_utils.get_dropout_layer(
                dropout_fraction=decoder_dropout_rate_by_level[i - 1],
                layer_name=this_name
            )(upconv_layer_by_level[i - 1])

        num_upconv_rows = upconv_layer_by_level[i - 1].get_shape()[1]
        num_desired_rows = last_conv_layer_matrix[-1, i - 1].get_shape()[1]
        num_padding_rows = num_desired_rows - num_upconv_rows

        num_upconv_columns = upconv_layer_by_level[i - 1].get_shape()[2]
        num_desired_columns = last_conv_layer_matrix[-1, i - 1].get_shape()[2]
        num_padding_columns = num_desired_columns - num_upconv_columns

        if num_padding_rows + num_padding_columns > 0:
            padding_arg = ((0, num_padding_rows), (0, num_padding_columns))
            this_name = 'padding_level{0:d}'.format(i - 1)

            upconv_layer_by_level[i - 1] = keras.layers.ZeroPadding2D(
                padding=padding_arg, name=this_name
            )(upconv_layer_by_level[i - 1])

        this_name = 'skip_level{0:d}'.format(i - 1)

        merged_layer_by_level[i - 1] = keras.layers.Concatenate(
            axis=-1, name=this_name
        )(
            [last_conv_layer_matrix[-1, i - 1], upconv_layer_by_level[i - 1]]
        )

    skip_layer_by_level[0] = architecture_utils.get_2d_conv_layer(
        num_kernel_rows=1, num_kernel_columns=1,
        num_rows_per_stride=1, num_columns_per_stride=1, num_filters=1,
        padding_type_string=architecture_utils.YES_PADDING_STRING,
        weight_regularizer=regularizer_object,
        layer_name='last_conv'
    )(skip_layer_by_level[0])

    skip_layer_by_level[0] = architecture_utils.get_activation_layer(
        activation_function_string=output_activ_function_name,
        alpha_for_relu=output_activ_function_alpha,
        alpha_for_elu=output_activ_function_alpha,
        layer_name='last_conv_activation'
    )(skip_layer_by_level[0])

    if mask_matrix is not None:
        this_matrix = numpy.expand_dims(
            mask_matrix.astype(float), axis=(0, -1)
        )
        skip_layer_by_level[0] = keras.layers.Multiply(name='mask')(
            [this_matrix, skip_layer_by_level[0]]
        )

    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=skip_layer_by_level[0]
    )

    model_object.compile(
        loss=loss_function, optimizer=keras.optimizers.Adam(),
        metrics=metric_function_list
    )

    model_object.summary()
    return model_object
