"""Implements CoordConv solution from Liu et al. (2018).

https://arxiv.org/abs/1807.03247
"""

import numpy
import keras.layers
from keras import backend as K
from gewittergefahr.gg_utils import error_checking


def add_spatial_coords_2d_new(input_layer_object):
    """Adds spatial coords to layer with two spatial dimensions.

    M = number of rows in grid
    N = number of columns in grid
    C = number of input channels

    :param input_layer_object: Input layer (instance of `keras.layers.Layer`).
    :return: output_layer_object: Same but with two extra output channels, one
        for each spatial coordinate.
    """

    input_shape = K.shape(input_layer_object)
    input_shape = [input_shape[i] for i in range(4)]
    batch_shape, dim1, dim2, channels = input_shape

    xx_ones = K.ones(K.stack([batch_shape, dim2]), dtype='int32')
    xx_ones = K.expand_dims(xx_ones, axis=-1)

    xx_range = K.tile(K.expand_dims(K.arange(0, dim1), axis=0),
                      K.stack([batch_shape, 1]))
    xx_range = K.expand_dims(xx_range, axis=1)
    xx_channels = K.batch_dot(xx_ones, xx_range, axes=[2, 1])
    xx_channels = K.expand_dims(xx_channels, axis=-1)
    xx_channels = K.permute_dimensions(xx_channels, [0, 2, 1, 3])

    yy_ones = K.ones(K.stack([batch_shape, dim1]), dtype='int32')
    yy_ones = K.expand_dims(yy_ones, axis=1)

    yy_range = K.tile(K.expand_dims(K.arange(0, dim2), axis=0),
                      K.stack([batch_shape, 1]))
    yy_range = K.expand_dims(yy_range, axis=-1)

    yy_channels = K.batch_dot(yy_range, yy_ones, axes=[2, 1])
    yy_channels = K.expand_dims(yy_channels, axis=-1)
    yy_channels = K.permute_dimensions(yy_channels, [0, 2, 1, 3])

    xx_channels = K.cast(xx_channels, K.floatx())
    xx_channels = xx_channels / K.cast(dim1 - 1, K.floatx())
    xx_channels = (xx_channels * 2) - 1.

    yy_channels = K.cast(yy_channels, K.floatx())
    yy_channels = yy_channels / K.cast(dim2 - 1, K.floatx())
    yy_channels = (yy_channels * 2) - 1.

    print(K.shape(yy_channels))

    return K.concatenate([input_layer_object, xx_channels, yy_channels], axis=-1)


def add_spatial_coords_2d(input_layer_object):
    """Adds spatial coords to layer with two spatial dimensions.

    M = number of rows in grid
    N = number of columns in grid
    C = number of input channels


    :param input_layer_object: Input layer (instance of `keras.layers.Layer`).
    :return: output_layer_object: Same but with two extra output channels, one
        for each spatial coordinate.
    """

    input_dimensions = numpy.array(
        input_layer_object.output.get_shape().as_list()[1:], dtype=int
    )

    error_checking.assert_is_geq(len(input_dimensions), 3)
    error_checking.assert_is_leq(len(input_dimensions), 4)

    has_time_dim = len(input_dimensions) == 4

    num_grid_rows = input_dimensions[0]
    num_grid_columns = input_dimensions[1]

    y_coords = K.arange(0, num_grid_rows, dtype=float)
    y_coords = y_coords - K.mean(y_coords)
    y_coords = y_coords / K.max(K.abs(y_coords))

    y_coord_tensor = K.expand_dims(y_coords, axis=-1)
    y_coord_tensor = K.repeat_elements(
        y_coord_tensor, rep=num_grid_columns, axis=-1
    )

    if has_time_dim:
        num_times = input_dimensions[-2]

        y_coord_tensor = K.expand_dims(y_coord_tensor, axis=-1)
        y_coord_tensor = K.repeat_elements(
            y_coord_tensor, rep=num_times, axis=-1
        )

    # Add batch and channel dimensions.
    y_coord_tensor = K.expand_dims(y_coord_tensor, axis=0)
    y_coord_tensor = K.expand_dims(y_coord_tensor, axis=-1)

    x_coords = K.arange(0, num_grid_columns, dtype=float)
    x_coords = x_coords - K.mean(x_coords)
    x_coords = x_coords / K.max(K.abs(x_coords))

    x_coord_tensor = K.expand_dims(x_coords, axis=0)
    x_coord_tensor = K.repeat_elements(
        x_coord_tensor, rep=num_grid_rows, axis=0
    )

    if has_time_dim:
        num_times = input_dimensions[-2]

        x_coord_tensor = K.expand_dims(x_coord_tensor, axis=-1)
        x_coord_tensor = K.repeat_elements(
            x_coord_tensor, rep=num_times, axis=-1
        )

    # Add batch and channel dimensions.
    x_coord_tensor = K.expand_dims(x_coord_tensor, axis=0)
    x_coord_tensor = K.expand_dims(x_coord_tensor, axis=-1)

    return keras.layers.concatenate(
        [input_tensor, y_coord_tensor, x_coord_tensor], axis=-1
    )


def add_spatial_coords_2d_old(input_tensor):
    """Adds spatial coords to tensor with two spatial dimensions.

    M = number of rows in grid
    N = number of columns in grid
    C = number of input channels

    :param input_tensor: Input tensor.  Second dimension must have length M
        (representing grid rows); third dimension must have length N
        (representing grid columns); and last dimension must have length C
        (representing channels).
    :return: output_tensor: Same as `input_tensor`, except that last dimension
        has length C + 2.
    """

    input_dimensions = K.shape(input_tensor)
    error_checking.assert_is_geq(len(input_dimensions), 4)
    error_checking.assert_is_leq(len(input_dimensions), 5)

    has_time_dim = len(input_dimensions) == 5

    num_examples = input_dimensions[0]
    num_grid_rows = input_dimensions[1]
    num_grid_columns = input_dimensions[2]

    y_coords = K.arange(0, num_grid_rows, dtype=K.dtype(input_tensor))
    y_coords = y_coords - K.mean(y_coords)
    y_coords = y_coords / K.max(K.abs(y_coords))

    y_coord_tensor = K.expand_dims(y_coords, axis=-1)
    y_coord_tensor = K.repeat_elements(
        y_coord_tensor, rep=num_grid_columns, axis=-1
    )

    y_coord_tensor = K.expand_dims(y_coord_tensor, axis=0)
    y_coord_tensor = K.repeat_elements(
        y_coord_tensor, rep=num_examples, axis=0
    )

    if has_time_dim:
        num_times = input_dimensions[-2]

        y_coord_tensor = K.expand_dims(y_coord_tensor, axis=-1)
        y_coord_tensor = K.repeat_elements(
            y_coord_tensor, rep=num_times, axis=-1
        )

    y_coord_tensor = K.expand_dims(y_coord_tensor, axis=-1)

    x_coords = K.arange(0, num_grid_columns, dtype=K.dtype(input_tensor))
    x_coords = x_coords - K.mean(x_coords)
    x_coords = x_coords / K.max(K.abs(x_coords))

    x_coord_tensor = K.expand_dims(x_coords, axis=0)
    x_coord_tensor = K.repeat_elements(
        x_coord_tensor, rep=num_grid_rows, axis=0
    )

    x_coord_tensor = K.expand_dims(x_coord_tensor, axis=0)
    x_coord_tensor = K.repeat_elements(
        x_coord_tensor, rep=num_examples, axis=0
    )

    if has_time_dim:
        num_times = input_dimensions[-2]

        x_coord_tensor = K.expand_dims(x_coord_tensor, axis=-1)
        x_coord_tensor = K.repeat_elements(
            x_coord_tensor, rep=num_times, axis=-1
        )

    x_coord_tensor = K.expand_dims(x_coord_tensor, axis=-1)

    return K.concatenate(
        [input_tensor, y_coord_tensor, x_coord_tensor], axis=-1
    )
