"""Implements CoordConv solution from Liu et al. (2018).

https://arxiv.org/abs/1807.03247
"""

from keras import backend as K
from gewittergefahr.gg_utils import error_checking


def add_spatial_coords_2d(input_tensor):
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
