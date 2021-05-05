"""Plotting methods for satellite data."""

import numpy
import matplotlib
matplotlib.use('agg')
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import error_checking
from ml4convection.plotting import satellite_plotting


def _grid_points_to_edges(grid_point_coords):
    """Converts grid-point coordinates to grid-cell-edge coordinates.

    P = number of grid points

    :param grid_point_coords: length-P numpy array of grid-point coordinates, in
        increasing order.
    :return: grid_cell_edge_coords: length-(P + 1) numpy array of grid-cell-edge
        coordinates, also in increasing order.
    """

    grid_cell_edge_coords = (grid_point_coords[:-1] + grid_point_coords[1:]) / 2
    first_edge_coords = (
        grid_point_coords[0] - numpy.diff(grid_point_coords[:2]) / 2
    )
    last_edge_coords = (
        grid_point_coords[-1] + numpy.diff(grid_point_coords[-2:]) / 2
    )

    return numpy.concatenate((
        first_edge_coords, grid_cell_edge_coords, last_edge_coords
    ))


def plot_2d_grid_regular(
        brightness_temp_matrix_kelvins, axes_object, latitudes_deg_n,
        longitudes_deg_e, cbar_orientation_string='vertical', font_size=30.):
    """Plots brightness temperature on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param brightness_temp_matrix_kelvins: M-by-N numpy array of brightness
        temperatures.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param latitudes_deg_n: length-M numpy array of grid-point latitudes (deg
        north).
    :param longitudes_deg_e: length-N numpy array of grid-point longitudes (deg
        east).
    :param cbar_orientation_string: Colour-bar orientation.  May be
        "horizontal", "vertical", or None.
    :param font_size: Font size.
    :return: colour_bar_object: Colour-bar handle (instance of
        `matplotlib.pyplot.colorbar`).
    """

    error_checking.assert_is_numpy_array(latitudes_deg_n, num_dimensions=1)
    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg_n)
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(latitudes_deg_n), 0.
    )

    error_checking.assert_is_numpy_array(longitudes_deg_e, num_dimensions=1)
    error_checking.assert_is_valid_lng_numpy_array(
        longitudes_deg_e, positive_in_west_flag=True
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(longitudes_deg_e), 0.
    )

    num_rows = len(latitudes_deg_n)
    num_columns = len(longitudes_deg_e)
    expected_dim = numpy.array([num_rows, num_columns], dtype=int)

    error_checking.assert_is_numpy_array(
        brightness_temp_matrix_kelvins, exact_dimensions=expected_dim
    )
    error_checking.assert_is_greater_numpy_array(
        brightness_temp_matrix_kelvins, 0., allow_nan=True
    )

    if cbar_orientation_string is not None:
        error_checking.assert_is_string(cbar_orientation_string)

    edge_latitudes_deg_n = _grid_points_to_edges(latitudes_deg_n)
    edge_longitudes_deg_e = _grid_points_to_edges(longitudes_deg_e)
    edge_temp_matrix_kelvins = grids.latlng_field_grid_points_to_edges(
        field_matrix=brightness_temp_matrix_kelvins,
        min_latitude_deg=1., min_longitude_deg=1.,
        lat_spacing_deg=1e-6, lng_spacing_deg=1e-6
    )[0]

    edge_temp_matrix_kelvins = numpy.ma.masked_where(
        numpy.isnan(edge_temp_matrix_kelvins), edge_temp_matrix_kelvins
    )
    colour_map_object, colour_norm_object = (
        satellite_plotting.get_colour_scheme()
    )

    if hasattr(colour_norm_object, 'boundaries'):
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]
    else:
        min_colour_value = colour_norm_object.vmin
        max_colour_value = colour_norm_object.vmax

    axes_object.pcolormesh(
        edge_longitudes_deg_e, edge_latitudes_deg_n, edge_temp_matrix_kelvins,
        cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None', zorder=-1e11
    )

    if cbar_orientation_string is None:
        return None

    return satellite_plotting.add_colour_bar(
        brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
        axes_object=axes_object, colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string=cbar_orientation_string, font_size=font_size
    )
