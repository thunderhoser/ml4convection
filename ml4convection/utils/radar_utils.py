"""Helper methods for radar data."""

import numpy
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking

RADAR_LATITUDES_DEG_N = numpy.array([25.07, 23.989, 23.1467, 21.899])
RADAR_LONGITUDES_DEG_E = numpy.array([121.77, 121.619, 120.086, 120.849])


def radar_sites_to_grid_points(grid_latitudes_deg_n, grid_longitudes_deg_e):
    """For each radar site, finds nearest grid point.

    M = number of rows in grid
    N = number of columns in grid
    R = number of radar sites

    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg E).
    :return: row_indices: length-R numpy array of grid rows.
    :return: column_indices: length-R numpy array of grid columns.
    """

    # Basic input-checking.
    error_checking.assert_is_numpy_array(grid_latitudes_deg_n, num_dimensions=1)
    error_checking.assert_is_valid_lat_numpy_array(grid_latitudes_deg_n)
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(grid_latitudes_deg_n), 0.
    )

    error_checking.assert_is_numpy_array(
        grid_longitudes_deg_e, num_dimensions=1
    )
    grid_longitudes_deg_e = (
        lng_conversion.convert_lng_positive_in_west(grid_longitudes_deg_e)
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(grid_longitudes_deg_e), 0.
    )

    # Make sure that all radars are inside the domain.
    error_checking.assert_is_greater(
        numpy.min(RADAR_LATITUDES_DEG_N), grid_latitudes_deg_n[0]
    )
    error_checking.assert_is_less_than(
        numpy.max(RADAR_LATITUDES_DEG_N), grid_latitudes_deg_n[-1]
    )
    error_checking.assert_is_greater(
        numpy.min(RADAR_LONGITUDES_DEG_E), grid_longitudes_deg_e[0]
    )
    error_checking.assert_is_less_than(
        numpy.max(RADAR_LONGITUDES_DEG_E), grid_longitudes_deg_e[-1]
    )

    row_indices = numpy.array([
        numpy.argmin(numpy.absolute(l - grid_latitudes_deg_n))
        for l in RADAR_LATITUDES_DEG_N
    ], dtype=int)

    column_indices = numpy.array([
        numpy.argmin(numpy.absolute(l - grid_longitudes_deg_e))
        for l in RADAR_LONGITUDES_DEG_E
    ], dtype=int)

    return row_indices, column_indices
