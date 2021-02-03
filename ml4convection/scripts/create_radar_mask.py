"""Creates mask for radar data.

This mask censors out regions with poor radar coverage.
"""

import argparse
import numpy
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import error_checking
from ml4convection.io import radar_io
from ml4convection.io import twb_radar_io
from ml4convection.utils import radar_utils

METRES_TO_KM = 0.001

MAX_DISTANCE_ARG_NAME = 'max_distance_metres'
OMIT_NORTH_RADAR_ARG_NAME = 'omit_north_radar'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MAX_DISTANCE_HELP_STRING = (
    'Max distance from nearest radar (d_nr).  Grid points with higher d_nr will'
    ' be masked out.'
)
OMIT_NORTH_RADAR_HELP_STRING = (
    'Boolean flag.  If 1, distances to the northernmost radar will not be '
    'calculated.  In other words, d_nr will include only the other radars.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by `radar_io.write_mask_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_DISTANCE_ARG_NAME, type=float, required=False, default=1e5,
    help=MAX_DISTANCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OMIT_NORTH_RADAR_ARG_NAME, type=int, required=False, default=1,
    help=OMIT_NORTH_RADAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(max_distance_metres, omit_north_radar, output_file_name):
    """Creates mask for radar data.

    This is effectively the main method.

    :param max_distance_metres: See documentation at top of file.
    :param omit_north_radar: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_greater(max_distance_metres, 0.)

    grid_latitudes_deg_n = twb_radar_io.GRID_LATITUDES_DEG_N
    grid_longitudes_deg_e = twb_radar_io.GRID_LONGITUDES_DEG_E
    radar_latitudes_deg_n = radar_utils.RADAR_LATITUDES_DEG_N
    radar_longitudes_deg_e = radar_utils.RADAR_LONGITUDES_DEG_E

    if omit_north_radar:
        radar_latitudes_deg_n = radar_latitudes_deg_n[1:]
        radar_longitudes_deg_e = radar_longitudes_deg_e[1:]

    num_rows = len(grid_latitudes_deg_n)
    num_columns = len(grid_longitudes_deg_e)
    num_radars = len(radar_latitudes_deg_n)
    mask_matrix = numpy.full((num_rows, num_columns), 0, dtype=bool)

    for k in range(num_radars):
        print((
            'Finding grid points within {0:.1f}-km radius of {1:d}th radar...'
        ).format(
            METRES_TO_KM * max_distance_metres,
            k + int(omit_north_radar)
        ))

        these_rows, these_columns = grids.get_latlng_grid_points_in_radius(
            test_latitude_deg=radar_latitudes_deg_n[k],
            test_longitude_deg=radar_longitudes_deg_e[k],
            effective_radius_metres=max_distance_metres,
            grid_point_latitudes_deg=grid_latitudes_deg_n,
            grid_point_longitudes_deg=grid_longitudes_deg_e
        )[:2]

        mask_matrix[these_rows, these_columns] = True

    print('Writing mask to: "{0:s}"...'.format(output_file_name))
    radar_io.write_mask_file(
        netcdf_file_name=output_file_name, mask_matrix=mask_matrix,
        latitudes_deg_n=grid_latitudes_deg_n,
        longitudes_deg_e=grid_longitudes_deg_e,
        max_distance_metres=max_distance_metres,
        omit_north_radar=omit_north_radar
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        max_distance_metres=getattr(INPUT_ARG_OBJECT, MAX_DISTANCE_ARG_NAME),
        omit_north_radar=bool(getattr(
            INPUT_ARG_OBJECT, OMIT_NORTH_RADAR_ARG_NAME
        )),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
