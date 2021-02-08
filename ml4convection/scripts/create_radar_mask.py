"""Creates mask for radar data.

This mask censors out regions with poor radar coverage.
"""

import copy
import argparse
import numpy
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import error_checking
from ml4convection.io import radar_io
from ml4convection.io import twb_radar_io
from ml4convection.io import example_io
from ml4convection.utils import radar_utils

METRES_TO_KM = 0.001
HALF_PARTIAL_GRID_SIZE_PX = 102

MAX_DISTANCE_ARG_NAME = 'max_distance_metres'
OMIT_NORTH_RADAR_ARG_NAME = 'omit_north_radar'
FULL_MASK_FILE_ARG_NAME = 'output_full_mask_file_name'
PARTIAL_MASK_FILE_ARG_NAME = 'output_partial_mask_file_name'

MAX_DISTANCE_HELP_STRING = (
    'Max distance from nearest radar (d_nr).  Grid points with higher d_nr will'
    ' be masked out.'
)
OMIT_NORTH_RADAR_HELP_STRING = (
    'Boolean flag.  If 1, distances to the northernmost radar will not be '
    'calculated.  In other words, d_nr will include only the other radars.'
)
FULL_MASK_FILE_HELP_STRING = (
    'Path to full-grid mask file.  Will be written by '
    '`radar_io.write_mask_file`.'
)
PARTIAL_MASK_FILE_HELP_STRING = (
    'Path to partial-grid mask file.  Will be written by '
    '`radar_io.write_mask_file`.'
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
    '--' + FULL_MASK_FILE_ARG_NAME, type=str, required=True,
    help=FULL_MASK_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PARTIAL_MASK_FILE_ARG_NAME, type=str, required=True,
    help=PARTIAL_MASK_FILE_HELP_STRING
)


def _run(max_distance_metres, omit_north_radar, full_mask_file_name,
         partial_mask_file_name):
    """Creates mask for radar data.

    This is effectively the main method.

    :param max_distance_metres: See documentation at top of file.
    :param omit_north_radar: Same.
    :param full_mask_file_name: Same.
    :param partial_mask_file_name: Same.
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
    full_mask_matrix = numpy.full((num_rows, num_columns), 0, dtype=bool)
    partial_mask_matrix = numpy.full((num_rows, num_columns), 0, dtype=bool)

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

        full_mask_matrix[these_rows, these_columns] = True

        if k != num_radars - 1:
            continue

        partial_mask_matrix[these_rows, these_columns] = True

    print('Writing full mask to: "{0:s}"...'.format(full_mask_file_name))
    radar_io.write_mask_file(
        netcdf_file_name=full_mask_file_name, mask_matrix=full_mask_matrix,
        latitudes_deg_n=grid_latitudes_deg_n,
        longitudes_deg_e=grid_longitudes_deg_e,
        max_distance_metres=max_distance_metres,
        omit_north_radar=omit_north_radar
    )

    center_row_indices, center_column_indices = (
        radar_utils.radar_sites_to_grid_points(
            grid_latitudes_deg_n=grid_latitudes_deg_n,
            grid_longitudes_deg_e=grid_longitudes_deg_e
        )
    )
    center_row_index = center_row_indices[-1]
    center_column_index = center_column_indices[-1]

    dummy_target_dict = {
        example_io.LATITUDES_KEY: grid_latitudes_deg_n,
        example_io.LONGITUDES_KEY: grid_longitudes_deg_e,
        example_io.TARGET_MATRIX_KEY:
            numpy.expand_dims(partial_mask_matrix, axis=0).astype(int),
        example_io.MASK_MATRIX_KEY: partial_mask_matrix
    }

    dummy_target_dict = example_io.subset_grid(
        predictor_or_target_dict=copy.deepcopy(dummy_target_dict),
        first_row=center_row_index - HALF_PARTIAL_GRID_SIZE_PX,
        last_row=center_row_index + HALF_PARTIAL_GRID_SIZE_PX,
        first_column=center_column_index - HALF_PARTIAL_GRID_SIZE_PX,
        last_column=center_column_index + HALF_PARTIAL_GRID_SIZE_PX,
    )

    print('Writing partial mask to: "{0:s}"...'.format(partial_mask_file_name))
    radar_io.write_mask_file(
        netcdf_file_name=partial_mask_file_name,
        mask_matrix=dummy_target_dict[example_io.MASK_MATRIX_KEY],
        latitudes_deg_n=dummy_target_dict[example_io.LATITUDES_KEY],
        longitudes_deg_e=dummy_target_dict[example_io.LONGITUDES_KEY],
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
        full_mask_file_name=getattr(INPUT_ARG_OBJECT, FULL_MASK_FILE_ARG_NAME),
        partial_mask_file_name=getattr(
            INPUT_ARG_OBJECT, PARTIAL_MASK_FILE_ARG_NAME
        )
    )
