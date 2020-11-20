"""Runs preliminary storm-tracking."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import projections
from gewittergefahr.gg_utils import gridrad_utils
from gewittergefahr.gg_utils import echo_top_tracking
from gewittergefahr.gg_utils import temporal_tracking
from gewittergefahr.gg_utils import radar_utils as gg_radar_utils
from gewittergefahr.gg_utils import error_checking
from ml4convection.io import radar_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DATE_FORMAT = radar_io.DATE_FORMAT
TIME_FORMAT_FOR_MESSAGES = '%Y-%m-%d-%H%M'

MAX_LINK_TIME_SECONDS = 600
CENTRAL_PROJ_LATITUDE_DEG = 23.5
CENTRAL_PROJ_LONGITUDE_DEG = 120.75

REFLECTIVITY_DIR_ARG_NAME = 'input_refl_dir_name'
ECHO_CLASSIFN_DIR_ARG_NAME = 'input_echo_classifn_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
CRITICAL_REFL_ARG_NAME = 'critical_refl_dbz'
MIN_ECHO_TOP_ARG_NAME = 'min_echo_top_m_asl'
MIN_SIZE_ARG_NAME = 'min_size_pixels'
MIN_INTERMAX_DISTANCE_ARG_NAME = 'min_intermax_distance_metres'
MAX_VELOCITY_DIFF_ARG_NAME = 'max_velocity_diff_m_s01'
MAX_LINK_DISTANCE_ARG_NAME = 'max_link_distance_m_s01'
OUTPUT_DIR_ARG_NAME = 'output_tracking_dir_name'

REFLECTIVITY_DIR_HELP_STRING = (
    'Name of top-level directory with reflectivity files.  Files therein will '
    'be found by `radar_io.find_file` and read by '
    '`radar_io.read_reflectivity_file`.'
)
ECHO_CLASSIFN_DIR_HELP_STRING = (
    'Name of top-level directory with echo-classification files.  Files therein'
    ' will be found by `radar_io.find_file` and read by '
    '`radar_io.read_echo_classifn_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  This script will continuously track storms for '
    'the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

CRITICAL_REFL_HELP_STRING = (
    'Critical reflectivity (used to define echo top).  For example, if this is '
    '40, the tracking field will be 40-dBZ echo top.'
)
MIN_ECHO_TOP_HELP_STRING = (
    'Minimum echo top (metres above sea level).  Lower echo tops will not be '
    'considered convective.'
)
MIN_SIZE_HELP_STRING = (
    'Minimum size.  Smaller connected regions of high echo tops will not be '
    'considered convective.'
)
MIN_INTERMAX_DISTANCE_HELP_STRING = (
    'Minimum distance between any pair of local maxima (storm objects) at the '
    'same time.  See `echo_top_tracking._remove_redundant_local_maxima` for '
    'details.'
)
MAX_VELOCITY_DIFF_HELP_STRING = (
    'Used to connect local maxima (storm objects) between times.  See '
    '`echo_top_tracking._link_local_maxima_in_time` for details.'
)
MAX_LINK_DISTANCE_HELP_STRING = (
    'Used to connect local maxima (storm objects) between times.  See '
    '`echo_top_tracking._link_local_maxima_in_time` for details.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of top-level directory.  Tracking files will be written here by '
    '`storm_tracking_io.write_processed_file`, to exact locations determined by'
    ' `storm_tracking_io.find_processed_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + REFLECTIVITY_DIR_ARG_NAME, type=str, required=True,
    help=REFLECTIVITY_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ECHO_CLASSIFN_DIR_ARG_NAME, type=str, required=True,
    help=ECHO_CLASSIFN_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CRITICAL_REFL_ARG_NAME, type=float, required=False, default=40.,
    help=CRITICAL_REFL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_ECHO_TOP_ARG_NAME, type=float, required=False, default=4.,
    help=MIN_ECHO_TOP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_SIZE_ARG_NAME, type=int, required=False, default=5,
    help=MIN_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_INTERMAX_DISTANCE_ARG_NAME, type=float, required=False,
    default=echo_top_tracking.DEFAULT_MIN_INTERMAX_DISTANCE_METRES,
    help=MIN_INTERMAX_DISTANCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_VELOCITY_DIFF_ARG_NAME, type=float, required=False,
    default=echo_top_tracking.DEFAULT_MAX_VELOCITY_DIFF_M_S01,
    help=MAX_VELOCITY_DIFF_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LINK_DISTANCE_ARG_NAME, type=float, required=False,
    default=echo_top_tracking.DEFAULT_MAX_LINK_DISTANCE_M_S01,
    help=MAX_LINK_DISTANCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run_tracking_one_time(
        reflectivity_dict, echo_classifn_dict, time_index,
        previous_local_max_dict, critical_refl_dbz, min_echo_top_m_asl,
        min_size_pixels, min_intermax_distance_metres, max_velocity_diff_m_s01,
        max_link_distance_m_s01):
    """Runs preliminary storm-tracking for one time step.

    :param reflectivity_dict: Dictionary returned by
        `radar_io.read_reflectivity_file`.
    :param echo_classifn_dict: Dictionary returned by
        `radar_io.read_echo_classifn_file`.
    :param time_index: Will run tracking for the [i]th time in the two
        dictionaries, where i = `time_index`.
    :param previous_local_max_dict: Dictionary with echo-top maxima at previous
        time step.  For list of keys, see
        `echo_top_tracking._local_maxima_to_polygons`.  If no previous time
        step, make this None.
    :param critical_refl_dbz: See documentation at top of file.
    :param min_echo_top_m_asl: Same.
    :param min_size_pixels: Same.
    :param min_intermax_distance_metres: Same.
    :param max_velocity_diff_m_s01: Same.
    :param max_link_distance_m_s01: Same.
    :return: local_max_dict: Same as `previous_local_max_dict` but for current
        time step.
    """

    # TODO(thunderhoser): Make sure times match.

    valid_time_unix_sec = (
        reflectivity_dict[radar_io.VALID_TIMES_KEY][time_index]
    )
    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT_FOR_MESSAGES
    )

    reflectivity_matrix_dbz = (
        reflectivity_dict[radar_io.REFLECTIVITY_KEY][time_index, ...]
    )
    reflectivity_matrix_dbz = numpy.swapaxes(reflectivity_matrix_dbz, 0, 2)
    reflectivity_matrix_dbz = numpy.swapaxes(reflectivity_matrix_dbz, 1, 2)

    echo_top_matrix_m_asl = gridrad_utils.get_echo_tops(
        reflectivity_matrix_dbz=reflectivity_matrix_dbz,
        grid_point_heights_m_asl=reflectivity_dict[radar_io.HEIGHTS_KEY],
        critical_reflectivity_dbz=critical_refl_dbz
    )

    convective_flag_matrix = (
        echo_classifn_dict[radar_io.CONVECTIVE_FLAGS_KEY][time_index, ...]
    )
    echo_top_matrix_m_asl[convective_flag_matrix == False] = 0.
    echo_top_matrix_m_asl = numpy.flip(echo_top_matrix_m_asl, axis=0)

    print((
        'Finding local maxima in {0:.1f}-dBZ echo top at {1:s}...'
    ).format(
        critical_refl_dbz, valid_time_string
    ))

    latitude_spacing_deg = (
        reflectivity_dict[radar_io.LATITUDES_KEY][1] -
        reflectivity_dict[radar_io.LATITUDES_KEY][0]
    )
    longitude_spacing_deg = (
        reflectivity_dict[radar_io.LONGITUDES_KEY][1] -
        reflectivity_dict[radar_io.LONGITUDES_KEY][0]
    )
    radar_metadata_dict = {
        gg_radar_utils.NW_GRID_POINT_LAT_COLUMN:
            reflectivity_dict[radar_io.LATITUDES_KEY][-1],
        gg_radar_utils.NW_GRID_POINT_LNG_COLUMN:
            reflectivity_dict[radar_io.LONGITUDES_KEY][0],
        gg_radar_utils.LAT_SPACING_COLUMN: latitude_spacing_deg,
        gg_radar_utils.LNG_SPACING_COLUMN: longitude_spacing_deg,
        gg_radar_utils.NUM_LAT_COLUMN:
            len(reflectivity_dict[radar_io.LATITUDES_KEY]),
        gg_radar_utils.NUM_LNG_COLUMN:
            len(reflectivity_dict[radar_io.LONGITUDES_KEY])
    }

    e_folding_radius_px = (
        echo_top_tracking.DEFAULT_SMOOTHING_RADIUS_DEG_LAT /
        latitude_spacing_deg
    )
    smoothed_et_matrix_m_asl = (
        echo_top_tracking._gaussian_smooth_radar_field(
            radar_matrix=echo_top_matrix_m_asl + 0.,
            e_folding_radius_pixels=e_folding_radius_px
        )
    )

    this_half_width_px = int(numpy.round(
        echo_top_tracking.DEFAULT_HALF_WIDTH_FOR_MAX_FILTER_DEG_LAT /
        latitude_spacing_deg
    ))
    local_max_dict = echo_top_tracking._find_local_maxima(
        radar_matrix=smoothed_et_matrix_m_asl,
        radar_metadata_dict=radar_metadata_dict,
        neigh_half_width_pixels=this_half_width_px
    )

    local_max_dict.update({
        temporal_tracking.VALID_TIME_KEY: valid_time_unix_sec
    })

    local_max_dict = echo_top_tracking._local_maxima_to_polygons(
        local_max_dict=local_max_dict,
        echo_top_matrix_km=echo_top_matrix_m_asl,
        min_echo_top_km=min_echo_top_m_asl,
        radar_metadata_dict=radar_metadata_dict,
        recompute_centroids=True
    )

    local_max_dict = echo_top_tracking._remove_small_polygons(
        local_max_dict=local_max_dict, min_size_pixels=min_size_pixels
    )

    projection_object = projections.init_azimuthal_equidistant_projection(
        central_latitude_deg=CENTRAL_PROJ_LATITUDE_DEG,
        central_longitude_deg=CENTRAL_PROJ_LONGITUDE_DEG
    )

    local_max_dict = echo_top_tracking._remove_redundant_local_maxima(
        local_max_dict=local_max_dict, projection_object=projection_object,
        min_intermax_distance_metres=min_intermax_distance_metres
    )

    if previous_local_max_dict is not None:
        print((
            'Linking local maxima at {0:s} with those at previous time...\n'
        ).format(
            valid_time_string
        ))

    current_to_prev_matrix = temporal_tracking.link_local_maxima_in_time(
        current_local_max_dict=local_max_dict,
        previous_local_max_dict=previous_local_max_dict,
        max_link_time_seconds=MAX_LINK_TIME_SECONDS,
        max_velocity_diff_m_s01=max_velocity_diff_m_s01,
        max_link_distance_m_s01=max_link_distance_m_s01
    )

    local_max_dict.update({
        temporal_tracking.CURRENT_TO_PREV_MATRIX_KEY: current_to_prev_matrix
    })

    return temporal_tracking.get_intermediate_velocities(
        current_local_max_dict=local_max_dict,
        previous_local_max_dict=previous_local_max_dict
    )


def _run(top_reflectivity_dir_name, top_echo_classifn_dir_name,
         first_date_string, last_date_string, critical_refl_dbz,
         min_echo_top_m_asl, min_size_pixels, min_intermax_distance_metres,
         max_velocity_diff_m_s01, max_link_distance_m_s01, top_output_dir_name):
    """Runs preliminary storm-tracking.

    This is effectively the main method.

    :param top_reflectivity_dir_name: See documentation at top of file.
    :param top_echo_classifn_dir_name: Same.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param critical_refl_dbz: Same.
    :param min_echo_top_m_asl: Same.
    :param min_size_pixels: Same.
    :param min_intermax_distance_metres: Same.
    :param max_velocity_diff_m_s01: Same.
    :param max_link_distance_m_s01: Same.
    :param top_output_dir_name: Same.
    """

    if min_size_pixels is None:
        min_size_pixels = 0

    error_checking.assert_is_integer(min_size_pixels)
    error_checking.assert_is_geq(min_size_pixels, 0)
    error_checking.assert_is_greater(min_echo_top_m_asl, 0.)

    reflectivity_file_names = radar_io.find_many_files(
        top_directory_name=top_reflectivity_dir_name,
        first_date_string=first_date_string, last_date_string=last_date_string,
        file_type_string=radar_io.REFL_TYPE_STRING,
        prefer_zipped=True, allow_other_format=True,
        raise_error_if_all_missing=True
    )

    date_strings = [
        radar_io.file_name_to_date(f) for f in reflectivity_file_names
    ]

    echo_classifn_file_names = [
        radar_io.find_file(
            top_directory_name=top_echo_classifn_dir_name, valid_date_string=d,
            file_type_string=radar_io.ECHO_CLASSIFN_TYPE_STRING,
            prefer_zipped=True, allow_other_format=True,
            raise_error_if_missing=True
        ) for d in date_strings
    ]

    num_days = len(date_strings)
    local_max_dict_by_time = []

    for i in range(num_days):
        print('Reading data from: "{0:s}"...'.format(
            reflectivity_file_names[i]
        ))
        reflectivity_dict = radar_io.read_reflectivity_file(
            netcdf_file_name=reflectivity_file_names[i], fill_nans=True
        )

        print('Reading data from: "{0:s}"...'.format(
            echo_classifn_file_names[i]
        ))
        echo_classifn_dict = radar_io.read_echo_classifn_file(
            echo_classifn_file_names[i]
        )

        assert numpy.array_equal(
            reflectivity_dict[radar_io.VALID_TIMES_KEY],
            echo_classifn_dict[radar_io.VALID_TIMES_KEY]
        )

        num_times = len(reflectivity_dict[radar_io.VALID_TIMES_KEY])

        for j in range(num_times):
            previous_local_max_dict = (
                None if len(local_max_dict_by_time) == 0
                else local_max_dict_by_time[-1]
            )

            current_local_max_dict = _run_tracking_one_time(
                reflectivity_dict=reflectivity_dict,
                echo_classifn_dict=echo_classifn_dict, time_index=j,
                previous_local_max_dict=previous_local_max_dict,
                critical_refl_dbz=critical_refl_dbz,
                min_echo_top_m_asl=min_echo_top_m_asl,
                min_size_pixels=min_size_pixels,
                min_intermax_distance_metres=min_intermax_distance_metres,
                max_velocity_diff_m_s01=max_velocity_diff_m_s01,
                max_link_distance_m_s01=max_link_distance_m_s01
            )

            local_max_dict_by_time.append(current_local_max_dict)

    print(SEPARATOR_STRING)

    valid_times_unix_sec = numpy.array([
        d[temporal_tracking.VALID_TIME_KEY] for d in local_max_dict_by_time
    ], dtype=int)

    print('Converting time series of echo-top maxima to storm tracks...')
    tracking_start_times_unix_sec, tracking_end_times_unix_sec = (
        echo_top_tracking._radar_times_to_tracking_periods(
            radar_times_unix_sec=valid_times_unix_sec,
            max_time_interval_sec=MAX_LINK_TIME_SECONDS
        )
    )

    first_numeric_id = 100 * time_conversion.string_to_unix_sec(
        date_strings[0], DATE_FORMAT
    )

    storm_object_table = temporal_tracking.local_maxima_to_storm_tracks(
        local_max_dict_by_time=local_max_dict_by_time,
        first_numeric_id=first_numeric_id
    )

    print('Computing storm ages...')
    storm_object_table = temporal_tracking.get_storm_ages(
        storm_object_table=storm_object_table,
        tracking_start_times_unix_sec=tracking_start_times_unix_sec,
        tracking_end_times_unix_sec=tracking_end_times_unix_sec,
        max_link_time_seconds=MAX_LINK_TIME_SECONDS
    )

    print('Computing storm velocities...')
    storm_object_table = temporal_tracking.get_storm_velocities(
        storm_object_table=storm_object_table
    )

    print(SEPARATOR_STRING)

    echo_top_tracking._write_new_tracks(
        storm_object_table=storm_object_table,
        top_output_dir_name=top_output_dir_name,
        valid_times_unix_sec=valid_times_unix_sec
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_reflectivity_dir_name=getattr(
            INPUT_ARG_OBJECT, REFLECTIVITY_DIR_ARG_NAME
        ),
        top_echo_classifn_dir_name=getattr(
            INPUT_ARG_OBJECT, ECHO_CLASSIFN_DIR_ARG_NAME
        ),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        critical_refl_dbz=getattr(INPUT_ARG_OBJECT, CRITICAL_REFL_ARG_NAME),
        min_echo_top_m_asl=getattr(INPUT_ARG_OBJECT, MIN_ECHO_TOP_ARG_NAME),
        min_size_pixels=getattr(INPUT_ARG_OBJECT, MIN_SIZE_ARG_NAME),
        min_intermax_distance_metres=getattr(
            INPUT_ARG_OBJECT, MIN_INTERMAX_DISTANCE_ARG_NAME
        ),
        max_velocity_diff_m_s01=getattr(
            INPUT_ARG_OBJECT, MAX_VELOCITY_DIFF_ARG_NAME
        ),
        max_link_distance_m_s01=getattr(
            INPUT_ARG_OBJECT, MAX_LINK_DISTANCE_ARG_NAME
        ),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
