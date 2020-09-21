"""Plots radar images for the given days."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from ml4convection.io import radar_io
from ml4convection.io import example_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DAYS_TO_SECONDS = 86400
TIME_FORMAT = '%Y-%m-%d-%H%M'

COMPOSITE_REFL_NAME = 'reflectivity_column_max_dbz'

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

REFLECTIVITY_DIR_ARG_NAME = 'input_reflectivity_dir_name'
ECHO_CLASSIFN_DIR_ARG_NAME = 'input_echo_classifn_dir_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
DAILY_TIMES_ARG_NAME = 'daily_times_seconds'
PLOT_RANDOM_ARG_NAME = 'plot_random_examples'
NUM_EXAMPLES_PER_DAY_ARG_NAME = 'num_examples_per_day'
PLOT_BASEMAP_ARG_NAME = 'plot_basemap'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

REFLECTIVITY_DIR_HELP_STRING = (
    'Name of directory with reflectivity data.  Files therein will be found by '
    '`radar_io.find_file` and read by `radar_io.read_reflectivity_file`.'
)
ECHO_CLASSIFN_DIR_HELP_STRING = (
    '[used only if `{0:s}` is specified] Name of directory with echo-'
    'classification data (files therein will be found by `radar_io.find_file` '
    'and read by `radar_io.read_echo_classifn_file`).  If specified, will plot '
    'black dot over each convective pixel.'
).format(REFLECTIVITY_DIR_ARG_NAME)

TARGET_DIR_HELP_STRING = (
    '[used only if `{0:s}` is left empty] Name of directory with target data.  '
    'Files therein will be found by `example_io.find_target_file` and read by '
    '`example_io.read_target_file`.'
).format(REFLECTIVITY_DIR_ARG_NAME)

DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will plot radar images for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

DAILY_TIMES_HELP_STRING = (
    'List of times to plot for each day.  All values should be in the range '
    '0...86399.'
)
PLOT_RANDOM_HELP_STRING = (
    '[used only if `{0:s}` is left alone] Boolean flag.  If 1, will randomly '
    'draw `{1:s}` examples from each day.  If 0, will draw the first `{1:s}` '
    'examples from each day.'
).format(DAILY_TIMES_ARG_NAME, NUM_EXAMPLES_PER_DAY_ARG_NAME)

NUM_EXAMPLES_PER_DAY_HELP_STRING = (
    'See documentation for `{0:s}`.'.format(PLOT_RANDOM_ARG_NAME)
)
PLOT_BASEMAP_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot radar images with (without) basemap.'
)
OUTPUT_DIR_HELP_STRING = 'Name of output directory.  Images will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + REFLECTIVITY_DIR_ARG_NAME, type=str, required=False, default='',
    help=REFLECTIVITY_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ECHO_CLASSIFN_DIR_ARG_NAME, type=str, required=False, default='',
    help=ECHO_CLASSIFN_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=False, default='',
    help=TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DAILY_TIMES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=DAILY_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_RANDOM_ARG_NAME, type=int, required=False, default=1,
    help=PLOT_RANDOM_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_PER_DAY_ARG_NAME, type=int, required=False, default=5,
    help=NUM_EXAMPLES_PER_DAY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_BASEMAP_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_BASEMAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_radar_one_example(
        reflectivity_dict, convective_flag_matrix, example_index, plot_basemap,
        output_dir_name):
    """Plots one radar image.

    M = number of rows in grid
    N = number of columns in grid

    :param reflectivity_dict: See doc for `_plot_radar_one_day`.
    :param convective_flag_matrix: M-by-N numpy array of Boolean flags.  If
        specified, will plot convective pixels only.  If None, will plot all
        pixels.
    :param example_index: Will plot [i]th example, where i = `example_index`.
    :param plot_basemap: See documentation at top of file.
    :param output_dir_name: Same.
    """

    latitudes_deg_n = reflectivity_dict[radar_io.LATITUDES_KEY]
    longitudes_deg_e = reflectivity_dict[radar_io.LONGITUDES_KEY]

    if plot_basemap:
        figure_object, axes_object, basemap_object = (
            plotting_utils.create_equidist_cylindrical_map(
                min_latitude_deg=numpy.min(latitudes_deg_n),
                max_latitude_deg=numpy.max(latitudes_deg_n),
                min_longitude_deg=numpy.min(longitudes_deg_e),
                max_longitude_deg=numpy.max(longitudes_deg_e),
                resolution_string='i'
            )
        )

        plotting_utils.plot_coastlines(
            basemap_object=basemap_object, axes_object=axes_object,
            line_colour=plotting_utils.DEFAULT_COUNTRY_COLOUR
        )
        plotting_utils.plot_countries(
            basemap_object=basemap_object, axes_object=axes_object
        )
        plotting_utils.plot_states_and_provinces(
            basemap_object=basemap_object, axes_object=axes_object
        )
        plotting_utils.plot_parallels(
            basemap_object=basemap_object, axes_object=axes_object,
            num_parallels=NUM_PARALLELS
        )
        plotting_utils.plot_meridians(
            basemap_object=basemap_object, axes_object=axes_object,
            num_meridians=NUM_MERIDIANS
        )
    else:
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

    valid_time_unix_sec = (
        reflectivity_dict[radar_io.VALID_TIMES_KEY][example_index]
    )
    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT
    )
    title_string = 'Composite reflectivity at {0:s}'.format(valid_time_string)

    # TODO(thunderhoser): Allow this script to plot reflectivity at any height.
    composite_refl_matrix_dbz = numpy.nanmax(
        reflectivity_dict[radar_io.REFLECTIVITY_KEY][example_index, ...],
        axis=-1
    )

    if convective_flag_matrix is not None:
        composite_refl_matrix_dbz[convective_flag_matrix == False] = 0.

    radar_plotting.plot_latlng_grid(
        field_matrix=composite_refl_matrix_dbz, field_name=COMPOSITE_REFL_NAME,
        axes_object=axes_object,
        min_grid_point_latitude_deg=numpy.min(latitudes_deg_n),
        min_grid_point_longitude_deg=numpy.min(longitudes_deg_e),
        latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0]
    )

    if not plot_basemap:
        tick_latitudes_deg_n = numpy.unique(numpy.round(latitudes_deg_n))
        tick_latitudes_deg_n = tick_latitudes_deg_n[
            tick_latitudes_deg_n >= numpy.min(latitudes_deg_n)
        ]
        tick_latitudes_deg_n = tick_latitudes_deg_n[
            tick_latitudes_deg_n <= numpy.max(latitudes_deg_n)
        ]

        tick_longitudes_deg_e = numpy.unique(numpy.round(longitudes_deg_e))
        tick_longitudes_deg_e = tick_longitudes_deg_e[
            tick_longitudes_deg_e >= numpy.min(longitudes_deg_e)
        ]
        tick_longitudes_deg_e = tick_longitudes_deg_e[
            tick_longitudes_deg_e <= numpy.max(longitudes_deg_e)
        ]

        axes_object.set_xticks(tick_longitudes_deg_e)
        axes_object.set_yticks(tick_latitudes_deg_n)
        axes_object.grid(
            b=True, which='major', axis='both', linestyle='--', linewidth=2
        )

        axes_object.set_xlabel(r'Longitude ($^{\circ}$E)')
        axes_object.set_ylabel(r'Latitude ($^{\circ}$N)')

    colour_map_object, colour_norm_object = (
        radar_plotting.get_default_colour_scheme(COMPOSITE_REFL_NAME)
    )

    plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=composite_refl_matrix_dbz,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=True
    )

    axes_object.set_title(title_string)

    output_file_name = '{0:s}/composite_reflectivity_{1:s}.jpg'.format(
        output_dir_name, valid_time_string
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_radar_one_day(
        reflectivity_dict, echo_classifn_dict, daily_times_seconds,
        plot_random_examples, num_examples, plot_basemap, output_dir_name):
    """Plots radar images for one day.

    :param reflectivity_dict: Dictionary in the format returned by
        `radar_io.read_reflectivity_file`.
    :param echo_classifn_dict: Dictionary in the format returned by
        `radar_io.read_echo_classifn_file`.  If specified, will plot convective
        pixels only.  If None, will plot all pixels.
    :param daily_times_seconds: See documentation at top of file.
    :param plot_random_examples: Same.
    :param num_examples: Same.
    :param plot_basemap: Same.
    :param output_dir_name: Same.
    """

    if echo_classifn_dict is None:
        convective_flag_matrix = None
    else:
        assert numpy.array_equal(
            reflectivity_dict[radar_io.VALID_TIMES_KEY],
            echo_classifn_dict[radar_io.VALID_TIMES_KEY]
        )
        convective_flag_matrix = (
            echo_classifn_dict[radar_io.CONVECTIVE_FLAGS_KEY]
        )

    if daily_times_seconds is not None:
        valid_times_unix_sec = reflectivity_dict[radar_io.VALID_TIMES_KEY]
        base_time_unix_sec = number_rounding.floor_to_nearest(
            valid_times_unix_sec[0], DAYS_TO_SECONDS
        )
        desired_times_unix_sec = numpy.round(
            base_time_unix_sec + daily_times_seconds
        ).astype(int)

        good_flags = numpy.array([
            t in valid_times_unix_sec for t in desired_times_unix_sec
        ], dtype=bool)

        if not numpy.any(good_flags):
            return

        desired_times_unix_sec = desired_times_unix_sec[good_flags]
        reflectivity_dict, desired_indices = radar_io.subset_by_time(
            reflectivity_dict=reflectivity_dict,
            desired_times_unix_sec=desired_times_unix_sec
        )

        if convective_flag_matrix is not None:
            convective_flag_matrix = (
                convective_flag_matrix[desired_indices, ...]
            )
    else:
        num_examples_total = len(reflectivity_dict[radar_io.VALID_TIMES_KEY])
        desired_indices = numpy.linspace(
            0, num_examples_total - 1, num=num_examples_total, dtype=int
        )

        if num_examples < num_examples_total:
            if plot_random_examples:
                desired_indices = numpy.random.choice(
                    desired_indices, size=num_examples, replace=False
                )
            else:
                desired_indices = desired_indices[:num_examples]

        reflectivity_dict = radar_io.subset_by_index(
            reflectivity_dict=reflectivity_dict, desired_indices=desired_indices
        )

        if convective_flag_matrix is not None:
            convective_flag_matrix = (
                convective_flag_matrix[desired_indices, ...]
            )

    num_examples = len(reflectivity_dict[radar_io.VALID_TIMES_KEY])

    for i in range(num_examples):
        _plot_radar_one_example(
            reflectivity_dict=reflectivity_dict,
            convective_flag_matrix=convective_flag_matrix[i, ...],
            example_index=i, plot_basemap=plot_basemap,
            output_dir_name=output_dir_name
        )


def _run(top_reflectivity_dir_name, top_echo_classifn_dir_name,
         top_target_dir_name, first_date_string, last_date_string,
         daily_times_seconds, plot_random_examples, num_examples_per_day,
         plot_basemap, output_dir_name):
    """Plots radar images for the given days.

    This is effectively the main method.

    :param top_reflectivity_dir_name: See documentation at top of file.
    :param top_echo_classifn_dir_name: Same.
    :param top_target_dir_name: Same.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param daily_times_seconds: Same.
    :param plot_random_examples: Same.
    :param num_examples_per_day: Same.
    :param plot_basemap: Same.
    :param output_dir_name: Same.
    :raises: ValueError:
        if `top_reflectivity_dir_name is None and top_target_dir_name is None`.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if top_reflectivity_dir_name == '':
        top_reflectivity_dir_name = None
    if top_target_dir_name == '':
        top_target_dir_name = None
    if top_echo_classifn_dir_name == '':
        top_echo_classifn_dir_name = None
    if len(daily_times_seconds) == 1 and daily_times_seconds[0] < 0:
        daily_times_seconds = None

    if daily_times_seconds is not None:
        error_checking.assert_is_geq_numpy_array(daily_times_seconds, 0)
        error_checking.assert_is_less_than_numpy_array(
            daily_times_seconds, DAYS_TO_SECONDS
        )
        plot_random_examples = False

    if top_reflectivity_dir_name is None and top_target_dir_name is None:
        raise ValueError((
            'One of the input args `{0:s}` and `{1:s}` must be specified.'
        ).format(
            REFLECTIVITY_DIR_ARG_NAME, TARGET_DIR_ARG_NAME
        ))

    if top_reflectivity_dir_name is None:
        top_echo_classifn_dir_name = None

        input_file_names = example_io.find_many_target_files(
            top_directory_name=top_target_dir_name,
            first_date_string=first_date_string,
            last_date_string=last_date_string,
            raise_error_if_any_missing=False
        )
    else:
        input_file_names = radar_io.find_many_files(
            top_directory_name=top_reflectivity_dir_name,
            first_date_string=first_date_string,
            last_date_string=last_date_string,
            file_type_string=radar_io.REFL_TYPE_STRING,
            raise_error_if_any_missing=False
        )

    if top_echo_classifn_dir_name is None:
        echo_classifn_file_names = None
    else:
        echo_classifn_file_names = [
            radar_io.find_file(
                top_directory_name=top_echo_classifn_dir_name,
                valid_date_string=radar_io.file_name_to_date(f),
                file_type_string=radar_io.ECHO_CLASSIFN_TYPE_STRING,
                raise_error_if_missing=True
            )
            for f in input_file_names
        ]

    for i in range(len(input_file_names)):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))

        if top_reflectivity_dir_name is None:
            target_dict = example_io.read_target_file(
                netcdf_file_name=input_file_names[i], read_targets=False,
                read_reflectivities=True
            )

            reflectivity_dict = {
                radar_io.REFLECTIVITY_KEY: numpy.expand_dims(
                    target_dict[example_io.COMPOSITE_REFL_MATRIX_KEY], axis=-1
                ),
                radar_io.VALID_TIMES_KEY: target_dict[
                    example_io.VALID_TIMES_KEY],
                radar_io.LATITUDES_KEY: target_dict[example_io.LATITUDES_KEY],
                radar_io.LONGITUDES_KEY: target_dict[example_io.LONGITUDES_KEY]
            }
        else:
            reflectivity_dict = radar_io.read_reflectivity_file(
                netcdf_file_name=input_file_names[i], fill_nans=False
            )

        if top_echo_classifn_dir_name is None:
            echo_classifn_dict = None
        else:
            echo_classifn_dict = radar_io.read_echo_classifn_file(
                echo_classifn_file_names[i]
            )

        _plot_radar_one_day(
            reflectivity_dict=reflectivity_dict,
            echo_classifn_dict=echo_classifn_dict,
            daily_times_seconds=daily_times_seconds,
            plot_random_examples=plot_random_examples,
            num_examples=num_examples_per_day,
            plot_basemap=plot_basemap, output_dir_name=output_dir_name
        )

        if i != len(input_file_names) - 1:
            print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_reflectivity_dir_name=getattr(
            INPUT_ARG_OBJECT, REFLECTIVITY_DIR_ARG_NAME
        ),
        top_echo_classifn_dir_name=getattr(
            INPUT_ARG_OBJECT, ECHO_CLASSIFN_DIR_ARG_NAME
        ),
        top_target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        daily_times_seconds=numpy.array(
            getattr(INPUT_ARG_OBJECT, DAILY_TIMES_ARG_NAME), dtype=int
        ),
        plot_random_examples=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_RANDOM_ARG_NAME
        )),
        num_examples_per_day=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_PER_DAY_ARG_NAME
        ),
        plot_basemap=bool(getattr(INPUT_ARG_OBJECT, PLOT_BASEMAP_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
