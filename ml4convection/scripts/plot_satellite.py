"""Plots satellite images for the given days."""

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
from ml4convection.io import satellite_io
from ml4convection.plotting import satellite_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DAYS_TO_SECONDS = 86400
TIME_FORMAT = '%Y-%m-%d-%H%M'

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
BAND_NUMBERS_ARG_NAME = 'band_numbers'
DAILY_TIMES_ARG_NAME = 'daily_times_seconds'
PLOT_RANDOM_ARG_NAME = 'plot_random_examples'
NUM_EXAMPLES_PER_DAY_ARG_NAME = 'num_examples_per_day'
PLOT_BASEMAP_ARG_NAME = 'plot_basemap'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

SATELLITE_DIR_HELP_STRING = (
    'Name of directory with satellite data.  Files therein will be found by '
    '`satellite_io.find_file` and read by `satellite_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will plot satellite images for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

BAND_NUMBERS_HELP_STRING = (
    'List of band numbers.  Will plot brightness temperatures for these '
    'spectral bands only.'
)
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
    'Boolean flag.  If 1 (0), will plot satellite images with (without) '
    'basemap.'
)
OUTPUT_DIR_HELP_STRING = 'Name of output directory.  Images will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SATELLITE_DIR_ARG_NAME, type=str, required=False, default='',
    help=SATELLITE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BAND_NUMBERS_ARG_NAME, type=int, nargs='+', required=True,
    help=BAND_NUMBERS_HELP_STRING
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


def _plot_one_satellite_image(
        satellite_dict, time_index, band_index, plot_basemap, output_dir_name):
    """Plots one satellite image.

    :param satellite_dict: Dictionary in format returned by
        `satellite_io.read_file`.
    :param time_index: Index of time to plot.
    :param band_index: Index of spectral band to plot.
    :param plot_basemap: See documentation at top of file.
    :param output_dir_name: Same.
    """

    latitudes_deg_n = satellite_dict[satellite_io.LATITUDES_KEY]
    longitudes_deg_e = satellite_dict[satellite_io.LONGITUDES_KEY]

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
        satellite_dict[satellite_io.VALID_TIMES_KEY][time_index]
    )
    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT
    )
    band_number = satellite_dict[satellite_io.BAND_NUMBERS_KEY][band_index]
    title_string = 'Band-{0:d} brightness temperature at {1:s}'.format(
        band_number, valid_time_string
    )

    brightness_temp_matrix_kelvins = (
        satellite_dict[satellite_io.BRIGHTNESS_TEMP_KEY][
            time_index, ..., band_index
        ]
    )

    satellite_plotting.plot_2d_grid(
        brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
        axes_object=axes_object,
        min_latitude_deg_n=numpy.min(latitudes_deg_n),
        min_longitude_deg_e=numpy.min(longitudes_deg_e),
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

    axes_object.set_title(title_string)

    output_file_name = (
        '{0:s}/brightness-temperature_band{1:02d}_{2:s}.jpg'
    ).format(
        output_dir_name, band_number, valid_time_string
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_satellite_one_day(
        satellite_file_name, band_numbers, daily_times_seconds,
        plot_random_examples, num_examples, plot_basemap, output_dir_name):
    """Plots satellite images for one day.

    :param satellite_file_name: Path to input file.  Will be read by
        `satellite_io.read_file`.
    :param band_numbers: See documentation at top of file.
    :param daily_times_seconds: Same.
    :param plot_random_examples: Same.
    :param num_examples: Same.
    :param plot_basemap: Same.
    :param output_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(satellite_file_name))
    satellite_dict = satellite_io.read_file(
        netcdf_file_name=satellite_file_name, fill_nans=False
    )
    satellite_dict = satellite_io.subset_by_band(
        satellite_dict=satellite_dict, band_numbers=band_numbers
    )

    if daily_times_seconds is not None:
        valid_times_unix_sec = satellite_dict[satellite_io.VALID_TIMES_KEY]
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
        satellite_dict = satellite_io.subset_by_time(
            satellite_dict=satellite_dict,
            desired_times_unix_sec=desired_times_unix_sec
        )[0]
    else:
        num_examples_total = len(satellite_dict[satellite_io.VALID_TIMES_KEY])
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

        satellite_dict = satellite_io.subset_by_index(
            satellite_dict=satellite_dict, desired_indices=desired_indices
        )

    num_times = len(satellite_dict[satellite_io.VALID_TIMES_KEY])
    num_bands = len(satellite_dict[satellite_io.BAND_NUMBERS_KEY])

    for i in range(num_times):
        for j in range(num_bands):
            _plot_one_satellite_image(
                satellite_dict=satellite_dict, time_index=i, band_index=j,
                plot_basemap=plot_basemap, output_dir_name=output_dir_name
            )


def _run(top_satellite_dir_name, first_date_string, last_date_string,
         band_numbers, daily_times_seconds, plot_random_examples,
         num_examples_per_day, plot_basemap, output_dir_name):
    """Plots satellite images for the given days.

    This is effectively the main method.

    :param top_satellite_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param band_numbers: Same.
    :param daily_times_seconds: Same.
    :param plot_random_examples: Same.
    :param num_examples_per_day: Same.
    :param plot_basemap: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if len(daily_times_seconds) == 1 and daily_times_seconds[0] < 0:
        daily_times_seconds = None

    if daily_times_seconds is not None:
        error_checking.assert_is_geq_numpy_array(daily_times_seconds, 0)
        error_checking.assert_is_less_than_numpy_array(
            daily_times_seconds, DAYS_TO_SECONDS
        )
        plot_random_examples = False

    satellite_file_names = satellite_io.find_many_files(
        top_directory_name=top_satellite_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        prefer_zipped=True, allow_other_format=True,
        raise_error_if_any_missing=False
    )

    for i in range(len(satellite_file_names)):
        _plot_satellite_one_day(
            satellite_file_name=satellite_file_names[i],
            band_numbers=band_numbers,
            daily_times_seconds=daily_times_seconds,
            plot_random_examples=plot_random_examples,
            num_examples=num_examples_per_day,
            plot_basemap=plot_basemap, output_dir_name=output_dir_name
        )

        if i != len(satellite_file_names) - 1:
            print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_satellite_dir_name=getattr(
            INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME
        ),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        band_numbers=numpy.array(
            getattr(INPUT_ARG_OBJECT, BAND_NUMBERS_ARG_NAME), dtype=int
        ),
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
