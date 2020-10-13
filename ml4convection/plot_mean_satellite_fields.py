"""Plots satellite images for the given days."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import satellite_io
import example_io
import border_io
import plotting_utils
import satellite_plotting

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DAYS_TO_SECONDS = 86400
TIME_FORMAT = '%Y-%m-%d-%H%M'

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
BAND_NUMBERS_ARG_NAME = 'band_numbers'
SPATIAL_DS_FACTOR_ARG_NAME = 'spatial_downsampling_factor'
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
SPATIAL_DS_FACTOR_HELP_STRING = (
    'Downsampling factor, used to coarsen spatial resolution.  If you do not '
    'want to coarsen spatial resolution, leave this alone.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Images will be saved here (one '
    'subdirectory per band).'
)

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
    '--' + SPATIAL_DS_FACTOR_ARG_NAME, type=int, required=False, default=1,
    help=SPATIAL_DS_FACTOR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_satellite_image(
        satellite_dict, time_index, band_index, border_latitudes_deg_n,
        border_longitudes_deg_e, top_output_dir_name):
    """Plots one satellite image.

    :param satellite_dict: Dictionary in format returned by
        `satellite_io.read_file`.
    :param time_index: Index of time to plot.
    :param band_index: Index of spectral band to plot.
    :param border_latitudes_deg_n: See doc for `_plot_satellite_one_day`.
    :param border_longitudes_deg_e: Same.
    :param top_output_dir_name: Same.
    """

    latitudes_deg_n = satellite_dict[satellite_io.LATITUDES_KEY]
    longitudes_deg_e = satellite_dict[satellite_io.LONGITUDES_KEY]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
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

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=latitudes_deg_n,
        plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=2., meridian_spacing_deg=2.
    )

    axes_object.set_title(title_string)

    output_file_name = (
        '{0:s}/band{1:02d}/brightness-temperature_band{1:02d}_{2:s}.jpg'
    ).format(
        top_output_dir_name, band_number, valid_time_string
    )
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(top_satellite_dir_name, first_date_string, last_date_string,
         band_numbers, spatial_downsampling_factor, top_output_dir_name):
    """Plots satellite images for the given days.

    This is effectively the main method.

    :param top_satellite_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param band_numbers: Same.
    :param spatial_downsampling_factor: Same.
    :param top_output_dir_name: Same.
    """

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    if spatial_downsampling_factor <= 1:
        spatial_downsampling_factor = None

    satellite_file_names = satellite_io.find_many_files(
        top_directory_name=top_satellite_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        prefer_zipped=True, allow_other_format=True,
        raise_error_if_any_missing=False
    )

    satellite_dicts = []

    for this_file_name in satellite_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_satellite_dict = satellite_io.read_file(
            netcdf_file_name=this_file_name, fill_nans=False
        )

        if spatial_downsampling_factor is not None:
            this_satellite_dict = example_io.downsample_data_in_space(
                satellite_dict=this_satellite_dict,
                downsampling_factor=spatial_downsampling_factor,
                change_coordinates=True
            )[0]

        this_satellite_dict = satellite_io.subset_by_band(
            satellite_dict=this_satellite_dict, band_numbers=band_numbers
        )
        this_satellite_dict[satellite_io.BRIGHTNESS_TEMP_KEY] = numpy.mean(
            this_satellite_dict[satellite_io.BRIGHTNESS_TEMP_KEY],
            axis=0, keepdims=True
        )
        this_satellite_dict[satellite_io.VALID_TIMES_KEY] = numpy.array(
            [0], dtype=int
        )

        satellite_dicts.append(this_satellite_dict)

    satellite_dict = satellite_io.concat_data(satellite_dicts)
    satellite_dict[satellite_io.BRIGHTNESS_TEMP_KEY] = numpy.mean(
        satellite_dict[satellite_io.BRIGHTNESS_TEMP_KEY],
        axis=0, keepdims=True
    )

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_date_string, satellite_io.DATE_FORMAT
    )
    satellite_dict[satellite_io.VALID_TIMES_KEY] = numpy.array(
        [first_time_unix_sec], dtype=int
    )

    print(SEPARATOR_STRING)

    for j in range(len(band_numbers)):
        _plot_one_satellite_image(
            satellite_dict=satellite_dict, time_index=0, band_index=j,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            top_output_dir_name=top_output_dir_name
        )


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
        spatial_downsampling_factor=getattr(
            INPUT_ARG_OBJECT, SPATIAL_DS_FACTOR_ARG_NAME
        ),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
