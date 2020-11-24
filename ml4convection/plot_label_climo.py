"""Plots climatology (long-term average frequency) of convection labels."""

import os
import sys
import pickle
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import radar_plotting
import gg_plotting_utils
import radar_io
import border_io
import plotting_utils

DUMMY_FIELD_NAME = 'reflectivity_column_max_dbz'

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_DIR_ARG_NAME = 'input_echo_classifn_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with echo classifications.  Files therein will'
    ' be found by `radar_io.find_file` and read by '
    '`radar_io.read_echo_classifn_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will plot average frequency over the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_FILE_HELP_STRING = 'Path to output file.  Figure will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(top_echo_classifn_dir_name, first_date_string, last_date_string,
         output_file_name):
    """Plots climatology (long-term average frequency) of convection labels.

    This is effectively the main method.

    :param top_echo_classifn_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    echo_classifn_file_names = radar_io.find_many_files(
        top_directory_name=top_echo_classifn_dir_name,
        first_date_string=first_date_string, last_date_string=last_date_string,
        file_type_string=radar_io.ECHO_CLASSIFN_TYPE_STRING,
        prefer_zipped=True, allow_other_format=True,
        raise_error_if_all_missing=True, raise_error_if_any_missing=True
    )

    num_times = 0
    convective_freq_matrix = None
    latitudes_deg_n = None
    longitudes_deg_e = None

    for this_file_name in echo_classifn_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_echo_classifn_dict = radar_io.read_echo_classifn_file(
            this_file_name
        )
        latitudes_deg_n = this_echo_classifn_dict[radar_io.LATITUDES_KEY]
        longitudes_deg_e = this_echo_classifn_dict[radar_io.LONGITUDES_KEY]

        this_convective_flag_matrix = (
            this_echo_classifn_dict[radar_io.CONVECTIVE_FLAGS_KEY]
        )
        num_times += this_convective_flag_matrix.shape[0]
        this_convective_freq_matrix = numpy.sum(
            this_convective_flag_matrix, axis=0
        )

        if convective_freq_matrix is None:
            convective_freq_matrix = this_convective_freq_matrix + 0
        else:
            convective_freq_matrix += this_convective_freq_matrix

    convective_freq_matrix = convective_freq_matrix / num_times

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()
    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
    )

    colour_map_object = pyplot.get_cmap('viridis')
    max_colour_value = numpy.max(convective_freq_matrix)
    colour_norm_object = pyplot.Normalize(vmin=0., vmax=max_colour_value)

    radar_plotting.plot_latlng_grid(
        field_matrix=convective_freq_matrix, field_name=DUMMY_FIELD_NAME,
        axes_object=axes_object,
        min_grid_point_latitude_deg=numpy.min(latitudes_deg_n),
        min_grid_point_longitude_deg=numpy.min(longitudes_deg_e),
        latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object
    )

    colour_bar_object = gg_plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=convective_freq_matrix,
        colour_map_object=colour_map_object,
        min_value=0., max_value=max_colour_value,
        orientation_string='vertical', extend_min=False, extend_max=True
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=latitudes_deg_n,
        plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=1., meridian_spacing_deg=1.
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    pickle_file_name = output_file_name.replace('.jpg', '.p')
    print(numpy.mean(convective_freq_matrix))
    print(numpy.median(convective_freq_matrix))
    print(numpy.max(convective_freq_matrix))

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(convective_freq_matrix, pickle_file_handle)
    pickle_file_handle.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_echo_classifn_dir_name=getattr(
            INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME
        ),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
