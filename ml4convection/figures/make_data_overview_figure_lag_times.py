"""Makes data-overview figure with multiple lag times."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import imagemagick_utils
from ml4convection.io import satellite_io
from ml4convection.io import radar_io
from ml4convection.io import border_io
from ml4convection.io import prediction_io
from ml4convection.plotting import prediction_plotting
from ml4convection.plotting import plotting_utils
from ml4convection.plotting import satellite_plotting
from ml4convection.scripts import plot_predictions

TIME_FORMAT = '%Y-%m-%d-%H%M'
DATE_FORMAT = '%Y%m%d'
COMPOSITE_REFL_NAME = 'reflectivity_column_max_dbz'

BAND_NUMBERS_TO_PLOT = numpy.array([8, 11, 16], dtype=int)

MASK_OUTLINE_COLOUR = numpy.full(3, 152. / 255)
MASK_OUTLINE_WIDTH = 4
BORDER_WIDTH = 4

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

TITLE_FONT_SIZE = 40
FONT_SIZE = 50

pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
REFLECTIVITY_DIR_ARG_NAME = 'input_reflectivity_dir_name'
ECHO_CLASSIFN_DIR_ARG_NAME = 'input_echo_classifn_dir_name'
MASK_FILE_ARG_NAME = 'input_mask_file_name'
SATELLITE_TIMES_ARG_NAME = 'satellite_time_strings'
RADAR_TIME_ARG_NAME = 'radar_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

SATELLITE_DIR_HELP_STRING = (
    'Name of directory with satellite data.  Files therein will be found by '
    '`satellite_io.find_file` and read by `satellite_io.read_file`.'
)
REFLECTIVITY_DIR_HELP_STRING = (
    'Name of directory with reflectivity data.  Files therein will be found by '
    '`radar_io.find_file` and read by `radar_io.read_reflectivity_file`.'
)
ECHO_CLASSIFN_DIR_HELP_STRING = (
    'Name of directory with echo-classification data (files therein will be '
    'found by `radar_io.find_file` and read by '
    '`radar_io.read_echo_classifn_file`).'
)
MASK_FILE_HELP_STRING = (
    'Name of mask file (will be read by `radar_io.read_mask_file`).  Unmasked '
    'area will be plotted with grey outline.  If you do not want to plot a '
    'mask, leave this alone.'
)
SATELLITE_TIMES_HELP_STRING = (
    'Will plot satellite data for these times (format "yyyy-mm-dd-HHMM").'
)
RADAR_TIME_HELP_STRING = (
    'Will plot radar data for this time (format "yyyy-mm-dd-HHMM").'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SATELLITE_DIR_ARG_NAME, type=str, required=True,
    help=SATELLITE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + REFLECTIVITY_DIR_ARG_NAME, type=str, required=True,
    help=REFLECTIVITY_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ECHO_CLASSIFN_DIR_ARG_NAME, type=str, required=True,
    help=ECHO_CLASSIFN_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MASK_FILE_ARG_NAME, type=str, required=False, default='',
    help=MASK_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SATELLITE_TIMES_ARG_NAME, type=str, nargs='+', required=True,
    help=SATELLITE_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_TIME_ARG_NAME, type=str, required=True,
    help=RADAR_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_satellite_image(
        satellite_dict, time_index, band_index, border_latitudes_deg_n,
        border_longitudes_deg_e, letter_label, cbar_orientation_string,
        output_dir_name, title_string=None):
    """Plots one satellite image.

    :param satellite_dict: Dictionary in format returned by
        `satellite_io.read_file`.
    :param time_index: Index of time to plot.
    :param band_index: Index of spectral band to plot.
    :param border_latitudes_deg_n: See doc for
        `plot_satellite._plot_satellite_one_day`.
    :param border_longitudes_deg_e: Same.
    :param letter_label: Letter label.
    :param cbar_orientation_string: See doc for
        `satellite_plotting.plot_2d_grid`.
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :param title_string: Title (will be plotted at top of figure).  To use
        default, leave this alone.
    :return: output_file_name: Path to output file.
    """

    latitudes_deg_n = satellite_dict[satellite_io.LATITUDES_KEY]
    longitudes_deg_e = satellite_dict[satellite_io.LONGITUDES_KEY]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object, line_width=BORDER_WIDTH
    )

    valid_time_unix_sec = (
        satellite_dict[satellite_io.VALID_TIMES_KEY][time_index]
    )
    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT
    )
    band_number = satellite_dict[satellite_io.BAND_NUMBERS_KEY][band_index]

    if title_string is None:
        title_string = 'Band-{0:d} '.format(band_number)
        title_string += r'$T_b$'
        title_string += ' (Kelvins), {0:s} UTC'.format(valid_time_string[-4:])

    brightness_temp_matrix_kelvins = (
        satellite_dict[satellite_io.BRIGHTNESS_TEMP_KEY][
            time_index, ..., band_index
        ]
    )

    satellite_plotting.plot_2d_grid_latlng(
        brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
        axes_object=axes_object,
        min_latitude_deg_n=numpy.min(latitudes_deg_n),
        min_longitude_deg_e=numpy.min(longitudes_deg_e),
        latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0],
        cbar_orientation_string=cbar_orientation_string, font_size=FONT_SIZE
    )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=latitudes_deg_n,
        plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=2., meridian_spacing_deg=2., font_size=FONT_SIZE
    )

    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)
    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='({0:s})'.format(letter_label)
    )

    output_file_name = (
        '{0:s}/brightness_temperature_{1:s}_band{2:02d}.jpg'
    ).format(
        output_dir_name, valid_time_string, band_number
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    return output_file_name


def _plot_radar_one_time(
        reflectivity_dict, echo_classifn_dict, mask_dict, example_index,
        border_latitudes_deg_n, border_longitudes_deg_e, letter_label,
        output_dir_name):
    """Plots radar images for one time step.

    :param reflectivity_dict: See doc for `plot_radar._plot_radar_one_day`.
    :param echo_classifn_dict: Same.
    :param mask_dict: Same.
    :param example_index: Will plot [i]th example, where i = `example_index`.
    :param border_latitudes_deg_n: See doc for `plot_radar._plot_radar_one_day`.
    :param border_longitudes_deg_e: Same.
    :param letter_label: Letter label.
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :return: output_file_name: Path to output file.
    """

    latitudes_deg_n = reflectivity_dict[radar_io.LATITUDES_KEY]
    longitudes_deg_e = reflectivity_dict[radar_io.LONGITUDES_KEY]

    valid_time_unix_sec = (
        reflectivity_dict[radar_io.VALID_TIMES_KEY][example_index]
    )
    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT
    )
    reflectivity_matrix_dbz = (
        reflectivity_dict[radar_io.REFLECTIVITY_KEY][example_index, ...]
    )
    colour_map_object, colour_norm_object = (
        radar_plotting.get_default_colour_scheme(COMPOSITE_REFL_NAME)
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object, line_width=BORDER_WIDTH
    )

    matrix_to_plot = numpy.nanmax(reflectivity_matrix_dbz, axis=-1)
    title_string = (
        'Reflectivity (dBZ)' if echo_classifn_dict is None
        else 'Reflectivity (dBZ) + labels'
    )
    title_string += ', {0:s} UTC'.format(valid_time_string[-4:])

    radar_plotting.plot_latlng_grid(
        field_matrix=matrix_to_plot, field_name=COMPOSITE_REFL_NAME,
        axes_object=axes_object,
        min_grid_point_latitude_deg=numpy.min(latitudes_deg_n),
        min_grid_point_longitude_deg=numpy.min(longitudes_deg_e),
        latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0]
    )

    if mask_dict is not None:
        pyplot.contour(
            longitudes_deg_e, latitudes_deg_n,
            mask_dict[radar_io.MASK_MATRIX_KEY].astype(int),
            numpy.array([0.999]),
            colors=(MASK_OUTLINE_COLOUR,), linewidths=MASK_OUTLINE_WIDTH,
            linestyles='solid', axes=axes_object
        )

    if echo_classifn_dict is not None:
        convective_flag_matrix = echo_classifn_dict[
            radar_io.CONVECTIVE_FLAGS_KEY
        ][example_index, ...]

        row_indices, column_indices = numpy.where(convective_flag_matrix)
        positive_latitudes_deg_n = latitudes_deg_n[row_indices]
        positive_longitudes_deg_e = longitudes_deg_e[column_indices]

        plotting_utils.plot_stippling(
            x_coords=positive_longitudes_deg_e,
            y_coords=positive_latitudes_deg_n,
            figure_object=figure_object, axes_object=axes_object,
            num_grid_columns=convective_flag_matrix.shape[1]
        )

    gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=matrix_to_plot,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=True,
        font_size=FONT_SIZE
    )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=latitudes_deg_n,
        plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=2., meridian_spacing_deg=2., font_size=FONT_SIZE
    )

    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)
    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='({0:s})'.format(letter_label)
    )

    output_file_name = '{0:s}/{1:s}_{2:s}.jpg'.format(
        output_dir_name,
        'reflectivity' if echo_classifn_dict else 'echo_classifn',
        valid_time_string
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    return output_file_name


def _plot_convection_mask_one_time(
        reflectivity_dict, echo_classifn_dict, mask_dict, example_index,
        border_latitudes_deg_n, border_longitudes_deg_e, letter_label,
        output_dir_name):
    """Plots convection mask for one time step.

    :param reflectivity_dict: See doc for `plot_radar._plot_radar_one_day`.
    :param echo_classifn_dict: Same.
    :param mask_dict: Same.
    :param example_index: Will plot [i]th example, where i = `example_index`.
    :param border_latitudes_deg_n: See doc for `plot_radar._plot_radar_one_day`.
    :param border_longitudes_deg_e: Same.
    :param letter_label: Letter label.
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    :return: output_file_name: Path to output file.
    """

    latitudes_deg_n = reflectivity_dict[radar_io.LATITUDES_KEY]
    longitudes_deg_e = reflectivity_dict[radar_io.LONGITUDES_KEY]
    valid_time_unix_sec = (
        reflectivity_dict[radar_io.VALID_TIMES_KEY][example_index]
    )
    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT
    )

    target_matrix = (
        echo_classifn_dict[radar_io.CONVECTIVE_FLAGS_KEY][0, ...]
    ).astype(float)

    prob_colour_map_object, prob_colour_norm_object = (
        prediction_plotting.get_prob_colour_scheme_hail(
            make_highest_prob_black=True
        )
    )

    dummy_prediction_dict = {
        prediction_io.VALID_TIMES_KEY:
            numpy.array([valid_time_unix_sec], dtype=int),
        prediction_io.PROBABILITY_MATRIX_KEY:
            numpy.expand_dims(target_matrix, axis=(0, -1)),
        prediction_io.TARGET_MATRIX_KEY:
            numpy.full((1,) + target_matrix.shape, 0, dtype=int),
        prediction_io.LATITUDES_KEY: latitudes_deg_n,
        prediction_io.LONGITUDES_KEY: longitudes_deg_e
    }

    figure_object, axes_object = plot_predictions._plot_predictions_one_example(
        prediction_dict=dummy_prediction_dict, example_index=0,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        mask_matrix=numpy.full(target_matrix.shape, 1, dtype=bool),
        plot_deterministic=False, probability_threshold=None,
        colour_map_object=prob_colour_map_object,
        colour_norm_object=prob_colour_norm_object,
        plot_colour_bar=False, output_dir_name=None,
        font_size=FONT_SIZE, latlng_visible=True
    )[1:]

    pyplot.contour(
        reflectivity_dict[radar_io.LONGITUDES_KEY],
        reflectivity_dict[radar_io.LATITUDES_KEY],
        mask_dict[radar_io.MASK_MATRIX_KEY].astype(int),
        numpy.array([0.999]),
        colors=(MASK_OUTLINE_COLOUR,), linewidths=MASK_OUTLINE_WIDTH,
        linestyles='solid', axes=axes_object
    )

    axes_object.set_title(
        'Convection mask, {0:s} UTC'.format(valid_time_string[-4:]),
        fontsize=TITLE_FONT_SIZE
    )
    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='({0:s})'.format(letter_label)
    )
    output_file_name = '{0:s}/convection_mask_{1:s}.jpg'.format(
        output_dir_name, valid_time_string
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    return output_file_name


def _run(top_satellite_dir_name, top_reflectivity_dir_name,
         top_echo_classifn_dir_name, mask_file_name, satellite_time_strings,
         radar_time_string, output_dir_name):
    """Makes data-overview figure.

    This is effectively the main method.

    :param top_satellite_dir_name: See documentation at top of file.
    :param top_reflectivity_dir_name: Same.
    :param top_echo_classifn_dir_name: Same.
    :param mask_file_name: Same.
    :param satellite_time_strings: Same.
    :param radar_time_string: Same.
    :param output_dir_name: Same.
    """

    if mask_file_name == '':
        mask_file_name = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    satellite_time_strings.sort()
    satellite_times_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in satellite_time_strings
    ], dtype=int)

    satellite_date_strings = [
        time_conversion.unix_sec_to_string(t, DATE_FORMAT)
        for t in satellite_times_unix_sec
    ]
    satellite_date_strings = list(set(satellite_date_strings))
    satellite_date_strings.sort()

    radar_time_unix_sec = time_conversion.string_to_unix_sec(
        radar_time_string, TIME_FORMAT
    )
    radar_date_string = time_conversion.unix_sec_to_string(
        radar_time_unix_sec, DATE_FORMAT
    )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    if mask_file_name is None:
        mask_dict = None
    else:
        print('Reading mask from: "{0:s}"...'.format(mask_file_name))
        mask_dict = radar_io.read_mask_file(mask_file_name)

    satellite_file_names = [
        satellite_io.find_file(
            top_directory_name=top_satellite_dir_name,
            valid_date_string=d, prefer_zipped=False, allow_other_format=True
        )
        for d in satellite_date_strings
    ]

    reflectivity_file_name = radar_io.find_file(
        top_directory_name=top_reflectivity_dir_name,
        valid_date_string=radar_date_string,
        file_type_string=radar_io.REFL_TYPE_STRING,
        prefer_zipped=False, allow_other_format=True
    )
    echo_classifn_file_name = radar_io.find_file(
        top_directory_name=top_echo_classifn_dir_name,
        valid_date_string=radar_date_string,
        file_type_string=radar_io.ECHO_CLASSIFN_TYPE_STRING,
        prefer_zipped=False, allow_other_format=True
    )

    satellite_dicts = [dict()] * len(satellite_file_names)

    for k in range(len(satellite_file_names)):
        print('Reading data from: "{0:s}"...'.format(satellite_file_names[k]))
        satellite_dicts[k] = satellite_io.read_file(
            netcdf_file_name=satellite_file_names[k], fill_nans=False
        )

    satellite_dict = satellite_io.concat_data(satellite_dicts)
    satellite_dict = satellite_io.subset_by_time(
        satellite_dict=satellite_dict,
        desired_times_unix_sec=satellite_times_unix_sec
    )[0]
    satellite_dict = satellite_io.subset_by_band(
        satellite_dict=satellite_dict, band_numbers=BAND_NUMBERS_TO_PLOT
    )

    print('Reading data from: "{0:s}"...'.format(reflectivity_file_name))
    reflectivity_dict = radar_io.read_reflectivity_file(
        netcdf_file_name=reflectivity_file_name, fill_nans=True
    )
    reflectivity_dict = radar_io.subset_by_time(
        refl_or_echo_classifn_dict=reflectivity_dict,
        desired_times_unix_sec=numpy.array([radar_time_unix_sec], dtype=int)
    )[0]

    print('Reading data from: "{0:s}"...'.format(echo_classifn_file_name))
    echo_classifn_dict = radar_io.read_echo_classifn_file(
        echo_classifn_file_name
    )
    echo_classifn_dict = radar_io.subset_by_time(
        refl_or_echo_classifn_dict=echo_classifn_dict,
        desired_times_unix_sec=numpy.array([radar_time_unix_sec], dtype=int)
    )[0]

    num_times = len(satellite_times_unix_sec)
    num_bands = len(satellite_dict[satellite_io.BAND_NUMBERS_KEY])

    letter_label = None
    linear_index = -1

    num_panels = num_times * num_bands + 2
    panel_file_names = [''] * num_panels

    for i in range(num_times):
        for j in range(num_bands):
            linear_index += 1

            if letter_label is None:
                letter_label = 'a'
            else:
                letter_label = chr(ord(letter_label) + 1)

            panel_file_names[linear_index] = _plot_one_satellite_image(
                satellite_dict=satellite_dict, time_index=i, band_index=j,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                letter_label=letter_label, output_dir_name=output_dir_name,
                cbar_orientation_string=(
                    'vertical' if i == num_times - 1 and j == num_bands - 1
                    else None
                )
            )

    letter_label = chr(ord(letter_label) + 1)

    panel_file_names[-2] = _plot_radar_one_time(
        reflectivity_dict=reflectivity_dict,
        echo_classifn_dict=None, mask_dict=mask_dict, example_index=0,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        letter_label=letter_label, output_dir_name=output_dir_name
    )

    letter_label = chr(ord(letter_label) + 1)

    panel_file_names[-1] = _plot_convection_mask_one_time(
        reflectivity_dict=reflectivity_dict,
        echo_classifn_dict=echo_classifn_dict,
        mask_dict=mask_dict, example_index=0,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        letter_label=letter_label, output_dir_name=output_dir_name
    )

    concat_figure_file_name = '{0:s}/data_overview_{1:s}.jpg'.format(
        output_dir_name, radar_time_string
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    num_panel_columns = len(BAND_NUMBERS_TO_PLOT)
    num_panel_rows = int(numpy.ceil(
        float(num_panels) / num_panel_columns
    ))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_satellite_dir_name=getattr(
            INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME
        ),
        top_reflectivity_dir_name=getattr(
            INPUT_ARG_OBJECT, REFLECTIVITY_DIR_ARG_NAME
        ),
        top_echo_classifn_dir_name=getattr(
            INPUT_ARG_OBJECT, ECHO_CLASSIFN_DIR_ARG_NAME
        ),
        mask_file_name=getattr(INPUT_ARG_OBJECT, MASK_FILE_ARG_NAME),
        satellite_time_strings=getattr(
            INPUT_ARG_OBJECT, SATELLITE_TIMES_ARG_NAME
        ),
        radar_time_string=getattr(INPUT_ARG_OBJECT, RADAR_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
