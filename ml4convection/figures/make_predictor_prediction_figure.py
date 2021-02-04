"""Makes figure with predictors/predictions for one t_valid and one t_lag."""

import os
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
from ml4convection.machine_learning import neural_net
from ml4convection.plotting import plotting_utils
from ml4convection.plotting import satellite_plotting
from ml4convection.plotting import prediction_plotting

TIME_FORMAT = '%Y-%m-%d-%H%M'
DATE_FORMAT = '%Y%m%d'
COMPOSITE_REFL_NAME = 'reflectivity_column_max_dbz'

MASK_OUTLINE_COLOUR = numpy.full(3, 152. / 255)

FONT_SIZE = 40
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

NUM_PANEL_COLUMNS = 3
CONCAT_FIGURE_SIZE_PX = int(1e7)

SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
RADAR_DIR_ARG_NAME = 'input_radar_dir_name'
PREDICTION_DIR_ARG_NAME = 'input_prediction_dir_name'
BAND_NUMBERS_ARG_NAME = 'band_numbers'
VALID_TIME_ARG_NAME = 'valid_time_string'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_px'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

SATELLITE_DIR_HELP_STRING = (
    'Name of directory with satellite data.  Files therein will be found by '
    '`satellite_io.find_file` and read by `satellite_io.read_file`.'
)
RADAR_DIR_HELP_STRING = (
    'Name of directory with radar data.  Files therein will be found by '
    '`radar_io.find_file` and read by `radar_io.read_reflectivity_file`.'
)
PREDICTION_DIR_HELP_STRING = (
    'Name of directory with predictions.  Files therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file`.'
)
BAND_NUMBERS_HELP_STRING = (
    'Will plot these band numbers for satellite data (list of integers).'
)
VALID_TIME_HELP_STRING = (
    'Will plot prediction at this valid time (format "yyyy-mm-dd-HHMM").'
)
SMOOTHING_RADIUS_HELP_STRING = (
    'Radius for Gaussian smoother.  If you do not want to smooth predictions, '
    'leave this alone.'
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
    '--' + RADAR_DIR_ARG_NAME, type=str, required=True,
    help=RADAR_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTION_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BAND_NUMBERS_ARG_NAME, type=int, nargs='+', required=False,
    default=satellite_io.BAND_NUMBERS, help=BAND_NUMBERS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VALID_TIME_ARG_NAME, type=str, required=True,
    help=VALID_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=float, required=False, default=-1,
    help=SMOOTHING_RADIUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_satellite_image(
        satellite_dict, time_index, band_index, border_latitudes_deg_n,
        border_longitudes_deg_e, mask_matrix, axes_object,
        cbar_orientation_string):
    """Plots brightness temperature for one band at one time.

    :param satellite_dict: Dictionary returned by `satellite_io.read_file`.
    :param time_index: Will plot the [i]th time, where i = `time_index`.
    :param band_index: Will plot the [j]th band, where j = `band_index`.
    :param border_latitudes_deg_n: See doc for `_plot_predictions`.
    :param border_longitudes_deg_e: Same.
    :param mask_matrix: Same.
    :param axes_object: Same.
    :param cbar_orientation_string: See doc for
        `satellite_plotting.plot_2d_grid`.
    """

    latitudes_deg_n = satellite_dict[satellite_io.LATITUDES_KEY]
    longitudes_deg_e = satellite_dict[satellite_io.LONGITUDES_KEY]
    band_number = satellite_dict[satellite_io.BAND_NUMBERS_KEY][band_index]

    valid_time_unix_sec = (
        satellite_dict[satellite_io.VALID_TIMES_KEY][time_index]
    )
    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT
    )
    title_string = 'Band-{0:d} brightness temp (K) at {1:s}'.format(
        band_number, valid_time_string
    )

    brightness_temp_matrix_kelvins = (
        satellite_dict[satellite_io.BRIGHTNESS_TEMP_KEY][
            time_index, ..., band_index
        ]
    )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
    )

    axes_object.contour(
        longitudes_deg_e, latitudes_deg_n, mask_matrix, numpy.array([0.999]),
        colors=(MASK_OUTLINE_COLOUR,), linewidths=2, linestyles='solid'
    )

    satellite_plotting.plot_2d_grid(
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

    axes_object.set_title(title_string)


def _plot_reflectivity(
        top_radar_dir_name, border_latitudes_deg_n, border_longitudes_deg_e,
        target_matrix, mask_matrix, valid_time_string, figure_object,
        axes_object):
    """Plots composite reflectivity.

    :param top_radar_dir_name: See documentation at top of file.
    :param border_latitudes_deg_n: See doc for `_plot_predictions`.
    :param border_longitudes_deg_e: Same.
    :param target_matrix: Same.
    :param mask_matrix: Same.
    :param valid_time_string: Same.
    :param figure_object: Same.
    :param axes_object: Same.
    """

    reflectivity_file_name = radar_io.find_file(
        top_directory_name=top_radar_dir_name,
        valid_date_string=valid_time_string[:10].replace('-', ''),
        file_type_string=radar_io.REFL_TYPE_STRING,
        prefer_zipped=True, allow_other_format=True,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(reflectivity_file_name))
    reflectivity_dict = radar_io.read_reflectivity_file(
        reflectivity_file_name
    )

    valid_time_unix_sec = time_conversion.string_to_unix_sec(
        valid_time_string, TIME_FORMAT
    )
    reflectivity_dict = radar_io.subset_by_time(
        refl_or_echo_classifn_dict=reflectivity_dict,
        desired_times_unix_sec=numpy.array([valid_time_unix_sec], dtype=int)
    )[0]

    reflectivity_dict = radar_io.expand_to_satellite_grid(
        any_radar_dict=reflectivity_dict, fill_nans=True
    )

    latitudes_deg_n = reflectivity_dict[radar_io.LATITUDES_KEY]
    longitudes_deg_e = reflectivity_dict[radar_io.LONGITUDES_KEY]
    reflectivity_matrix_dbz = numpy.nanmax(
        reflectivity_dict[radar_io.REFLECTIVITY_KEY][0, ...], axis=-1
    )

    colour_map_object, colour_norm_object = (
        radar_plotting.get_default_colour_scheme(COMPOSITE_REFL_NAME)
    )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
    )

    axes_object.contour(
        longitudes_deg_e, latitudes_deg_n, mask_matrix, numpy.array([0.999]),
        colors=(MASK_OUTLINE_COLOUR,), linewidths=2, linestyles='solid'
    )

    radar_plotting.plot_latlng_grid(
        field_matrix=reflectivity_matrix_dbz, field_name=COMPOSITE_REFL_NAME,
        axes_object=axes_object,
        min_grid_point_latitude_deg=numpy.min(latitudes_deg_n),
        min_grid_point_longitude_deg=numpy.min(longitudes_deg_e),
        latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0]
    )

    row_indices, column_indices = numpy.where(target_matrix)
    positive_latitudes_deg_n = latitudes_deg_n[row_indices]
    positive_longitudes_deg_e = longitudes_deg_e[column_indices]

    plotting_utils.plot_stippling(
        x_coords=positive_longitudes_deg_e,
        y_coords=positive_latitudes_deg_n,
        figure_object=figure_object, axes_object=axes_object,
        num_grid_columns=target_matrix.shape[1]
    )

    gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=reflectivity_matrix_dbz,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='horizontal', extend_min=False, extend_max=True,
        font_size=FONT_SIZE
    )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=latitudes_deg_n,
        plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=2., meridian_spacing_deg=2., font_size=FONT_SIZE
    )

    title_string = 'Composite reflectivity (dBZ) at {0:s}'.format(
        valid_time_string
    )
    axes_object.set_title(title_string)


def _plot_predictions(
        top_prediction_dir_name, border_latitudes_deg_n,
        border_longitudes_deg_e, valid_time_string, title_string,
        figure_object, axes_object, smoothing_radius_px=None):
    """Plots predictions (convection probabilities).

    P = number of points in border set
    M = number of rows in grid
    N = number of columns in grid

    :param top_prediction_dir_name: Name of top-level directory with
        predictions.
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg N).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg E).
    :param valid_time_string: See documentation at top of file.
    :param title_string: Figure title.
    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param smoothing_radius_px: Same.
    :return: target_matrix: M-by-N numpy array of Boolean flags, True where
        convection actually occurs.
    :return: model_metadata_dict: Dictionary returned by
        `neural_net.read_metafile`.
    """

    prediction_file_name = prediction_io.find_file(
        top_directory_name=top_prediction_dir_name,
        valid_date_string=valid_time_string[:10].replace('-', ''),
        radar_number=None, prefer_zipped=True, allow_other_format=True,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    valid_time_unix_sec = time_conversion.string_to_unix_sec(
        valid_time_string, TIME_FORMAT
    )
    prediction_dict = prediction_io.subset_by_time(
        prediction_dict=prediction_dict,
        desired_times_unix_sec=numpy.array([valid_time_unix_sec], dtype=int)
    )[0]

    if smoothing_radius_px is not None:
        prediction_dict = prediction_io.smooth_probabilities(
            prediction_dict=prediction_dict,
            smoothing_radius_px=smoothing_radius_px
        )

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(model_file_name)

    print('Reading model metadata from: "{0:s}"...'.format(
        model_metafile_name
    ))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    mask_matrix = model_metadata_dict[neural_net.FULL_MASK_MATRIX_KEY]

    target_matrix = prediction_dict[prediction_io.TARGET_MATRIX_KEY][0, ...]
    target_matrix[mask_matrix == False] = 0
    probability_matrix = (
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][0, ...]
    )

    latitudes_deg_n = prediction_dict[prediction_io.LATITUDES_KEY]
    longitudes_deg_e = prediction_dict[prediction_io.LONGITUDES_KEY]

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
    )

    axes_object.contour(
        longitudes_deg_e, latitudes_deg_n, mask_matrix, numpy.array([0.999]),
        colors=(MASK_OUTLINE_COLOUR,), linewidths=2, linestyles='solid'
    )

    prediction_plotting.plot_probabilistic(
        target_matrix=target_matrix, probability_matrix=probability_matrix,
        figure_object=figure_object, axes_object=axes_object,
        min_latitude_deg_n=latitudes_deg_n[0],
        min_longitude_deg_e=longitudes_deg_e[0],
        latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0],
        max_prob_in_colour_bar=1.
    )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=latitudes_deg_n,
        plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=2., meridian_spacing_deg=2., font_size=FONT_SIZE
    )

    axes_object.set_title(title_string)

    return target_matrix, model_metadata_dict


def _run(top_satellite_dir_name, top_radar_dir_name, top_prediction_dir_name,
         band_numbers, valid_time_string, smoothing_radius_px, output_dir_name):
    """Makes figure with predictors/predictions for one t_valid and one t_lag.

    This is effectively the main method.

    :param top_satellite_dir_name: See documentation at top of file.
    :param top_radar_dir_name: Same.
    :param top_prediction_dir_name: Same.
    :param band_numbers: Same.
    :param valid_time_string: Same.
    :param smoothing_radius_px: Same.
    :param output_dir_name: Same.
    """

    if smoothing_radius_px <= 0:
        smoothing_radius_px = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    valid_time_unix_sec = time_conversion.string_to_unix_sec(
        valid_time_string, TIME_FORMAT
    )
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    # Plot forecast probabilities.
    num_bands = len(band_numbers)
    num_panels = num_bands + 2
    panel_file_names = [''] * num_panels

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    title_string = 'Convection probabilities at {0:s}'.format(valid_time_string)

    target_matrix, model_metadata_dict = _plot_predictions(
        top_prediction_dir_name=top_prediction_dir_name,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        valid_time_string=valid_time_string, title_string=title_string,
        figure_object=figure_object, axes_object=axes_object,
        smoothing_radius_px=smoothing_radius_px
    )

    colour_map_object, colour_norm_object = (
        prediction_plotting.get_prob_colour_scheme(1.)
    )

    gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=numpy.array([0, 1], dtype=float),
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='horizontal',
        extend_min=False, extend_max=False, font_size=FONT_SIZE
    )

    letter_label = chr(ord('a') + num_bands + 1)
    gg_plotting_utils.label_axes(
        axes_object=axes_object,
        label_string='({0:s})'.format(letter_label)
    )

    panel_file_names[-1] = '{0:s}/{1:s}_predictions.jpg'.format(
        output_dir_name, valid_time_string
    )

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot composite reflectivity.
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    mask_matrix = model_metadata_dict[neural_net.FULL_MASK_MATRIX_KEY]

    _plot_reflectivity(
        top_radar_dir_name=top_radar_dir_name,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        target_matrix=target_matrix, mask_matrix=mask_matrix,
        valid_time_string=valid_time_string,
        figure_object=figure_object, axes_object=axes_object
    )

    letter_label = chr(ord('a') + num_bands)
    gg_plotting_utils.label_axes(
        axes_object=axes_object,
        label_string='({0:s})'.format(letter_label)
    )

    panel_file_names[-2] = '{0:s}/{1:s}_reflectivity.jpg'.format(
        output_dir_name, valid_time_string
    )

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-2]))
    figure_object.savefig(
        panel_file_names[-2], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot brightness temperatures (predictors).
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    lead_time_seconds = training_option_dict[neural_net.LEAD_TIME_KEY]
    lag_times_seconds = training_option_dict[neural_net.LAG_TIMES_KEY]

    satellite_times_unix_sec = (
        valid_time_unix_sec - lead_time_seconds - lag_times_seconds
    )
    satellite_date_strings = [
        time_conversion.unix_sec_to_string(t, DATE_FORMAT)
        for t in satellite_times_unix_sec
    ]
    satellite_date_strings = list(set(satellite_date_strings))

    satellite_file_names = [
        satellite_io.find_file(
            top_directory_name=top_satellite_dir_name, valid_date_string=d,
            prefer_zipped=True, allow_other_format=True,
            raise_error_if_missing=True
        ) for d in satellite_date_strings
    ]

    satellite_dicts = []

    for this_file_name in satellite_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_satellite_dict = satellite_io.read_file(
            netcdf_file_name=this_file_name, fill_nans=False
        )
        this_satellite_dict = satellite_io.subset_by_band(
            satellite_dict=this_satellite_dict, band_numbers=band_numbers
        )

        satellite_dicts.append(this_satellite_dict)

    satellite_dict = satellite_io.concat_data(satellite_dicts)
    satellite_dict = satellite_io.subset_by_time(
        satellite_dict=satellite_dict,
        desired_times_unix_sec=satellite_times_unix_sec
    )[0]

    num_lag_times = len(lag_times_seconds)
    num_panel_rows = int(numpy.ceil(
        float(num_panels) / NUM_PANEL_COLUMNS
    ))

    for i in range(num_lag_times):
        for j in range(num_bands):
            figure_object, axes_object = pyplot.subplots(
                1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
            )

            _plot_one_satellite_image(
                satellite_dict=satellite_dict, time_index=i, band_index=j,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                mask_matrix=mask_matrix, axes_object=axes_object,
                cbar_orientation_string=(
                    'horizontal' if j == num_bands - 1 else ''
                )
            )

            letter_label = chr(ord('a') + j)
            gg_plotting_utils.label_axes(
                axes_object=axes_object,
                label_string='({0:s})'.format(letter_label)
            )

            panel_file_names[j] = (
                '{0:s}/{1:s}_satellite_band{2:02d}_lag-time-sec={3:05d}.jpg'
            ).format(
                output_dir_name, valid_time_string, band_numbers[j],
                lag_times_seconds[i]
            )

            print('Saving figure to file: "{0:s}"...'.format(
                panel_file_names[j]
            ))
            figure_object.savefig(
                panel_file_names[j], dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

        concat_figure_file_name = (
            '{0:s}/{1:s}_concat_lag-time-sec={2:05d}.jpg'
        ).format(
            output_dir_name, valid_time_string, lag_times_seconds[i]
        )
        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))

        imagemagick_utils.concatenate_images(
            input_file_names=panel_file_names,
            output_file_name=concat_figure_file_name,
            num_panel_rows=num_panel_rows, num_panel_columns=NUM_PANEL_COLUMNS
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

        for j in range(num_bands):
            os.remove(panel_file_names[j])

    os.remove(panel_file_names[-2])
    os.remove(panel_file_names[-1])


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_satellite_dir_name=getattr(
            INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME
        ),
        top_radar_dir_name=getattr(INPUT_ARG_OBJECT, RADAR_DIR_ARG_NAME),
        top_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME
        ),
        band_numbers=numpy.array(
            getattr(INPUT_ARG_OBJECT, BAND_NUMBERS_ARG_NAME), dtype=int
        ),
        valid_time_string=getattr(INPUT_ARG_OBJECT, VALID_TIME_ARG_NAME),
        smoothing_radius_px=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
