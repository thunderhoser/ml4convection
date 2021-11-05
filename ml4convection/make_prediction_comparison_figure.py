"""Makes figure with predictions from different models."""

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
import error_checking
import radar_plotting
import imagemagick_utils
import gg_plotting_utils
import radar_io
import border_io
import prediction_io
import neural_net
import plotting_utils
import prediction_plotting

TIME_FORMAT = '%Y-%m-%d-%H%M'
COMPOSITE_REFL_NAME = 'reflectivity_column_max_dbz'

MASK_OUTLINE_COLOUR = numpy.full(3, 152. / 255)

MIN_PLOT_LATITUDE_DEG_N = 21.
MAX_PLOT_LATITUDE_DEG_N = 25.
MIN_PLOT_LONGITUDE_DEG_E = 119.
MAX_PLOT_LONGITUDE_DEG_E = 123.

FIGURE_WIDTH_INCHES = 15.
FIGURE_HEIGHT_INCHES = 15.
FIGURE_RESOLUTION_DPI = 300
PANEL_FIGURE_SIZE_PX = int(3e5)
CONCAT_FIGURE_SIZE_PX = int(1e7)

DEFAULT_FONT_SIZE = 60
LATLNG_FONT_SIZE = 50

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

PREDICTION_DIRS_ARG_NAME = 'input_prediction_dir_names'
MODEL_DESCRIPTIONS_ARG_NAME = 'model_description_strings'
VALID_TIME_ARG_NAME = 'valid_time_string'
RADAR_DIR_ARG_NAME = 'input_radar_dir_name'
SMOOTHING_RADII_ARG_NAME = 'smoothing_radii_px'
PANEL_LETTERS_ARG_NAME = 'panel_letters'
NUM_PANEL_ROWS_ARG_NAME = 'num_panel_rows'
NUM_PANEL_COLUMNS_ARG_NAME = 'num_panel_columns'
RADAR_INDEX_ARG_NAME = 'radar_panel_index'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_DIRS_HELP_STRING = (
    'List of input directories, one per model.  Within each directory, the '
    'prediction file will be found by `prediction_io.find_file` and read by '
    '`prediction_io.read_file`.'
)
MODEL_DESCRIPTIONS_HELP_STRING = (
    'Space-separated list of model descriptions (one per model), to be used as '
    'panel titles.  Underscores will be replaced by spaces.'
)
VALID_TIME_HELP_STRING = (
    'Will plot predictions for this valid time (format "yyyy-mm-dd-HHMM").'
)
RADAR_DIR_HELP_STRING = (
    'Name of directory with radar data.  The relevant file will be found by '
    '`radar_io.read_file` and read by `radar_io.read_reflectivity_file`.  If '
    'you do not want to plot radar data, leave this alone.'
)
SMOOTHING_RADII_HELP_STRING = (
    'Radii for Gaussian smoother (one per model).  Use non-positive number for '
    'no smoothing.'
)
PANEL_LETTERS_HELP_STRING = (
    'Space-separated list of letters (one per panel) used for labels.  Num '
    'panels = num models + (1 if plotting radar, else 0).'
)
NUM_PANEL_ROWS_HELP_STRING = (
    'Number of panel rows.  If you want number of rows to be determined '
    'automatically, leave this argument alone.'
)
NUM_PANEL_COLUMNS_HELP_STRING = 'Same as `{0:s}` but for columns.'.format(
    NUM_PANEL_ROWS_ARG_NAME
)
RADAR_INDEX_HELP_STRING = (
    '[used only if `{0:s}` is specified] Array index of panel with radar data.'
).format(RADAR_DIR_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_DIRS_ARG_NAME, nargs='+', type=str, required=True,
    help=PREDICTION_DIRS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_DESCRIPTIONS_ARG_NAME, nargs='+', type=str, required=True,
    help=MODEL_DESCRIPTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VALID_TIME_ARG_NAME, type=str, required=True,
    help=VALID_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_DIR_ARG_NAME, type=str, required=False, default='',
    help=RADAR_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADII_ARG_NAME, type=float, nargs='+', required=True,
    help=SMOOTHING_RADII_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PANEL_LETTERS_ARG_NAME, nargs='+', type=str, required=True,
    help=PANEL_LETTERS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PANEL_ROWS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_PANEL_ROWS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PANEL_COLUMNS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_PANEL_COLUMNS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_INDEX_ARG_NAME, type=int, required=False, default=-1,
    help=RADAR_INDEX_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_reflectivity(
        top_radar_dir_name, border_latitudes_deg_n, border_longitudes_deg_e,
        target_matrix, mask_matrix, valid_time_string, figure_object,
        axes_object):
    """Plots composite reflectivity.

    :param top_radar_dir_name: See documentation at top of file.
    :param border_latitudes_deg_n: See doc for `_plot_predictions_one_model`.
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

    # reflectivity_dict = radar_io.expand_to_satellite_grid(
    #     any_radar_dict=reflectivity_dict, fill_nans=True
    # )

    good_lat_indices = numpy.where(numpy.logical_and(
        reflectivity_dict[radar_io.LATITUDES_KEY] >= MIN_PLOT_LATITUDE_DEG_N,
        reflectivity_dict[radar_io.LATITUDES_KEY] <= MAX_PLOT_LATITUDE_DEG_N
    ))[0]

    good_lng_indices = numpy.where(numpy.logical_and(
        reflectivity_dict[radar_io.LONGITUDES_KEY] >= MIN_PLOT_LONGITUDE_DEG_E,
        reflectivity_dict[radar_io.LONGITUDES_KEY] <= MAX_PLOT_LONGITUDE_DEG_E
    ))[0]

    reflectivity_dict[radar_io.LATITUDES_KEY] = (
        reflectivity_dict[radar_io.LATITUDES_KEY][good_lat_indices]
    )
    reflectivity_dict[radar_io.LONGITUDES_KEY] = (
        reflectivity_dict[radar_io.LONGITUDES_KEY][good_lng_indices]
    )
    reflectivity_dict[radar_io.REFLECTIVITY_KEY] = (
        reflectivity_dict[radar_io.REFLECTIVITY_KEY][:, good_lat_indices, ...]
    )
    reflectivity_dict[radar_io.REFLECTIVITY_KEY] = (
        reflectivity_dict[radar_io.REFLECTIVITY_KEY][
            :, :, good_lng_indices, ...
        ]
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
        orientation_string='vertical', extend_min=False, extend_max=True,
        font_size=DEFAULT_FONT_SIZE
    )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=latitudes_deg_n,
        plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=2., meridian_spacing_deg=2.,
        font_size=LATLNG_FONT_SIZE
    )

    axes_object.set_title('Reflectivity (dBZ)')


def _plot_predictions_one_model(
        top_prediction_dir_name, border_latitudes_deg_n,
        border_longitudes_deg_e, valid_time_string, title_string,
        figure_object, axes_object, smoothing_radius_px=None):
    """Plots predictions (and targets) for one model.

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
    :return: mask_matrix: M-by-N numpy array of Boolean flags, True for unmasked
        grid points.
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

    good_lat_indices = numpy.where(numpy.logical_and(
        prediction_dict[prediction_io.LATITUDES_KEY] >= MIN_PLOT_LATITUDE_DEG_N,
        prediction_dict[prediction_io.LATITUDES_KEY] <= MAX_PLOT_LATITUDE_DEG_N
    ))[0]

    good_lng_indices = numpy.where(numpy.logical_and(
        prediction_dict[prediction_io.LONGITUDES_KEY] >=
        MIN_PLOT_LONGITUDE_DEG_E,
        prediction_dict[prediction_io.LONGITUDES_KEY] <=
        MAX_PLOT_LONGITUDE_DEG_E
    ))[0]

    prediction_dict[prediction_io.LATITUDES_KEY] = (
        prediction_dict[prediction_io.LATITUDES_KEY][good_lat_indices]
    )
    prediction_dict[prediction_io.LONGITUDES_KEY] = (
        prediction_dict[prediction_io.LONGITUDES_KEY][good_lng_indices]
    )

    these_keys = [
        prediction_io.TARGET_MATRIX_KEY, prediction_io.PROBABILITY_MATRIX_KEY
    ]
    for this_key in these_keys:
        prediction_dict[this_key] = (
            prediction_dict[this_key][:, good_lat_indices, :]
        )
        prediction_dict[this_key] = (
            prediction_dict[this_key][..., good_lng_indices]
        )

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
    mask_matrix = mask_matrix[good_lat_indices, :]
    mask_matrix = mask_matrix[:, good_lng_indices]

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

    colour_map_object, colour_norm_object = (
        prediction_plotting.get_prob_colour_scheme(
            max_probability=1., make_lowest_prob_grey=False
        )
    )

    prediction_plotting.plot_probabilistic(
        target_matrix=target_matrix, probability_matrix=probability_matrix,
        figure_object=figure_object, axes_object=axes_object,
        min_latitude_deg_n=latitudes_deg_n[0],
        min_longitude_deg_e=longitudes_deg_e[0],
        latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object
    )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=latitudes_deg_n,
        plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=2., meridian_spacing_deg=2.,
        font_size=LATLNG_FONT_SIZE
    )

    axes_object.set_title(title_string)

    return target_matrix, mask_matrix


def _run(top_prediction_dir_names, model_descriptions_abbrev, valid_time_string,
         top_radar_dir_name, smoothing_radii_px, panel_letters, num_panel_rows,
         num_panel_columns, radar_panel_index, output_dir_name):
    """Makes figure with predictions from different models.

    This is effectively the main method.

    :param top_prediction_dir_names: See documentation at top of file.
    :param model_descriptions_abbrev: Same.
    :param valid_time_string: Same.
    :param top_radar_dir_name: Same.
    :param smoothing_radii_px: Same.
    :param panel_letters: Same.
    :param num_panel_rows: Same.
    :param num_panel_columns: Same.
    :param radar_panel_index: Same.
    :param output_dir_name: Same.
    """

    # Process input args.
    model_descriptions_verbose = [
        s.replace('_', ' ') for s in model_descriptions_abbrev
    ]
    model_descriptions_verbose = [
        s.replace('inf', r'$\infty$') for s in model_descriptions_verbose
    ]
    model_descriptions_abbrev = [
        s.replace('_', '-').lower() for s in model_descriptions_abbrev
    ]

    if top_radar_dir_name == '':
        top_radar_dir_name = None

    num_models = len(top_prediction_dir_names)
    num_panels = num_models + int(top_radar_dir_name is not None)

    if top_radar_dir_name is not None:
        error_checking.assert_is_geq(radar_panel_index, 0)
        error_checking.assert_is_less_than(radar_panel_index, num_panels)

    if num_panel_rows < 0:
        num_panel_rows = int(numpy.ceil(
            numpy.sqrt(num_panels)
        ))
    if num_panel_columns < 0:
        num_panel_columns = int(numpy.ceil(
            float(num_panels) / num_panel_rows
        ))

    expected_dim = numpy.array([num_models], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(model_descriptions_abbrev), exact_dimensions=expected_dim
    )
    error_checking.assert_is_numpy_array(
        smoothing_radii_px, exact_dimensions=expected_dim
    )

    expected_dim = numpy.array([num_panels], dtype=int)
    print(panel_letters)
    error_checking.assert_is_numpy_array(
        numpy.array(panel_letters), exact_dimensions=expected_dim
    )

    smoothing_radii_px = [r if r >= 0 else None for r in smoothing_radii_px]

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Do actual stuff.
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    while len(panel_letters) < num_panel_rows * num_panel_columns:
        panel_letters.append('')

    panel_letter_matrix = numpy.reshape(
        numpy.array(panel_letters), (num_panel_rows, num_panel_columns),
    )

    target_matrix = None
    mask_matrix = None
    panel_file_names = [''] * num_panels

    for k in range(num_models):
        if top_radar_dir_name is not None and k >= radar_panel_index:
            panel_index = k + 1
        else:
            panel_index = k + 0

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        target_matrix, mask_matrix = _plot_predictions_one_model(
            top_prediction_dir_name=top_prediction_dir_names[k],
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            valid_time_string=valid_time_string,
            title_string=model_descriptions_verbose[k],
            figure_object=figure_object, axes_object=axes_object,
            smoothing_radius_px=smoothing_radii_px[k]
        )

        row_index, column_index = numpy.unravel_index(
            panel_index, (num_panel_rows, num_panel_columns)
        )
        gg_plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(
                panel_letter_matrix[row_index, column_index]
            ),
            x_coord_normalized=-0.025, font_size=DEFAULT_FONT_SIZE
        )

        if k == num_models - 1:
            colour_map_object, colour_norm_object = (
                prediction_plotting.get_prob_colour_scheme(
                    max_probability=1., make_lowest_prob_grey=False
                )
            )

            if top_radar_dir_name is None:
                orientation_string = 'vertical'
            else:
                radar_row_index, radar_column_index = numpy.unravel_index(
                    radar_panel_index, (num_panel_rows, num_panel_columns)
                )

                if (
                        radar_row_index == row_index and
                        radar_column_index == column_index + 1
                ):
                    orientation_string = 'horizontal'
                else:
                    orientation_string = 'vertical'

            gg_plotting_utils.plot_colour_bar(
                axes_object_or_matrix=axes_object,
                data_matrix=numpy.array([0, 1], dtype=float),
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object,
                orientation_string=orientation_string,
                extend_min=False, extend_max=False, font_size=DEFAULT_FONT_SIZE
            )

        panel_file_names[panel_index] = (
            '{0:s}/predictions_{1:s}_{2:s}.jpg'
        ).format(
            output_dir_name, valid_time_string, model_descriptions_abbrev[k]
        )

        print('Saving figure to file: "{0:s}"...'.format(
            panel_file_names[panel_index]
        ))
        figure_object.savefig(
            panel_file_names[panel_index], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        imagemagick_utils.resize_image(
            input_file_name=panel_file_names[panel_index],
            output_file_name=panel_file_names[panel_index],
            output_size_pixels=PANEL_FIGURE_SIZE_PX
        )

    if top_radar_dir_name is not None:
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )

        _plot_reflectivity(
            top_radar_dir_name=top_radar_dir_name,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            target_matrix=target_matrix, mask_matrix=mask_matrix,
            valid_time_string=valid_time_string,
            figure_object=figure_object, axes_object=axes_object
        )

        row_index, column_index = numpy.unravel_index(
            radar_panel_index, (num_panel_rows, num_panel_columns)
        )
        gg_plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(
                panel_letter_matrix[row_index, column_index]
            ),
            x_coord_normalized=-0.025, font_size=DEFAULT_FONT_SIZE
        )

        panel_file_names[radar_panel_index] = (
            '{0:s}/predictions_{1:s}_radar.jpg'
        ).format(output_dir_name, valid_time_string)

        print('Saving figure to file: "{0:s}"...'.format(
            panel_file_names[radar_panel_index]
        ))
        figure_object.savefig(
            panel_file_names[radar_panel_index], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        imagemagick_utils.resize_image(
            input_file_name=panel_file_names[radar_panel_index],
            output_file_name=panel_file_names[radar_panel_index],
            output_size_pixels=PANEL_FIGURE_SIZE_PX
        )

    while len(panel_file_names) < num_panel_rows * num_panel_columns:
        panel_file_names.append('')

    panel_file_name_matrix = numpy.reshape(
        numpy.array(panel_file_names), (num_panel_rows, num_panel_columns)
    )
    panel_file_names = numpy.reshape(
        panel_file_name_matrix, panel_file_name_matrix.size, order='C'
    )
    panel_file_names = panel_file_names[:num_panels]

    concat_figure_file_name = '{0:s}/predictions_{1:s}_comparison.jpg'.format(
        output_dir_name, valid_time_string
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns,
        border_width_pixels=15
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

    for k in range(num_panels):
        os.remove(panel_file_names[k])


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_names=getattr(
            INPUT_ARG_OBJECT, PREDICTION_DIRS_ARG_NAME
        ),
        model_descriptions_abbrev=getattr(
            INPUT_ARG_OBJECT, MODEL_DESCRIPTIONS_ARG_NAME
        ),
        valid_time_string=getattr(INPUT_ARG_OBJECT, VALID_TIME_ARG_NAME),
        top_radar_dir_name=getattr(INPUT_ARG_OBJECT, RADAR_DIR_ARG_NAME),
        smoothing_radii_px=numpy.array(
            getattr(INPUT_ARG_OBJECT, SMOOTHING_RADII_ARG_NAME), dtype=float
        ),
        panel_letters=getattr(INPUT_ARG_OBJECT, PANEL_LETTERS_ARG_NAME),
        num_panel_rows=getattr(INPUT_ARG_OBJECT, NUM_PANEL_ROWS_ARG_NAME),
        num_panel_columns=getattr(INPUT_ARG_OBJECT, NUM_PANEL_COLUMNS_ARG_NAME),
        radar_panel_index=getattr(INPUT_ARG_OBJECT, RADAR_INDEX_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
