"""Makes figure with predictions from different models."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from ml4convection.io import radar_io
from ml4convection.io import border_io
from ml4convection.io import prediction_io
from ml4convection.machine_learning import neural_net
from ml4convection.plotting import plotting_utils
from ml4convection.plotting import prediction_plotting

TIME_FORMAT = '%Y-%m-%d-%H%M'
COMPOSITE_REFL_NAME = 'reflectivity_column_max_dbz'

MASK_OUTLINE_COLOUR = numpy.full(3, 152. / 255)

NUM_PANEL_COLUMNS = 2

FIGURE_WIDTH_INCHES = 15.
FIGURE_HEIGHT_INCHES = 15.
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

FONT_SIZE = 50
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

PREDICTION_DIRS_ARG_NAME = 'input_prediction_dir_names'
MODEL_DESCRIPTIONS_ARG_NAME = 'model_description_strings'
VALID_TIME_ARG_NAME = 'valid_time_string'
RADAR_DIR_ARG_NAME = 'input_radar_dir_name'
SMOOTHING_RADII_ARG_NAME = 'smoothing_radii_px'
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
        orientation_string='vertical', extend_min=False, extend_max=True,
        font_size=FONT_SIZE
    )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=latitudes_deg_n,
        plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=2., meridian_spacing_deg=2., font_size=FONT_SIZE
    )

    axes_object.set_title('Composite reflectivity (dBZ)')
    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='(b)', x_coord_normalized=-0.025
    )


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
        parallel_spacing_deg=2., meridian_spacing_deg=2., font_size=FONT_SIZE
    )

    axes_object.set_title(title_string)

    return target_matrix, mask_matrix


def _run(top_prediction_dir_names, model_descriptions_abbrev, valid_time_string,
         top_radar_dir_name, smoothing_radii_px, output_dir_name):
    """Makes figure with predictions from different models.

    This is effectively the main method.

    :param top_prediction_dir_names: See documentation at top of file.
    :param model_descriptions_abbrev: Same.
    :param valid_time_string: Same.
    :param top_radar_dir_name: Same.
    :param smoothing_radii_px: Same.
    :param output_dir_name: Same.
    """

    # Process input args.
    model_descriptions_verbose = [
        s.replace('_', ' ') for s in model_descriptions_abbrev
    ]
    model_descriptions_abbrev = [
        s.replace('_', '-').lower() for s in model_descriptions_abbrev
    ]

    if top_radar_dir_name == '':
        top_radar_dir_name = None

    num_models = len(top_prediction_dir_names)
    num_panels = num_models + int(top_radar_dir_name is not None)

    expected_dim = numpy.array([num_models], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(model_descriptions_abbrev), exact_dimensions=expected_dim
    )
    error_checking.assert_is_numpy_array(
        smoothing_radii_px, exact_dimensions=expected_dim
    )

    smoothing_radii_px = [r if r >= 0 else None for r in smoothing_radii_px]

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Do actual stuff.
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    target_matrix = None
    mask_matrix = None
    panel_file_names = [''] * num_panels

    for k in range(num_models):
        if top_radar_dir_name is not None and k >= 1:
            panel_index = k + 1
            letter_label = chr(ord('a') + k + 1)
        else:
            panel_index = k + 0
            letter_label = chr(ord('a') + k)

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

        gg_plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(letter_label),
            x_coord_normalized=-0.025
        )

        if k == num_models - 1:
            colour_map_object, colour_norm_object = (
                prediction_plotting.get_prob_colour_scheme(
                    max_probability=1., make_lowest_prob_grey=False
                )
            )

            gg_plotting_utils.plot_colour_bar(
                axes_object_or_matrix=axes_object,
                data_matrix=numpy.array([0, 1], dtype=float),
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object,
                orientation_string='vertical',
                extend_min=False, extend_max=False, font_size=FONT_SIZE
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

        panel_file_names[1] = '{0:s}/predictions_{1:s}_radar.jpg'.format(
            output_dir_name, valid_time_string
        )

        print('Saving figure to file: "{0:s}"...'.format(panel_file_names[1]))
        figure_object.savefig(
            panel_file_names[1], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    concat_figure_file_name = '{0:s}/predictions_{1:s}_comparison.jpg'.format(
        output_dir_name, valid_time_string
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    num_panel_rows = int(numpy.ceil(
        float(num_panels) / NUM_PANEL_COLUMNS
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
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
