"""Makes figure with predictions from different models."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import radar_plotting
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
FIGURE_RESOLUTION_DPI = 300

PREDICTION_DIRS_ARG_NAME = 'input_prediction_dir_names'
MODEL_DESCRIPTIONS_ARG_NAME = 'model_description_strings'
VALID_TIME_ARG_NAME = 'valid_time_string'
RADAR_DIR_ARG_NAME = 'input_radar_dir_name'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_px'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

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
SMOOTHING_RADIUS_HELP_STRING = (
    'Radius for Gaussian smoother.  If you do not want to smooth predictions, '
    'leave this alone.'
)
OUTPUT_FILE_HELP_STRING = 'Name of output file.  Figure will be saved here.'

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
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=float, required=False, default=-1,
    help=SMOOTHING_RADIUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
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
        any_radar_dict=reflectivity_dict
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

    pyplot.contour(
        longitudes_deg_e, latitudes_deg_n, mask_matrix, numpy.array([0.999]),
        colors=(MASK_OUTLINE_COLOUR,), linewidths=2, linestyles='solid',
        axes=axes_object
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
        orientation_string='vertical', extend_min=False, extend_max=True
    )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=latitudes_deg_n,
        plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=2., meridian_spacing_deg=2.
    )

    axes_object.set_title('(b) Composite reflectivity, in dBZ')


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

    pyplot.contour(
        longitudes_deg_e, latitudes_deg_n, mask_matrix, numpy.array([0.999]),
        colors=(MASK_OUTLINE_COLOUR,), linewidths=2, linestyles='solid',
        axes=axes_object
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
        parallel_spacing_deg=2., meridian_spacing_deg=2.
    )

    axes_object.set_title(title_string)

    return target_matrix, mask_matrix


def _run(top_prediction_dir_names, model_descriptions_abbrev, valid_time_string,
         top_radar_dir_name, smoothing_radius_px, output_file_name):
    """Makes figure with predictions from different models.

    This is effectively the main method.

    :param top_prediction_dir_names: See documentation at top of file.
    :param model_descriptions_abbrev: Same.
    :param valid_time_string: Same.
    :param top_radar_dir_name: Same.
    :param smoothing_radius_px: Same.
    :param output_file_name: Same.
    """

    # Process input args.
    model_descriptions_verbose = [
        s.replace('_', ' ') for s in model_descriptions_abbrev
    ]

    if top_radar_dir_name == '':
        top_radar_dir_name = None
    if smoothing_radius_px <= 0:
        smoothing_radius_px = None

    num_models = len(top_prediction_dir_names)
    num_panels = num_models + int(top_radar_dir_name is not None)

    expected_dim = numpy.array([num_models], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(model_descriptions_abbrev), exact_dimensions=expected_dim
    )

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    # Do actual stuff.
    num_panel_rows = int(numpy.ceil(
        float(num_panels) / NUM_PANEL_COLUMNS
    ))
    figure_object, axes_object_matrix = gg_plotting_utils.create_paneled_figure(
        num_rows=num_panel_rows, num_columns=NUM_PANEL_COLUMNS
    )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    letter_label = None
    target_matrix = None
    mask_matrix = None
    axes_used_matrix = numpy.full(axes_object_matrix.shape, 0, dtype=bool)

    for k in range(num_models):
        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        if top_radar_dir_name is not None and k >= 1:
            i, j = numpy.unravel_index(k + 1, axes_object_matrix.shape)
        else:
            i, j = numpy.unravel_index(k, axes_object_matrix.shape)

        axes_used_matrix[i, j] = True

        title_string = '({0:s}) {1:s}'.format(
            letter_label, model_descriptions_verbose[k]
        )

        target_matrix, mask_matrix = _plot_predictions_one_model(
            top_prediction_dir_name=top_prediction_dir_names[k],
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            valid_time_string=valid_time_string, title_string=title_string,
            figure_object=figure_object, axes_object=axes_object_matrix[i, j],
            smoothing_radius_px=smoothing_radius_px
        )

    colour_map_object, colour_norm_object = (
        prediction_plotting.get_prob_colour_scheme(1.)
    )

    gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object_matrix[-1, 0],
        data_matrix=numpy.array([0, 1], dtype=float),
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='horizontal', extend_min=False, extend_max=False
    )

    if top_radar_dir_name is not None:
        axes_used_matrix[0, 1] = True

        _plot_reflectivity(
            top_radar_dir_name=top_radar_dir_name,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            target_matrix=target_matrix, mask_matrix=mask_matrix,
            valid_time_string=valid_time_string, figure_object=figure_object,
            axes_object=axes_object_matrix[0, 1]
        )

    for i in range(num_panel_rows):
        for j in range(NUM_PANEL_COLUMNS):
            if axes_used_matrix[i, j]:
                continue

            axes_object_matrix[i, j].axis('off')

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


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
        smoothing_radius_px=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
