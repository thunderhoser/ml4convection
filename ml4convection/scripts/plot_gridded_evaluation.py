"""Plots evaluation scores on grid."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import radar_plotting
from ml4convection.io import border_io
from ml4convection.utils import evaluation
from ml4convection.plotting import plotting_utils

TOLERANCE = 1e-6
DUMMY_FIELD_NAME = 'reflectivity_column_max_dbz'

MAX_COLOUR_PERCENTILE = 99.

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

INPUT_FILE_ARG_NAME = 'input_advanced_score_file_name'
SEQ_COLOUR_MAP_ARG_NAME = 'sequential_colour_map_name'
DIV_COLOUR_MAP_ARG_NAME = 'diverging_colour_map_name'
BIAS_COLOUR_MAP_ARG_NAME = 'bias_colour_map_name'
PROB_THRESHOLD_ARG_NAME = 'probability_threshold'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to file with advanced evaluation scores.  Will be read by '
    '`evaluation.read_advanced_score_file`.'
)
SEQ_COLOUR_MAP_HELP_STRING = (
    'Name of sequential colour map (must be accepted by '
    '`matplotlib.pyplot.get_cmap`).  Will be used for POD, success ratio, CSI, '
    'Brier score, and climatological event frequency.'
)
DIV_COLOUR_MAP_HELP_STRING = (
    'Name of diverging colour map (must be accepted by '
    '`matplotlib.pyplot.get_cmap`).  Will be used for Brier skill score and '
    'fractions skill score.'
)
BIAS_COLOUR_MAP_HELP_STRING = (
    'Name of colour map for frequency bias (must be accepted by '
    '`matplotlib.pyplot.get_cmap`).'
)
PROB_THRESHOLD_HELP_STRING = (
    'Probability threshold used to compute POD, success ratio, CSI, and bias.  '
    'If you do not want to plot the aforelisted scores, leave this argument '
    'alone.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SEQ_COLOUR_MAP_ARG_NAME, type=str, required=False, default='plasma',
    help=SEQ_COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DIV_COLOUR_MAP_ARG_NAME, type=str, required=False, default='seismic',
    help=DIV_COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BIAS_COLOUR_MAP_ARG_NAME, type=str, required=False,
    default='seismic', help=BIAS_COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PROB_THRESHOLD_ARG_NAME, type=float, required=False, default=-1,
    help=PROB_THRESHOLD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_bias_colour_scheme(colour_map_name, max_colour_value):
    """Returns colour scheme for frequency bias.

    :param colour_map_name: Name of colour scheme (must be accepted by
        `matplotlib.pyplot.get_cmap`).
    :param max_colour_value: Max value in colour scheme.
    :return: colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :return: colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    """

    orig_colour_map_object = pyplot.get_cmap(colour_map_name)

    negative_values = numpy.linspace(0, 1, num=1001, dtype=float)
    positive_values = numpy.linspace(1, max_colour_value, num=1001, dtype=float)
    bias_values = numpy.concatenate((negative_values, positive_values))

    normalized_values = numpy.linspace(0, 1, num=len(bias_values), dtype=float)
    rgb_matrix = orig_colour_map_object(normalized_values)[:, :-1]

    colour_map_object = matplotlib.colors.ListedColormap(rgb_matrix)
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        bias_values, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def _plot_one_score(
        score_matrix, advanced_score_table_xarray, border_latitudes_deg_n,
        border_longitudes_deg_e, colour_map_name, is_frequency_bias,
        maybe_negative, output_file_name, title_string=None, panel_letter=None):
    """Plots one score on 2-D georeferenced grid.

    M = number of rows in grid
    N = number of columns in grid
    P = number of points in border set

    :param score_matrix: M-by-N numpy array of scores.
    :param advanced_score_table_xarray: xarray table returned by
        `evaluation.read_advanced_score_file`, which will be used for metadata.
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg N).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg E).
    :param colour_map_name: Name of colour scheme (must be accepted by
        `matplotlib.pyplot.get_cmap`).
    :param is_frequency_bias: Boolean flag.  If True, the score being plotted is
        frequency bias.
    :param maybe_negative: Boolean flag.  If True, the score being plotted may
        be negative.
    :param output_file_name: Path to output file (figure will be saved here).
    :param title_string: Title (will be added above figure).  If you do not want
        a title, make this None.
    :param panel_letter: Panel letter.  For example, if the letter is "a", will
        add "(a)" at top-left of figure, assuming that it will eventually be a
        panel in a larger figure.  If you do not want a panel letter, make this
        None.
    """

    latitudes_deg_n = (
        advanced_score_table_xarray.coords[evaluation.LATITUDE_DIM].values
    )
    longitudes_deg_e = (
        advanced_score_table_xarray.coords[evaluation.LONGITUDE_DIM].values
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
    )

    if is_frequency_bias:
        this_offset = numpy.nanpercentile(
            numpy.absolute(score_matrix - 1.), MAX_COLOUR_PERCENTILE
        )
        min_colour_value = 0.
        max_colour_value = 1. + this_offset

        colour_map_object, colour_norm_object = _get_bias_colour_scheme(
            colour_map_name=colour_map_name, max_colour_value=max_colour_value
        )
    else:
        colour_map_object = pyplot.get_cmap(colour_map_name)

        if maybe_negative:
            min_colour_value = -1.
        else:
            min_colour_value = numpy.nanpercentile(
                score_matrix, 100. - MAX_COLOUR_PERCENTILE
            )

        max_colour_value = numpy.nanpercentile(
            score_matrix, MAX_COLOUR_PERCENTILE
        )
        colour_norm_object = pyplot.Normalize(
            vmin=min_colour_value, vmax=max_colour_value
        )

    radar_plotting.plot_latlng_grid(
        field_matrix=score_matrix, field_name=DUMMY_FIELD_NAME,
        axes_object=axes_object,
        min_grid_point_latitude_deg=numpy.min(latitudes_deg_n),
        min_grid_point_longitude_deg=numpy.min(longitudes_deg_e),
        latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object
    )

    if is_frequency_bias:
        colour_bar_object = gg_plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object, data_matrix=score_matrix,
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical', fraction_of_axis_length=0.8,
            extend_min=False, extend_max=True
        )

        tick_values = colour_bar_object.get_ticks()
        tick_strings = ['{0:.1g}'.format(v) for v in tick_values]
        # tick_strings = ['{0:.1f}'.format(v) for v in tick_values]
    else:
        colour_bar_object = gg_plotting_utils.plot_linear_colour_bar(
            axes_object_or_matrix=axes_object, data_matrix=score_matrix,
            colour_map_object=colour_map_object,
            min_value=min_colour_value, max_value=max_colour_value,
            orientation_string='vertical', fraction_of_axis_length=0.8,
            extend_min=maybe_negative or min_colour_value > TOLERANCE,
            extend_max=max_colour_value < 1. - TOLERANCE
        )

        tick_values = colour_bar_object.get_ticks()
        tick_strings = ['{0:.1g}'.format(v) for v in tick_values]
        # tick_strings = ['{0:.2f}'.format(v) for v in tick_values]

    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    if title_string is not None:
        axes_object.set_title(title_string)

    if panel_letter is not None:
        gg_plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(panel_letter)
        )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(advanced_score_file_name, sequential_colour_map_name,
         diverging_colour_map_name, bias_colour_map_name, probability_threshold,
         output_dir_name):
    """Plots evaluation scores on grid.

    This is effectively the main method.

    :param advanced_score_file_name: See documentation at top of file.
    :param sequential_colour_map_name: Same.
    :param diverging_colour_map_name: Same.
    :param bias_colour_map_name: Same.
    :param probability_threshold: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if desired probability threshold cannot be found.
    """

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    if probability_threshold <= 0:
        probability_threshold = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(advanced_score_file_name))
    advanced_score_table_xarray = evaluation.read_advanced_score_file(
        advanced_score_file_name
    )

    _plot_one_score(
        score_matrix=
        advanced_score_table_xarray[evaluation.BRIER_SCORE_KEY].values,
        advanced_score_table_xarray=advanced_score_table_xarray,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        colour_map_name=sequential_colour_map_name,
        is_frequency_bias=False, maybe_negative=False,
        output_file_name='{0:s}/brier_score.jpg'.format(output_dir_name),
        title_string='Brier score'
    )

    _plot_one_score(
        score_matrix=
        advanced_score_table_xarray[evaluation.BRIER_SKILL_SCORE_KEY].values,
        advanced_score_table_xarray=advanced_score_table_xarray,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        colour_map_name=diverging_colour_map_name,
        is_frequency_bias=False, maybe_negative=True,
        output_file_name='{0:s}/brier_skill_score.jpg'.format(output_dir_name),
        title_string='Brier skill score'
    )

    _plot_one_score(
        score_matrix=advanced_score_table_xarray[evaluation.FSS_KEY].values,
        advanced_score_table_xarray=advanced_score_table_xarray,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        colour_map_name=diverging_colour_map_name,
        is_frequency_bias=False, maybe_negative=True,
        output_file_name=
        '{0:s}/fractions_skill_score.jpg'.format(output_dir_name),
        title_string='Fractions skill score'
    )

    _plot_one_score(
        score_matrix=
        advanced_score_table_xarray[evaluation.TRAINING_EVENT_FREQ_KEY].values,
        advanced_score_table_xarray=advanced_score_table_xarray,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        colour_map_name=sequential_colour_map_name,
        is_frequency_bias=False, maybe_negative=False,
        output_file_name=
        '{0:s}/climo_event_frequency.jpg'.format(output_dir_name),
        title_string='Climatological event frequency (in training data)'
    )

    if probability_threshold is None:
        return

    all_prob_thresholds = advanced_score_table_xarray.coords[
        evaluation.PROBABILITY_THRESHOLD_DIM
    ].values

    prob_threshold_index = numpy.argmin(numpy.absolute(
        all_prob_thresholds - probability_threshold
    ))
    min_difference = (
        all_prob_thresholds[prob_threshold_index] - probability_threshold
    )

    if min_difference > TOLERANCE:
        error_string = (
            'Cannot find desired probability threshold ({0:.6f}).  Nearest is '
            '{1:.6f}.'
        ).format(
            probability_threshold, all_prob_thresholds[prob_threshold_index]
        )

        raise ValueError(error_string)

    a = advanced_score_table_xarray

    _plot_one_score(
        score_matrix=a[evaluation.POD_KEY].values[..., prob_threshold_index],
        advanced_score_table_xarray=a,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        colour_map_name=sequential_colour_map_name,
        is_frequency_bias=False, maybe_negative=False,
        output_file_name='{0:s}/pod.jpg'.format(output_dir_name),
        title_string='Probability of detection'
    )

    _plot_one_score(
        score_matrix=
        a[evaluation.SUCCESS_RATIO_KEY].values[..., prob_threshold_index],
        advanced_score_table_xarray=a,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        colour_map_name=sequential_colour_map_name,
        is_frequency_bias=False, maybe_negative=False,
        output_file_name='{0:s}/success_ratio.jpg'.format(output_dir_name),
        title_string='Success ratio'
    )

    _plot_one_score(
        score_matrix=
        a[evaluation.FREQUENCY_BIAS_KEY].values[..., prob_threshold_index],
        advanced_score_table_xarray=a,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        colour_map_name=bias_colour_map_name,
        is_frequency_bias=True, maybe_negative=False,
        output_file_name='{0:s}/frequency_bias.jpg'.format(output_dir_name),
        title_string='Frequency bias'
    )

    _plot_one_score(
        score_matrix=a[evaluation.CSI_KEY].values[..., prob_threshold_index],
        advanced_score_table_xarray=a,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        colour_map_name=sequential_colour_map_name,
        is_frequency_bias=False, maybe_negative=False,
        output_file_name='{0:s}/csi.jpg'.format(output_dir_name),
        title_string='Critical success index'
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        advanced_score_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        sequential_colour_map_name=getattr(
            INPUT_ARG_OBJECT, SEQ_COLOUR_MAP_HELP_STRING
        ),
        diverging_colour_map_name=getattr(
            INPUT_ARG_OBJECT, DIV_COLOUR_MAP_ARG_NAME
        ),
        bias_colour_map_name=getattr(
            INPUT_ARG_OBJECT, BIAS_COLOUR_MAP_ARG_NAME
        ),
        probability_threshold=getattr(
            INPUT_ARG_OBJECT, PROB_THRESHOLD_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
