"""Makes figure with FSS and CSI vs. lead time for U-nets and persistence."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors
import matplotlib.patches
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4convection.utils import evaluation
from ml4convection.plotting import evaluation_plotting as eval_plotting

TOLERANCE = 1e-6

MARKER_TYPE = 'o'
MARKER_SIZE = 16
LINE_WIDTH = 4

GRID_LINE_WIDTH = 1.
GRID_LINE_COLOUR = numpy.full(3, 0.)

FSS_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
CSI_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
BSS_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
POLYGON_OPACITY = 0.5

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

LEAD_TIMES_ARG_NAME = 'lead_times_minutes'
U_NET_FILES_ARG_NAME = 'input_u_net_eval_file_names'
PERSISTENCE_FILES_ARG_NAME = 'input_persistence_eval_file_names'
U_NET_THRESHOLDS_ARG_NAME = 'u_net_prob_thresholds'
PERSISTENCE_THRESHOLDS_ARG_NAME = 'persistence_prob_thresholds'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

LEAD_TIMES_HELP_STRING = 'List of lead times.'
U_NET_FILES_HELP_STRING = (
    'List of paths to evaluation files for U-nets.  num_evaluation_files must ='
    ' num_lead_times.  Each of these files will be read by '
    '`evaluation.read_advanced_score_file`.'
)
PERSISTENCE_FILES_HELP_STRING = (
    'List of paths to evaluation files for persistence models.  '
    'num_evaluation_files must = num_lead_times_above_zero.  Each of these '
    'files will be read by `evaluation.read_advanced_score_file`.'
)
U_NET_THRESHOLDS_HELP_STRING = (
    'List of probability thresholds for U-nets, at which CSI will be computed.'
    '  num_thresholds must = num_lead_times.'
)
PERSISTENCE_THRESHOLDS_HELP_STRING = (
    'List of probability thresholds for persistence models, at which CSI will '
    'be computed.  num_thresholds must = num_lead_times_above_zero.'
)
CONFIDENCE_LEVEL_HELP_STRING = (
    'Level for confidence intervals, ranging from 0...1.'
)
OUTPUT_FILE_HELP_STRING = 'Path to output file.  Figure will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + LEAD_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=LEAD_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + U_NET_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=U_NET_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PERSISTENCE_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=PERSISTENCE_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + U_NET_THRESHOLDS_ARG_NAME, type=float, nargs='+', required=True,
    help=U_NET_THRESHOLDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PERSISTENCE_THRESHOLDS_ARG_NAME, type=float, nargs='+',
    required=True, help=PERSISTENCE_THRESHOLDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _plot_figure(
        lead_times_minutes, u_net_csi_matrix, u_net_fss_matrix,
        u_net_bss_matrix, persistence_csi_matrix, persistence_fss_matrix,
        persistence_bss_matrix, confidence_level, output_file_name):
    """Plots figure showing CSI and FSS vs. lead time.

    T = number of lead times
    B = number of bootstrap replicates

    :param lead_times_minutes: length-T numpy array of lead times.
    :param u_net_csi_matrix: T-by-B numpy array of CSI values.
    :param u_net_fss_matrix: T-by-B numpy array of FSS values.
    :param u_net_bss_matrix: T-by-B numpy array of BSS values.
    :param persistence_csi_matrix: T-by-B numpy array of CSI values.
    :param persistence_fss_matrix: T-by-B numpy array of FSS values.
    :param persistence_bss_matrix: T-by-B numpy array of BSS values.
    :param confidence_level: See documentation at top of file.
    :param output_file_name: Same.
    """

    # Housekeeping.
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    # Plot U-net FSS.
    this_handle = axes_object.plot(
        lead_times_minutes, numpy.mean(u_net_fss_matrix, axis=1),
        color=FSS_COLOUR, linewidth=LINE_WIDTH, linestyle='solid',
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=FSS_COLOUR, markeredgecolor=FSS_COLOUR
    )[0]

    legend_handles = [this_handle]
    legend_strings = ['U-net FSS']

    num_bootstrap_reps = u_net_fss_matrix.shape[1]

    if num_bootstrap_reps > 1:
        x_value_matrix = numpy.expand_dims(lead_times_minutes, axis=-1)
        x_value_matrix = numpy.repeat(
            x_value_matrix, axis=-1, repeats=num_bootstrap_reps
        )
        polygon_coord_matrix = eval_plotting.confidence_interval_to_polygon(
            x_value_matrix=numpy.transpose(x_value_matrix),
            y_value_matrix=numpy.transpose(u_net_fss_matrix),
            confidence_level=confidence_level, same_order=False
        )

        polygon_colour = matplotlib.colors.to_rgba(FSS_COLOUR, POLYGON_OPACITY)
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        axes_object.add_patch(patch_object)

    # Plot U-net CSI.
    this_handle = axes_object.plot(
        lead_times_minutes, numpy.mean(u_net_csi_matrix, axis=1),
        color=CSI_COLOUR, linewidth=LINE_WIDTH, linestyle='solid',
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=CSI_COLOUR, markeredgecolor=CSI_COLOUR
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('U-net CSI')

    if num_bootstrap_reps > 1:
        x_value_matrix = numpy.expand_dims(lead_times_minutes, axis=-1)
        x_value_matrix = numpy.repeat(
            x_value_matrix, axis=-1, repeats=num_bootstrap_reps
        )
        polygon_coord_matrix = eval_plotting.confidence_interval_to_polygon(
            x_value_matrix=numpy.transpose(x_value_matrix),
            y_value_matrix=numpy.transpose(u_net_csi_matrix),
            confidence_level=confidence_level, same_order=False
        )

        polygon_colour = matplotlib.colors.to_rgba(CSI_COLOUR, POLYGON_OPACITY)
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        axes_object.add_patch(patch_object)

    # Plot U-net BSS.
    this_handle = axes_object.plot(
        lead_times_minutes, numpy.mean(u_net_bss_matrix, axis=1),
        color=BSS_COLOUR, linewidth=LINE_WIDTH, linestyle='solid',
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=BSS_COLOUR, markeredgecolor=BSS_COLOUR
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('U-net BSS')

    if num_bootstrap_reps > 1:
        x_value_matrix = numpy.expand_dims(lead_times_minutes, axis=-1)
        x_value_matrix = numpy.repeat(
            x_value_matrix, axis=-1, repeats=num_bootstrap_reps
        )
        polygon_coord_matrix = eval_plotting.confidence_interval_to_polygon(
            x_value_matrix=numpy.transpose(x_value_matrix),
            y_value_matrix=numpy.transpose(u_net_bss_matrix),
            confidence_level=confidence_level, same_order=False
        )

        polygon_colour = matplotlib.colors.to_rgba(BSS_COLOUR, POLYGON_OPACITY)
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        axes_object.add_patch(patch_object)

    # Plot persistence-model FSS.
    this_handle = axes_object.plot(
        lead_times_minutes, numpy.mean(persistence_fss_matrix, axis=1),
        color=FSS_COLOUR, linewidth=LINE_WIDTH, linestyle='dashed',
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=FSS_COLOUR, markeredgecolor=FSS_COLOUR
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('Persistence FSS')

    if num_bootstrap_reps > 1:
        x_value_matrix = numpy.expand_dims(lead_times_minutes, axis=-1)
        x_value_matrix = numpy.repeat(
            x_value_matrix, axis=-1, repeats=num_bootstrap_reps
        )
        polygon_coord_matrix = eval_plotting.confidence_interval_to_polygon(
            x_value_matrix=numpy.transpose(x_value_matrix),
            y_value_matrix=numpy.transpose(persistence_fss_matrix),
            confidence_level=confidence_level, same_order=False
        )

        polygon_colour = matplotlib.colors.to_rgba(FSS_COLOUR, POLYGON_OPACITY)
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        axes_object.add_patch(patch_object)

    # Plot persistence-model CSI.
    this_handle = axes_object.plot(
        lead_times_minutes, numpy.mean(persistence_csi_matrix, axis=1),
        color=CSI_COLOUR, linewidth=LINE_WIDTH, linestyle='dashed',
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=CSI_COLOUR, markeredgecolor=CSI_COLOUR
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('Persistence CSI')

    if num_bootstrap_reps > 1:
        x_value_matrix = numpy.expand_dims(lead_times_minutes, axis=-1)
        x_value_matrix = numpy.repeat(
            x_value_matrix, axis=-1, repeats=num_bootstrap_reps
        )
        polygon_coord_matrix = eval_plotting.confidence_interval_to_polygon(
            x_value_matrix=numpy.transpose(x_value_matrix),
            y_value_matrix=numpy.transpose(persistence_csi_matrix),
            confidence_level=confidence_level, same_order=False
        )

        polygon_colour = matplotlib.colors.to_rgba(CSI_COLOUR, POLYGON_OPACITY)
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        axes_object.add_patch(patch_object)

    # Plot persistence-model BSS.
    this_handle = axes_object.plot(
        lead_times_minutes, numpy.mean(persistence_bss_matrix, axis=1),
        color=BSS_COLOUR, linewidth=LINE_WIDTH, linestyle='dashed',
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=BSS_COLOUR, markeredgecolor=BSS_COLOUR
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('Persistence BSS')

    if num_bootstrap_reps > 1:
        x_value_matrix = numpy.expand_dims(lead_times_minutes, axis=-1)
        x_value_matrix = numpy.repeat(
            x_value_matrix, axis=-1, repeats=num_bootstrap_reps
        )
        polygon_coord_matrix = eval_plotting.confidence_interval_to_polygon(
            x_value_matrix=numpy.transpose(x_value_matrix),
            y_value_matrix=numpy.transpose(persistence_bss_matrix),
            confidence_level=confidence_level, same_order=False
        )

        polygon_colour = matplotlib.colors.to_rgba(BSS_COLOUR, POLYGON_OPACITY)
        patch_object = matplotlib.patches.Polygon(
            polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
        )
        axes_object.add_patch(patch_object)

    axes_object.set_xticks(lead_times_minutes)
    axes_object.set_xlabel('Lead time (minutes)')

    axes_object.grid(
        b=True, which='major', axis='y', linestyle='--',
        linewidth=GRID_LINE_WIDTH, color=GRID_LINE_COLOUR
    )

    axes_object.legend(
        legend_handles, legend_strings, loc='lower center',
        bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=2
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(lead_times_minutes, u_net_eval_file_names, persistence_eval_file_names,
         u_net_prob_thresholds, persistence_prob_thresholds,
         confidence_level, output_file_name):
    """Makes figure with FSS and CSI vs. lead time for U-nets and persistence.

    This is effectively the main method.

    :param lead_times_minutes: See documentation at top of file.
    :param u_net_eval_file_names: Same.
    :param persistence_eval_file_names: Same.
    :param u_net_prob_thresholds: Same.
    :param persistence_prob_thresholds: Same.
    :param confidence_level: Same.
    :param output_file_name: Same.
    """

    # Check input args.
    error_checking.assert_is_geq_numpy_array(lead_times_minutes, 0)
    num_lead_times = len(lead_times_minutes)
    num_nonzero_lead_times = numpy.sum(lead_times_minutes > 0)

    error_checking.assert_is_numpy_array(
        numpy.unique(lead_times_minutes),
        exact_dimensions=numpy.array([num_lead_times], dtype=int)
    )
    error_checking.assert_is_numpy_array(
        numpy.array(u_net_eval_file_names),
        exact_dimensions=numpy.array([num_lead_times], dtype=int)
    )
    error_checking.assert_is_numpy_array(
        numpy.array(persistence_eval_file_names),
        exact_dimensions=numpy.array([num_nonzero_lead_times], dtype=int)
    )
    error_checking.assert_is_numpy_array(
        u_net_prob_thresholds,
        exact_dimensions=numpy.array([num_lead_times], dtype=int)
    )
    error_checking.assert_is_numpy_array(
        persistence_prob_thresholds,
        exact_dimensions=numpy.array([num_nonzero_lead_times], dtype=int)
    )

    error_checking.assert_is_geq_numpy_array(u_net_prob_thresholds, 0.)
    error_checking.assert_is_leq_numpy_array(u_net_prob_thresholds, 1.)
    error_checking.assert_is_geq_numpy_array(persistence_prob_thresholds, 0.)
    error_checking.assert_is_leq_numpy_array(persistence_prob_thresholds, 1.)
    error_checking.assert_is_geq(confidence_level, 0.9)
    error_checking.assert_is_less_than(confidence_level, 1.)
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    if numpy.any(lead_times_minutes == 0):
        zero_index = numpy.where(lead_times_minutes == 0)[0][0]
        persistence_eval_file_names.insert(zero_index, None)

        persistence_prob_thresholds = persistence_prob_thresholds.tolist()
        persistence_prob_thresholds.insert(zero_index, numpy.nan)
        persistence_prob_thresholds = numpy.array(persistence_prob_thresholds)

    sort_indices = numpy.argsort(lead_times_minutes)
    lead_times_minutes = lead_times_minutes[sort_indices]
    u_net_eval_file_names = [u_net_eval_file_names[k] for k in sort_indices]
    persistence_eval_file_names = [
        persistence_eval_file_names[k] for k in sort_indices
    ]
    u_net_prob_thresholds = u_net_prob_thresholds[sort_indices]
    persistence_prob_thresholds = persistence_prob_thresholds[sort_indices]

    # Do actual stuff.
    u_net_csi_matrix = None
    u_net_fss_matrix = None
    u_net_bss_matrix = None
    persistence_csi_matrix = None
    persistence_fss_matrix = None
    persistence_bss_matrix = None

    for i in range(num_lead_times):
        print('Reading data from: "{0:s}"...'.format(u_net_eval_file_names[i]))
        et = evaluation.read_advanced_score_file(u_net_eval_file_names[i])

        these_prob_diffs = numpy.absolute(
            et.coords[evaluation.PROBABILITY_THRESHOLD_DIM].values -
            u_net_prob_thresholds[i]
        )
        assert numpy.min(these_prob_diffs) <= TOLERANCE
        this_threshold_index = numpy.argmin(these_prob_diffs)

        if i == 0:
            num_bootstrap_reps = len(
                et.coords[evaluation.BOOTSTRAP_REPLICATE_DIM].values
            )
            these_dim = (num_lead_times, num_bootstrap_reps)

            u_net_csi_matrix = numpy.full(these_dim, numpy.nan)
            u_net_fss_matrix = numpy.full(these_dim, numpy.nan)
            u_net_bss_matrix = numpy.full(these_dim, numpy.nan)
            persistence_csi_matrix = numpy.full(these_dim, numpy.nan)
            persistence_fss_matrix = numpy.full(these_dim, numpy.nan)
            persistence_bss_matrix = numpy.full(these_dim, numpy.nan)

        u_net_csi_matrix[i, :] = (
            et[evaluation.CSI_KEY].values[:, this_threshold_index]
        )
        u_net_fss_matrix[i, :] = et[evaluation.FSS_KEY].values[:, 0]
        u_net_bss_matrix[i, :] = (
            et[evaluation.BRIER_SKILL_SCORE_KEY].values[:, 0]
        )

        if lead_times_minutes[i] == 0:
            continue

        print('Reading data from: "{0:s}"...'.format(
            persistence_eval_file_names[i]
        ))
        et = evaluation.read_advanced_score_file(persistence_eval_file_names[i])

        these_prob_diffs = numpy.absolute(
            et.coords[evaluation.PROBABILITY_THRESHOLD_DIM].values -
            persistence_prob_thresholds[i]
        )
        assert numpy.min(these_prob_diffs) <= TOLERANCE
        this_threshold_index = numpy.argmin(these_prob_diffs)

        persistence_csi_matrix[i, :] = (
            et[evaluation.CSI_KEY].values[:, this_threshold_index]
        )
        persistence_fss_matrix[i, :] = et[evaluation.FSS_KEY].values[:, 0]
        persistence_bss_matrix[i, :] = (
            et[evaluation.BRIER_SKILL_SCORE_KEY].values[:, 0]
        )

    _plot_figure(
        lead_times_minutes=lead_times_minutes,
        u_net_csi_matrix=u_net_csi_matrix,
        u_net_fss_matrix=u_net_fss_matrix,
        u_net_bss_matrix=u_net_bss_matrix,
        persistence_csi_matrix=persistence_csi_matrix,
        persistence_fss_matrix=persistence_fss_matrix,
        persistence_bss_matrix=persistence_bss_matrix,
        confidence_level=confidence_level,
        output_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        lead_times_minutes=numpy.array(
            getattr(INPUT_ARG_OBJECT, LEAD_TIMES_ARG_NAME), dtype=int
        ),
        u_net_eval_file_names=getattr(INPUT_ARG_OBJECT, U_NET_FILES_ARG_NAME),
        persistence_eval_file_names=getattr(
            INPUT_ARG_OBJECT, PERSISTENCE_FILES_ARG_NAME
        ),
        u_net_prob_thresholds=numpy.array(
            getattr(INPUT_ARG_OBJECT, U_NET_THRESHOLDS_ARG_NAME), dtype=float
        ),
        persistence_prob_thresholds=numpy.array(
            getattr(INPUT_ARG_OBJECT, PERSISTENCE_THRESHOLDS_ARG_NAME),
            dtype=float
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
