"""Plots scores on hyperparameter grid for Experiment 2."""

import sys
import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import pixelwise_evaluation as pixelwise_eval
import spatial_evaluation as spatial_eval
import plotting_utils
import file_system_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

POSITIVE_CLASS_WEIGHTS = numpy.array(
    [1, 10, 25, 50, 75, 100, 150, 200], dtype=int
)
CONV_LAYER_DROPOUT_RATES = numpy.array([0, 0.175, 0.35, 0.525, 0.7])
L2_WEIGHTS = numpy.logspace(-4, -2, num=5)

AUC_KEY = pixelwise_eval.AUC_KEY
AUPD_KEY = pixelwise_eval.AUPD_KEY
MAX_CSI_KEY = 'max_csi'
FSS_ARRAY_KEY = 'fractions_skill_scores'
HALF_WINDOW_SIZES_FOR_EVAL_KEY = 'half_window_sizes_for_eval_px'

FONT_SIZE = 20
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

EXPERIMENT_DIR_ARG_NAME = 'experiment_dir_name'
EXPERIMENT_DIR_HELP_STRING = 'Name of top-level directory with models.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT_DIR_HELP_STRING
)


def _plot_scores_2d(
        score_matrix, min_colour_value, max_colour_value, x_tick_labels,
        y_tick_labels, colour_map_object=pyplot.get_cmap('plasma')
):
    """Plots scores on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of scores.
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param x_tick_labels: length-N list of tick labels.
    :param y_tick_labels: length-M list of tick labels.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.imshow(
        score_matrix, cmap=colour_map_object, origin='lower',
        vmin=min_colour_value, vmax=max_colour_value
    )

    x_tick_values = numpy.linspace(
        0, score_matrix.shape[1] - 1, num=score_matrix.shape[1], dtype=float
    )
    y_tick_values = numpy.linspace(
        0, score_matrix.shape[0] - 1, num=score_matrix.shape[0], dtype=float
    )

    pyplot.xticks(x_tick_values, x_tick_labels)
    pyplot.yticks(y_tick_values, y_tick_labels)

    colour_norm_object = matplotlib.colors.Normalize(
        vmin=min_colour_value, vmax=max_colour_value, clip=False
    )

    colour_bar_object = plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=score_matrix[numpy.invert(numpy.isnan(score_matrix))],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='horizontal', extend_min=False, extend_max=False,
        fraction_of_axis_length=0.8, font_size=FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _read_scores_one_model(model_dir_name):
    """Reads scores for one model.

    E = number of scales at which fractions skill score was computed

    :param model_dir_name: Name of directory with trained model and evaluation
        data.
    :return: score_dict: Dictionary with the following keys.
    score_dict['area_under_roc_curve']: Area under ROC curve.
    score_dict['area_under_perf_diagram']: Area under performance diagram.
    score_dict['max_csi']: Maximum critical success index.
    score_dict['fractions_skill_scores']: length-E numpy array of fractions
        skill scores.
    score_dict['half_window_sizes_for_eval_px']: length-E numpy array of
        corresponding scales.
    """

    pixelwise_eval_file_name = (
        '{0:s}/validation/evaluation/advanced_scores.nc'.format(model_dir_name)
    )
    spatial_eval_file_name = (
        '{0:s}/validation/evaluation/spatial_evaluation.p'
    ).format(model_dir_name)

    if not (
            os.path.isfile(pixelwise_eval_file_name) and
            os.path.isfile(spatial_eval_file_name)
    ):
        return None

    print('Reading data from: "{0:s}"...'.format(pixelwise_eval_file_name))
    advanced_score_table_xarray = (
        pixelwise_eval.read_file(pixelwise_eval_file_name)
    )

    score_dict = {
        AUC_KEY: advanced_score_table_xarray.attrs[pixelwise_eval.AUC_KEY],
        AUPD_KEY: advanced_score_table_xarray.attrs[pixelwise_eval.AUPD_KEY],
        MAX_CSI_KEY: numpy.max(
            advanced_score_table_xarray[pixelwise_eval.CSI_KEY].values
        )
    }

    print('Reading data from: "{0:s}"...'.format(spatial_eval_file_name))
    fractions_skill_scores, half_window_sizes_for_eval_px = (
        spatial_eval.read_file(spatial_eval_file_name)
    )

    score_dict[FSS_ARRAY_KEY] = fractions_skill_scores
    score_dict[HALF_WINDOW_SIZES_FOR_EVAL_KEY] = half_window_sizes_for_eval_px
    return score_dict


def _print_ranking_one_score(score_matrix, score_name):
    """Prints ranking for one score.

    L = number of scales for loss function
    C = number of conv-layer dropout rates
    W = number of L_2 weights

    :param score_matrix: L-by-C-by-W numpy array of scores.
    :param score_name: Name of score.
    """

    scores_1d = numpy.ravel(score_matrix)
    scores_1d[numpy.isnan(scores_1d)] = -numpy.inf
    sort_indices_1d = numpy.argsort(-scores_1d)
    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        sort_indices_1d, score_matrix.shape
    )

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        print((
            '{0:d}th-highest {1:s} = {2:.4g} ... '
            'positive-class weight = {3:d} ... '
            'conv-layer dropout rate = {4:.3f} ... L_2 weight = 10^{5:.1f}'
        ).format(
            m + 1, score_name, score_matrix[i, j, k],
            POSITIVE_CLASS_WEIGHTS[i], CONV_LAYER_DROPOUT_RATES[j],
            numpy.log10(L2_WEIGHTS[k])
        ))


def _run(experiment_dir_name):
    """Plots scores on hyperparameter grid for Experiment 1.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    """

    num_pos_class_weights = len(POSITIVE_CLASS_WEIGHTS)
    num_dropout_rates = len(CONV_LAYER_DROPOUT_RATES)
    num_l2_weights = len(L2_WEIGHTS)
    dimensions = (
        num_pos_class_weights, num_dropout_rates, num_l2_weights
    )

    max_csi_matrix = numpy.full(dimensions, numpy.nan)
    auc_matrix = numpy.full(dimensions, numpy.nan)
    aupd_matrix = numpy.full(dimensions, numpy.nan)
    fss_matrix = None
    half_window_sizes_for_eval_px = None

    y_tick_labels = [
        '{0:.3f}'.format(d).replace('-1', '0')
        for d in CONV_LAYER_DROPOUT_RATES
    ]
    x_tick_labels = [
        r'10${0:.1f}$'.format(numpy.log10(w)) for w in L2_WEIGHTS
    ]
    y_axis_label = 'Conv-layer dropout rate'
    x_axis_label = r'L$_{2}$ weight'

    for i in range(num_pos_class_weights):
        for j in range(num_dropout_rates):
            for k in range(num_l2_weights):
                this_model_dir_name = (
                    '{0:s}/positive-class-weight={1:03d}_'
                    'conv-dropout={2:.3f}_l2-weight={3:.6f}'
                ).format(
                    experiment_dir_name, POSITIVE_CLASS_WEIGHTS[i],
                    CONV_LAYER_DROPOUT_RATES[j], L2_WEIGHTS[k]
                )

                this_score_dict = _read_scores_one_model(this_model_dir_name)

                if fss_matrix is None:
                    half_window_sizes_for_eval_px = (
                        this_score_dict[HALF_WINDOW_SIZES_FOR_EVAL_KEY] + 0
                    )
                    these_dim = (
                        num_pos_class_weights, num_dropout_rates,
                        num_l2_weights, len(half_window_sizes_for_eval_px)
                    )
                    fss_matrix = numpy.full(these_dim, numpy.nan)

                auc_matrix[i, j, k] = this_score_dict[AUC_KEY]
                aupd_matrix[i, j, k] = this_score_dict[AUPD_KEY]
                max_csi_matrix[i, j, k] = this_score_dict[MAX_CSI_KEY]
                fss_matrix[i, j, k, :] = this_score_dict[FSS_ARRAY_KEY]

    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=auc_matrix, score_name='AUC')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=aupd_matrix, score_name='AUPD')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=max_csi_matrix, score_name='Max CSI')
    print(SEPARATOR_STRING)

    num_window_sizes_for_eval = len(half_window_sizes_for_eval_px)

    for m in range(num_window_sizes_for_eval):
        this_window_size_px = int(numpy.round(
            2 * half_window_sizes_for_eval_px[m] + 1
        ))
        _print_ranking_one_score(
            score_matrix=fss_matrix[..., m],
            score_name='{0:d}-by-{0:d} FSS'.format(this_window_size_px)
        )
        print(SEPARATOR_STRING)

    for i in range(num_pos_class_weights):
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=auc_matrix[i, ...],
            min_colour_value=numpy.nanpercentile(auc_matrix, 1),
            max_colour_value=numpy.nanpercentile(auc_matrix, 99),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        title_string = 'AUC with positive-class weight = {0:d}'.format(
            POSITIVE_CLASS_WEIGHTS[i]
        )
        axes_object.set_title(title_string)

        figure_file_name = (
            '{0:s}/positive-class-weight={1:03d}_auc_grid.jpg'
        ).format(
            experiment_dir_name, POSITIVE_CLASS_WEIGHTS[i]
        )

        file_system_utils.mkdir_recursive_if_necessary(
            file_name=figure_file_name
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        figure_object, axes_object = _plot_scores_2d(
            score_matrix=aupd_matrix[i, ...],
            min_colour_value=numpy.nanpercentile(aupd_matrix, 1),
            max_colour_value=numpy.nanpercentile(aupd_matrix, 99),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        title_string = 'AUPD with positive-class weight = {0:d}'.format(
            POSITIVE_CLASS_WEIGHTS[i]
        )
        axes_object.set_title(title_string)

        figure_file_name = (
            '{0:s}/positive-class-weight={1:03d}_aupd_grid.jpg'
        ).format(
            experiment_dir_name, POSITIVE_CLASS_WEIGHTS[i]
        )

        file_system_utils.mkdir_recursive_if_necessary(
            file_name=figure_file_name
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        figure_object, axes_object = _plot_scores_2d(
            score_matrix=max_csi_matrix[i, ...],
            min_colour_value=numpy.nanpercentile(max_csi_matrix, 1),
            max_colour_value=numpy.nanpercentile(max_csi_matrix, 99),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        title_string = 'Max CSI with positive-class weight = {0:d}'.format(
            POSITIVE_CLASS_WEIGHTS[i]
        )
        axes_object.set_title(title_string)

        figure_file_name = (
            '{0:s}/positive-class-weight={1:03d}_csi_grid.jpg'
        ).format(
            experiment_dir_name, POSITIVE_CLASS_WEIGHTS[i]
        )

        file_system_utils.mkdir_recursive_if_necessary(
            file_name=figure_file_name
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        for m in range(num_window_sizes_for_eval):
            this_window_size_for_eval_px = int(numpy.round(
                2 * half_window_sizes_for_eval_px[m] + 1
            ))

            figure_object, axes_object = _plot_scores_2d(
                score_matrix=fss_matrix[i, ..., m],
                min_colour_value=numpy.nanpercentile(fss_matrix[..., m], 1),
                max_colour_value=numpy.nanpercentile(fss_matrix[..., m], 99),
                x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
            )

            axes_object.set_xlabel(x_axis_label)
            axes_object.set_ylabel(y_axis_label)
            title_string = (
                '{0:d}-by-{0:d} FSS with positive-class weight = {1:d}'
            ).format(
                this_window_size_for_eval_px, POSITIVE_CLASS_WEIGHTS[i]
            )
            axes_object.set_title(title_string)

            figure_file_name = (
                '{0:s}/positive-class-weight={1:03d}_fss{2:d}by{2:d}_grid.jpg'
            ).format(
                experiment_dir_name, POSITIVE_CLASS_WEIGHTS[i],
                this_window_size_for_eval_px
            )

            file_system_utils.mkdir_recursive_if_necessary(
                file_name=figure_file_name
            )

            print('Saving figure to: "{0:s}"...'.format(figure_file_name))
            figure_object.savefig(
                figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME)
    )
