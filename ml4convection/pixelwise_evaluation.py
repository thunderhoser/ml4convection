"""Pixelwise (grid-point-by-grid-point) evaluation."""

import os
import sys
import pickle
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import histograms
import gg_model_evaluation as gg_model_eval
import gg_general_utils
import file_system_utils
import error_checking
import prediction_io

DEFAULT_NUM_PROB_THRESHOLDS = 201
DEFAULT_NUM_RELIA_BINS = 20

PROBABILITY_THRESHOLD_DIM = 'probability_threshold'
RELIABILITY_BIN_DIM = 'reliability_bin'

NUM_TRUE_POSITIVES_KEY = 'num_true_positives'
NUM_FALSE_POSITIVES_KEY = 'num_false_positives'
NUM_FALSE_NEGATIVES_KEY = 'num_false_negatives'
NUM_TRUE_NEGATIVES_KEY = 'num_true_negatives'
NUM_EXAMPLES_KEY = 'num_examples'
EVENT_FREQUENCY_KEY = 'event_frequency'
MEAN_FORECAST_PROB_KEY = 'mean_forecast_prob'
CLIMO_EVENT_FREQ_KEY = 'climo_event_frequency'

POD_KEY = 'probability_of_detection'
POFD_KEY = 'probability_of_false_detection'
SUCCESS_RATIO_KEY = 'success_ratio'
FREQUENCY_BIAS_KEY = 'frequency_bias'
CSI_KEY = 'critical_success_index'
ACCURACY_KEY = 'accuracy'
HEIDKE_SCORE_KEY = 'heidke_score'
AUC_KEY = 'area_under_roc_curve'
AUPD_KEY = 'area_under_perf_diagram'
BRIER_SCORE_KEY = 'brier_score'
BRIER_SKILL_SCORE_KEY = 'brier_skill_score'


def _update_basic_scores(basic_score_table_xarray, prediction_file_name,
                         prediction_dict=None):
    """Updates basic scores.

    :param basic_score_table_xarray: xarray table in format produced by
        `get_basic_scores`.
    :param prediction_file_name: Path to input file (will be read by
        `prediction_io.read_file`).  Predictions in this file will be used to
        update scores in the table.
    :param prediction_dict: Leave this alone.  Used for testing only.
    :return: basic_score_table_xarray: Same as input but with different values.
    :return: num_examples_new: Number of examples in new file.
    :return: num_positive_examples_new: Number of positive examples in new file.
    """

    if prediction_dict is None:
        print('Reading data from: "{0:s}"...'.format(prediction_file_name))
        prediction_dict = prediction_io.read_file(prediction_file_name)

    forecast_probabilities = numpy.ravel(
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY]
    )
    target_classes = numpy.ravel(
        prediction_dict[prediction_io.TARGET_MATRIX_KEY]
    )

    print('Updating reliability info...')

    num_bins_for_reliability = len(
        basic_score_table_xarray.coords[RELIABILITY_BIN_DIM].values
    )
    example_to_bin_indices = histograms.create_histogram(
        input_values=forecast_probabilities,
        num_bins=num_bins_for_reliability, min_value=0., max_value=1.
    )[0]

    for k in range(num_bins_for_reliability):
        these_example_indices = numpy.where(example_to_bin_indices == k)[0]

        if len(these_example_indices) == 0:
            continue

        these_mean_forecast_probs = numpy.array([
            basic_score_table_xarray[MEAN_FORECAST_PROB_KEY].values[k],
            numpy.mean(forecast_probabilities[these_example_indices])
        ])
        these_weights = numpy.array([
            basic_score_table_xarray[NUM_EXAMPLES_KEY].values[k],
            len(these_example_indices)
        ])
        basic_score_table_xarray[MEAN_FORECAST_PROB_KEY].values[k] = (
            numpy.average(a=these_mean_forecast_probs, weights=these_weights)
        )

        these_event_frequencies = numpy.array([
            basic_score_table_xarray[EVENT_FREQUENCY_KEY].values[k],
            numpy.mean(target_classes[these_example_indices])
        ])
        these_weights = numpy.array([
            basic_score_table_xarray[NUM_EXAMPLES_KEY].values[k],
            len(these_example_indices)
        ])
        basic_score_table_xarray[EVENT_FREQUENCY_KEY].values[k] = (
            numpy.average(a=these_event_frequencies, weights=these_weights)
        )

        basic_score_table_xarray[NUM_EXAMPLES_KEY].values[k] += (
            len(these_example_indices)
        )

    print('Updating contingency tables...')

    probability_thresholds = (
        basic_score_table_xarray.coords[PROBABILITY_THRESHOLD_DIM].values
    )
    num_prob_thresholds = len(probability_thresholds)

    sort_indices = numpy.argsort(target_classes)
    target_classes = target_classes[sort_indices]
    forecast_probabilities = forecast_probabilities[sort_indices]
    num_points = len(target_classes)

    if numpy.max(target_classes) == 0:
        negative_example_indices = numpy.linspace(
            0, num_points - 1, num=num_points, dtype=int
        )
        positive_example_indices = numpy.array([], dtype=int)
    else:
        first_positive_index = gg_general_utils.find_nearest_value(
            sorted_input_values=target_classes, test_value=1
        )[1]

        negative_example_indices = numpy.linspace(
            0, first_positive_index - 1, num=first_positive_index, dtype=int
        )
        positive_example_indices = numpy.linspace(
            first_positive_index, num_points - 1,
            num=num_points - first_positive_index, dtype=int
        )

    for k in range(num_prob_thresholds):
        if numpy.mod(k, 10) == 0:
            print((
                'Have updated contingency tables for {0:d} of {1:d} probability'
                ' thresholds...'
            ).format(
                k, num_prob_thresholds
            ))

        these_forecast_classes = (
            forecast_probabilities >= probability_thresholds[k]
        ).astype(int)

        basic_score_table_xarray[NUM_TRUE_POSITIVES_KEY].values[k] += (
            numpy.sum(these_forecast_classes[positive_example_indices] == 1)
        )
        basic_score_table_xarray[NUM_FALSE_POSITIVES_KEY].values[k] += (
            numpy.sum(these_forecast_classes[negative_example_indices] == 1)
        )
        basic_score_table_xarray[NUM_FALSE_NEGATIVES_KEY].values[k] += (
            numpy.sum(these_forecast_classes[positive_example_indices] == 0)
        )
        basic_score_table_xarray[NUM_TRUE_NEGATIVES_KEY].values[k] += (
            numpy.sum(these_forecast_classes[negative_example_indices] == 0)
        )

    print((
        'Have updated contingency tables for all {0:d} probability thresholds!'
    ).format(
        num_prob_thresholds
    ))

    return (
        basic_score_table_xarray,
        len(target_classes),
        numpy.sum(target_classes == 1)
    )


def get_basic_scores(
        prediction_file_names, event_frequency_in_training=None,
        num_prob_thresholds=DEFAULT_NUM_PROB_THRESHOLDS,
        num_bins_for_reliability=DEFAULT_NUM_RELIA_BINS, test_mode=False):
    """Computes basic scores (contingency tables and reliability info).

    T = number of probability thresholds
    B = number of bins for reliability

    :param prediction_file_names: 1-D list of paths to prediction files (will be
        read by `prediction_io.read_file`).
    :param event_frequency_in_training: Event frequency in training data.  If
        None, this method will compute event frequency in new data.
    :param num_prob_thresholds: Number of probability thresholds (T above).
    :param num_bins_for_reliability: Number of bins for reliability (B above).
    :param test_mode: Leave this alone.
    :return: basic_score_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    error_checking.assert_is_integer(num_bins_for_reliability)
    error_checking.assert_is_geq(num_bins_for_reliability, 10)
    error_checking.assert_is_boolean(test_mode)

    if not test_mode:
        error_checking.assert_is_string_list(prediction_file_names)

    num_examples_total = 0
    num_positive_examples_total = 0

    if event_frequency_in_training is not None:
        error_checking.assert_is_geq(event_frequency_in_training, 0.)
        error_checking.assert_is_leq(event_frequency_in_training, 1.)

    probability_thresholds = gg_model_eval.get_binarization_thresholds(
        threshold_arg=num_prob_thresholds
    )
    num_prob_thresholds = len(probability_thresholds)

    bin_indices = numpy.linspace(
        0, num_bins_for_reliability - 1, num=num_bins_for_reliability, dtype=int
    )
    metadata_dict = {
        PROBABILITY_THRESHOLD_DIM: probability_thresholds,
        RELIABILITY_BIN_DIM: bin_indices
    }

    these_dim = (PROBABILITY_THRESHOLD_DIM,)
    this_array = numpy.full(num_prob_thresholds, 0, dtype=int)
    main_data_dict = {
        NUM_TRUE_POSITIVES_KEY: (these_dim, this_array + 0),
        NUM_FALSE_POSITIVES_KEY: (these_dim, this_array + 0),
        NUM_FALSE_NEGATIVES_KEY: (these_dim, this_array + 0),
        NUM_TRUE_NEGATIVES_KEY: (these_dim, this_array + 0)
    }

    these_dim = (RELIABILITY_BIN_DIM,)
    this_integer_array = numpy.full(num_bins_for_reliability, 0, dtype=int)
    this_float_array = numpy.full(num_bins_for_reliability, 0, dtype=float)
    new_dict = {
        NUM_EXAMPLES_KEY: (these_dim, this_integer_array + 0),
        EVENT_FREQUENCY_KEY: (these_dim, this_float_array + 0.),
        MEAN_FORECAST_PROB_KEY: (these_dim, this_float_array + 0.)
    }

    main_data_dict.update(new_dict)
    basic_score_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    basic_score_table_xarray.attrs[CLIMO_EVENT_FREQ_KEY] = (
        event_frequency_in_training
    )

    for i in range(len(prediction_file_names)):
        (
            basic_score_table_xarray,
            num_examples_new,
            num_positive_examples_new
        ) = _update_basic_scores(
            basic_score_table_xarray=basic_score_table_xarray,
            prediction_file_name=prediction_file_names[i]
        )

        num_examples_total += num_examples_new
        num_positive_examples_total += num_positive_examples_new

        if i != len(prediction_file_names) - 1:
            print('\n')

    if (
            basic_score_table_xarray.attrs[CLIMO_EVENT_FREQ_KEY] is None
            and not test_mode
    ):
        basic_score_table_xarray.attrs[CLIMO_EVENT_FREQ_KEY] = (
            float(num_positive_examples_total) / num_examples_total
        )

    return basic_score_table_xarray


def get_advanced_scores(basic_score_table_xarray):
    """Computes advanced scores.

    :param basic_score_table_xarray: See doc for `get_basic_scores`.
    :return: advanced_score_table_xarray: xarray table with results (variable
        and dimension names should make the table self-explanatory).
    """

    # result_table_xarray.attrs[PREDICTION_FILE_KEY] = prediction_file_name

    probability_thresholds = (
        basic_score_table_xarray.coords[PROBABILITY_THRESHOLD_DIM].values
    )
    num_prob_thresholds = len(probability_thresholds)
    bin_indices = basic_score_table_xarray.coords[RELIABILITY_BIN_DIM].values

    metadata_dict = {
        PROBABILITY_THRESHOLD_DIM: probability_thresholds,
        RELIABILITY_BIN_DIM: bin_indices
    }

    these_dim = (PROBABILITY_THRESHOLD_DIM,)
    this_array = numpy.full(num_prob_thresholds, numpy.nan)
    main_data_dict = {
        POD_KEY: (these_dim, this_array + 0.),
        POFD_KEY: (these_dim, this_array + 0.),
        SUCCESS_RATIO_KEY: (these_dim, this_array + 0.),
        FREQUENCY_BIAS_KEY: (these_dim, this_array + 0.),
        CSI_KEY: (these_dim, this_array + 0.),
        ACCURACY_KEY: (these_dim, this_array + 0.),
        HEIDKE_SCORE_KEY: (these_dim, this_array + 0.)
    }

    advanced_score_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )

    for k in range(num_prob_thresholds):
        this_contingency_table = {
            gg_model_eval.NUM_TRUE_POSITIVES_KEY:
                basic_score_table_xarray[NUM_TRUE_POSITIVES_KEY].values[k],
            gg_model_eval.NUM_FALSE_POSITIVES_KEY:
                basic_score_table_xarray[NUM_FALSE_POSITIVES_KEY].values[k],
            gg_model_eval.NUM_FALSE_NEGATIVES_KEY:
                basic_score_table_xarray[NUM_FALSE_NEGATIVES_KEY].values[k],
            gg_model_eval.NUM_TRUE_NEGATIVES_KEY:
                basic_score_table_xarray[NUM_TRUE_NEGATIVES_KEY].values[k]
        }

        advanced_score_table_xarray[POD_KEY].values[k] = (
            gg_model_eval.get_pod(this_contingency_table)
        )
        advanced_score_table_xarray[POFD_KEY].values[k] = (
            gg_model_eval.get_pofd(this_contingency_table)
        )
        advanced_score_table_xarray[SUCCESS_RATIO_KEY].values[k] = (
            gg_model_eval.get_success_ratio(this_contingency_table)
        )
        advanced_score_table_xarray[FREQUENCY_BIAS_KEY].values[k] = (
            gg_model_eval.get_frequency_bias(this_contingency_table)
        )
        advanced_score_table_xarray[CSI_KEY].values[k] = (
            gg_model_eval.get_csi(this_contingency_table)
        )
        advanced_score_table_xarray[ACCURACY_KEY].values[k] = (
            gg_model_eval.get_accuracy(this_contingency_table)
        )
        advanced_score_table_xarray[HEIDKE_SCORE_KEY].values[k] = (
            gg_model_eval.get_heidke_score(this_contingency_table)
        )

    auc = gg_model_eval.get_area_under_roc_curve(
        pofd_by_threshold=advanced_score_table_xarray[POFD_KEY].values,
        pod_by_threshold=advanced_score_table_xarray[POD_KEY].values
    )
    aupd = gg_model_eval.get_area_under_perf_diagram(
        success_ratio_by_threshold=
        advanced_score_table_xarray[SUCCESS_RATIO_KEY].values,
        pod_by_threshold=advanced_score_table_xarray[POD_KEY].values
    )
    this_dict = gg_model_eval.get_brier_skill_score(
        mean_forecast_prob_by_bin=
        basic_score_table_xarray[MEAN_FORECAST_PROB_KEY].values,
        mean_observed_label_by_bin=
        basic_score_table_xarray[EVENT_FREQUENCY_KEY].values,
        num_examples_by_bin=basic_score_table_xarray[NUM_EXAMPLES_KEY].values,
        climatology=basic_score_table_xarray.attrs[CLIMO_EVENT_FREQ_KEY]
    )

    advanced_score_table_xarray.attrs[AUC_KEY] = auc
    advanced_score_table_xarray.attrs[AUPD_KEY] = aupd
    advanced_score_table_xarray.attrs[BRIER_SCORE_KEY] = (
        this_dict[gg_model_eval.BRIER_SCORE_KEY]
    )
    advanced_score_table_xarray.attrs[BRIER_SKILL_SCORE_KEY] = (
        this_dict[gg_model_eval.BSS_KEY]
    )

    return advanced_score_table_xarray


def write_file(score_table_xarray, output_file_name):
    """Writes evaluation results to NetCDF or Pickle file.

    :param score_table_xarray: Table created by `get_basic_scores` or
        `get_advanced_scores`.
    :param output_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    # score_table_xarray.to_netcdf(
    #     path=output_file_name, mode='w', format='NETCDF3_64BIT_OFFSET'
    # )

    if NUM_TRUE_POSITIVES_KEY in score_table_xarray:
        pickle_file_handle = open(output_file_name, 'wb')
        pickle.dump(output_file_name, pickle_file_handle)
        pickle_file_handle.close()
    else:
        score_table_xarray.to_netcdf(
            path=output_file_name, mode='w', format='NETCDF3_64BIT'
        )


def read_file(input_file_name):
    """Reads evaluation results from NetCDF or Pickle file.

    :param input_file_name: Path to input file.
    :return: score_table_xarray: Table created by `get_basic_scores` or
        `get_advanced_scores`.
    """

    error_checking.assert_file_exists(input_file_name)

    try:
        pickle_file_handle = open(input_file_name, 'rb')
        score_table_xarray = pickle.load(pickle_file_handle)
        pickle_file_handle.close()
    except:
        score_table_xarray = xarray.open_dataset(input_file_name)

    return score_table_xarray
