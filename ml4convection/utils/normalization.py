"""Methods for normalizing satellite variables (predictors)."""

import copy
import pickle
import numpy
import scipy.stats
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4convection.io import satellite_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_VALUES_KEY = 'num_values'
MEAN_VALUE_KEY = 'mean_value'
MEAN_OF_SQUARES_KEY = 'mean_of_squares'

BAND_NUMBERS_KEY = 'band_numbers'
SAMPLED_VALUES_KEY = 'sampled_value_matrix'
MEAN_VALUES_KEY = 'mean_values'
STANDARD_DEVIATIONS_KEY = 'standard_deviations'

MIN_CUMULATIVE_DENSITY = 1e-6
MAX_CUMULATIVE_DENSITY = 1. - 1e-6


def _update_normalization_params(normalization_param_dict, new_data_matrix):
    """Updates normalization parameters.

    :param normalization_param_dict: Dictionary with the following keys.
    normalization_param_dict['num_values']: Number of values on which current
        estimates are based.
    normalization_param_dict['mean_value']: Current mean.
    normalization_param_dict['mean_of_squares']: Current mean of squared values.

    :param new_data_matrix: numpy array with new values.  Will be used to
        update estimates in `normalization_param_dict`.
    :return: normalization_param_dict: Same as input, but with new estimates.
    """

    these_means = numpy.array([
        normalization_param_dict[MEAN_VALUE_KEY], numpy.mean(new_data_matrix)
    ])
    these_weights = numpy.array([
        normalization_param_dict[NUM_VALUES_KEY], new_data_matrix.size
    ])
    normalization_param_dict[MEAN_VALUE_KEY] = numpy.average(
        these_means, weights=these_weights
    )

    these_means = numpy.array([
        normalization_param_dict[MEAN_OF_SQUARES_KEY],
        numpy.mean(new_data_matrix ** 2)
    ])
    these_weights = numpy.array([
        normalization_param_dict[NUM_VALUES_KEY], new_data_matrix.size
    ])
    normalization_param_dict[MEAN_OF_SQUARES_KEY] = numpy.average(
        these_means, weights=these_weights
    )

    normalization_param_dict[NUM_VALUES_KEY] += new_data_matrix.size
    return normalization_param_dict


def _get_standard_deviation(normalization_param_dict):
    """Computes standard deviation.

    :param normalization_param_dict: See doc for `_update_normalization_params`.
    :return: standard_deviation: Standard deviation.
    """

    multiplier = float(
        normalization_param_dict[NUM_VALUES_KEY]
    ) / (normalization_param_dict[NUM_VALUES_KEY] - 1)

    return numpy.sqrt(multiplier * (
        normalization_param_dict[MEAN_OF_SQUARES_KEY] -
        normalization_param_dict[MEAN_VALUE_KEY] ** 2
    ))


def _actual_to_uniform_dist(actual_values_new, actual_values_training):
    """Converts values from actual to uniform distribution.

    :param actual_values_new: numpy array of actual (physical) values to
        convert.
    :param actual_values_training: numpy array of actual (physical) values in
        training data.
    :return: uniform_values_new: numpy array (same shape as `actual_values_new`)
        with rescaled values from 0...1.
    """

    actual_values_new_1d = numpy.ravel(actual_values_new)

    indices = numpy.searchsorted(
        numpy.sort(numpy.ravel(actual_values_training)), actual_values_new_1d,
        side='left'
    ).astype(float)

    num_values = actual_values_training.size
    uniform_values_new_1d = indices / (num_values - 1)
    uniform_values_new_1d = numpy.minimum(uniform_values_new_1d, 1.)

    return numpy.reshape(uniform_values_new_1d, actual_values_new.shape)


def _uniform_to_actual_dist(uniform_values_new, actual_values_training):
    """Converts values from uniform to actual distribution.

    This method is the inverse of `_actual_to_uniform_dist`.

    :param uniform_values_new: See doc for `_actual_to_uniform_dist`.
    :param actual_values_training: Same.
    :return: actual_values_new: Same.
    """

    uniform_values_new_1d = numpy.ravel(uniform_values_new)

    actual_values_new_1d = numpy.percentile(
        numpy.ravel(actual_values_training), 100 * uniform_values_new_1d,
        interpolation='linear'
    )

    return numpy.reshape(actual_values_new_1d, uniform_values_new.shape)


def _normalize_one_variable(actual_values_new, actual_values_training=None,
                            mean_value_training=None, stdev_training=None):
    """Normalizes one variable.

    If `actual_values_training is not None`, will convert new values to uniform
    distribution, then normal distribution.

    If `actual_values_training is None`, will convert directly to normal
    distribution, using `mean_value_training` and `stdev_training`.

    :param actual_values_new: See doc for `_actual_to_uniform_dist`.
    :param actual_values_training: Same.
    :param mean_value_training: Mean value in training data.
    :param stdev_training: Standard deviation in training data.
    :return: normalized_values_new: numpy array (same shape as
        `actual_values_new`) with normalized values (z-scores).
    """

    if actual_values_training is None:
        return (actual_values_new - mean_value_training) / stdev_training

    uniform_values_new = _actual_to_uniform_dist(
        actual_values_new=actual_values_new,
        actual_values_training=actual_values_training
    )

    uniform_values_new = numpy.maximum(
        uniform_values_new, MIN_CUMULATIVE_DENSITY
    )
    uniform_values_new = numpy.minimum(
        uniform_values_new, MAX_CUMULATIVE_DENSITY
    )
    return scipy.stats.norm.ppf(uniform_values_new, loc=0., scale=1.)


def _denorm_one_variable(normalized_values_new, actual_values_training=None,
                         mean_value_training=None, stdev_training=None):
    """Denormalizes one variable.

    This method is the inverse of `_normalize_one_variable`.

    :param normalized_values_new: See doc for `_normalize_one_variable`.
    :param actual_values_training: Same.
    :param mean_value_training: Same.
    :param stdev_training: Same.
    :return: actual_values_new: Same.
    """

    if actual_values_training is None:
        return mean_value_training + normalized_values_new * stdev_training

    uniform_values_new = scipy.stats.norm.cdf(
        normalized_values_new, loc=0., scale=1.
    )

    return _uniform_to_actual_dist(
        uniform_values_new=uniform_values_new,
        actual_values_training=actual_values_training
    )


def normalize_data(satellite_dict, uniformize=False,
                   norm_dict_for_temperature=None, norm_dict_for_count=None):
    """Normalizes all predictor variables.

    At least one of `norm_dict_for_temperature` and `norm_dict_for_count` must
    be specified.

    :param satellite_dict: See doc for `satellite_io.read_file`.
    :param uniformize: Boolean flag.  If True, will convert predictor values to
        uniform distribution, then normal distribution.  If False, will convert
        directly to normal distribution.
    :param norm_dict_for_temperature: See doc for `get_normalization_params`.
    :param norm_dict_for_count: Same.
    :return: satellite_dict: Same but with normalized predictor values.
    """

    # TODO(thunderhoser): Need to remove NaN's from predictor matrices
    # somewhere.

    do_temperatures = norm_dict_for_temperature is not None
    do_counts = norm_dict_for_count is not None
    error_checking.assert_is_greater(int(do_temperatures) + int(do_counts), 0)

    error_checking.assert_is_boolean(uniformize)
    band_numbers_new = satellite_dict[satellite_io.BAND_NUMBERS_KEY]

    for j in range(len(band_numbers_new)):
        print((
            'Normalizing brightness {0:s} in band {1:d}, '
            '{2:s} uniformization...'
        ).format(
            'temperatures and counts' if do_temperatures and do_counts
            else 'temperatures' if do_temperatures
            else 'counts',
            band_numbers_new[j],
            'with' if uniformize else 'without'
        ))

        if do_temperatures:
            main_key = satellite_io.BRIGHTNESS_TEMP_KEY
            k = numpy.where(
                norm_dict_for_temperature[BAND_NUMBERS_KEY] ==
                band_numbers_new[j]
            )[0][0]

            if uniformize:
                satellite_dict[main_key][..., j] = _normalize_one_variable(
                    actual_values_new=satellite_dict[main_key][..., j],
                    actual_values_training=
                    norm_dict_for_temperature[SAMPLED_VALUES_KEY][:, k]
                )
            else:
                satellite_dict[main_key][..., j] = _normalize_one_variable(
                    actual_values_new=satellite_dict[main_key][..., j],
                    mean_value_training=
                    norm_dict_for_temperature[MEAN_VALUES_KEY][k],
                    stdev_training=
                    norm_dict_for_temperature[STANDARD_DEVIATIONS_KEY][k]
                )

        if do_counts:
            main_key = satellite_io.BRIGHTNESS_COUNT_KEY
            k = numpy.where(
                norm_dict_for_count[BAND_NUMBERS_KEY] == band_numbers_new[j]
            )[0][0]

            if uniformize:
                satellite_dict[main_key][..., j] = _normalize_one_variable(
                    actual_values_new=satellite_dict[main_key][..., j],
                    actual_values_training=
                    norm_dict_for_count[SAMPLED_VALUES_KEY][:, k]
                )
            else:
                satellite_dict[main_key][..., j] = _normalize_one_variable(
                    actual_values_new=satellite_dict[main_key][..., j],
                    mean_value_training=norm_dict_for_count[MEAN_VALUES_KEY][k],
                    stdev_training=
                    norm_dict_for_count[STANDARD_DEVIATIONS_KEY][k]
                )

    return satellite_dict


def denormalize_data(satellite_dict, uniformize=False,
                     norm_dict_for_temperature=None, norm_dict_for_count=None):
    """Denormalizes all predictor variables.

    At least one of `norm_dict_for_temperature` and `norm_dict_for_count` must
    be specified.

    :param satellite_dict: See doc for `normalize_data`.
    :param uniformize: Same.
    :param norm_dict_for_temperature: Same.
    :param norm_dict_for_count: Same.
    :return: satellite_dict: Same.
    """

    do_temperatures = norm_dict_for_temperature is not None
    do_counts = norm_dict_for_count is not None
    error_checking.assert_is_greater(int(do_temperatures) + int(do_counts), 0)

    error_checking.assert_is_boolean(uniformize)
    band_numbers_new = satellite_dict[satellite_io.BAND_NUMBERS_KEY]

    for j in range(len(band_numbers_new)):
        print((
            'Denormalizing brightness {0:s} in band {1:d}, '
            '{2:s} uniformization...'
        ).format(
            'temperatures and counts' if do_temperatures and do_counts
            else 'temperatures' if do_temperatures
            else 'counts',
            band_numbers_new[j],
            'with' if uniformize else 'without'
        ))

        if do_temperatures:
            main_key = satellite_io.BRIGHTNESS_TEMP_KEY
            k = numpy.where(
                norm_dict_for_temperature[BAND_NUMBERS_KEY] ==
                band_numbers_new[j]
            )[0][0]

            if uniformize:
                satellite_dict[main_key][..., j] = _denorm_one_variable(
                    normalized_values_new=satellite_dict[main_key][..., j],
                    actual_values_training=
                    norm_dict_for_temperature[SAMPLED_VALUES_KEY][:, k]
                )
            else:
                satellite_dict[main_key][..., j] = _denorm_one_variable(
                    normalized_values_new=satellite_dict[main_key][..., j],
                    mean_value_training=
                    norm_dict_for_temperature[MEAN_VALUES_KEY][k],
                    stdev_training=
                    norm_dict_for_temperature[STANDARD_DEVIATIONS_KEY][k]
                )

        if do_counts:
            main_key = satellite_io.BRIGHTNESS_COUNT_KEY
            k = numpy.where(
                norm_dict_for_count[BAND_NUMBERS_KEY] == band_numbers_new[j]
            )[0][0]

            if uniformize:
                satellite_dict[main_key][..., j] = _denorm_one_variable(
                    normalized_values_new=satellite_dict[main_key][..., j],
                    actual_values_training=
                    norm_dict_for_count[SAMPLED_VALUES_KEY][:, k]
                )
            else:
                satellite_dict[main_key][..., j] = _denorm_one_variable(
                    normalized_values_new=satellite_dict[main_key][..., j],
                    mean_value_training=norm_dict_for_count[MEAN_VALUES_KEY][k],
                    stdev_training=
                    norm_dict_for_count[STANDARD_DEVIATIONS_KEY][k]
                )

    return satellite_dict


def get_normalization_params(
        top_satellite_dir_name, first_date_string, last_date_string,
        num_values_per_band, do_temperatures, do_counts):
    """Computes normalization parameters (mean and stdev for each band).

    C = number of spectral bands
    T = `num_values_per_band`

    :param top_satellite_dir_name: See doc for `satellite_io.find_many_files`.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param num_values_per_band: Number of values to save from each band (will be
        used for uniformization).  These values will be randomly sampled.
    :param do_temperatures: Boolean flag.  If True, will find normalization
        params for brightness temperatures.
    :param do_counts: Boolean flag.  If True, will find normalization params for
        brightness counts.
    :return: norm_dict_for_temperature: Dictionary with the following keys.
        If `do_temperatures == False`, this will be None.
    norm_dict_for_temperature['band_numbers']: length-C numpy array of band
        numbers (integers).
    norm_dict_for_temperature['sampled_value_matrix']: T-by-C numpy array of
        values randomly sampled from files.
    norm_dict_for_temperature['mean_values']: length-C numpy array of means.
    norm_dict_for_temperature['standard_deviations']: length-C numpy array of
        standard deviations.

    :return: norm_dict_for_count: Same but for brightness counts
        instead of temperatures.  If `do_counts == False`, this will be None.
    """

    error_checking.assert_is_integer(num_values_per_band)
    error_checking.assert_is_geq(num_values_per_band, 10000)
    error_checking.assert_is_boolean(do_temperatures)
    error_checking.assert_is_boolean(do_counts)
    error_checking.assert_is_greater(int(do_temperatures) + int(do_counts), 0)

    satellite_file_names = satellite_io.find_many_files(
        top_directory_name=top_satellite_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        raise_error_if_all_missing=True, raise_error_if_any_missing=True
    )

    num_files = len(satellite_file_names)
    num_values_per_band_per_file = int(numpy.ceil(
        float(num_values_per_band) / num_files
    ))
    num_values_per_band = num_values_per_band_per_file * num_files

    band_numbers = satellite_io.BAND_NUMBERS
    num_bands = len(band_numbers)

    original_param_dict = {
        NUM_VALUES_KEY: 0,
        MEAN_VALUE_KEY: 0.,
        MEAN_OF_SQUARES_KEY: 0.
    }

    if do_temperatures:
        param_dicts_for_temperature = (
            [copy.deepcopy(original_param_dict)] * num_bands
        )
        sampled_temperature_matrix_kelvins = numpy.full(
            (num_values_per_band, num_bands), numpy.nan
        )

    if do_counts:
        param_dicts_for_count = (
            [copy.deepcopy(original_param_dict)] * num_bands
        )
        sampled_count_matrix = numpy.full(
            (num_values_per_band, num_bands), numpy.nan
        )

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(satellite_file_names[i]))
        this_satellite_dict = satellite_io.read_file(
            netcdf_file_name=satellite_file_names[i],
            read_temperatures=do_temperatures, read_counts=do_counts
        )

        this_satellite_dict = satellite_io.subset_by_band(
            satellite_dict=this_satellite_dict, band_numbers=band_numbers
        )

        this_first_index = i * num_values_per_band_per_file
        this_last_index = this_first_index + num_values_per_band_per_file

        for j in range(num_bands):
            if do_temperatures:
                these_temperatures_kelvins = numpy.ravel(
                    this_satellite_dict[satellite_io.BRIGHTNESS_TEMP_KEY][
                        ..., j
                    ]
                )
                these_temperatures_kelvins = these_temperatures_kelvins[
                    numpy.isfinite(these_temperatures_kelvins)
                ]

                sampled_temperature_matrix_kelvins[
                    this_first_index:this_last_index, j
                ] = numpy.random.choice(
                    these_temperatures_kelvins,
                    size=num_values_per_band_per_file, replace=False
                )

                param_dicts_for_temperature[j] = _update_normalization_params(
                    normalization_param_dict=param_dicts_for_temperature[j],
                    new_data_matrix=these_temperatures_kelvins
                )

            if do_counts:
                these_counts = numpy.ravel(
                    this_satellite_dict[satellite_io.BRIGHTNESS_COUNT_KEY][
                        ..., j
                    ]
                )
                these_counts = these_counts[numpy.isfinite(these_counts)]

                sampled_count_matrix[this_first_index:this_last_index, j] = (
                    numpy.random.choice(
                        these_counts,
                        size=num_values_per_band_per_file, replace=False
                    )
                )

                param_dicts_for_count[j] = _update_normalization_params(
                    normalization_param_dict=param_dicts_for_count[j],
                    new_data_matrix=these_counts
                )

    if do_temperatures:
        error_checking.assert_is_numpy_array_without_nan(
            sampled_temperature_matrix_kelvins
        )

        norm_dict_for_temperature = {
            BAND_NUMBERS_KEY: band_numbers,
            SAMPLED_VALUES_KEY: sampled_temperature_matrix_kelvins,
            MEAN_VALUES_KEY: numpy.array([
                d[MEAN_VALUE_KEY] for d in param_dicts_for_temperature
            ]),
            STANDARD_DEVIATIONS_KEY: numpy.array([
                _get_standard_deviation(d) for d in param_dicts_for_temperature
            ])
        }

        print(SEPARATOR_STRING)

        for j in range(num_bands):
            print((
                'Band number = {0:d} ... mean temperature = {1:.2f} K ... '
                'standard deviation = {2:.3f} K'
            ).format(
                norm_dict_for_temperature['band_numbers'][j],
                norm_dict_for_temperature['mean_values'][j],
                norm_dict_for_temperature['standard_deviations'][j]
            ))
    else:
        norm_dict_for_temperature = None

    if do_counts:
        error_checking.assert_is_numpy_array_without_nan(sampled_count_matrix)

        norm_dict_for_count = {
            BAND_NUMBERS_KEY: band_numbers,
            SAMPLED_VALUES_KEY: sampled_count_matrix,
            MEAN_VALUES_KEY: numpy.array([
                d[MEAN_VALUE_KEY] for d in param_dicts_for_count
            ]),
            STANDARD_DEVIATIONS_KEY: numpy.array([
                _get_standard_deviation(d) for d in param_dicts_for_count
            ])
        }

        print(SEPARATOR_STRING)

        for j in range(num_bands):
            print((
                'Band number = {0:d} ... mean count = {1:.2f} ... '
                'standard deviation = {2:.3f}'
            ).format(
                norm_dict_for_count['band_numbers'][j],
                norm_dict_for_count['mean_values'][j],
                norm_dict_for_count['standard_deviations'][j]
            ))
    else:
        norm_dict_for_count = None

    return norm_dict_for_temperature, norm_dict_for_count


def write_file(pickle_file_name, norm_dict_for_temperature,
               norm_dict_for_count):
    """Writes normalization parameters to Pickle file.

    :param pickle_file_name: Path to output file.
    :param norm_dict_for_temperature: See doc for `get_normalization_params`.
    :param norm_dict_for_count: Same.
    """

    have_temperatures = norm_dict_for_temperature is not None
    have_counts = norm_dict_for_count is not None
    error_checking.assert_is_greater(
        int(have_temperatures) + int(have_counts), 0
    )

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(norm_dict_for_temperature, pickle_file_handle)
    pickle.dump(norm_dict_for_count, pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads normalization parameters from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: norm_dict_for_temperature: See doc for `get_normalization_params`.
    :return: norm_dict_for_count: Same.
    """

    error_checking.assert_file_exists(pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'rb')
    norm_dict_for_temperature = pickle.load(pickle_file_handle)
    norm_dict_for_count = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return norm_dict_for_temperature, norm_dict_for_count
