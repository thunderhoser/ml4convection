"""Methods for training ML models."""

import copy
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml4convection.io import satellite_io
from ml4convection.io import radar_io
from ml4convection.utils import normalization

TOLERANCE = 1e-6

DAYS_TO_SECONDS = 86400
DATE_FORMAT = '%Y%m%d'

SATELLITE_DIRECTORY_KEY = 'top_satellite_dir_name'
RADAR_DIRECTORY_KEY = 'top_radar_dir_name'
BATCH_SIZE_KEY = 'num_examples_per_batch'
BAND_NUMBERS_KEY = 'band_numbers'
LEAD_TIME_KEY = 'lead_time_seconds'
REFL_THRESHOLD_KEY = 'reflectivity_threshold_dbz'
FIRST_TIME_KEY = 'first_time_unix_sec'
LAST_TIME_KEY = 'last_time_unix_sec'
NORMALIZATION_FILE_KEY = 'normalization_file_name'
UNIFORMIZE_FLAG_KEY = 'uniformize'

DEFAULT_GENERATOR_OPTION_DICT = {
    BATCH_SIZE_KEY: 256,
    BAND_NUMBERS_KEY: satellite_io.BAND_NUMBERS,
    REFL_THRESHOLD_KEY: 35.,
    UNIFORMIZE_FLAG_KEY: False
}


def _check_generator_args(option_dict):
    """Error-checks input arguments for generator.

    :param option_dict: See doc for `data_generator`.
    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_GENERATOR_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_numpy_array(
        option_dict[BAND_NUMBERS_KEY], num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(option_dict[BAND_NUMBERS_KEY])

    error_checking.assert_is_integer(option_dict[BATCH_SIZE_KEY])
    error_checking.assert_is_geq(option_dict[BATCH_SIZE_KEY], 2)
    error_checking.assert_is_integer(option_dict[LEAD_TIME_KEY])
    error_checking.assert_is_geq(option_dict[LEAD_TIME_KEY], 0)
    error_checking.assert_is_less_than(
        option_dict[LEAD_TIME_KEY], DAYS_TO_SECONDS
    )
    error_checking.assert_is_greater(option_dict[REFL_THRESHOLD_KEY], 0.)

    return option_dict


def _read_inputs_one_day(
        valid_date_string, satellite_file_names, band_numbers,
        norm_dict_for_count, uniformize, radar_file_names, lead_time_seconds,
        reflectivity_threshold_dbz, num_examples_to_read):
    """Reads inputs (satellite and radar data) for one day.

    :param valid_date_string: Valid date (format "yyyymmdd").
    :param satellite_file_names: 1-D list of paths to satellite files (readable
        by `satellite_io.read_file`).
    :param band_numbers: See doc for `data_generator`.
    :param norm_dict_for_count: Dictionary returned by
        `normalization.read_file`.  Will use this to normalize satellite data.
    :param uniformize: See doc for `data_generator`.
    :param radar_file_names: 1-D list of paths to radar files (readable by
        `radar_io.read_file`).
    :param lead_time_seconds: See doc for `data_generator`.
    :param reflectivity_threshold_dbz: Same.
    :param num_examples_to_read: Number of examples to read.
    :return: predictor_matrix: See doc for `data_generator`.
    :return: target_matrix: Same.
    """

    radar_date_strings = [
        radar_io.file_name_to_date(f) for f in radar_file_names
    ]
    index = radar_date_strings.index(valid_date_string)
    desired_radar_file_name = radar_file_names[index]

    satellite_date_strings = [
        satellite_io.file_name_to_date(f) for f in satellite_file_names
    ]
    index = satellite_date_strings.index(valid_date_string)
    desired_satellite_file_names = [satellite_file_names[index]]

    if lead_time_seconds > 0:
        desired_satellite_file_names.insert(0, satellite_file_names[index - 1])
    
    print('Reading data from: "{0:s}"...'.format(desired_radar_file_name))
    radar_dict = radar_io.read_2d_file(desired_radar_file_name)
    
    satellite_dicts = []

    for this_file_name in desired_satellite_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_satellite_dict = satellite_io.read_file(
            netcdf_file_name=this_file_name, read_temperatures=False,
            read_counts=True
        )
        this_satellite_dict = satellite_io.subset_by_band(
            satellite_dict=this_satellite_dict, band_numbers=band_numbers
        )
        satellite_dicts.append(this_satellite_dict)

    satellite_dict = satellite_io.concat_data(satellite_dicts)

    assert numpy.allclose(
        radar_dict[radar_io.LATITUDES_KEY],
        satellite_dict[satellite_io.LATITUDES_KEY],
        atol=TOLERANCE
    )

    assert numpy.allclose(
        radar_dict[radar_io.LONGITUDES_KEY],
        satellite_dict[satellite_io.LONGITUDES_KEY],
        atol=TOLERANCE
    )
    
    valid_times_unix_sec = radar_dict[radar_io.VALID_TIMES_KEY]
    init_times_unix_sec = valid_times_unix_sec - lead_time_seconds
    
    good_flags = numpy.array([
        t in satellite_dict[satellite_io.VALID_TIMES_KEY]
        for t in init_times_unix_sec
    ], dtype=bool)
    
    if not numpy.any(good_flags):
        return None, None

    good_indices = numpy.where(good_flags)[0]
    valid_times_unix_sec = valid_times_unix_sec[good_indices]
    init_times_unix_sec = init_times_unix_sec[good_indices]

    radar_dict = radar_io.subset_by_time(
        radar_dict=radar_dict, desired_times_unix_sec=valid_times_unix_sec
    )[0]
    satellite_dict = satellite_io.subset_by_time(
        satellite_dict=satellite_dict,
        desired_times_unix_sec=init_times_unix_sec
    )[0]
    num_examples = len(good_indices)

    if num_examples >= num_examples_to_read:
        desired_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        )
        desired_indices = numpy.random.choice(
            desired_indices, size=num_examples_to_read, replace=False
        )

        radar_dict = radar_io.subset_by_index(
            radar_dict=radar_dict, desired_indices=desired_indices
        )
        satellite_dict = satellite_io.subset_by_index(
            satellite_dict=satellite_dict, desired_indices=desired_indices
        )

    satellite_dict = normalization.normalize_data(
        satellite_dict=satellite_dict, uniformize=uniformize,
        norm_dict_for_count=norm_dict_for_count
    )
    predictor_matrix = satellite_dict[satellite_io.BRIGHTNESS_COUNT_KEY]

    target_matrix = (
            radar_dict[
                radar_io.COMPOSITE_REFL_KEY] >= reflectivity_threshold_dbz
    ).astype(int)
    target_matrix = numpy.expand_dims(target_matrix, axis=-1)

    return predictor_matrix, target_matrix


def data_generator(option_dict):
    """Generates training data.

    E = number of examples per batch
    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (spectral bands)

    :param option_dict: Dictionary with the following keys.
    option_dict['top_satellite_dir_name']: Name of top-level directory with
        satellite data (predictors).  Files therein will be found by
        `satellite_io.find_file` and read by `satellite_io.read_file`.
    option_dict['top_radar_dir_name']: Name of top-level directory with radar
        data (targets).  Files therein will be found by `radar_io.find_file` and
        read by `radar_io.read_2d_file`.
    option_dict['num_examples_per_batch']: Batch size.
    option_dict['band_numbers']: 1-D numpy array of band numbers (integers) for
        satellite data.  Will use only these spectral bands as predictors.
    option_dict['lead_time_seconds']: Lead time for prediction.
    option_dict['reflectivity_threshold_dbz']: Reflectivity threshold for
       convection.  Only grid cells with composite (column-max) reflectivity >=
       threshold will be called convective.
    option_dict['first_valid_date_string']: First valid date (format
        "yyyymmdd").  Will not generate examples with radar data before this
        time.
    option_dict['last_valid_date_string']: Last valid date (format "yyyymmdd").
        Will not generate examples with radar data after this time.
    option_dict['normalization_file_name']: File with training examples to use
        for normalization (will be read by `example_io.read_file`).`
    option_dict['uniformize']: Boolean flag.  If True, will convert satellite
        values to uniform distribution before normal distribution.  If False,
        will go directly to normal distribution.

    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values,
        based on satellite data.
    :return: target_matrix: E-by-M-by-N numpy array of target values (integers
        in 0...1, indicating whether or not convection occurs at the given lead
        time).
    """

    # TODO(thunderhoser): Allow generator to read brightness temperatures
    # instead of counts.

    option_dict = _check_generator_args(option_dict)

    top_satellite_dir_name = option_dict[SATELLITE_DIRECTORY_KEY]
    top_radar_dir_name = option_dict[RADAR_DIRECTORY_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    band_numbers = option_dict[BAND_NUMBERS_KEY]
    lead_time_seconds = option_dict[LEAD_TIME_KEY]
    reflectivity_threshold_dbz = option_dict[REFL_THRESHOLD_KEY]
    first_valid_date_string = option_dict[FIRST_DATE_KEY]
    last_valid_date_string = option_dict[LAST_DATE_KEY]
    normalization_file_name = option_dict[NORMALIZATION_FILE_KEY]
    uniformize = option_dict[UNIFORMIZE_FLAG_KEY]

    # TODO(thunderhoser): Figure out this fuckery.

    if lead_time_seconds == 0:
        first_init_date_string = copy.deepcopy(first_valid_date_string)
        last_init_date_string = copy.deepcopy(last_valid_date_string)
    else:
        first_valid_time_unix_sec = time_conversion.string_to_unix_sec(
            first_valid_date_string, DATE_FORMAT
        )
        first_init_date_string = time_conversion.unix_sec_to_string(
            first_valid_time_unix_sec - DAYS_TO_SECONDS, DATE_FORMAT
        )

        last_valid_time_unix_sec = time_conversion.string_to_unix_sec(
            last_valid_date_string, DATE_FORMAT
        )
        last_init_date_string = time_conversion.unix_sec_to_string(
            last_valid_time_unix_sec - DAYS_TO_SECONDS, DATE_FORMAT
        )

    print('Reading normalization parameters from: "{0:s}"...'.format(
        normalization_file_name
    ))
    norm_dict_for_count = normalization.read_file(normalization_file_name)[1]

    satellite_file_names = satellite_io.find_many_files(
        top_directory_name=top_satellite_dir_name,
        first_date_string=first_init_date_string,
        last_date_string=last_init_date_string,
        raise_error_if_any_missing=True
    )

    radar_file_names = radar_io.find_many_files(
        top_directory_name=top_radar_dir_name,
        first_date_string=first_init_date_string,
        last_date_string=last_init_date_string,
        with_3d=False, raise_error_if_any_missing=True
    )

    num_examples_in_memory = 0
    file_index = 0

    while True:
        if num_examples_in_memory >= num_examples_per_batch:
            raise StopIteration

        num_examples_in_memory = 0
        predictor_matrix = None
        target_values = None

        while num_examples_in_memory < num_examples_per_batch:
            # TODO(thunderhoser): Figure out this fuckery.
            # TODO(thunderhoser): Write method to read single file.

            if example_index == len(all_desired_id_strings):
                if for_inference:
                    if predictor_matrix is None:
                        raise StopIteration

                    break

                example_index = 0

            this_num_examples = num_examples_per_batch - num_examples_in_memory