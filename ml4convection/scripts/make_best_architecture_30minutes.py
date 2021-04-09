"""Makes architecture for best 30-minute U-net in journal paper."""

import os
import copy
import numpy
from gewittergefahr.deep_learning import architecture_utils
from ml4convection.io import radar_io
from ml4convection.machine_learning import neural_net
from ml4convection.machine_learning import u_net_architecture

# Dimensions of predictors (satellite images).
NUM_GRID_ROWS = 205
NUM_GRID_COLUMNS = 205
NUM_SPECTRAL_BANDS = 7

# Best L_2 weight and number of lag times.  The exact lag times are
# {0, 20, 40, 60} minutes, but these are not specified in the architecture; they
# are specified only at training time, when feeding the input data to the model.
L2_WEIGHT = 10 ** -5.5
NUM_LAG_TIMES = 4

# Number of levels in U-net.
NUM_LEVELS = 5

# File containing mask, which ensures that only grid points within 100 km of the
# three southernmost radars are used in the loss function.  The mask is
# referenced to a radar-centered domain, with 205 x 205 grid points.
THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
MASK_FILE_NAME = '{0:s}/radar_mask_100km_omit-north_partial.nc'.format(
    THIS_DIRECTORY_NAME
)

# Loss function (fractions skill score with 9-by-9 window, or half-window size
# of grid points).
LOSS_FUNCTION_NAME = 'fss_neigh4'

# Metrics (other evaluation scores tracked during training).
METRIC_NAMES = [
    'fss_neigh0', 'fss_neigh1', 'fss_neigh2', 'fss_neigh3',
    'csi_neigh0', 'csi_neigh1', 'csi_neigh2', 'csi_neigh3',
    'bias_neigh0', 'bias_neigh1', 'bias_neigh2', 'bias_neigh3',
    'iou_neigh0', 'iou_neigh1', 'iou_neigh2', 'iou_neigh3',
    'dice_neigh0', 'dice_neigh1', 'dice_neigh2', 'dice_neigh3'
]

# DEFAULT_OPTION_DICT contains fixed hyperparameters, which were not varied
# during the hyperparameter experiment presented in the paper.
DEFAULT_OPTION_DICT = {
    u_net_architecture.NUM_LEVELS_KEY: NUM_LEVELS,
    u_net_architecture.NUM_CONV_LAYERS_KEY: 2,
    u_net_architecture.CONV_LAYER_CHANNEL_COUNTS_KEY:
        numpy.array([16, 24, 32, 48, 64, 96], dtype=int),
    u_net_architecture.CONV_LAYER_DROPOUT_RATES_KEY:
        numpy.full(NUM_LEVELS + 1, 0.),
    u_net_architecture.UPCONV_LAYER_DROPOUT_RATES_KEY:
        numpy.full(NUM_LEVELS, 0.),
    u_net_architecture.SKIP_LAYER_DROPOUT_RATES_KEY:
        numpy.full(NUM_LEVELS, 0.),
    u_net_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    u_net_architecture.OUTPUT_ACTIV_FUNCTION_KEY:
        architecture_utils.SIGMOID_FUNCTION_STRING,
    u_net_architecture.OUTPUT_ACTIV_FUNCTION_ALPHA_KEY: 0.,
    u_net_architecture.L1_WEIGHT_KEY: 0.,
    u_net_architecture.USE_BATCH_NORM_KEY: True
}


def _run():
    """Makes architecture for best 30-minute U-net in journal paper.

    This is effectively the main method.
    """
    
    # Read mask.
    print('Reading partial mask from: "{0:s}"...'.format(
        MASK_FILE_NAME
    ))
    mask_matrix = radar_io.read_mask_file(MASK_FILE_NAME)[
        radar_io.MASK_MATRIX_KEY
    ]
    
    # Create loss function.
    loss_function = neural_net.get_metrics(
        metric_names=[LOSS_FUNCTION_NAME], mask_matrix=mask_matrix,
        use_as_loss_function=True
    )[0][0]
    
    # Assign experimental hyperparameters.
    option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
    option_dict[u_net_architecture.L2_WEIGHT_KEY] = L2_WEIGHT
    option_dict[u_net_architecture.INPUT_DIMENSIONS_KEY] = numpy.array([
        NUM_GRID_ROWS, NUM_GRID_COLUMNS, NUM_LAG_TIMES * NUM_SPECTRAL_BANDS
    ], dtype=int)

    # This next command creates the U-net architecture and compiles the U-net.
    # Thus, it returns a U-net that is completely ready to train.  It also
    # prints a very long flow chart, containing details on each layer in the
    # model.  I have not included this flow chart in the paper, because it is
    # difficult to read and I feel that Figure 5 does a better job of
    # documenting the architecture.

    model_object = u_net_architecture.create_model(
        option_dict=option_dict, loss_function=loss_function,
        mask_matrix=mask_matrix, metric_names=METRIC_NAMES
    )


if __name__ == '__main__':
    _run()
