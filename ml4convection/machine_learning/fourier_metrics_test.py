"""Unit tests for fourier_metrics.py."""

import unittest
import numpy
from keras import backend as K
from ml4convection.machine_learning import fourier_metrics
from ml4convection.machine_learning import custom_metrics_test

TOLERANCE = 1e-3
SMALL_NUMBER = K.eval(K.epsilon())

TARGET_MATRIX = custom_metrics_test.TARGET_MATRIX
PREDICTION_MATRIX = custom_metrics_test.PREDICTION_MATRIX
TARGET_TENSOR = custom_metrics_test.TARGET_TENSOR
PREDICTION_TENSOR = custom_metrics_test.PREDICTION_TENSOR
MASK_MATRIX = custom_metrics_test.MASK_MATRIX

SPATIAL_COEFF_MATRIX = numpy.full((30, 36), 1.)
FREQUENCY_COEFF_MATRIX = numpy.full((30, 36), 1.)

BRIER_SCORE = custom_metrics_test.BRIER_SCORE_NEIGH0
CSI_VALUE = custom_metrics_test.CSI_NEIGH0
FREQUENCY_BIAS = 41.6 / (40 + SMALL_NUMBER)
POSITIVE_IOU_VALUE = custom_metrics_test.POSITIVE_IOU_NEIGH0
ALL_CLASS_IOU_VALUE = custom_metrics_test.ALL_CLASS_IOU_NEIGH0
DICE_COEFF = custom_metrics_test.DICE_COEFF_NEIGH0

ACTUAL_MSE = numpy.mean(
    numpy.expand_dims(MASK_MATRIX, axis=0) *
    (TARGET_MATRIX - PREDICTION_MATRIX) ** 2
)
REFERENCE_MSE = numpy.mean(
    numpy.expand_dims(MASK_MATRIX, axis=0) *
    (TARGET_MATRIX ** 2 + PREDICTION_MATRIX ** 2)
)
PIXELWISE_FSS = 1. - ACTUAL_MSE / REFERENCE_MSE


class FourierMetricsTests(unittest.TestCase):
    """Each method is a unit test for fourier_metrics.py."""

    def test_brier_score(self):
        """Ensures correct output from brier_score()."""

        this_function = fourier_metrics.brier_score(
            spatial_coeff_matrix=SPATIAL_COEFF_MATRIX,
            frequency_coeff_matrix=FREQUENCY_COEFF_MATRIX,
            mask_matrix=MASK_MATRIX
        )
        this_brier_score = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_brier_score, BRIER_SCORE, atol=TOLERANCE
        ))

    def test_csi(self):
        """Ensures correct output from csi()."""

        this_function = fourier_metrics.csi(
            spatial_coeff_matrix=SPATIAL_COEFF_MATRIX,
            frequency_coeff_matrix=FREQUENCY_COEFF_MATRIX,
            mask_matrix=MASK_MATRIX, use_as_loss_function=False
        )
        this_csi = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(numpy.isclose(this_csi, CSI_VALUE, atol=TOLERANCE))

    def test_frequency_bias(self):
        """Ensures correct output from frequency_bias()."""

        this_function = fourier_metrics.frequency_bias(
            spatial_coeff_matrix=SPATIAL_COEFF_MATRIX,
            frequency_coeff_matrix=FREQUENCY_COEFF_MATRIX,
            mask_matrix=MASK_MATRIX
        )
        this_bias = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(numpy.isclose(
            this_bias, FREQUENCY_BIAS, atol=TOLERANCE
        ))

    def test_pixelwise_fss(self):
        """Ensures correct output from pixelwise_fss()."""

        this_function = fourier_metrics.pixelwise_fss(
            spatial_coeff_matrix=SPATIAL_COEFF_MATRIX,
            frequency_coeff_matrix=FREQUENCY_COEFF_MATRIX,
            mask_matrix=MASK_MATRIX, use_as_loss_function=False
        )
        this_fss = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(numpy.isclose(
            this_fss, PIXELWISE_FSS, atol=TOLERANCE
        ))

    def test_iou(self):
        """Ensures correct output from iou()."""

        this_function = fourier_metrics.iou(
            spatial_coeff_matrix=SPATIAL_COEFF_MATRIX,
            frequency_coeff_matrix=FREQUENCY_COEFF_MATRIX,
            mask_matrix=MASK_MATRIX, use_as_loss_function=False
        )
        this_iou = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(numpy.isclose(
            this_iou, POSITIVE_IOU_VALUE, atol=TOLERANCE
        ))

    def test_all_class_iou(self):
        """Ensures correct output from all_class_iou()."""

        this_function = fourier_metrics.all_class_iou(
            spatial_coeff_matrix=SPATIAL_COEFF_MATRIX,
            frequency_coeff_matrix=FREQUENCY_COEFF_MATRIX,
            mask_matrix=MASK_MATRIX, use_as_loss_function=False
        )
        this_iou = K.eval(this_function(TARGET_TENSOR, PREDICTION_TENSOR))

        self.assertTrue(numpy.isclose(
            this_iou, ALL_CLASS_IOU_VALUE, atol=TOLERANCE
        ))

    def test_dice_coeff(self):
        """Ensures correct output from dice_coeff()."""

        this_function = fourier_metrics.dice_coeff(
            spatial_coeff_matrix=SPATIAL_COEFF_MATRIX,
            frequency_coeff_matrix=FREQUENCY_COEFF_MATRIX,
            mask_matrix=MASK_MATRIX, use_as_loss_function=False
        )
        this_dice_coeff = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )

        self.assertTrue(numpy.isclose(
            this_dice_coeff, DICE_COEFF, atol=TOLERANCE
        ))

    def test_metrics(self):
        """Ensures correct output from metrics()."""

        these_metric_names = [
            fourier_metrics.BRIER_SCORE_NAME + '_foo',
            fourier_metrics.CSI_NAME + '_foo',
            fourier_metrics.FREQUENCY_BIAS_NAME + '_foo',
            fourier_metrics.FSS_NAME + '_foo',
            fourier_metrics.IOU_NAME + '_foo',
            fourier_metrics.ALL_CLASS_IOU_NAME + '_foo',
            fourier_metrics.DICE_COEFF_NAME + '_foo',
            fourier_metrics.REAL_FREQ_MSE_NAME + '_foo',
            fourier_metrics.IMAGINARY_FREQ_MSE_NAME + '_foo',
            fourier_metrics.FREQ_MSE_NAME + '_foo'
        ]

        this_function = fourier_metrics.metrics(
            spatial_coeff_matrix=SPATIAL_COEFF_MATRIX,
            frequency_coeff_matrix=FREQUENCY_COEFF_MATRIX,
            mask_matrix=MASK_MATRIX,
            metric_names=these_metric_names, function_name='metrics_foo'
        )

        metrics_dict = this_function(TARGET_TENSOR, PREDICTION_TENSOR)

        self.assertTrue(numpy.isclose(
            K.eval(metrics_dict[fourier_metrics.BRIER_SCORE_NAME + '_foo']),
            BRIER_SCORE, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            K.eval(metrics_dict[fourier_metrics.CSI_NAME + '_foo']),
            CSI_VALUE, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            K.eval(metrics_dict[fourier_metrics.FREQUENCY_BIAS_NAME + '_foo']),
            FREQUENCY_BIAS, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            K.eval(metrics_dict[fourier_metrics.FSS_NAME + '_foo']),
            PIXELWISE_FSS, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            K.eval(metrics_dict[fourier_metrics.IOU_NAME + '_foo']),
            POSITIVE_IOU_VALUE, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            K.eval(metrics_dict[fourier_metrics.ALL_CLASS_IOU_NAME + '_foo']),
            ALL_CLASS_IOU_VALUE, atol=TOLERANCE
        ))
        self.assertTrue(numpy.isclose(
            K.eval(metrics_dict[fourier_metrics.DICE_COEFF_NAME + '_foo']),
            DICE_COEFF, atol=TOLERANCE
        ))

        self.assertTrue(
            K.eval(metrics_dict[fourier_metrics.REAL_FREQ_MSE_NAME + '_foo'])
            > 0.
        )
        self.assertTrue(
            K.eval(
                metrics_dict[fourier_metrics.IMAGINARY_FREQ_MSE_NAME + '_foo']
            ) > 0.
        )
        self.assertTrue(
            K.eval(metrics_dict[fourier_metrics.FREQ_MSE_NAME + '_foo'])
            > 0.
        )

    def test_frequency_domain_mse_real(self):
        """Ensures correct output from frequency_domain_mse_real."""

        this_function = fourier_metrics.frequency_domain_mse_real(
            spatial_coeff_matrix=SPATIAL_COEFF_MATRIX,
            frequency_coeff_matrix=FREQUENCY_COEFF_MATRIX
        )

        this_mse = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )
        self.assertTrue(this_mse > 0.)

    def test_frequency_domain_mse_imag(self):
        """Ensures correct output from frequency_domain_mse_imag."""

        this_function = fourier_metrics.frequency_domain_mse_imag(
            spatial_coeff_matrix=SPATIAL_COEFF_MATRIX,
            frequency_coeff_matrix=FREQUENCY_COEFF_MATRIX
        )

        this_mse = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )
        self.assertTrue(this_mse > 0.)

    def test_frequency_domain_mse(self):
        """Ensures correct output from frequency_domain_mse."""

        this_function = fourier_metrics.frequency_domain_mse(
            spatial_coeff_matrix=SPATIAL_COEFF_MATRIX,
            frequency_coeff_matrix=FREQUENCY_COEFF_MATRIX
        )

        this_mse = K.eval(
            this_function(TARGET_TENSOR, PREDICTION_TENSOR)
        )
        self.assertTrue(this_mse > 0.)


if __name__ == '__main__':
    unittest.main()
