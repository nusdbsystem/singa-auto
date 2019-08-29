# functionality for weighted loss revision

import torch
import torch.nn.functional as F


def revised_weighted_activation(dense_outputs, slope, dim=1):
    """
    Calculate revised weighted activation based on dense outputs.

    :param dense_outputs: outputs from dense layer
    :param slope: slope value when changing loss function calculation
    :param dim: a dimension along which log_softmax will be computed, 1 as default.
    :return: revised weighted activation result after log_softmax
    """

    dense_outputs = torch.div(dense_outputs, slope)
    activation_outputs = F.log_softmax(dense_outputs, dim)
    return activation_outputs


def revised_weighted_loss(activation_outputs, ground_truth, slope):
    """
    Calculate revised weighted loss based on revised weighted activation outputs.

    :param activation_outputs: revised weighted activation result after log_softmax
    :param ground_truth: ground truth class for each sample
    :param slope: slope value when changing loss function calculation
    :return: revised weighted loss
    """

    revised_loss = - slope * F.nll_loss(activation_outputs, ground_truth)
    return revised_loss
