# functionality for self-paced learning (spl)

import math

def calculate_threshold_by_epoch(epoch, threshold_init, mu):
    """
    Calculate threshold for spl using epoch ID.

    :param epoch: current epoch ID (starting from 1)
    :param threshold_init: initial value setting for computing threshold, initial threshold is "1 / threshold_init"
    :param mu: each time, divide "threshold_init" by "mu" times to increase the threshold
    :return: threshold value for current epoch to decide a sample is easy/hard based on its loss
    """

    threshold_epoch = threshold_init / math.pow(mu, epoch - 1)
    updated_threshold = 1 / threshold_epoch
    return updated_threshold
