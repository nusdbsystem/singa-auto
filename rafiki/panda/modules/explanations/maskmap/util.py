from __future__ import division

import numpy as np
import argparse


def evaluate_explanation_score(x, y, w, h, saliency_map):
    """
    :param x: normalized coordinate x int
    :param y: normalized coordinate y int
    :param w: normalized coordinate w int
    :param h: normalized coordinate h int
    :param saliency_map: saliency map
    :return:
    """
    bbox = np.zeros_like(saliency_map)
    bbox[int(y):int(y + h), int(x):int(x + w)] = 1

    '''
    fig, ax = plt.subplots()
    im = ax.imshow(bbox)
    plt.show()

    '''

    corr = np.mean(np.multiply(bbox, saliency_map))

    return corr


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="/home/feiyi/workspace/git-repo/PANDAmodels/datasets/chestxray/", type=str)
    parser.add_argument("--model", default="densenet", type=str)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--drop_rate", default=0.0, type=float)
    parser.add_argument("--epochs", default=0, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--seed", default=123456, type=int)
    parser.add_argument("--tag", default="relabeled", type=str)
    parser.add_argument("--toy", action="store_false")### true
    parser.add_argument("--save_path", default="saliencymap_densenet", type=str)
    parser.add_argument("--scale", default=512, type=int)
    parser.add_argument("--horizontal_flip", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--train_weighted", action="store_true")
    parser.add_argument("--valid_weighted", action="store_true")
    parser.add_argument("--size", default=None, type=str)
    parser.add_argument("--kernel_x", default=64, type=int)
    parser.add_argument("--kernel_y", default=64, type=int)
    parser.add_argument("--stride_x", default=32, type=int)
    parser.add_argument("--stride_y", default=32, type=int)
    return parser
