#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np


def average(x):
    if len(x) == 0:
        return 0
    else:
        return sum(x) / len(x)


def medium(x):
    if len(x) == 0:
        return 0
    ranked = list(sorted(x))
    if len(x) % 2 == 1:
        return ranked[len(x) // 2]
    else:
        return (ranked[len(x) // 2] + ranked[len(x) // 2 + 1]) / 2

def get_acc_dict(scores, target):
    n = len(scores)
    ranklist = list(sorted(range(n), key=lambda x: scores[x], reverse=True))
    output = ranklist[0]
    acc_dict = dict(
        top1=1 if ranklist[0] == target else 0,
        top3=1 if target in ranklist[:3] else 0,
        top5=1 if target in ranklist[:5] else 0,
        medium_rank=list(filter(lambda k: ranklist[k] == target, range(n)))[0] + 1,
        average_rank=list(filter(lambda k: ranklist[k] == target, range(n)))[0] + 1,
    )
    return output, acc_dict

def make_dict_default(d, k, v):
    if k not in d:
        d[k] = v


def dump_confusing_matrix(matrix, name, labels, logger=print, plot=True, folder=None, filename=None):
    if matrix.sum() < 1e-9:
        print("%s has no element." % name)
        return
    logger(name)
    num_format = '%3.0f' if matrix.dtype == np.int else '%4.3f'
    for i in range(len(labels)):
        for j in range(len(labels)):
            logger(num_format % float(matrix[i, j]), end=' ')
        logger('')
    if plot:
        plot_confusing_matrix(matrix, folder, filename, labels, labels, title=name)


def plot_confusing_matrix(x, folder, filename, x_labels=None, y_labels=None, title=None):
    if title is None:
        title = filename
    n, m = x.shape[0], x.shape[1]
    if x_labels is None:
        x_labels = [i for i in range(n)]
    else:
        assert len(x_labels) == n
    if y_labels is None:
        y_labels = [i for i in range(m)]
    else:
        assert len(y_labels) == n

    os.makedirs(folder, exist_ok=True)
    fullname = os.path.join(folder, filename)

    fig, ax = plt.subplots()
    im = ax.imshow(x)

    # We want to show all ticks...
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(m))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(n):
        for j in range(m):
            text = ax.text(j, i, "%.2f" % x[i, j], ha="center", va="center", color="w")
    plt.xlim((-0.5, m - 0.5))
    plt.ylim((-0.5, n - 0.5))
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(fullname)
    plt.close()
