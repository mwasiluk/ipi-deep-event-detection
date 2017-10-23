from __future__ import print_function

import errno
import os

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score


def sliding_window(iterable, left, right, padding=None, step=1):
    """Make a sliding window iterator with padding.

    Iterate over `iterable` with a step size `step` producing a tuple for each element:
        ( ... left items, item, right_items ... )
    such that item visits all elements of `iterable` `step`-steps aside, the length of
    the left_items and right_items is `left` and `right` respectively, and any missing
    elements at the start and the end of the iteration are padded with the `padding`
    item.

    For example:

    >>> list( sliding_window( range(5), 1, 2 ) )
    [(None, 0, 1, 2), (0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, None), (3, 4, None, None)]

    """
    from itertools import islice, repeat, chain
    from collections import deque

    n = left + right + 1

    iterator = chain(iterable, repeat(padding, right))

    elements = deque(repeat(padding, left), n)
    elements.extend(islice(iterator, right - step + 1))

    while True:
        for i in range(step):
            elements.append(next(iterator))
        yield tuple(elements)


def print_stats(y_true, y_pred, labels_index, binary):
    if not binary:
        y_true = map(lambda v: v.argmax(), y_true)
        y_pred = map(lambda v: v.argmax(), y_pred)



    conf_mat = confusion_matrix(y_true, y_pred)
    print(conf_mat)

    acc = accuracy_score(y_true, y_pred)
    print("acc: ", acc)

    p_r_f = precision_recall_fscore_support(y_true, y_pred)
    print_p_r_f(p_r_f, labels_index)
    return p_r_f, acc


def print_p_r_f(p_r_f, labels_index):
    print("")
    print("  \t", end='')

    for key, value in sorted(labels_index.iteritems(), key=lambda x: x[1]):
        print(key + "\t\t", end='')
    print("")
    print("P \t", end='')
    for p in p_r_f[0]:
        print(str(p) + "\t", end='')
    print("")
    print("R \t", end='')
    for p in p_r_f[1]:
        print(str(p) + "\t", end='')
    print("")
    print("F1\t", end='')
    for p in p_r_f[2]:
        print(str(p) + "\t", end='')

    print("")


def split(y_data, split_ratio, balanced=True):
    if balanced:
        return balanced_split(y_data, split_ratio)

    return simple_split(y_data, split_ratio)


def simple_split(y_data, split_ratio):

    y1_target_len = split_ratio * len(y_data)
    y1 = []
    y2 = []
    for i in range(len(y_data)):
        if i < y1_target_len:
            y1.append(i)
        else:
            y2.append(i)
    return y1, y2


def balanced_split(y_data, split_ratio):
    y_true = map(lambda v: v.argmax(), y_data)

    class_num = len(np.unique(y_true))
    print(class_num)
    class_num_arr = np.zeros(class_num)
    for y in y_true:
        class_num_arr[y] += 1

    y1_class_num_arr = class_num_arr * split_ratio
    y1_class_num_arr_curr = np.zeros(class_num)
    y1 = []
    y2 = []

    for i, y_d in enumerate(y_data):
        y = y_d.argmax()

        if y1_class_num_arr_curr[y] < y1_class_num_arr[y]:
            y1.append(i)
            y1_class_num_arr_curr[y] +=1
        else:
            y2.append(i)

    print(y1_class_num_arr, y1_class_num_arr_curr)
    return y1, y2


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def merge_several_folds_results(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a