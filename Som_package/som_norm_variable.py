# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:30:55 2024

@author: sdd380
"""

import numpy as np
from som_set import som_set

def som_norm_variable(x, method, operation):
    """
    Normalize or denormalize a scalar variable.

    Parameters:
    - x: numpy array, a set of values of a scalar variable for which the (de)normalization is performed.
    - method: dict or str, identifier for a normalization method.
              If str: 'var', 'range', 'log', 'logistic', 'histD', or 'histC'.
              If dict: normalization struct with keys 'type', 'method', 'params', 'status'.
    - operation: str, the operation to be performed: 'init', 'do', or 'undo'.

    Returns:
    - x_new: numpy array, appropriately processed values.
    - sNorm: dict, updated normalization struct.
    """
    # Convert method to dict if it's a string
    if isinstance(method, str):
        method = {'type': 'som_norm', 'method': method, 'params': [], 'status': 'uninit'}
    
    # Initialize sNorm
    sNorm = method.copy()

    # Initialize normalization parameters if needed
    if operation == 'init' or (operation == 'do' and sNorm['status'] == 'uninit'):
        if sNorm['method'] == 'var':
            sNorm['params'] = norm_variance_init(x)
        elif sNorm['method'] == 'range':
            sNorm['params'] = norm_scale01_init(x)
        elif sNorm['method'] == 'log':
            sNorm['params'] = norm_log_init(x)
        elif sNorm['method'] == 'logistic':
            sNorm['params'] = norm_logistic_init(x)
        elif sNorm['method'] == 'histD':
            sNorm['params'] = norm_histeqD_init(x)
        elif sNorm['method'] == 'histC':
            sNorm['params'] = norm_histeqC_init(x)
        else:
            raise ValueError("Unrecognized method: {}".format(sNorm['method']))
        sNorm['status'] = 'undone'

    # Apply or undo normalization
    if operation == 'do':
        if sNorm['method'] == 'var':
            x_new = norm_scale_do(x, sNorm['params'])
        elif sNorm['method'] == 'range':
            x_new = norm_scale_do(x, sNorm['params'])
        elif sNorm['method'] == 'log':
            x_new = norm_log_do(x, sNorm['params'])
        elif sNorm['method'] == 'logistic':
            x_new = norm_logistic_do(x, sNorm['params'])
        elif sNorm['method'] == 'histD':
            x_new = norm_histeqD_do(x, sNorm['params'])
        elif sNorm['method'] == 'histC':
            x_new = norm_histeqC_do(x, sNorm['params'])
        else:
            raise ValueError("Unrecognized method: {}".format(sNorm['method']))
        sNorm['status'] = 'done'
    elif operation == 'undo':
        if sNorm['status'] == 'uninit':
            raise ValueError("Could not undo: uninitialized normalization struct.")
        if sNorm['method'] == 'var':
            x_new = norm_scale_undo(x, sNorm['params'])
        elif sNorm['method'] == 'range':
            x_new = norm_scale_undo(x, sNorm['params'])
        elif sNorm['method'] == 'log':
            x_new = norm_log_undo(x, sNorm['params'])
        elif sNorm['method'] == 'logistic':
            x_new = norm_logistic_undo(x, sNorm['params'])
        elif sNorm['method'] == 'histD':
            x_new = norm_histeqD_undo(x, sNorm['params'])
        elif sNorm['method'] == 'histC':
            x_new = norm_histeqC_undo(x, sNorm['params'])
        else:
            raise ValueError("Unrecognized method: {}".format(sNorm['method']))
        sNorm['status'] = 'undone'
    else:
        raise ValueError("Unrecognized operation: {}".format(operation))

    return x_new, sNorm


def norm_variance_init(x):
    inds = np.where(~np.isnan(x) & np.isfinite(x))[0]
    mean_x = np.mean(x[inds])
    std_x = np.std(x[inds])
    if std_x == 0:
        std_x = 1
    return [mean_x, std_x]


def norm_scale01_init(x):
    inds = np.where(~np.isnan(x) & np.isfinite(x))[0]
    min_x = np.min(x[inds])
    max_x = np.max(x[inds])
    if min_x == max_x:
        return [min_x, 1]
    else:
        return [min_x, max_x - min_x]


def norm_log_init(x):
    inds = np.where(~np.isnan(x) & np.isfinite(x))[0]
    min_x = np.min(x[inds])
    return min_x


def norm_logistic_init(x):
    inds = np.where(~np.isnan(x) & np.isfinite(x))[0]
    mean_x = np.mean(x[inds])
    std_x = np.std(x[inds])
    if std_x == 0:
        std_x = 1
    return [mean_x, std_x]


def norm_histeqD_init(x):
    inds = np.where(~np.isnan(x) & ~np.isinf(x))[0]
    unique_values = np.unique(x[inds])
    return unique_values


def norm_histeqC_init(x):
    inds = np.where(~np.isnan(x) & ~np.isinf(x))[0]
    unique_values = np.unique(x[inds])
    lims = int(np.ceil(np.sqrt(len(unique_values))))
    if lims == 1:
        return [unique_values[0], unique_values[0] + 1]
    elif lims == 2:
        return [unique_values[0], unique_values[-1]]
    else:
        p = np.zeros(lims)
        p[0] = unique_values[0]
        p[-1] = unique_values[-1]
        binsize = np.zeros(lims - 1)
        b = 0
        avebinsize = len(inds) / (lims - 1)
        for i in range(len(unique_values) - 1):
            binsize[b] += np.sum(x == unique_values[i])
            if binsize[b] >= avebinsize:
                b += 1
                p[b] = (unique_values[i] + unique_values[i + 1]) / 2
            if b == (lims - 1):
                binsize[b] = len(inds) - np.sum(binsize)
                break
            else:
                avebinsize = (len(inds) - np.sum(binsize)) / (lims - 1 - b)
        return p


def norm_scale_do(x, p):
    return (x - p[0]) / p[1]


def norm_log_do(x, p):
    return np.log(x - p + 1)


def norm_logistic_do(x, p):
    x_scaled = (x - p[0]) / p[1]
    return 1 / (1 + np.exp(-x_scaled))


def norm_histeqD_do(x, p):
    x_new = np.zeros_like(x)
    inds = np.where(~np.isnan(x) & ~np.isinf(x))[0]
    for i in inds:
        ind = np.argmin(np.abs(x[i] - p))
        if x[i] > p[ind] and ind < len(p) - 1:
            x_new[i] = ind + 1
        else:
            x_new[i] = ind
    x_new /= (len(p) - 1)
    return x_new


def norm_histeqC_do(x, p):
    x_new = np.copy(x)
    lims = len(p)
    r = p[1] - p[0]
    inds = np.where((x <= p[0]) & np.isfinite(x))[0]
    if len(inds) > 0:
        x_new[inds] = 0 - (p[0] - x[inds]) / r

    r = p[-1] - p[-2]
    inds = np.where((x > p[-1]) & np.isfinite(x))[0]
    if len(inds) > 0:
        x_new[inds] = lims - 1 + (x[inds] - p[-1]) / r

    for i in range(1, lims - 1):
        r0 = p[i]
        r1 = p[i + 1]
        r = r1 - r0
        inds = np.where((x > r0) & (x <= r1) & np.isfinite(x))[0]
        if len(inds) > 0:
            x_new[inds] = i - 1 + (x[inds] - r0) / r

    x_new /= (lims - 1)
    return x_new


def norm_scale_undo(x, p):
    return x * p[1] + p[0]


def norm_log_undo(x, p):
    return np.exp(x) - 1 + p


def norm_logistic_undo(x, p):
    x = np.log(x / (1 - x))
    return x * p[1] + p[0]


def norm_histeqD_undo(x, p):
    x_new = np.round(x * (len(p) - 1) + 1).astype(int)
    inds = np.where(~np.isnan(x_new) & ~np.isinf(x_new))[0]
    x_new[inds] = p[x_new[inds] - 1]
    return x_new


def norm_histeqC_undo(x, p):
    x_new = x * (len(p) - 1)

    r = p[1] - p[0]
    inds = np.where((x_new <= 0) & np.isfinite(x_new))[0]
    if len(inds) > 0:
        x_new[inds] = x_new[inds] * r + p[0]

    r = p[-1] - p[-2]
    inds = np.where((x_new >= len(p) - 1) & np.isfinite(x_new))[0]
    if len(inds) > 0:
        x_new[inds] = (x_new[inds] - (len(p) - 1)) * r + p[-1]

    for i in range(1, len(p) - 1):
        r0 = p[i]
        r1 = p[i + 1]
        r = r1 - r0
        inds = np.where((x_new > i - 1) & (x_new <= i) & np.isfinite(x_new))[0]
        if len(inds) > 0:
            x_new[inds] = (x_new[inds] - (i - 1)) * r + r0

    return x_new