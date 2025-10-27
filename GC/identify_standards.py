"""
Chain length of standards (alkanes) is expected to be proportional to retention time.
We can use this and the fact that C16 is the first compound to identify them by finding points closest to the regression line.
We can further easily define the expected slope of the regression line.
"""
import random
from typing import Iterable, Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment, minimize, brute, minimize_scalar
from sklearn.linear_model import HuberRegressor

# expected abundances of standards, should be adjusted over time
# map label to weight
PATTERN_WEIGHTS = {16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: .5,
                   28: 1, 29: .5, 30: .8, 32: .5, 34: .5}
PEAK_POSITIONS = [
    15.56, 17.71, 19.97, 22.26, 24.54, 26.78, 28.96, 31.07,
    33.12, 35.1, 37.01, 38.833, 40.64, 41.93, 44.03, 47.22, 50.36
]
LABELS = [
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34
]
NAMES = [f'C{l}' for l in LABELS]
# fit label(rt) = a * rt^2 + b * rt + c
A0, B0, C0 = np.polyfit(PEAK_POSITIONS, LABELS, deg=2)

# add squalane
SQUALANE = dict(rt=38.065, name='Squalane', label=26.5)
PEAK_POSITIONS.append(SQUALANE['rt'])
LABELS.append(SQUALANE['label'])
NAMES.append(SQUALANE['name'])
PATTERN_WEIGHTS[SQUALANE['label']] = .7


def convert_rt_to_expected_label(x):
    """Get expected labels by converting retention times"""
    return A0 * x ** 2 + B0 * x + C0


def calc_r2(y_meas, y_pred):
    y_meas = np.asarray(y_meas)
    y_pred = np.asarray(y_pred)
    meas_mean = np.mean(y_meas)

    SS_res = np.sum((y_meas - y_pred) ** 2)
    SS_tot = np.sum((y_meas - meas_mean))
    return 1 - SS_res / SS_tot


def remove_biggest_outlier(
        peak_positions: Iterable,
        expected_slope: float = None,
        chain_lengths=None,
        expected_peak_positions=None
):
    """iteratively remove peaks to maximize R^2"""
    # make sure it is sorted and we create a copy
    peak_positions = sorted(peak_positions)
    # match chain lengths to retention times
    if chain_lengths is None:
        # assume first peak is 16 and last one is 32
        chain_lengths = initialize_peak_labels(peak_positions, TARGET_CHAIN_LENGTHS)
        print(dict(zip(peak_positions, chain_lengths)))

    if expected_slope is not None:
        # fix slope and only fit offset
        ...

    # test all removals
    leave_out_idx_to_r2 = {}
    for idx in range(len(peak_positions)):
        new_peak_positions = peak_positions.copy()
        new_chain_lengths = chain_lengths.copy()
        removed_chain_length = new_chain_lengths.pop(idx)
        removed_val = new_peak_positions.pop(idx)
        # chain length = b * rt + a
        (b, a), *out = np.polyfit(x=new_peak_positions, y=new_chain_lengths, deg=1, full=True)
        # calc R^2 val
        pred_chain_lengths = [b * pp + a for pp in new_peak_positions]
        # R2 = calc_r2(new_peak_positions, pred_chain_lengths)
        leave_out_idx_to_r2[idx] = out[0], removed_val, removed_chain_length

    return leave_out_idx_to_r2


def gaussian(x, mu, sigma, normalize_area: bool = False):
    A = 1 / np.sqrt(2 * np.pi * sigma ** 2) if normalize_area else 1
    return A * np.exp(-1 / 2 * (x - mu) ** 2 / sigma ** 2)


def get_xs_for_peaks(peak_positions: Iterable, delta_rt: float) -> np.ndarray:
    min_ = min(peak_positions)
    max_ = max(peak_positions)
    span = max_ - min_
    rts = np.arange(  # give some slack at the boundaries
        min_ - span * .1,
        max_ + span * .1,
        delta_rt
    )
    return rts


def full_spec(xs, pos, hs, sigma):
    full = np.zeros_like(xs)
    for p, h in zip(pos, hs):
        full += gaussian(xs, p, sigma) * h
    return full


def full_spec_similarity(a, b):
    return np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))


def by_pattern_matching(peak_positions, peak_heights, sigma=.2, delta_rt=.01, pattern_std=None, plts=False, ax=None):
    """shift and scale pattern_weights to maximize score"""
    def transform(params):
        a, b, c = params

        def inner(xs):
            pred_pos = [a * l ** 2 + b * l + c for l in xs]
            return pred_pos

        return inner

    def score(params) -> float:
        pred_pos = transform(params)(peak_positions)
        measured_transformed = full_spec(rts, pred_pos, peak_heights, sigma)

        ab = np.dot(measured_transformed, pattern_full)
        abs_a = np.dot(measured_transformed, measured_transformed)
        return -ab / (np.sqrt(abs_a) * sqrt_abs_b)

    if pattern_std is None:
        pattern_std = PATTERN_WEIGHTS

    # ensure proper scaling
    peak_heights = [p / max(peak_heights) for p in peak_heights]
    pred_heights = pattern_std.values()

    rts = get_xs_for_peaks(peak_positions, delta_rt)
    pattern_full = full_spec(rts, PEAK_POSITIONS, pattern_std.values(), sigma)
    sqrt_abs_b = np.sqrt(np.dot(pattern_full, pattern_full))

    x0 = (0, 1, 0)
    print(f'initial score: {score(x0)}')
    bounds = (
        (-.001, .001),
        (1 - .2, 1 + .2),
        (-5, 5)
    )
    out = minimize(
        score, x0=x0, method='Nelder-Mead',  # 'Nelder-Mead', "TNC", 'L-BFGS-B'
        bounds=bounds
    )
    params = out['x']
    print(f'final score: {score(params)}')

    if not plts:
        return params, transform(params)

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(rts, full_spec(rts, peak_positions, peak_heights, sigma), label='measured', linestyle='--')

    pred_pos = transform(x0)(PEAK_POSITIONS)
    ax.plot(rts, full_spec(rts, pred_pos, pred_heights, sigma), label='pattern')

    pred_pos = transform(params)(peak_positions)
    ax.plot(rts, full_spec(rts, pred_pos, peak_heights, sigma), label='transformed measured')
    ax.legend()
    return params, transform(params), ax


def predict_labels_from_fit(peak_positions, trafo):
    """Need to match peaks to labels"""
    # transform peak positions to match pattern and then predict labels based on PATTERN-RT - LABEL fit
    transformed_peaks = trafo(peak_positions)
    predicted_labels = [A0 * l ** 2 + B0 * l + C0 for l in transformed_peaks]
    return predicted_labels


def assign_labels_from_fit(peak_labels, max_tol_assignment=.2):
    assigned_peaks = {}
    # target whole numbers and squalane (only non-integer)
    peak_labels = np.array(peak_labels)
    peak_still_available = np.ones_like(peak_labels, dtype=bool)

    dists_sql = [abs(p - SQUALANE['label']) for p in peak_labels]
    sql_idx = np.argmin(dists_sql)
    if dists_sql[sql_idx] < max_tol_assignment:
        assigned_peaks[sql_idx] = SQUALANE['name']
        peak_still_available[sql_idx] = False
    # for all others just round and compare to tolerance
    possible_labels = np.unique(np.around(peak_labels[peak_still_available]).astype(int))

    for l in possible_labels:
        diffs = np.abs(peak_labels - l)
        diffs[~peak_still_available] = np.inf
        idx = np.argmin(diffs)
        if diffs[idx] < max_tol_assignment:
            assigned_peaks[idx] = f'C{l}'
            peak_still_available[idx] = False
    unassigned_peaks = dict(zip(np.arange(len(peak_labels))[peak_still_available],
                                np.around(peak_labels)[peak_still_available]))
    return assigned_peaks, unassigned_peaks


def find_scaling_of_pattern(
        peak_positions, peak_heights, label_first_peak=16, width_tri=None
):
    """scale pattern_weights to maximize score"""

    def score(scale):
        pred_pos = [l * scale for l in PATTERN_WEIGHTS]
        # get height
        s = 0
        for pos in pred_pos:
            dists = np.abs(peak_positions - pos)
            peak_idx = np.argmin(dists)
            dist = dists[peak_idx]
            if dist > width_tri:
                continue
            dist_f = dist / width_tri
            height = peak_heights[peak_idx]
            s += height * dist_f
        return max_score - s

    max_score = sum(PATTERN_WEIGHTS.values())

    # ensure proper scaling
    peak_heights = [p / max(peak_heights) for p in peak_heights]

    peak_positions = np.array(peak_positions)
    shift = peak_positions.min()
    peak_positions -= shift

    if width_tri is None:  # this controls how steeply the triangular kernels vanish
        span = peak_positions.max()
        width_tri = span / 100

    target_labels = np.array(TARGET_CHAIN_LENGTHS)
    target_labels -= label_first_peak

    # obtain rough scaling factor and offset from linear spacing assumption
    estimated_labels = initialize_peak_labels(peak_positions, target_labels)
    weights = [PATTERN_WEIGHTS[round(el) + label_first_peak] for el in estimated_labels]

    scaling_factor0, shift0 = np.polyfit(estimated_labels, peak_positions, w=weights, deg=1)

    print(estimated_labels)

    return -shift, scaling_factor0

    assert abs(shift0) < 1e-6

    # return -shift, scaling_factor0

    bounds = (scaling_factor0 / 1.5, scaling_factor0 * 1.5)

    params = minimize_scalar(
        score, bounds
    )
    return -shift, params.x


def plot_matched_pattern(peak_positions, peak_heights, a, b, c, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    peak_heights = np.array(peak_heights) / max(peak_heights)

    y = np.array(list(PATTERN_WEIGHTS.keys()))
    x = (-b + np.sqrt(b ** 2 - 4 * a * (c - y))) / (2 * a)

    ax.stem(peak_positions, peak_heights, linefmt='C0--', markerfmt='')
    ax.stem(x, -np.array(list(PATTERN_WEIGHTS.values())), linefmt='C1--', markerfmt='')

    return ax


def assignment_problem(peak_positions, labels, metric: Callable = None):
    """Only use this on it's own if you are sure there are no missing or extra peaks"""
    if metric is None:
        metric = np.abs

    peak_positions = sorted(peak_positions)
    labels = sorted(labels)

    # reshape inputs for sklearn
    labels_arr = np.array(labels).reshape(1, -1)  # row vector
    peaks_arr = np.array(peak_positions).reshape(-1, 1)  # column vector

    init_labels = initialize_peak_labels(peak_positions, labels)
    slope, intercept = np.polyfit(x=init_labels, y=peak_positions, deg=1, full=False)

    # predicted positions for each label
    pred_pos = slope * labels_arr + intercept

    # cost matrix
    C = metric(peaks_arr - pred_pos)

    # solve assignment problem
    row_ind, col_ind = linear_sum_assignment(C)

    X = np.zeros((len(peak_positions), len(labels)), dtype=bool)
    X[row_ind, col_ind] = True

    fit_error = (X * C).sum()

    assignments = {peak_positions[r]: labels[c] for r, c in zip(row_ind, col_ind)}
    # sort
    assignments = {p: assignments[p] for p in sorted(assignments)}

    return assignments, fit_error


def test_1to1(noise_amplitude: float = 0):
    """should always return correct result up to a noise amplitude of 0.64
    np.diff(right_peak_positions_wo_sq).min() / 2
    """
    right_peak_positions_wo_sq = [
        15.56, 17.71, 19.97, 22.26, 24.54, 26.78, 28.96, 31.07,
        33.12, 35.1, 37.01, 38.833, 40.64, 41.93, 44.03, 47.22, 50.36
    ]

    labels_wo_sq = [
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34
    ]

    # add noise to peak positions
    right_peak_positions_wo_sq = [
        pp + (np.random.random() * 2 - 1) * noise_amplitude
        for pp in right_peak_positions_wo_sq
    ]

    shuffled = right_peak_positions_wo_sq.copy()
    np.random.shuffle(shuffled)

    assignments, error = assignment_problem(shuffled, labels=labels_wo_sq)
    # assert assignments == dict(zip(right_peak_positions_wo_sq, labels_wo_sq))
    return assignments, error


def assignment_with_missing(max_missing=2):
    ...


if __name__ == '__main__':
    # values from the sample
    peak_positions = [
        15.56, 17.71, 19.97, 20.32, 22.26, 24.54, 26.78, 26.93, 28.96, 31.07,
        33.12, 35.1, 37.01, 38.07, 40.64, 41.93, 44.03, 47.22, 50.36, 59.48
    ]
    peak_heights = [42.94, 41.15, 41.31, 23.43, 41.65, 42.55, 42.51, 43.02,
                    42.33, 40.17, 40.54, 39.55, 37.44, 28.67,
                    31.01, 33.58, 27.19, 22.09, 17.54, 4.36]

    # add random offsets
    # peak_positions = [p + .5 - p * .01 for p in peak_positions]

    # remove one
    # removed = peak_positions.pop(4)
    # peak_heights.pop(4)

    # right_peak_positions = [
    #     15.56, 17.71, 19.97, 22.26, 24.54, 26.78, 28.96, 31.07,
    #     33.12, 35.1, 37.01, 38.07, 38.833, 40.64, 41.93, 44.03, 47.22, 50.36
    # ]
    sigma = .1
    delta_rt = .01

    params, trafo, ax = by_pattern_matching(
        peak_positions=peak_positions, peak_heights=peak_heights, sigma=sigma, delta_rt=delta_rt, plts=True
    )
    print(params)

    # ax.vlines(removed, *ax.get_ylim(), linestyles='dashed', color='r')
    plt.show()

    predicted_labels = predict_labels_from_fit(peak_positions, trafo)
    # assign labels
    assigned_labels, additional_labels = assign_labels_from_fit(predicted_labels)

    #
    # plt.figure()
    # plt.plot(rts, full_meas, label='measured')
    # plt.plot(rts, full_pattern, label='expected')
    # plt.legend()
    # plt.show()
    #
    # plt.figure()
    # plt.plot(rts, full_meas, label='measured')
    # plt.plot(rts, full_transformed, label='fitted pattern')
    # plt.legend()
    # plt.show()
    #
    # print(full_spec_similarity(full_meas, full_pattern))
    # print(full_spec_similarity(full_meas, full_transformed))
