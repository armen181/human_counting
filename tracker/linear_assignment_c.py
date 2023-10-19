from __future__ import absolute_import

import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

from . import kalman_filter
import sys

sys.path.append("/Users/armen/IdeaProjects/human_counting/tracker/build/lib.macosx-10.9-x86_64-cpython-38/tracker")
import tracking_module

INFTY_COST = 1e+5


def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    return tracking_module.min_cost_matching(
        distance_metric, max_distance, tracks, detections,
        track_indices, detection_indices)


def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
    return tracking_module.matching_cascade(distance_metric, max_distance, cascade_depth, tracks, detections,
                                            track_indices, detection_indices)


def gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix
