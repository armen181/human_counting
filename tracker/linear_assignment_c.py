from __future__ import absolute_import

import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

from . import kalman_filter

INFTY_COST = 1e+5

def min_cost_matching(max_distance, tracks, detections, samples, kf, track_indices=None,
                      detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = gated_metric(tracks, detections, track_indices, detection_indices, samples, kf)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    return min_cost_part_2(cost_matrix, detection_indices, max_distance, track_indices)


def iou_cost_matching(max_distance, tracks, detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = iou_cost(tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    return min_cost_part_2(cost_matrix, detection_indices, max_distance, track_indices)


def min_cost_part_2(cost_matrix, detection_indices, max_distance, track_indices):
    nan_mask = np.isnan(cost_matrix)
    inf_mask = np.isinf(cost_matrix)
    default_value = 0
    cost_matrix[nan_mask] = default_value
    cost_matrix[inf_mask] = default_value
    row_indices, col_indices = linear_assignment(cost_matrix)
    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(max_distance, cascade_depth, tracks, detections, samples, kf, track_indices=None,
                     detection_indices=None):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break

        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        matches_l, _, unmatched_detections = min_cost_matching(max_distance,
                                                               tracks,
                                                               detections,
                                                               samples,
                                                               kf,
                                                               track_indices_l,
                                                               unmatched_detections)
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


def gated_metric(tracks, detections, track_indices, detection_indices, samples, kf):
    features = np.array([detections[i].feature for i in detection_indices])
    targets = np.array([tracks[i].track_id for i in track_indices])
    cost_matrix = distance(samples, features, targets)
    return gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices)


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


def match(tracks, detections, max_dist, samples, kf, max_age, max_iou_distance):
    confirmed_tracks = [i for i, t in enumerate(tracks) if t.is_confirmed()]
    unconfirmed_tracks = [i for i, t in enumerate(tracks) if not t.is_confirmed()]
    matches_a, unmatched_tracks_a, unmatched_detections = matching_cascade(max_dist, max_age, tracks, detections,
                                                                           samples, kf, confirmed_tracks,)

    iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if tracks[k].time_since_update == 1]
    unmatched_tracks_a = [k for k in unmatched_tracks_a if tracks[k].time_since_update != 1]
    matches_b, unmatched_tracks_b, unmatched_detections = iou_cost_matching(max_iou_distance, tracks, detections,
                                                                            iou_track_candidates, unmatched_detections)
    matches = matches_a + matches_b
    unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
    return matches, unmatched_tracks, unmatched_detections


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix


def iou(bbox, candidates):
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
    np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
    np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def distance(samples, features, targets):
    cost_matrix = np.zeros((len(targets), len(features)))
    for i, target in enumerate(targets):
        distances = 1. - np.dot(samples[target], features.T)
        cost_matrix[i, :] = distances.min(axis=0)
    return cost_matrix
