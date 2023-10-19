# tracking_module.pyx

import numpy as np
cimport numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

cdef double INFTY_COST = 1e+5

cpdef min_cost_matching(
        distance_metric, double max_distance, list tracks, list detections,
        list track_indices=None,
        list detection_indices=None):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices

    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    nan_mask = np.isnan(cost_matrix)
    inf_mask = np.isinf(cost_matrix)

    cdef double default_value = 0
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

def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices, detection_indices):

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if not unmatched_detections:  # No detections left
            break

        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        if not track_indices_l:  # Nothing to match at this level
            continue

        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)
        matches += matches_l

    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections