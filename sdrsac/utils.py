from . import _utils


def generate_weight_matrix(m, b, d_diff_thresh, pair_dist_thresh):
    return _utils.generate_weight_matrix(m, b, d_diff_thresh, pair_dist_thresh)


def generate_sdp_constraint_map(weight, n, k):
    return _utils.generate_sdp_constraint_map(weight, n, k)