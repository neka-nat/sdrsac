import numpy as np
import picos
from cvxopt import matrix, spmatrix
from . import linear_projection
from . import utils
from . import _kabsch


def to_spmatrix(sp):
    coo = sp.astype(np.double).tocoo()
    return spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=sp.shape)


def solve_sdp(w, k):
    nsquare = w.shape[0]
    n = int(np.sqrt(nsquare))
    ww = np.c_[w, np.zeros((nsquare, 1))]
    ww = np.r_[ww, np.zeros((1, nsquare + 1))]
    maps, mapeq, maps_b, mapeq_b = utils.generate_sdp_constraint_map(ww, n, k)

    sdp = picos.Problem()
    y = sdp.add_variable('y', (nsquare + 1, nsquare + 1), 'symmetric')
    sdp.add_constraint(y >> 0)
    sdp.add_constraint(y[nsquare, nsquare] == 1.0)
    sdp.add_constraint(y[nsquare, :] == y[:, nsquare].T)
    for i, m in enumerate(maps):
        sdp.add_constraint(to_spmatrix(m) | y <= maps_b.astype(np.double)[i, 0])
    sdp.add_constraint(to_spmatrix(mapeq) | y == matrix(mapeq_b))
    sdp.add_constraint(picos.trace(y) == k + 1)
    sdp.set_objective('max', matrix(ww) | y)
    sdp.set_option('tol', 0.1)
    sdp.set_option('verbose', 0)
    # print(sdp)
    sdp.solve()

    y = y.value
    t = np.array(y[-1, :-1])
    x1 = t.reshape((n, n))
    x = linear_projection.linear_projection(x1.ravel(order='F'))

    idx, = np.where(x.ravel(order='F')==1.0)
    score = x1.ravel(order='F')[idx]
    sidx = np.argsort(score)
    idx = np.delete(idx, sidx[(n - k):])
    x.ravel(order='F')[idx] = 0.0
    return x


def count_correspondence(m, tree, eps):
    dist, idx = tree.query(m.T)
    in_idx, = np.where(dist <= eps)
    return len(in_idx)


def get_correspondences(x):
    idxs = np.argmax(x, axis=1)
    y = x[np.arange(x.shape[0]), idxs]
    corr_m = np.where(y > 0)[0]
    corr_b = idxs[corr_m]
    return corr_m, corr_b


def sdp_reg(m, b, k, d_diff_thresh, pair_dist_thresh, n_pair_thres):
    w, n_accepted_pairs = utils.generate_weight_matrix(m, b,
                                                       d_diff_thresh,
                                                       pair_dist_thresh)
    if n_accepted_pairs <= n_pair_thres:
        return None, None, None, None

    x = solve_sdp(w, k)
    corr_m, corr_b = get_correspondences(x)
    sm = m[:, corr_m]
    sb = b[:, corr_b]
    rot, t = _kabsch.kabsch(sm, sb)
    return rot, t, corr_m, corr_b