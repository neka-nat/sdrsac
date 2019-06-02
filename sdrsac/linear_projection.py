import numpy as np
from cvxopt import matrix, solvers


def linear_projection(xp):
    f = xp.copy()
    f[f < 0.0] = 0.0
    f = -f
    nn = xp.shape[0]
    n = int(np.round(np.sqrt(nn)))

    aa = np.zeros((4 * n, nn))
    bb = np.tile(np.array([1.0, -1.0]), 2 * n)
    for i in range(n):
        ta = np.zeros((n, n))
        ta[i, :] = 1.0
        tar = ta.ravel(order='F')
        aa[4 * i, :] = tar
        aa[4 * i + 1, :] = -tar

        ta = np.zeros((n, n))
        ta[:, i] = 1.0
        tar = ta.ravel(order='F')
        aa[4 * i + 2, :] = tar
        aa[4 * i + 3, :] = -tar

    aa = np.r_[aa, -np.identity(nn)]
    bb = np.r_[bb, np.zeros(nn)]
    sol = solvers.lp(matrix(f), matrix(aa), matrix(bb))
    return np.array(sol['x']).reshape((n, n))