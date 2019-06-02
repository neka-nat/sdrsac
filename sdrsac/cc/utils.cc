#include "utils.h"

using namespace sdrsac;

WeightResult sdrsac::generateWeightMatrix(const Matrix3X& m, const Matrix3X& b,
                                          const Float d_diff_thresh,
                                          const Float pair_dist_thresh) {
    assert(m.cols() == b.cols());
    const Integer n = m.cols();
    Matrix w = -1.0e4 * Matrix::Ones(n * n, n * n);
    Integer n_accepted_pairs = 0;
    for (Integer p = 0; p < n; ++p) {
        for (Integer q = 0; q < n; ++q) {
            for (Integer s = 0; s < n; ++s) {
                for (Integer t = 0; t < n; ++t) {
                    if (p == s || q == t) continue;
                    auto i = p + (q - 1) * n;
                    auto j = s + (t - 1) * n;
                    auto ps_dist = (m.col(p) - m.col(s)).norm();
                    auto qt_dist = (b.col(q) - b.col(t)).norm();
                    auto d_diff = std::abs(ps_dist - qt_dist);
                    if (d_diff <= d_diff_thresh &&
                        ps_dist >= pair_dist_thresh &&
                        qt_dist >= pair_dist_thresh) {
                            w(i, j) = std::exp(-d_diff);
                            ++n_accepted_pairs;
                    }
                }
            }
        }
    }
    return {w, n_accepted_pairs};
}

ConstraintMap sdrsac::generateSDPConstraintMap(const Matrix& weight, Integer n, Integer k) {
    SpMatArray maps(2 * n);
    Integer nn_1 = n * n + 1;
    for (Integer i = 0; i < n; ++i) {
        Matrix a = Matrix::Zero(n, n);
        a.row(i).array() = 1.0;
        const Vector av = (Vector(nn_1) << Eigen::Map<Vector>(a.data(), a.size()), 0.0).finished();
        SparseMatrix m = SparseMatrix(nn_1, nn_1);
        for (Integer j = 0; j < nn_1; ++j) {
            m.insert(nn_1 - 1, j) = av[j];
        }
        maps[i] = m;
    }
    for (Integer i = 0; i < n; ++i) {
        Matrix a = Matrix::Zero(n, n);
        a.col(i).array() = 1.0;
        const Vector av = (Vector(nn_1) << Eigen::Map<Vector>(a.data(), a.size()), 0.0).finished();
        SparseMatrix m = SparseMatrix(nn_1, nn_1);
        for (Integer j = 0; j < nn_1; ++j) {
            m.insert(nn_1 - 1, j) = av[j];
        }
        maps[n + i] = m;
    }

    Matrix maps_b = Matrix::Ones(2 * n, 1);
    SparseMatrix m = SparseMatrix(nn_1, nn_1);
    for (Integer j = 0; j < nn_1; ++j) {
        m.insert(nn_1 - 1, j) = 1.0;
    }

    return {maps, m, maps_b, k};
}