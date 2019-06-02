#ifndef __sdrsac_utils_h__
#define __sdrsac_utils_h__
#include "types.h"
#include <tuple>
#include <vector>

namespace sdrsac
{

typedef std::pair<Matrix, Integer> WeightResult;
typedef std::vector<SparseMatrix> SpMatArray;
typedef std::tuple<SpMatArray, SparseMatrix, Matrix, Integer> ConstraintMap;

WeightResult generateWeightMatrix(const Matrix3X& m, const Matrix3X& b,
                                  const Float d_diff_thresh,
                                  const Float pair_dist_thresh);

ConstraintMap generateSDPConstraintMap(const Matrix& weight, Integer n, Integer k);

} // namespace sdrsac

#endif