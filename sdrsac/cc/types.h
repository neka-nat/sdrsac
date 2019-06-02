#ifndef __sdrsac_types_h__
#define __sdrsac_types_h__

#include <Eigen/Core>
#include <Eigen/Sparse>
//#define USE_DOUBLE

namespace sdrsac {
#ifdef USE_DOUBLE
typedef double Float;
typedef int Integer;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
typedef Eigen::VectorXi VectorXi;
typedef Eigen::Matrix3d Matrix3;
typedef Eigen::Vector3d Vector3;
#else
typedef float Float;
typedef int Integer;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::VectorXf Vector;
typedef Eigen::VectorXi VectorXi;
typedef Eigen::Matrix3f Matrix3;
typedef Eigen::Vector3f Vector3;
#endif
typedef Eigen::Matrix<Float, 3, Eigen::Dynamic> Matrix3X;
typedef Eigen::Matrix<Float, Eigen::Dynamic, 3> MatrixX3;
typedef Eigen::SparseMatrix<Float> SparseMatrix;
}  // namespace sdrsac

#endif