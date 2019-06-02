#include "types.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <Eigen/Geometry>
#include <iostream>

using namespace sdrsac;
namespace py = pybind11;

PYBIND11_MODULE(_kabsch, m) {
    m.def("kabsch", [](const Matrix3X& model, const Matrix3X& target) -> std::pair<Matrix3, Vector3> {
        const auto res = Eigen::umeyama(model, target, false);
        return std::make_pair(res.block<3, 3>(0, 0), res.col(3).head<3>());
    });

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}