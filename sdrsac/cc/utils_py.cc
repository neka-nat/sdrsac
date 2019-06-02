#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include "utils.h"

namespace py = pybind11;
using namespace sdrsac;

PYBIND11_MODULE(_utils, m) {
    m.def("generate_weight_matrix", &generateWeightMatrix);
    m.def("generate_sdp_constraint_map", &generateSDPConstraintMap);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}