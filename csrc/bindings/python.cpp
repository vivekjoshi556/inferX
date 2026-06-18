#include "loader.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_inferX, m) {
    m.doc() = "InferX is an Inference module.";

    m.def("load_model", &inferX::load_model, py::arg("model_id"),
          py::arg("revision") = "main", py::arg("auth_token") = "",
          "Loading a Model using it's model identifier");
}
