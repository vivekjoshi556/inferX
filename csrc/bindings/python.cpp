#include "loader.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_inferX, m) {
    m.doc() = "InferX is an Inference module.";

    m.def("hello", &hello, "Hello from inferX Module");
}
