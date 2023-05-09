#include "noa/quant/heston.hh"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "noncentral_chisquare",
        &noa::quant::noncentral_chisquare,
        py::call_guard<py::gil_scoped_release>(),
        "Generates samples from a noncentral chi-square distribution using Andersen's approximate Quadratic Exponential scheme"
    );
    m.def(
        "generate_cir",
        &noa::quant::generate_cir,
        py::call_guard<py::gil_scoped_release>(),
        "Generates paths of Cox-Ingersoll-Ross (CIR) process using Andersen's Quadratic Exponential scheme"
    );
    m.def(
        "generate_heston",
        &noa::quant::generate_heston,
        py::call_guard<py::gil_scoped_release>(),
        "Generates time series following Heston model, using Andersen's Quadratic Exponential scheme"
    );
}
