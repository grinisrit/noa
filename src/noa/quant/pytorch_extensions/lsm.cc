#include "noa/quant/lsm.hh"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
            "price_american_put_lsm",
            &noa::quant::price_american_put_lsm,
            py::call_guard<py::gil_scoped_release>(),
            "Calculate the price of American put option using the Longstaff-Schwartz method"
    );
}
