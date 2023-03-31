#include "noa/quant/bsm.hh"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
            "price_american_put_bs",
            &noa::quant::price_american_put_bs,
            py::call_guard<py::gil_scoped_release>(),
            "Calculate the value of American put option under the Black-Scholes"
            "model, using the Brennan-Schwartz algorithm"
    );
    m.def(
            "find_early_exercise",
            &noa::quant::find_early_exercise,
            py::call_guard<py::gil_scoped_release>(),
            "Calculates the early exercise curve for American put option under"
            "the Black-Scholes model"
    );
}
