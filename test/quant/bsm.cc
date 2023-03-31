#include <noa/quant/bsm.hh>

#include <cstdint>

using namespace torch::indexing;

int main(int argc, char* argv[]) {
    double STRIKE = 50;
    double T = 1;
    double RATE = 0.01;
    double SIGMA = 0.3;
    double S_min = 20;
    double S_max = 100;
    int64_t npoints_S = 500;
    int64_t npoints_t = 500;

    auto [V, S_array, t_array] = noa::quant::price_american_put_bs(
            STRIKE, T, RATE, SIGMA, S_min, S_max, npoints_S, npoints_t);

    auto [stop_line_V, stop_line_S] = noa::quant::find_early_exercise(
            V, S_array, t_array, STRIKE);

    return 0;
}
