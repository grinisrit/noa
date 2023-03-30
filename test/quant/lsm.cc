#include <noa/quant/lsm.hh>
#include <noa/quant/bsm.hh>

#include <cmath>
#include <iostream>

#include <torch/torch.h>

int main(int argc, char* argv[]) {
    int64_t n_paths = 1'000'000;
    int64_t n_steps = 250;
    double dt = 1.0/250;

    double sigma = 0.3;
    double S0 = 100.0;
    double strike = S0 * 0.85;
    double rate = 0.03;

    // LSM
    torch::Tensor paths_gbm = S0 * torch::cumprod(
            1 + rate * dt + sigma * std::sqrt(dt) * torch::randn({n_paths, n_steps}, torch::kFloat64),
            1);
    paths_gbm = torch::hstack({S0 * torch::ones({n_paths, 1}, torch::kFloat64), paths_gbm});

    double price_lsm = noa::quant::price_american_put_lsm(paths_gbm, dt, strike, rate);

    // BSM
    double T = static_cast<double>(n_steps) * dt;
    double S_min = 30;
    double S_max = 200;
    int64_t npoints_S = 3000;
    int64_t npoints_t = 3000;

    auto [V_bsm, S_arr, t_arr] = noa::quant::price_american_put_bs(
            strike, T, rate, sigma, S_min, S_max, npoints_S, npoints_t);
    auto price_bsm =
            V_bsm.index({torch::argmin(torch::abs(S_arr - S0)), -1}).item<double>();

    // results
    double rel_diff = (price_lsm - price_bsm) / price_bsm;
    std::cout << std::endl;
    std::cout << "LSM price: " << price_lsm << std::endl;
    std::cout << "BSM price: " << price_bsm << std::endl;
    std::cout << "Relative difference: " << rel_diff * 100 << "%" << std::endl;

    return 0;
}
