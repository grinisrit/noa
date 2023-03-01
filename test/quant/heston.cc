/** For results visualization see docs/quant/heston.ipynb */

#include <noa/quant/heston.hh>

#include <cstdint>
#include <cmath>
#include <iostream>

#include <torch/torch.h>

using namespace torch::indexing;
using namespace noa::quant;

int64_t n_paths = 10000;
int64_t n_steps = 2500;
double dt = 1.0/250;
double rho = -0.6;
double S0 = 100.0;

double xi = 0.009;
double theta = 0.01;
double v0 = theta;
double kappa = 0.06;
/*
double xi = 0.1;
double theta = 1;
double v0 = theta;
double kappa = 1;
*/

double nonc_chi2_moment_true(double df, double nonc, int moment_num) {
    if (moment_num == 1)
        return df + nonc;
    else if (moment_num == 2)
        return 2*df + 4*nonc;
    else if (moment_num == 3)
        return std::pow(2.0, 1.5) * (df + 3*nonc) /
               std::pow(df + 2*nonc, 1.5);
    else
        throw std::invalid_argument("Only 1-3 moments are implemented");
};

double nonc_chi2_moment_mc(const torch::Tensor& sample, int moment_num) {
    if (moment_num == 1)
        return sample.mean().item<double>();
    else if (moment_num == 2)
        return sample.std().pow(2).item<double>();
    else if (moment_num == 3)
        return sample.mean().pow(3).item<double>() /
               sample.std().pow(3).item<double>();
    else
        throw std::invalid_argument("Only 1-3 moments are implemented");
}

void test_noncentral_chi2() {
    std::cout << "Running functional test of `noncentral_chisquare()`" << std::endl;
    int sample_size = 100000;
    // pairs of values for which s^2/m^2 < 1.5 (see `heston.hh`)
    double df_quad[5] =   {1.5, 1.5, 1.5, 2, 3.0};
    double nonc_quad[5] = {0.5, 1.2, 2.0, 3.0, 4.0};
    // pairs of values for which s^2/m^2 > 1.5
    double df_exp[5] =   {0.7, 0.6, 0.5, 0.25, 0.5};
    double nonc_exp[5] = {0.7, 1.0, 1.1, 1.5, 0.5};

    std::cout << "Quadratic scheme\n________________" << std::endl;
    for (int i = 0; i < 5; i++) {
        torch::Tensor sample = noncentral_chisquare(
                df_quad[i]*torch::ones(sample_size),
                nonc_quad[i]*torch::ones(sample_size));
        double m1_mc = nonc_chi2_moment_mc(sample, 1);
        double m2_mc = nonc_chi2_moment_mc(sample, 2);
        double m3_mc = nonc_chi2_moment_mc(sample, 3);
        double m1_true = nonc_chi2_moment_true(df_quad[i], nonc_quad[i], 1);
        double m2_true = nonc_chi2_moment_true(df_quad[i], nonc_quad[i], 2);
        double m3_true = nonc_chi2_moment_true(df_quad[i], nonc_quad[i], 3);

        std::cout << "df = " << df_quad[i] << ", nonc = " << nonc_quad[i] << std::endl;
        std::cout << "1st moment. Monte-Carlo: " << m1_mc << ", true: " << m1_true << std::endl;
        std::cout << "2nd moment. Monte-Carlo: " << m2_mc << ", true: " << m2_true << std::endl;
        std::cout << "3rd moment. Monte-Carlo: " << m3_mc << ", true: " << m3_true << std::endl << std::endl;
    }
    std::cout << "Exponential scheme\n________________" << std::endl;
    for (int i = 0; i < 5; i++) {
        torch::Tensor sample = noncentral_chisquare(
                df_exp[i]*torch::ones(sample_size),
                nonc_exp[i]*torch::ones(sample_size));
        double m1_mc = nonc_chi2_moment_mc(sample, 1);
        double m2_mc = nonc_chi2_moment_mc(sample, 2);
        double m3_mc = nonc_chi2_moment_mc(sample, 3);
        double m1_true = nonc_chi2_moment_true(df_exp[i], nonc_exp[i], 1);
        double m2_true = nonc_chi2_moment_true(df_exp[i], nonc_exp[i], 2);
        double m3_true = nonc_chi2_moment_true(df_exp[i], nonc_exp[i], 3);

        std::cout << "df = " << df_exp[i] << ", nonc = " << nonc_exp[i] << std::endl;
        std::cout << "1st moment. Monte-Carlo: " << m1_mc << ", true: " << m1_true << std::endl;
        std::cout << "2nd moment. Monte-Carlo: " << m2_mc << ", true: " << m2_true << std::endl;
        std::cout << "3rd moment. Monte-Carlo: " << m3_mc << ", true: " << m3_true << std::endl << std::endl;
    }
}

void test_cir() {
    std::cout << "Running functional test of `generate_cir()`" << std::endl;
    torch::Tensor init_state_var = v0 * torch::ones(n_paths, torch::dtype<double>());
    torch::Tensor cir_paths = noa::quant::generate_cir(n_paths, n_steps, dt, init_state_var, kappa, theta, xi);
    torch::save(cir_paths, "cir_paths.pt");
}

void test_heston() {
    std::cout << "Running functional test of `generate_heston()`" << std::endl;
    torch::Tensor init_state_var = v0 * torch::ones(n_paths,torch::dtype<double>());
    torch::Tensor init_state_price = S0 * torch::ones(n_paths,torch::dtype<double>());

    torch::Tensor heston_paths, var_paths;
    std::tie(heston_paths, var_paths) = noa::quant::generate_heston(
            n_paths, n_steps, dt, init_state_price, init_state_var,
            kappa, theta, xi, rho
    );
    torch::save(heston_paths, "heston_paths.pt");
    torch::save(var_paths, "var_paths.pt");
}

int main(int argc, char* argv[]) {
    test_noncentral_chi2();
    test_cir();
    test_heston();
    std::cout << "Done." << std::endl;
    return 0;
}
