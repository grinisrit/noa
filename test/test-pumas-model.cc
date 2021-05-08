#include "test-data.hh"

#include <noa/pms/pumas-model.hh>
#include <noa/pms/constants.hh>
#include <noa/utils/common.hh>

#include <gtest/gtest.h>

using namespace noa::utils;
using namespace noa::pms;
using namespace noa::pms::dcs;


TEST(DCS, CoulombHardScattering) {
    const auto kinetic_energies = DCSData::get_kinetic_energies();
    const int nkin = kinetic_energies.size(0);
    auto fCM = torch::zeros({nkin, 2}, torch::dtype(torch::kDouble).layout(torch::kStrided));
    auto screen = torch::zeros({nkin, 9}, torch::dtype(torch::kDouble).layout(torch::kStrided));
    auto fspin = torch::zeros_like(kinetic_energies);
    auto invlambda = torch::zeros_like(kinetic_energies);

    default_coulomb_data(
            fCM, screen, fspin, invlambda,
            DCSData::get_kinetic_energies(), STANDARD_ROCK, MUON_MASS);

    ASSERT_TRUE(relative_error(screen.slice(1, 0, 3).reshape_as(DCSData::get_pumas_screening()),
                                       DCSData::get_pumas_screening())
                        .item<Scalar>() < 1E-10);
    ASSERT_TRUE(mean_error(invlambda, DCSData::get_pumas_invlambda()).item<Scalar>() < 1E-10);

    auto G = torch::zeros_like(fCM);
    default_coulomb_transport(
            G, screen, fspin, torch::tensor(1.0, torch::dtype(torch::kDouble)));
    ASSERT_TRUE(relative_error(G.view_as(DCSData::get_pumas_transport()),
                                       DCSData::get_pumas_transport())
                        .item<Scalar>() < 1E-10);

    G = G.view({1, nkin, 2});
    fCM = fCM.view({1, nkin, 2});
    screen = screen.view({1, nkin, 9});
    invlambda = invlambda.view({1, nkin});
    fspin = fspin.view({1, nkin});

    auto lb_h = torch::zeros_like(DCSData::get_kinetic_energies());
    auto mu0 = torch::zeros_like(DCSData::get_kinetic_energies());

    default_hard_scattering(
            mu0, lb_h, G, fCM, screen, invlambda, fspin);

    ASSERT_TRUE(relative_error(mu0, DCSData::get_pumas_mu0()).item<Scalar>() < 1E-11);
    ASSERT_TRUE(relative_error(lb_h, DCSData::get_pumas_lb_h()).item<Scalar>() < 1E-11);
}

TEST(DCS, CoulombSoftScattering) {
    auto result = torch::zeros_like(DCSData::get_kinetic_energies());
    default_soft_scattering(result, DCSData::get_kinetic_energies(), STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error(result, DCSData::get_pumas_soft_scatter()).item<Scalar>() < 1E-12);
}
