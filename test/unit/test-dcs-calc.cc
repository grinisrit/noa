#include "test-data.hh"

#include <noa/pms/dcs.hh>
#include <noa/pms/physics.hh>
#include <noa/utils/common.hh>

#include <gtest/gtest.h>

using namespace noa::pms;
using namespace noa::utils;

TEST(DCS, Bremsstrahlung) {
    const auto kinetic_energies = DCSData::get_kinetic_energies();
    const auto recoil_energies = DCSData::get_recoil_energies();
    const auto pumas_brems = DCSData::get_pumas_brems();
    const auto result = torch::zeros_like(kinetic_energies);
    dcs::vmap(dcs::pumas::bremsstrahlung)(
            result, kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error(result, pumas_brems).item<Scalar>() < 1E-11);
}

TEST(DCS, DELBremsstrahlung) {
    const auto result = torch::zeros_like(DCSData::get_kinetic_energies());
    dcs::vmap_integral(
            dcs::recoil_integral(dcs::pumas::bremsstrahlung, dcs::del_integrand))(
            result,
            DCSData::get_kinetic_energies(),
            dcs::pumas::X_FRACTION, STANDARD_ROCK, MUON_MASS, 180);
    ASSERT_TRUE(relative_error(result, DCSData::get_pumas_brems_del()).item<Scalar>() < 1E-7);
}

TEST(DCS, CELBremsstrahlung) {
    const auto result = torch::zeros_like(DCSData::get_kinetic_energies());
    dcs::vmap_integral(
            dcs::recoil_integral(dcs::pumas::bremsstrahlung, dcs::cel_integrand))(
            result,
            DCSData::get_kinetic_energies(),
            dcs::pumas::X_FRACTION, STANDARD_ROCK, MUON_MASS, 180);
    ASSERT_TRUE(relative_error(result, DCSData::get_pumas_brems_cel()).item<Scalar>() < 1E-7);
}

TEST(DCS, PairProduction) {
    const auto result = torch::zeros_like(DCSData::get_kinetic_energies());
    dcs::vmap(dcs::pumas::pair_production)(
            result,
            DCSData::get_kinetic_energies(),
            DCSData::get_recoil_energies(),
            STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error(result, DCSData::get_pumas_pprod()).item<Scalar>() < 1E-7);
}


TEST(DCS, DELPairProduction) {
    const auto result = torch::zeros_like(DCSData::get_kinetic_energies());
    dcs::vmap_integral(
            dcs::recoil_integral(dcs::pumas::pair_production, dcs::del_integrand))(
            result,
            DCSData::get_kinetic_energies(),
            dcs::pumas::X_FRACTION, STANDARD_ROCK, MUON_MASS, 180);
    ASSERT_TRUE(relative_error(result, DCSData::get_pumas_pprod_del()).item<Scalar>() < 1E-7);
}

TEST(DCS, CELPairProduction) {
    const auto result = torch::zeros_like(DCSData::get_kinetic_energies());
    dcs::vmap_integral(
            dcs::recoil_integral(dcs::pumas::pair_production, dcs::cel_integrand))(
            result,
            DCSData::get_kinetic_energies(),
            dcs::pumas::X_FRACTION, STANDARD_ROCK, MUON_MASS, 180);
    ASSERT_TRUE(relative_error(result, DCSData::get_pumas_pprod_cel()).item<Scalar>() < 1E-7);
}

TEST(DCS, Photonuclear) {
    const auto result = torch::zeros_like(DCSData::get_kinetic_energies());
    dcs::vmap(dcs::pumas::photonuclear)(
            result,
            DCSData::get_kinetic_energies(),
            DCSData::get_recoil_energies(),
            STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error(result, DCSData::get_pumas_photo()).item<Scalar>() < 1E-9);
}

TEST(DCS, DELPhotonuclear) {
    const auto result = torch::zeros_like(DCSData::get_kinetic_energies());
    dcs::vmap_integral(
            dcs::recoil_integral(dcs::pumas::photonuclear, dcs::del_integrand))(
            result,
            DCSData::get_kinetic_energies(),
            dcs::pumas::X_FRACTION, STANDARD_ROCK, MUON_MASS, 180);
    ASSERT_TRUE(relative_error(result, DCSData::get_pumas_photo_del()).item<Scalar>() < 1E-9);
}

TEST(DCS, CELPhotonuclear) {
    const auto result = torch::zeros_like(DCSData::get_kinetic_energies());
    dcs::vmap_integral(
            dcs::recoil_integral(dcs::pumas::photonuclear, dcs::cel_integrand))(
            result,
            DCSData::get_kinetic_energies(),
            dcs::pumas::X_FRACTION, STANDARD_ROCK, MUON_MASS, 180);
    ASSERT_TRUE(relative_error(result, DCSData::get_pumas_photo_cel()).item<Scalar>() < 1E-9);
}


TEST(DCS, Ionisation) {
    const auto result = torch::zeros_like(DCSData::get_kinetic_energies());
    dcs::vmap(dcs::pumas::ionisation)(
            result,
            DCSData::get_kinetic_energies(),
            DCSData::get_recoil_energies(),
            STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error(result, DCSData::get_pumas_ion()).item<Scalar>() < 1E-9);
}

TEST(DCS, DELIonisation) {
    const auto result = torch::zeros_like(DCSData::get_kinetic_energies());
    dcs::vmap_integral(
            dcs::recoil_integral(dcs::pumas::ionisation, dcs::del_integrand))(
            result,
            DCSData::get_kinetic_energies(),
            dcs::pumas::X_FRACTION, STANDARD_ROCK, MUON_MASS, 180);
    ASSERT_TRUE(relative_error(result, DCSData::get_pumas_ion_del()).item<Scalar>() < 1E-9);
}

TEST(DCS, CELIonisation) {
    const auto result = torch::zeros_like(DCSData::get_kinetic_energies());
    dcs::vmap_integral(
            dcs::recoil_integral(dcs::pumas::ionisation, dcs::cel_integrand))(
            result,
            DCSData::get_kinetic_energies(),
            dcs::pumas::X_FRACTION, STANDARD_ROCK, MUON_MASS, 180);
    ASSERT_TRUE(relative_error(result, DCSData::get_pumas_ion_cel()).item<Scalar>() < 1E-9);
}

TEST(DCS, CoulombHardScattering) {
    const auto kinetic_energies = DCSData::get_kinetic_energies();
    const Index nkin = kinetic_energies.size(0);
    auto fCM = torch::zeros({nkin, 2}, torch::dtype(torch::kDouble).layout(torch::kStrided));
    auto screen = torch::zeros({nkin, 9}, torch::dtype(torch::kDouble).layout(torch::kStrided));
    auto fspin = torch::zeros_like(kinetic_energies);
    auto invlambda = torch::zeros_like(kinetic_energies);

    dcs::pumas::coulomb_data(
            fCM, screen, fspin, invlambda,
            DCSData::get_kinetic_energies(), STANDARD_ROCK, MUON_MASS);

    ASSERT_TRUE(relative_error(screen.slice(1, 0, 3).reshape_as(DCSData::get_pumas_screening()),
                               DCSData::get_pumas_screening())
                        .item<Scalar>() < 1E-10);
    ASSERT_TRUE(mean_error(invlambda, DCSData::get_pumas_invlambda()).item<Scalar>() < 1E-10);

    auto G = torch::zeros_like(fCM);
    dcs::pumas::coulomb_transport(
            G, screen, fspin, torch::tensor(1.0, torch::dtype(torch::kDouble)));
    ASSERT_TRUE(relative_error(G.view_as(DCSData::get_pumas_transport()),
                               DCSData::get_pumas_transport())
                        .item<Scalar>() < 1E-10);

    G = G.view({1, nkin, 2});
    fCM = fCM.view({1, nkin, 2});
    screen = screen.view({1, nkin, 9});
    invlambda = invlambda.view({1, nkin});
    fspin = fspin.view({1, nkin});

    const auto lb_h = torch::zeros_like(DCSData::get_kinetic_energies());
    const auto mu0 = torch::zeros_like(DCSData::get_kinetic_energies());

    dcs::pumas::hard_scattering(
            mu0, lb_h, G, fCM, screen, invlambda, fspin);

    ASSERT_TRUE(relative_error(mu0, DCSData::get_pumas_mu0()).item<Scalar>() < 1E-11);
    ASSERT_TRUE(relative_error(lb_h, DCSData::get_pumas_lb_h()).item<Scalar>() < 1E-11);
}

TEST(DCS, CoulombSoftScattering) {
    const auto result = torch::zeros_like(DCSData::get_kinetic_energies());
    dcs::pumas::soft_scattering(result, DCSData::get_kinetic_energies(), STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error(result, DCSData::get_pumas_soft_scatter()).item<Scalar>() < 1E-12);
}
