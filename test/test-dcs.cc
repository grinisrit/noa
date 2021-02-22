#include "unit-pms.hh"

#include <gtest/gtest.h>

using namespace ghmc::utils;
using namespace ghmc::pms;
using namespace ghmc::pms::dcs;

TEST(DCS, Bremsstrahlung)
{
    auto result = torch::zeros_like(kinetic_energies);
    map_kernel(default_bremsstrahlung)(
        result,
        kinetic_energies,
        recoil_energies,
        STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error<double>(result, pumas_brems).item<double>() < 1E-11);
}

TEST(DCS, DELBremsstrahlung)
{
    auto result = torch::zeros_like(kinetic_energies);
    map_compute_integral(default_bremsstrahlung)(
        result,
        kinetic_energies,
        X_FRACTION, STANDARD_ROCK, MUON_MASS, 180, false);
    ASSERT_TRUE(relative_error<double>(result, pumas_brems_del).item<double>() < 1E-7);
}

TEST(DCS, CELBremsstrahlung)
{
    auto result = torch::zeros_like(kinetic_energies);
    map_compute_integral(default_bremsstrahlung)(
        result,
        kinetic_energies,
        X_FRACTION, STANDARD_ROCK, MUON_MASS, 180, true);
    ASSERT_TRUE(relative_error<double>(result, pumas_brems_cel).item<double>() < 1E-7);
}

TEST(DCS, PairProduction)
{
    auto result = torch::zeros_like(kinetic_energies);
    map_kernel(default_pair_production)(
        result,
        kinetic_energies,
        recoil_energies,
        STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error<double>(result, pumas_pprod).item<double>() < 1E-7);
}

TEST(DCS, DELPairProduction)
{
    auto result = torch::zeros_like(kinetic_energies);
    map_compute_integral(default_pair_production)(
        result,
        kinetic_energies,
        X_FRACTION, STANDARD_ROCK, MUON_MASS, 180, false);
    ASSERT_TRUE(relative_error<double>(result, pumas_pprod_del).item<double>() < 1E-7);
}

TEST(DCS, CELPairProduction)
{
    auto result = torch::zeros_like(kinetic_energies);
    map_compute_integral(default_pair_production)(
        result,
        kinetic_energies,
        X_FRACTION, STANDARD_ROCK, MUON_MASS, 180, true);
    ASSERT_TRUE(relative_error<double>(result, pumas_pprod_cel).item<double>() < 1E-7);
}

TEST(DCS, Photonuclear)
{
    auto result = torch::zeros_like(kinetic_energies);
    map_kernel(default_photonuclear)(
        result,
        kinetic_energies,
        recoil_energies,
        STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error<double>(result, pumas_photo).item<double>() < 1E-9);
}

TEST(DCS, DELPhotonuclear)
{
    auto result = torch::zeros_like(kinetic_energies);
    map_compute_integral(default_photonuclear)(
        result,
        kinetic_energies,
        X_FRACTION, STANDARD_ROCK, MUON_MASS, 180, false);
    ASSERT_TRUE(relative_error<double>(result, pumas_photo_del).item<double>() < 1E-9);
}

TEST(DCS, CELPhotonuclear)
{
    auto result = torch::zeros_like(kinetic_energies);
    map_compute_integral(default_photonuclear)(
        result,
        kinetic_energies,
        X_FRACTION, STANDARD_ROCK, MUON_MASS, 180, true);
    ASSERT_TRUE(relative_error<double>(result, pumas_photo_cel).item<double>() < 1E-9);
}

TEST(DCS, Ionisation)
{
    auto result = torch::zeros_like(kinetic_energies);
    map_kernel(default_ionisation)(
        result,
        kinetic_energies,
        recoil_energies,
        STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error<double>(result, pumas_ion).item<double>() < 1E-9);
}

TEST(DCS, DELIonisation)
{
    auto result = torch::zeros_like(kinetic_energies);
    map_compute_integral(default_ionisation)(
        result,
        kinetic_energies,
        X_FRACTION, STANDARD_ROCK, MUON_MASS, 180, false);
    ASSERT_TRUE(relative_error<double>(result, pumas_ion_del).item<double>() < 1E-9);
}

TEST(DCS, CELIonisation)
{
    auto result = torch::zeros_like(kinetic_energies);
    map_compute_integral(default_ionisation)(
        result,
        kinetic_energies,
        X_FRACTION, STANDARD_ROCK, MUON_MASS, 180, true);
    ASSERT_TRUE(relative_error<double>(result, pumas_ion_cel).item<double>() < 1E-9);
}

TEST(DCS, CoulombHardScattering)
{
    const int nkin = kinetic_energies.numel();
    auto fCM = torch::zeros({nkin, 2}, torch::kFloat64);
    auto screen = torch::zeros({nkin, 9}, torch::kFloat64);
    auto fspin = torch::zeros_like(kinetic_energies);
    auto invlambda = torch::zeros_like(kinetic_energies);

    default_coulomb_data(
        fCM, screen, fspin, invlambda,
        kinetic_energies, STANDARD_ROCK, MUON_MASS);

    ASSERT_TRUE(relative_error<double>(screen.slice(1, 0, 3).reshape_as(pumas_screening), pumas_screening).item<double>() < 1E-10);
    ASSERT_TRUE(mean_error<double>(invlambda, pumas_invlambda).item<double>() < 1E-10);

    auto G = torch::zeros_like(fCM);
    default_coulomb_transport(
        G, screen, fspin, torch::tensor(1.0, torch::kFloat64));
    ASSERT_TRUE(relative_error<double>(G.view_as(pumas_transport), pumas_transport).item<double>() < 1E-10);

    G = G.view({1, nkin, 2});
    fCM = fCM.view({1, nkin, 2});
    screen = screen.view({1, nkin, 9});
    invlambda = invlambda.view({1, nkin});
    fspin = fspin.view({1, nkin});

    auto lb_h = torch::zeros_like(kinetic_energies);
    auto mu0 = torch::zeros_like(kinetic_energies);

    default_hard_scattering(
        mu0, lb_h, G, fCM, screen, invlambda, fspin);

    ASSERT_TRUE(relative_error<double>(mu0, pumas_mu0).item<double>() < 1E-11);
    ASSERT_TRUE(relative_error<double>(lb_h, pumas_lb_h).item<double>() < 1E-11);
}

TEST(DSC, CoulombSoftScattering)
{
    auto result = torch::zeros_like(kinetic_energies);
    default_soft_scattering(result, kinetic_energies, STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error<double>(result, pumas_soft_scatter).item<double>() < 1E-12);
}
