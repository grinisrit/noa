#include "unit-pms.hh"

#include <gtest/gtest.h>

using namespace ghmc::utils;

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

TEST(DCS, CoulombFrameParameters)
{
    const auto A = STANDARD_ROCK.A * ATOM_ENERGY * 1E-3; // GeV
    const auto cutoff = 1E-9;
    auto fCM = torch::zeros({kinetic_energies.numel(), 2}, torch::kFloat64);
    int i = 0;
    auto result = vmap<double>(kinetic_energies, [&](const auto &k) {
        return coulomb_frame_parameters(
            fCM[i++], k, A, MUON_MASS, cutoff);
    });
    ASSERT_TRUE(mean_error<double>(result, pumas_fcm_kinetic).item<double>() < 1E-6);
    ASSERT_TRUE(mean_error<double>(fCM.view_as(pumas_fcm_lorentz), pumas_fcm_lorentz).item<double>() < 1E-6);
}