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

TEST(DCS, CoulombDataComputation)
{
    const int N = kinetic_energies.numel();
    const auto A = STANDARD_ROCK.A * ATOM_ENERGY * 1E-3; // GeV
    const auto cutoff = 1E-9;

    auto fCM = torch::zeros({N, 2}, torch::kFloat64);
    auto sceening = torch::zeros({N, 3}, torch::kFloat64);
    auto a = torch::zeros({N, 3}, torch::kFloat64);
    auto b = torch::zeros({N, 3}, torch::kFloat64);

    auto kinetic0 = vmapi<double>(
        kinetic_energies,
        [&](const int i, const double &k) { return coulomb_frame_parameters(fCM[i], k, A, MUON_MASS, cutoff); });

    for_eachi<double>(
        kinetic0,
        [&](const int i, const double &k) { coulomb_screening_parameters(
                                                sceening[i], a[i], b[i], k, STANDARD_ROCK, MUON_MASS); });

    //auto fspins = vmap<double>(kinetic0, [](const auto &k) { return coulomb_spin_factor(k, MUON_MASS); });

    
    ASSERT_TRUE(mean_error<double>(fCM.view_as(pumas_fcm_lorentz), pumas_fcm_lorentz).item<double>() < 1E-9);
    ASSERT_TRUE(relative_error<double>(sceening.view_as(pumas_screening), pumas_screening).item<double>() < 1E-10);
}
