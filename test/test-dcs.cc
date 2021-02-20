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

TEST(DCS, CoulombScattering)
{
    const int N = kinetic_energies.numel();
    auto G = torch::zeros({N, 2}, torch::kFloat64);
    auto fCM = torch::zeros({N, 2}, torch::kFloat64);
    auto screen = torch::zeros({N, 9}, torch::kFloat64);

    auto kinetic0 = vmapi<double>(
        kinetic_energies,
        [&](const int i, const double &k) { return coulomb_frame_parameters(fCM[i], k, STANDARD_ROCK, MUON_MASS); });

    auto invlambda = vmapi<double>(
        kinetic0,
        [&](const int i, const double &k) { return coulomb_screening_parameters(
                                                screen[i], k, STANDARD_ROCK, MUON_MASS); });

    auto fspin = vmap<double>(kinetic0, [](const auto &k) { return coulomb_spin_factor(k, MUON_MASS); });

    for_eachi<double>(
        fspin,
        [&](const int i, const double &fspin) { coulomb_transport_coefficients(
                                                    G[i], screen[i], fspin, 1.); });

    ASSERT_TRUE(relative_error<double>(screen.slice(1, 0, 3).reshape_as(pumas_screening), pumas_screening).item<double>() < 1E-10);
    ASSERT_TRUE(relative_error<double>(G.view_as(pumas_transport), pumas_transport).item<double>() < 1E-10);
    ASSERT_TRUE(mean_error<double>(invlambda, pumas_invlambda).item<double>() < 1E-10);

    G = G.view({N, 1, 2});
    fCM = fCM.view({N, 1, 2});
    screen = screen.view({N, 1, 9});
    invlambda = invlambda.view({N, 1});
    fspin = fspin.view({N, 1});

    const auto lb_h = torch::zeros_like(kinetic_energies);
    Scalar *plb_h = lb_h.data_ptr<Scalar>();
    const auto mu0 = torch::zeros_like(kinetic_energies);
    Scalar *pmu0 = mu0.data_ptr<Scalar>();

    for (int i = 0; i < N; i++)
        default_hard_scattering(
            plb_h[i], pmu0[i], G[i], fCM[i], screen[i], invlambda[i], fspin[i]);

    ASSERT_TRUE(relative_error<double>(mu0, pumas_mu0).item<double>() < 1E-11);
    ASSERT_TRUE(relative_error<double>(lb_h, pumas_lb_h).item<double>() < 1E-11);
}
