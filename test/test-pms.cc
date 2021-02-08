#include "unit-pms.hh"

#include <gtest/gtest.h>

TEST(PMS, DCSBremsstrahlung)
{
    auto brems = dcs_bremsstrahlung(STANDARD_ROCK, MUON_MASS, kinetic, qi);
    ASSERT_TRUE(torch::abs(brems - expected_brems).sum().item<float>() < 1E-12f);
}

TEST(PMS, DCSPairProduction)
{
    auto pairs = dcs_pair_production(STANDARD_ROCK, MUON_MASS, kinetic, qi);
    ASSERT_TRUE(torch::abs(pairs - expected_pairs).sum().item<float>() < 1E-12f);
}

TEST(PMS, DCSPhotonuclear)
{
    auto photo = dcs_photonuclear(STANDARD_ROCK, MUON_MASS, kinetic, qi);
    ASSERT_TRUE(torch::abs(photo - expected_photo).sum().item<float>() < 1E-12f);
}

TEST(PMS, DCSIonisation)
{
    auto ion = dcs_ionisation(STANDARD_ROCK, MUON_MASS, kinetic, qi);
    ASSERT_TRUE(torch::abs(ion - expected_ion).sum().item<float>() < 1E-11f);
}

TEST(PMS, CELBremsstrahlung)
{
    auto cel_brems = map_cs(dcs_bremsstrahlung_kernel)(
        STANDARD_ROCK, MUON_MASS, kinetic, X_FRACTION, 180, true);
    ASSERT_TRUE(torch::abs(cel_brems - expected_cel_brems).sum().item<float>() < 1E-11f);
}

TEST(PMS, CSIonisation)
{
    auto cs_ion = map_cs(dcs_ionisation_kernel)(
        STANDARD_ROCK, MUON_MASS, kinetic, X_FRACTION, 180, false);
    ASSERT_TRUE(torch::abs(cs_ion - expected_cs_ion).sum().item<float>() < 1E-12f);
}

TEST(PMS, CSIonisationAnalytic)
{
    auto cs_analytic_ion = map_cs(dcs_ionisation_kernel)(
        STANDARD_ROCK, MUON_MASS, kinetic_analytic_ion, X_FRACTION, 180, false);
    ASSERT_TRUE(torch::abs(cs_analytic_ion - expected_cs_analytic_ion).sum().item<float>() < 1E-11f);
}