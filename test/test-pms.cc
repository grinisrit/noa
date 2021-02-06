#include "pms.hh"

#include <gtest/gtest.h>

TEST(PMS, DCSBremsstrahlung)
{
    auto brems = dcs_bremsstrahlung(ghmc::pms::STANDARD_ROCK,
                                    ghmc::pms::MUON_MASS, kinetic, qi);
    ASSERT_TRUE(torch::abs(brems - expected_brems).sum().item<float>() < 1E-12f);
}

TEST(PMS, DCSPairProduction)
{
    auto pairs = dcs_pair_production(ghmc::pms::STANDARD_ROCK,
                                     ghmc::pms::MUON_MASS, kinetic, qi);
    ASSERT_TRUE(torch::abs(pairs - expected_pairs).sum().item<float>() < 1E-12f);
}

TEST(PMS, DCSPhotonuclear)
{
    auto photo = dcs_photonuclear(ghmc::pms::STANDARD_ROCK,
                                  ghmc::pms::MUON_MASS, kinetic, qi);                         
    ASSERT_TRUE(torch::abs(photo - expected_photo).sum().item<float>() < 1E-12f);
}

TEST(PMS, DCSIonisation)
{
    auto ion = dcs_ionisation(ghmc::pms::STANDARD_ROCK,
                              ghmc::pms::MUON_MASS, kinetic, qi);
    ASSERT_TRUE(torch::abs(ion - expected_ion).sum().item<float>() < 1E-11f);
}