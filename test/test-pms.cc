#include "pms.hh"

#include <gtest/gtest.h>

TEST(PMS, DCSBremsstrahlung)
{
    auto brems = torch::zeros_like(kinetic);
    dcs_bremsstrahlung(ghmc::pms::STANDARD_ROCK.Z,
                               ghmc::pms::STANDARD_ROCK.A,
                               ghmc::pms::MUON_MASS, kinetic, qi, brems);
    ASSERT_TRUE(torch::abs(brems - expected_brems).sum().item<float>() < 1E-11f);
}