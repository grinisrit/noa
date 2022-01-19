#include "test-data.hh"

#include <noa/pms/dcs.hh>
#include <noa/pms/physics.hh>
#include <noa/utils/common.hh>

#include <gtest/gtest.h>

using namespace noa::utils;
using namespace noa::pms;

TEST(DCS, BremsstrahlungCUDA) {
    const auto kinetic_energies = DCSData::get_kinetic_energies().to(torch::kCUDA);
    const auto recoil_energies = DCSData::get_recoil_energies().to(torch::kCUDA);
    const auto pumas_brems = DCSData::get_pumas_brems().to(torch::kCUDA);
    const auto result = torch::zeros_like(kinetic_energies);
    dcs::cuda::vmap_bremsstrahlung(result, kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error(result, pumas_brems).item<Scalar>() < 1E-11);
}
