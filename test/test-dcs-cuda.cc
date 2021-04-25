#include "test-data.hh"
#include "dcs-kernels.hh"

#include <noa/pms/constants.hh>
#include <noa/utils/common.hh>

#include <gtest/gtest.h>

using namespace noa::pms;

TEST(DCS, BremsstrahlungCUDA) {
    auto kinetic_energies = DCSData::get_kinetic_energies().to(torch::kCUDA);
    auto recoil_energies = DCSData::get_recoil_energies().to(torch::kCUDA);
    auto pumas_brems = DCSData::get_pumas_brems().to(torch::kCUDA);
    auto result = torch::zeros_like(kinetic_energies);
    bremsstrahlung_cuda<Scalar>(result, kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error(result, pumas_brems).item<Scalar>() < 1E-11);
}
