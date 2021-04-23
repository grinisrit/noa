#include "test-data.hh"

#include <noa/pms/dcs_cuda.hh>
#include <noa/pms/constants.hh>

#include <gtest/gtest.h>

using namespace noa::pms;

TEST(DCS, BremsstrahlungCUDA) {
    auto kinetic_energies = DCSData::get_kinetic_energies().to(torch::kCUDA);
    auto recoil_energies = DCSData::get_recoil_energies().to(torch::kCUDA);
    auto pumas_brems = DCSData::get_pumas_brems().to(torch::kCUDA);
    auto res = bremsstrahlung_cuda(kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);

    std::cout << pumas_brems.slice(0,0,5) << "\n"
        << res.slice(0,0,5) << "\n";

}