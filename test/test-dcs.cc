#include "test-data.hh"

#include <noa/pms/dcs.hh>
#include <noa/pms/constants.hh>
#include <noa/utils/common.hh>

#include <gtest/gtest.h>

using namespace noa::pms;

TEST(DCS, BremsstrahlungCPU) {
    auto kinetic_energies = DCSData::get_kinetic_energies();
    auto recoil_energies = DCSData::get_recoil_energies();
    auto pumas_brems = DCSData::get_pumas_brems();
    auto result = torch::zeros_like(kinetic_energies);
    dcs::vmap<Scalar>(dcs::pumas::bremsstrahlung)(
            result, kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error(result, pumas_brems).item<Scalar>() < 1E-11);
}

TEST(DCS, PairProductionCPU) {
    auto result = torch::zeros_like(DCSData::get_kinetic_energies());
    dcs::vmap<Scalar>(dcs::pumas::pair_production)(
            result,
            DCSData::get_kinetic_energies(),
            DCSData::get_recoil_energies(),
            STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error(result, DCSData::get_pumas_pprod()).item<Scalar>() < 1E-7);
}

TEST(DCS, PhotonuclearCPU) {
    auto result = torch::zeros_like(DCSData::get_kinetic_energies());
    dcs::vmap<Scalar>(dcs::pumas::photonuclear)(
            result,
            DCSData::get_kinetic_energies(),
            DCSData::get_recoil_energies(),
            STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error(result, DCSData::get_pumas_photo()).item<Scalar>() < 1E-9);
}

TEST(DCS, IonisationCPU) {
    auto result = torch::zeros_like(DCSData::get_kinetic_energies());
    dcs::vmap<Scalar>(dcs::pumas::ionisation)(
            result,
            DCSData::get_kinetic_energies(),
            DCSData::get_recoil_energies(),
            STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error(result, DCSData::get_pumas_ion()).item<Scalar>() < 1E-9);
}
