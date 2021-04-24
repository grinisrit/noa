#include "test-data.hh"

#include <noa/pms/dcs_cuda.hh>
#include <noa/pms/constants.hh>
#include <noa/utils/common.hh>

#include <gtest/gtest.h>

using namespace noa::pms;

TEST(DCS, BremsstrahlungCUDA) {
    auto kinetic_energies = DCSData::get_kinetic_energies().to(torch::kCUDA);
    auto recoil_energies = DCSData::get_recoil_energies().to(torch::kCUDA);
    auto pumas_brems = DCSData::get_pumas_brems().to(torch::kCUDA);
    auto result = torch::zeros_like(kinetic_energies);
    bremsstrahlung_cuda(result, kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error(result, pumas_brems).item<Scalar>() < 1E-11);
}

TEST(DCS, BremsstrahlungCUDA32) {
    auto kinetic_energies = DCSData::get_kinetic_energies().to(torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto recoil_energies = DCSData::get_recoil_energies().to(torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto pumas_brems = DCSData::get_pumas_brems().to(torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto result = torch::zeros_like(kinetic_energies);
    bremsstrahlung_cuda(result, kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error(result, pumas_brems).item<float>() < 1E-6);
}

TEST(DCS, BremsstrahlungCUDAptr) {
    auto kinetic_energies = DCSData::get_kinetic_energies().to(torch::kCUDA);
    auto recoil_energies = DCSData::get_recoil_energies().to(torch::kCUDA);
    auto pumas_brems = DCSData::get_pumas_brems().to(torch::kCUDA);
    auto result = torch::zeros_like(kinetic_energies);
    bremsstrahlung_cuda_ptr<Scalar>(result, kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error(result, pumas_brems).item<Scalar>() < 1E-11);
}

TEST(DCS, BremsstrahlungCUDA32ptr) {
    auto kinetic_energies = DCSData::get_kinetic_energies().to(torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto recoil_energies = DCSData::get_recoil_energies().to(torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto pumas_brems = DCSData::get_pumas_brems().to(torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto result = torch::zeros_like(kinetic_energies);
    bremsstrahlung_cuda_ptr<float>(result, kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error(result, pumas_brems).item<float>() < 1E-6);
}

TEST(DCS, BremsstrahlungCPU) {
    auto kinetic_energies = DCSData::get_kinetic_energies();
    auto recoil_energies = DCSData::get_recoil_energies();
    auto pumas_brems = DCSData::get_pumas_brems();
    auto result = torch::zeros_like(kinetic_energies);
    bremsstrahlung_cpu(result, kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error(result, pumas_brems).item<Scalar>() < 1E-11);
}

TEST(DCS, BremsstrahlungCPU32) {
    auto kinetic_energies = DCSData::get_kinetic_energies().to(torch::dtype(torch::kFloat32));
    auto recoil_energies = DCSData::get_recoil_energies().to(torch::dtype(torch::kFloat32));
    auto pumas_brems = DCSData::get_pumas_brems().to(torch::dtype(torch::kFloat32));
    auto result = torch::zeros_like(kinetic_energies);
    bremsstrahlung_cpu(result, kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
    ASSERT_TRUE(relative_error(result, pumas_brems).item<float>() < 1E-6);
}