#include "noa-bench.hh"

#include <noa/pms/dcs_cuda.hh>
#include <noa/pms/constants.hh>

#include <benchmark/benchmark.h>

using namespace noa::pms;

BENCHMARK_F(DCSBenchmark, BremsstrahlungVectorisedCPUptr)
(benchmark::State &state) {
    const auto kinetic_energies = DCSData::get_kinetic_energies();
    const auto recoil_energies = DCSData::get_recoil_energies();
    auto result = torch::zeros_like(kinetic_energies);
    const double A = STANDARD_ROCK.A;
    const int Z = STANDARD_ROCK.Z;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        bremsstrahlung_cpu_kernel_ptr<double>(result, kinetic_energies, recoil_energies,Z, A, mu);
}

BENCHMARK_F(DCSBenchmark, BremsstrahlungVectorisedCPU)
(benchmark::State &state) {
    const auto kinetic_energies = DCSData::get_kinetic_energies();
    const auto recoil_energies = DCSData::get_recoil_energies();
    auto result = torch::zeros_like(kinetic_energies);
    const double A = STANDARD_ROCK.A;
    const int Z = STANDARD_ROCK.Z;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        bremsstrahlung_cpu_kernel<double>(result, kinetic_energies, recoil_energies,Z, A, mu);
}


BENCHMARK_F(DCSBenchmark, BremsstrahlungVectorisedCPU32)
(benchmark::State &state) {
    const auto kinetic_energies = DCSData::get_kinetic_energies().to(torch::dtype(torch::kFloat32));
    const auto recoil_energies = DCSData::get_recoil_energies().to(torch::dtype(torch::kFloat32));
    auto result = torch::zeros_like(kinetic_energies);
    const float A = STANDARD_ROCK.A;
    const int Z = STANDARD_ROCK.Z;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        bremsstrahlung_cpu_kernel<float>(result, kinetic_energies, recoil_energies,  Z, A, mu);
}

BENCHMARK_F(DCSBenchmark, BremsstrahlungVectorisedCUDAptr)
(benchmark::State &state) {
    const auto kinetic_energies = DCSData::get_kinetic_energies().to(torch::kCUDA);
    const auto recoil_energies = DCSData::get_recoil_energies().to(torch::kCUDA);
    const auto result = torch::zeros_like(kinetic_energies);
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        bremsstrahlung_cuda_ptr<Scalar>(result, kinetic_energies, recoil_energies, element, mu);
}

BENCHMARK_F(DCSBenchmark, BremsstrahlungVectorisedCUDA)
(benchmark::State &state) {
    const auto kinetic_energies = DCSData::get_kinetic_energies().to(torch::kCUDA);
    const auto recoil_energies = DCSData::get_recoil_energies().to(torch::kCUDA);
    auto result = torch::zeros_like(kinetic_energies);
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        bremsstrahlung_cuda(result, kinetic_energies, recoil_energies, element, mu);
}

BENCHMARK_F(DCSBenchmark, BremsstrahlungVectorisedCUDA32)
(benchmark::State &state) {
    const auto kinetic_energies = DCSData::get_kinetic_energies().to(torch::dtype(torch::kFloat32).device(torch::kCUDA));
    const auto recoil_energies = DCSData::get_recoil_energies().to(torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto result = torch::zeros_like(kinetic_energies);
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        bremsstrahlung_cuda(result, kinetic_energies, recoil_energies, element, mu);
}

BENCHMARK_F(DCSBenchmark, BremsstrahlungVectorisedCUDA32ptr)
(benchmark::State &state) {
    const auto kinetic_energies = DCSData::get_kinetic_energies().to(torch::dtype(torch::kFloat32).device(torch::kCUDA));
    const auto recoil_energies = DCSData::get_recoil_energies().to(torch::dtype(torch::kFloat32).device(torch::kCUDA));
    const auto result = torch::zeros_like(kinetic_energies);
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        bremsstrahlung_cuda_ptr<float>(result, kinetic_energies, recoil_energies, element, mu);
}