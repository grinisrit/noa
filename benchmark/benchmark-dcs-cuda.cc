#include "noa-bench.hh"
#include "dcs-kernels.hh"

#include <noa/pms/constants.hh>
#include <benchmark/benchmark.h>

using namespace noa::pms;

BENCHMARK_F(DCSBenchmark, BremsstrahlungVectorisedCUDA)
(benchmark::State &state) {
    const auto kinetic_energies = DCSData::get_kinetic_energies().to(torch::kCUDA);
    const auto recoil_energies = DCSData::get_recoil_energies().to(torch::kCUDA);
    const auto result = torch::zeros_like(kinetic_energies);
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        bremsstrahlung_cuda<Scalar>(result, kinetic_energies, recoil_energies, element, mu);
}


BENCHMARK_F(DCSBenchmark, BremsstrahlungVectorisedLargeCUDA)
(benchmark::State &state) {
    const auto kinetic_energies = DCSData::get_kinetic_energies().to(torch::kCUDA).repeat_interleave(1000);
    const auto recoil_energies = DCSData::get_recoil_energies().to(torch::kCUDA).repeat_interleave(1000);
    const auto result = torch::zeros_like(kinetic_energies);
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        bremsstrahlung_cuda<Scalar>(result, kinetic_energies, recoil_energies, element, mu);
}
