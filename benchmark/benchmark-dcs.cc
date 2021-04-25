#include "noa-bench.hh"
#include "dcs-kernels.hh"

#include <noa/pms/constants.hh>
#include <benchmark/benchmark.h>

using namespace noa::pms;

BENCHMARK_F(DCSBenchmark, BremsstrahlungVectorisedCPU)
(benchmark::State &state) {
    const auto kinetic_energies = DCSData::get_kinetic_energies();
    const auto recoil_energies = DCSData::get_recoil_energies();
    auto result = torch::zeros_like(kinetic_energies);
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        bremsstrahlung_cpu<Scalar>(result, kinetic_energies, recoil_energies, element, mu);
}

BENCHMARK_F(DCSBenchmark, BremsstrahlungVectorisedLargeCPU)
(benchmark::State &state) {
    const auto kinetic_energies = DCSData::get_kinetic_energies().repeat_interleave(1000);
    const auto recoil_energies = DCSData::get_recoil_energies().repeat_interleave(1000);
    auto result = torch::zeros_like(kinetic_energies);
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        bremsstrahlung_cpu_p<Scalar>(result, kinetic_energies, recoil_energies, element, mu);
}