#include "noa-bench.hh"

#include <noa/pms/constants.hh>
#include <noa/pms/dcs.hh>

#include <benchmark/benchmark.h>

using namespace noa::pms;

BENCHMARK_F(DCSBenchmark, Bremsstrahlung)
(benchmark::State &state)
{
    const auto k = DCSData::get_kinetic_energies()[65].item<Scalar>();
    const auto q = DCSData::get_recoil_energies()[65].item<Scalar>();
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        dcs::pumas::bremsstrahlung(k, q, element, mu);
}

BENCHMARK_F(DCSBenchmark, BremsstrahlungVectorisedCPU)
(benchmark::State &state) {
    const auto kinetic_energies = DCSData::get_kinetic_energies();
    const auto recoil_energies = DCSData::get_recoil_energies();
    auto result = torch::zeros_like(kinetic_energies);
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        dcs::vmap<Scalar>(dcs::pumas::bremsstrahlung)(
                result, kinetic_energies, recoil_energies, element, mu);
}

BENCHMARK_F(DCSBenchmark, BremsstrahlungVectorisedLargeCPU)
(benchmark::State &state) {
    const auto kinetic_energies = DCSData::get_kinetic_energies().repeat_interleave(1000);
    const auto recoil_energies = DCSData::get_recoil_energies().repeat_interleave(1000);
    auto result = torch::zeros_like(kinetic_energies);
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        dcs::vmap<Scalar>(dcs::pumas::bremsstrahlung)(
                result, kinetic_energies, recoil_energies, element, mu);
}

BENCHMARK_F(DCSBenchmark, BremsstrahlungVectorisedLargeCPUOpenMP)
(benchmark::State &state) {
    const auto kinetic_energies = DCSData::get_kinetic_energies().repeat_interleave(1000);
    const auto recoil_energies = DCSData::get_recoil_energies().repeat_interleave(1000);
    auto result = torch::zeros_like(kinetic_energies);
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        dcs::pvmap<Scalar>(dcs::pumas::bremsstrahlung)(
                result, kinetic_energies, recoil_energies, element, mu);
}