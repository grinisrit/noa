#include "pms.hh"

#include <benchmark/benchmark.h>

static void BM_DCSBremsstrahlung(benchmark::State &state)
{
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    const auto k = kinetic[0].item<float>();
    const auto q = qi[0].item<float>();
    for (auto _ : state)
        dcs_bremsstrahlung_kernel(element, mu, k, q);
}
BENCHMARK(BM_DCSBremsstrahlung);

static void BM_DCSPairProduction(benchmark::State &state)
{
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    const auto k = kinetic[0].item<float>();
    const auto q = qi[0].item<float>();
    for (auto _ : state)
        dcs_pair_production_kernel(element, mu, k, q);
}
BENCHMARK(BM_DCSPairProduction);

static void BM_DCSPhotonuclear(benchmark::State &state)
{
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    const auto k = kinetic[0].item<float>();
    const auto q = qi[0].item<float>();
    for (auto _ : state)
        dcs_photonuclear_kernel(element, mu, k, q);
}
BENCHMARK(BM_DCSPhotonuclear);

static void BM_DCSIonisation(benchmark::State &state)
{
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    const auto k = kinetic[0].item<float>();
    const auto q = qi[0].item<float>();
    for (auto _ : state)
        dcs_ionisation_kernel(element, mu, k, q);
}
BENCHMARK(BM_DCSIonisation);