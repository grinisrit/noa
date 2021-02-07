#include "unit-pms.hh"

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

static void BM_CSIonisationAnalytic(benchmark::State &state)
{
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    const auto k = kinetic_analytic_ion[0].item<float>();
    const auto xlow = ghmc::pms::X_FRACTION;
    for (auto _ : state)
        cs_ionisation_analytic_kernel(
        element, mu, k, xlow); 
}
BENCHMARK(BM_CSIonisationAnalytic);

static void BM_CELBremsstrahlung(benchmark::State &state)
{
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    const auto k = kinetic_analytic_ion[0].item<float>();
    const auto xlow = ghmc::pms::X_FRACTION;
    for (auto _ : state)
        compute_dcs_integral_kernel(dcs_bremsstrahlung_kernel)(
        element, mu, k, xlow, true); 
}
BENCHMARK(BM_CELBremsstrahlung);