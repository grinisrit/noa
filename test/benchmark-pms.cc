#include "unit-pms.hh"

#include <benchmark/benchmark.h>

static void PMS_DCSBremsstrahlung(benchmark::State &state)
{
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    const auto k = kinetic[0].item<float>();
    const auto q = qi[0].item<float>();
    for (auto _ : state)
        dcs_bremsstrahlung_kernel(element, mu, k, q);
}
BENCHMARK(PMS_DCSBremsstrahlung);

static void PMS_DCSBremsstrahlungVect(benchmark::State &state)
{
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    const auto k = table_K;
    const auto q = 0.05 * k;
    auto r = torch::zeros_like(k);
    for (auto _ : state)
        eval_dcs(dcs_bremsstrahlung_kernel)(r, element, mu, k, q);
}
BENCHMARK(PMS_DCSBremsstrahlungVect);

static void PMS_DCSPairProduction(benchmark::State &state)
{
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    const auto k = kinetic[0].item<float>();
    const auto q = qi[0].item<float>();
    for (auto _ : state)
        dcs_pair_production_kernel(element, mu, k, q);
}
BENCHMARK(PMS_DCSPairProduction);

static void PMS_DCSPhotonuclear(benchmark::State &state)
{
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    const auto k = kinetic[0].item<float>();
    const auto q = qi[0].item<float>();
    for (auto _ : state)
        dcs_photonuclear_kernel(element, mu, k, q);
}
BENCHMARK(PMS_DCSPhotonuclear);

static void PMS_DCSIonisation(benchmark::State &state)
{
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    const auto k = kinetic[0].item<float>();
    const auto q = qi[0].item<float>();
    for (auto _ : state)
        dcs_ionisation_kernel(element, mu, k, q);
}
BENCHMARK(PMS_DCSIonisation);

static void PMS_AnalyticCSIonisation(benchmark::State &state)
{
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    const auto k = kinetic_analytic_ion[0].item<float>();
    const auto xlow = X_FRACTION;
    for (auto _ : state)
        cs_ionisation_analytic_kernel(
            element, mu, k, xlow);
}
BENCHMARK(PMS_AnalyticCSIonisation);

static void PMS_CELBremsstrahlung(benchmark::State &state)
{
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    const auto k = kinetic[0].item<float>();
    const auto xlow = X_FRACTION;
    for (auto _ : state)
        cs_kernel(dcs_bremsstrahlung_kernel)(
            element, mu, k, xlow, 180, true);
}
BENCHMARK(PMS_CELBremsstrahlung);

static void PMS_CSBremsstrahlung(benchmark::State &state)
{
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    const auto k = kinetic[0].item<float>();
    const auto xlow = X_FRACTION;
    for (auto _ : state)
        cs_kernel(dcs_bremsstrahlung_kernel)(
            element, mu, k, xlow, 180, false);
}
BENCHMARK(PMS_CSBremsstrahlung);

static void PMS_CSBremsstrahlungVect(benchmark::State &state)
{
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    const auto k = table_K;
    auto r = torch::zeros_like(k);
    const auto xlow = X_FRACTION;
    for (auto _ : state)
        eval_cs(dcs_bremsstrahlung_kernel)(
            r, element, mu, k, xlow, 180, false);
}
BENCHMARK(PMS_CSBremsstrahlungVect);

static void PMS_CSPairProduction(benchmark::State &state)
{
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    const auto k = kinetic[0].item<float>();
    const auto xlow = X_FRACTION;
    for (auto _ : state)
        cs_kernel(dcs_pair_production_kernel)(
            element, mu, k, xlow, 180, false);
}
BENCHMARK(PMS_CSPairProduction);

static void PMS_CSPhotonuclear(benchmark::State &state)
{
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    const auto k = kinetic[0].item<float>();
    const auto xlow = X_FRACTION;
    for (auto _ : state)
        cs_kernel(dcs_photonuclear_kernel)(
            element, mu, k, xlow, 180, false);
}
BENCHMARK(PMS_CSPhotonuclear);

static void PMS_CSIonisation(benchmark::State &state)
{
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    const auto k = kinetic[0].item<float>();
    const auto xlow = X_FRACTION;
    for (auto _ : state)
        cs_kernel(dcs_ionisation_kernel)(
            element, mu, k, xlow, 180, false);
}
BENCHMARK(PMS_CSIonisation);