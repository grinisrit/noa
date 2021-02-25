#include "unit-pms.hh"

#include <benchmark/benchmark.h>

using namespace ghmc::pms;
using namespace ghmc::pms::dcs;

static void DCS_Bremsstrahlung(benchmark::State &state)
{
    const auto k = kinetic_energies[65].item<Scalar>();
    const auto q = recoil_energies[65].item<Scalar>();
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        default_bremsstrahlung(k, q, element, mu);
}
BENCHMARK(DCS_Bremsstrahlung);

static void DCS_BremsstrahlungVectorised(benchmark::State &state)
{
    const auto r = torch::zeros_like(kinetic_energies);
    const auto k = kinetic_energies;
    const auto q = recoil_energies;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_kernel(default_bremsstrahlung)(r, k, q, element, mu);
}
BENCHMARK(DCS_BremsstrahlungVectorised);

static void DCS_DELBremsstrahlung(benchmark::State &state)
{
    const auto k = kinetic_energies[65].item<Scalar>();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        compute_integral(default_bremsstrahlung)(
            k, xlow, element, mu, 180, false);
}
BENCHMARK(DCS_DELBremsstrahlung);

static void DCS_DELBremsstrahlungVectorised(benchmark::State &state)
{
    const auto r = torch::zeros_like(kinetic_energies);
    const auto k = kinetic_energies;
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_compute_integral(default_bremsstrahlung)(
            r, k, xlow, element, mu, 180, false);
}
BENCHMARK(DCS_DELBremsstrahlungVectorised);

static void DCS_CELBremsstrahlung(benchmark::State &state)
{
    const auto k = kinetic_energies[65].item<Scalar>();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        compute_integral(default_bremsstrahlung)(
            k, xlow, element, mu, 180, true);
}
BENCHMARK(DCS_CELBremsstrahlung);

static void DCS_CELBremsstrahlungVectorised(benchmark::State &state)
{
    const auto r = torch::zeros_like(kinetic_energies);
    const auto k = kinetic_energies;
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_compute_integral(default_bremsstrahlung)(
            r, k, xlow, element, mu, 180, true);
}
BENCHMARK(DCS_CELBremsstrahlungVectorised);

static void DCS_PairProduction(benchmark::State &state)
{
    const auto k = kinetic_energies[65].item<Scalar>();
    const auto q = recoil_energies[65].item<Scalar>();
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        default_pair_production(k, q, element, mu);
}
BENCHMARK(DCS_PairProduction);

static void DCS_PairProductionVectorised(benchmark::State &state)
{
    const auto r = torch::zeros_like(kinetic_energies);
    const auto k = kinetic_energies;
    const auto q = recoil_energies;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_kernel(default_pair_production)(r, k, q, element, mu);
}
BENCHMARK(DCS_PairProductionVectorised);

static void DCS_DELPairProduction(benchmark::State &state)
{
    const auto k = kinetic_energies[65].item<Scalar>();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        compute_integral(default_pair_production)(
            k, xlow, element, mu, 180, false);
}
BENCHMARK(DCS_DELPairProduction);

static void DCS_DELPairProductionVectorised(benchmark::State &state)
{
    const auto r = torch::zeros_like(kinetic_energies);
    const auto k = kinetic_energies;
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_compute_integral(default_pair_production)(
            r, k, xlow, element, mu, 180, false);
}
BENCHMARK(DCS_DELPairProductionVectorised);

static void DCS_CELPairProduction(benchmark::State &state)
{
    const auto k = kinetic_energies[65].item<Scalar>();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        compute_integral(default_pair_production)(
            k, xlow, element, mu, 180, true);
}
BENCHMARK(DCS_CELPairProduction);

static void DCS_CELPairProductionVectorised(benchmark::State &state)
{
    const auto r = torch::zeros_like(kinetic_energies);
    const auto k = kinetic_energies;
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_compute_integral(default_pair_production)(
            r, k, xlow, element, mu, 180, true);
}
BENCHMARK(DCS_CELPairProductionVectorised);

static void DCS_Photonuclear(benchmark::State &state)
{
    const auto k = kinetic_energies[65].item<Scalar>();
    const auto q = recoil_energies[65].item<Scalar>();
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        default_photonuclear(k, q, element, mu);
}
BENCHMARK(DCS_Photonuclear);

static void DCS_PhotonuclearVectorised(benchmark::State &state)
{
    const auto r = torch::zeros_like(kinetic_energies);
    const auto k = kinetic_energies;
    const auto q = recoil_energies;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_kernel(default_photonuclear)(r, k, q, element, mu);
}
BENCHMARK(DCS_PhotonuclearVectorised);

static void DCS_DELPhotonuclear(benchmark::State &state)
{
    const auto k = kinetic_energies[65].item<Scalar>();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        compute_integral(default_photonuclear)(
            k, xlow, element, mu, 180, false);
}
BENCHMARK(DCS_DELPhotonuclear);

static void DCS_DELPhotonuclearVectorised(benchmark::State &state)
{
    const auto r = torch::zeros_like(kinetic_energies);
    const auto k = kinetic_energies;
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_compute_integral(default_photonuclear)(
            r, k, xlow, element, mu, 180, false);
}
BENCHMARK(DCS_DELPhotonuclearVectorised);

static void DCS_CELPhotonuclear(benchmark::State &state)
{
    const auto k = kinetic_energies[65].item<Scalar>();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        compute_integral(default_photonuclear)(
            k, xlow, element, mu, 180, true);
}
BENCHMARK(DCS_CELPhotonuclear);

static void DCS_CELPhotonuclearVectorised(benchmark::State &state)
{
    const auto r = torch::zeros_like(kinetic_energies);
    const auto k = kinetic_energies;
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_compute_integral(default_photonuclear)(
            r, k, xlow, element, mu, 180, true);
}
BENCHMARK(DCS_CELPhotonuclearVectorised);

static void DCS_Ionisation(benchmark::State &state)
{
    const auto k = kinetic_energies[69].item<Scalar>();
    const auto q = recoil_energies[69].item<Scalar>();
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        default_ionisation(k, q, element, mu);
}
BENCHMARK(DCS_Ionisation);

static void DCS_IonisationVectorised(benchmark::State &state)
{
    const auto r = torch::zeros_like(kinetic_energies);
    const auto k = kinetic_energies;
    const auto q = recoil_energies;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_kernel(default_ionisation)(r, k, q, element, mu);
}
BENCHMARK(DCS_IonisationVectorised);

static void DCS_DELIonisation(benchmark::State &state)
{
    const auto k = kinetic_energies[69].item<Scalar>();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        compute_integral(default_ionisation)(
            k, xlow, element, mu, 180, false);
}
BENCHMARK(DCS_DELIonisation);

static void DCS_DELIonisationVectorised(benchmark::State &state)
{
    const auto r = torch::zeros_like(kinetic_energies);
    const auto k = kinetic_energies;
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_compute_integral(default_ionisation)(
            r, k, xlow, element, mu, 180, false);
}
BENCHMARK(DCS_DELIonisationVectorised);

static void DCS_CELIonisation(benchmark::State &state)
{
    const auto k = kinetic_energies[69].item<Scalar>();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        compute_integral(default_ionisation)(
            k, xlow, element, mu, 180, true);
}
BENCHMARK(DCS_CELIonisation);

static void DCS_CELIonisationVectorised(benchmark::State &state)
{
    const auto r = torch::zeros_like(kinetic_energies);
    const auto k = kinetic_energies;
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_compute_integral(default_ionisation)(
            r, k, xlow, element, mu, 180, true);
}
BENCHMARK(DCS_CELIonisationVectorised);
