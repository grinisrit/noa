#include "noa-bench.hh"

#include <noa/pms/conf.hh>
#include <noa/pms/dcs.hh>
#include <noa/pms/physics.hh>


#include <benchmark/benchmark.h>

using namespace noa::pms;
using namespace noa::pms::dcs;

BENCHMARK_F(DCSBenchmark, DCS_Bremsstrahlung)
(benchmark::State &state)
{
    const auto k = DCSData::get_kinetic_energies()[65].item<Scalar>();
    const auto q = DCSData::get_recoil_energies()[65].item<Scalar>();
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        default_bremsstrahlung(k, q, element, mu);
}

BENCHMARK_F(DCSBenchmark, DCS_BremsstrahlungVectorised)
(benchmark::State &state)
{
    const auto r = torch::zeros_like(DCSData::get_kinetic_energies());
    const auto k = DCSData::get_kinetic_energies();
    const auto q = DCSData::get_recoil_energies();
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_kernel(default_bremsstrahlung)(r, k, q, element, mu);
}

BENCHMARK_F(DCSBenchmark, DCS_DELBremsstrahlung)
(benchmark::State &state)
{
    const auto k = DCSData::get_kinetic_energies()[65].item<Scalar>();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        compute_integral(default_bremsstrahlung)(
            k, xlow, element, mu, 180, false);
}

BENCHMARK_F(DCSBenchmark, DCS_DELBremsstrahlungVectorised)
(benchmark::State &state)
{
    const auto r = torch::zeros_like(DCSData::get_kinetic_energies());
    const auto k = DCSData::get_kinetic_energies();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_compute_integral(default_bremsstrahlung)(
            r, k, xlow, element, mu, 180, false);
}

BENCHMARK_F(DCSBenchmark, DCS_CELBremsstrahlung)
(benchmark::State &state)
{
    const auto k = DCSData::get_kinetic_energies()[65].item<Scalar>();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        compute_integral(default_bremsstrahlung)(
            k, xlow, element, mu, 180, true);
}

BENCHMARK_F(DCSBenchmark, DCS_CELBremsstrahlungVectorised)
(benchmark::State &state)
{
    const auto r = torch::zeros_like(DCSData::get_kinetic_energies());
    const auto k = DCSData::get_kinetic_energies();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_compute_integral(default_bremsstrahlung)(
            r, k, xlow, element, mu, 180, true);
}

BENCHMARK_F(DCSBenchmark, DCS_PairProduction)
(benchmark::State &state)
{
    const auto k = DCSData::get_kinetic_energies()[65].item<Scalar>();
    const auto q = DCSData::get_recoil_energies()[65].item<Scalar>();
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        default_pair_production(k, q, element, mu);
}

BENCHMARK_F(DCSBenchmark, DCS_PairProductionVectorised)
(benchmark::State &state)
{
    const auto r = torch::zeros_like(DCSData::get_kinetic_energies());
    const auto k = DCSData::get_kinetic_energies();
    const auto q = DCSData::get_recoil_energies();
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_kernel(default_pair_production)(r, k, q, element, mu);
}

BENCHMARK_F(DCSBenchmark, DCS_DELPairProduction)
(benchmark::State &state)
{
    const auto k = DCSData::get_kinetic_energies()[65].item<Scalar>();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        compute_integral(default_pair_production)(
            k, xlow, element, mu, 180, false);
}

BENCHMARK_F(DCSBenchmark, DCS_DELPairProductionVectorised)
(benchmark::State &state)
{
    const auto r = torch::zeros_like(DCSData::get_kinetic_energies());
    const auto k = DCSData::get_kinetic_energies();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_compute_integral(default_pair_production)(
            r, k, xlow, element, mu, 180, false);
}

BENCHMARK_F(DCSBenchmark, DCS_CELPairProduction)
(benchmark::State &state)
{
    const auto k = DCSData::get_kinetic_energies()[65].item<Scalar>();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        compute_integral(default_pair_production)(
            k, xlow, element, mu, 180, true);
}

BENCHMARK_F(DCSBenchmark, DCS_CELPairProductionVectorised)
(benchmark::State &state)
{
    const auto r = torch::zeros_like(DCSData::get_kinetic_energies());
    const auto k = DCSData::get_kinetic_energies();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_compute_integral(default_pair_production)(
            r, k, xlow, element, mu, 180, true);
}

BENCHMARK_F(DCSBenchmark, DCS_Photonuclear)
(benchmark::State &state)
{
    const auto k = DCSData::get_kinetic_energies()[65].item<Scalar>();
    const auto q = DCSData::get_recoil_energies()[65].item<Scalar>();
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        default_photonuclear(k, q, element, mu);
}

BENCHMARK_F(DCSBenchmark, DCS_PhotonuclearVectorised)
(benchmark::State &state)
{
    const auto r = torch::zeros_like(DCSData::get_kinetic_energies());
    const auto k = DCSData::get_kinetic_energies();
    const auto q = DCSData::get_recoil_energies();
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_kernel(default_photonuclear)(r, k, q, element, mu);
}

BENCHMARK_F(DCSBenchmark, DCS_DELPhotonuclear)
(benchmark::State &state)
{
    const auto k = DCSData::get_kinetic_energies()[65].item<Scalar>();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        compute_integral(default_photonuclear)(
            k, xlow, element, mu, 180, false);
}

BENCHMARK_F(DCSBenchmark, DCS_DELPhotonuclearVectorised)
(benchmark::State &state)
{
    const auto r = torch::zeros_like(DCSData::get_kinetic_energies());
    const auto k = DCSData::get_kinetic_energies();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_compute_integral(default_photonuclear)(
            r, k, xlow, element, mu, 180, false);
}

BENCHMARK_F(DCSBenchmark, DCS_CELPhotonuclear)
(benchmark::State &state)
{
    const auto k = DCSData::get_kinetic_energies()[65].item<Scalar>();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        compute_integral(default_photonuclear)(
            k, xlow, element, mu, 180, true);
}

BENCHMARK_F(DCSBenchmark, DCS_CELPhotonuclearVectorised)
(benchmark::State &state)
{
    const auto r = torch::zeros_like(DCSData::get_kinetic_energies());
    const auto k = DCSData::get_kinetic_energies();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_compute_integral(default_photonuclear)(
            r, k, xlow, element, mu, 180, true);
}

BENCHMARK_F(DCSBenchmark, DCS_Ionisation)
(benchmark::State &state)
{
    const auto k = DCSData::get_kinetic_energies()[69].item<Scalar>();
    const auto q = DCSData::get_recoil_energies()[69].item<Scalar>();
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        default_ionisation(k, q, element, mu);
}

BENCHMARK_F(DCSBenchmark, DCS_IonisationVectorised)
(benchmark::State &state)
{
    const auto r = torch::zeros_like(DCSData::get_kinetic_energies());
    const auto k = DCSData::get_kinetic_energies();
    const auto q = DCSData::get_recoil_energies();
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_kernel(default_ionisation)(r, k, q, element, mu);
}

BENCHMARK_F(DCSBenchmark, DCS_DELIonisation)
(benchmark::State &state)
{
    const auto k = DCSData::get_kinetic_energies()[69].item<Scalar>();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        compute_integral(default_ionisation)(
            k, xlow, element, mu, 180, false);
}

BENCHMARK_F(DCSBenchmark, DCS_DELIonisationVectorised)
(benchmark::State &state)
{
    const auto r = torch::zeros_like(DCSData::get_kinetic_energies());
    const auto k = DCSData::get_kinetic_energies();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_compute_integral(default_ionisation)(
            r, k, xlow, element, mu, 180, false);
}

BENCHMARK_F(DCSBenchmark, DCS_CELIonisation)
(benchmark::State &state)
{
    const auto k = DCSData::get_kinetic_energies()[69].item<Scalar>();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        compute_integral(default_ionisation)(
            k, xlow, element, mu, 180, true);
}

BENCHMARK_F(DCSBenchmark, DCS_CELIonisationVectorised)
(benchmark::State &state)
{
    const auto r = torch::zeros_like(DCSData::get_kinetic_energies());
    const auto k = DCSData::get_kinetic_energies();
    const auto xlow = X_FRACTION;
    const auto element = STANDARD_ROCK;
    const auto mu = MUON_MASS;
    for (auto _ : state)
        map_compute_integral(default_ionisation)(
            r, k, xlow, element, mu, 180, true);
}
