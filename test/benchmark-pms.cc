#include "pms.hh"

#include <benchmark/benchmark.h>

static void BM_DCSBremsstrahlung(benchmark::State &state)
{
    const auto Z = STANDARD_ROCK.Z;
    const auto A = STANDARD_ROCK.A;
    const auto mu = MUON_MASS;
    const auto k = kinetic[0].item<float>();
    const auto q = qi[0].item<float>();
    for (auto _ : state)
        dcs_bremsstrahlung_kernel(Z,A,mu,k,q);
}
BENCHMARK(BM_DCSBremsstrahlung);

static void BM_DCSPairProduction(benchmark::State &state)
{
    const auto Z = STANDARD_ROCK.Z;
    const auto A = STANDARD_ROCK.A;
    const auto mu = MUON_MASS;
    const auto k = kinetic[0].item<float>();
    const auto q = qi[0].item<float>();
    for (auto _ : state)
        dcs_pair_production_kernel(Z,A,mu,k,q);
}
BENCHMARK(BM_DCSPairProduction);

static void BM_DCSPhotonuclear(benchmark::State &state)
{
    const auto Z = STANDARD_ROCK.Z;
    const auto A = STANDARD_ROCK.A;
    const auto mu = MUON_MASS;
    const auto k = kinetic[0].item<float>();
    const auto q = qi[0].item<float>();
    for (auto _ : state)
        dcs_photonuclear_kernel(Z,A,mu,k,q);
}
BENCHMARK(BM_DCSPhotonuclear);