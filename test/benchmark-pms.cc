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
        default_dcs_bremsstrahlung(Z,A,mu,k,q);
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
        default_dcs_pair_production(Z,A,mu,k,q);
}
BENCHMARK(BM_DCSPairProduction);