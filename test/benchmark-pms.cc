#include "pms.hh"

#include <benchmark/benchmark.h>

static void BM_DCSBremsstrahlung(benchmark::State &state)
{
    auto res = torch::zeros_like(kinetic);
    for (auto _ : state)
        dcs_bremsstrahlung(
            STANDARD_ROCK.Z,
            STANDARD_ROCK.A,
            MUON_MASS, kinetic,
            qi, res);
}
BENCHMARK(BM_DCSBremsstrahlung);