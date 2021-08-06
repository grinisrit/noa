#include "measure-dcs-calc.hh"

#include <noa/pms/dcs.hh>

#include <benchmark/benchmark.h>

using namespace noa::pms;


BENCHMARK_F(DCSBenchmark, Bremsstrahlung)
(benchmark::State &state) {
    single_calculation(state, dcs::pumas::bremsstrahlung);
}

BENCHMARK_F(DCSBenchmark, BremsstrahlungVectorised)
(benchmark::State &state) {
    vectorised_calculation(state, dcs::pumas::bremsstrahlung);
}

BENCHMARK_F(DCSBenchmark, BremsstrahlungVectorisedLarge)
(benchmark::State &state) {
    large_vectorised_calculation(state, dcs::pumas::bremsstrahlung);
}

BENCHMARK_F(DCSBenchmark, BremsstrahlungVectorisedLargeOpenMP)
(benchmark::State &state) {
    large_vectorised_openmp_calculation(state, dcs::pumas::bremsstrahlung);
}

BENCHMARK_F(DCSBenchmark, DELBremsstrahlung)
(benchmark::State &state) {
    single_recoil_integral_calculation(state, dcs::pumas::bremsstrahlung, dcs::del_integrand);
}

BENCHMARK_F(DCSBenchmark, DELBremsstrahlungVectorised)
(benchmark::State &state) {
    vectorised_recoil_integral_calculation(state, dcs::pumas::bremsstrahlung, dcs::del_integrand);
}

BENCHMARK_F(DCSBenchmark, CELBremsstrahlung)
(benchmark::State &state) {
    single_recoil_integral_calculation(state, dcs::pumas::bremsstrahlung, dcs::cel_integrand);
}

BENCHMARK_F(DCSBenchmark, CELBremsstrahlungVectorised)
(benchmark::State &state) {
    vectorised_recoil_integral_calculation(state, dcs::pumas::bremsstrahlung, dcs::cel_integrand);
}

BENCHMARK_F(DCSBenchmark, PairProduction)
(benchmark::State &state) {
    single_calculation(state, dcs::pumas::pair_production);
}

BENCHMARK_F(DCSBenchmark, PairProductionVectorised)
(benchmark::State &state) {
    vectorised_calculation(state, dcs::pumas::pair_production);
}

BENCHMARK_F(DCSBenchmark, DELPairProduction)
(benchmark::State &state) {
    single_recoil_integral_calculation(state, dcs::pumas::pair_production, dcs::del_integrand);
}

BENCHMARK_F(DCSBenchmark, DELPairProductionVectorised)
(benchmark::State &state) {
    vectorised_recoil_integral_calculation(state, dcs::pumas::pair_production, dcs::del_integrand);
}

BENCHMARK_F(DCSBenchmark, CELPairProduction)
(benchmark::State &state) {
    single_recoil_integral_calculation(state, dcs::pumas::pair_production, dcs::cel_integrand);
}

BENCHMARK_F(DCSBenchmark, CELPairProductionVectorised)
(benchmark::State &state) {
    vectorised_recoil_integral_calculation(state, dcs::pumas::pair_production, dcs::cel_integrand);
}

BENCHMARK_F(DCSBenchmark, Photonuclear)
(benchmark::State &state) {
    single_calculation(state, dcs::pumas::photonuclear);
}

BENCHMARK_F(DCSBenchmark, PhotonuclearVectorised)
(benchmark::State &state) {
    vectorised_calculation(state, dcs::pumas::photonuclear);
}

BENCHMARK_F(DCSBenchmark, DELPhotonuclear)
(benchmark::State &state) {
    single_recoil_integral_calculation(state, dcs::pumas::photonuclear, dcs::del_integrand);
}

BENCHMARK_F(DCSBenchmark, DELPhotonuclearVectorised)
(benchmark::State &state) {
    vectorised_recoil_integral_calculation(state, dcs::pumas::photonuclear, dcs::del_integrand);
}

BENCHMARK_F(DCSBenchmark, CELPhotonuclear)
(benchmark::State &state) {
    single_recoil_integral_calculation(state, dcs::pumas::photonuclear, dcs::cel_integrand);
}

BENCHMARK_F(DCSBenchmark, CELPhotonuclearVectorised)
(benchmark::State &state) {
    vectorised_recoil_integral_calculation(state, dcs::pumas::photonuclear, dcs::cel_integrand);
}


BENCHMARK_F(DCSBenchmark, Ionisation)
(benchmark::State &state) {
    single_calculation(state, dcs::pumas::ionisation);
}

BENCHMARK_F(DCSBenchmark, IonisationVectorised)
(benchmark::State &state) {
    vectorised_calculation(state, dcs::pumas::ionisation);
}

BENCHMARK_F(DCSBenchmark, DELIonisation)
(benchmark::State &state) {
    single_recoil_integral_calculation(state, dcs::pumas::ionisation, dcs::del_integrand);
}

BENCHMARK_F(DCSBenchmark, DELIonisationVectorised)
(benchmark::State &state) {
    vectorised_recoil_integral_calculation(state, dcs::pumas::ionisation, dcs::del_integrand);
}

BENCHMARK_F(DCSBenchmark, CELIonisation)
(benchmark::State &state) {
    single_recoil_integral_calculation(state, dcs::pumas::ionisation, dcs::cel_integrand);
}

BENCHMARK_F(DCSBenchmark, CELIonisationVectorised)
(benchmark::State &state) {
    vectorised_recoil_integral_calculation(state, dcs::pumas::ionisation, dcs::cel_integrand);
}
