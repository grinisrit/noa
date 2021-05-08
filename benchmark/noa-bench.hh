#pragma once

#include "test-data.hh"

#include <noa/pms/constants.hh>
#include <noa/pms/dcs.hh>

#include <benchmark/benchmark.h>

using namespace noa::pms;

struct DCSBenchmark : benchmark::Fixture {
    DCSBenchmark() {
        DCSData::get_all();
    }

    template<typename DCSFunc>
    inline void single_calculation(benchmark::State &state, const DCSFunc & dcs_func){
        const auto k = DCSData::get_kinetic_energies()[65].item<Scalar>();
        const auto q = DCSData::get_recoil_energies()[65].item<Scalar>();
        const auto element = STANDARD_ROCK;
        const auto mu = MUON_MASS;
        for (auto _ : state)
            dcs_func(k, q, element, mu);
    }

    template<typename DCSFunc>
    inline void vectorised_calculation(benchmark::State &state, const DCSFunc & dcs_func){
        const auto kinetic_energies = DCSData::get_kinetic_energies();
        const auto recoil_energies = DCSData::get_recoil_energies();
        auto result = torch::zeros_like(kinetic_energies);
        const auto element = STANDARD_ROCK;
        const auto mu = MUON_MASS;
        for (auto _ : state)
            dcs::vmap<Scalar>(dcs_func)(
                    result, kinetic_energies, recoil_energies, element, mu);
    }

    template<typename DCSFunc>
    inline void large_vectorised_calculation(benchmark::State &state, const DCSFunc & dcs_func){
        const auto kinetic_energies = DCSData::get_kinetic_energies().repeat_interleave(1000);
        const auto recoil_energies = DCSData::get_recoil_energies().repeat_interleave(1000);
        auto result = torch::zeros_like(kinetic_energies);
        const auto element = STANDARD_ROCK;
        const auto mu = MUON_MASS;
        for (auto _ : state)
            dcs::vmap<Scalar>(dcs_func)(
                    result, kinetic_energies, recoil_energies, element, mu);
    }

    template<typename DCSFunc>
    inline void large_vectorised_openmp_calculation(benchmark::State &state, const DCSFunc & dcs_func){
        const auto kinetic_energies = DCSData::get_kinetic_energies().repeat_interleave(1000);
        const auto recoil_energies = DCSData::get_recoil_energies().repeat_interleave(1000);
        auto result = torch::zeros_like(kinetic_energies);
        const auto element = STANDARD_ROCK;
        const auto mu = MUON_MASS;
        for (auto _ : state)
            dcs::pvmap<Scalar>(dcs_func)(
                    result, kinetic_energies, recoil_energies, element, mu);
    }

    template<typename DCSFunc>
    inline void single_del_calculation(benchmark::State &state, const DCSFunc & dcs_func) {
        const auto k = DCSData::get_kinetic_energies()[65].item<Scalar>();
        const auto xlow = X_FRACTION;
        const auto element = STANDARD_ROCK;
        const auto mu = MUON_MASS;
        for (auto _ : state)
            dcs::del_integral<Scalar>(dcs_func)(
                    k, xlow, element, mu, 180);
    }

    template<typename DCSFunc>
    inline void single_cel_calculation(benchmark::State &state, const DCSFunc & dcs_func) {
        const auto k = DCSData::get_kinetic_energies()[65].item<Scalar>();
        const auto xlow = X_FRACTION;
        const auto element = STANDARD_ROCK;
        const auto mu = MUON_MASS;
        for (auto _ : state)
            dcs::cel_integral<Scalar>(dcs_func)(
                    k, xlow, element, mu, 180);
    }

    template<typename DCSFunc>
    inline void vectorised_del_calculation(benchmark::State &state, const DCSFunc & dcs_func) {
        const auto r = torch::zeros_like(DCSData::get_kinetic_energies());
        const auto k = DCSData::get_kinetic_energies();
        const auto xlow = X_FRACTION;
        const auto element = STANDARD_ROCK;
        const auto mu = MUON_MASS;
        for (auto _ : state)
            dcs::vmap_integral<Scalar>(
                    dcs::del_integral<Scalar>(dcs_func))(
                    r, k, xlow, element, mu, 180);
    }

    template<typename DCSFunc>
    inline void vectorised_cel_calculation(benchmark::State &state, const DCSFunc & dcs_func) {
        const auto r = torch::zeros_like(DCSData::get_kinetic_energies());
        const auto k = DCSData::get_kinetic_energies();
        const auto xlow = X_FRACTION;
        const auto element = STANDARD_ROCK;
        const auto mu = MUON_MASS;
        for (auto _ : state)
            dcs::vmap_integral<Scalar>(
                    dcs::cel_integral<Scalar>(dcs_func))(
                    r, k, xlow, element, mu, 180);
    }
};


