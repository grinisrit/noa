#pragma once

#include "test-data.hh"

#include <benchmark/benchmark.h>


struct DCSBenchmark : benchmark::Fixture {
    DCSBenchmark() {
        DCSData::get_all();
    }
};
