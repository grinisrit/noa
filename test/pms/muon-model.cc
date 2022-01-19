

#include <iostream>
#include <chrono>
#include <gflags/gflags.h>

using namespace std::chrono;



DEFINE_string(materials, "pumas-materials", "Path to PUMAS materials data");


auto main(int argc, char **argv) -> int {

    gflags::SetUsageMessage("Functional tests for Muon PMS component");

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    gflags::ShutDownCommandLineFlags();

    return 0;
}