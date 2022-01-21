#include <gflags/gflags.h>

DEFINE_string(materials, "pumas-materials", "Path to PUMAS materials data");


auto main(int argc, char **argv) -> int {

    gflags::SetUsageMessage("Functional tests for PUMAS bindings in PMS");

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    gflags::ShutDownCommandLineFlags();

    return 0;
}