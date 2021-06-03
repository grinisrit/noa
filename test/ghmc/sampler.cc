#include "routines.hh"

#include <gflags/gflags.h>

DEFINE_bool(normal, false, "Run GHMC sampler for Normal Distribution");
DEFINE_bool(funnel, false, "Run GHMC sampler for Funnel Distribution");

auto main(int argc, char **argv) -> int {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_normal)
        if (!sample_normal_distribution("ghmc_sample_normal_distribution.pt"))
            return 1;

    if (FLAGS_funnel)
        if (!sample_funnel_distribution("ghmc_sample_funnel_distribution.pt"))
            return 1;

    return 0;
}

