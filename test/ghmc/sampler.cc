#include "routines.hh"

#include <gflags/gflags.h>

DEFINE_bool(normal, false, "Test GHMC sampler for Normal Distribution");
DEFINE_bool(funnel, false, "Test GHMC sampler for Funnel Distribution");
DEFINE_bool(bnet, false, "Test GHMC sampler for Bayesian Deep Learning");
DEFINE_bool(cuda, false, "Run on GPU");

auto main(int argc, char **argv) -> int {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    auto device = FLAGS_cuda ? torch::kCUDA : torch::kCPU;

    if(FLAGS_cuda && !torch::cuda::is_available()){
        std::cerr << "No CUDA available, run tests on CPU only.\n";
        return 1;
    }

    if (FLAGS_normal)
        if (!sample_normal_distribution("ghmc_sample_normal_distribution.pt", device))
            return 1;

    if (FLAGS_funnel)
        if (!sample_funnel_distribution("ghmc_sample_funnel_distribution.pt", device))
            return 1;

    if (FLAGS_bnet)
        if (!sample_bayesian_net("ghmc_sample_bayesian_net.pt", device))
            return 1;

    return 0;
}

