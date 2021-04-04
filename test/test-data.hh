#pragma once

#include <noa/utils/common.hh>

using namespace noa::utils;

inline torch::Tensor lazy_load_or_fail(TensorOpt &tensor, const Path &path)
{
    if (tensor.has_value())
    {
        return tensor.value();
    }
    else
    {
        if (load_tensor(tensor, path))
        {
            return tensor.value();
        }
        else
        {
            throw std::runtime_error("Corrupted test data");
        }
    }
}

inline const auto noa_test_data = noa::utils::Path{"noa-test-data"};

inline const auto ghmc_dir = noa_test_data / "ghmc";

inline const auto pms_dir = noa_test_data / "pms";

inline const auto theta_pt = ghmc_dir / "theta.pt";

inline const auto momentum_pt = ghmc_dir / "momentum.pt";

inline const auto expected_fisher_pt = ghmc_dir / "expected_fisher.pt";

inline const auto expected_spectrum_pt = ghmc_dir / "expected_spectrum.pt";

inline const auto expected_energy_pt = ghmc_dir / "expected_energy.pt";

inline const auto expected_flow_theta_pt = ghmc_dir / "expected_flow_theta.pt";

inline const auto expected_flow_moment_pt = ghmc_dir / "expected_flow_moment.pt";
