#pragma once

#include "test-data.hh"

#include <noa/utils/common.hh>

#include <gtest/gtest.h>

using namespace noa::utils;

class GHMCData : public ::testing::Environment
{

    inline static TensorOpt theta = std::nullopt;
    inline static TensorOpt momentum = std::nullopt;
    inline static TensorOpt expected_fisher = std::nullopt;
    inline static TensorOpt expected_spectrum = std::nullopt;
    inline static TensorOpt expected_energy = std::nullopt;
    inline static TensorOpt expected_flow_theta = std::nullopt;
    inline static TensorOpt expected_flow_moment = std::nullopt;

public:
    static torch::Tensor get_theta()
    {
        return lazy_load_or_fail(theta, theta_pt);
    }

    static torch::Tensor get_momentum()
    {
        return lazy_load_or_fail(momentum, momentum_pt);
    }

    static torch::Tensor get_expected_fisher()
    {
        return lazy_load_or_fail(expected_fisher, expected_fisher_pt);
    }

    static torch::Tensor get_expected_spectrum()
    {
        return lazy_load_or_fail(expected_spectrum, expected_spectrum_pt);
    }

    static torch::Tensor get_expected_energy()
    {
        return lazy_load_or_fail(expected_energy, expected_energy_pt);
    }

    static torch::Tensor get_expected_flow_theta()
    {
        return lazy_load_or_fail(expected_flow_theta, expected_flow_theta_pt);
    }

    static torch::Tensor get_expected_flow_moment()
    {
        return lazy_load_or_fail(expected_flow_moment, expected_flow_moment_pt);
    }

    virtual void SetUp()
    {
        get_theta();
        get_momentum();
        get_expected_fisher();
        get_expected_spectrum();
        get_expected_energy();
        get_expected_flow_theta();
        get_expected_flow_moment();
    }
};