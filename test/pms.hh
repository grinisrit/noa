#pragma once

#include <ghmc/ghmc.hh>

#include <gtest/gtest.h>
#include <torch/torch.h>

inline const auto kinetic = torch::tensor({0.0010f, 0.0012f, 0.0014f, 0.0017f});
inline const auto qi =  torch::tensor({5.0168871f, 6.0202645f, 7.0236419f,8.528708f}) * 1E-5f;

inline const auto brems =  torch::tensor({6.1020778f, 5.8831033f, 5.6975736f, 5.4640792f}) * 1E-9f;

