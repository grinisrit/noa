#pragma once

#include <ghmc/pms/dcs.hh>
#include <ghmc/pms/physics.hh>

using namespace ghmc::pms;

inline const auto kinetic = torch::tensor({100.0f, 120.0f, 140.0f});
inline const auto qi = torch::tensor({5.0168871f, 6.0202645f, 7.0236419f});

inline const auto expected_brems = torch::tensor({3.9684804E-6f, 4.0105072E-6f, 4.0428550E-6f});
inline const auto expected_pairs = torch::tensor({6.6595556E-6f, 6.8945456E-6f, 7.0870052E-6f});