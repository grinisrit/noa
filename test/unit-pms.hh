#pragma once

#include <ghmc/pms/dcs.hh>
#include <ghmc/pms/physics.hh>

using namespace ghmc::pms;

inline const auto dcs_bremsstrahlung = map_dcs(dcs_bremsstrahlung_kernel);
inline const auto dcs_pair_production = map_dcs(dcs_pair_production_kernel);
inline const auto dcs_photonuclear = map_dcs(dcs_photonuclear_kernel);
inline const auto dcs_ionisation = map_dcs(dcs_ionisation_kernel);

inline const auto kinetic = torch::tensor({100.0f, 120.0f, 140.0f});
inline const auto qi = torch::tensor({5.0168871f, 6.0202645f, 7.0236419f});

inline const auto expected_brems = torch::tensor({3.9684804E-6f, 4.0105072E-6f, 4.0428550E-6f});
inline const auto expected_pairs = torch::tensor({6.6597247E-6f, 6.8944339E-6f, 7.0868744E-6f});
inline const auto expected_photo = torch::tensor({2.3121508E-6f, 2.2614157E-6f, 2.2223203E-6f});
inline const auto expected_ion = torch::tensor({3.0583396E-5f, 2.5583496E-5f, 2.1998339E-5f});

inline const auto expected_cel_brems = torch::tensor({0.9992710E-5f, 1.2361067E-5f, 1.4781266E-5f});
inline const auto expected_cs_ion = torch::tensor({1.3020459E-6f, 1.0920769E-6f, 0.9411282E-6f});

inline const auto kinetic_analytic_ion = torch::tensor({8.f, 9.f, 10.f});
inline const auto expected_cs_analytic_ion = torch::tensor({1.2356270E-5f, 1.1261894E-5f, 1.0343668E-5f});

inline const auto table_K = torch::arange(100.f, 1E+6f, 9999.f);
inline const auto table_q = 0.05f * table_K;