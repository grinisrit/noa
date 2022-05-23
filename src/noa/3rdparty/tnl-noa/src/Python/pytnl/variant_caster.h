#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mpark/variant.hpp>   // backport of std::variant from C++17

namespace pybind11 { namespace detail {

// add specialization for concrete variant type
// (variant_caster is implemented in pybind11 and used for C++17's std::variant casting)
template<class... Args> struct type_caster<mpark::variant<Args...>>
    : variant_caster<mpark::variant<Args...>> {};

}} // namespace pybind11::detail
