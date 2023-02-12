// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

namespace noa::TNL {
namespace Meshes {
namespace Templates {

/**
 * One of the possible implementation of the conjuction operator.
 *
 * This one is taken from https://en.cppreference.com/w/cpp/types/conjunction
 */

template< class... >
struct conjuction : std::true_type
{};

template< class Type >
struct conjuction< Type > : Type
{};

template< class Head, class... Tail >
struct conjuction< Head, Tail... > : std::conditional_t< bool( Head::value ), conjuction< Tail... >, Head >
{};

template< class... Types >
constexpr bool conjunction_v = conjuction< Types... >::value;

/**
 * One of the possible implementation of the conjuction operator.
 *
 * This one is taken from https://en.cppreference.com/w/cpp/types/disjunction
 */

template< class... >
struct disjunction : std::false_type
{};

template< class Type >
struct disjunction< Type > : Type
{};

template< class Head, class... Tail >
struct disjunction< Head, Tail... > : std::conditional_t< bool( Head::value ), Head, disjunction< Tail... > >
{};

template< class... Types >
constexpr bool disjunction_v = disjunction< Types... >::value;

}  // namespace Templates
}  // namespace Meshes
}  // namespace noa::TNL
