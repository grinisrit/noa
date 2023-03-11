// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>
#include <type_traits>

namespace noa::TNL {
namespace Algorithms {

namespace detail {

template< typename Index, Index begin, typename Func, Index... idx, typename... ArgTypes >
constexpr void
static_for_impl( Func&& f, std::integer_sequence< Index, idx... >, ArgTypes&&... args )
{
   // C++17 fold expression using the comma operator
   ( f( std::integral_constant< Index, begin + idx >{}, std::forward< ArgTypes >( args )... ), ... );
}

}  // namespace detail

/**
 * \brief Generic loop with constant bounds and indices usable in constant
 * expressions.
 *
 * \e staticFor is a generic C++17 implementation of a static for-loop using
 * \e constexpr functions and template metaprogramming. It is equivalent to
 * executing a function `f(i, args...)` for arguments `i` from the integral
 * range `[begin, end)`, but with the type \ref std::integral_constant rather
 * than `int` or `std::size_t` representing the indices. Hence, each index has
 * its own distinct C++ type and the \e value of the index can be deduced from
 * the type. The `args...` are additional user-supplied arguments that are
 * forwarded to the \e staticFor function.
 *
 * Also note that thanks to `constexpr` cast operator, the argument `i` can be
 * used in constant expressions and the \e staticFor function can be used from
 * the host code as well as CUDA kernels (TNL requires the
 * `--expt-relaxed-constexpr` parameter when compiled by `nvcc`).
 *
 * \tparam Index is the type of the loop indices.
 * \tparam begin is the left bound of the iteration range `[begin, end)`.
 * \tparam end is the right bound of the iteration range `[begin, end)`.
 * \tparam Func is the type of the functor (it is usually deduced from the
 *    argument used in the function call).
 * \tparam ArgTypes are the types of additional arguments passed to the
 *    function.
 *
 * \param f is the functor to be called in each iteration.
 * \param args are additional user-supplied arguments that are forwarded
 *    to each call of \e f.
 *
 * \par Example
 * \include Algorithms/staticForExample.cpp
 * \par Output
 * \include staticForExample.out
 */
template< typename Index, Index begin, Index end, typename Func, typename... ArgTypes >
constexpr void
staticFor( Func&& f, ArgTypes&&... args )
{
   if constexpr( begin < end ) {
      detail::static_for_impl< Index, begin >(
         std::forward< Func >( f ), std::make_integer_sequence< Index, end - begin >{}, std::forward< ArgTypes >( args )... );
   }
}

}  // namespace Algorithms
}  // namespace noa::TNL
