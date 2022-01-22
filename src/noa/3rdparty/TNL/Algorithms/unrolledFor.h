// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

namespace noaTNL {
namespace Algorithms {

namespace detail {

// special dispatch for empty loop
template< typename Index, Index begin, Index end, Index unrollFactor,  typename Func >
constexpr std::enable_if_t< (begin >= end) >
unrolled_for_dispatch( Func&& f )
{}

// special dispatch for 1 iteration
template< typename Index, Index begin, Index end, Index unrollFactor,  typename Func >
constexpr std::enable_if_t< (begin < end && end - begin == 1) >
unrolled_for_dispatch( Func&& f )
{
   f( begin );
}

// specialization for unrolling short loops (at least 2, but at most unrollFactor iterations)
template< typename Index, Index begin, Index end, Index unrollFactor,  typename Func >
constexpr std::enable_if_t< (begin < end && end - begin >= 2 && end - begin <= unrollFactor) >
unrolled_for_dispatch( Func&& f )
{
   constexpr Index mid = begin + (end - begin) / 2;
   unrolled_for_dispatch< Index, begin, mid, unrollFactor >( std::forward< Func >( f ) );
   unrolled_for_dispatch< Index, mid, end, unrollFactor >( std::forward< Func >( f ) );
}

// specialization for long loops - normal for-loop
template< typename Index, Index begin, Index end, Index unrollFactor,  typename Func >
constexpr std::enable_if_t< (begin < end && end - begin > 1 && end - begin > unrollFactor) >
unrolled_for_dispatch( Func&& f )
{
   for( Index i = begin; i < end; i++ )
      f( i );
}

} // namespace detail

/**
 * \brief Generic for-loop with explicit unrolling.
 *
 * \e unrolledFor performs explicit loop unrolling of short loops which can
 * improve performance in some cases. The bounds of the for-loop must be constant
 * (i.e. known at the compile time). Loops longer than \e unrollFactor are not
 * unrolled and executed as a normal for-loop.
 *
 * The unroll factor is configurable, but note that full unrolling does not
 * make sense for very long loops. It might even trigger the compiler's limit
 * on recursive template instantiation. Also note that the compiler will (at
 * least partially) unroll loops with static bounds anyway.
 *
 * \tparam Index is the type of the loop indices.
 * \tparam begin is the left bound of the iteration range `[begin, end)`.
 * \tparam end is the right bound of the iteration range `[begin, end)`.
 * \tparam unrollFactor is the maximum length of loops to fully unroll via
 *    recursive template instantiation.
 * \tparam Func is the type of the functor (it is usually deduced from the
 *    argument used in the function call).
 *
 * \param f is the functor to be called in each iteration.
 *
 * \par Example
 * \include Algorithms/unrolledForExample.cpp
 * \par Output
 * \include unrolledForExample.out
 */
template< typename Index, Index begin, Index end, Index unrollFactor = 8,  typename Func >
constexpr void unrolledFor( Func&& f )
{
   detail::unrolled_for_dispatch< Index, begin, end, unrollFactor >( std::forward< Func >( f ) );
}

} // namespace Algorithms
} // namespace noaTNL
