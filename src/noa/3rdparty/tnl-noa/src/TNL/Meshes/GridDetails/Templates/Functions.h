// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/StaticVector.h>
#include <type_traits>

namespace noa::TNL {
namespace Meshes {
namespace Templates {

constexpr size_t
pow( size_t Value, size_t Power )
{
   size_t result = 1;

   for( size_t i = 0; i < Power; i++ )
      result *= Value;

   return result;
}

template< typename Index >
constexpr Index
product( Index from, Index to )
{
   Index result = 1;

   if( from <= to )
      for( Index i = from; i <= to; i++ )
         result *= i;

   return result;
}

template< typename Index >
constexpr Index
combination( Index k, Index n )
{
   return product< Index >( k + 1, n ) / product< Index >( 1, n - k );
}

/**
 * @brief A help method to calculate collapsed index the next way:
 *        base ^ 0 * (power_0 + base/2) + base ^ 1 * (power_1 + base/2)
 *
 * @tparam Powers
 * @param base - base value of the series
 * @param powers - powers values
 * @return constexpr int
 */
template< typename Index, int Size >
constexpr Index
makeCollapsedIndex( const Index base, const TNL::Containers::StaticVector< Size, Index > powers )
{
   Index index = 0;
   Index currentBase = 1;
   Index halfBase = base >> 1;

   for( Index i = 0; i < powers.getSize(); i++ ) {
      index += ( powers[ i ] + halfBase ) * currentBase;
      currentBase *= base;
   }

   return index;
}

template< typename Index, Index... Powers >
constexpr Index
makeCollapsedIndex( const int base )
{
   Index index = 0;
   Index currentBase = 1;
   Index halfBase = base >> 1;

   for( const auto x : { Powers... } ) {
      index += ( x + halfBase ) * currentBase;
      currentBase *= base;
   }

   return index;
}

template< typename Index >
constexpr Index
firstKCombinationSum( Index k, Index n )
{
   if( k == 0 )
      return 0;

   if( k == n )
      return ( 1 << n ) - 1;

   Index result = 0;

   // Fraction simplification of k-combination
   for( Index i = 0; i < k; i++ )
      result += combination( i, n );

   return result;
}

constexpr bool
isInClosedInterval( int lower, int value, int upper )
{
   return lower <= value && value <= upper;
}

constexpr bool
isInLeftClosedRightOpenInterval( int lower, int value, int upper )
{
   return lower <= value && value < upper;
}

}  // namespace Templates
}  // namespace Meshes
}  // namespace noa::TNL
