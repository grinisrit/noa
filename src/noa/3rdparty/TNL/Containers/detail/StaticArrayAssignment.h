// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/TypeTraits.h>
#include <noa/3rdparty/TNL/Algorithms/unrolledFor.h>

namespace noa::TNL {
namespace Containers {
namespace detail {

template< typename StaticArray,
          typename T,
          bool isStaticArrayType = IsStaticArrayType< T >::value >
struct StaticArrayAssignment;

/**
 * \brief Specialization for array-array assignment.
 */
template< typename StaticArray,
          typename T >
struct StaticArrayAssignment< StaticArray, T, true >
{
   static constexpr void assign( StaticArray& a, const T& v )
   {
      static_assert( StaticArray::getSize() == T::getSize(),
                     "Cannot assign static arrays with different size." );
      Algorithms::unrolledFor< int, 0, StaticArray::getSize() >(
         [&] ( int i ) mutable {
            a[ i ] = v[ i ];
         }
      );
   }
};

/**
 * \brief Specialization for array-value assignment for other types. We assume
 * that T is convertible to StaticArray::ValueType.
 */
template< typename StaticArray,
          typename T >
struct StaticArrayAssignment< StaticArray, T, false >
{
   static constexpr void assign( StaticArray& a, const T& v )
   {
      Algorithms::unrolledFor< int, 0, StaticArray::getSize() >(
         [&] ( int i ) mutable {
            a[ i ] = v;
         }
      );
   }
};

} // namespace detail
} // namespace Containers
} // namespace noa::TNL
