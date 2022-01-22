// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/TypeTraits.h>
#include <noa/3rdparty/TNL/Algorithms/MemoryOperations.h>
#include <noa/3rdparty/TNL/Algorithms/MultiDeviceMemoryOperations.h>

namespace noaTNL {
namespace Containers {
namespace detail {

template< typename Array,
          typename T,
          bool isArrayType = IsArrayType< T >::value >
struct ArrayAssignment;

/**
 * \brief Specialization for array-array assignment with containers implementing
 * getArrayData method.
 */
template< typename Array,
          typename T >
struct ArrayAssignment< Array, T, true >
{
   static void resize( Array& a, const T& t )
   {
      a.setSize( t.getSize() );
   }

   static void assign( Array& a, const T& t )
   {
      TNL_ASSERT_EQ( a.getSize(), ( decltype( a.getSize() ) ) t.getSize(), "The sizes of the arrays must be equal." );
      // skip assignment of empty arrays
      if( a.getSize() == 0 )
         return;
      Algorithms::MultiDeviceMemoryOperations< typename Array::DeviceType, typename T::DeviceType >::template
         copy< typename Array::ValueType, typename T::ValueType, typename Array::IndexType >
         ( a.getArrayData(), t.getArrayData(), t.getSize() );
   }
};

/**
 * \brief Specialization for array-value assignment for other types. We assume
 * that T is convertible to Array::ValueType.
 */
template< typename Array,
          typename T >
struct ArrayAssignment< Array, T, false >
{
   static void resize( Array& a, const T& t )
   {
   }

   static void assign( Array& a, const T& t )
   {
      // skip assignment to an empty array
      if( a.getSize() == 0 )
         return;
      Algorithms::MemoryOperations< typename Array::DeviceType >::template
         set< typename Array::ValueType, typename Array::IndexType >
         ( a.getArrayData(), ( typename Array::ValueType ) t, a.getSize() );
   }
};

} // namespace detail
} // namespace Containers
} // namespace noaTNL
