// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/detail/Contains.h>

namespace TNL {
namespace Algorithms {

/**
 * \brief Checks if an array/vector/view contains an element with given value.
 *
 * By default, all elements of the array are checked. If \e begin or \e end is
 * set to a non-zero value, only elements in the sub-interval `[begin, end)` are
 * checked.
 *
 * \param array The array to be searched.
 * \param value The value to be checked.
 * \param begin The beginning of the array sub-interval. It is 0 by default.
 * \param end The end of the array sub-interval. The default value is 0 which
 *            is, however, replaced with the array size.
 * \return `true` if there is _at least one_ element in the sub-interval
 *         `[begin, end)` which has the value \e value.
 */
template< typename Array >
bool
contains( const Array& array,
          typename Array::ValueType value,
          typename Array::IndexType begin = 0,
          typename Array::IndexType end = 0 )
{
   TNL_ASSERT_TRUE( array.getData(), "Attempted to check a value of an empty array." );
   if( end == 0 )
      end = array.getSize();
   return detail::Contains< typename Array::DeviceType >()( array.getData() + begin, end - begin, value );
}

/**
 * \brief Checks if all elements of an array/vector/view have the given value.
 *
 * By default, all elements of the array are checked. If \e begin or \e end is
 * set to a non-zero value, only elements in the sub-interval `[begin, end)` are
 * checked.
 *
 * \param array The array to be searched.
 * \param value The value to be checked.
 * \param begin The beginning of the array sub-interval. It is 0 by default.
 * \param end The end of the array sub-interval. The default value is 0 which
 *            is, however, replaced with the array size.
 * \return `true` if _all_ elements in the sub-interval `[begin, end)` have the
 *         same value \e value.
 */
template< typename Array >
bool
containsOnlyValue( const Array& array,
                   typename Array::ValueType value,
                   typename Array::IndexType begin = 0,
                   typename Array::IndexType end = 0 )
{
   TNL_ASSERT_TRUE( array.getData(), "Attempted to check a value of an empty array." );
   if( end == 0 )
      end = array.getSize();
   return detail::ContainsOnlyValue< typename Array::DeviceType >()( array.getData() + begin, end - begin, value );
}

} // namespace Algorithms
} // namespace TNL
