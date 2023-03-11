// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <climits>
#include <type_traits>

namespace noa::TNL {

/**
 * \brief Function takes a value and swaps its endianness.
 *
 * Reference: https://stackoverflow.com/a/4956493
 */
template< typename T >
T
swapEndianness( T u )
{
   static_assert( CHAR_BIT == 8, "CHAR_BIT != 8" );
   static_assert( std::is_fundamental< T >::value, "swap_endian works only for fundamental types" );

   union
   {
      T u;
      unsigned char u8[ sizeof( T ) ];
   } source, dest;

   source.u = u;

   for( std::size_t k = 0; k < sizeof( T ); k++ )
      dest.u8[ k ] = source.u8[ sizeof( T ) - k - 1 ];

   return dest.u;
}

/**
 * \brief Function returns `true` iff the system executing the program is little endian.
 */
inline bool
isLittleEndian()
{
   const unsigned int tmp1 = 1;
   const auto* tmp2 = reinterpret_cast< const unsigned char* >( &tmp1 );
   return *tmp2 != 0;
}

/**
 * \brief Function takes a value and returns its big endian representation.
 */
template< typename T >
T
forceBigEndian( T value )
{
   static bool swap = isLittleEndian();
   if( swap )
      return swapEndianness( value );
   return value;
}

}  // namespace noa::TNL
