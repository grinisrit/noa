// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/TypeInfo.h>
#include <noa/3rdparty/TNL/Math.h>
#include <noa/3rdparty/TNL/Containers/StaticArray.h>
#include <noa/3rdparty/TNL/Containers/detail/StaticArrayAssignment.h>
#include <noa/3rdparty/TNL/Algorithms/unrolledFor.h>

namespace noaTNL {
namespace Containers {

namespace detail {

// StaticArrayComparator does static loop unrolling of array comparison
template< int Size, typename LeftValue, typename RightValue, int Index >
struct StaticArrayComparator
{
   __cuda_callable__
   static bool EQ( const StaticArray< Size, LeftValue >& left,
                   const StaticArray< Size, RightValue >& right )
   {
      if( left[ Index ] == right[ Index ] )
         return StaticArrayComparator< Size, LeftValue, RightValue, Index + 1 >::EQ( left, right );
      return false;
   }
};

template< int Size, typename LeftValue, typename RightValue >
struct StaticArrayComparator< Size, LeftValue, RightValue, Size >
{
   __cuda_callable__
   static bool EQ( const StaticArray< Size, LeftValue >& left,
                   const StaticArray< Size, RightValue >& right )
   {
      return true;
   }
};

////
// Static array sort does static loop unrolling of array sort.
// It performs static variant of bubble sort as follows:
//
// for( int k = Size - 1; k > 0; k--)
//   for( int i = 0; i < k; i++ )
//      if( data[ i ] > data[ i+1 ] )
//         swap( data[ i ], data[ i+1 ] );
template< int k, int i, typename Value >
struct StaticArraySort
{
   __cuda_callable__
   static void exec( Value* data ) {
      if( data[ i ] > data[  i + 1 ] )
         swap( data[ i ], data[ i+1 ] );
      StaticArraySort< k, i + 1, Value >::exec( data );
   }
};

template< int k, typename Value >
struct StaticArraySort< k, k, Value >
{
   __cuda_callable__
   static void exec( Value* data ) {
      StaticArraySort< k - 1, 0, Value >::exec( data );
   }
};

template< typename Value >
struct StaticArraySort< 0, 0, Value >
{
   __cuda_callable__
   static void exec( Value* data ) {}
};

} // namespace detail


template< int Size, typename Value >
__cuda_callable__
constexpr int StaticArray< Size, Value >::getSize()
{
   return Size;
}

template< int Size, typename Value >
__cuda_callable__
StaticArray< Size, Value >::StaticArray()
{
}

template< int Size, typename Value >
   template< typename _unused >
__cuda_callable__
StaticArray< Size, Value >::StaticArray( const Value v[ Size ] )
{
   Algorithms::unrolledFor< int, 0, Size >(
      [&] ( int i ) mutable {
         (*this)[ i ] = v[ i ];
      }
   );
}

template< int Size, typename Value >
__cuda_callable__
StaticArray< Size, Value >::StaticArray( const Value& v )
{
   Algorithms::unrolledFor< int, 0, Size >(
      [&] ( int i ) mutable {
         (*this)[ i ] = v;
      }
   );
}

template< int Size, typename Value >
__cuda_callable__
StaticArray< Size, Value >::StaticArray( const StaticArray< Size, Value >& v )
{
   Algorithms::unrolledFor< int, 0, Size >(
      [&] ( int i ) mutable {
         (*this)[ i ] = v[ i ];
      }
   );
}

template< int Size, typename Value >
__cuda_callable__
StaticArray< Size, Value >::StaticArray( const std::initializer_list< Value > &elems)
{
   auto it = elems.begin();
   for( int i = 0; i < getSize(); i++ )
      data[ i ] = *it++;
}

template< int Size, typename Value >
 __cuda_callable__
StaticArray< Size, Value >::StaticArray( const Value& v1, const Value& v2 )
{
   static_assert( Size == 2, "This constructor can be called only for arrays with Size = 2." );
   data[ 0 ] = v1;
   data[ 1 ] = v2;
}

template< int Size, typename Value >
 __cuda_callable__
StaticArray< Size, Value >::StaticArray( const Value& v1, const Value& v2, const Value& v3 )
{
   static_assert( Size == 3, "This constructor can be called only for arrays with Size = 3." );
   data[ 0 ] = v1;
   data[ 1 ] = v2;
   data[ 2 ] = v3;
}

template< int Size, typename Value >
__cuda_callable__
Value* StaticArray< Size, Value >::getData()
{
   return data;
}

template< int Size, typename Value >
__cuda_callable__
const Value* StaticArray< Size, Value >::getData() const
{
   return data;
}

template< int Size, typename Value >
__cuda_callable__
const Value& StaticArray< Size, Value >::operator[]( int i ) const
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, Size, "Element index is out of bounds." );
   return data[ i ];
}

template< int Size, typename Value >
__cuda_callable__
Value& StaticArray< Size, Value >::operator[]( int i )
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, Size, "Element index is out of bounds." );
   return data[ i ];
}

template< int Size, typename Value >
__cuda_callable__
const Value& StaticArray< Size, Value >::operator()( int i ) const
{
   return operator[]( i );
}

template< int Size, typename Value >
__cuda_callable__
Value& StaticArray< Size, Value >::operator()( int i )
{
   return operator[]( i );
}

template< int Size, typename Value >
__cuda_callable__
Value& StaticArray< Size, Value >::x()
{
   return data[ 0 ];
}

template< int Size, typename Value >
__cuda_callable__
const Value& StaticArray< Size, Value >::x() const
{
   return data[ 0 ];
}

template< int Size, typename Value >
__cuda_callable__
Value& StaticArray< Size, Value >::y()
{
   static_assert( Size > 1, "Cannot call StaticArray< Size, Value >::y() for arrays with Size < 2." );
   return data[ 1 ];
}

template< int Size, typename Value >
__cuda_callable__
const Value& StaticArray< Size, Value >::y() const
{
   static_assert( Size > 1, "Cannot call StaticArray< Size, Value >::y() for arrays with Size < 2." );
   return data[ 1 ];
}

template< int Size, typename Value >
__cuda_callable__
Value& StaticArray< Size, Value >::z()
{
   static_assert( Size > 1, "Cannot call StaticArray< Size, Value >::z() for arrays with Size < 3." );
   return data[ 2 ];
}

template< int Size, typename Value >
__cuda_callable__
const Value& StaticArray< Size, Value >::z() const
{
   static_assert( Size > 1, "Cannot call StaticArray< Size, Value >::z() for arrays with Size < 3." );
   return data[ 2 ];
}

template< int Size, typename Value >
__cuda_callable__
StaticArray< Size, Value >& StaticArray< Size, Value >::operator=( const StaticArray< Size, Value >& array )
{
   Algorithms::unrolledFor< int, 0, Size >(
      [&] ( int i ) mutable {
         (*this)[ i ] = array[ i ];
      }
   );
   return *this;
}

template< int Size, typename Value >
   template< typename T >
__cuda_callable__
StaticArray< Size, Value >& StaticArray< Size, Value >::operator=( const T& v )
{
   detail::StaticArrayAssignment< StaticArray, T >::assign( *this, v );
   return *this;
}

template< int Size, typename Value >
   template< typename Array >
__cuda_callable__
bool StaticArray< Size, Value >::operator==( const Array& array ) const
{
   return detail::StaticArrayComparator< Size, Value, typename Array::ValueType, 0 >::EQ( *this, array );
}

template< int Size, typename Value >
   template< typename Array >
__cuda_callable__
bool StaticArray< Size, Value >::operator!=( const Array& array ) const
{
   return ! this->operator==( array );
}

template< int Size, typename Value >
   template< typename OtherValue >
__cuda_callable__
StaticArray< Size, Value >::
operator StaticArray< Size, OtherValue >() const
{
   StaticArray< Size, OtherValue > aux;
   aux.operator=( *this );
   return aux;
}

template< int Size, typename Value >
__cuda_callable__
void StaticArray< Size, Value >::setValue( const ValueType& val )
{
   Algorithms::unrolledFor< int, 0, Size >(
      [&] ( int i ) mutable {
         (*this)[ i ] = val;
      }
   );
}

template< int Size, typename Value >
bool StaticArray< Size, Value >::save( File& file ) const
{
   file.save( getData(), Size );
   return true;
}

template< int Size, typename Value >
bool StaticArray< Size, Value >::load( File& file)
{
   file.load( getData(), Size );
   return true;
}

template< int Size, typename Value >
void StaticArray< Size, Value >::sort()
{
   detail::StaticArraySort< Size - 1, 0, Value >::exec( getData() );
}

template< int Size, typename Value >
std::ostream& StaticArray< Size, Value >::write( std::ostream& str, const char* separator ) const
{
   for( int i = 0; i < Size - 1; i++ )
      str << data[ i ] << separator;
   str << data[ Size - 1 ];
   return str;
}

template< int Size, typename Value >
std::ostream& operator<<( std::ostream& str, const StaticArray< Size, Value >& a )
{
   str << "[ ";
   a.write( str, ", " );
   str << " ]";
   return str;
}

// Serialization of arrays into binary files.
template< int Size, typename Value >
File& operator<<( File& file, const StaticArray< Size, Value >& array )
{
   for( int i = 0; i < Size; i++ )
      file.save( &array[ i ] );
   return file;
}

template< int Size, typename Value >
File& operator<<( File&& file, const StaticArray< Size, Value >& array )
{
   File& f = file;
   return f << array;
}

// Deserialization of arrays from binary files.
template< int Size, typename Value >
File& operator>>( File& file, StaticArray< Size, Value >& array )
{
   for( int i = 0; i < Size; i++ )
      file.load( &array[ i ] );
   return file;
}

template< int Size, typename Value >
File& operator>>( File&& file, StaticArray< Size, Value >& array )
{
   File& f = file;
   return f >> array;
}

} // namespace Containers
} // namespace noaTNL
