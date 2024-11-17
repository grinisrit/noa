// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/MemoryOperations.h>

namespace noa::TNL {
namespace Algorithms {

template< typename Element, typename Index >
__cuda_callable__
void
MemoryOperations< Devices::Sequential >::construct( Element* data, Index size )
{
   TNL_ASSERT_TRUE( data, "Attempted to create elements through a nullptr." );
   for( Index i = 0; i < size; i++ )
      // placement-new
      ::new( (void*) ( data + i ) ) Element();
}

template< typename Element, typename Index, typename... Args >
__cuda_callable__
void
MemoryOperations< Devices::Sequential >::construct( Element* data, Index size, const Args&... args )
{
   TNL_ASSERT_TRUE( data, "Attempted to create elements through a nullptr." );
   for( Index i = 0; i < size; i++ )
      // placement-new
      // (note that args are passed by reference to the constructor, not via
      // std::forward since move-semantics does not apply for the construction
      // of multiple elements)
      ::new( (void*) ( data + i ) ) Element( args... );
}

template< typename Element, typename Index >
__cuda_callable__
void
MemoryOperations< Devices::Sequential >::destruct( Element* data, Index size )
{
   TNL_ASSERT_TRUE( data, "Attempted to destroy elements through a nullptr." );
   for( Index i = 0; i < size; i++ )
      ( data + i )->~Element();
}

template< typename Element >
__cuda_callable__
void
MemoryOperations< Devices::Sequential >::setElement( Element* data, const Element& value )
{
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
   *data = value;
}

template< typename Element >
__cuda_callable__
Element
MemoryOperations< Devices::Sequential >::getElement( const Element* data )
{
   TNL_ASSERT_TRUE( data, "Attempted to get data through a nullptr." );
   return *data;
}

template< typename Element, typename Index >
__cuda_callable__
void
MemoryOperations< Devices::Sequential >::set( Element* data, const Element& value, Index size )
{
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
   for( Index i = 0; i < size; i++ )
      data[ i ] = value;
}

template< typename DestinationElement, typename SourceElement, typename Index >
__cuda_callable__
void
MemoryOperations< Devices::Sequential >::copy( DestinationElement* destination, const SourceElement* source, Index size )
{
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( destination, "Attempted to copy data to a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to copy data from a nullptr." );

   for( Index i = 0; i < size; i++ )
      destination[ i ] = source[ i ];
}

template< typename DestinationElement, typename Index, typename SourceIterator >
void
MemoryOperations< Devices::Sequential >::copyFromIterator( DestinationElement* destination,
                                                           Index destinationSize,
                                                           SourceIterator first,
                                                           SourceIterator last )
{
   Index i = 0;
   while( i < destinationSize && first != last )
      destination[ i++ ] = *first++;
   if( first != last )
      throw std::length_error( "Source iterator is larger than the destination array." );
}

template< typename Element1, typename Element2, typename Index >
__cuda_callable__
bool
MemoryOperations< Devices::Sequential >::compare( const Element1* destination, const Element2* source, Index size )
{
   if( size == 0 )
      return true;
   TNL_ASSERT_TRUE( destination, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to compare data through a nullptr." );

   for( Index i = 0; i < size; i++ )
      if( ! ( destination[ i ] == source[ i ] ) )
         return false;
   return true;
}

}  // namespace Algorithms
}  // namespace noa::TNL
