// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <stdexcept>
#include <algorithm>  // std::copy, std::equal

#include <noa/3rdparty/TNL/Algorithms/MemoryOperations.h>
#include <noa/3rdparty/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/TNL/Algorithms/reduce.h>

namespace noa::TNL {
namespace Algorithms {

template< typename Element, typename Index >
void
MemoryOperations< Devices::Host >::
construct( Element* data,
           const Index size )
{
   TNL_ASSERT_TRUE( data, "Attempted to create elements through a nullptr." );
   auto kernel = [data]( Index i )
   {
      // placement-new
      ::new( (void*) (data + i) ) Element();
   };
   ParallelFor< Devices::Host >::exec( (Index) 0, size, kernel );
}

template< typename Element, typename Index, typename... Args >
void
MemoryOperations< Devices::Host >::
construct( Element* data,
           const Index size,
           const Args&... args )
{
   TNL_ASSERT_TRUE( data, "Attempted to create elements through a nullptr." );
   auto kernel = [data, &args...]( Index i )
   {
      // placement-new
      // (note that args are passed by reference to the constructor, not via
      // std::forward since move-semantics does not apply for the construction
      // of multiple elements)
      ::new( (void*) (data + i) ) Element( args... );
   };
   ParallelFor< Devices::Host >::exec( (Index) 0, size, kernel );
}

template< typename Element, typename Index >
void
MemoryOperations< Devices::Host >::
destruct( Element* data,
          const Index size )
{
   TNL_ASSERT_TRUE( data, "Attempted to destroy data through a nullptr." );
   auto kernel = [data]( Index i )
   {
      (data + i)->~Element();
   };
   ParallelFor< Devices::Host >::exec( (Index) 0, size, kernel );
}

template< typename Element >
__cuda_callable__ // only to avoid nvcc warning
void
MemoryOperations< Devices::Host >::
setElement( Element* data,
            const Element& value )
{
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
   *data = value;
}

template< typename Element >
__cuda_callable__ // only to avoid nvcc warning
Element
MemoryOperations< Devices::Host >::
getElement( const Element* data )
{
   TNL_ASSERT_TRUE( data, "Attempted to get data through a nullptr." );
   return *data;
}

template< typename Element, typename Index >
void
MemoryOperations< Devices::Host >::
set( Element* data,
     const Element& value,
     const Index size )
{
   if( size == 0 ) return;
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
   auto kernel = [data, value]( Index i )
   {
      data[ i ] = value;
   };
   ParallelFor< Devices::Host >::exec( (Index) 0, size, kernel );
}

template< typename DestinationElement,
          typename SourceElement,
          typename Index >
void
MemoryOperations< Devices::Host >::
copy( DestinationElement* destination,
      const SourceElement* source,
      const Index size )
{
   if( size == 0 ) return;
   TNL_ASSERT_TRUE( destination, "Attempted to copy data to a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to copy data from a nullptr." );

   // our ParallelFor version is faster than std::copy iff we use more than 1 thread
   if( Devices::Host::isOMPEnabled() && Devices::Host::getMaxThreadsCount() > 1 ) {
      auto kernel = [destination, source]( Index i )
      {
         destination[ i ] = source[ i ];
      };
      ParallelFor< Devices::Host >::exec( (Index) 0, size, kernel );
   }
   else {
      // std::copy usually uses std::memcpy for TriviallyCopyable types
      std::copy( source, source + size, destination );
   }
}

template< typename DestinationElement,
          typename Index,
          typename SourceIterator >
void
MemoryOperations< Devices::Host >::
copyFromIterator( DestinationElement* destination,
                  Index destinationSize,
                  SourceIterator first,
                  SourceIterator last )
{
   MemoryOperations< Devices::Sequential >::copyFromIterator( destination, destinationSize, first, last );
}

template< typename DestinationElement,
          typename SourceElement,
          typename Index >
bool
MemoryOperations< Devices::Host >::
compare( const DestinationElement* destination,
         const SourceElement* source,
         const Index size )
{
   if( size == 0 ) return true;
   TNL_ASSERT_TRUE( destination, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to compare data through a nullptr." );

   if( Devices::Host::isOMPEnabled() && Devices::Host::getMaxThreadsCount() > 1 ) {
      auto fetch = [destination, source] ( Index i ) -> bool { return destination[ i ] == source[ i ]; };
      return reduce< Devices::Host >( ( Index ) 0, size, fetch, std::logical_and<>{}, true );
   }
   else {
      // sequential algorithm can return as soon as it finds a mismatch
      return std::equal( source, source + size, destination );
   }
}

} // namespace Algorithms
} // namespace noa::TNL
