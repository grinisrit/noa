// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <memory>  // std::unique_ptr
#include <stdexcept>

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/MemoryOperations.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/MultiDeviceMemoryOperations.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/reduce.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/CudaSupportMissing.h>

namespace noa::TNL {
namespace Algorithms {

template< typename Element, typename Index >
void
MemoryOperations< Devices::Cuda >::construct( Element* data, Index size )
{
   TNL_ASSERT_TRUE( data, "Attempted to create elements through a nullptr." );
   auto kernel = [ data ] __cuda_callable__( Index i )
   {
      // placement-new
      ::new( (void*) ( data + i ) ) Element();
   };
   ParallelFor< Devices::Cuda >::exec( (Index) 0, size, kernel );
}

template< typename Element, typename Index, typename... Args >
void
MemoryOperations< Devices::Cuda >::construct( Element* data, Index size, const Args&... args )
{
   TNL_ASSERT_TRUE( data, "Attempted to create elements through a nullptr." );
   // NOTE: nvcc does not allow __cuda_callable__ lambdas with a variadic capture
   auto kernel = [ data ] __cuda_callable__( Index i, Args... args )
   {
      // placement-new
      // (note that args are passed by value to the constructor, not via
      // std::forward or even by reference, since move-semantics does not apply for
      // the construction of multiple elements and pass-by-reference cannot be used
      // with CUDA kernels)
      ::new( (void*) ( data + i ) ) Element( args... );
   };
   ParallelFor< Devices::Cuda >::exec( (Index) 0, size, kernel, args... );
}

template< typename Element, typename Index >
void
MemoryOperations< Devices::Cuda >::destruct( Element* data, Index size )
{
   TNL_ASSERT_TRUE( data, "Attempted to destroy data through a nullptr." );
   auto kernel = [ data ] __cuda_callable__( Index i )
   {
      ( data + i )->~Element();
   };
   ParallelFor< Devices::Cuda >::exec( (Index) 0, size, kernel );
}

template< typename Element >
__cuda_callable__
void
MemoryOperations< Devices::Cuda >::setElement( Element* data, const Element& value )
{
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
#ifdef __CUDA_ARCH__
   *data = value;
#else
   // NOTE: calling `MemoryOperations< Devices::Cuda >::set( data, value, 1 );`
   // does not work here due to `#ifdef __CUDA_ARCH__` above. It would involve
   // launching a CUDA kernel with an extended lambda, which would be discarded
   // by nvcc (never called).
   MultiDeviceMemoryOperations< Devices::Cuda, void >::copy( data, &value, 1 );
#endif
}

template< typename Element >
__cuda_callable__
Element
MemoryOperations< Devices::Cuda >::getElement( const Element* data )
{
   TNL_ASSERT_TRUE( data, "Attempted to get data through a nullptr." );
#ifdef __CUDA_ARCH__
   return *data;
#else
   Element result;
   MultiDeviceMemoryOperations< void, Devices::Cuda >::copy( &result, data, 1 );
   return result;
#endif
}

template< typename Element, typename Index >
void
MemoryOperations< Devices::Cuda >::set( Element* data, const Element& value, Index size )
{
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( data, "Attempted to set data through a nullptr." );
   auto kernel = [ data, value ] __cuda_callable__( Index i )
   {
      data[ i ] = value;
   };
   ParallelFor< Devices::Cuda >::exec( (Index) 0, size, kernel );
}

template< typename DestinationElement, typename SourceElement, typename Index >
void
MemoryOperations< Devices::Cuda >::copy( DestinationElement* destination, const SourceElement* source, Index size )
{
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( destination, "Attempted to copy data to a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to copy data from a nullptr." );

   // our ParallelFor kernel is faster than cudaMemcpy
   auto kernel = [ destination, source ] __cuda_callable__( Index i )
   {
      destination[ i ] = source[ i ];
   };
   ParallelFor< Devices::Cuda >::exec( (Index) 0, size, kernel );
}

template< typename DestinationElement, typename Index, typename SourceIterator >
void
MemoryOperations< Devices::Cuda >::copyFromIterator( DestinationElement* destination,
                                                     Index destinationSize,
                                                     SourceIterator first,
                                                     SourceIterator last )
{
   using BaseType = typename std::remove_cv< DestinationElement >::type;
   const int buffer_size = TNL::min( Cuda::getTransferBufferSize() / sizeof( BaseType ), destinationSize );
   std::unique_ptr< BaseType[] > buffer{ new BaseType[ buffer_size ] };
   Index copiedElements = 0;
   while( copiedElements < destinationSize && first != last ) {
      Index i = 0;
      while( i < buffer_size && first != last )
         buffer[ i++ ] = *first++;
      MultiDeviceMemoryOperations< Devices::Cuda, void >::copy( &destination[ copiedElements ], buffer.get(), i );
      copiedElements += i;
   }
   if( first != last )
      throw std::length_error( "Source iterator is larger than the destination array." );
}

template< typename Element1, typename Element2, typename Index >
bool
MemoryOperations< Devices::Cuda >::compare( const Element1* destination, const Element2* source, Index size )
{
   if( size == 0 )
      return true;
   TNL_ASSERT_TRUE( destination, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to compare data through a nullptr." );

   auto fetch = [ = ] __cuda_callable__( Index i ) -> bool
   {
      return destination[ i ] == source[ i ];
   };
   return reduce< Devices::Cuda >( (Index) 0, size, fetch, std::logical_and<>{}, true );
}

}  // namespace Algorithms
}  // namespace noa::TNL
