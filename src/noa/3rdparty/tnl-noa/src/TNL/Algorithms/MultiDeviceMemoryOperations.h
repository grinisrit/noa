// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/MemoryOperations.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/LaunchHelpers.h>  // getTransferBufferSize

namespace noa::TNL {
namespace Algorithms {

template< typename DestinationDevice, typename SourceDevice = DestinationDevice >
struct MultiDeviceMemoryOperations
{
   template< typename DestinationElement, typename SourceElement, typename Index >
   static void
   copy( DestinationElement* destination, const SourceElement* source, Index size )
   {
      // use DestinationDevice, unless it is void
      using Device = std::conditional_t< std::is_void< DestinationDevice >::value, SourceDevice, DestinationDevice >;
      MemoryOperations< Device >::copy( destination, source, size );
   }

   template< typename DestinationElement, typename SourceElement, typename Index >
   static bool
   compare( const DestinationElement* destination, const SourceElement* source, Index size )
   {
      // use DestinationDevice, unless it is void
      using Device = std::conditional_t< std::is_void< DestinationDevice >::value, SourceDevice, DestinationDevice >;
      return MemoryOperations< Device >::compare( destination, source, size );
   }
};

template< typename DeviceType >
struct MultiDeviceMemoryOperations< Devices::Cuda, DeviceType >
{
   template< typename DestinationElement, typename SourceElement, typename Index >
   static void
   copy( DestinationElement* destination, const SourceElement* source, Index size );

   template< typename DestinationElement, typename SourceElement, typename Index >
   static bool
   compare( const DestinationElement* destination, const SourceElement* source, Index size );
};

template< typename DeviceType >
struct MultiDeviceMemoryOperations< DeviceType, Devices::Cuda >
{
   template< typename DestinationElement, typename SourceElement, typename Index >
   static void
   copy( DestinationElement* destination, const SourceElement* source, Index size );

   template< typename Element1, typename Element2, typename Index >
   static bool
   compare( const Element1* destination, const Element2* source, Index size );
};

// CUDA <-> CUDA to disambiguate from partial specializations below
template<>
struct MultiDeviceMemoryOperations< Devices::Cuda, Devices::Cuda >
{
   template< typename DestinationElement, typename SourceElement, typename Index >
   static void
   copy( DestinationElement* destination, const SourceElement* source, Index size )
   {
      MemoryOperations< Devices::Cuda >::copy( destination, source, size );
   }

   template< typename DestinationElement, typename SourceElement, typename Index >
   static bool
   compare( const DestinationElement* destination, const SourceElement* source, Index size )
   {
      return MemoryOperations< Devices::Cuda >::compare( destination, source, size );
   }
};

/****
 * Operations CUDA -> Host
 */
template< typename DeviceType >
template< typename DestinationElement, typename SourceElement, typename Index >
void
MultiDeviceMemoryOperations< DeviceType, Devices::Cuda >::copy( DestinationElement* destination,
                                                                const SourceElement* source,
                                                                Index size )
{
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( destination, "Attempted to copy data to a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to copy data from a nullptr." );
#ifdef HAVE_CUDA
   if( std::is_same< std::remove_cv_t< DestinationElement >, std::remove_cv_t< SourceElement > >::value ) {
      if( cudaMemcpy( destination, source, size * sizeof( DestinationElement ), cudaMemcpyDeviceToHost ) != cudaSuccess )
         std::cerr << "Transfer of data from CUDA device to host failed." << std::endl;
      TNL_CHECK_CUDA_DEVICE;
   }
   else {
      using BaseType = std::remove_cv_t< SourceElement >;
      const int buffer_size = TNL::min( Cuda::getTransferBufferSize() / sizeof( BaseType ), size );
      std::unique_ptr< BaseType[] > buffer{ new BaseType[ buffer_size ] };
      Index i = 0;
      while( i < size ) {
         if( cudaMemcpy( (void*) buffer.get(),
                         (void*) &source[ i ],
                         TNL::min( size - i, buffer_size ) * sizeof( SourceElement ),
                         cudaMemcpyDeviceToHost )
             != cudaSuccess )
            std::cerr << "Transfer of data from CUDA device to host failed." << std::endl;
         TNL_CHECK_CUDA_DEVICE;
         int j = 0;
         while( j < buffer_size && i + j < size ) {
            destination[ i + j ] = buffer[ j ];
            j++;
         }
         i += j;
      }
   }
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename DeviceType >
template< typename Element1, typename Element2, typename Index >
bool
MultiDeviceMemoryOperations< DeviceType, Devices::Cuda >::compare( const Element1* destination,
                                                                   const Element2* source,
                                                                   Index size )
{
   if( size == 0 )
      return true;
   /***
    * Here, destination is on host and source is on CUDA device.
    */
   TNL_ASSERT_TRUE( destination, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_GE( size, (Index) 0, "Array size must be non-negative." );
#ifdef HAVE_CUDA
   const int buffer_size = TNL::min( Cuda::getTransferBufferSize() / sizeof( Element2 ), size );
   std::unique_ptr< Element2[] > host_buffer{ new Element2[ buffer_size ] };
   Index compared = 0;
   while( compared < size ) {
      const int transfer = TNL::min( size - compared, buffer_size );
      if( cudaMemcpy(
             (void*) host_buffer.get(), (void*) &source[ compared ], transfer * sizeof( Element2 ), cudaMemcpyDeviceToHost )
          != cudaSuccess )
         std::cerr << "Transfer of data from CUDA device to host failed." << std::endl;
      TNL_CHECK_CUDA_DEVICE;
      if( ! MemoryOperations< Devices::Host >::compare( &destination[ compared ], host_buffer.get(), transfer ) )
         return false;
      compared += transfer;
   }
   return true;
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

/****
 * Operations Host -> CUDA
 */
template< typename DeviceType >
template< typename DestinationElement, typename SourceElement, typename Index >
void
MultiDeviceMemoryOperations< Devices::Cuda, DeviceType >::copy( DestinationElement* destination,
                                                                const SourceElement* source,
                                                                Index size )
{
   if( size == 0 )
      return;
   TNL_ASSERT_TRUE( destination, "Attempted to copy data to a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to copy data from a nullptr." );
   TNL_ASSERT_GE( size, (Index) 0, "Array size must be non-negative." );
#ifdef HAVE_CUDA
   if( std::is_same< std::remove_cv_t< DestinationElement >, std::remove_cv_t< SourceElement > >::value ) {
      if( cudaMemcpy( destination, source, size * sizeof( DestinationElement ), cudaMemcpyHostToDevice ) != cudaSuccess )
         std::cerr << "Transfer of data from host to CUDA device failed." << std::endl;
      TNL_CHECK_CUDA_DEVICE;
   }
   else {
      const int buffer_size = TNL::min( Cuda::getTransferBufferSize() / sizeof( DestinationElement ), size );
      std::unique_ptr< DestinationElement[] > buffer{ new DestinationElement[ buffer_size ] };
      Index i = 0;
      while( i < size ) {
         int j = 0;
         while( j < buffer_size && i + j < size ) {
            buffer[ j ] = source[ i + j ];
            j++;
         }
         if( cudaMemcpy(
                (void*) &destination[ i ], (void*) buffer.get(), j * sizeof( DestinationElement ), cudaMemcpyHostToDevice )
             != cudaSuccess )
            std::cerr << "Transfer of data from host to CUDA device failed." << std::endl;
         TNL_CHECK_CUDA_DEVICE;
         i += j;
      }
   }
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename DeviceType >
template< typename Element1, typename Element2, typename Index >
bool
MultiDeviceMemoryOperations< Devices::Cuda, DeviceType >::compare( const Element1* destination,
                                                                   const Element2* source,
                                                                   Index size )
{
   if( size == 0 )
      return true;
   TNL_ASSERT_TRUE( destination, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_TRUE( source, "Attempted to compare data through a nullptr." );
   TNL_ASSERT_GE( size, (Index) 0, "Array size must be non-negative." );
   return MultiDeviceMemoryOperations< DeviceType, Devices::Cuda >::compare( source, destination, size );
}

}  // namespace Algorithms
}  // namespace noa::TNL
