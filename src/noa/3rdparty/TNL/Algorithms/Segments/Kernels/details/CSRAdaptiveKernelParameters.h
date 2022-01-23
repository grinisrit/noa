// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noa::TNL {
   namespace Algorithms {
      namespace Segments {
         namespace detail {

// This can be used for tunning the number of CUDA threads per block depending on the size of Value
// TODO: Perform some tests
static constexpr int CSRAdaptiveKernelParametersCudaBlockSizes[] = { 256, 256, 256, 256, 256, 256 };

template< int SizeOfValue = 1,
          int StreamedSharedMemory_ = 24576 >
struct CSRAdaptiveKernelParameters
{
   static constexpr int MaxValueSizeLog = 6;

   static constexpr int getSizeValueLogConstexpr( const int i );

   static constexpr int getSizeOfValue() { return SizeOfValue; };

   static constexpr int SizeOfValueLog = getSizeValueLogConstexpr( SizeOfValue );

   static_assert( SizeOfValueLog < MaxValueSizeLog, "Parameter SizeOfValue is too large." );

   /**
    * \brief Computes number of CUDA threads per block depending on Value type.
    *
    * \return CUDA block size.
    */
   static constexpr int CudaBlockSize() { return CSRAdaptiveKernelParametersCudaBlockSizes[ SizeOfValueLog ]; };
   //{ return SizeOfValue == 8 ? 128 : 256; };

   /**
    * \brief Returns amount of shared memory dedicated for stream CSR kernel.
    *
    * \return Stream shared memory.
    */
   static constexpr size_t StreamedSharedMemory() { return StreamedSharedMemory_; };

   /**
    * \brief Number of elements fitting into streamed shared memory.
    */
   static constexpr size_t StreamedSharedElementsCount() { return StreamedSharedMemory() / SizeOfValue; };

   /**
    * \brief Computes number of warps in one CUDA block.
    */
   static constexpr size_t WarpsCount() { return CudaBlockSize() / Cuda::getWarpSize(); };

   /**
    * \brief Computes number of elements to be streamed into the shared memory.
    *
    * \return Number of elements to be streamed into the shared memory.
    */
   static constexpr size_t StreamedSharedElementsPerWarp() { return StreamedSharedElementsCount() / WarpsCount(); };

   /**
    * \brief Returns maximum number of elements per warp for vector and hybrid kernel.
    *
    * \return Maximum number of elements per warp for vector and hybrid kernel.
    */
   static constexpr int MaxVectorElementsPerWarp() { return 384; };

   /**
    * \brief Returns maximum number of elements per warp for adaptive kernel.
    *
    * \return Maximum number of elements per warp for adaptive kernel.
    */
   static constexpr int MaxAdaptiveElementsPerWarp() { return 512; };

   static int getSizeValueLog( const int i )
   {
      if( i ==  1 ) return 0;
      if( i ==  2 ) return 1;
      if( i <=  4 ) return 2;
      if( i <=  8 ) return 3;
      if( i <= 16 ) return 4;
      return 5;
   }
};


template< int SizeOfValue,
          int StreamedSharedMemory_ >
constexpr int
CSRAdaptiveKernelParameters< SizeOfValue, StreamedSharedMemory_ >::
getSizeValueLogConstexpr( const int i )
{
   if( i ==  1 ) return 0;
   if( i ==  2 ) return 1;
   if( i <=  4 ) return 2;
   if( i <=  8 ) return 3;
   if( i <= 16 ) return 4;
   if( i <= 32 ) return 5;
   return 6;
};

         } // namespace detail
      } // namespace Segments
   }  // namespace Algorithms
} // namespace noa::TNL
