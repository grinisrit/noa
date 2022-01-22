// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Math.h>
#include <noa/3rdparty/TNL/Cuda/SharedMemory.h>
#include <noa/3rdparty/TNL/Exceptions/CudaBadAlloc.h>
#include <noa/3rdparty/TNL/Containers/Array.h>
#include "ScanType.h"

namespace noaTNL {
namespace Algorithms {
namespace detail {

#ifdef HAVE_CUDA
/* Template for cooperative scan across the CUDA block of threads.
 * It is a *cooperative* operation - all threads must call the operation,
 * otherwise it will deadlock!
 *
 * The default implementation is generic and the reduction is done using
 * shared memory. Specializations can be made based on `Reduction` and
 * `ValueType`, e.g. using the `__shfl_sync` intrinsics for supported
 * value types.
 */
template< ScanType scanType,
          int blockSize,
          typename Reduction,
          typename ValueType >
struct CudaBlockScan
{
   // storage to be allocated in shared memory
   struct Storage
   {
      ValueType chunkResults[ blockSize + blockSize / Cuda::getNumberOfSharedMemoryBanks() ];  // accessed via Cuda::getInterleaving()
      ValueType warpResults[ Cuda::getWarpSize() ];
   };

   /* Cooperative scan across the CUDA block - each thread will get the
    * result of the scan according to its ID.
    *
    * \param reduction    The binary reduction functor.
    * \param identity     Neutral element for given reduction operation, i.e.
    *                     value such that `reduction(identity, x) == x` for any `x`.
    * \param threadValue  Value of the calling thread to be reduced.
    * \param tid          Index of the calling thread (usually `threadIdx.x`,
    *                     unless you know what you are doing).
    * \param storage      Auxiliary storage (must be allocated as a __shared__
    *                     variable).
    */
   __device__ static
   ValueType
   scan( const Reduction& reduction,
         ValueType identity,
         ValueType threadValue,
         int tid,
         Storage& storage )
   {
      // verify the configuration
      TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in CudaBlockScan::scan" );
      static_assert( blockSize / Cuda::getWarpSize() <= Cuda::getWarpSize(),
                     "blockSize is too large, it would not be possible to scan warpResults using one warp" );

      // store the threadValue in the shared memory
      const int chunkResultIdx = Cuda::getInterleaving( tid );
      storage.chunkResults[ chunkResultIdx ] = threadValue;
      __syncthreads();

      // perform the parallel scan on chunkResults inside warps
      const int lane_id = tid % Cuda::getWarpSize();
      const int warp_id = tid / Cuda::getWarpSize();
      #pragma unroll
      for( int stride = 1; stride < Cuda::getWarpSize(); stride *= 2 ) {
         if( lane_id >= stride ) {
            storage.chunkResults[ chunkResultIdx ] = reduction( storage.chunkResults[ chunkResultIdx ], storage.chunkResults[ Cuda::getInterleaving( tid - stride ) ] );
         }
         __syncwarp();
      }
      threadValue = storage.chunkResults[ chunkResultIdx ];

      // the last thread in warp stores the intermediate result in warpResults
      if( lane_id == Cuda::getWarpSize() - 1 )
         storage.warpResults[ warp_id ] = threadValue;
      __syncthreads();

      // perform the scan of warpResults using one warp
      if( warp_id == 0 )
         #pragma unroll
         for( int stride = 1; stride < blockSize / Cuda::getWarpSize(); stride *= 2 ) {
            if( lane_id >= stride )
               storage.warpResults[ tid ] = reduction( storage.warpResults[ tid ], storage.warpResults[ tid - stride ] );
            __syncwarp();
         }
      __syncthreads();

      // shift threadValue by the warpResults
      if( warp_id > 0 )
         threadValue = reduction( threadValue, storage.warpResults[ warp_id - 1 ] );

      // shift the result for exclusive scan
      if( scanType == ScanType::Exclusive ) {
         storage.chunkResults[ chunkResultIdx ] = threadValue;
         __syncthreads();
         threadValue = (tid == 0) ? identity : storage.chunkResults[ Cuda::getInterleaving( tid - 1 ) ];
      }

      __syncthreads();
      return threadValue;
   }
};

template< ScanType scanType,
          int __unused,  // the __shfl implementation does not depend on the blockSize
          typename Reduction,
          typename ValueType >
struct CudaBlockScanShfl
{
   // storage to be allocated in shared memory
   struct Storage
   {
      ValueType warpResults[ Cuda::getWarpSize() ];
   };

   /* Cooperative scan across the CUDA block - each thread will get the
    * result of the scan according to its ID.
    *
    * \param reduction    The binary reduction functor.
    * \param identity     Neutral element for given reduction operation, i.e.
    *                     value such that `reduction(identity, x) == x` for any `x`.
    * \param threadValue  Value of the calling thread to be reduced.
    * \param tid          Index of the calling thread (usually `threadIdx.x`,
    *                     unless you know what you are doing).
    * \param storage      Auxiliary storage (must be allocated as a __shared__
    *                     variable).
    */
   __device__ static
   ValueType
   scan( const Reduction& reduction,
         ValueType identity,
         ValueType threadValue,
         int tid,
         Storage& storage )
   {
      const int lane_id = tid % Cuda::getWarpSize();
      const int warp_id = tid / Cuda::getWarpSize();

      // perform the parallel scan across warps
      ValueType total;
      threadValue = warpScan< scanType >( reduction, identity, threadValue, lane_id, total );

      // the last thread in warp stores the result of inclusive scan in warpResults
      if( lane_id == Cuda::getWarpSize() - 1 )
         storage.warpResults[ warp_id ] = total;
      __syncthreads();

      // the first warp performs the scan of warpResults
      if( warp_id == 0 ) {
         // read from shared memory only if that warp existed
         if( tid < blockDim.x / Cuda::getWarpSize() )
            total = storage.warpResults[ lane_id ];
         else
            total = identity;
         storage.warpResults[ lane_id ] = warpScan< ScanType::Inclusive >( reduction, identity, total, lane_id, total );
      }
      __syncthreads();

      // shift threadValue by the warpResults
      if( warp_id > 0 )
         threadValue = reduction( threadValue, storage.warpResults[ warp_id - 1 ] );

      __syncthreads();
      return threadValue;
   }

   /* Helper function.
    * Cooperative scan across the warp - each thread will get the result of the
    * scan according to its ID.
    * return value = thread's result of the *warpScanType* scan
    * total = thread's result of the *inclusive* scan
    */
   template< ScanType warpScanType >
   __device__ static
   ValueType
   warpScan( const Reduction& reduction,
             ValueType identity,
             ValueType threadValue,
             int lane_id,
             ValueType& total )
   {
      constexpr unsigned mask = 0xffffffff;

      // perform an inclusive scan
      #pragma unroll
      for( int stride = 1; stride < Cuda::getWarpSize(); stride *= 2 ) {
         const ValueType otherValue = __shfl_up_sync( mask, threadValue, stride );
         if( lane_id >= stride )
            threadValue = reduction( threadValue, otherValue );
      }

      // set the result of the inclusive scan
      total = threadValue;

      // shift the result for exclusive scan
      if( warpScanType == ScanType::Exclusive ) {
         threadValue = __shfl_up_sync( mask, threadValue, 1 );
         if( lane_id == 0 )
            threadValue = identity;
      }

      return threadValue;
   }
};

template< ScanType scanType,
          int blockSize,
          typename Reduction >
struct CudaBlockScan< scanType, blockSize, Reduction, int >
: public CudaBlockScanShfl< scanType, blockSize, Reduction, int >
{};

template< ScanType scanType,
          int blockSize,
          typename Reduction >
struct CudaBlockScan< scanType, blockSize, Reduction, unsigned int >
: public CudaBlockScanShfl< scanType, blockSize, Reduction, unsigned int >
{};

template< ScanType scanType,
          int blockSize,
          typename Reduction >
struct CudaBlockScan< scanType, blockSize, Reduction, long >
: public CudaBlockScanShfl< scanType, blockSize, Reduction, long >
{};

template< ScanType scanType,
          int blockSize,
          typename Reduction >
struct CudaBlockScan< scanType, blockSize, Reduction, unsigned long >
: public CudaBlockScanShfl< scanType, blockSize, Reduction, unsigned long >
{};

template< ScanType scanType,
          int blockSize,
          typename Reduction >
struct CudaBlockScan< scanType, blockSize, Reduction, long long >
: public CudaBlockScanShfl< scanType, blockSize, Reduction, long long >
{};

template< ScanType scanType,
          int blockSize,
          typename Reduction >
struct CudaBlockScan< scanType, blockSize, Reduction, unsigned long long >
: public CudaBlockScanShfl< scanType, blockSize, Reduction, unsigned long long >
{};

template< ScanType scanType,
          int blockSize,
          typename Reduction >
struct CudaBlockScan< scanType, blockSize, Reduction, float >
: public CudaBlockScanShfl< scanType, blockSize, Reduction, float >
{};

template< ScanType scanType,
          int blockSize,
          typename Reduction >
struct CudaBlockScan< scanType, blockSize, Reduction, double >
: public CudaBlockScanShfl< scanType, blockSize, Reduction, double >
{};

/* Template for cooperative scan of a data tile in the global memory.
 * It is a *cooperative* operation - all threads must call the operation,
 * otherwise it will deadlock!
 */
template< ScanType scanType,
          int blockSize,
          int valuesPerThread,
          typename Reduction,
          typename ValueType >
struct CudaTileScan
{
   using BlockScan = CudaBlockScan< ScanType::Exclusive, blockSize, Reduction, ValueType >;

   // storage to be allocated in shared memory
   struct Storage
   {
      ValueType data[ blockSize * valuesPerThread ];
      typename BlockScan::Storage blockScanStorage;
   };

   /* Cooperative scan of a data tile in the global memory - each thread will
    * get the result of its chunk (i.e. the last value of the (inclusive) scan
    * in the chunk) according to the thread ID.
    *
    * \param input        The input array to be scanned.
    * \param output       The array where the result will be stored.
    * \param begin        The first element in the array to be scanned.
    * \param end          the last element in the array to be scanned.
    * \param outputBegin  The first element in the output array to be written. There
    *                     must be at least `end - begin` elements in the output
    *                     array starting at the position given by `outputBegin`.
    * \param reduction    The binary reduction functor.
    * \param identity     Neutral element for given reduction operation, i.e.
    *                     value such that `reduction(identity, x) == x` for any `x`.
    * \param shift        A global shift to be applied to all elements in the
    *                     chunk processed by this thread.
    * \param storage      Auxiliary storage (must be allocated as a __shared__
    *                     variable).
    */
   template< typename InputView,
             typename OutputView >
   __device__ static
   ValueType
   scan( const InputView input,
         OutputView output,
         typename InputView::IndexType begin,
         typename InputView::IndexType end,
         typename OutputView::IndexType outputBegin,
         const Reduction& reduction,
         ValueType identity,
         ValueType shift,
         Storage& storage )
   {
      // verify the configuration
      TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in CudaTileScan::scan" );
      static_assert( valuesPerThread % 2,
                     "valuesPerThread must be odd, otherwise there would be shared memory bank conflicts "
                     "when threads access their chunks in shared memory sequentially" );

      // calculate indices
      constexpr int maxElementsInBlock = blockSize * valuesPerThread;
      const int remainingElements = end - begin - blockIdx.x * maxElementsInBlock;
      const int elementsInBlock = noaTNL::min( remainingElements, maxElementsInBlock );

      // update global array offsets for the thread
      const int threadOffset = blockIdx.x * maxElementsInBlock + threadIdx.x;
      begin += threadOffset;
      outputBegin += threadOffset;

      // Load data into the shared memory.
      {
         int idx = threadIdx.x;
         while( idx < elementsInBlock )
         {
            storage.data[ idx ] = input[ begin ];
            begin += blockDim.x;
            idx += blockDim.x;
         }
         // fill the remaining (maxElementsInBlock - elementsInBlock) values with identity
         // (this helps to avoid divergent branches in the blocks below)
         while( idx < maxElementsInBlock )
         {
            storage.data[ idx ] = identity;
            idx += blockDim.x;
         }
      }
      __syncthreads();

      // Perform sequential reduction of the thread's chunk in shared memory.
      const int chunkOffset = threadIdx.x * valuesPerThread;
      ValueType value = storage.data[ chunkOffset ];
      #pragma unroll
      for( int i = 1; i < valuesPerThread; i++ )
         value = reduction( value, storage.data[ chunkOffset + i ] );

      // Scan the spine to obtain the initial value ("offset") for the downsweep.
      value = BlockScan::scan( reduction, identity, value, threadIdx.x, storage.blockScanStorage );

      // Apply the global shift.
      value = reduction( value, shift );

      // Downsweep step: scan the chunks and use the result of spine scan as the initial value.
      #pragma unroll
      for( int i = 0; i < valuesPerThread; i++ )
      {
         const ValueType inputValue = storage.data[ chunkOffset + i ];
         if( scanType == ScanType::Exclusive )
            storage.data[ chunkOffset + i ] = value;
         value = reduction( value, inputValue );
         if( scanType == ScanType::Inclusive )
            storage.data[ chunkOffset + i ] = value;
      }
      __syncthreads();

      // Store the result back in the global memory.
      {
         int idx = threadIdx.x;
         while( idx < elementsInBlock )
         {
            output[ outputBegin ] = storage.data[ idx ];
            outputBegin += blockDim.x;
            idx += blockDim.x;
         }
      }

      // Return the last (inclusive) scan value of the chunk processed by this thread.
      return value;
   }
};

/* CudaScanKernelUpsweep - compute partial reductions per each CUDA block.
 */
template< int blockSize,
          int valuesPerThread,
          typename InputView,
          typename Reduction,
          typename ValueType >
__global__ void
CudaScanKernelUpsweep( const InputView input,
                       typename InputView::IndexType begin,
                       typename InputView::IndexType end,
                       Reduction reduction,
                       ValueType identity,
                       ValueType* reductionResults )
{
   // verify the configuration
   TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in CudaScanKernelUpsweep" );
   static_assert( valuesPerThread % 2,
                  "valuesPerThread must be odd, otherwise there would be shared memory bank conflicts "
                  "when threads access their chunks in shared memory sequentially" );

   // allocate shared memory
   using BlockReduce = CudaBlockReduce< blockSize, Reduction, ValueType >;
   union Shared {
      ValueType data[ blockSize * valuesPerThread ];
      typename BlockReduce::Storage blockReduceStorage;

      // initialization is not allowed for __shared__ variables, so we need to
      // disable initialization in the implicit default constructor
      __device__ Shared() {}
   };
   __shared__ Shared storage;

   // calculate indices
   constexpr int maxElementsInBlock = blockSize * valuesPerThread;
   const int remainingElements = end - begin - blockIdx.x * maxElementsInBlock;
   const int elementsInBlock = noaTNL::min( remainingElements, maxElementsInBlock );

   // update global array offset for the thread
   const int threadOffset = blockIdx.x * maxElementsInBlock + threadIdx.x;
   begin += threadOffset;

   // Load data into the shared memory.
   {
      int idx = threadIdx.x;
      while( idx < elementsInBlock )
      {
         storage.data[ idx ] = input[ begin ];
         begin += blockDim.x;
         idx += blockDim.x;
      }
      // fill the remaining (maxElementsInBlock - elementsInBlock) values with identity
      // (this helps to avoid divergent branches in the blocks below)
      while( idx < maxElementsInBlock )
      {
         storage.data[ idx ] = identity;
         idx += blockDim.x;
      }
   }
   __syncthreads();

   // Perform sequential reduction of the thread's chunk in shared memory.
   const int chunkOffset = threadIdx.x * valuesPerThread;
   ValueType value = storage.data[ chunkOffset ];
   #pragma unroll
   for( int i = 1; i < valuesPerThread; i++ )
      value = reduction( value, storage.data[ chunkOffset + i ] );
   __syncthreads();

   // Perform the parallel reduction.
   value = BlockReduce::reduce( reduction, identity, value, threadIdx.x, storage.blockReduceStorage );

   // Store the block result in the global memory.
   if( threadIdx.x == 0 )
      reductionResults[ blockIdx.x ] = value;
}

/* CudaScanKernelDownsweep - scan each tile of the input separately in each CUDA
 * block and use the result of spine scan as the initial value
 */
template< ScanType scanType,
          int blockSize,
          int valuesPerThread,
          typename InputView,
          typename OutputView,
          typename Reduction >
__global__ void
CudaScanKernelDownsweep( const InputView input,
                         OutputView output,
                         typename InputView::IndexType begin,
                         typename InputView::IndexType end,
                         typename OutputView::IndexType outputBegin,
                         Reduction reduction,
                         typename OutputView::ValueType identity,
                         typename OutputView::ValueType shift,
                         const typename OutputView::ValueType* reductionResults )
{
   using ValueType = typename OutputView::ValueType;
   using TileScan = CudaTileScan< scanType, blockSize, valuesPerThread, Reduction, ValueType >;

   // allocate shared memory
   union Shared {
      typename TileScan::Storage tileScanStorage;

      // initialization is not allowed for __shared__ variables, so we need to
      // disable initialization in the implicit default constructor
      __device__ Shared() {}
   };
   __shared__ Shared storage;

   // load the reduction of the previous tiles
   shift = reduction( shift, reductionResults[ blockIdx.x ] );

   // scan from input into output
   TileScan::scan( input, output, begin, end, outputBegin, reduction, identity, shift, storage.tileScanStorage );
}

/* CudaScanKernelParallel - scan each tile of the input separately in each CUDA
 * block (first phase to be followed by CudaScanKernelUniformShift when there
 * are multiple CUDA blocks).
 */
template< ScanType scanType,
          int blockSize,
          int valuesPerThread,
          typename InputView,
          typename OutputView,
          typename Reduction >
__global__ void
CudaScanKernelParallel( const InputView input,
                        OutputView output,
                        typename InputView::IndexType begin,
                        typename InputView::IndexType end,
                        typename OutputView::IndexType outputBegin,
                        Reduction reduction,
                        typename OutputView::ValueType identity,
                        typename OutputView::ValueType* blockResults )
{
   using ValueType = typename OutputView::ValueType;
   using TileScan = CudaTileScan< scanType, blockSize, valuesPerThread, Reduction, ValueType >;

   // allocate shared memory
   union Shared {
      typename TileScan::Storage tileScanStorage;

      // initialization is not allowed for __shared__ variables, so we need to
      // disable initialization in the implicit default constructor
      __device__ Shared() {}
   };
   __shared__ Shared storage;

   // scan from input into output
   const ValueType value = TileScan::scan( input, output, begin, end, outputBegin, reduction, identity, identity, storage.tileScanStorage );

   // The last thread of the block stores the block result in the global memory.
   if( blockResults && threadIdx.x == blockDim.x - 1 )
      blockResults[ blockIdx.x ] = value;
}

/* CudaScanKernelUniformShift - apply a uniform shift to a pre-scanned output
 * array.
 *
 * \param blockResults  An array of per-block shifts coming from the first phase
 *                      (computed by CudaScanKernelParallel)
 * \param shift         A global shift to be applied to all elements of the
 *                      output array.
 */
template< int blockSize,
          int valuesPerThread,
          typename OutputView,
          typename Reduction >
__global__ void
CudaScanKernelUniformShift( OutputView output,
                            typename OutputView::IndexType outputBegin,
                            typename OutputView::IndexType outputEnd,
                            Reduction reduction,
                            const typename OutputView::ValueType* blockResults,
                            typename OutputView::ValueType shift )
{
   // load the block result into a __shared__ variable first
   union Shared {
      typename OutputView::ValueType blockResult;

      // initialization is not allowed for __shared__ variables, so we need to
      // disable initialization in the implicit default constructor
      __device__ Shared() {}
   };
   __shared__ Shared storage;
   if( threadIdx.x == 0 )
      storage.blockResult = blockResults[ blockIdx.x ];

   // update the output offset for the thread
   TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in CudaScanKernelUniformShift" );
   constexpr int maxElementsInBlock = blockSize * valuesPerThread;
   const int threadOffset = blockIdx.x * maxElementsInBlock + threadIdx.x;
   outputBegin += threadOffset;

   // update the block shift
   __syncthreads();
   shift = reduction( shift, storage.blockResult );

   int valueIdx = 0;
   while( valueIdx < valuesPerThread && outputBegin < outputEnd )
   {
      output[ outputBegin ] = reduction( output[ outputBegin ], shift );
      outputBegin += blockDim.x;
      valueIdx++;
   }
}

/**
 * \tparam blockSize  The CUDA block size to be used for kernel launch.
 * \tparam valuesPerThread  Number of elements processed by each thread sequentially.
 */
template< ScanType scanType,
          ScanPhaseType phaseType,
          typename ValueType,
          // use blockSize=256 for 32-bit value types, scale with sizeof(ValueType)
          // to keep shared memory requirements constant
          int blockSize = 256 * 4 / sizeof(ValueType),
          // valuesPerThread should be odd to avoid shared memory bank conflicts
          int valuesPerThread = 7 >
struct CudaScanKernelLauncher
{
   /****
    * \brief Performs both phases of prefix sum.
    *
    * \param input the input array to be scanned
    * \param output the array where the result will be stored
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param outputBegin the first element in the output array to be written. There
    *                    must be at least `end - begin` elements in the output
    *                    array starting at the position given by `outputBegin`.
    * \param reduction Symmetric binary function representing the reduction operation
    *                  (usually addition, i.e. an instance of \ref std::plus).
    * \param identity Neutral element for given reduction operation, i.e.
    *                 value such that `reduction(identity, x) == x` for any `x`.
    */
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
   static void
   perform( const InputArray& input,
            OutputArray& output,
            typename InputArray::IndexType begin,
            typename InputArray::IndexType end,
            typename OutputArray::IndexType outputBegin,
            Reduction&& reduction,
            typename OutputArray::ValueType identity )
   {
      const auto blockShifts = performFirstPhase(
         input,
         output,
         begin,
         end,
         outputBegin,
         reduction,
         identity );

      // if the first-phase kernel was launched with just one block, skip the second phase
      if( blockShifts.getSize() <= 2 )
         return;

      performSecondPhase(
         input,
         output,
         blockShifts,
         begin,
         end,
         outputBegin,
         reduction,
         identity,
         identity );
   }

   /****
    * \brief Performs the first phase of prefix sum.
    *
    * \param input the input array to be scanned
    * \param output the array where the result will be stored
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param outputBegin the first element in the output array to be written. There
    *                    must be at least `end - begin` elements in the output
    *                    array starting at the position given by `outputBegin`.
    * \param reduction Symmetric binary function representing the reduction operation
    *                  (usually addition, i.e. an instance of \ref std::plus).
    * \param identity Neutral element for given reduction operation, i.e.
    *                 value such that `reduction(identity, x) == x` for any `x`.
    */
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
   static auto
   performFirstPhase( const InputArray& input,
                      OutputArray& output,
                      typename InputArray::IndexType begin,
                      typename InputArray::IndexType end,
                      typename OutputArray::IndexType outputBegin,
                      Reduction&& reduction,
                      typename OutputArray::ValueType identity )
   {
      static_assert( std::is_same< ValueType, typename OutputArray::ValueType >::value, "invalid configuration of ValueType" );
      using Index = typename InputArray::IndexType;

      if( end - begin <= blockSize * valuesPerThread ) {
         // allocate array for the block results
         Containers::Array< typename OutputArray::ValueType, Devices::Cuda > blockResults;
         blockResults.setSize( 2 );
         blockResults.setElement( 0, identity );

         // run the kernel with just 1 block
         if( end - begin <= blockSize )
            CudaScanKernelParallel< scanType, blockSize, 1 ><<< 1, blockSize >>>
               ( input.getConstView(),
                 output.getView(),
                 begin,
                 end,
                 outputBegin,
                 reduction,
                 identity,
                 // blockResults are shifted by 1, because the 0-th element should stay identity
                 &blockResults.getData()[ 1 ] );
         else if( end - begin <= blockSize * 3 )
            CudaScanKernelParallel< scanType, blockSize, 3 ><<< 1, blockSize >>>
               ( input.getConstView(),
                 output.getView(),
                 begin,
                 end,
                 outputBegin,
                 reduction,
                 identity,
                 // blockResults are shifted by 1, because the 0-th element should stay identity
                 &blockResults.getData()[ 1 ] );
         else if( end - begin <= blockSize * 5 )
            CudaScanKernelParallel< scanType, blockSize, 5 ><<< 1, blockSize >>>
               ( input.getConstView(),
                 output.getView(),
                 begin,
                 end,
                 outputBegin,
                 reduction,
                 identity,
                 // blockResults are shifted by 1, because the 0-th element should stay identity
                 &blockResults.getData()[ 1 ] );
         else
            CudaScanKernelParallel< scanType, blockSize, valuesPerThread ><<< 1, blockSize >>>
               ( input.getConstView(),
                 output.getView(),
                 begin,
                 end,
                 outputBegin,
                 reduction,
                 identity,
                 // blockResults are shifted by 1, because the 0-th element should stay identity
                 &blockResults.getData()[ 1 ] );

         // synchronize the null-stream
         cudaStreamSynchronize(0);
         TNL_CHECK_CUDA_DEVICE;

         // Store the number of CUDA grids for the purpose of unit testing, i.e.
         // to check if we test the algorithm with more than one CUDA grid.
         gridsCount() = 1;

         // blockResults now contains shift values for each block - to be used in the second phase
         return blockResults;
      }
      else {
         // compute the number of grids
         constexpr int maxElementsInBlock = blockSize * valuesPerThread;
         const Index numberOfBlocks = roundUpDivision( end - begin, maxElementsInBlock );
         const Index numberOfGrids = Cuda::getNumberOfGrids( numberOfBlocks, maxGridSize() );

         // allocate array for the block results
         Containers::Array< typename OutputArray::ValueType, Devices::Cuda > blockResults;
         blockResults.setSize( numberOfBlocks + 1 );

         // loop over all grids
         for( Index gridIdx = 0; gridIdx < numberOfGrids; gridIdx++ ) {
            // compute current grid offset and size of data to be scanned
            const Index gridOffset = gridIdx * maxGridSize() * maxElementsInBlock;
            const Index currentSize = noaTNL::min( end - begin - gridOffset, maxGridSize() * maxElementsInBlock );

            // setup block and grid size
            dim3 cudaBlockSize, cudaGridSize;
            cudaBlockSize.x = blockSize;
            cudaGridSize.x = roundUpDivision( currentSize, maxElementsInBlock );

            // run the kernel
            switch( phaseType )
            {
               case ScanPhaseType::WriteInFirstPhase:
                  CudaScanKernelParallel< scanType, blockSize, valuesPerThread ><<< cudaGridSize, cudaBlockSize >>>
                     ( input.getConstView(),
                       output.getView(),
                       begin + gridOffset,
                       begin + gridOffset + currentSize,
                       outputBegin + gridOffset,
                       reduction,
                       identity,
                       &blockResults.getData()[ gridIdx * maxGridSize() ] );
                  break;

               case ScanPhaseType::WriteInSecondPhase:
                  CudaScanKernelUpsweep< blockSize, valuesPerThread ><<< cudaGridSize, cudaBlockSize >>>
                     ( input.getConstView(),
                       begin + gridOffset,
                       begin + gridOffset + currentSize,
                       reduction,
                       identity,
                       &blockResults.getData()[ gridIdx * maxGridSize() ] );
                  break;
            }
         }

         // synchronize the null-stream after all grids
         cudaStreamSynchronize(0);
         TNL_CHECK_CUDA_DEVICE;

         // blockResults now contains scan results for each block. The first phase
         // ends by computing an exclusive scan of this array.
         CudaScanKernelLauncher< ScanType::Exclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::perform(
            blockResults,
            blockResults,
            0,
            blockResults.getSize(),
            0,
            reduction,
            identity );

         // Store the number of CUDA grids for the purpose of unit testing, i.e.
         // to check if we test the algorithm with more than one CUDA grid.
         gridsCount() = numberOfGrids;

         // blockResults now contains shift values for each block - to be used in the second phase
         return blockResults;
      }
   }

   /****
    * \brief Performs the second phase of prefix sum.
    *
    * \param input the input array to be scanned
    * \param output the array where the result will be stored
    * \param blockShifts  Pointer to a GPU array containing the block shifts. It is the
    *                     result of the first phase.
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param outputBegin the first element in the output array to be written. There
    *                    must be at least `end - begin` elements in the output
    *                    array starting at the position given by `outputBegin`.
    * \param reduction Symmetric binary function representing the reduction operation
    *                  (usually addition, i.e. an instance of \ref std::plus).
    * \param identity Neutral element for given reduction operation, i.e.
    *                 value such that `reduction(identity, x) == x` for any `x`.
    * \param shift A constant shifting all elements of the array (usually
    *              `identity`, i.e. the neutral value).
    */
   template< typename InputArray,
             typename OutputArray,
             typename BlockShifts,
             typename Reduction >
   static void
   performSecondPhase( const InputArray& input,
                       OutputArray& output,
                       const BlockShifts& blockShifts,
                       typename InputArray::IndexType begin,
                       typename InputArray::IndexType end,
                       typename OutputArray::IndexType outputBegin,
                       Reduction&& reduction,
                       typename OutputArray::ValueType identity,
                       typename OutputArray::ValueType shift )
   {
      static_assert( std::is_same< ValueType, typename OutputArray::ValueType >::value, "invalid configuration of ValueType" );
      using Index = typename InputArray::IndexType;

      // if the input was already scanned with just one block in the first phase,
      // it must be shifted uniformly in the second phase
      if( end - begin <= blockSize * valuesPerThread ) {
         CudaScanKernelUniformShift< blockSize, valuesPerThread ><<< 1, blockSize >>>
            ( output.getView(),
              outputBegin,
              outputBegin + end - begin,
              reduction,
              blockShifts.getData(),
              shift );
      }
      else {
         // compute the number of grids
         constexpr int maxElementsInBlock = blockSize * valuesPerThread;
         const Index numberOfBlocks = roundUpDivision( end - begin, maxElementsInBlock );
         const Index numberOfGrids = Cuda::getNumberOfGrids( numberOfBlocks, maxGridSize() );

         // loop over all grids
         for( Index gridIdx = 0; gridIdx < numberOfGrids; gridIdx++ ) {
            // compute current grid offset and size of data to be scanned
            const Index gridOffset = gridIdx * maxGridSize() * maxElementsInBlock;
            const Index currentSize = noaTNL::min( end - begin - gridOffset, maxGridSize() * maxElementsInBlock );

            // setup block and grid size
            dim3 cudaBlockSize, cudaGridSize;
            cudaBlockSize.x = blockSize;
            cudaGridSize.x = roundUpDivision( currentSize, maxElementsInBlock );

            // run the kernel
            switch( phaseType )
            {
               case ScanPhaseType::WriteInFirstPhase:
                  CudaScanKernelUniformShift< blockSize, valuesPerThread ><<< cudaGridSize, cudaBlockSize >>>
                     ( output.getView(),
                       outputBegin + gridOffset,
                       outputBegin + gridOffset + currentSize,
                       reduction,
                       &blockShifts.getData()[ gridIdx * maxGridSize() ],
                       shift );
                  break;

               case ScanPhaseType::WriteInSecondPhase:
                  CudaScanKernelDownsweep< scanType, blockSize, valuesPerThread ><<< cudaGridSize, cudaBlockSize >>>
                     ( input.getConstView(),
                       output.getView(),
                       begin + gridOffset,
                       begin + gridOffset + currentSize,
                       outputBegin + gridOffset,
                       reduction,
                       identity,
                       shift,
                       &blockShifts.getData()[ gridIdx * maxGridSize() ] );
                  break;
            }
         }
      }

      // synchronize the null-stream after all grids
      cudaStreamSynchronize(0);
      TNL_CHECK_CUDA_DEVICE;
   }

   // The following serves for setting smaller maxGridSize so that we can force
   // the scan in CUDA to run with more than one grid in unit tests.
   static int& maxGridSize()
   {
      static int maxGridSize = Cuda::getMaxGridSize();
      return maxGridSize;
   }

   static void resetMaxGridSize()
   {
      maxGridSize() = Cuda::getMaxGridSize();
      gridsCount() = -1;
   }

   static int& gridsCount()
   {
      static int gridsCount = -1;
      return gridsCount;
   }
};

#endif

} // namespace detail
} // namespace Algorithms
} // namespace noaTNL
