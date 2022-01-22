// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>  // std::pair

#include <noa/3rdparty/TNL/Assert.h>
#include <noa/3rdparty/TNL/Math.h>
#include <noa/3rdparty/TNL/Cuda/DeviceInfo.h>
#include <noa/3rdparty/TNL/Algorithms/CudaReductionBuffer.h>
#include <noa/3rdparty/TNL/Algorithms/MultiDeviceMemoryOperations.h>
#include <noa/3rdparty/TNL/Exceptions/CudaSupportMissing.h>

namespace noaTNL {
namespace Algorithms {
namespace detail {

#ifdef HAVE_CUDA
/* Template for cooperative reduction across the CUDA block of threads.
 * It is a *cooperative* operation - all threads must call the operation,
 * otherwise it will deadlock!
 *
 * The default implementation is generic and the reduction is done using
 * shared memory. Specializations can be made based on `Reduction` and
 * `ValueType`, e.g. using the `__shfl_sync` intrinsics for supported
 * value types.
 */
template< int blockSize,
          typename Reduction,
          typename ValueType >
struct CudaBlockReduce
{
   // storage to be allocated in shared memory
   struct Storage
   {
      // when there is only one warp per blockSize.x, we need to allocate two warps
      // worth of shared memory so that we don't index shared memory out of bounds
      ValueType data[ (blockSize <= 32) ? 2 * blockSize : blockSize ];
   };

   /* Cooperative reduction across the CUDA block - each thread will get the
    * result of the reduction
    *
    * \param reduction   The binary reduction functor.
    * \param identity     Neutral element for given reduction operation, i.e.
    *                     value such that `reduction(identity, x) == x` for any `x`.
    * \param threadValue Value of the calling thread to be reduced.
    * \param tid         Index of the calling thread (usually `threadIdx.x`,
    *                    unless you know what you are doing).
    * \param storage     Auxiliary storage (must be allocated as a __shared__
    *                    variable).
    */
   __device__ static
   ValueType
   reduce( const Reduction& reduction,
           ValueType identity,
           ValueType threadValue,
           int tid,
           Storage& storage )
   {
      storage.data[ tid ] = threadValue;
      __syncthreads();

      if( blockSize >= 1024 ) {
         if( tid < 512 )
            storage.data[ tid ] = reduction( storage.data[ tid ], storage.data[ tid + 512 ] );
         __syncthreads();
      }
      if( blockSize >= 512 ) {
         if( tid < 256 )
            storage.data[ tid ] = reduction( storage.data[ tid ], storage.data[ tid + 256 ] );
         __syncthreads();
      }
      if( blockSize >= 256 ) {
         if( tid < 128 )
            storage.data[ tid ] = reduction( storage.data[ tid ], storage.data[ tid + 128 ] );
         __syncthreads();
      }
      if( blockSize >= 128 ) {
         if( tid <  64 )
            storage.data[ tid ] = reduction( storage.data[ tid ], storage.data[ tid + 64 ] );
         __syncthreads();
      }

      // This runs in one warp so we use __syncwarp() instead of __syncthreads().
      if( tid < 32 ) {
         if( blockSize >= 64 )
            storage.data[ tid ] = reduction( storage.data[ tid ], storage.data[ tid + 32 ] );
         __syncwarp();
         // Note that here we do not have to check if tid < 16 etc, because we have
         // 2 * blockSize.x elements of shared memory per block, so we do not
         // access out of bounds. The results for the upper half will be undefined,
         // but unused anyway.
         if( blockSize >= 32 )
            storage.data[ tid ] = reduction( storage.data[ tid ], storage.data[ tid + 16 ] );
         __syncwarp();
         if( blockSize >= 16 )
            storage.data[ tid ] = reduction( storage.data[ tid ], storage.data[ tid + 8 ] );
         __syncwarp();
         if( blockSize >=  8 )
            storage.data[ tid ] = reduction( storage.data[ tid ], storage.data[ tid + 4 ] );
         __syncwarp();
         if( blockSize >=  4 )
            storage.data[ tid ] = reduction( storage.data[ tid ], storage.data[ tid + 2 ] );
         __syncwarp();
         if( blockSize >=  2 )
            storage.data[ tid ] = reduction( storage.data[ tid ], storage.data[ tid + 1 ] );
      }

      __syncthreads();
      return storage.data[ 0 ];
   }
};

template< int blockSize,
          typename Reduction,
          typename ValueType >
struct CudaBlockReduceShfl
{
   // storage to be allocated in shared memory
   struct Storage
   {
      ValueType warpResults[ Cuda::getWarpSize() ];
   };

   /* Cooperative reduction across the CUDA block - each thread will get the
    * result of the reduction
    *
    * \param reduction   The binary reduction functor.
    * \param identity     Neutral element for given reduction operation, i.e.
    *                     value such that `reduction(identity, x) == x` for any `x`.
    * \param threadValue Value of the calling thread to be reduced.
    * \param tid         Index of the calling thread (usually `threadIdx.x`,
    *                    unless you know what you are doing).
    * \param storage     Auxiliary storage (must be allocated as a __shared__
    *                    variable).
    */
   __device__ static
   ValueType
   reduce( const Reduction& reduction,
           ValueType identity,
           ValueType threadValue,
           int tid,
           Storage& storage )
   {
      // verify the configuration
      static_assert( blockSize / Cuda::getWarpSize() <= Cuda::getWarpSize(),
                     "blockSize is too large, it would not be possible to reduce warpResults using one warp" );

      int lane_id = threadIdx.x % warpSize;
      int warp_id = threadIdx.x / warpSize;

      // perform the parallel reduction across warps
      threadValue = warpReduce( reduction, threadValue );

      // the first thread of each warp writes the result into the shared memory
      if( lane_id == 0 )
         storage.warpResults[ warp_id ] = threadValue;
      __syncthreads();

      // the first warp performs the final reduction
      if( warp_id == 0 ) {
         // read from shared memory only if that warp existed
         if( tid < blockSize / Cuda::getWarpSize() )
            threadValue = storage.warpResults[ lane_id ];
         else
            threadValue = identity;
         threadValue = warpReduce( reduction, threadValue );
      }

      // the first thread writes the result into the shared memory
      if( tid == 0 )
         storage.warpResults[ 0 ] = threadValue;

      __syncthreads();
      return storage.warpResults[ 0 ];
   }

   /* Helper function.
    * Cooperative reduction across the warp - each thread will get the result
    * of the reduction
    */
   __device__ static
   ValueType
   warpReduce( const Reduction& reduction,
               ValueType threadValue )
   {
      constexpr unsigned mask = 0xffffffff;
      #pragma unroll
      for( int i = Cuda::getWarpSize() / 2; i > 0; i /= 2 ) {
         const ValueType otherValue = __shfl_xor_sync( mask, threadValue, i );
         threadValue = reduction( threadValue, otherValue );
      }
      return threadValue;
   }
};

template< int blockSize,
          typename Reduction >
struct CudaBlockReduce< blockSize, Reduction, int >
: public CudaBlockReduceShfl< blockSize, Reduction, int >
{};

template< int blockSize,
          typename Reduction >
struct CudaBlockReduce< blockSize, Reduction, unsigned int >
: public CudaBlockReduceShfl< blockSize, Reduction, unsigned int >
{};

template< int blockSize,
          typename Reduction >
struct CudaBlockReduce< blockSize, Reduction, long >
: public CudaBlockReduceShfl< blockSize, Reduction, long >
{};

template< int blockSize,
          typename Reduction >
struct CudaBlockReduce< blockSize, Reduction, unsigned long >
: public CudaBlockReduceShfl< blockSize, Reduction, unsigned long >
{};

template< int blockSize,
          typename Reduction >
struct CudaBlockReduce< blockSize, Reduction, long long >
: public CudaBlockReduceShfl< blockSize, Reduction, long long >
{};

template< int blockSize,
          typename Reduction >
struct CudaBlockReduce< blockSize, Reduction, unsigned long long >
: public CudaBlockReduceShfl< blockSize, Reduction, unsigned long long >
{};

template< int blockSize,
          typename Reduction >
struct CudaBlockReduce< blockSize, Reduction, float >
: public CudaBlockReduceShfl< blockSize, Reduction, float >
{};

template< int blockSize,
          typename Reduction >
struct CudaBlockReduce< blockSize, Reduction, double >
: public CudaBlockReduceShfl< blockSize, Reduction, double >
{};

/* Template for cooperative reduction with argument across the CUDA block of
 * threads. It is a *cooperative* operation - all threads must call the
 * operation, otherwise it will deadlock!
 *
 * The default implementation is generic and the reduction is done using
 * shared memory. Specializations can be made based on `Reduction` and
 * `ValueType`, e.g. using the `__shfl_sync` intrinsics for supported
 * value types.
 */
template< int blockSize,
          typename Reduction,
          typename ValueType,
          typename IndexType >
struct CudaBlockReduceWithArgument
{
   // storage to be allocated in shared memory
   struct Storage
   {
      // when there is only one warp per blockSize.x, we need to allocate two warps
      // worth of shared memory so that we don't index shared memory out of bounds
      ValueType data[ (blockSize <= 32) ? 2 * blockSize : blockSize ];
      IndexType idx [ (blockSize <= 32) ? 2 * blockSize : blockSize ];
   };

   /* Cooperative reduction with argument across the CUDA block - each thread
    * will get the pair of the result of the reduction and the index
    *
    * \param reduction   The binary reduction functor.
    * \param identity     Neutral element for given reduction operation, i.e.
    *                     value such that `reduction(identity, x) == x` for any `x`.
    * \param threadValue Value of the calling thread to be reduced.
    * \param threadIndex Index value of the calling thread to be reduced.
    * \param tid         Index of the calling thread (usually `threadIdx.x`,
    *                    unless you know what you are doing).
    * \param storage     Auxiliary storage (must be allocated as a __shared__
    *                    variable).
    */
   __device__ static
   std::pair< ValueType, IndexType >
   reduceWithArgument( const Reduction& reduction,
                       ValueType identity,
                       ValueType threadValue,
                       IndexType threadIndex,
                       int tid,
                       Storage& storage )
   {
      storage.data[ tid ] = threadValue;
      storage.idx[ tid ] = threadIndex;
      __syncthreads();

      if( blockSize >= 1024 ) {
         if( tid < 512 )
            reduction( storage.data[ tid ], storage.data[ tid + 512 ], storage.idx[ tid ], storage.idx[ tid + 512 ] );
         __syncthreads();
      }
      if( blockSize >= 512 ) {
         if( tid < 256 )
            reduction( storage.data[ tid ], storage.data[ tid + 256 ], storage.idx[ tid ], storage.idx[ tid + 256 ] );
         __syncthreads();
      }
      if( blockSize >= 256 ) {
         if( tid < 128 )
            reduction( storage.data[ tid ], storage.data[ tid + 128 ], storage.idx[ tid ], storage.idx[ tid + 128 ] );
         __syncthreads();
      }
      if( blockSize >= 128 ) {
         if( tid <  64 )
            reduction( storage.data[ tid ], storage.data[ tid + 64 ], storage.idx[ tid ], storage.idx[ tid + 64 ] );
         __syncthreads();
      }

      // This runs in one warp so we use __syncwarp() instead of __syncthreads().
      if( tid < 32 ) {
         if( blockSize >= 64 )
            reduction( storage.data[ tid ], storage.data[ tid + 32 ], storage.idx[ tid ], storage.idx[ tid + 32 ] );
         __syncwarp();
         // Note that here we do not have to check if tid < 16 etc, because we have
         // 2 * blockSize.x elements of shared memory per block, so we do not
         // access out of bounds. The results for the upper half will be undefined,
         // but unused anyway.
         if( blockSize >= 32 )
            reduction( storage.data[ tid ], storage.data[ tid + 16 ], storage.idx[ tid ], storage.idx[ tid + 16 ] );
         __syncwarp();
         if( blockSize >= 16 )
            reduction( storage.data[ tid ], storage.data[ tid + 8 ], storage.idx[ tid ], storage.idx[ tid + 8 ] );
         __syncwarp();
         if( blockSize >=  8 )
            reduction( storage.data[ tid ], storage.data[ tid + 4 ], storage.idx[ tid ], storage.idx[ tid + 4 ] );
         __syncwarp();
         if( blockSize >=  4 )
            reduction( storage.data[ tid ], storage.data[ tid + 2 ], storage.idx[ tid ], storage.idx[ tid + 2 ] );
         __syncwarp();
         if( blockSize >=  2 )
            reduction( storage.data[ tid ], storage.data[ tid + 1 ], storage.idx[ tid ], storage.idx[ tid + 1 ] );
      }

      __syncthreads();
      return std::make_pair( storage.data[ 0 ], storage.idx[ 0 ] );
   }
};
#endif

/****
 * The performance of this kernel is very sensitive to register usage.
 * Compile with --ptxas-options=-v and configure these constants for given
 * architecture so that there are no local memory spills.
 */
static constexpr int Reduction_maxThreadsPerBlock = 256;  // must be a power of 2
static constexpr int Reduction_registersPerThread = 32;   // empirically determined optimal value

#ifdef HAVE_CUDA
// __CUDA_ARCH__ is defined only in device code!
#if (__CUDA_ARCH__ == 750 )
   // Turing has a limit of 1024 threads per multiprocessor
   static constexpr int Reduction_minBlocksPerMultiprocessor = 4;
#else
   static constexpr int Reduction_minBlocksPerMultiprocessor = 8;
#endif

template< int blockSize,
          typename DataFetcher,
          typename Reduction,
          typename Result,
          typename Index >
__global__ void
__launch_bounds__( Reduction_maxThreadsPerBlock, Reduction_minBlocksPerMultiprocessor )
CudaReductionKernel( DataFetcher dataFetcher,
                     const Reduction reduction,
                     Result identity,
                     Index begin,
                     Index end,
                     Result* output )
{
   TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in CudaReductionKernel" );

   // allocate shared memory
   using BlockReduce = CudaBlockReduce< blockSize, Reduction, Result >;
   union Shared {
      typename BlockReduce::Storage blockReduceStorage;

      // initialization is not allowed for __shared__ variables, so we need to
      // disable initialization in the implicit default constructor
      __device__ Shared() {}
   };
   __shared__ Shared storage;

   // Calculate the grid size (stride of the sequential reduction loop).
   const Index gridSize = blockDim.x * gridDim.x;
   // Shift the input lower bound by the thread index in the grid.
   begin += blockIdx.x * blockDim.x + threadIdx.x;

   // Start with the sequential reduction and push the result into the shared memory.
   Result result = identity;
   while( begin + 4 * gridSize < end ) {
      result = reduction( result, dataFetcher( begin ) );
      result = reduction( result, dataFetcher( begin + gridSize ) );
      result = reduction( result, dataFetcher( begin + 2 * gridSize ) );
      result = reduction( result, dataFetcher( begin + 3 * gridSize ) );
      begin += 4 * gridSize;
   }
   while( begin + 2 * gridSize < end ) {
      result = reduction( result, dataFetcher( begin ) );
      result = reduction( result, dataFetcher( begin + gridSize ) );
      begin += 2 * gridSize;
   }
   while( begin < end ) {
      result = reduction( result, dataFetcher( begin ) );
      begin += gridSize;
   }
   __syncthreads();

   // Perform the parallel reduction.
   result = BlockReduce::reduce( reduction, identity, result, threadIdx.x, storage.blockReduceStorage );

   // Store the result back in the global memory.
   if( threadIdx.x == 0 )
      output[ blockIdx.x ] = result;
}

template< int blockSize,
          typename DataFetcher,
          typename Reduction,
          typename Result,
          typename Index >
__global__ void
__launch_bounds__( Reduction_maxThreadsPerBlock, Reduction_minBlocksPerMultiprocessor )
CudaReductionWithArgumentKernel( DataFetcher dataFetcher,
                                 const Reduction reduction,
                                 Result identity,
                                 Index begin,
                                 Index end,
                                 Result* output,
                                 Index* idxOutput,
                                 const Index* idxInput = nullptr )
{
   TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in CudaReductionKernel" );

   // allocate shared memory
   using BlockReduce = CudaBlockReduceWithArgument< blockSize, Reduction, Result, Index >;
   union Shared {
      typename BlockReduce::Storage blockReduceStorage;

      // initialization is not allowed for __shared__ variables, so we need to
      // disable initialization in the implicit default constructor
      __device__ Shared() {}
   };
   __shared__ Shared storage;

   // Calculate the grid size (stride of the sequential reduction loop).
   const Index gridSize = blockDim.x * gridDim.x;
   // Shift the input lower bound by the thread index in the grid.
   begin += blockIdx.x * blockDim.x + threadIdx.x;

   // TODO: initialIndex should be passed as an argument to the kernel
   Index initialIndex;

   // Start with the sequential reduction and push the result into the shared memory.
   Result result = identity;
   if( idxInput ) {
      if( begin < end ) {
         result = dataFetcher( begin );
         initialIndex = idxInput[ begin ];
         begin += gridSize;
      }
      while( begin + 4 * gridSize < end ) {
         reduction( result, dataFetcher( begin ), initialIndex, idxInput[ begin ] );
         reduction( result, dataFetcher( begin + gridSize ), initialIndex, idxInput[ begin + gridSize ] );
         reduction( result, dataFetcher( begin + 2 * gridSize ), initialIndex, idxInput[ begin + 2 * gridSize ] );
         reduction( result, dataFetcher( begin + 3 * gridSize ), initialIndex, idxInput[ begin + 3 * gridSize ] );
         begin += 4 * gridSize;
      }
      while( begin + 2 * gridSize < end ) {
         reduction( result, dataFetcher( begin ), initialIndex, idxInput[ begin ] );
         reduction( result, dataFetcher( begin + gridSize ), initialIndex, idxInput[ begin + gridSize ] );
         begin += 2 * gridSize;
      }
      while( begin < end ) {
         reduction( result, dataFetcher( begin ), initialIndex, idxInput[ begin ] );
         begin += gridSize;
      }
   }
   else {
      if( begin < end ) {
         result = dataFetcher( begin );
         initialIndex = begin;
         begin += gridSize;
      }
      while( begin + 4 * gridSize < end ) {
         reduction( result, dataFetcher( begin ), initialIndex, begin );
         reduction( result, dataFetcher( begin + gridSize ), initialIndex, begin + gridSize );
         reduction( result, dataFetcher( begin + 2 * gridSize ), initialIndex, begin + 2 * gridSize );
         reduction( result, dataFetcher( begin + 3 * gridSize ), initialIndex, begin + 3 * gridSize );
         begin += 4 * gridSize;
      }
      while( begin + 2 * gridSize < end ) {
         reduction( result, dataFetcher( begin ), initialIndex, begin );
         reduction( result, dataFetcher( begin + gridSize ), initialIndex, begin + gridSize );
         begin += 2 * gridSize;
      }
      while( begin < end ) {
         reduction( result, dataFetcher( begin ), initialIndex, begin );
         begin += gridSize;
      }
   }
   __syncthreads();

   // Perform the parallel reduction.
   const std::pair< Result, Index > result_pair = BlockReduce::reduceWithArgument( reduction, identity, result, initialIndex, threadIdx.x, storage.blockReduceStorage );

   // Store the result back in the global memory.
   if( threadIdx.x == 0 ) {
      output[ blockIdx.x ] = result_pair.first;
      idxOutput[ blockIdx.x ] = result_pair.second;
   }
}
#endif


template< typename Index,
          typename Result >
struct CudaReductionKernelLauncher
{
   // The number of blocks should be a multiple of the number of multiprocessors
   // to ensure optimum balancing of the load. This is very important, because
   // we run the kernel with a fixed number of blocks, so the amount of work per
   // block increases with enlarging the problem, so even small imbalance can
   // cost us dearly.
   // Therefore,  desGridSize = blocksPerMultiprocessor * numberOfMultiprocessors
   // where blocksPerMultiprocessor is determined according to the number of
   // available registers on the multiprocessor.
   // On Tesla K40c, desGridSize = 8 * 15 = 120.
   //
   // Update:
   // It seems to be better to map only one CUDA block per one multiprocessor or maybe
   // just slightly more. Therefore we omit blocksdPerMultiprocessor in the following.
   CudaReductionKernelLauncher( const Index begin, const Index end )
   : activeDevice( Cuda::DeviceInfo::getActiveDevice() ),
     blocksdPerMultiprocessor( Cuda::DeviceInfo::getRegistersPerMultiprocessor( activeDevice )
                               / ( Reduction_maxThreadsPerBlock * Reduction_registersPerThread ) ),
     //desGridSize( blocksdPerMultiprocessor * Cuda::DeviceInfo::getCudaMultiprocessors( activeDevice ) ),
     desGridSize( Cuda::DeviceInfo::getCudaMultiprocessors( activeDevice ) ),
     begin( begin ), end( end )
   {
   }

   template< typename DataFetcher,
             typename Reduction >
   int start( const Reduction& reduction,
              DataFetcher& dataFetcher,
              const Result& identity,
              Result*& output )
   {
      // create reference to the reduction buffer singleton and set size
      const std::size_t buf_size = 2 * desGridSize * sizeof( Result );
      CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
      cudaReductionBuffer.setSize( buf_size );
      output = cudaReductionBuffer.template getData< Result >();

      this->reducedSize = this->launch( begin, end, reduction, dataFetcher, identity, output );
      return this->reducedSize;
   }

   template< typename DataFetcher,
             typename Reduction >
   int startWithArgument( const Reduction& reduction,
                          DataFetcher& dataFetcher,
                          const Result& identity,
                          Result*& output,
                          Index*& idxOutput )
   {
      // create reference to the reduction buffer singleton and set size
      const std::size_t buf_size = 2 * desGridSize * ( sizeof( Result ) + sizeof( Index ) );
      CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
      cudaReductionBuffer.setSize( buf_size );
      output = cudaReductionBuffer.template getData< Result >();
      idxOutput = reinterpret_cast< Index* >( &output[ 2 * desGridSize ] );

      this->reducedSize = this->launchWithArgument( begin, end, reduction, dataFetcher, identity, output, idxOutput, nullptr );
      return this->reducedSize;
   }

   template< typename Reduction >
   Result
   finish( const Reduction& reduction,
           const Result& identity )
   {
      // Input is the first half of the buffer, output is the second half
      CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
      Result* input = cudaReductionBuffer.template getData< Result >();
      Result* output = &input[ desGridSize ];

      while( this->reducedSize > 1 )
      {
         // this lambda has to be defined inside the loop, because the captured variable changes
         auto copyFetch = [input] __cuda_callable__ ( Index i ) { return input[ i ]; };
         this->reducedSize = this->launch( 0, this->reducedSize, reduction, copyFetch, identity, output );
         std::swap( input, output );
      }

      // swap again to revert the swap from the last iteration
      // AND to solve the case when this->reducedSize was 1 since the beginning
      std::swap( input, output );

      // Copy result on CPU
      Result result;
      MultiDeviceMemoryOperations< void, Devices::Cuda >::copy( &result, output, 1 );
      return result;
   }

   template< typename Reduction >
   std::pair< Result, Index >
   finishWithArgument( const Reduction& reduction,
                       const Result& identity )
   {
      // Input is the first half of the buffer, output is the second half
      CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
      Result* input = cudaReductionBuffer.template getData< Result >();
      Result* output = &input[ desGridSize ];
      Index* idxInput = reinterpret_cast< Index* >( &output[ desGridSize ] );
      Index* idxOutput = &idxInput[ desGridSize ];

      while( this->reducedSize > 1 )
      {
         // this lambda has to be defined inside the loop, because the captured variable changes
         auto copyFetch = [input] __cuda_callable__ ( Index i ) { return input[ i ]; };
         this->reducedSize = this->launchWithArgument( ( Index ) 0, this->reducedSize, reduction, copyFetch, identity, output, idxOutput, idxInput );
         std::swap( input, output );
         std::swap( idxInput, idxOutput );
      }

      // swap again to revert the swap from the last iteration
      // AND to solve the case when this->reducedSize was 1 since the beginning
      std::swap( input, output );
      std::swap( idxInput, idxOutput );

      ////
      // Copy result on CPU
      std::pair< Result, Index > result;
      MultiDeviceMemoryOperations< void, Devices::Cuda >::copy( &result.first, output, 1 );
      MultiDeviceMemoryOperations< void, Devices::Cuda >::copy( &result.second, idxOutput, 1 );
      return result;
   }


   protected:
      template< typename DataFetcher,
                typename Reduction >
      int launch( const Index begin,
                  const Index end,
                  const Reduction& reduction,
                  DataFetcher& dataFetcher,
                  const Result& identity,
                  Result* output )
      {
#ifdef HAVE_CUDA
         const Index size = end - begin;
         dim3 blockSize, gridSize;
         blockSize.x = Reduction_maxThreadsPerBlock;
         gridSize.x = noaTNL::min( Cuda::getNumberOfBlocks( size, blockSize.x ), desGridSize );

         // This is "general", but this method always sets blockSize.x to a specific value,
         // so runtime switch is not necessary - it only prolongs the compilation time.
/*
         // Depending on the blockSize we generate appropriate template instance.
         switch( blockSize.x )
         {
            case 512:
               CudaReductionKernel< 512 >
               <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, size, output);
               break;
            case 256:
               cudaFuncSetCacheConfig(CudaReductionKernel< 256, DataFetcher, Reduction, Result, Index >, cudaFuncCachePreferShared);

               CudaReductionKernel< 256 >
               <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, size, output);
               break;
            case 128:
               cudaFuncSetCacheConfig(CudaReductionKernel< 128, DataFetcher, Reduction, Result, Index >, cudaFuncCachePreferShared);

               CudaReductionKernel< 128 >
               <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, size, output);
               break;
            case  64:
               cudaFuncSetCacheConfig(CudaReductionKernel<  64, DataFetcher, Reduction, Result, Index >, cudaFuncCachePreferShared);

               CudaReductionKernel<  64 >
               <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, size, output);
               break;
            case  32:
               cudaFuncSetCacheConfig(CudaReductionKernel<  32, DataFetcher, Reduction, Result, Index >, cudaFuncCachePreferShared);

               CudaReductionKernel<  32 >
               <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, size, output);
               break;
            case  16:
               cudaFuncSetCacheConfig(CudaReductionKernel<  16, DataFetcher, Reduction, Result, Index >, cudaFuncCachePreferShared);

               CudaReductionKernel<  16 >
               <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, size, output);
               break;
           case   8:
               cudaFuncSetCacheConfig(CudaReductionKernel<   8, DataFetcher, Reduction, Result, Index >, cudaFuncCachePreferShared);

               CudaReductionKernel<   8 >
               <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, size, output);
               break;
            case   4:
               cudaFuncSetCacheConfig(CudaReductionKernel<   4, DataFetcher, Reduction, Result, Index >, cudaFuncCachePreferShared);

               CudaReductionKernel<   4 >
               <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, size, output);
               break;
            case   2:
               cudaFuncSetCacheConfig(CudaReductionKernel<   2, DataFetcher, Reduction, Result, Index >, cudaFuncCachePreferShared);

               CudaReductionKernel<   2 >
               <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, size, output);
               break;
            case   1:
               TNL_ASSERT( false, std::cerr << "blockSize should not be 1." << std::endl );
            default:
               TNL_ASSERT( false, std::cerr << "Block size is " << blockSize. x << " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." );
         }
         cudaStreamSynchronize(0);
         TNL_CHECK_CUDA_DEVICE;
*/

         // Check just to future-proof the code setting blockSize.x
         if( blockSize.x == Reduction_maxThreadsPerBlock ) {
            cudaFuncSetCacheConfig(CudaReductionKernel< Reduction_maxThreadsPerBlock, DataFetcher, Reduction, Result, Index >, cudaFuncCachePreferShared);

            // shared memory is allocated statically inside the kernel
            CudaReductionKernel< Reduction_maxThreadsPerBlock >
            <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, begin, end, output);
            cudaStreamSynchronize(0);
            TNL_CHECK_CUDA_DEVICE;
         }
         else {
            TNL_ASSERT( false, std::cerr << "Block size was expected to be " << Reduction_maxThreadsPerBlock << ", but " << blockSize.x << " was specified." << std::endl; );
         }

         // Return the size of the output array on the CUDA device
         return gridSize.x;
#else
         throw Exceptions::CudaSupportMissing();
#endif
      }

      template< typename DataFetcher,
                typename Reduction >
      int launchWithArgument( const Index begin,
                              const Index end,
                              const Reduction& reduction,
                              DataFetcher& dataFetcher,
                              const Result& identity,
                              Result* output,
                              Index* idxOutput,
                              const Index* idxInput )
      {
#ifdef HAVE_CUDA
         dim3 blockSize, gridSize;
         const Index size = end - begin;
         blockSize.x = Reduction_maxThreadsPerBlock;
         gridSize.x = noaTNL::min( Cuda::getNumberOfBlocks( size, blockSize.x ), desGridSize );

         // This is "general", but this method always sets blockSize.x to a specific value,
         // so runtime switch is not necessary - it only prolongs the compilation time.
/*
         // Depending on the blockSize we generate appropriate template instance.
         switch( blockSize.x )
         {
            case 512:
               CudaReductionWithArgumentKernel< 512 >
               <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, size, output, idxOutput, idxInput );
               break;
            case 256:
               cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel< 256, DataFetcher, Reduction, Result, Index >, cudaFuncCachePreferShared);

               CudaReductionWithArgumentKernel< 256 >
               <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, size, output, idxOutput, idxInput );
               break;
            case 128:
               cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel< 128, DataFetcher, Reduction, Result, Index >, cudaFuncCachePreferShared);

               CudaReductionWithArgumentKernel< 128 >
               <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, size, output, idxOutput, idxInput );
               break;
            case  64:
               cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel<  64, DataFetcher, Reduction, Result, Index >, cudaFuncCachePreferShared);

               CudaReductionWithArgumentKernel<  64 >
               <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, size, output, idxOutput, idxInput );
               break;
            case  32:
               cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel<  32, DataFetcher, Reduction, Result, Index >, cudaFuncCachePreferShared);

               CudaReductionWithArgumentKernel<  32 >
               <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, size, output, idxOutput, idxInput );
               break;
            case  16:
               cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel<  16, DataFetcher, Reduction, Result, Index >, cudaFuncCachePreferShared);

               CudaReductionWithArgumentKernel<  16 >
               <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, size, output, idxOutput, idxInput );
               break;
           case   8:
               cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel<   8, DataFetcher, Reduction, Result, Index >, cudaFuncCachePreferShared);

               CudaReductionWithArgumentKernel<   8 >
               <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, size, output, idxOutput, idxInput );
               break;
            case   4:
               cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel<   4, DataFetcher, Reduction, Result, Index >, cudaFuncCachePreferShared);

               CudaReductionWithArgumentKernel<   4 >
               <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, size, output, idxOutput, idxInput );
               break;
            case   2:
               cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel<   2, DataFetcher, Reduction, Result, Index >, cudaFuncCachePreferShared);

               CudaReductionWithArgumentKernel<   2 >
               <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, size, output, idxOutput, idxInput );
               break;
            case   1:
               TNL_ASSERT( false, std::cerr << "blockSize should not be 1." << std::endl );
            default:
               TNL_ASSERT( false, std::cerr << "Block size is " << blockSize. x << " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." );
         }
         cudaStreamSynchronize(0);
         TNL_CHECK_CUDA_DEVICE;
*/

         // Check just to future-proof the code setting blockSize.x
         if( blockSize.x == Reduction_maxThreadsPerBlock ) {
            cudaFuncSetCacheConfig(CudaReductionWithArgumentKernel< Reduction_maxThreadsPerBlock, DataFetcher, Reduction, Result, Index >, cudaFuncCachePreferShared);

            // shared memory is allocated statically inside the kernel
            CudaReductionWithArgumentKernel< Reduction_maxThreadsPerBlock >
            <<< gridSize, blockSize >>>( dataFetcher, reduction, identity, begin, end, output, idxOutput, idxInput );
            cudaStreamSynchronize(0);
            TNL_CHECK_CUDA_DEVICE;
         }
         else {
            TNL_ASSERT( false, std::cerr << "Block size was expected to be " << Reduction_maxThreadsPerBlock << ", but " << blockSize.x << " was specified." << std::endl; );
         }

         // return the size of the output array on the CUDA device
         return gridSize.x;
#else
         throw Exceptions::CudaSupportMissing();
#endif
      }


      const int activeDevice;
      const int blocksdPerMultiprocessor;
      const int desGridSize;
      //const Index originalSize;
      const Index begin, end;
      Index reducedSize;
};

} // namespace detail
} // namespace Algorithms
} // namespace noaTNL
