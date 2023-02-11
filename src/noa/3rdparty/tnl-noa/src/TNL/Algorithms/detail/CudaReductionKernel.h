// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>  // std::pair

#include <noa/3rdparty/tnl-noa/src/TNL/Assert.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Math.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/DeviceInfo.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/KernelLaunch.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/CudaReductionBuffer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/MultiDeviceMemoryOperations.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/CudaSupportMissing.h>

namespace noa::TNL {
namespace Algorithms {
namespace detail {

#ifdef __CUDACC__
/* Template for cooperative reduction across the CUDA block of threads.
 * It is a *cooperative* operation - all threads must call the operation,
 * otherwise it will deadlock!
 *
 * The default implementation is generic and the reduction is done using
 * shared memory. Specializations can be made based on `Reduction` and
 * `ValueType`, e.g. using the `__shfl_sync` intrinsics for supported
 * value types.
 */
template< int blockSize, typename Reduction, typename ValueType >
struct CudaBlockReduce
{
   // storage to be allocated in shared memory
   struct Storage
   {
      // when there is only one warp per blockSize.x, we need to allocate two warps
      // worth of shared memory so that we don't index shared memory out of bounds
      ValueType data[ ( blockSize <= 32 ) ? 2 * blockSize : blockSize ];
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
   __device__
   static ValueType
   reduce( const Reduction& reduction, ValueType identity, ValueType threadValue, int tid, Storage& storage )
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
         if( tid < 64 )
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
         if( blockSize >= 8 )
            storage.data[ tid ] = reduction( storage.data[ tid ], storage.data[ tid + 4 ] );
         __syncwarp();
         if( blockSize >= 4 )
            storage.data[ tid ] = reduction( storage.data[ tid ], storage.data[ tid + 2 ] );
         __syncwarp();
         if( blockSize >= 2 )
            storage.data[ tid ] = reduction( storage.data[ tid ], storage.data[ tid + 1 ] );
      }

      __syncthreads();
      return storage.data[ 0 ];
   }
};

template< int blockSize, typename Reduction, typename ValueType >
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
   __device__
   static ValueType
   reduce( const Reduction& reduction, ValueType identity, ValueType threadValue, int tid, Storage& storage )
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
   __device__
   static ValueType
   warpReduce( const Reduction& reduction, ValueType threadValue )
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

template< int blockSize, typename Reduction >
struct CudaBlockReduce< blockSize, Reduction, int > : public CudaBlockReduceShfl< blockSize, Reduction, int >
{};

template< int blockSize, typename Reduction >
struct CudaBlockReduce< blockSize, Reduction, unsigned int > : public CudaBlockReduceShfl< blockSize, Reduction, unsigned int >
{};

template< int blockSize, typename Reduction >
struct CudaBlockReduce< blockSize, Reduction, long > : public CudaBlockReduceShfl< blockSize, Reduction, long >
{};

template< int blockSize, typename Reduction >
struct CudaBlockReduce< blockSize, Reduction, unsigned long >
: public CudaBlockReduceShfl< blockSize, Reduction, unsigned long >
{};

template< int blockSize, typename Reduction >
struct CudaBlockReduce< blockSize, Reduction, long long > : public CudaBlockReduceShfl< blockSize, Reduction, long long >
{};

template< int blockSize, typename Reduction >
struct CudaBlockReduce< blockSize, Reduction, unsigned long long >
: public CudaBlockReduceShfl< blockSize, Reduction, unsigned long long >
{};

template< int blockSize, typename Reduction >
struct CudaBlockReduce< blockSize, Reduction, float > : public CudaBlockReduceShfl< blockSize, Reduction, float >
{};

template< int blockSize, typename Reduction >
struct CudaBlockReduce< blockSize, Reduction, double > : public CudaBlockReduceShfl< blockSize, Reduction, double >
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
template< int blockSize, typename Reduction, typename ValueType, typename IndexType >
struct CudaBlockReduceWithArgument
{
   // storage to be allocated in shared memory
   struct Storage
   {
      // when there is only one warp per blockSize.x, we need to allocate two warps
      // worth of shared memory so that we don't index shared memory out of bounds
      ValueType data[ ( blockSize <= 32 ) ? 2 * blockSize : blockSize ];
      IndexType idx[ ( blockSize <= 32 ) ? 2 * blockSize : blockSize ];
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
   __device__
   static std::pair< ValueType, IndexType >
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
         if( tid < 64 )
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
         if( blockSize >= 8 )
            reduction( storage.data[ tid ], storage.data[ tid + 4 ], storage.idx[ tid ], storage.idx[ tid + 4 ] );
         __syncwarp();
         if( blockSize >= 4 )
            reduction( storage.data[ tid ], storage.data[ tid + 2 ], storage.idx[ tid ], storage.idx[ tid + 2 ] );
         __syncwarp();
         if( blockSize >= 2 )
            reduction( storage.data[ tid ], storage.data[ tid + 1 ], storage.idx[ tid ], storage.idx[ tid + 1 ] );
      }

      __syncthreads();
      return std::make_pair( storage.data[ 0 ], storage.idx[ 0 ] );
   }
};
#endif

template< int blockSize, typename DataFetcher, typename Reduction, typename Result, typename Index >
__global__
void
CudaReductionKernel( DataFetcher dataFetcher,
                     const Reduction reduction,
                     Result identity,
                     Index begin,
                     Index end,
                     Result* output )
{
#ifdef __CUDACC__
   TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in CudaReductionKernel" );

   // allocate shared memory
   using BlockReduce = CudaBlockReduce< blockSize, Reduction, Result >;
   union Shared
   {
      typename BlockReduce::Storage blockReduceStorage;

      // initialization is not allowed for __shared__ variables, so we need to
      // disable initialization in the implicit default constructor
      __device__
      Shared() {}
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
#endif
}

template< int blockSize, typename DataFetcher, typename Reduction, typename Result, typename Index >
__global__
void
CudaReductionWithArgumentKernel( DataFetcher dataFetcher,
                                 const Reduction reduction,
                                 Result identity,
                                 Index begin,
                                 Index end,
                                 Result* output,
                                 Index* idxOutput,
                                 const Index* idxInput = nullptr )
{
#ifdef __CUDACC__
   TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in CudaReductionKernel" );

   // allocate shared memory
   using BlockReduce = CudaBlockReduceWithArgument< blockSize, Reduction, Result, Index >;
   union Shared
   {
      typename BlockReduce::Storage blockReduceStorage;

      // initialization is not allowed for __shared__ variables, so we need to
      // disable initialization in the implicit default constructor
      __device__
      Shared() {}
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
   const std::pair< Result, Index > result_pair =
      BlockReduce::reduceWithArgument( reduction, identity, result, initialIndex, threadIdx.x, storage.blockReduceStorage );

   // Store the result back in the global memory.
   if( threadIdx.x == 0 ) {
      output[ blockIdx.x ] = result_pair.first;
      idxOutput[ blockIdx.x ] = result_pair.second;
   }
#endif
}

template< typename Index, typename Result >
struct CudaReductionKernelLauncher
{
   // must be a power of 2
   static constexpr int maxThreadsPerBlock = 256;

   // The number of blocks should be a multiple of the number of multiprocessors
   // to ensure optimum balancing of the load. This is very important, because
   // we run the kernel with a fixed number of blocks, so the amount of work per
   // block increases with enlarging the problem, so even small imbalance can
   // cost us dearly.
   // Therefore,  desGridSize = blocksPerMultiprocessor * numberOfMultiprocessors
   // where the maximum value of blocksPerMultiprocessor can be determined
   // according to the number of available registers on the multiprocessor.
   // However, it seems to be better to map only one CUDA block per multiprocessor,
   // or maybe just slightly more.
   CudaReductionKernelLauncher( const Index begin, const Index end )
   : activeDevice( Cuda::DeviceInfo::getActiveDevice() ),
     desGridSize( Cuda::DeviceInfo::getCudaMultiprocessors( activeDevice ) ), begin( begin ), end( end )
   {}

   template< typename DataFetcher, typename Reduction >
   int
   start( const Reduction& reduction, DataFetcher& dataFetcher, const Result& identity, Result*& output )
   {
      // create reference to the reduction buffer singleton and set size
      const std::size_t buf_size = 2 * desGridSize * sizeof( Result );
      CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
      cudaReductionBuffer.setSize( buf_size );
      output = cudaReductionBuffer.template getData< Result >();

      this->reducedSize = this->launch( begin, end, reduction, dataFetcher, identity, output );
      return this->reducedSize;
   }

   template< typename DataFetcher, typename Reduction >
   int
   startWithArgument( const Reduction& reduction,
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
   finish( const Reduction& reduction, const Result& identity )
   {
      // Input is the first half of the buffer, output is the second half
      CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
      Result* input = cudaReductionBuffer.template getData< Result >();
      Result* output = &input[ desGridSize ];

      while( this->reducedSize > 1 ) {
         // this lambda has to be defined inside the loop, because the captured variable changes
         auto copyFetch = [ input ] __cuda_callable__( Index i )
         {
            return input[ i ];
         };
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
   finishWithArgument( const Reduction& reduction, const Result& identity )
   {
      // Input is the first half of the buffer, output is the second half
      CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
      Result* input = cudaReductionBuffer.template getData< Result >();
      Result* output = &input[ desGridSize ];
      Index* idxInput = reinterpret_cast< Index* >( &output[ desGridSize ] );
      Index* idxOutput = &idxInput[ desGridSize ];

      while( this->reducedSize > 1 ) {
         // this lambda has to be defined inside the loop, because the captured variable changes
         auto copyFetch = [ input ] __cuda_callable__( Index i )
         {
            return input[ i ];
         };
         this->reducedSize = this->launchWithArgument(
            (Index) 0, this->reducedSize, reduction, copyFetch, identity, output, idxOutput, idxInput );
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
   template< typename DataFetcher, typename Reduction >
   int
   launch( const Index begin,
           const Index end,
           const Reduction& reduction,
           DataFetcher& dataFetcher,
           const Result& identity,
           Result* output )
   {
      const Index size = end - begin;
      Cuda::LaunchConfiguration launch_config;
      launch_config.blockSize.x = maxThreadsPerBlock;
      launch_config.gridSize.x = TNL::min( Cuda::getNumberOfBlocks( size, launch_config.blockSize.x ), desGridSize );
      // shared memory is allocated statically inside the kernel

      // Check just to future-proof the code setting blockSize.x
      if( launch_config.blockSize.x == maxThreadsPerBlock ) {
         cudaFuncSetCacheConfig( CudaReductionKernel< maxThreadsPerBlock, DataFetcher, Reduction, Result, Index >,
                                 cudaFuncCachePreferShared );
         Cuda::launchKernelSync( CudaReductionKernel< maxThreadsPerBlock, DataFetcher, Reduction, Result, Index >,
                                 launch_config,
                                 dataFetcher,
                                 reduction,
                                 identity,
                                 begin,
                                 end,
                                 output );
      }
      else {
         throw std::runtime_error( "Block size was expected to be " + std::to_string( maxThreadsPerBlock ) + ", but "
                                   + std::to_string( launch_config.blockSize.x ) + " was specified." );
      }

      // Return the size of the output array on the CUDA device
      return launch_config.gridSize.x;
   }

   template< typename DataFetcher, typename Reduction >
   int
   launchWithArgument( const Index begin,
                       const Index end,
                       const Reduction& reduction,
                       DataFetcher& dataFetcher,
                       const Result& identity,
                       Result* output,
                       Index* idxOutput,
                       const Index* idxInput )
   {
      const Index size = end - begin;
      Cuda::LaunchConfiguration launch_config;
      launch_config.blockSize.x = maxThreadsPerBlock;
      launch_config.gridSize.x = TNL::min( Cuda::getNumberOfBlocks( size, launch_config.blockSize.x ), desGridSize );
      // shared memory is allocated statically inside the kernel

      // Check just to future-proof the code setting blockSize.x
      if( launch_config.blockSize.x == maxThreadsPerBlock ) {
         cudaFuncSetCacheConfig( CudaReductionWithArgumentKernel< maxThreadsPerBlock, DataFetcher, Reduction, Result, Index >,
                                 cudaFuncCachePreferShared );
         Cuda::launchKernelSync( CudaReductionWithArgumentKernel< maxThreadsPerBlock, DataFetcher, Reduction, Result, Index >,
                                 launch_config,
                                 dataFetcher,
                                 reduction,
                                 identity,
                                 begin,
                                 end,
                                 output,
                                 idxOutput,
                                 idxInput );
      }
      else {
         throw std::runtime_error( "Block size was expected to be " + std::to_string( maxThreadsPerBlock ) + ", but "
                                   + std::to_string( launch_config.blockSize.x ) + " was specified." );
      }

      // return the size of the output array on the CUDA device
      return launch_config.gridSize.x;
   }

   const int activeDevice;
   const int desGridSize;
   // const Index originalSize;
   const Index begin, end;
   Index reducedSize;
};

}  // namespace detail
}  // namespace Algorithms
}  // namespace noa::TNL
