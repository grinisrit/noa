// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Assert.h>
#include <noa/3rdparty/TNL/Cuda/LaunchHelpers.h>
#include <noa/3rdparty/TNL/Containers/VectorView.h>
#include <noa/3rdparty/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/detail/LambdaAdapter.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/Kernels/CSRScalarKernel.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/Kernels/CSRAdaptiveKernelView.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/Kernels/details/CSRAdaptiveKernelBlockDescriptor.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/Kernels/details/CSRAdaptiveKernelParameters.h>

namespace noaTNL {
   namespace Algorithms {
      namespace Segments {

#ifdef HAVE_CUDA

template< typename BlocksView,
          typename Offsets,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Real,
          typename... Args >
__global__ void
reduceSegmentsCSRAdaptiveKernel( BlocksView blocks,
                                    int gridIdx,
                                    Offsets offsets,
                                    Index first,
                                    Index last,
                                    Fetch fetch,
                                    Reduction reduce,
                                    ResultKeeper keep,
                                    Real zero,
                                    Args... args )
{
   using BlockType = detail::CSRAdaptiveKernelBlockDescriptor< Index >;
   constexpr int CudaBlockSize = detail::CSRAdaptiveKernelParameters< sizeof( Real ) >::CudaBlockSize();
   constexpr int WarpSize = Cuda::getWarpSize();
   constexpr int WarpsCount = detail::CSRAdaptiveKernelParameters< sizeof( Real ) >::WarpsCount();
   constexpr size_t StreamedSharedElementsPerWarp  = detail::CSRAdaptiveKernelParameters< sizeof( Real ) >::StreamedSharedElementsPerWarp();

   __shared__ Real streamShared[ WarpsCount ][ StreamedSharedElementsPerWarp ];
   __shared__ Real multivectorShared[ CudaBlockSize / WarpSize ];
   //__shared__ BlockType sharedBlocks[ WarpsCount ];

   const Index index = ( ( gridIdx * noaTNL::Cuda::getMaxGridXSize() + blockIdx.x ) * blockDim.x ) + threadIdx.x;
   const Index blockIdx = index / WarpSize;
   if( blockIdx >= blocks.getSize() - 1 )
      return;

   if( threadIdx.x < CudaBlockSize / WarpSize )
      multivectorShared[ threadIdx.x ] = zero;
   Real result = zero;
   bool compute( true );
   const Index laneIdx = threadIdx.x & 31; // & is cheaper than %
   /*if( laneIdx == 0 )
      sharedBlocks[ warpIdx ] = blocks[ blockIdx ];
   __syncthreads();
   const auto& block = sharedBlocks[ warpIdx ];*/
   const BlockType block = blocks[ blockIdx ];
   const Index firstSegmentIdx = block.getFirstSegment();
   const Index begin = offsets[ firstSegmentIdx ];

   if( block.getType() == detail::Type::STREAM ) // Stream kernel - many short segments per warp
   {
      const Index warpIdx = threadIdx.x / 32;
      const Index end = begin + block.getSize();

      // Stream data to shared memory
      for( Index globalIdx = laneIdx + begin; globalIdx < end; globalIdx += WarpSize )
         streamShared[ warpIdx ][ globalIdx - begin ] = fetch( globalIdx, compute );
      const Index lastSegmentIdx = firstSegmentIdx + block.getSegmentsInBlock();

      for( Index i = firstSegmentIdx + laneIdx; i < lastSegmentIdx; i += WarpSize )
      {
         const Index sharedEnd = offsets[ i + 1 ] - begin; // end of preprocessed data
         result = zero;
         // Scalar reduction
         for( Index sharedIdx = offsets[ i ] - begin; sharedIdx < sharedEnd; sharedIdx++ )
            result = reduce( result, streamShared[ warpIdx ][ sharedIdx ] );
         keep( i, result );
      }
   }
   else if( block.getType() == detail::Type::VECTOR ) // Vector kernel - one segment per warp
   {
      const Index end = begin + block.getSize();
      const Index segmentIdx = block.getFirstSegment();

      for( Index globalIdx = begin + laneIdx; globalIdx < end; globalIdx += WarpSize )
         result = reduce( result, fetch( globalIdx, compute ) );

      // Parallel reduction
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 16 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result,  8 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result,  4 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result,  2 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result,  1 ) );
      if( laneIdx == 0 )
         keep( segmentIdx, result );
   }
   else // block.getType() == Type::LONG - several warps per segment
   {
      const Index segmentIdx = block.getFirstSegment();//block.index[0];
      const Index end = offsets[segmentIdx + 1];

      TNL_ASSERT_GT( block.getWarpsCount(), 0, "" );
      result = zero;
      for( Index globalIdx = begin + laneIdx + noaTNL::Cuda::getWarpSize() * block.getWarpIdx();
           globalIdx < end;
           globalIdx += noaTNL::Cuda::getWarpSize() * block.getWarpsCount() )
      {
         result = reduce( result, fetch( globalIdx, compute ) );
      }

      result += __shfl_down_sync(0xFFFFFFFF, result, 16);
      result += __shfl_down_sync(0xFFFFFFFF, result, 8);
      result += __shfl_down_sync(0xFFFFFFFF, result, 4);
      result += __shfl_down_sync(0xFFFFFFFF, result, 2);
      result += __shfl_down_sync(0xFFFFFFFF, result, 1);

      const Index warpID = threadIdx.x / 32;
      if( laneIdx == 0 )
         multivectorShared[ warpID ] = result;

      __syncthreads();
      // Reduction in multivectorShared
      if( block.getWarpIdx() == 0 && laneIdx < 16 )
      {
         constexpr int totalWarps = CudaBlockSize / WarpSize;
         if( totalWarps >= 32 )
         {
            multivectorShared[ laneIdx ] =  reduce( multivectorShared[ laneIdx ], multivectorShared[ laneIdx + 16 ] );
            __syncwarp();
         }
         if( totalWarps >= 16 )
         {
            multivectorShared[ laneIdx ] =  reduce( multivectorShared[ laneIdx ], multivectorShared[ laneIdx +  8 ] );
            __syncwarp();
         }
         if( totalWarps >= 8 )
         {
            multivectorShared[ laneIdx ] =  reduce( multivectorShared[ laneIdx ], multivectorShared[ laneIdx +  4 ] );
            __syncwarp();
         }
         if( totalWarps >= 4 )
         {
            multivectorShared[ laneIdx ] =  reduce( multivectorShared[ laneIdx ], multivectorShared[ laneIdx +  2 ] );
            __syncwarp();
         }
         if( totalWarps >= 2 )
         {
            multivectorShared[ laneIdx ] =  reduce( multivectorShared[ laneIdx ], multivectorShared[ laneIdx +  1 ] );
            __syncwarp();
         }
         if( laneIdx == 0 )
         {
            //printf( "Long: segmentIdx %d -> %d \n", segmentIdx, multivectorShared[ 0 ] );
            keep( segmentIdx, multivectorShared[ 0 ] );
         }
      }
   }
}
#endif

template< typename Index,
          typename Device,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          bool DispatchScalarCSR =
            detail::CheckFetchLambda< Index, Fetch >::hasAllParameters() ||
            std::is_same< Device, Devices::Host >::value >
struct CSRAdaptiveKernelreduceSegmentsDispatcher;

template< typename Index,
          typename Device,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper >
struct CSRAdaptiveKernelreduceSegmentsDispatcher< Index, Device, Fetch, Reduction, ResultKeeper, true >
{

   template< typename BlocksView,
             typename Offsets,
             typename Real,
             typename... Args >
   static void reduce( const Offsets& offsets,
                       const BlocksView& blocks,
                       Index first,
                       Index last,
                       Fetch& fetch,
                       const Reduction& reduction,
                       ResultKeeper& keeper,
                       const Real& zero,
                       Args... args)
   {
      noaTNL::Algorithms::Segments::CSRScalarKernel< Index, Device >::
         reduceSegments( offsets, first, last, fetch, reduction, keeper, zero, args... );
   }
};

template< typename Index,
          typename Device,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper >
struct CSRAdaptiveKernelreduceSegmentsDispatcher< Index, Device, Fetch, Reduction, ResultKeeper, false >
{
   template< typename BlocksView,
             typename Offsets,
             typename Real,
             typename... Args >
   static void reduce( const Offsets& offsets,
                       const BlocksView& blocks,
                       Index first,
                       Index last,
                       Fetch& fetch,
                       const Reduction& reduction,
                       ResultKeeper& keeper,
                       const Real& zero,
                       Args... args)
   {
#ifdef HAVE_CUDA

      Index blocksCount;

      const Index threads = detail::CSRAdaptiveKernelParameters< sizeof( Real ) >::CudaBlockSize();
      constexpr size_t maxGridSize = noaTNL::Cuda::getMaxGridXSize();

      // Fill blocks
      size_t neededThreads = blocks.getSize() * noaTNL::Cuda::getWarpSize(); // one warp per block
      // Execute kernels on device
      for (Index gridIdx = 0; neededThreads != 0; gridIdx++ )
      {
         if( maxGridSize * threads >= neededThreads )
         {
            blocksCount = roundUpDivision( neededThreads, threads );
            neededThreads = 0;
         }
         else
         {
            blocksCount = maxGridSize;
            neededThreads -= maxGridSize * threads;
         }

         reduceSegmentsCSRAdaptiveKernel<
               BlocksView,
               Offsets,
               Index, Fetch, Reduction, ResultKeeper, Real, Args... >
            <<<blocksCount, threads>>>(
               blocks,
               gridIdx,
               offsets,
               first,
               last,
               fetch,
               reduction,
               keeper,
               zero,
               args... );
      }
      cudaStreamSynchronize(0);
      TNL_CHECK_CUDA_DEVICE;
#endif
   }
};

template< typename Index,
          typename Device >
void
CSRAdaptiveKernelView< Index, Device >::
setBlocks( BlocksType& blocks, const int idx )
{
   this->blocksArray[ idx ].bind( blocks );
}

template< typename Index,
          typename Device >
auto
CSRAdaptiveKernelView< Index, Device >::
getView() -> ViewType
{
   return *this;
};

template< typename Index,
          typename Device >
auto
CSRAdaptiveKernelView< Index, Device >::
getConstView() const -> ConstViewType
{
   return *this;
}

template< typename Index,
          typename Device >
noaTNL::String
CSRAdaptiveKernelView< Index, Device >::
getKernelType()
{
   return "Adaptive";
}

template< typename Index,
          typename Device >
   template< typename OffsetsView,
               typename Fetch,
               typename Reduction,
               typename ResultKeeper,
               typename Real,
               typename... Args >
void
CSRAdaptiveKernelView< Index, Device >::
reduceSegments( const OffsetsView& offsets,
                   Index first,
                   Index last,
                   Fetch& fetch,
                   const Reduction& reduction,
                   ResultKeeper& keeper,
                   const Real& zero,
                   Args... args ) const
{
   int valueSizeLog = getSizeValueLog( sizeof( Real ) );

   if( detail::CheckFetchLambda< Index, Fetch >::hasAllParameters() || valueSizeLog >= MaxValueSizeLog )
   {
      noaTNL::Algorithms::Segments::CSRScalarKernel< Index, Device >::
         reduceSegments( offsets, first, last, fetch, reduction, keeper, zero, args... );
      return;
   }

   CSRAdaptiveKernelreduceSegmentsDispatcher< Index, Device, Fetch, Reduction, ResultKeeper  >::template
      reduce< BlocksView, OffsetsView, Real, Args... >( offsets, this->blocksArray[ valueSizeLog ], first, last, fetch, reduction, keeper, zero, args... );
}

template< typename Index,
          typename Device >
CSRAdaptiveKernelView< Index, Device >&
CSRAdaptiveKernelView< Index, Device >::
operator=( const CSRAdaptiveKernelView< Index, Device >& kernelView )
{
   for( int i = 0; i < MaxValueSizeLog; i++ )
      this->blocksArray[ i ].bind( kernelView.blocksArray[ i ] );
   return *this;
}

template< typename Index,
          typename Device >
void
CSRAdaptiveKernelView< Index, Device >::
printBlocks( int idx ) const
{
   auto& blocks = this->blocksArray[ idx ];
   for( Index i = 0; i < this->blocks.getSize(); i++ )
   {
      auto block = blocks.getElement( i );
      std::cout << "Block " << i << " : " << block << std::endl;
   }

}

      } // namespace Segments
   }  // namespace Algorithms
} // namespace noaTNL
