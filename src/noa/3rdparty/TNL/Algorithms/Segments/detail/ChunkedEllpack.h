// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/ChunkedEllpackSegmentView.h>
#include <TNL/Algorithms/Segments/detail/CheckLambdas.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {
         namespace detail {

/***
 * In the ChunkedEllpack, the segments are split into slices. This is done
 * in ChunkedEllpack::resolveSliceSizes. All segments elements in each slice
 * are split into chunks. All chunks in one slice have the same size, but the size
 * of chunks can be different in each slice.
 */
template< typename Index >
struct ChunkedEllpackSliceInfo
{
   /**
    * The size of the slice, it means the number of the segments covered by
    * the slice.
    */
   Index size;

   /**
    * The chunk size, i.e. maximal number of non-zero elements that can be stored
    * in the chunk.
    */
   Index chunkSize;

   /**
    * Index of the first segment covered be this slice.
    */
   Index firstSegment;

   /**
    * Position of the first element of this slice.
    */
   Index pointer;
};

template< typename Index,
          typename Device,
          ElementsOrganization Organization >
class ChunkedEllpack
{
   public:

      using DeviceType = Device;
      using IndexType = Index;
      static constexpr ElementsOrganization getOrganization() { return Organization; }
      using OffsetsContainer = Containers::Vector< IndexType, DeviceType, IndexType >;
      using OffsetsHolderView = typename OffsetsContainer::ConstViewType;
      using SegmentsSizes = OffsetsContainer;
      using ChunkedEllpackSliceInfoType = detail::ChunkedEllpackSliceInfo< IndexType >;
      using ChunkedEllpackSliceInfoAllocator = typename Allocators::Default< Device >::template Allocator< ChunkedEllpackSliceInfoType >;
      using ChunkedEllpackSliceInfoContainer = Containers::Array< ChunkedEllpackSliceInfoType, DeviceType, IndexType, ChunkedEllpackSliceInfoAllocator >;
      using ChunkedEllpackSliceInfoContainerView = typename ChunkedEllpackSliceInfoContainer::ConstViewType;
      using SegmentViewType = ChunkedEllpackSegmentView< IndexType, Organization >;

      __cuda_callable__ static
      IndexType getSegmentSizeDirect( const OffsetsHolderView& segmentsToSlicesMapping,
                                      const ChunkedEllpackSliceInfoContainerView& slices,
                                      const OffsetsHolderView& segmentsToChunksMapping,
                                      const IndexType segmentIdx )
      {
         const IndexType& sliceIndex = segmentsToSlicesMapping[ segmentIdx ];
         IndexType firstChunkOfSegment( 0 );
         if( segmentIdx != slices[ sliceIndex ].firstSegment )
            firstChunkOfSegment = segmentsToChunksMapping[ segmentIdx - 1 ];

         const IndexType lastChunkOfSegment = segmentsToChunksMapping[ segmentIdx ];
         const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
         return chunkSize * segmentChunksCount;
      }

      static
      IndexType getSegmentSize( const OffsetsHolderView& segmentsToSlicesMapping,
                                const ChunkedEllpackSliceInfoContainerView& slices,
                                const OffsetsHolderView& segmentsToChunksMapping,
                                const IndexType segmentIdx )
      {
         const IndexType& sliceIndex = segmentsToSlicesMapping.getElement( segmentIdx );
         IndexType firstChunkOfSegment( 0 );
         if( segmentIdx != slices.getElement( sliceIndex ).firstSegment )
            firstChunkOfSegment = segmentsToChunksMapping.getElement( segmentIdx - 1 );

         const IndexType lastChunkOfSegment = segmentsToChunksMapping.getElement( segmentIdx );
         const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         const IndexType chunkSize = slices.getElement( sliceIndex ).chunkSize;
         return chunkSize * segmentChunksCount;
      }

      __cuda_callable__ static
      IndexType getGlobalIndexDirect( const OffsetsHolderView& segmentsToSlicesMapping,
                                      const ChunkedEllpackSliceInfoContainerView& slices,
                                      const OffsetsHolderView& segmentsToChunksMapping,
                                      const IndexType chunksInSlice,
                                      const IndexType segmentIdx,
                                      const IndexType localIdx )
      {
         const IndexType& sliceIndex = segmentsToSlicesMapping[ segmentIdx ];
         IndexType firstChunkOfSegment( 0 );
         if( segmentIdx != slices[ sliceIndex ].firstSegment )
            firstChunkOfSegment = segmentsToChunksMapping[ segmentIdx - 1 ];

         //const IndexType lastChunkOfSegment = segmentsToChunksMapping[ segmentIdx ];
         //const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         const IndexType sliceOffset = slices[ sliceIndex ].pointer;
         const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
         //TNL_ASSERT_LE( localIdx, segmentChunksCount * chunkSize, "" );
         TNL_ASSERT_LE( localIdx, ( segmentsToChunksMapping[ segmentIdx ] - firstChunkOfSegment ) * chunkSize, "" );

         if( Organization == RowMajorOrder )
            return sliceOffset + firstChunkOfSegment * chunkSize + localIdx;
         else
         {
            const IndexType inChunkOffset = localIdx % chunkSize;
            const IndexType chunkIdx = localIdx / chunkSize;
            return sliceOffset + inChunkOffset * chunksInSlice + firstChunkOfSegment + chunkIdx;
         }
      }

      static
      IndexType getGlobalIndex( const OffsetsHolderView& segmentsToSlicesMapping,
                                const ChunkedEllpackSliceInfoContainerView& slices,
                                const OffsetsHolderView& segmentsToChunksMapping,
                                const IndexType chunksInSlice,
                                const IndexType segmentIdx,
                                const IndexType localIdx )
      {
         const IndexType& sliceIndex = segmentsToSlicesMapping.getElement( segmentIdx );
         IndexType firstChunkOfSegment( 0 );
         if( segmentIdx != slices.getElement( sliceIndex ).firstSegment )
            firstChunkOfSegment = segmentsToChunksMapping.getElement( segmentIdx - 1 );

         //const IndexType lastChunkOfSegment = segmentsToChunksMapping.getElement( segmentIdx );
         //const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         const IndexType sliceOffset = slices.getElement( sliceIndex ).pointer;
         const IndexType chunkSize = slices.getElement( sliceIndex ).chunkSize;
         //TNL_ASSERT_LE( localIdx, segmentChunksCount * chunkSize, "" );
         TNL_ASSERT_LE( localIdx, ( segmentsToChunksMapping.getElement( segmentIdx ) - firstChunkOfSegment ) * chunkSize, "" );

         if( Organization == RowMajorOrder )
            return sliceOffset + firstChunkOfSegment * chunkSize + localIdx;
         else
         {
            const IndexType inChunkOffset = localIdx % chunkSize;
            const IndexType chunkIdx = localIdx / chunkSize;
            return sliceOffset + inChunkOffset * chunksInSlice + firstChunkOfSegment + chunkIdx;
         }
      }

      static __cuda_callable__
      SegmentViewType getSegmentViewDirect( const OffsetsHolderView& segmentsToSlicesMapping,
                                            const ChunkedEllpackSliceInfoContainerView& slices,
                                            const OffsetsHolderView& segmentsToChunksMapping,
                                            const IndexType& chunksInSlice,
                                            const IndexType& segmentIdx )
      {
         const IndexType& sliceIndex = segmentsToSlicesMapping[ segmentIdx ];
         IndexType firstChunkOfSegment( 0 );
         if( segmentIdx != slices[ sliceIndex ].firstSegment )
            firstChunkOfSegment = segmentsToChunksMapping[ segmentIdx - 1 ];

         const IndexType lastChunkOfSegment = segmentsToChunksMapping[ segmentIdx ];
         const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         const IndexType sliceOffset = slices[ sliceIndex ].pointer;
         const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
         const IndexType segmentSize = segmentChunksCount * chunkSize;

         if( Organization == RowMajorOrder )
            return SegmentViewType( segmentIdx,
                                    sliceOffset + firstChunkOfSegment * chunkSize,
                                    segmentSize,
                                    chunkSize,
                                    chunksInSlice );
         else
            return SegmentViewType( segmentIdx,
                                    sliceOffset + firstChunkOfSegment,
                                    segmentSize,
                                    chunkSize,
                                    chunksInSlice );
      }

      static __cuda_callable__
      SegmentViewType getSegmentView( const OffsetsHolderView& segmentsToSlicesMapping,
                                      const ChunkedEllpackSliceInfoContainerView& slices,
                                      const OffsetsHolderView& segmentsToChunksMapping,
                                      const IndexType chunksInSlice,
                                      const IndexType segmentIdx )
      {
         const IndexType& sliceIndex = segmentsToSlicesMapping.getElement( segmentIdx );
         IndexType firstChunkOfSegment( 0 );
         if( segmentIdx != slices.getElement( sliceIndex ).firstSegment )
            firstChunkOfSegment = segmentsToChunksMapping.getElement( segmentIdx - 1 );

         const IndexType lastChunkOfSegment = segmentsToChunksMapping.getElement( segmentIdx );
         const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         const IndexType sliceOffset = slices.getElement( sliceIndex ).pointer;
         const IndexType chunkSize = slices.getElement( sliceIndex ).chunkSize;
         const IndexType segmentSize = segmentChunksCount * chunkSize;

         if( Organization == RowMajorOrder )
            return SegmentViewType( segmentIdx,
                                    sliceOffset + firstChunkOfSegment * chunkSize,
                                    segmentSize,
                                    chunkSize,
                                    chunksInSlice );
         else
            return SegmentViewType( segmentIdx,
                                    sliceOffset + firstChunkOfSegment,
                                    segmentSize,
                                    chunkSize,
                                    chunksInSlice );
      }
};

#ifdef HAVE_CUDA
template< typename Index,
          typename Fetch,
          bool HasAllParameters = detail::CheckFetchLambda< Index, Fetch >::hasAllParameters() >
struct ChunkedEllpackreduceSegmentsDispatcher{};

template< typename Index, typename Fetch >
struct ChunkedEllpackreduceSegmentsDispatcher< Index, Fetch, true >
{
   template< typename View,
             typename Reduction,
             typename ResultKeeper,
             typename Real,
             typename... Args >
   __device__
   static void exec( View chunkedEllpack,
                     Index gridIdx,
                     Index first,
                     Index last,
                     Fetch fetch,
                     Reduction reduction,
                     ResultKeeper keeper,
                     Real zero,
                     Args... args )
   {
      chunkedEllpack.reduceSegmentsKernelWithAllParameters( gridIdx, first, last, fetch, reduction, keeper, zero, args... );
   }
};

template< typename Index, typename Fetch >
struct ChunkedEllpackreduceSegmentsDispatcher< Index, Fetch, false >
{
   template< typename View,
             typename Reduction,
             typename ResultKeeper,
             typename Real,
             typename... Args >
   __device__
   static void exec( View chunkedEllpack,
                     Index gridIdx,
                     Index first,
                     Index last,
                     Fetch fetch,
                     Reduction reduction,
                     ResultKeeper keeper,
                     Real zero,
                     Args... args )
   {
      chunkedEllpack.reduceSegmentsKernel( gridIdx, first, last, fetch, reduction, keeper, zero, args... );
   }
};

template< typename View,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Real,
          typename... Args >
__global__
void ChunkedEllpackreduceSegmentsKernel( View chunkedEllpack,
                                            Index gridIdx,
                                            Index first,
                                            Index last,
                                            Fetch fetch,
                                            Reduction reduction,
                                            ResultKeeper keeper,
                                            Real zero,
                                            Args... args )
{
   ChunkedEllpackreduceSegmentsDispatcher< Index, Fetch >::exec( chunkedEllpack, gridIdx, first, last, fetch, reduction, keeper, zero, args... );
}
#endif

         } //namespace detail
      } //namespace Segments
   } //namespace Algorithms
} //namepsace TNL
