// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <noa/3rdparty/TNL/TypeTraits.h>
#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/ElementsOrganization.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/ChunkedEllpackSegmentView.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/detail/ChunkedEllpack.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/SegmentsPrinting.h>

namespace noaTNL {
   namespace Algorithms {
      namespace Segments {


template< typename Device,
          typename Index,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization() >
class ChunkedEllpackView
{
   public:

      using DeviceType = Device;
      using IndexType = std::remove_const_t< Index >;
      using OffsetsView = typename Containers::VectorView< Index, DeviceType, IndexType >;
      using ConstOffsetsView = typename OffsetsView::ConstViewType;
      using ViewType = ChunkedEllpackView;
      template< typename Device_, typename Index_ >
      using ViewTemplate = ChunkedEllpackView< Device_, Index_, Organization >;
      using ConstViewType = ChunkedEllpackView< Device, std::add_const_t< Index >, Organization >;
      using SegmentViewType = ChunkedEllpackSegmentView< IndexType, Organization >;
      using ChunkedEllpackSliceInfoType = detail::ChunkedEllpackSliceInfo< IndexType >;
      using ChunkedEllpackSliceInfoAllocator = typename Allocators::Default< Device >::template Allocator< ChunkedEllpackSliceInfoType >;
      using ChunkedEllpackSliceInfoContainer = Containers::Array< typename noaTNL::copy_const< ChunkedEllpackSliceInfoType >::template from< Index >::type, DeviceType, IndexType, ChunkedEllpackSliceInfoAllocator >;
      using ChunkedEllpackSliceInfoContainerView = typename ChunkedEllpackSliceInfoContainer::ViewType;

      static constexpr bool havePadding() { return true; };

      __cuda_callable__
      ChunkedEllpackView() = default;

      __cuda_callable__
      ChunkedEllpackView( const IndexType size,
                          const IndexType storageSize,
                          const IndexType chunksInSlice,
                          const IndexType desiredChunkSize,
                          const OffsetsView& rowToChunkMapping,
                          const OffsetsView& rowToSliceMapping,
                          const OffsetsView& chunksToSegmentsMapping,
                          const OffsetsView& rowPointers,
                          const ChunkedEllpackSliceInfoContainerView& slices,
                          const IndexType numberOfSlices );

      __cuda_callable__
      ChunkedEllpackView( const IndexType size,
                          const IndexType storageSize,
                          const IndexType chunksInSlice,
                          const IndexType desiredChunkSize,
                          const OffsetsView&& rowToChunkMapping,
                          const OffsetsView&& rowToSliceMapping,
                          const OffsetsView&& chunksToSegmentsMapping,
                          const OffsetsView&& rowPointers,
                          const ChunkedEllpackSliceInfoContainerView&& slices,
                          const IndexType numberOfSlices );

      __cuda_callable__
      ChunkedEllpackView( const ChunkedEllpackView& chunked_ellpack_view ) = default;

      __cuda_callable__
      ChunkedEllpackView( ChunkedEllpackView&& chunked_ellpack_view ) = default;

      static String getSerializationType();

      static String getSegmentsType();

      __cuda_callable__
      ViewType getView();

      __cuda_callable__
      const ConstViewType getConstView() const;

      /**
       * \brief Number of segments.
       */
      __cuda_callable__
      IndexType getSegmentsCount() const;

      /***
       * \brief Returns size of the segment number \r segmentIdx
       */
      __cuda_callable__
      IndexType getSegmentSize( const IndexType segmentIdx ) const;

      /***
       * \brief Returns number of elements managed by all segments.
       */
      __cuda_callable__
      IndexType getSize() const;

      /***
       * \brief Returns number of elements that needs to be allocated.
       */
      __cuda_callable__
      IndexType getStorageSize() const;

      __cuda_callable__
      IndexType getGlobalIndex( const Index segmentIdx, const Index localIdx ) const;

      __cuda_callable__
      SegmentViewType getSegmentView( const IndexType segmentIdx ) const;

      /***
       * \brief Go over all segments and for each segment element call
       * function 'f' with arguments 'args'. The return type of 'f' is bool.
       * When its true, the for-loop continues. Once 'f' returns false, the for-loop
       * is terminated.
       */
      template< typename Function >
      void forElements( IndexType begin, IndexType end, Function&& f ) const;

      template< typename Function >
      void forAllElements( Function&& f ) const;

      template< typename Function >
      void forSegments( IndexType begin, IndexType end, Function&& f ) const;

      template< typename Function >
      void forAllSegments( Function&& f ) const;

      /***
       * \brief Go over all segments and perform a reduction in each of them.
       */
      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
      void reduceSegments( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const;

      template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
      void reduceAllSegments( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const;

      ChunkedEllpackView& operator=( const ChunkedEllpackView& view );

      void save( File& file ) const;

      template< typename Fetch >
      SegmentsPrinter< ChunkedEllpackView, Fetch > print( Fetch&& fetch ) const;

      void printStructure( std::ostream& str ) const;

   protected:

#ifdef HAVE_CUDA
      template< typename Fetch,
                typename Reduction,
                typename ResultKeeper,
                typename Real >
      __device__
      void reduceSegmentsKernelWithAllParameters( IndexType gridIdx,
                                                     IndexType first,
                                                     IndexType last,
                                                     Fetch fetch,
                                                     Reduction reduction,
                                                     ResultKeeper keeper,
                                                     Real zero ) const;

      template< typename Fetch,
                typename Reduction,
                typename ResultKeeper,
                typename Real >
      __device__
      void reduceSegmentsKernel( IndexType gridIdx,
                                    IndexType first,
                                    IndexType last,
                                    Fetch fetch,
                                    Reduction reduction,
                                    ResultKeeper keeper,
                                    Real zero ) const;
#endif

      IndexType size = 0, storageSize = 0, numberOfSlices = 0;

      IndexType chunksInSlice = 256, desiredChunkSize = 16;

      /**
       * For each segment, this keeps index of the slice which contains the
       * segment.
       */
      OffsetsView rowToSliceMapping;

      /**
       * For each row, this keeps index of the first chunk within a slice.
       */
      OffsetsView rowToChunkMapping;

      OffsetsView chunksToSegmentsMapping;

      /**
       * Keeps index of the first segment index.
       */
      OffsetsView rowPointers;

      ChunkedEllpackSliceInfoContainerView slices;

#ifdef HAVE_CUDA
      template< typename View_,
                typename Index_,
                typename Fetch_,
                typename Reduction_,
                typename ResultKeeper_,
                typename Real_ >
      friend __global__
      void ChunkedEllpackreduceSegmentsKernel( View_ chunkedEllpack,
                                                  Index_ gridIdx,
                                                  Index_ first,
                                                  Index_ last,
                                                  Fetch_ fetch,
                                                  Reduction_ reduction,
                                                  ResultKeeper_ keeper,
                                                  Real_ zero );

      template< typename Index_, typename Fetch_, bool B_ >
      friend struct detail::ChunkedEllpackreduceSegmentsDispatcher;
#endif
};
      } // namespace Segments
   }  // namespace Algorithms
} // namespace noaTNL

#include <noa/3rdparty/TNL/Algorithms/Segments/ChunkedEllpackView.hpp>
