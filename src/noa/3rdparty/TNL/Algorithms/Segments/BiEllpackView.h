// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/ElementsOrganization.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/BiEllpackSegmentView.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/detail/BiEllpack.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/SegmentsPrinting.h>

namespace noaTNL {
   namespace Algorithms {
      namespace Segments {


template< typename Device,
          typename Index,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int WarpSize = 32 >
class BiEllpackView
{
   public:

      using DeviceType = Device;
      using IndexType = std::remove_const_t< Index >;
      using OffsetsView = typename Containers::VectorView< Index, DeviceType, IndexType >;
      using ConstOffsetsView = typename OffsetsView::ConstViewType;
      using ViewType = BiEllpackView;
      template< typename Device_, typename Index_ >
      using ViewTemplate = BiEllpackView< Device_, Index_, Organization, WarpSize >;
      using ConstViewType = BiEllpackView< Device, std::add_const_t< Index >, Organization, WarpSize >;
      using SegmentViewType = BiEllpackSegmentView< IndexType, Organization, WarpSize >;

      static constexpr bool havePadding() { return true; };

      __cuda_callable__
      BiEllpackView() = default;

      __cuda_callable__
      BiEllpackView( const IndexType size,
                     const IndexType storageSize,
                     const IndexType virtualRows,
                     const OffsetsView& rowPermArray,
                     const OffsetsView& groupPointers );

      __cuda_callable__
      BiEllpackView( const IndexType size,
                     const IndexType storageSize,
                     const IndexType virtualRows,
                     const OffsetsView&& rowPermArray,
                     const OffsetsView&& groupPointers );

      __cuda_callable__
      BiEllpackView( const BiEllpackView& chunked_ellpack_view ) = default;

      __cuda_callable__
      BiEllpackView( BiEllpackView&& chunked_ellpack_view ) = default;

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
      void forElements( IndexType first, IndexType last, Function&& f ) const;

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

      BiEllpackView& operator=( const BiEllpackView& view );

      void save( File& file ) const;

      void load( File& file );

      template< typename Fetch >
      SegmentsPrinter< BiEllpackView, Fetch > print( Fetch&& fetch ) const;

      void printStructure( std::ostream& str ) const;

   protected:

      static constexpr int getWarpSize() { return WarpSize; };

      static constexpr int getLogWarpSize() { return std::log2( WarpSize ); };

      IndexType size = 0, storageSize = 0;

      IndexType virtualRows = 0;

      OffsetsView rowPermArray;

      OffsetsView groupPointers;

#ifdef HAVE_CUDA
      template< typename Fetch,
                typename Reduction,
                typename ResultKeeper,
                typename Real,
                int BlockDim >
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
                typename Real_,
                int BlockDim >
      __device__
      void reduceSegmentsKernel( IndexType gridIdx,
                                    IndexType first,
                                    IndexType last,
                                    Fetch fetch,
                                    Reduction reduction,
                                    ResultKeeper keeper,
                                    Real_ zero ) const;

      template< typename View_,
                typename Index_,
                typename Fetch_,
                typename Reduction_,
                typename ResultKeeper_,
                typename Real_,
                int BlockDim >
      friend __global__
      void BiEllpackreduceSegmentsKernel( View_ chunkedEllpack,
                                             Index_ gridIdx,
                                             Index_ first,
                                             Index_ last,
                                             Fetch_ fetch,
                                             Reduction_ reduction,
                                             ResultKeeper_ keeper,
                                             Real_ zero );

      template< typename Index_, typename Fetch_, int BlockDim_, int WarpSize_, bool B_ >
      friend struct detail::BiEllpackreduceSegmentsDispatcher;
#endif
};

template <typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
std::ostream& operator<<( std::ostream& str, const BiEllpackView< Device, Index, Organization, WarpSize >& segments ) { return printSegments( str, segments ); }


      } // namespace Segments
   }  // namespace Algorithms
} // namespace noaTNL

#include <noa/3rdparty/TNL/Algorithms/Segments/BiEllpackView.hpp>
