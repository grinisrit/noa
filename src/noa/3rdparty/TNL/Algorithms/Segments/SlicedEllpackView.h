// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/ElementsOrganization.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/SegmentView.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/SegmentsPrinting.h>

namespace noa::TNL {
   namespace Algorithms {
      namespace Segments {

template< typename Device,
          typename Index,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int SliceSize = 32 >
class SlicedEllpackView
{
   public:

      using DeviceType = Device;
      using IndexType = std::remove_const_t< Index >;
      using OffsetsView = typename Containers::VectorView< Index, DeviceType, IndexType >;
      static constexpr int getSliceSize() { return SliceSize; }
      static constexpr ElementsOrganization getOrganization() { return Organization; }
      template< typename Device_, typename Index_ >
      using ViewTemplate = SlicedEllpackView< Device_, Index_, Organization, SliceSize >;
      using ViewType = SlicedEllpackView;
      using ConstViewType = SlicedEllpackView< Device, std::add_const_t< Index >, Organization, SliceSize >;
      using SegmentViewType = SegmentView< IndexType, Organization >;

      static constexpr bool havePadding() { return true; };

      __cuda_callable__
      SlicedEllpackView();

      __cuda_callable__
      SlicedEllpackView( IndexType size,
                         IndexType alignedSize,
                         IndexType segmentsCount,
                         OffsetsView&& sliceOffsets,
                         OffsetsView&& sliceSegmentSizes );

      __cuda_callable__
      SlicedEllpackView( const SlicedEllpackView& slicedEllpackView ) = default;

      __cuda_callable__
      SlicedEllpackView( SlicedEllpackView&& slicedEllpackView ) = default;

      static String getSerializationType();

      static String getSegmentsType();

      __cuda_callable__
      ViewType getView();

      __cuda_callable__
      const ConstViewType getConstView() const;

      __cuda_callable__
      IndexType getSegmentsCount() const;

      __cuda_callable__
      IndexType getSegmentSize( const IndexType segmentIdx ) const;

      /**
       * \brief Number segments.
       */
      __cuda_callable__
      IndexType getSize() const;

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

      SlicedEllpackView& operator=( const SlicedEllpackView& view );

      void save( File& file ) const;

      void load( File& file );

      template< typename Fetch >
      SegmentsPrinter< SlicedEllpackView, Fetch > print( Fetch&& fetch ) const;

   protected:

      IndexType size, alignedSize, segmentsCount;

      OffsetsView sliceOffsets, sliceSegmentSizes;
};

template <typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
std::ostream& operator<<( std::ostream& str, const SlicedEllpackView< Device, Index, Organization, SliceSize >& segments ) { return printSegments( str, segments ); }

      } // namespace Segements
   }  // namespace Algorithms
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Algorithms/Segments/SlicedEllpackView.hpp>
