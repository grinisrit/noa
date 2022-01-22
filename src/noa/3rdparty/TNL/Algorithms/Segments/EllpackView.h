// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/SegmentView.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/ElementsOrganization.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/SegmentsPrinting.h>


namespace noaTNL {
   namespace Algorithms {
      namespace Segments {

enum EllpackKernelType { Scalar, Vector, Vector2, Vector4, Vector8, Vector16 };

template< typename Device,
          typename Index,
          ElementsOrganization Organization = Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int Alignment = 32 >
class EllpackView
{
   public:

      using DeviceType = Device;
      using IndexType = std::remove_const_t< Index >;
      static constexpr int getAlignment() { return Alignment; }
      static constexpr ElementsOrganization getOrganization() { return Organization; }
      using OffsetsContainer = Containers::Vector< IndexType, DeviceType, IndexType >;
      using SegmentsSizes = OffsetsContainer;
      template< typename Device_, typename Index_ >
      using ViewTemplate = EllpackView< Device_, Index_, Organization, Alignment >;
      using ViewType = EllpackView;
      using ConstViewType = ViewType;
      using SegmentViewType = SegmentView< IndexType, Organization >;

      static constexpr bool havePadding() { return true; };

      __cuda_callable__
      EllpackView();

      __cuda_callable__
      EllpackView( IndexType segmentsCount, IndexType segmentSize, IndexType alignedSize );

      __cuda_callable__
      EllpackView( IndexType segmentsCount, IndexType segmentSize );

      __cuda_callable__
      EllpackView( const EllpackView& ellpackView ) = default;

      __cuda_callable__
      EllpackView( EllpackView&& ellpackView ) = default;

      static String getSerializationType();

      static String getSegmentsType();

      __cuda_callable__
      ViewType getView();

      __cuda_callable__
      const ConstViewType getConstView() const;

      /**
       * \brief Number segments.
       */
      __cuda_callable__
      IndexType getSegmentsCount() const;

      __cuda_callable__
      IndexType getSegmentSize( const IndexType segmentIdx ) const;

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

      EllpackView& operator=( const EllpackView& view );

      void save( File& file ) const;

      void load( File& file );

      template< typename Fetch >
      SegmentsPrinter< EllpackView, Fetch > print( Fetch&& fetch ) const;

   protected:

      IndexType segmentSize, segmentsCount, alignedSize;
};

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int Alignment >
std::ostream& operator<<( std::ostream& str, const EllpackView< Device, Index, Organization, Alignment >& ellpack ) { return printSegments( str, ellpack ); }

      } // namespace Segments
   }  // namespace Algorithms
} // namespace noaTNL

#include <noa/3rdparty/TNL/Algorithms/Segments/EllpackView.hpp>
