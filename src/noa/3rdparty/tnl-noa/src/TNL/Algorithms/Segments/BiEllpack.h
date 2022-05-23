// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Allocators/Default.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/BiEllpackView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/SegmentView.h>

namespace noa::TNL {
namespace Algorithms {
namespace Segments {

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int WarpSize = 32 >
class BiEllpack
{
public:
   using DeviceType = Device;
   using IndexType = std::remove_const_t< Index >;
   using OffsetsContainer = Containers::Vector< IndexType, DeviceType, IndexType, IndexAllocator >;
   static constexpr ElementsOrganization
   getOrganization()
   {
      return Organization;
   }
   using ViewType = BiEllpackView< Device, Index, Organization, WarpSize >;
   template< typename Device_, typename Index_ >
   using ViewTemplate = BiEllpackView< Device_, Index_, Organization, WarpSize >;
   using ConstViewType = typename ViewType::ConstViewType;
   using SegmentViewType = typename ViewType::SegmentViewType;

   static constexpr bool
   havePadding()
   {
      return true;
   };

   BiEllpack() = default;

   template< typename SizesContainer >
   BiEllpack( const SizesContainer& sizes );

   template< typename ListIndex >
   BiEllpack( const std::initializer_list< ListIndex >& segmentsSizes );

   BiEllpack( const BiEllpack& segments ) = default;

   BiEllpack( BiEllpack&& segments ) noexcept = default;

   static std::string
   getSerializationType();

   static String
   getSegmentsType();

   ViewType
   getView();

   const ConstViewType
   getConstView() const;

   /**
    * \brief Number of segments.
    */
   __cuda_callable__
   IndexType
   getSegmentsCount() const;

   /**
    * \brief Set sizes of particular segments.
    */
   template< typename SizesHolder = OffsetsContainer >
   void
   setSegmentsSizes( const SizesHolder& sizes );

   void
   reset();

   IndexType
   getSegmentSize( IndexType segmentIdx ) const;

   /**
    * \brief Number segments.
    */
   __cuda_callable__
   IndexType
   getSize() const;

   __cuda_callable__
   IndexType
   getStorageSize() const;

   __cuda_callable__
   IndexType
   getGlobalIndex( IndexType segmentIdx, IndexType localIdx ) const;

   __cuda_callable__
   SegmentViewType
   getSegmentView( IndexType segmentIdx ) const;

   /***
    * \brief Go over all segments and for each segment element call
    * function 'f' with arguments 'args'. The return type of 'f' is bool.
    * When its true, the for-loop continues. Once 'f' returns false, the for-loop
    * is terminated.
    */
   template< typename Function >
   void
   forElements( IndexType first, IndexType last, Function&& f ) const;

   template< typename Function >
   void
   forAllElements( Function&& f ) const;

   template< typename Function >
   void
   forSegments( IndexType begin, IndexType end, Function&& f ) const;

   template< typename Function >
   void
   forAllSegments( Function&& f ) const;

   /***
    * \brief Go over all segments and perform a reduction in each of them.
    */
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
   void
   reduceSegments( IndexType first,
                   IndexType last,
                   Fetch& fetch,
                   const Reduction& reduction,
                   ResultKeeper& keeper,
                   const Real& zero ) const;

   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
   void
   reduceAllSegments( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const;

   BiEllpack&
   operator=( const BiEllpack& source ) = default;

   template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
   BiEllpack&
   operator=( const BiEllpack< Device_, Index_, IndexAllocator_, Organization_, WarpSize >& source );

   void
   save( File& file ) const;

   void
   load( File& file );

   template< typename Fetch >
   SegmentsPrinter< BiEllpack, Fetch >
   print( Fetch&& fetch ) const;

   void
   printStructure( std::ostream& str ) const;

   // TODO: nvcc needs this public because of lambda function used inside
   template< typename SizesHolder = OffsetsContainer >
   void
   performRowBubbleSort( const SizesHolder& segmentsSize );

   // TODO: the same as  above
   template< typename SizesHolder = OffsetsContainer >
   void
   computeColumnSizes( const SizesHolder& segmentsSizes );

protected:
   static constexpr int
   getWarpSize()
   {
      return WarpSize;
   };

   static constexpr int
   getLogWarpSize()
   {
      return std::log2( WarpSize );
   };

   template< typename SizesHolder = OffsetsContainer >
   void
   verifyRowPerm( const SizesHolder& segmentsSizes );

   template< typename SizesHolder = OffsetsContainer >
   void
   verifyRowLengths( const SizesHolder& segmentsSizes );

   IndexType
   getStripLength( IndexType stripIdx ) const;

   IndexType
   getGroupLength( IndexType strip, IndexType group ) const;

   IndexType size = 0, storageSize = 0;

   IndexType virtualRows = 0;

   OffsetsContainer rowPermArray;

   OffsetsContainer groupPointers;

   // TODO: Replace later
   __cuda_callable__
   Index
   power( const IndexType number, const IndexType exponent ) const
   {
      if( exponent >= 0 ) {
         IndexType result = 1;
         for( IndexType i = 0; i < exponent; i++ )
            result *= number;
         return result;
      }
      return 0;
   };

   template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_, int WarpSize_ >
   friend class BiEllpack;
};

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int WarpSize >
std::ostream&
operator<<( std::ostream& str, const BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >& segments )
{
   return printSegments( segments, str );
}

}  // namespace Segments
}  // namespace Algorithms
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/BiEllpack.hpp>
