// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/EllpackView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/SegmentView.h>

namespace noa::TNL {
namespace Algorithms {
namespace Segments {

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          ElementsOrganization Organization = Segments::DefaultElementsOrganization< Device >::getOrganization(),
          int Alignment = 32 >
class Ellpack
{
public:
   using DeviceType = Device;
   using IndexType = std::remove_const_t< Index >;
   static constexpr int
   getAlignment()
   {
      return Alignment;
   }
   static constexpr ElementsOrganization
   getOrganization()
   {
      return Organization;
   }
   using OffsetsContainer = Containers::Vector< IndexType, DeviceType, IndexType >;
   using SegmentsSizes = OffsetsContainer;
   template< typename Device_, typename Index_ >
   using ViewTemplate = EllpackView< Device_, Index_, Organization, Alignment >;
   using ViewType = EllpackView< Device, Index, Organization, Alignment >;
   using ConstViewType = typename ViewType::ConstViewType;
   using SegmentViewType = SegmentView< IndexType, Organization >;

   static constexpr bool
   havePadding()
   {
      return true;
   }

   Ellpack() = default;

   template< typename SizesContainer >
   Ellpack( const SizesContainer& sizes );

   template< typename ListIndex >
   Ellpack( const std::initializer_list< ListIndex >& segmentsSizes );

   Ellpack( IndexType segmentsCount, IndexType segmentSize );

   Ellpack( const Ellpack& segments ) = default;

   Ellpack( Ellpack&& segments ) noexcept = default;

   static std::string
   getSerializationType();

   static String
   getSegmentsType();

   ViewType
   getView();

   ConstViewType
   getConstView() const;

   /**
    * \brief Set sizes of particular segments.
    */
   template< typename SizesHolder = OffsetsContainer >
   void
   setSegmentsSizes( const SizesHolder& sizes );

   void
   setSegmentsSizes( IndexType segmentsCount, IndexType segmentSize );

   void
   reset();

   /**
    * \brief Number segments.
    */
   __cuda_callable__
   IndexType
   getSegmentsCount() const;

   __cuda_callable__
   IndexType
   getSegmentSize( IndexType segmentIdx ) const;

   __cuda_callable__
   IndexType
   getSize() const;

   /**
    * \brief Returns number of elements that needs to be allocated by a container connected to this segments.
    *
    * \return size of container connected to this segments.
    */
   __cuda_callable__
   IndexType
   getStorageSize() const;

   __cuda_callable__
   IndexType
   getGlobalIndex( Index segmentIdx, Index localIdx ) const;

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

   /**
    * \brief Call \e reduceSegments for all segments.
    */
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
   void
   reduceAllSegments( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const;

   Ellpack&
   operator=( const Ellpack& source ) = default;

   template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_, int Alignment_ >
   Ellpack&
   operator=( const Ellpack< Device_, Index_, IndexAllocator_, Organization_, Alignment_ >& source );

   void
   save( File& file ) const;

   void
   load( File& file );

   template< typename Fetch >
   SegmentsPrinter< Ellpack, Fetch >
   print( Fetch&& fetch ) const;

protected:
   IndexType segmentSize = 0;
   IndexType size = 0;
   IndexType alignedSize = 0;
};

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
std::ostream&
operator<<( std::ostream& str, const Ellpack< Device, Index, IndexAllocator, Organization, Alignment >& segments )
{
   return printSegments( segments, str );
}

}  // namespace Segments
}  // namespace Algorithms
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/Ellpack.hpp>
