// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Allocators/Default.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/ChunkedEllpackView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/SegmentView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/SegmentsPrinting.h>

namespace noa::TNL {
namespace Algorithms {
namespace Segments {

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index >,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization() >
class ChunkedEllpack
{
public:
   using DeviceType = Device;
   using IndexType = std::remove_const_t< Index >;
   using OffsetsContainer = Containers::Vector< Index, DeviceType, IndexType, IndexAllocator >;
   static constexpr ElementsOrganization
   getOrganization()
   {
      return Organization;
   }
   using ViewType = ChunkedEllpackView< Device, Index, Organization >;
   template< typename Device_, typename Index_ >
   using ViewTemplate = ChunkedEllpackView< Device_, Index_, Organization >;
   using ConstViewType = typename ViewType::ConstViewType;
   using SegmentViewType = typename ViewType::SegmentViewType;
   using ChunkedEllpackSliceInfoType = typename ViewType::ChunkedEllpackSliceInfoType;
   using ChunkedEllpackSliceInfoAllocator =
      typename Allocators::Default< Device >::template Allocator< ChunkedEllpackSliceInfoType >;
   using ChunkedEllpackSliceInfoContainer =
      Containers::Array< typename TNL::copy_const< ChunkedEllpackSliceInfoType >::template from< Index >::type,
                         DeviceType,
                         IndexType,
                         ChunkedEllpackSliceInfoAllocator >;

   static constexpr bool
   havePadding()
   {
      return true;
   }

   ChunkedEllpack() = default;

   template< typename SizesContainer >
   ChunkedEllpack( const SizesContainer& sizes );

   template< typename ListIndex >
   ChunkedEllpack( const std::initializer_list< ListIndex >& segmentsSizes );

   ChunkedEllpack( const ChunkedEllpack& segments ) = default;

   ChunkedEllpack( ChunkedEllpack&& segments ) noexcept = default;

   static std::string
   getSerializationType();

   static String
   getSegmentsType();

   ViewType
   getView();

   ConstViewType
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

   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
   void
   reduceAllSegments( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const;

   ChunkedEllpack&
   operator=( const ChunkedEllpack& source ) = default;

   template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
   ChunkedEllpack&
   operator=( const ChunkedEllpack< Device_, Index_, IndexAllocator_, Organization_ >& source );

   void
   save( File& file ) const;

   void
   load( File& file );

   template< typename Fetch >
   SegmentsPrinter< ChunkedEllpack, Fetch >
   print( Fetch&& fetch ) const;

   void
   printStructure( std::ostream& str );  // TODO const;

protected:
   template< typename SegmentsSizes >
   void
   resolveSliceSizes( SegmentsSizes& rowLengths );

   template< typename SegmentsSizes >
   bool
   setSlice( SegmentsSizes& rowLengths, IndexType sliceIdx, IndexType& elementsToAllocation );

   IndexType size = 0, storageSize = 0;

   IndexType chunksInSlice = 256, desiredChunkSize = 16;

   /**
    * For each segment, this keeps index of the slice which contains the
    * segment.
    */
   OffsetsContainer rowToSliceMapping;

   /**
    * For each row, this keeps index of the first chunk within a slice.
    */
   OffsetsContainer rowToChunkMapping;

   OffsetsContainer chunksToSegmentsMapping;

   /**
    * Keeps index of the first segment index.
    */
   OffsetsContainer rowPointers;

   ChunkedEllpackSliceInfoContainer slices;

   IndexType numberOfSlices = 0;

   template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
   friend class ChunkedEllpack;
};

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization >
std::ostream&
operator<<( std::ostream& str, const ChunkedEllpack< Device, Index, IndexAllocator, Organization >& segments )
{
   return printSegments( segments, str );
}

}  // namespace Segments
}  // namespace Algorithms
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/ChunkedEllpack.hpp>
