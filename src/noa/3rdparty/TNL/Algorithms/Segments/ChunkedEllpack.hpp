// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/TNL/Algorithms/scan.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/ChunkedEllpack.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/Ellpack.h>

namespace noa::TNL {
   namespace Algorithms {
      namespace Segments {

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
   template< typename SizesContainer >
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
ChunkedEllpack( const SizesContainer& segmentsSizes )
{
   this->setSegmentsSizes( segmentsSizes );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
   template< typename ListIndex >
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
ChunkedEllpack( const std::initializer_list< ListIndex >& segmentsSizes )
{
   this->setSegmentsSizes( Containers::Vector< IndexType, DeviceType, IndexType >( segmentsSizes ) );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
String
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
getSerializationType()
{
   // FIXME: the serialized data DEPEND on the Organization parameter, so it should be reflected in the serialization type
   return "ChunkedEllpack< [any_device], " + noa::TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
String
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
getSegmentsType()
{
   return ViewType::getSegmentsType();
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
typename ChunkedEllpack< Device, Index, IndexAllocator, Organization >::ViewType
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
getView()
{
   return ViewType( size, storageSize, chunksInSlice, desiredChunkSize,
                    rowToChunkMapping.getView(),
                    rowToSliceMapping.getView(),
                    chunksToSegmentsMapping.getView(),
                    rowPointers.getView(),
                    slices.getView(),
                    numberOfSlices );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
auto ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
getConstView() const -> const ConstViewType
{
   return ConstViewType( size, storageSize, chunksInSlice, desiredChunkSize,
                         rowToChunkMapping.getConstView(),
                         rowToSliceMapping.getConstView(),
                         chunksToSegmentsMapping.getConstView(),
                         rowPointers.getConstView(),
                         slices.getConstView(),
                         numberOfSlices );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
   template< typename SegmentsSizes >
void
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
resolveSliceSizes( SegmentsSizes& segmentsSizes )
{
   /****
    * Iterate over rows and allocate slices so that each slice has
    * approximately the same number of allocated elements
    */
   const IndexType desiredElementsInSlice =
            this->chunksInSlice * this->desiredChunkSize;

   IndexType segmentIdx( 0 ),
             sliceSize( 0 ),
             allocatedElementsInSlice( 0 );
   numberOfSlices = 0;
   while( segmentIdx < segmentsSizes.getSize() )
   {
      /****
       * Add one row to the current slice until we reach the desired
       * number of elements in a slice.
       */
      allocatedElementsInSlice += segmentsSizes[ segmentIdx ];
      sliceSize++;
      segmentIdx++;
      if( allocatedElementsInSlice < desiredElementsInSlice  )
          if( segmentIdx < segmentsSizes.getSize() && sliceSize < chunksInSlice ) continue;
      TNL_ASSERT( sliceSize >0, );
      this->slices[ numberOfSlices ].size = sliceSize;
      this->slices[ numberOfSlices ].firstSegment = segmentIdx - sliceSize;
      this->slices[ numberOfSlices ].pointer = allocatedElementsInSlice; // this is only temporary
      sliceSize = 0;
      numberOfSlices++;
      allocatedElementsInSlice = 0;
   }
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
   template< typename SegmentsSizes >
bool
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
setSlice( SegmentsSizes& rowLengths,
          const IndexType sliceIndex,
          IndexType& elementsToAllocation )
{
   /****
    * Now, compute the number of chunks per each row.
    * Each row get one chunk by default.
    * Then each row will get additional chunks w.r. to the
    * number of the elements in the row. If there are some
    * free chunks left, repeat it again.
    */
   const IndexType sliceSize = this->slices[ sliceIndex ].size;
   const IndexType sliceBegin = this->slices[ sliceIndex ].firstSegment;
   const IndexType allocatedElementsInSlice = this->slices[ sliceIndex ].pointer;
   const IndexType sliceEnd = sliceBegin + sliceSize;

   IndexType freeChunks = this->chunksInSlice - sliceSize;
   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
      this->rowToChunkMapping.setElement( i, 1 );

   int totalAddedChunks( 0 );
   int maxRowLength( rowLengths[ sliceBegin ] );
   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
   {
      double rowRatio( 0.0 );
      if( allocatedElementsInSlice != 0 )
         rowRatio = ( double ) rowLengths[ i ] / ( double ) allocatedElementsInSlice;
      const IndexType addedChunks = freeChunks * rowRatio;
      totalAddedChunks += addedChunks;
      this->rowToChunkMapping[ i ] += addedChunks;
      if( maxRowLength < rowLengths[ i ] )
         maxRowLength = rowLengths[ i ];
   }
   freeChunks -= totalAddedChunks;
   while( freeChunks )
      for( IndexType i = sliceBegin; i < sliceEnd && freeChunks; i++ )
         if( rowLengths[ i ] == maxRowLength )
         {
            this->rowToChunkMapping[ i ]++;
            freeChunks--;
         }

   /****
    * Compute the chunk size
    */
   IndexType maxChunkInSlice( 0 );
   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
   {
      TNL_ASSERT_NE( this->rowToChunkMapping[ i ], 0, "" );
      maxChunkInSlice = noa::TNL::max( maxChunkInSlice,
                              roundUpDivision( rowLengths[ i ], this->rowToChunkMapping[ i ] ) );
   }

   /****
    * Set-up the slice info.
    */
   this->slices[ sliceIndex ].chunkSize = maxChunkInSlice;
   this->slices[ sliceIndex ].pointer = elementsToAllocation;
   elementsToAllocation += this->chunksInSlice * maxChunkInSlice;

   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
      this->rowToSliceMapping[ i ] = sliceIndex;

   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
   {
      this->rowPointers[ i + 1 ] = maxChunkInSlice*rowToChunkMapping[ i ];
      TNL_ASSERT( this->rowPointers[ i ] >= 0,
                 std::cerr << "this->rowPointers[ i ] = " << this->rowPointers[ i ] );
      TNL_ASSERT( this->rowPointers[ i + 1 ] >= 0,
                 std::cerr << "this->rowPointers[ i + 1 ] = " << this->rowPointers[ i + 1 ] );
   }

   /****
    * Finish the row to chunk mapping by computing the prefix sum.
    */
   for( IndexType j = sliceBegin + 1; j < sliceEnd; j++ )
      rowToChunkMapping[ j ] += rowToChunkMapping[ j - 1 ];
   return true;
}


template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
   template< typename SizesHolder >
void
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
setSegmentsSizes( const SizesHolder& segmentsSizes )
{
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      this->size = segmentsSizes.getSize();
      this->slices.setSize( this->size );
      this->rowToChunkMapping.setSize( this->size );
      this->rowToSliceMapping.setSize( this->size );
      this->rowPointers.setSize( this->size + 1 );

      this->resolveSliceSizes( segmentsSizes );
      this->rowPointers.setElement( 0, 0 );
      this->storageSize = 0;
      for( IndexType sliceIndex = 0; sliceIndex < numberOfSlices; sliceIndex++ )
         this->setSlice( segmentsSizes, sliceIndex, storageSize );
      inplaceInclusiveScan( this->rowPointers );
      IndexType chunksCount = this->numberOfSlices * this->chunksInSlice;
      this->chunksToSegmentsMapping.setSize( chunksCount );
      IndexType chunkIdx( 0 );
      for( IndexType segmentIdx = 0; segmentIdx < this->size; segmentIdx++ )
      {
         const IndexType& sliceIdx = rowToSliceMapping[ segmentIdx ];
         IndexType firstChunkOfSegment( 0 );
         if( segmentIdx != slices[ sliceIdx ].firstSegment )
               firstChunkOfSegment = rowToChunkMapping[ segmentIdx - 1 ];

         const IndexType lastChunkOfSegment = rowToChunkMapping[ segmentIdx ];
         const IndexType segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
         for( IndexType i = 0; i < segmentChunksCount; i++ )
            this->chunksToSegmentsMapping[ chunkIdx++ ] = segmentIdx;
      }
   }
   else
   {
      ChunkedEllpack< Devices::Host, Index, typename Allocators::Default< Devices::Host >::template Allocator< Index >, Organization > hostSegments;
      Containers::Vector< IndexType, Devices::Host, IndexType > hostSegmentsSizes;
      hostSegmentsSizes = segmentsSizes;
      hostSegments.setSegmentsSizes( hostSegmentsSizes );
      *this = hostSegments;
   }
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
void
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
reset()
{
   this->size = 0;
   this->storageSize = 0;
   this->rowToSliceMapping.reset();
   this->rowToChunkMapping.reset();
   this->chunksToSegmentsMapping.reset();
   this->rowPointers.reset();
   this->slices.reset();
   this->numberOfSlices = 0;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
__cuda_callable__ auto ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
getSegmentsCount() const -> IndexType
{
   return this->size;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
auto ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getSegmentSize(
      rowToSliceMapping.getConstView(),
      slices.getConstView(),
      rowToChunkMapping.getConstView(),
      segmentIdx );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
__cuda_callable__ auto ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
getSize() const -> IndexType
{
   return this->size;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
__cuda_callable__ auto ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
getStorageSize() const -> IndexType
{
   return this->storageSize;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
__cuda_callable__ auto ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
getGlobalIndex( const Index segmentIdx, const Index localIdx ) const -> IndexType
{
      return detail::ChunkedEllpack< IndexType, DeviceType, Organization >::getGlobalIndex(
         rowToSliceMapping,
         slices,
         rowToChunkMapping,
         chunksInSlice,
         segmentIdx,
         localIdx );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
__cuda_callable__ auto ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
   template< typename Function >
void
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
forElements( IndexType first, IndexType last, Function&& f ) const
{
   this->getConstView().forElements( first, last, f );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
   template< typename Function >
void
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
forAllElements( Function&& f ) const
{
   this->forElements( 0, this->getSegmentsCount(), f );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
   template< typename Function >
void
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
forSegments( IndexType begin, IndexType end, Function&& f ) const
{
   this->getConstView().forSegments( begin, end, f );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
   template< typename Function >
void
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
forAllSegments( Function&& f ) const
{
   this->getConstView().forAllSegments( f );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
void
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
reduceSegments( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const
{
   this->getConstView().reduceSegments( first, last, fetch, reduction, keeper, zero );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
void
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
reduceAllSegments( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const
{
   this->reduceSegments( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
   template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
ChunkedEllpack< Device, Index, IndexAllocator, Organization >&
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
operator=( const ChunkedEllpack< Device_, Index_, IndexAllocator_, Organization_ >& source )
{
   this->size = source.size;
   this->storageSize = source.storageSize;
   this->chunksInSlice = source.chunksInSlice;
   this->desiredChunkSize = source.desiredChunkSize;
   this->rowToChunkMapping = source.rowToChunkMapping;
   this->rowToSliceMapping = source.rowToSliceMapping;
   this->rowPointers = source.rowPointers;
   this->chunksToSegmentsMapping = source.chunksToSegmentsMapping;
   this->slices = source.slices;
   this->numberOfSlices = source.numberOfSlices;
   return *this;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
void
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
save( File& file ) const
{
   file.save( &this->size );
   file.save( &this->storageSize );
   file.save( &this->chunksInSlice );
   file.save( &this->desiredChunkSize );
   file << this->rowToChunkMapping
        << this->rowToSliceMapping
        << this->rowPointers
        << this->chunksToSegmentsMapping
        << this->slices;
   file.save( this->numberOfSlices );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
void
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
load( File& file )
{
   file.load( &this->size );
   file.load( &this->storageSize );
   file.load( &this->chunksInSlice );
   file.load( &this->desiredChunkSize );
   file >> this->rowToChunkMapping
        >> this->rowToSliceMapping
        >> this->chunksToSegmentsMapping
        >> this->rowPointers
        >> this->slices;
   file.load( &this->numberOfSlices );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
      template< typename Fetch >
auto
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
print( Fetch&& fetch ) const -> SegmentsPrinter< ChunkedEllpack, Fetch >
{
   return SegmentsPrinter< ChunkedEllpack, Fetch >( *this, fetch );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization >
void
ChunkedEllpack< Device, Index, IndexAllocator, Organization >::
printStructure( std::ostream& str )
{
   this->getView().printStructure( str );
}

      } // namespace Segments
   }  // namespace Algorithms
} // namespace noa::TNL
