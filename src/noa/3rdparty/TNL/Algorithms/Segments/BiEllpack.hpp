// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <math.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Algorithms/Segments/BiEllpack.h>
#include <TNL/Algorithms/Segments/Ellpack.h>

namespace TNL {
   namespace Algorithms {
      namespace Segments {

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename SizesContainer >
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
BiEllpack( const SizesContainer& segmentsSizes )
{
   this->setSegmentsSizes( segmentsSizes );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename ListIndex >
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
BiEllpack( const std::initializer_list< ListIndex >& segmentsSizes )
{
   this->setSegmentsSizes( Containers::Vector< IndexType, DeviceType, IndexType >( segmentsSizes ) );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
String
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
getSerializationType()
{
   // FIXME: the serialized data DEPEND on the Organization and WarpSize parameters, so it should be reflected in the serialization type
   return "BiEllpack< [any_device], " + TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
String
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
getSegmentsType()
{
   return ViewType::getSegmentsType();
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
typename BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::ViewType
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
getView()
{
   return ViewType( size, storageSize, virtualRows, rowPermArray.getView(), groupPointers.getView() );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
auto BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
getConstView() const -> const ConstViewType
{
   return ConstViewType( size, storageSize, virtualRows, rowPermArray.getConstView(), groupPointers.getConstView() );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
auto BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
getSegmentsCount() const -> IndexType
{
   return this->size;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename SizesHolder >
void BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
performRowBubbleSort( const SizesHolder& segmentsSizes )
{
   if( segmentsSizes.getSize() == 0 )
      return;

   this->rowPermArray.forAllElements( [] __cuda_callable__ ( const IndexType idx, IndexType& value ) { value = idx; } );

   //if( std::is_same< DeviceType, Devices::Host >::value )
   {
      IndexType strips = this->virtualRows / getWarpSize();
      for( IndexType i = 0; i < strips; i++ )
      {
         IndexType begin = i * getWarpSize();
         IndexType end = ( i + 1 ) * getWarpSize() - 1;
         if(this->getSize() - 1 < end)
            end = this->getSize() - 1;
         bool sorted = false;
         IndexType permIndex1, permIndex2, offset = 0;
         while( !sorted )
         {
            sorted = true;
            for( IndexType j = begin + offset; j < end - offset; j++ )
            {
               for( IndexType k = begin; k < end + 1; k++ )
               {
                  if( this->rowPermArray.getElement( k ) == j )
                     permIndex1 = k;
                  if( this->rowPermArray.getElement( k ) == j + 1 )
                     permIndex2 = k;
               }
               if( segmentsSizes.getElement( permIndex1 ) < segmentsSizes.getElement( permIndex2 ) )
               {
                  IndexType temp = this->rowPermArray.getElement( permIndex1 );
                  this->rowPermArray.setElement( permIndex1, this->rowPermArray.getElement( permIndex2 ) );
                  this->rowPermArray.setElement( permIndex2, temp );
                  sorted = false;
               }
            }
            for( IndexType j = end - 1 - offset; j > begin + offset; j-- )
            {
               for( IndexType k = begin; k < end + 1; k++ )
               {
                  if( this->rowPermArray.getElement( k ) == j )
                     permIndex1 = k;
                  if( this->rowPermArray.getElement( k ) == j - 1 )
                     permIndex2 = k;
               }
               if( segmentsSizes.getElement( permIndex2 ) < segmentsSizes.getElement( permIndex1 ) )
               {
                  IndexType temp = this->rowPermArray.getElement( permIndex1 );
                  this->rowPermArray.setElement( permIndex1, this->rowPermArray.getElement( permIndex2 ) );
                  this->rowPermArray.setElement( permIndex2, temp );
                  sorted = false;
               }
            }
            offset++;
         }
      }
   }
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename SizesHolder >
void BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
computeColumnSizes( const SizesHolder& segmentsSizes )
{
   IndexType numberOfStrips = this->virtualRows / getWarpSize();
   auto groupPointersView = this->groupPointers.getView();
   auto segmentsPermutationView = this->rowPermArray.getView();
   auto segmentsSizesView = segmentsSizes.getConstView();
   const IndexType size = this->getSize();
   auto createGroups = [=] __cuda_callable__ ( const IndexType strip ) mutable {
      IndexType firstSegment = strip * getWarpSize();
      IndexType groupBegin = strip * ( getLogWarpSize() + 1 );
      IndexType emptyGroups = 0;

      ////
      // The last strip can be shorter
      if( strip == numberOfStrips - 1 )
      {
         IndexType segmentsCount = size - firstSegment;
         while( segmentsCount <= TNL::pow( 2, getLogWarpSize() - 1 - emptyGroups ) - 1 )
            emptyGroups++;
         for( IndexType group = groupBegin; group < groupBegin + emptyGroups; group++ )
            groupPointersView[ group ] = 0;
      }

      IndexType allocatedColumns = 0;
      for( IndexType groupIdx = emptyGroups; groupIdx < getLogWarpSize(); groupIdx++ )
      {
         IndexType segmentIdx = TNL::pow( 2, getLogWarpSize() - 1 - groupIdx ) - 1;
         IndexType permSegm = 0;
         while( segmentsPermutationView[ permSegm + firstSegment ] != segmentIdx + firstSegment )
            permSegm++;
         const IndexType groupWidth = segmentsSizesView[ permSegm + firstSegment ] - allocatedColumns;
         const IndexType groupHeight = TNL::pow( 2, getLogWarpSize() - groupIdx );
         const IndexType groupSize = groupWidth * groupHeight;
         allocatedColumns = segmentsSizesView[ permSegm + firstSegment ];
         groupPointersView[ groupIdx + groupBegin ] = groupSize;
      }
   };
   Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, this->virtualRows / getWarpSize(), createGroups );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename SizesHolder >
void BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
verifyRowPerm( const SizesHolder& segmentsSizes )
{
   bool ok = true;
   IndexType numberOfStrips = this->virtualRows / getWarpSize();
   for( IndexType strip = 0; strip < numberOfStrips; strip++ )
   {
      IndexType begin = strip * getWarpSize();
      IndexType end = ( strip + 1 ) * getWarpSize();
      if( this->getSize() < end )
         end = this->getSize();
      for( IndexType i = begin; i < end - 1; i++ )
      {
         IndexType permIndex1, permIndex2;
         bool first = false;
         bool second = false;
         for( IndexType j = begin; j < end; j++ )
         {
            if( this->rowPermArray.getElement( j ) == i )
            {
               permIndex1 = j;
               first = true;
            }
            if( this->rowPermArray.getElement( j ) == i + 1 )
            {
               permIndex2 = j;
               second = true;
            }
         }
         if( !first || !second )
            std::cout << "Wrong permutation!" << std::endl;
         if( segmentsSizes.getElement( permIndex1 ) >= segmentsSizes.getElement( permIndex2 ) )
            continue;
         else
            ok = false;
      }
   }
   if( !ok )
      throw( std::logic_error( "Segments permutation verification failed." ) );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename SizesHolder >
void BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
verifyRowLengths( const SizesHolder& segmentsSizes )
{
   std::cerr << "segmentsSizes = " << segmentsSizes << std::endl;
   for( IndexType segmentIdx = 0; segmentIdx < this->getSize(); segmentIdx++ )
   {
      const IndexType strip = segmentIdx / getWarpSize();
      const IndexType stripLength = this->getStripLength( strip );
      const IndexType groupBegin = ( getLogWarpSize() + 1 ) * strip;
      const IndexType rowStripPerm = this->rowPermArray.getElement( segmentIdx ) - strip * getWarpSize();
      const IndexType begin = this->groupPointers.getElement( groupBegin ) * getWarpSize() + rowStripPerm * stripLength;
      IndexType elementPtr = begin;
      IndexType rowLength = 0;
      const IndexType groupsCount = detail::BiEllpack< Index, Device, Organization, WarpSize >::getActiveGroupsCount( this->rowPermArray.getConstView(), segmentIdx );
      for( IndexType group = 0; group < groupsCount; group++ )
      {
         std::cerr << "groupIdx = " << group << " groupLength = " << this->getGroupLength( strip, group ) << std::endl;
         for( IndexType i = 0; i < this->getGroupLength( strip, group ); i++ )
         {
            IndexType biElementPtr = elementPtr;
            for( IndexType j = 0; j < this->power( 2, group ); j++ )
            {
               rowLength++;
               biElementPtr += this->power( 2, getLogWarpSize() - group ) * stripLength;
            }
            elementPtr++;
         }
      }
      if( segmentsSizes.getElement( segmentIdx ) > rowLength )
         throw( std::logic_error( "Segments capacities verification failed." ) );
   }
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename SizesHolder >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
setSegmentsSizes( const SizesHolder& segmentsSizes )
{
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      this->size = segmentsSizes.getSize();
      if( this->size % WarpSize != 0 )
         this->virtualRows = this->size + getWarpSize() - ( this->size % getWarpSize() );
      else
         this->virtualRows = this->size;
      IndexType strips = this->virtualRows / getWarpSize();
      this->rowPermArray.setSize( this->size );
      this->groupPointers.setSize( strips * ( getLogWarpSize() + 1 ) + 1 );
      this->groupPointers = 0;

      this->performRowBubbleSort( segmentsSizes );
      this->computeColumnSizes( segmentsSizes );

      inplaceExclusiveScan( this->groupPointers );

      this->verifyRowPerm( segmentsSizes );
      //this->verifyRowLengths( segmentsSizes ); // TODO: I am not sure what this test is doing.
      this->storageSize =  getWarpSize() * this->groupPointers.getElement( strips * ( getLogWarpSize() + 1 ) );
   }
   else
   {
      BiEllpack< Devices::Host, Index, typename Allocators::Default< Devices::Host >::template Allocator< IndexType >, Organization > hostSegments;
      Containers::Vector< IndexType, Devices::Host, IndexType > hostSegmentsSizes;
      hostSegmentsSizes = segmentsSizes;
      hostSegments.setSegmentsSizes( hostSegmentsSizes );
      *this = hostSegments;
   }
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
reset()
{
   this->size = 0;
   this->storageSize = 0;
   this->virtualRows = 0;
   rowPermArray.reset();
   groupPointers.reset();
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
auto BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   return detail::BiEllpack< IndexType, DeviceType, Organization >::getSegmentSize(
      rowPermArray.getConstView(),
      groupPointers.getConstView(),
      segmentIdx );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
__cuda_callable__ auto BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
getSize() const -> IndexType
{
   return this->size;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
__cuda_callable__ auto BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
getStorageSize() const -> IndexType
{
   return this->storageSize;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
__cuda_callable__ auto BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
getGlobalIndex( const IndexType segmentIdx, const IndexType localIdx ) const -> IndexType
{
      return detail::BiEllpack< IndexType, DeviceType, Organization >::getGlobalIndex(
         rowPermArray.getConstView(),
         groupPointers.getConstView(),
         segmentIdx,
         localIdx );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
__cuda_callable__ auto BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename Function >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
forElements( IndexType first, IndexType last, Function&& f ) const
{
   this->getConstView().forElements( first, last, f );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename Function >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
forAllElements( Function&& f ) const
{
   this->forElements( 0, this->getSegmentsCount(), f );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename Function >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
forSegments( IndexType begin, IndexType end, Function&& f ) const
{
   this->getConstView().forSegments( begin, end, f );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename Function >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
forAllSegments( Function&& f ) const
{
   this->getConstView().forAllSegments( f );
}


template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
reduceSegments( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const
{
   this->getConstView().reduceSegments( first, last, fetch, reduction, keeper, zero );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
reduceAllSegments( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const
{
   this->reduceSegments( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_ >
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >&
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
operator=( const BiEllpack< Device_, Index_, IndexAllocator_, Organization_, WarpSize >& source )
{
   this->size = source.size;
   this->storageSize = source.storageSize;
   this->virtualRows = source.virtualRows;
   this->rowPermArray = source.rowPermArray;
   this->groupPointers = source.groupPointers;
   return *this;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
save( File& file ) const
{
   file.save( &this->size );
   file.save( &this->storageSize );
   file.save( &this->virtualRows );
   file << this->rowPermArray
        << this->groupPointers;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
load( File& file )
{
   file.load( &this->size );
   file.load( &this->storageSize );
   file.load( &this->virtualRows );
   file >> this->rowPermArray
        >> this->groupPointers;
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
      template< typename Fetch >
auto
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
print( Fetch&& fetch ) const -> SegmentsPrinter< BiEllpack, Fetch >
{
   return SegmentsPrinter< BiEllpack, Fetch >( *this, fetch );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
void
BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
printStructure( std::ostream& str ) const
{
   this->view.printStructure( str );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
auto BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
getStripLength( const IndexType stripIdx ) const -> IndexType
{
   return detail::BiEllpack< Index, Device, Organization, WarpSize >::getStripLength( this->groupPointers.getConstView(), stripIdx );
}

template< typename Device,
          typename Index,
          typename IndexAllocator,
          ElementsOrganization Organization,
          int WarpSize >
auto BiEllpack< Device, Index, IndexAllocator, Organization, WarpSize >::
getGroupLength( const IndexType strip, const IndexType group ) const -> IndexType
{
   return this->groupPointers.getElement( strip * ( getLogWarpSize() + 1 ) + group + 1 )
           - this->groupPointers.getElement( strip * ( getLogWarpSize() + 1 ) + group );
}

      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
