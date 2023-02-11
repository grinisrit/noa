// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/Ellpack.h>

namespace noa::TNL {
namespace Algorithms {
namespace Segments {

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
template< typename SizesContainer >
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::Ellpack( const SizesContainer& segmentsSizes )
{
   this->setSegmentsSizes( segmentsSizes );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
template< typename ListIndex >
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::Ellpack(
   const std::initializer_list< ListIndex >& segmentsSizes )
{
   this->setSegmentsSizes( Containers::Vector< IndexType, DeviceType, IndexType >( segmentsSizes ) );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::Ellpack( const IndexType segmentsCount,
                                                                            const IndexType segmentSize )
{
   this->setSegmentsSizes( segmentsCount, segmentSize );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
std::string
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::getSerializationType()
{
   // FIXME: the serialized data DEPEND on the Organization and Alignment parameters, so it should be reflected in the
   // serialization type
   return "Ellpack< [any_device], " + TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
String
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::getSegmentsType()
{
   return ViewType::getSegmentsType();
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
auto
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::getView() -> ViewType
{
   return { size, segmentSize, alignedSize };
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
auto
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::getConstView() const -> ConstViewType
{
   return { size, segmentSize, alignedSize };
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
template< typename SizesHolder >
void
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::setSegmentsSizes( const SizesHolder& sizes )
{
   this->segmentSize = max( sizes );
   this->size = sizes.getSize();
   if( Organization == RowMajorOrder )
      this->alignedSize = this->size;
   else
      this->alignedSize = roundUpDivision( size, this->getAlignment() ) * this->getAlignment();
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
void
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::reset()
{
   this->segmentSize = 0;
   this->size = 0;
   this->alignedSize = 0;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
void
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::setSegmentsSizes( const IndexType segmentsCount,
                                                                                     const IndexType segmentSize )
{
   this->segmentSize = segmentSize;
   this->size = segmentsCount;
   if( Organization == RowMajorOrder )
      this->alignedSize = this->size;
   else
      this->alignedSize = roundUpDivision( size, this->getAlignment() ) * this->getAlignment();
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::getSegmentsCount() const -> IndexType
{
   return this->size;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::getSegmentSize( const IndexType segmentIdx ) const
   -> IndexType
{
   return this->segmentSize;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::getSize() const -> IndexType
{
   return this->size * this->segmentSize;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::getStorageSize() const -> IndexType
{
   return this->alignedSize * this->segmentSize;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::getGlobalIndex( const Index segmentIdx,
                                                                                   const Index localIdx ) const -> IndexType
{
   if( Organization == RowMajorOrder )
      return segmentIdx * this->segmentSize + localIdx;
   else
      return segmentIdx + this->alignedSize * localIdx;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::getSegmentView( const IndexType segmentIdx ) const
   -> SegmentViewType
{
   if( Organization == RowMajorOrder )
      return SegmentViewType( segmentIdx, segmentIdx * this->segmentSize, this->segmentSize, 1 );
   else
      return SegmentViewType( segmentIdx, segmentIdx, this->segmentSize, this->alignedSize );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
template< typename Function >
void
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::forElements( IndexType first,
                                                                                IndexType last,
                                                                                Function&& f ) const
{
   this->getConstView().forElements( first, last, f );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
template< typename Function >
void
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::forAllElements( Function&& f ) const
{
   this->forElements( 0, this->getSegmentsCount(), f );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
template< typename Function >
void
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::forSegments( IndexType begin,
                                                                                IndexType end,
                                                                                Function&& f ) const
{
   this->getConstView().forSegments( begin, end, f );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
template< typename Function >
void
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::forAllSegments( Function&& f ) const
{
   this->getConstView().forAllSegments( f );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
void
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::reduceSegments( IndexType first,
                                                                                   IndexType last,
                                                                                   Fetch& fetch,
                                                                                   const Reduction& reduction,
                                                                                   ResultKeeper& keeper,
                                                                                   const Real& zero ) const
{
   this->getConstView().reduceSegments( first, last, fetch, reduction, keeper, zero );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
void
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::reduceAllSegments( Fetch& fetch,
                                                                                      const Reduction& reduction,
                                                                                      ResultKeeper& keeper,
                                                                                      const Real& zero ) const
{
   this->reduceSegments( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
template< typename Device_, typename Index_, typename IndexAllocator_, ElementsOrganization Organization_, int Alignment_ >
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >&
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::operator=(
   const Ellpack< Device_, Index_, IndexAllocator_, Organization_, Alignment_ >& source )
{
   this->segmentSize = source.segmentSize;
   this->size = source.size;
   this->alignedSize = roundUpDivision( size, this->getAlignment() ) * this->getAlignment();
   return *this;
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
void
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::save( File& file ) const
{
   file.save( &segmentSize );
   file.save( &size );
   file.save( &alignedSize );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
void
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::load( File& file )
{
   file.load( &segmentSize );
   file.load( &size );
   file.load( &alignedSize );
}

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int Alignment >
template< typename Fetch >
auto
Ellpack< Device, Index, IndexAllocator, Organization, Alignment >::print( Fetch&& fetch ) const
   -> SegmentsPrinter< Ellpack, Fetch >
{
   return SegmentsPrinter< Ellpack, Fetch >( *this, fetch );
}

}  // namespace Segments
}  // namespace Algorithms
}  // namespace noa::TNL
