// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/CSR.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/detail/CSR.h>

namespace noa::TNL {
   namespace Algorithms {
      namespace Segments {


template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
CSR< Device, Index, Kernel, IndexAllocator >::
CSR()
{
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
   template< typename SizesContainer >
CSR< Device, Index, Kernel, IndexAllocator >::
CSR( const SizesContainer& segmentsSizes )
{
   this->setSegmentsSizes( segmentsSizes );
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
   template< typename ListIndex >
CSR< Device, Index, Kernel, IndexAllocator >::
CSR( const std::initializer_list< ListIndex >& segmentsSizes )
{
   this->setSegmentsSizes( Containers::Vector< IndexType, DeviceType, IndexType >( segmentsSizes ) );
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
String
CSR< Device, Index, Kernel, IndexAllocator >::
getSerializationType()
{
   return "CSR< [any_device], " +
      noa::TNL::getSerializationType< IndexType >() + ", " +
      // FIXME: the serialized data do not depend on the the kernel type so it should not be in the serialization type
      noa::TNL::getSerializationType< KernelType >() + " >";
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
String
CSR< Device, Index, Kernel, IndexAllocator >::
getSegmentsType()
{
   return ViewType::getSegmentsType();
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
   template< typename SizesHolder >
void
CSR< Device, Index, Kernel, IndexAllocator >::
setSegmentsSizes( const SizesHolder& sizes )
{
   detail::CSR< Device, Index >::setSegmentsSizes( sizes, this->offsets );
   this->kernel.init( this->offsets );
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
void
CSR< Device, Index, Kernel, IndexAllocator >::
reset()
{
   this->offsets.setSize( 1 );
   this->offsets = 0;
   this->kernel.reset();
}


template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
typename CSR< Device, Index, Kernel, IndexAllocator >::ViewType
CSR< Device, Index, Kernel, IndexAllocator >::
getView()
{
   return ViewType( this->offsets.getView(), this->kernel.getView() );
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
auto
CSR< Device, Index, Kernel, IndexAllocator >::
getConstView() const -> const ConstViewType
{
   return ConstViewType( this->offsets.getConstView(), this->kernel.getConstView() );
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
__cuda_callable__ auto CSR< Device, Index, Kernel, IndexAllocator >::
getSegmentsCount() const -> IndexType
{
   return this->offsets.getSize() - 1;
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
__cuda_callable__ auto CSR< Device, Index, Kernel, IndexAllocator >::
getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   return detail::CSR< Device, Index >::getSegmentSize( this->offsets, segmentIdx );
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
__cuda_callable__ auto CSR< Device, Index, Kernel, IndexAllocator >::
getSize() const -> IndexType
{
   return this->getStorageSize();
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
__cuda_callable__ auto CSR< Device, Index, Kernel, IndexAllocator >::
getStorageSize() const -> IndexType
{
   return detail::CSR< Device, Index >::getStorageSize( this->offsets );
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
__cuda_callable__ auto CSR< Device, Index, Kernel, IndexAllocator >::
getGlobalIndex( const Index segmentIdx, const Index localIdx ) const -> IndexType
{
   if( ! std::is_same< DeviceType, Devices::Host >::value )
   {
#ifdef __CUDA_ARCH__
      return offsets[ segmentIdx ] + localIdx;
#else
      return offsets.getElement( segmentIdx ) + localIdx;
#endif
   }
   return offsets[ segmentIdx ] + localIdx;
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
__cuda_callable__
auto
CSR< Device, Index, Kernel, IndexAllocator >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   return SegmentViewType( segmentIdx, offsets[ segmentIdx ], offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ] );
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
auto
CSR< Device, Index, Kernel, IndexAllocator >::
getOffsets() const -> const OffsetsContainer&
{
   return this->offsets;
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
auto
CSR< Device, Index, Kernel, IndexAllocator >::
getOffsets() -> OffsetsContainer&
{
   return this->offsets;
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
   template< typename Function >
void
CSR< Device, Index, Kernel, IndexAllocator >::
forElements( IndexType begin, IndexType end, Function&& f ) const
{
   this->getConstView().forElements( begin, end, f );
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
   template< typename Function >
void
CSR< Device, Index, Kernel, IndexAllocator >::
forAllElements( Function&& f ) const
{
   this->forElements( 0, this->getSegmentsCount(), f );
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
   template< typename Function >
void
CSR< Device, Index, Kernel, IndexAllocator >::
forSegments( IndexType begin, IndexType end, Function&& f ) const
{
   this->getConstView().forSegments( begin, end, f );
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
   template< typename Function >
void
CSR< Device, Index, Kernel, IndexAllocator >::
forAllSegments( Function&& f ) const
{
   this->getConstView().forAllSegments( f );
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
   template< typename Function >
void
CSR< Device, Index, Kernel, IndexAllocator >::
sequentialForSegments( IndexType begin, IndexType end, Function&& f ) const
{
   this->getConstView().sequentialForSegments( begin, end, f );
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
   template< typename Function >
void
CSR< Device, Index, Kernel, IndexAllocator >::
sequentialForAllSegments( Function&& f ) const
{
   this->getConstView().sequentialForAllSegments( f );
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
void
CSR< Device, Index, Kernel, IndexAllocator >::
reduceSegments( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const
{
   this->getConstView().reduceSegments( first, last, fetch, reduction, keeper, zero );
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
void
CSR< Device, Index, Kernel, IndexAllocator >::
reduceAllSegments( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const
{
   this->reduceSegments( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero );
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
   template< typename Device_, typename Index_, typename Kernel_, typename IndexAllocator_ >
CSR< Device, Index, Kernel, IndexAllocator >&
CSR< Device, Index, Kernel, IndexAllocator >::
operator=( const CSR< Device_, Index_, Kernel_, IndexAllocator_ >& source )
{
   this->offsets = source.offsets;
   this->kernel = kernel;
   return *this;
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
void
CSR< Device, Index, Kernel, IndexAllocator >::
save( File& file ) const
{
   file << this->offsets;
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
void
CSR< Device, Index, Kernel, IndexAllocator >::
load( File& file )
{
   file >> this->offsets;
   this->kernel.init( this->offsets );
}

template< typename Device,
          typename Index,
          typename Kernel,
          typename IndexAllocator >
      template< typename Fetch >
auto
CSR< Device, Index, Kernel, IndexAllocator >::
print( Fetch&& fetch ) const -> SegmentsPrinter< CSR, Fetch >
{
   return SegmentsPrinter< CSR, Fetch >( *this, fetch );
}

      } // namespace Segments
   }  // namespace Algorithms
} // namespace noa::TNL
