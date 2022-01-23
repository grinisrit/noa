// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/SlicedEllpackView.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/detail/LambdaAdapter.h>

#include "SlicedEllpackView.h"

namespace noa::TNL {
   namespace Algorithms {
      namespace Segments {


template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
__cuda_callable__
SlicedEllpackView< Device, Index, Organization, SliceSize >::
SlicedEllpackView()
   : size( 0 ), alignedSize( 0 ), segmentsCount( 0 )
{
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
__cuda_callable__
SlicedEllpackView< Device, Index, Organization, SliceSize >::
SlicedEllpackView(  IndexType size,
                    IndexType alignedSize,
                    IndexType segmentsCount,
                    OffsetsView&& sliceOffsets,
                    OffsetsView&& sliceSegmentSizes )
   : size( size ), alignedSize( alignedSize ), segmentsCount( segmentsCount ),
     sliceOffsets( std::forward< OffsetsView >( sliceOffsets ) ), sliceSegmentSizes( std::forward< OffsetsView >( sliceSegmentSizes ) )
{
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
String
SlicedEllpackView< Device, Index, Organization, SliceSize >::
getSerializationType()
{
   // FIXME: the serialized data DEPEND on the Organization and Alignment parameters, so it should be reflected in the serialization type
   return "SlicedEllpack< [any_device], " + noa::TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
String
SlicedEllpackView< Device, Index, Organization, SliceSize >::
getSegmentsType()
{
   return "SlicedEllpack";
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
__cuda_callable__
typename SlicedEllpackView< Device, Index, Organization, SliceSize >::ViewType
SlicedEllpackView< Device, Index, Organization, SliceSize >::
getView()
{
   return ViewType( size, alignedSize, segmentsCount, sliceOffsets, sliceSegmentSizes );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
__cuda_callable__
auto
SlicedEllpackView< Device, Index, Organization, SliceSize >::
getConstView() const -> const ConstViewType
{
   return ConstViewType( size, alignedSize, segmentsCount, sliceOffsets.getConstView(), sliceSegmentSizes.getConstView() );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
__cuda_callable__ auto SlicedEllpackView< Device, Index, Organization, SliceSize >::
getSegmentsCount() const -> IndexType
{
   return this->segmentsCount;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
__cuda_callable__ auto SlicedEllpackView< Device, Index, Organization, SliceSize >::
getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   const Index sliceIdx = segmentIdx / SliceSize;
   if( std::is_same< DeviceType, Devices::Host >::value )
      return this->sliceSegmentSizes[ sliceIdx ];
   else
   {
#ifdef __CUDA_ARCH__
   return this->sliceSegmentSizes[ sliceIdx ];
#else
   return this->sliceSegmentSizes.getElement( sliceIdx );
#endif
   }
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
__cuda_callable__ auto SlicedEllpackView< Device, Index, Organization, SliceSize >::
getSize() const -> IndexType
{
   return this->size;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
__cuda_callable__ auto SlicedEllpackView< Device, Index, Organization, SliceSize >::
getStorageSize() const -> IndexType
{
   return this->alignedSize;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
__cuda_callable__ auto SlicedEllpackView< Device, Index, Organization, SliceSize >::
getGlobalIndex( const Index segmentIdx, const Index localIdx ) const -> IndexType
{
   const IndexType sliceIdx = segmentIdx / SliceSize;
   const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
   IndexType sliceOffset, segmentSize;
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      sliceOffset = this->sliceOffsets[ sliceIdx ];
      segmentSize = this->sliceSegmentSizes[ sliceIdx ];
   }
   else
   {
#ifdef __CUDA_ARCH__
      sliceOffset = this->sliceOffsets[ sliceIdx ];
      segmentSize = this->sliceSegmentSizes[ sliceIdx ];
#else
      sliceOffset = this->sliceOffsets.getElement( sliceIdx );
      segmentSize = this->sliceSegmentSizes.getElement( sliceIdx );
#endif
   }
   if( Organization == RowMajorOrder )
      return sliceOffset + segmentInSliceIdx * segmentSize + localIdx;
   else
      return sliceOffset + segmentInSliceIdx + SliceSize * localIdx;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
__cuda_callable__
auto
SlicedEllpackView< Device, Index, Organization, SliceSize >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   const IndexType sliceIdx = segmentIdx / SliceSize;
   const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
   const IndexType& sliceOffset = this->sliceOffsets[ sliceIdx ];
   const IndexType& segmentSize = this->sliceSegmentSizes[ sliceIdx ];

   if( Organization == RowMajorOrder )
      return SegmentViewType( segmentIdx, sliceOffset + segmentInSliceIdx * segmentSize, segmentSize, 1 );
   else
      return SegmentViewType( segmentIdx, sliceOffset + segmentInSliceIdx, segmentSize, SliceSize );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
   template< typename Function >
void
SlicedEllpackView< Device, Index, Organization, SliceSize >::
forElements( IndexType first, IndexType last, Function&& f ) const
{
   const auto sliceSegmentSizes_view = this->sliceSegmentSizes.getConstView();
   const auto sliceOffsets_view = this->sliceOffsets.getConstView();
   if( Organization == RowMajorOrder )
   {
      auto l = [=] __cuda_callable__ ( const IndexType segmentIdx ) mutable {
         const IndexType sliceIdx = segmentIdx / SliceSize;
         const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
         const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
         const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx * segmentSize;
         const IndexType end = begin + segmentSize;
         IndexType localIdx( 0 );
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx++  )
         {
            // The following is a workaround of a bug in nvcc 11.2
#if CUDART_VERSION == 11020
             f( segmentIdx, localIdx, globalIdx );
             localIdx++;
#else
             f( segmentIdx, localIdx++, globalIdx );
#endif
         }
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l );
   }
   else
   {
      auto l = [=] __cuda_callable__ ( const IndexType segmentIdx ) mutable {
         const IndexType sliceIdx = segmentIdx / SliceSize;
         const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
         //const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
         const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx;
         const IndexType end = sliceOffsets_view[ sliceIdx + 1 ];
         IndexType localIdx( 0 );
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SliceSize )
         {
            // The following is a workaround of a bug in nvcc 11.2
#if CUDART_VERSION == 11020
            f( segmentIdx, localIdx, globalIdx );
            localIdx++;
#else
            f( segmentIdx, localIdx++, globalIdx );
#endif
         }
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l );
   }
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
   template< typename Function >
void
SlicedEllpackView< Device, Index, Organization, SliceSize >::
forAllElements( Function&& f ) const
{
   this->forElements( 0, this->getSegmentsCount(), f );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
   template< typename Function >
void
SlicedEllpackView< Device, Index, Organization, SliceSize >::
forSegments( IndexType begin, IndexType end, Function&& function ) const
{
   auto view = this->getConstView();
   auto f = [=] __cuda_callable__ ( IndexType segmentIdx ) mutable {
      auto segment = view.getSegmentView( segmentIdx );
      function( segment );
   };
   noa::TNL::Algorithms::ParallelFor< DeviceType >::exec( begin, end, f );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
   template< typename Function >
void
SlicedEllpackView< Device, Index, Organization, SliceSize >::
forAllSegments( Function&& f ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
void
SlicedEllpackView< Device, Index, Organization, SliceSize >::
reduceSegments( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const
{
   using RealType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   //using RealType = decltype( fetch( IndexType(), IndexType(), IndexType(), std::declval< bool& >() ) );
   const auto sliceSegmentSizes_view = this->sliceSegmentSizes.getConstView();
   const auto sliceOffsets_view = this->sliceOffsets.getConstView();
   if( Organization == RowMajorOrder )
   {
      auto l = [=] __cuda_callable__ ( const IndexType segmentIdx ) mutable {
         const IndexType sliceIdx = segmentIdx / SliceSize;
         const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
         const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
         const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx * segmentSize;
         const IndexType end = begin + segmentSize;
         RealType aux( zero );
         IndexType localIdx( 0 );
         bool compute( true );
         for( IndexType globalIdx = begin; globalIdx< end; globalIdx++  )
            aux = reduction( aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx, compute ) );
         keeper( segmentIdx, aux );
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l );
   }
   else
   {
      auto l = [=] __cuda_callable__ ( const IndexType segmentIdx ) mutable {
         const IndexType sliceIdx = segmentIdx / SliceSize;
         const IndexType segmentInSliceIdx = segmentIdx % SliceSize;
         //const IndexType segmentSize = sliceSegmentSizes_view[ sliceIdx ];
         const IndexType begin = sliceOffsets_view[ sliceIdx ] + segmentInSliceIdx;
         const IndexType end = sliceOffsets_view[ sliceIdx + 1 ];
         RealType aux( zero );
         IndexType localIdx( 0 );
         bool compute( true );
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx += SliceSize  )
            aux = reduction( aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx, compute ) );
         keeper( segmentIdx, aux );
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l );
   }
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
void
SlicedEllpackView< Device, Index, Organization, SliceSize >::
reduceAllSegments( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const
{
   this->reduceSegments( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
SlicedEllpackView< Device, Index, Organization, SliceSize >&
SlicedEllpackView< Device, Index, Organization, SliceSize >::
operator=( const SlicedEllpackView< Device, Index, Organization, SliceSize >& view )
{
   this->size = view.size;
   this->alignedSize = view.alignedSize;
   this->segmentsCount = view.segmentsCount;
   this->sliceOffsets.bind( view.sliceOffsets );
   this->sliceSegmentSizes.bind( view.sliceSegmentSizes );
   return *this;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
void
SlicedEllpackView< Device, Index, Organization, SliceSize >::
save( File& file ) const
{
   file.save( &size );
   file.save( &alignedSize );
   file.save( &segmentsCount );
   file << this->sliceOffsets;
   file << this->sliceSegmentSizes;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
void
SlicedEllpackView< Device, Index, Organization, SliceSize >::
load( File& file )
{
   file.load( &size );
   file.load( &alignedSize );
   file.load( &segmentsCount );
   file >> this->sliceOffsets;
   file >> this->sliceSegmentSizes;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int SliceSize >
      template< typename Fetch >
auto
SlicedEllpackView< Device, Index, Organization, SliceSize >::
print( Fetch&& fetch ) const -> SegmentsPrinter< SlicedEllpackView, Fetch >
{
   return SegmentsPrinter< SlicedEllpackView, Fetch >( *this, fetch );
}

      } // namespace Segments
   }  // namespace Algorithms
} // namespace noa::TNL
