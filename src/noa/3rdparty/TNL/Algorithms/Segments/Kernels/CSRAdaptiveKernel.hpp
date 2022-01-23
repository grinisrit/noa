// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Assert.h>
#include <noa/3rdparty/TNL/Cuda/LaunchHelpers.h>
#include <noa/3rdparty/TNL/Containers/VectorView.h>
#include <noa/3rdparty/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/detail/LambdaAdapter.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/Kernels/CSRScalarKernel.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/Kernels/details/CSRAdaptiveKernelBlockDescriptor.h>

namespace noa::TNL {
   namespace Algorithms {
      namespace Segments {

template< typename Index,
          typename Device >
noa::TNL::String
CSRAdaptiveKernel< Index, Device >::
getKernelType()
{
   return ViewType::getKernelType();
};

template< typename Index,
          typename Device >
   template< typename Offsets >
void
CSRAdaptiveKernel< Index, Device >::
init( const Offsets& offsets )
{
   if( max( offsets ) == 0 )
   {
      for( int i = 0; i < MaxValueSizeLog(); i++ )
      {
         this->blocksArray[ i ].reset();
         this->view.setBlocks( this->blocksArray[ i ], i );
      }
      return;
   }

   this->template initValueSize<  1 >( offsets );
   this->template initValueSize<  2 >( offsets );
   this->template initValueSize<  4 >( offsets );
   this->template initValueSize<  8 >( offsets );
   this->template initValueSize< 16 >( offsets );
   this->template initValueSize< 32 >( offsets );
   for( int i = 0; i < MaxValueSizeLog(); i++ )
      this->view.setBlocks( this->blocksArray[ i ], i );
}


template< typename Index,
          typename Device >
void
CSRAdaptiveKernel< Index, Device >::
reset()
{
   for( int i = 0; i < MaxValueSizeLog(); i++ )
   {
      this->blocksArray[ i ].reset();
      this->view.setBlocks( this->blocksArray[ i ], i );
   }
}

template< typename Index,
          typename Device >
auto
CSRAdaptiveKernel< Index, Device >::
getView() -> ViewType
{
   return this->view;
}

template< typename Index,
          typename Device >
auto
CSRAdaptiveKernel< Index, Device >::
getConstView() const -> ConstViewType
{
   return this->view;
};

template< typename Index,
          typename Device >
   template< typename OffsetsView,
               typename Fetch,
               typename Reduction,
               typename ResultKeeper,
               typename Real,
               typename... Args >
void
CSRAdaptiveKernel< Index, Device >::
reduceSegments( const OffsetsView& offsets,
                   Index first,
                   Index last,
                   Fetch& fetch,
                   const Reduction& reduction,
                   ResultKeeper& keeper,
                   const Real& zero,
                   Args... args ) const
{
   view.reduceSegments( offsets, first, last, fetch, reduction, keeper, zero, args... );
}

template< typename Index,
          typename Device >
   template< int SizeOfValue,
             typename Offsets >
Index
CSRAdaptiveKernel< Index, Device >::
findLimit( const Index start,
           const Offsets& offsets,
           const Index size,
           detail::Type &type,
           size_t &sum )
{
   sum = 0;
   for( Index current = start; current < size - 1; current++ )
   {
      Index elements = offsets[ current + 1 ] - offsets[ current ];
      sum += elements;
      if( sum > detail::CSRAdaptiveKernelParameters< SizeOfValue >::StreamedSharedElementsPerWarp() )
      {
         if( current - start > 0 ) // extra row
         {
            type = detail::Type::STREAM;
            return current;
         }
         else
         {                  // one long row
            if( sum <= 2 * detail::CSRAdaptiveKernelParameters< SizeOfValue >::MaxAdaptiveElementsPerWarp() ) //MAX_ELEMENTS_PER_WARP_ADAPT )
               type = detail::Type::VECTOR;
            else
               type = detail::Type::LONG;
            return current + 1;
         }
      }
   }
   type = detail::Type::STREAM;
   return size - 1; // return last row pointer
}

template< typename Index,
          typename Device >
   template< int SizeOfValue,
             typename Offsets >
void
CSRAdaptiveKernel< Index, Device >::
initValueSize( const Offsets& offsets )
{
   using HostOffsetsType = noa::TNL::Containers::Vector< typename Offsets::IndexType, noa::TNL::Devices::Host, typename Offsets::IndexType >;
   HostOffsetsType hostOffsets( offsets );
   const Index rows = offsets.getSize();
   Index start( 0 ), nextStart( 0 );
   size_t sum;

   // Fill blocks
   std::vector< detail::CSRAdaptiveKernelBlockDescriptor< Index > > inBlocks;
   inBlocks.reserve( rows );

   while( nextStart != rows - 1 )
   {
      detail::Type type;
      nextStart = findLimit< SizeOfValue >( start, hostOffsets, rows, type, sum );
      if( type == detail::Type::LONG )
      {
         const Index blocksCount = inBlocks.size();
         const Index warpsPerCudaBlock = detail::CSRAdaptiveKernelParameters< SizeOfValue >::CudaBlockSize() / noa::TNL::Cuda::getWarpSize();
         Index warpsLeft = roundUpDivision( blocksCount, warpsPerCudaBlock ) * warpsPerCudaBlock - blocksCount;
         if( warpsLeft == 0 )
            warpsLeft = warpsPerCudaBlock;
         for( Index index = 0; index < warpsLeft; index++ )
            inBlocks.emplace_back( start, detail::Type::LONG, index, warpsLeft );
      }
      else
      {
         inBlocks.emplace_back(start, type,
               nextStart,
               offsets.getElement(nextStart),
               offsets.getElement(start) );
      }
      start = nextStart;
   }
   inBlocks.emplace_back(nextStart);
   this->blocksArray[ getSizeValueLog( SizeOfValue ) ] = inBlocks;
}

      } // namespace Segments
   }  // namespace Algorithms
} // namespace noa::TNL
