// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/SegmentView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/Kernels/CSRScalarKernel.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/Kernels/CSRVectorKernel.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/Kernels/CSRHybridKernel.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/Kernels/CSRLightKernel.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/Kernels/CSRAdaptiveKernel.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/SegmentsPrinting.h>

namespace noa::TNL {
namespace Algorithms {
namespace Segments {

template< typename Device, typename Index, typename Kernel = CSRScalarKernel< std::remove_const_t< Index >, Device > >
class CSRView
{
public:
   using DeviceType = Device;
   using IndexType = std::remove_const_t< Index >;
   using KernelType = Kernel;
   using OffsetsView = Containers::VectorView< Index, DeviceType, IndexType >;
   using ConstOffsetsView = typename OffsetsView::ConstViewType;
   using KernelView = typename Kernel::ViewType;
   using ViewType = CSRView;
   template< typename Device_, typename Index_ >
   using ViewTemplate = CSRView< Device_, Index_, Kernel >;
   using ConstViewType = CSRView< Device, std::add_const_t< Index >, Kernel >;
   using SegmentViewType = SegmentView< IndexType, RowMajorOrder >;

   static constexpr bool
   havePadding()
   {
      return false;
   }

   __cuda_callable__
   CSRView() = default;

   __cuda_callable__
   CSRView( const OffsetsView& offsets, const KernelView& kernel );

   __cuda_callable__
   CSRView( OffsetsView&& offsets, KernelView&& kernel );

   __cuda_callable__
   CSRView( const CSRView& csr_view ) = default;

   template< typename Index2 >
   __cuda_callable__
   CSRView( const CSRView< Device, Index2, Kernel >& csr_view );

   __cuda_callable__
   CSRView( CSRView&& csr_view ) noexcept = default;

   static std::string
   getSerializationType();

   static String
   getSegmentsType();

   __cuda_callable__
   ViewType
   getView();

   __cuda_callable__
   ConstViewType
   getConstView() const;

   /**
    * \brief Number segments.
    */
   __cuda_callable__
   IndexType
   getSegmentsCount() const;

   /***
    * \brief Returns size of the segment number \r segmentIdx
    */
   __cuda_callable__
   IndexType
   getSegmentSize( IndexType segmentIdx ) const;

   /***
    * \brief Returns number of elements managed by all segments.
    */
   __cuda_callable__
   IndexType
   getSize() const;

   /***
    * \brief Returns number of elements that needs to be allocated.
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
    * function 'f'. The return type of 'f' is bool.
    * When its true, the for-loop continues. Once 'f' returns false, the for-loop
    * is terminated.
    */
   template< typename Function >
   void
   forElements( IndexType begin, IndexType end, Function&& f ) const;

   template< typename Function >
   void
   forAllElements( Function&& f ) const;

   template< typename Function >
   void
   forSegments( IndexType begin, IndexType end, Function&& f ) const;

   template< typename Function >
   void
   forAllSegments( Function&& f ) const;

   template< typename Function >
   void
   sequentialForSegments( IndexType begin, IndexType end, Function&& f ) const;

   template< typename Function >
   void
   sequentialForAllSegments( Function&& f ) const;

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

   CSRView&
   operator=( const CSRView& view );

   void
   save( File& file ) const;

   void
   load( File& file );

   template< typename Fetch >
   SegmentsPrinter< CSRView, Fetch >
   print( Fetch&& fetch ) const;

   OffsetsView
   getOffsets()
   {
      return offsets;
   }

   ConstOffsetsView
   getOffsets() const
   {
      return offsets.getConstView();
   }

   KernelType&
   getKernel()
   {
      return kernel;
   }

   const KernelType&
   getKernel() const
   {
      return kernel;
   }

protected:
   OffsetsView offsets;

   KernelView kernel;
};

template< typename Device, typename Index, typename Kernel >
std::ostream&
operator<<( std::ostream& str, const CSRView< Device, Index, Kernel >& segments )
{
   return printSegments( str, segments );
}

template< typename Device, typename Index >
using CSRViewScalar = CSRView< Device, Index, CSRScalarKernel< std::remove_const_t< Index >, Device > >;

template< typename Device, typename Index >
using CSRViewVector = CSRView< Device, Index, CSRVectorKernel< std::remove_const_t< Index >, Device > >;

template< typename Device, typename Index, int ThreadsInBlock = 256 >
using CSRViewHybrid = CSRView< Device, Index, CSRHybridKernel< std::remove_const_t< Index >, Device, ThreadsInBlock > >;

template< typename Device, typename Index >
using CSRViewLight = CSRView< Device, Index, CSRLightKernel< std::remove_const_t< Index >, Device > >;

template< typename Device, typename Index >
using CSRViewAdaptive = CSRView< Device, Index, CSRAdaptiveKernel< std::remove_const_t< Index >, Device > >;

template< typename Device, typename Index >
using CSRViewDefault = CSRViewScalar< Device, Index >;

}  // namespace Segments
}  // namespace Algorithms
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/CSRView.hpp>
