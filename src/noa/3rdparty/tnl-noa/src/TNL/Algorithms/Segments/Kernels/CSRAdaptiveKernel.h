// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Assert.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/LaunchHelpers.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/VectorView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/detail/LambdaAdapter.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/Kernels/CSRScalarKernel.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/Kernels/CSRAdaptiveKernelView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/Kernels/details/CSRAdaptiveKernelBlockDescriptor.h>

namespace noa::TNL {
namespace Algorithms {
namespace Segments {

#ifdef __CUDACC__

template< int CudaBlockSize,
          int warpSize,
          int WARPS,
          int SHARED_PER_WARP,
          int MAX_ELEM_PER_WARP,
          typename BlocksView,
          typename Offsets,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Real,
          typename... Args >
__global__
void
reduceSegmentsCSRAdaptiveKernel( BlocksView blocks,
                                 int gridIdx,
                                 Offsets offsets,
                                 Index first,
                                 Index last,
                                 Fetch fetch,
                                 Reduction reduce,
                                 ResultKeeper keep,
                                 Real zero,
                                 Args... args );
#endif

template< typename Index, typename Device >
struct CSRAdaptiveKernel
{
   using IndexType = Index;
   using DeviceType = Device;
   using ViewType = CSRAdaptiveKernelView< Index, Device >;
   using ConstViewType = CSRAdaptiveKernelView< Index, Device >;
   using BlocksType = typename ViewType::BlocksType;
   using BlocksView = typename BlocksType::ViewType;

   static constexpr int
   MaxValueSizeLog()
   {
      return ViewType::MaxValueSizeLog;
   }

   static int
   getSizeValueLog( const int& i )
   {
      return detail::CSRAdaptiveKernelParameters<>::getSizeValueLog( i );
   }

   static TNL::String
   getKernelType();

   template< typename Offsets >
   void
   init( const Offsets& offsets );

   void
   reset();

   __cuda_callable__
   ViewType
   getView();

   __cuda_callable__
   ConstViewType
   getConstView() const;

   template< typename OffsetsView, typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
   void
   reduceSegments( const OffsetsView& offsets,
                   Index first,
                   Index last,
                   Fetch& fetch,
                   const Reduction& reduction,
                   ResultKeeper& keeper,
                   const Real& zero,
                   Args... args ) const;

protected:
   template< int SizeOfValue, typename Offsets >
   Index
   findLimit( Index start, const Offsets& offsets, Index size, detail::Type& type, size_t& sum );

   template< int SizeOfValue, typename Offsets >
   void
   initValueSize( const Offsets& offsets );

   /**
    * \brief  blocksArray[ i ] stores blocks for sizeof( Value ) == 2^i.
    */
   BlocksType blocksArray[ MaxValueSizeLog() ];

   ViewType view;
};

}  // namespace Segments
}  // namespace Algorithms
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/Kernels/CSRAdaptiveKernel.hpp>
