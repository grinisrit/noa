// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/Kernels/details/CSRAdaptiveKernelBlockDescriptor.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/Kernels/details/CSRAdaptiveKernelParameters.h>

namespace noa::TNL {
   namespace Algorithms {
      namespace Segments {

template< typename Index,
          typename Device >
struct CSRAdaptiveKernelView
{
   using IndexType = Index;
   using DeviceType = Device;
   using ViewType = CSRAdaptiveKernelView< Index, Device >;
   using ConstViewType = CSRAdaptiveKernelView< Index, Device >;
   using BlocksType = noa::TNL::Containers::Vector< detail::CSRAdaptiveKernelBlockDescriptor< Index >, Device, Index >;
   using BlocksView = typename BlocksType::ViewType;

   static constexpr int MaxValueSizeLog = detail::CSRAdaptiveKernelParameters<>::MaxValueSizeLog;

   static int getSizeValueLog( const int& i ) { return detail::CSRAdaptiveKernelParameters<>::getSizeValueLog( i ); };

   CSRAdaptiveKernelView() = default;

   void setBlocks( BlocksType& blocks, const int idx );

   ViewType getView();

   ConstViewType getConstView() const;

   static noa::TNL::String getKernelType();

   template< typename OffsetsView,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Real,
             typename... Args >
   void reduceSegments( const OffsetsView& offsets,
                        Index first,
                        Index last,
                        Fetch& fetch,
                        const Reduction& reduction,
                        ResultKeeper& keeper,
                        const Real& zero,
                        Args... args ) const;

   CSRAdaptiveKernelView& operator=( const CSRAdaptiveKernelView< Index, Device >& kernelView );

   void printBlocks( int idx ) const;

   protected:
      BlocksView blocksArray[ MaxValueSizeLog ];
};

      } // namespace Segments
   }  // namespace Algorithms
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Algorithms/Segments/Kernels/CSRAdaptiveKernelView.hpp>
