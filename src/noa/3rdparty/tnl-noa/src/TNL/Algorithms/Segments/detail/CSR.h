// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/scan.h>

namespace noa::TNL {
namespace Algorithms {
namespace Segments {
namespace detail {

template< typename Device, typename Index >
class CSR
{
public:
   using DeviceType = Device;
   using IndexType = Index;

   template< typename SizesHolder, typename CSROffsets >
   static void
   setSegmentsSizes( const SizesHolder& sizes, CSROffsets& offsets )
   {
      offsets.setSize( sizes.getSize() + 1 );
      // GOTCHA: when sizes.getSize() == 0, getView returns a full view with size == 1
      if( sizes.getSize() > 0 ) {
         auto view = offsets.getView( 0, sizes.getSize() );
         view = sizes;
      }
      offsets.setElement( sizes.getSize(), 0 );
      inplaceExclusiveScan( offsets );
   }

   template< typename CSROffsets >
   __cuda_callable__
   static IndexType
   getSegmentsCount( const CSROffsets& offsets )
   {
      return offsets.getSize() - 1;
   }

   /***
    * \brief Returns size of the segment number \r segmentIdx
    */
   template< typename CSROffsets >
   __cuda_callable__
   static IndexType
   getSegmentSize( const CSROffsets& offsets, const IndexType segmentIdx )
   {
      if( ! std::is_same< DeviceType, Devices::Host >::value ) {
#ifdef __CUDA_ARCH__
         return offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
#else
         return offsets.getElement( segmentIdx + 1 ) - offsets.getElement( segmentIdx );
#endif
      }
      return offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
   }

   /***
    * \brief Returns number of elements that needs to be allocated.
    */
   template< typename CSROffsets >
   __cuda_callable__
   static IndexType
   getStorageSize( const CSROffsets& offsets )
   {
      if( ! std::is_same< DeviceType, Devices::Host >::value ) {
#ifdef __CUDA_ARCH__
         return offsets[ getSegmentsCount( offsets ) ];
#else
         return offsets.getElement( getSegmentsCount( offsets ) );
#endif
      }
      return offsets[ getSegmentsCount( offsets ) ];
   }

   __cuda_callable__
   IndexType
   getGlobalIndex( Index segmentIdx, Index localIdx ) const;

   __cuda_callable__
   void
   getSegmentAndLocalIndex( Index globalIdx, Index& segmentIdx, Index& localIdx ) const;

   /***
    * \brief Go over all segments and for each segment element call
    * function 'f' with arguments 'args'. The return type of 'f' is bool.
    * When its true, the for-loop continues. Once 'f' returns false, the for-loop
    * is terminated.
    */
   template< typename Function, typename... Args >
   void
   forElements( IndexType first, IndexType last, Function& f, Args... args ) const;

   template< typename Function, typename... Args >
   void
   forAllElements( Function& f, Args... args ) const;

   /***
    * \brief Go over all segments and perform a reduction in each of them.
    */
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
   void
   reduceSegments( IndexType first,
                   IndexType last,
                   Fetch& fetch,
                   Reduction& reduction,
                   ResultKeeper& keeper,
                   const Real& zero,
                   Args... args ) const;

   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real, typename... Args >
   void
   reduceAllSegments( Fetch& fetch, Reduction& reduction, ResultKeeper& keeper, const Real& zero, Args... args ) const;
};
}  // namespace detail
}  // namespace Segments
}  // namespace Algorithms
}  // namespace noa::TNL
