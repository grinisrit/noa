// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <noa/3rdparty/TNL/Containers/Array.h>

namespace noa::TNL {
   namespace Algorithms {
      namespace Segments {

/**
 * \brief Print segments sizes, i.e. the segments setup.
 *
 * \tparam Segments is type of segments.
 * \param segments is an instance of segments.
 * \param str is output stream.
 * \return reference to the output stream.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsPrintingExample-1.cpp
 * \par Output
 * \include SegmentsPrintingExample-1.out
 */
template< typename Segments >
std::ostream& printSegments( const Segments& segments, std::ostream& str )
{
   using IndexType = typename Segments::IndexType;
   using DeviceType = typename Segments::DeviceType;

   auto segmentsCount = segments.getSegmentsCount();
   str << " [";
   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ )
   {
      auto segmentSize = segments.getSegmentSize( segmentIdx );
      str << " " << segmentSize;
      if( segmentIdx < segmentsCount )
         str << ",";
   }
   str << " ] ";
   return str;
}

/// This is to prevent from appearing in Doxygen documentation.
/// \cond HIDDEN_CLASS
template< typename Segments,
          typename Fetch >
struct SegmentsPrinter
{
   SegmentsPrinter( const Segments& segments, Fetch&& fetch )
   : segments( segments ), fetch( fetch ) {}

   std::ostream& print( std::ostream& str ) const
   {
      using IndexType = typename Segments::IndexType;
      using DeviceType = typename Segments::DeviceType;
      using ValueType = decltype( fetch( IndexType() ) );

      noa::TNL::Containers::Array< ValueType, DeviceType, IndexType > aux( 1 );
      auto view = segments.getConstView();
      for( IndexType segmentIdx = 0; segmentIdx < segments.getSegmentsCount(); segmentIdx++ )
      {
         str << "Seg. " << segmentIdx << ": [ ";
         auto segmentSize = segments.getSegmentSize( segmentIdx );
         for( IndexType localIdx = 0; localIdx < segmentSize; localIdx++ )
         {
            aux.forAllElements( [=] __cuda_callable__ ( IndexType elementIdx, double& v ) mutable {
               //printf( "####### localIdx = %d, globalIdx = %d \n", localIdx, view.getGlobalIndex( segmentIdx, localIdx ) );
               //v = view.getGlobalIndex( segmentIdx, localIdx );
               v = fetch( view.getGlobalIndex( segmentIdx, localIdx ) );
            } );
            auto value = aux.getElement( 0 );
            str << value;
            if( localIdx < segmentSize - 1 )
               str << ", ";
         }
         str << " ] " << std::endl;
      }
      return str;
   }

   protected:

   const Segments& segments;

   Fetch fetch;
};

/*template< typename Segments,
          typename Fetch >
std::ostream& operator<<( std::ostream& str, const SegmentsPrinter< Segments, Fetch >& printer )
{
   return printer.print( str );
}*/

template< typename Segments,
          typename Fetch >
std::ostream& printSegments( const Segments& segments, Fetch&& fetch, std::ostream& str )
{
   using IndexType = typename Segments::IndexType;
   using DeviceType = typename Segments::DeviceType;
   using ValueType = decltype( fetch( IndexType() ) );

   noa::TNL::Containers::Array< ValueType, DeviceType, IndexType > aux( 1 );
   auto view = segments.getConstView();
   for( IndexType segmentIdx = 0; segmentIdx < segments.getSegmentsCount(); segmentIdx++ )
   {
      str << "Seg. " << segmentIdx << ": [ ";
      auto segmentSize = segments.getSegmentSize( segmentIdx );
      //std::cerr << "Segment size = " << segmentSize << std::endl;
      for( IndexType localIdx = 0; localIdx < segmentSize; localIdx++ )
      {
         aux.forAllElements( [=] __cuda_callable__ ( IndexType elementIdx, double& v ) mutable {
            //printf( "####### localIdx = %d, globalIdx = %d \n", localIdx, view.getGlobalIndex( segmentIdx, localIdx ) );
            v = fetch( view.getGlobalIndex( segmentIdx, localIdx ) );
            //v = view.getGlobalIndex( segmentIdx, localIdx );
         } );
         auto value = aux.getElement( 0 );
         str << value;
         if( localIdx < segmentSize - 1 )
            str << ", ";
      }
      str << " ] " << std::endl;
   }
   return str;
}
/// \endcond

      } // namespace Segments
   } // namespace Algorithms
} // namespace noa::TNL
