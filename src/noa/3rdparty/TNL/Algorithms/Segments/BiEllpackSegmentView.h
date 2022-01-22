// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <math.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/ElementsOrganization.h>
#include <noa/3rdparty/TNL/Containers/StaticVector.h>

namespace noaTNL {
   namespace Algorithms {
      namespace Segments {

template< typename Index,
          ElementsOrganization Organization,
          int WarpSize = 32 >
class BiEllpackSegmentView
{
   public:

      static constexpr int getWarpSize() { return WarpSize; };

      static constexpr int getLogWarpSize() { static_assert( WarpSize == 32, "nvcc does not allow constexpr log2" ); return 5; }// TODO: return std::log2( WarpSize ); };

      static constexpr int getGroupsCount() { return getLogWarpSize() + 1; };

      using IndexType = Index;
      using GroupsWidthType = Containers::StaticVector< getGroupsCount(), IndexType >;


      /**
       * \brief Constructor.
       *
       * \param offset is offset of the first group of the strip the segment belongs to.
       * \param size is the segment size
       * \param inStripIdx is index of the segment within its strip.
       * \param groupsWidth is a static vector containing widths of the strip groups
       */
      __cuda_callable__
      BiEllpackSegmentView( const IndexType segmentIdx,
                            const IndexType offset,
                            const IndexType inStripIdx,
                            const GroupsWidthType& groupsWidth )
      : segmentIdx( segmentIdx ), groupOffset( offset ), inStripIdx( inStripIdx ), segmentSize( noaTNL::sum( groupsWidth ) ), groupsWidth( groupsWidth ){};

      __cuda_callable__
      IndexType getSize() const
      {
         return this->segmentSize;
      };

      __cuda_callable__
      IndexType getGlobalIndex( IndexType localIdx ) const
      {
         //std::cerr << "SegmentView: localIdx = " << localIdx << " groupWidth = " << groupsWidth << std::endl;
         IndexType groupIdx( 0 ), offset( groupOffset ), groupHeight( getWarpSize() );
         while( localIdx >= groupsWidth[ groupIdx ] )
         {
            //std::cerr << "ROW: groupIdx = " << groupIdx << " groupWidth = " << groupsWidth[ groupIdx ]
            //          << " groupSize = " << groupsWidth[ groupIdx ] * groupHeight << std::endl;
            localIdx -= groupsWidth[ groupIdx ];
            offset += groupsWidth[ groupIdx++ ] * groupHeight;
            groupHeight /= 2;
         }
         TNL_ASSERT_LE( groupIdx, noaTNL::log2( getWarpSize() - inStripIdx + 1 ), "Local index exceeds segment bounds." );
         if( Organization == RowMajorOrder )
         {
            //std::cerr << " offset = " << offset << " inStripIdx = " << inStripIdx << " localIdx = " << localIdx 
            //          << " return = " << offset + inStripIdx * groupsWidth[ groupIdx ] + localIdx << std::endl;
            return offset + inStripIdx * groupsWidth[ groupIdx ] + localIdx;
         }
         else
            return offset + inStripIdx + localIdx * groupHeight;
      };

      __cuda_callable__
      const IndexType& getSegmentIndex() const
      {
         return this->segmentIdx;
      };

      protected:

         IndexType segmentIdx, groupOffset, inStripIdx, segmentSize;

         GroupsWidthType groupsWidth;
};

      } //namespace Segments
   } //namespace Algorithms
} //namespace noaTNL
