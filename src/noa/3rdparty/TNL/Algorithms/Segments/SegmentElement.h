// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <noa/3rdparty/TNL/Cuda/CudaCallable.h>

namespace noaTNL {
   namespace Algorithms {
      namespace Segments {

/**
 * \brief Simple structure representing one element of a segment.
 *
 * \tparam Index is type used for indexing of the elements.
 */
template< typename Index >
class SegmentElement
{
   public:

      /**
       * \brief Type used for indexing of the elements.
       */
      using IndexType = Index;

      /**
       * \brief Constructor of the segment element with all parameters.
       *
       * \param segmentIdx is in index of the parent segment.
       * \param localIdx is a rank of the element in the segment.
       * \param globalIdx is an index of the element in the related container.
       */
      __cuda_callable__
      SegmentElement( const IndexType& segmentIdx,
                      const IndexType& localIdx,
                      const IndexType globalIdx )
      : segmentIdx( segmentIdx ), localIdx( localIdx ), globalIdx( globalIdx ) {};

      /**
       * \brief Returns index of the parent segment.
       *
       * \return index of the parent segment.
       */
      __cuda_callable__
      const IndexType& segmentIndex() const { return segmentIdx; };

      /**
       * \brief Returns rank of the element in the segment.
       *
       * \return rank of the element in the segment.
       */
      __cuda_callable__
      const IndexType& localIndex() const { return localIdx; };

      /**
       * \brief Returns index of the element in the related container.
       *
       * \return index of the element in the related container.
       */
      __cuda_callable__
      const IndexType& globalIndex() const { return globalIdx; };

   protected:

      const IndexType& segmentIdx;

      const IndexType& localIdx;

      const IndexType globalIdx;
};

      } // namespace Segments
   } // namespace Algorithms
} // namespace noaTNL
