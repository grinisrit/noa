// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include "SegmentedScan.h"

#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/NotImplementedError.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/CudaSupportMissing.h>

namespace noa::TNL {
namespace Algorithms {

template< detail::ScanType Type >
template< typename Vector, typename Reduction, typename Flags >
void
SegmentedScan< Devices::Sequential, Type >::perform( Vector& v,
                                                     Flags& flags,
                                                     typename Vector::IndexType begin,
                                                     typename Vector::IndexType end,
                                                     const Reduction& reduction,
                                                     typename Vector::ValueType identity )
{
   using ValueType = typename Vector::ValueType;
   using IndexType = typename Vector::IndexType;

   if( Type == detail::ScanType::Inclusive ) {
      for( IndexType i = begin + 1; i < end; i++ )
         if( ! flags[ i ] )
            v[ i ] = reduction( v[ i ], v[ i - 1 ] );
   }
   else  // Exclusive scan
   {
      ValueType aux( v[ begin ] );
      v[ begin ] = identity;
      for( IndexType i = begin + 1; i < end; i++ ) {
         ValueType x = v[ i ];
         if( flags[ i ] )
            aux = identity;
         v[ i ] = aux;
         aux = reduction( aux, x );
      }
   }
}

template< detail::ScanType Type >
template< typename Vector, typename Reduction, typename Flags >
void
SegmentedScan< Devices::Host, Type >::perform( Vector& v,
                                               Flags& flags,
                                               typename Vector::IndexType begin,
                                               typename Vector::IndexType end,
                                               const Reduction& reduction,
                                               typename Vector::ValueType identity )
{
#ifdef HAVE_OPENMP
   // TODO: parallelize with OpenMP
   SegmentedScan< Devices::Sequential, Type >::perform( v, flags, begin, end, reduction, identity );
#else
   SegmentedScan< Devices::Sequential, Type >::perform( v, flags, begin, end, reduction, identity );
#endif
}

template< detail::ScanType Type >
template< typename Vector, typename Reduction, typename Flags >
void
SegmentedScan< Devices::Cuda, Type >::perform( Vector& v,
                                               Flags& flags,
                                               typename Vector::IndexType begin,
                                               typename Vector::IndexType end,
                                               const Reduction& reduction,
                                               typename Vector::ValueType identity )
{
#ifdef __CUDACC__
   throw Exceptions::NotImplementedError( "Segmented scan (prefix sum) is not implemented for CUDA." );
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

}  // namespace Algorithms
}  // namespace noa::TNL
