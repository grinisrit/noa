// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Xuan Thang Nguyen, Tomas Oberhuber

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Sorting/detail/bitonicSort.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/NotImplementedError.h>

namespace noa::TNL {
namespace Algorithms {
namespace Sorting {

struct BitonicSort
{
   template< typename Array >
   void static sort( Array& array )
   {
      bitonicSort( array );
   }

   template< typename Array, typename Compare >
   void static sort( Array& array, const Compare& compare )
   {
      bitonicSort( array, compare );
   }

   template< typename Device, typename Index, typename Compare, typename Swap >
   void static inplaceSort( const Index begin, const Index end, const Compare& compare, const Swap& swap )
   {
      if( std::is_same< Device, Devices::Cuda >::value )
         bitonicSort( begin, end, compare, swap );
      else
         throw Exceptions::NotImplementedError( "inplace bitonic sort is implemented only for CUDA" );
   }
};

}  // namespace Sorting
}  // namespace Algorithms
}  // namespace noa::TNL
