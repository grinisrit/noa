// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber

#pragma once

#include <algorithm>

#include <TNL/Assert.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
   namespace Algorithms {
      namespace Sorting {

struct BubbleSort
{
   template< typename Device, typename Index, typename Compare, typename Swap >
   void static inplaceSort( const Index begin, const Index end, Compare& compare, Swap& swap )
   {
      if( std::is_same< Device, Devices::Cuda >::value )
         throw Exceptions::NotImplementedError( "inplace bubble sort is not implemented for CUDA" );

      Index left = begin;
      Index right = end - 1;
      while( left < right )
      {
         //Index lastChange = end - 1;
         for( Index j = left; j < right - 1; j++ )
         {
            TNL_ASSERT_LT( j+1, end, "" );
            if( ! compare( j, j+1 ) )
            {
               swap( j, j+1 );
               //lastChange = j;
            }
         }
         right--; //lastChange;
         for( Index j = right; j >= left; j-- )
         {
            TNL_ASSERT_LT( j+1, end, "" );
            if( ! compare( j, j+1 ) )
            {
               swap( j, j+1 );
               //lastChange = j;
            }
         }
         left++; //lastChange;
      }
   }
};

      } // namespace Sorting
   } // namespace Algorithms
} //namespace TNL
