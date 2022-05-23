// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Sorting/detail/helpers.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Array.h>

namespace noa::TNL {
namespace Algorithms {
namespace Sorting {

#ifdef HAVE_CUDA

/**
 * IMPORTANT: all threads in block have to call this function to work properly
 * the size of src isn't limited, but for optimal efficiency, no more than 8*blockDim.x should be used
 * Description: sorts src and writes into dst within a block
 * works independently from other concurrent blocks
 * @param sharedMem sharedMem pointer has to be able to store all of src elements
 * */
template< typename Value, typename CMP >
__device__
void
bitonicSort_Block( TNL::Containers::ArrayView< Value, TNL::Devices::Cuda > src,
                   TNL::Containers::ArrayView< Value, TNL::Devices::Cuda > dst,
                   Value* sharedMem,
                   const CMP& Cmp )
{
   // copy from globalMem into sharedMem
   for( int i = threadIdx.x; i < src.getSize(); i += blockDim.x )
      sharedMem[ i ] = src[ i ];
   __syncthreads();

   //------------------------------------------
   // bitonic activity
   {
      int paddedSize = closestPow2_ptx( src.getSize() );

      for( int monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2 ) {
         for( int bitonicLen = monotonicSeqLen; bitonicLen > 1; bitonicLen /= 2 ) {
            for( int i = threadIdx.x;; i += blockDim.x )  // simulates other blocks in case src.size > blockDim.x*2
            {
               // calculates which 2 indexes will be compared and swap
               int part = i / ( bitonicLen / 2 );
               int s = part * bitonicLen + ( i & ( ( bitonicLen / 2 ) - 1 ) );
               int e = s + bitonicLen / 2;

               if( e >= src.getSize() )  // touching virtual padding, the order dont swap
                  break;

               // calculate the direction of swapping
               int monotonicSeqIdx = i / ( monotonicSeqLen / 2 );
               bool ascending = ( monotonicSeqIdx & 1 ) != 0;
               if( ( monotonicSeqIdx + 1 ) * monotonicSeqLen >= src.getSize() )  // special case for parts with no "partner"
                  ascending = true;

               cmpSwap( sharedMem[ s ], sharedMem[ e ], ascending, Cmp );
            }

            __syncthreads();  // only 1 synchronization needed
         }
      }
   }

   //------------------------------------------
   // writeback to global memory
   for( int i = threadIdx.x; i < dst.getSize(); i += blockDim.x )
      dst[ i ] = sharedMem[ i ];
}

/**
 * IMPORTANT: all threads in block have to call this function to work properly
 * IMPORTANT: unlike the counterpart with shared memory, this function only works in-place
 * the size of src isn't limited, but for optimal efficiency, no more than 8*blockDim.x should be used
 * Description: sorts src in place using bitonic sort
 * works independently from other concurrent blocks
 * this version doesnt use shared memory and is prefered for Value with big size
 * */
template< typename Value, typename CMP >
__device__
void
bitonicSort_Block( TNL::Containers::ArrayView< Value, TNL::Devices::Cuda > src, const CMP& Cmp )
{
   int paddedSize = closestPow2_ptx( src.getSize() );

   for( int monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2 ) {
      for( int bitonicLen = monotonicSeqLen; bitonicLen > 1; bitonicLen /= 2 ) {
         for( int i = threadIdx.x;; i += blockDim.x )  // simulates other blocks in case src.size > blockDim.x*2
         {
            // calculates which 2 indexes will be compared and swap
            int part = i / ( bitonicLen / 2 );
            int s = part * bitonicLen + ( i & ( ( bitonicLen / 2 ) - 1 ) );
            int e = s + bitonicLen / 2;

            if( e >= src.getSize() )
               break;

            // calculate the direction of swapping
            int monotonicSeqIdx = i / ( monotonicSeqLen / 2 );
            bool ascending = ( monotonicSeqIdx & 1 ) != 0;
            if( ( monotonicSeqIdx + 1 ) * monotonicSeqLen >= src.getSize() )  // special case for parts with no "partner"
               ascending = true;

            cmpSwap( src[ s ], src[ e ], ascending, Cmp );
         }
         __syncthreads();
      }
   }
}

#endif
}  // namespace Sorting
}  // namespace Algorithms
}  // namespace noa::TNL
