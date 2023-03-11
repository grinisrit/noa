// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Atomic.h>

namespace noa::TNL {
namespace Algorithms {

template< typename Device >
struct AtomicOperations;

template<>
struct AtomicOperations< Devices::Host >
{
   // this is __cuda_callable__ only to silence nvcc warnings (all methods inside class
   // template specializations must have the same execution space specifier, otherwise
   // nvcc complains)
   TNL_NVCC_HD_WARNING_DISABLE
   template< typename Value >
   __cuda_callable__
   static Value
   add( Value& v, const Value& a )
   {
      Value old;
#ifdef HAVE_OPENMP
      #pragma omp atomic capture
#endif
      {
         old = v;
         v += a;
      }
      return old;
   }
};

template<>
struct AtomicOperations< Devices::Sequential >
{
   // this is __cuda_callable__ only to silence nvcc warnings (all methods inside class
   // template specializations must have the same execution space specifier, otherwise
   // nvcc complains)
   TNL_NVCC_HD_WARNING_DISABLE
   template< typename Value >
   __cuda_callable__
   static Value
   add( Value& v, const Value& a )
   {
      const Value old = v;
      v += a;
      return old;
   }
};

template<>
struct AtomicOperations< Devices::Cuda >
{
   template< typename Value >
   __cuda_callable__
   static Value
   add( Value& v, const Value& a )
   {
#ifdef __CUDA_ARCH__
      // atomicAdd is __device__, cannot be used from the host side
      return atomicAdd( &v, a );
#else
      return 0;
#endif
   }

   __cuda_callable__
   static short int
   add( short int& v, const short int& a )
   {
#ifdef __CUDACC__
      TNL_ASSERT_TRUE( false, "Atomic add for short int is not supported on CUDA." );
#endif
      return 0;
   }
};

}  // namespace Algorithms
}  // namespace noa::TNL
