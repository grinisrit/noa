// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <cstdlib>  // std::atexit

#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/CheckDevice.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/CudaBadAlloc.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/CudaSupportMissing.h>

namespace noa::TNL {
namespace Algorithms {

class CudaReductionBuffer
{
public:
   inline static CudaReductionBuffer&
   getInstance()
   {
      static CudaReductionBuffer instance;
      return instance;
   }

   inline void
   setSize( std::size_t size )
   {
#ifdef HAVE_CUDA
      if( size > this->size ) {
         this->free();
         if( cudaMalloc( (void**) &this->data, size ) != cudaSuccess ) {
            this->data = 0;
            throw Exceptions::CudaBadAlloc();
         }
         this->size = size;
      }
#else
      throw Exceptions::CudaSupportMissing();
#endif
   }

   template< typename Type >
   Type*
   getData()
   {
      return (Type*) this->data;
   }

   // copy-constructor and copy-assignment are meaningless for a singleton class
   CudaReductionBuffer( CudaReductionBuffer const& copy ) = delete;
   CudaReductionBuffer&
   operator=( CudaReductionBuffer const& copy ) = delete;

private:
   // private constructor of the singleton
   inline CudaReductionBuffer( std::size_t size = 0 )
   {
#ifdef HAVE_CUDA
      setSize( size );
      std::atexit( CudaReductionBuffer::free_atexit );
#endif
   }

   inline static void
   free_atexit()
   {
      CudaReductionBuffer::getInstance().free();
   }

protected:
   inline void
   free()
   {
#ifdef HAVE_CUDA
      if( data ) {
         cudaFree( data );
         data = nullptr;
         TNL_CHECK_CUDA_DEVICE;
      }
#endif
   }

   void* data = nullptr;

   std::size_t size = 0;
};

}  // namespace Algorithms
}  // namespace noa::TNL
