// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/CheckDevice.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/CudaBadAlloc.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/CudaSupportMissing.h>

namespace noa::TNL {
namespace Algorithms {

class CudaReductionBuffer
{
public:
   static CudaReductionBuffer&
   getInstance()
   {
      // note that this ensures construction on first use, and thus also correct
      // destruction before the CUDA context is destroyed
      // https://stackoverflow.com/questions/335369/finding-c-static-initialization-order-problems#335746
      static CudaReductionBuffer instance;
      return instance;
   }

   void
   setSize( std::size_t size )  // NOLINT(readability-convert-member-functions-to-static)
   {
#ifdef __CUDACC__
      if( size > this->size ) {
         this->reset();
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

   void
   reset()
   {
#ifdef __CUDACC__
      if( data ) {
         cudaFree( data );
         data = nullptr;
         size = 0;
         TNL_CHECK_CUDA_DEVICE;
      }
#endif
   }

   template< typename Type >
   Type*
   getData()
   {
      return reinterpret_cast< Type* >( this->data );
   }

   ~CudaReductionBuffer()
   {
      reset();
   }

   // copy-constructor and copy-assignment are meaningless for a singleton class
   CudaReductionBuffer( CudaReductionBuffer const& copy ) = delete;
   CudaReductionBuffer&
   operator=( CudaReductionBuffer const& copy ) = delete;

private:
   // private constructor of the singleton
   CudaReductionBuffer( std::size_t size = 0 )
   {
      setSize( size );
   }

   void* data = nullptr;

#ifdef __CUDACC__
   std::size_t size = 0;
#endif
};

}  // namespace Algorithms
}  // namespace noa::TNL
