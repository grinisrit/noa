// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <stdlib.h>

#include <noa/3rdparty/TNL/Cuda/CheckDevice.h>
#include <noa/3rdparty/TNL/Exceptions/CudaBadAlloc.h>
#include <noa/3rdparty/TNL/Exceptions/CudaSupportMissing.h>

namespace noaTNL {
namespace Algorithms {

class CudaReductionBuffer
{
   public:
      inline static CudaReductionBuffer& getInstance()
      {
         static CudaReductionBuffer instance;
         return instance;
      }

      inline void setSize( size_t size )
      {
#ifdef HAVE_CUDA
         if( size > this->size )
         {
            this->free();
            if( cudaMalloc( ( void** ) &this->data, size ) != cudaSuccess ) {
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
      Type* getData()
      {
         return ( Type* ) this->data;
      }

   private:
      // stop the compiler generating methods of copy the object
      CudaReductionBuffer( CudaReductionBuffer const& copy );            // Not Implemented
      CudaReductionBuffer& operator=( CudaReductionBuffer const& copy ); // Not Implemented

      // private constructor of the singleton
      inline CudaReductionBuffer( size_t size = 0 )
      {
#ifdef HAVE_CUDA
         setSize( size );
         atexit( CudaReductionBuffer::free_atexit );
#endif
      }

      inline static void free_atexit( void )
      {
         CudaReductionBuffer::getInstance().free();
      }

   protected:
      inline void free( void )
      {
#ifdef HAVE_CUDA
         if( data )
         {
            cudaFree( data );
            data = nullptr;
            TNL_CHECK_CUDA_DEVICE;
         }
#endif
      }

      void* data = nullptr;

      size_t size = 0;
};

} // namespace Algorithms
} // namespace noaTNL
